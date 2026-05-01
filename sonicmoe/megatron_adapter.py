# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""
Megatron-Core / TransformerEngine integration adapter for SonicMoE.

Drop-in replacement for `megatron.core.transformer.moe.experts.TEGroupedMLP`:
the rest of the MoE stack (router, AlltoAllTokenDispatcher, permute / unpermute)
is reused from Megatron-Core. SonicMoE only owns the per-expert grouped GEMMs
on tokens that have already been routed to the local-expert slice.

Forward contract (matches TEGroupedMLP):
    forward(
        permuted_local_hidden_states: [TK_local, H],
        tokens_per_expert: [num_local_experts] int32/int64,
        permuted_probs: [TK_local] | None    (optional, ignored — unpermute applies probs)
    ) -> (output [TK_local, H], output_bias [H] | None)

EP / TP plumbing lives in Megatron's dispatcher; this module is process-group-
agnostic. At EP > 1 each rank instantiates its own `SonicMoEExperts` with
`num_local_experts = global_num_experts // EP`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .enums import ActivationType, is_glu
from .functional import _DownProjection, _UpProjection


_ACTIVATION_BY_NAME = {
    "swiglu": ActivationType.SWIGLU,
    "geglu": ActivationType.GEGLU,
    "reglu": ActivationType.REGLU,
}


def _resolve_activation(name_or_enum) -> ActivationType:
    if isinstance(name_or_enum, ActivationType):
        act = name_or_enum
    else:
        key = str(name_or_enum).lower()
        if key not in _ACTIVATION_BY_NAME:
            raise ValueError(
                f"SonicMoE adapter currently supports GLU activations only "
                f"(swiglu/geglu/reglu); got {name_or_enum!r}"
            )
        act = _ACTIVATION_BY_NAME[key]
    if not is_glu(act):
        raise ValueError(f"SonicMoE adapter requires a GLU activation, got {act}")
    return act


class SonicMoEExperts(nn.Module):
    """
    SonicMoE experts module compatible with Megatron-Core MoELayer.

    Weight layout (matches the kernel's stride requirement (2, 0, 1)):
        weight1 : [2 * intermediate_size, hidden_size, num_local_experts]
        weight2 : [hidden_size,           intermediate_size, num_local_experts]

    The Megatron dispatcher hands us tokens already permuted by local expert id,
    so each row in `permuted_local_hidden_states` activates exactly one expert.
    We therefore build trivial routing metadata (identity scatter/gather) and
    call SonicMoE's `_UpProjection` / `_DownProjection` directly, bypassing the
    Triton routing-metadata kernel.
    """

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: ActivationType | str = ActivationType.SWIGLU,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        init_std: float | None = None,
        concat_layout: bool = False,
    ) -> None:
        super().__init__()
        if bias:
            raise NotImplementedError(
                "SonicMoEExperts: bias=True not supported yet. "
                "Megatron MoE configs typically set add_bias_linear=False."
            )

        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = _resolve_activation(activation)
        self.dtype = dtype
        # concat_layout=True matches TE/Megatron's [gate_rows | up_rows] fused
        # weight layout. concat_layout=False uses [gate, up] interleaved per
        # row pair — the kernel's default for from-scratch training.
        self.concat_layout = concat_layout

        # Stride order (2, 0, 1): dim E (last) is slowest. A contiguous
        # (E, out, in) tensor permuted to (out, in, E) has strides
        # (in, 1, out*in) — exactly what the kernel needs.
        device = torch.device(device) if device is not None else torch.device("cuda")
        I = intermediate_size
        H = hidden_size
        E = num_local_experts

        w1 = torch.empty(E, 2 * I, H, dtype=dtype, device=device).permute(1, 2, 0)
        w2 = torch.empty(E, H, I, dtype=dtype, device=device).permute(1, 2, 0)

        self.weight1 = nn.Parameter(w1)
        self.weight2 = nn.Parameter(w2)

        self.reset_parameters(std=init_std)

    @torch.no_grad()
    def reset_parameters(self, std: float | None = None) -> None:
        if std is None:
            std = (2.0 / (5.0 * self.hidden_size)) ** 0.5
        nn.init.normal_(self.weight1, mean=0.0, std=std)
        nn.init.normal_(self.weight2, mean=0.0, std=std)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = permuted_local_hidden_states
        TK_local, H = x.shape
        assert H == self.hidden_size, (
            f"SonicMoEExperts: hidden_size mismatch (got {H}, expected {self.hidden_size})"
        )
        E = self.num_local_experts
        device = x.device

        # ---- empty-input fast path (no tokens routed to this rank's experts) ----
        if TK_local == 0:
            return torch.empty_like(x), None

        # ---- build identity routing metadata ----
        # Each row of `x` is its own (token, expert) pair → K_eff = 1.
        # Tokens are already grouped by expert id (Megatron dispatcher contract),
        # so x_gather_idx / s_scatter_idx / s_reverse_scatter_idx are all arange.
        if tokens_per_expert.dtype != torch.int32:
            tokens_per_expert_i32 = tokens_per_expert.to(torch.int32)
        else:
            tokens_per_expert_i32 = tokens_per_expert
        if tokens_per_expert_i32.device != device:
            tokens_per_expert_i32 = tokens_per_expert_i32.to(device)

        # expert_frequency_offset has length E + 1, leading zero, then cumsum.
        expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
        expert_frequency_offset[0] = 0
        torch.cumsum(tokens_per_expert_i32, dim=0, dtype=torch.int32, out=expert_frequency_offset[1:])

        identity = torch.arange(TK_local, dtype=torch.int32, device=device)
        x_gather_idx = identity
        s_scatter_idx = identity
        s_reverse_scatter_idx = identity
        num_activated_expert_per_token_offset = torch.arange(
            TK_local + 1, dtype=torch.int32, device=device
        )

        # Recent Megatron-Core (≥ 0.13) applies probs inside the experts module
        # for FP8 numerical stability — `permuted_probs` arrives as a 1-D scaling
        # factor per (token, expert) row, and the dispatcher's unpermute then
        # only does the scatter-add (no probs multiplication). Pass it directly
        # to _DownProjection as router_scores.
        if permuted_probs is None:
            router_scores = torch.ones(TK_local, dtype=torch.float32, device=device)
        else:
            router_scores = permuted_probs.view(-1).to(torch.float32)
            assert router_scores.numel() == TK_local, (
                f"permuted_probs has {router_scores.numel()} entries but expected {TK_local}"
            )

        is_inference = not self.training

        a, h = _UpProjection.apply(
            x,
            self.weight1,
            None,                                # b1
            expert_frequency_offset,
            TK_local,                            # total_expert_freq (== TK)
            None,                                # K — unused on the variable-K path
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            True,                                # is_each_token_has_variable_activated_experts
            self.activation,
            is_inference,
            self.concat_layout,
        )

        out = _DownProjection.apply(
            a,
            h,
            self.weight2,
            None,                                # b2
            router_scores,
            expert_frequency_offset,
            TK_local,                            # T (output rows == TK_local here)
            None,                                # K
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
            True,                                # is_varlen_K
            self.activation,
        )

        return out, None

    def extra_repr(self) -> str:
        return (
            f"num_local_experts={self.num_local_experts}, "
            f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, "
            f"activation={self.activation.value}, dtype={self.dtype}"
        )


# ---------------------------------------------------------------------------
# Megatron-Core MoELayer-compatible wrapper
# ---------------------------------------------------------------------------
class SonicMoEExpertsForMcore(SonicMoEExperts):
    """
    `SonicMoEExperts` with the constructor signature MoELayer expects:

        experts(num_local_experts, config)

    Plug it into `MoESubmodules.experts` to replace `TEGroupedMLP`. The router
    and dispatcher remain Megatron's; only the expert MLPs are swapped.
    """

    def __init__(self, num_local_experts, config, submodules=None, **kwargs):  # noqa: ARG002
        # Megatron passes additional kwargs in newer versions (e.g. pg_collection,
        # cp_comm_type). Accept and discard — SonicMoE doesn't need them at the
        # experts level; EP/CP plumbing happens in the dispatcher.
        moe_ffn = getattr(config, "moe_ffn_hidden_size", None) or config.ffn_hidden_size
        if getattr(config, "add_bias_linear", False):
            raise ValueError("SonicMoEExpertsForMcore: add_bias_linear=True is not supported.")
        if not getattr(config, "gated_linear_unit", True):
            raise ValueError("SonicMoEExpertsForMcore: requires gated_linear_unit=True (SwiGLU).")
        super().__init__(
            num_local_experts=num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=moe_ffn,
            activation=ActivationType.SWIGLU,
            bias=False,
            dtype=config.params_dtype,
        )


# ---------------------------------------------------------------------------
# Megatron-Core integration helper (functional)
# ---------------------------------------------------------------------------
def build_sonic_moe_experts_from_config(transformer_config, num_local_experts: int):
    """
    Construct `SonicMoEExperts` from a `megatron.core.transformer.TransformerConfig`.

    Imported lazily so users without megatron-core installed can still use
    `SonicMoEExperts` directly.
    """
    try:
        from megatron.core.transformer import TransformerConfig  # noqa: F401
    except ImportError as exc:  # pragma: no cover — dependency-only check
        raise ImportError(
            "megatron-core is not available. Install it to use "
            "build_sonic_moe_experts_from_config()."
        ) from exc

    activation = "swiglu" if getattr(transformer_config, "gated_linear_unit", True) else None
    if activation is None:
        raise ValueError("SonicMoEExperts requires a GLU activation (gated_linear_unit=True).")
    if getattr(transformer_config, "add_bias_linear", False):
        raise ValueError("SonicMoEExperts requires add_bias_linear=False.")

    return SonicMoEExperts(
        num_local_experts=num_local_experts,
        hidden_size=transformer_config.hidden_size,
        intermediate_size=transformer_config.moe_ffn_hidden_size or transformer_config.ffn_hidden_size,
        activation=activation,
        bias=False,
        dtype=transformer_config.params_dtype,
    )
