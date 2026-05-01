# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""
Convert TransformerEngine / Megatron-Core MoE expert weights into SonicMoE
layout.

TE/Megatron-Core's `TEGroupedMLP` for SwiGLU stores per-expert weights either
as separate parameters (`linear_fc1.weight0` … `weight{E-1}`) or as a single
fused tensor. In all cases the row layout for fc1 is concatenated:
    [gate_rows (I), up_rows (I)]

SonicMoE wants weights in a (out, in, E) tensor with stride order (2, 0, 1).
For SwiGLU the row order can be either:
    concat_layout=True  : [gate_rows, up_rows]   (TE-native — no reshape)
    concat_layout=False : interleaved [gate0, up0, gate1, up1, …]
                                                  (kernel default)

This module provides:

    load_te_weights_into_sonic(experts, te_state_dict_or_module, *, prefix="")
    sonic_weights_from_te(num_local_experts, hidden_size, intermediate_size,
                          fc1_per_expert, fc2_per_expert, *, concat_layout=True)

Both validate shapes and stride after the conversion.
"""

from __future__ import annotations

from typing import List, Mapping, Sequence, Union

import torch

from .megatron_adapter import SonicMoEExperts


_ParamSource = Union[Mapping[str, torch.Tensor], "torch.nn.Module"]


def _gather_te_per_expert_weights(
    src: _ParamSource,
    num_local_experts: int,
    prefix: str,
) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Return (fc1_list, fc2_list), each of length num_local_experts. Each fc1
    tensor is [2*I, H] in concat layout (gate then up). Each fc2 tensor is
    [H, I].

    Accepts either a state_dict or an nn.Module; tries the standard Megatron
    naming first (`linear_fc1.weight{e}`, `linear_fc2.weight{e}`) and falls
    back to fused tensors (`linear_fc1.weight` of shape [E*2I, H]).
    """
    state = src.state_dict() if hasattr(src, "state_dict") else dict(src)

    def k(name: str) -> str:
        return f"{prefix}{name}" if prefix else name

    # ---- Per-expert layout ----
    if k("linear_fc1.weight0") in state:
        fc1 = [state[k(f"linear_fc1.weight{e}")] for e in range(num_local_experts)]
        fc2 = [state[k(f"linear_fc2.weight{e}")] for e in range(num_local_experts)]
        return fc1, fc2

    # ---- Fused layout (TE GroupedLinear) ----
    if k("linear_fc1.weight") in state:
        w1_fused = state[k("linear_fc1.weight")]    # [E*2I, H]
        w2_fused = state[k("linear_fc2.weight")]    # [H, E*I]
        E = num_local_experts
        two_I = w1_fused.shape[0] // E
        I_dim = w2_fused.shape[1] // E
        fc1 = [w1_fused[e * two_I : (e + 1) * two_I] for e in range(E)]
        fc2 = [w2_fused[:, e * I_dim : (e + 1) * I_dim] for e in range(E)]
        return fc1, fc2

    raise KeyError(
        f"Could not find expert weights under prefix={prefix!r}. "
        f"Looked for `linear_fc1.weight0`/`weight` patterns. "
        f"Available keys: {list(state.keys())[:20]}…"
    )


def sonic_weights_from_te(
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    fc1_per_expert: Sequence[torch.Tensor],
    fc2_per_expert: Sequence[torch.Tensor],
    *,
    concat_layout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stack per-expert TE weights into the SonicMoE-native layout.

    Returns (weight1, weight2) where:
        weight1: [2I, H, E]   stride (2, 0, 1)
        weight2: [H,  I, E]   stride (2, 0, 1)
    """
    E = num_local_experts
    H = hidden_size
    I = intermediate_size
    assert len(fc1_per_expert) == E and len(fc2_per_expert) == E

    for e, (w1, w2) in enumerate(zip(fc1_per_expert, fc2_per_expert)):
        assert w1.shape == (2 * I, H), (
            f"expert {e}: fc1 weight expected ({2 * I}, {H}), got {tuple(w1.shape)}"
        )
        assert w2.shape == (H, I), (
            f"expert {e}: fc2 weight expected ({H}, {I}), got {tuple(w2.shape)}"
        )

    # Stack per-expert into (E, 2I, H) and (E, H, I).
    w1_stack = torch.stack(list(fc1_per_expert), dim=0).contiguous()  # [E, 2I, H]
    w2_stack = torch.stack(list(fc2_per_expert), dim=0).contiguous()  # [E, H, I]

    if not concat_layout:
        # Reorder fc1 rows from [gate(I), up(I)] → interleaved [gate, up, gate, up, …]
        gate = w1_stack[:, :I, :]                          # [E, I, H]
        up = w1_stack[:, I:, :]                            # [E, I, H]
        w1_inter = torch.empty_like(w1_stack)
        w1_inter[:, 0::2, :] = gate
        w1_inter[:, 1::2, :] = up
        w1_stack = w1_inter

    # Stack is contiguous (E, 2I, H). permute(1, 2, 0) gives shape (2I, H, E)
    # with strides (H, 1, 2I*H) → stride order (2, 0, 1). No copy needed.
    weight1 = w1_stack.permute(1, 2, 0)
    weight2 = w2_stack.permute(1, 2, 0)

    _assert_stride_order_201(weight1, "weight1")
    _assert_stride_order_201(weight2, "weight2")

    return weight1, weight2


def _assert_stride_order_201(t: torch.Tensor, name: str) -> None:
    s = t.stride()
    # Expect s[2] > s[0] >= s[1] = 1 (stride order = (2, 0, 1))
    if not (s[2] >= s[0] >= s[1] and s[1] == 1):
        raise AssertionError(
            f"{name}: stride order is not (2, 0, 1) (got strides={s}, shape={tuple(t.shape)})"
        )


def load_te_weights_into_sonic(
    experts: SonicMoEExperts,
    te_source: _ParamSource,
    *,
    prefix: str = "",
) -> None:
    """
    Convert TE/Megatron-Core grouped-MLP weights and load them into a
    `SonicMoEExperts` module (in-place).

    `te_source` may be a state_dict or any module implementing `.state_dict()`.
    Use `prefix` if the experts are nested (e.g. "decoder.layers.0.mlp.experts.").
    The target `experts.concat_layout` flag selects the row layout used.
    """
    fc1, fc2 = _gather_te_per_expert_weights(
        te_source, experts.num_local_experts, prefix=prefix
    )

    weight1, weight2 = sonic_weights_from_te(
        num_local_experts=experts.num_local_experts,
        hidden_size=experts.hidden_size,
        intermediate_size=experts.intermediate_size,
        fc1_per_expert=fc1,
        fc2_per_expert=fc2,
        concat_layout=experts.concat_layout,
    )

    with torch.no_grad():
        experts.weight1.copy_(weight1.to(dtype=experts.weight1.dtype, device=experts.weight1.device))
        experts.weight2.copy_(weight2.to(dtype=experts.weight2.dtype, device=experts.weight2.device))
