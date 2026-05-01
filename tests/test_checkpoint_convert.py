# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""
Round-trip test for the TE → SonicMoE checkpoint converter.

We can't easily depend on TransformerEngine inside this test, so we synthesize
a state_dict that follows Megatron-Core's two supported naming conventions:

  1. Per-expert layout: linear_fc1.weight0 … linear_fc1.weight{E-1}
  2. Fused layout:      linear_fc1.weight  of shape [E*2I, H]

In both cases the row layout for fc1 is concatenated [gate(I), up(I)], which
is what real TE GroupedLinear weights look like.
"""

import torch
import torch.nn.functional as F

from sonicmoe._checkpoint_convert import load_te_weights_into_sonic, sonic_weights_from_te
from sonicmoe.enums import ActivationType
from sonicmoe.megatron_adapter import SonicMoEExperts

from .test_commons import TestCommons


_SEED = 7


def _swiglu(h: torch.Tensor) -> torch.Tensor:
    g = h[..., ::2]
    u = h[..., 1::2]
    return F.silu(g) * u


def _swiglu_concat(h: torch.Tensor) -> torch.Tensor:
    """SwiGLU when h is [..., 2I] in [gate(I), up(I)] concat layout."""
    I = h.size(-1) // 2
    g = h[..., :I]
    u = h[..., I:]
    return F.silu(g) * u


def _torch_grouped_concat(
    permuted_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    fc1_per_expert,
    fc2_per_expert,
) -> torch.Tensor:
    """Reference using TE-style concat-layout fc1 weights."""
    out = torch.empty_like(permuted_tokens)
    offset = 0
    for e in range(tokens_per_expert.numel()):
        n = int(tokens_per_expert[e].item())
        if n == 0:
            continue
        x_e = permuted_tokens[offset : offset + n]
        h = F.linear(x_e, fc1_per_expert[e])      # [n, 2I] concat layout
        a = _swiglu_concat(h)                     # [n, I]
        out[offset : offset + n] = F.linear(a, fc2_per_expert[e])
        offset += n
    return out


class CheckpointConvertTest(TestCommons):
    def _build_fake_te_state(self, E: int, I: int, H: int, dtype, device, fused: bool):
        fc1_list = [
            0.02 * torch.randn(2 * I, H, dtype=dtype, device=device) for _ in range(E)
        ]
        fc2_list = [
            0.02 * torch.randn(H, I, dtype=dtype, device=device) for _ in range(E)
        ]
        if fused:
            fused_fc1 = torch.cat(fc1_list, dim=0)              # [E*2I, H]
            fused_fc2 = torch.cat(fc2_list, dim=1)              # [H, E*I]
            state = {
                "linear_fc1.weight": fused_fc1,
                "linear_fc2.weight": fused_fc2,
            }
        else:
            state = {}
            for e, w in enumerate(fc1_list):
                state[f"linear_fc1.weight{e}"] = w
            for e, w in enumerate(fc2_list):
                state[f"linear_fc2.weight{e}"] = w
        return state, fc1_list, fc2_list

    def _routing(self, T: int, E: int, K: int, device, gen):
        idxs = torch.empty(T, K, dtype=torch.long, device=device)
        for t in range(T):
            idxs[t] = torch.randperm(E, generator=gen, device=device)[:K]
        flat = idxs.flatten()
        order = torch.argsort(flat, stable=True)
        tokens_per_expert = torch.bincount(flat[order], minlength=E).to(torch.int32)
        return order, tokens_per_expert

    def _run_case(self, fused: bool, concat_layout: bool):
        self.set_seed(_SEED)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        T, H, I, E, K = 1024, 512, 256, 16, 4

        with torch.device(device):
            experts = SonicMoEExperts(
                num_local_experts=E,
                hidden_size=H,
                intermediate_size=I,
                activation=ActivationType.SWIGLU,
                bias=False,
                dtype=dtype,
                concat_layout=concat_layout,
            )

        state, fc1_list, fc2_list = self._build_fake_te_state(E, I, H, dtype, device, fused)

        load_te_weights_into_sonic(experts, state)

        gen = torch.Generator(device=device).manual_seed(_SEED)
        sort_order, tokens_per_expert = self._routing(T, E, K, device, gen)

        x_unique = 0.02 * torch.randn(T, H, device=device, dtype=dtype)
        permuted = x_unique.repeat_interleave(K, dim=0)[sort_order].contiguous()

        y_kernel, _ = experts(permuted, tokens_per_expert)
        y_ref = _torch_grouped_concat(permuted, tokens_per_expert, fc1_list, fc2_list)

        self.assert_equal_tensors(
            y_kernel.float(),
            y_ref.float(),
            False,
            atol_bfloat16=1.4e-2,
            rtol_bfloat16=2e-2,
            dtype=dtype,
        )

    def test_per_expert_concat_layout(self):
        self._run_case(fused=False, concat_layout=True)

    def test_per_expert_interleaved_layout(self):
        self._run_case(fused=False, concat_layout=False)

    def test_fused_concat_layout(self):
        self._run_case(fused=True, concat_layout=True)

    def test_sonic_weights_from_te_smoke(self):
        device = torch.device("cuda")
        dtype = torch.bfloat16
        E, I, H = 4, 64, 128
        fc1 = [torch.randn(2 * I, H, dtype=dtype, device=device) for _ in range(E)]
        fc2 = [torch.randn(H, I, dtype=dtype, device=device) for _ in range(E)]
        w1, w2 = sonic_weights_from_te(E, H, I, fc1, fc2, concat_layout=True)
        assert w1.shape == (2 * I, H, E)
        assert w2.shape == (H, I, E)
        s1, s2 = w1.stride(), w2.stride()
        assert s1[2] >= s1[0] and s1[1] == 1, f"weight1 strides {s1}"
        assert s2[2] >= s2[0] and s2[1] == 1, f"weight2 strides {s2}"
