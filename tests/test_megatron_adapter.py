# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""
Correctness test for `sonicmoe.megatron_adapter.SonicMoEExperts`.

Simulates Megatron-Core's `AlltoAllTokenDispatcher` output (tokens already
permuted by local expert id) and compares forward + backward against a torch
reference that runs each expert independently with `F.linear`.
"""

import torch
import torch.nn.functional as F
from parameterized import parameterized

from sonicmoe.enums import ActivationType
from sonicmoe.megatron_adapter import SonicMoEExperts

from .test_commons import TestCommons


_SEED = 42


def _swiglu(h: torch.Tensor) -> torch.Tensor:
    g = h[..., ::2]
    u = h[..., 1::2]
    return F.silu(g) * u


def _torch_grouped_experts(
    permuted_tokens: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation: per-expert F.linear → SwiGLU → F.linear.

    weight1: [2I, H, E_local] (stride (2,0,1))
    weight2: [H,  I, E_local]
    """
    H = permuted_tokens.size(1)
    E = tokens_per_expert.numel()
    out = torch.empty_like(permuted_tokens)
    offset = 0
    for e in range(E):
        n = int(tokens_per_expert[e].item())
        if n == 0:
            continue
        x_e = permuted_tokens[offset : offset + n]
        w1_e = weight1[..., e]                       # [2I, H]
        w2_e = weight2[..., e]                       # [H, I]
        h = F.linear(x_e, w1_e)                      # [n, 2I]
        a = _swiglu(h)                               # [n, I]
        y = F.linear(a, w2_e)                        # [n, H]
        out[offset : offset + n] = y
        offset += n
    assert offset == permuted_tokens.size(0)
    return out


def _make_routing(T: int, E: int, K: int, device, generator: torch.Generator):
    """
    Build (permutation, tokens_per_expert) the same way Megatron's dispatcher
    would: each token picks K distinct experts; we then sort by expert id.
    Returns the per-(token,expert) row count == T*K and the bin counts.
    """
    indices = torch.empty(T, K, dtype=torch.long, device=device)
    for t in range(T):
        indices[t] = torch.randperm(E, generator=generator, device=device)[:K]
    flat_expert = indices.flatten()                     # [TK]
    sort_order = torch.argsort(flat_expert, stable=True)
    sorted_expert = flat_expert[sort_order]
    tokens_per_expert = torch.bincount(sorted_expert, minlength=E).to(torch.int32)
    return sort_order, tokens_per_expert


class MegatronAdapterTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [torch.device("cuda")],
            [torch.bfloat16],
            # (T, H, I, E_local, K)
            [
                (1024, 512, 256, 16, 4),
                (2048, 768, 512, 32, 4),
                (4096, 1024, 1024, 64, 8),
                (8192, 2048, 1024, 128, 8),
            ],
        )
    )
    def test_forward_backward(
        self,
        device: torch.device,
        dtype: torch.dtype,
        problem_shape: tuple[int, int, int, int, int],
    ) -> None:
        self.set_seed(_SEED)
        T, H, I, E, K = problem_shape

        with torch.device(device):
            experts = SonicMoEExperts(
                num_local_experts=E,
                hidden_size=H,
                intermediate_size=I,
                activation=ActivationType.SWIGLU,
                bias=False,
                dtype=dtype,
            )
            torch.nn.init.normal_(experts.weight1, mean=0, std=0.02)
            torch.nn.init.normal_(experts.weight2, mean=0, std=0.02)

        gen = torch.Generator(device=device).manual_seed(_SEED)
        sort_order, tokens_per_expert = _make_routing(T, E, K, device, gen)

        # Build the permuted hidden-states tensor: dispatcher would have
        # gathered token h(t) into row r if sort_order[r] == t*K + slot.
        x_unique = 0.02 * torch.randn(T, H, device=device, dtype=dtype)
        permuted = x_unique.repeat_interleave(K, dim=0)[sort_order].contiguous()

        # ---- forward ----
        permuted_kernel = permuted.clone().detach().requires_grad_()
        permuted_ref = permuted.clone().detach().requires_grad_()

        weight1_kernel = experts.weight1
        weight2_kernel = experts.weight2

        # For the reference, materialize a contiguous copy with the same
        # logical values but in default layout — easier to differentiate.
        weight1_ref = weight1_kernel.detach().clone().requires_grad_()
        weight2_ref = weight2_kernel.detach().clone().requires_grad_()

        y_kernel, _ = experts(permuted_kernel, tokens_per_expert)
        y_ref = _torch_grouped_experts(
            permuted_ref, tokens_per_expert, weight1_ref, weight2_ref
        )

        self.assert_equal_tensors(
            y_kernel.float(),
            y_ref.float(),
            False,
            atol_bfloat16=1.4e-2,
            rtol_bfloat16=2e-2,
            dtype=dtype,
        )

        # ---- backward ----
        dy = 0.02 * torch.randn_like(y_kernel)

        gx_kernel, gw1_kernel, gw2_kernel = torch.autograd.grad(
            y_kernel, [permuted_kernel, weight1_kernel, weight2_kernel],
            grad_outputs=dy, retain_graph=False,
        )
        gx_ref, gw1_ref, gw2_ref = torch.autograd.grad(
            y_ref, [permuted_ref, weight1_ref, weight2_ref],
            grad_outputs=dy, retain_graph=False,
        )

        for k_grad, r_grad in (
            (gx_kernel, gx_ref),
            (gw1_kernel, gw1_ref),
            (gw2_kernel, gw2_ref),
        ):
            self.assert_equal_tensors(
                k_grad.float(),
                r_grad.float(),
                False,
                atol_bfloat16=2e-2,
                rtol_bfloat16=2e-2,
                dtype=dtype,
            )

    def test_empty_input_fast_path(self) -> None:
        device = torch.device("cuda")
        with torch.device(device):
            experts = SonicMoEExperts(
                num_local_experts=8,
                hidden_size=128,
                intermediate_size=256,
                activation=ActivationType.SWIGLU,
                bias=False,
                dtype=torch.bfloat16,
            )
        permuted = torch.empty(0, 128, dtype=torch.bfloat16, device=device)
        tokens_per_expert = torch.zeros(8, dtype=torch.int32, device=device)
        y, bias = experts(permuted, tokens_per_expert)
        assert y.shape == (0, 128)
        assert bias is None
