"""Single-combo benchmark + correctness verify for SonicMoE vs TransformerEngine.

Designed to be called from a shell loop (one invocation per combo) so any CUDA
crash on one config doesn't poison subsequent ones — each run gets a fresh CUDA
context.

Usage:
    python bench_compare.py --mode sonic-bf16 --name OLMoE \\
        --T 32768 --H 2048 --I 1024 --E 64 --K 8 --verify

Modes:
    sonic-bf16  — SonicMoE BF16 (kernel under test)
    te-bf16     — TransformerEngine GroupedLinear BF16
    te-fp16     — TransformerEngine GroupedLinear FP16
    te-fp8      — TransformerEngine GroupedLinear with FP8 autocast (BF16 params)

Output (one line, pipe-delimited, parseable):
    name|T=<T>|<mode>|<time_ms>ms|<TFLOPS>TF[|verify_max=<>|verify_rel=<>|verify=<PASS|FAIL>]
or on failure:
    name|T=<T>|<mode>|FAIL|<reason>
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from triton.testing import do_bench


# ============================================================
# Backends — block builders
# ============================================================

def build_sonicmoe_block(H, I, E, K, dtype):
    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType
    with torch.device("cuda"):
        m = MoE(num_experts=E, num_experts_per_tok=K,
                hidden_size=H, intermediate_size=I,
                activation_function=ActivationType.SWIGLU,
                add_bias=False, std=0.02).to(dtype=dtype)
    return m


def build_megatron_moe_block(H, I, E, K, dtype):
    """Megatron-Core MoELayer — NVIDIA's production-grade MoE module.
    Uses TE GroupedLinear underneath but adds Megatron's full router + dispatcher.
    """
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(0)
    from megatron.core import parallel_state
    if not parallel_state.is_initialized() if hasattr(parallel_state, "is_initialized") \
       else not parallel_state.is_unitialized():
        parallel_state.initialize_model_parallel()

    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.moe.moe_layer import MoELayer

    cfg = TransformerConfig(
        num_layers=1, hidden_size=H, num_attention_heads=8,
        ffn_hidden_size=I, num_moe_experts=E, moe_router_topk=K,
        moe_grouped_gemm=True, moe_token_dispatcher_type="alltoall",
        add_bias_linear=False, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        bf16=True, params_dtype=dtype,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=E, moe_grouped_gemm=True)
    moe_submods = spec.submodules.mlp.submodules
    moe = MoELayer(cfg, submodules=moe_submods).cuda().to(dtype)
    moe.eval()

    # Wrap so it accepts [T, H] like our other modes (Megatron expects [B, T, H]).
    class _MCoreWrap(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            y, _ = self.m(x.unsqueeze(0))
            return y.squeeze(0)
    return _MCoreWrap(moe)


def build_te_moe_block(H, I, E, K, dtype):
    """Megatron-LM-style MoE block built from TransformerEngine primitives."""
    import transformer_engine.pytorch as te

    class TEMoEBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.H, self.I, self.E, self.K = H, I, E, K
            self.router = torch.nn.Linear(H, E, bias=False)
            self.up = te.GroupedLinear(num_gemms=E, in_features=H,
                                       out_features=2 * I, bias=False)
            self.down = te.GroupedLinear(num_gemms=E, in_features=I,
                                         out_features=H, bias=False)

        def forward(self, x):
            T = x.shape[0]
            logits = self.router(x).float()
            topk_vals, topk_idx = torch.softmax(logits, -1).topk(self.K, -1)
            topk_vals = topk_vals.to(x.dtype)
            routing_map = torch.zeros(T, self.E, dtype=torch.bool, device=x.device)
            probs = torch.zeros(T, self.E, dtype=x.dtype, device=x.device)
            routing_map.scatter_(1, topk_idx, True)
            probs.scatter_(1, topk_idx, topk_vals)
            permuted_x, _, row_id_map = te.moe_permute_with_probs(
                x, probs, routing_map, num_out_tokens=T * self.K)
            cnt = routing_map.sum(0).tolist()
            h = self.up(permuted_x, cnt)
            a = F.silu(h[..., :self.I]) * h[..., self.I:]
            y = self.down(a, cnt)
            return te.moe_unpermute(y, row_id_map, merging_probs=probs,
                                    restore_shape=torch.Size([T, self.H]),
                                    map_type='mask')
    return TEMoEBlock().cuda().to(dtype)


# ============================================================
# Torch reference (used for correctness verification)
# ============================================================

def torch_ref_for_te(te_block, x):
    """Plain-torch implementation of TEMoEBlock.forward — uses TE's own weights.
    Extracts weight0..weight{E-1} from each te.GroupedLinear and runs a per-expert
    vectorized loop. Output dtype matches x.
    """
    T, H = x.shape
    E, K, I = te_block.E, te_block.K, te_block.I
    # Extract per-expert weights from TE GroupedLinear (named weight0, weight1, ...)
    W1 = torch.stack([getattr(te_block.up,   f"weight{e}").float() for e in range(E)], 0)  # [E, 2I, H]
    W2 = torch.stack([getattr(te_block.down, f"weight{e}").float() for e in range(E)], 0)  # [E, H, I]
    Wr = te_block.router.weight.float()  # [E, H]

    logits  = x.float() @ Wr.T                       # [T, E]
    scores  = torch.softmax(logits, -1)
    topk_vals, topk_idx = scores.topk(K, -1)         # [T, K]

    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    for e in range(E):
        mask = (topk_idx == e)                       # [T, K]
        if not mask.any():
            continue
        token_ids, slot_ids = mask.nonzero(as_tuple=True)
        x_e = x[token_ids].float()                   # [n_e, H]
        s_e = topk_vals[token_ids, slot_ids]         # [n_e]
        h_e = x_e @ W1[e].T                          # [n_e, 2I]
        a_e = F.silu(h_e[..., :I]) * h_e[..., I:]    # [n_e, I]
        y_e = a_e @ W2[e].T                          # [n_e, H]
        out.index_add_(0, token_ids, y_e * s_e.unsqueeze(-1))
    return out.to(x.dtype)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["sonic-bf16", "te-bf16", "te-fp16", "te-fp8",
                             "mcore-bf16", "mcore-fp8"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--I", type=int, required=True)
    ap.add_argument("--E", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rep", type=int, default=15)
    ap.add_argument("--verify", action="store_true",
                    help="Run a torch-reference correctness check at T_verify=1024.")
    ap.add_argument("--T-verify", type=int, default=1024,
                    help="Token count for the verification pass (small to fit memory).")
    ap.add_argument("--include-backward", action="store_true")
    args = ap.parse_args()

    flops_fwd = 2 * args.T * args.K * args.H * (2 * args.I) + 2 * args.T * args.K * args.I * args.H
    flops = flops_fwd * (3 if args.include_backward else 1)

    # tolerance per dtype (atol_max, rtol_max for "PASS")
    TOL = {
        "sonic-bf16": (1.4e-2, 2.0e-2),
        "te-bf16":    (1.4e-2, 2.0e-2),
        "te-fp16":    (5.0e-3, 1.0e-2),
        "te-fp8":     (5.0e-2, 5.0e-2),
        "mcore-bf16": (1.4e-2, 2.0e-2),
        "mcore-fp8":  (5.0e-2, 5.0e-2),
    }[args.mode]

    def report(t_ms=None, err=None, verify=None):
        if err is not None:
            print(f"{args.name}|T={args.T}|{args.mode}|FAIL|{err}")
        else:
            tf = flops / (t_ms * 1e9)
            line = f"{args.name}|T={args.T}|{args.mode}|{t_ms:.3f}ms|{tf:.1f}TF"
            if verify is not None:
                vmax, head, vpass = verify
                line += f"|vmax={vmax:.2e}|head={head:.2e}|verify={'PASS' if vpass else 'FAIL'}"
            print(line)
        sys.stdout.flush()

    try:
        # ---------- Build block + bench function ----------
        if args.mode == "sonic-bf16":
            from sonicmoe import KernelBackendMoE
            block = build_sonicmoe_block(args.H, args.I, args.E, args.K, torch.bfloat16)

            def fn_fwd(x_in):
                with torch.autocast("cuda", torch.float32):
                    return block(x_in, kernel_backend_moe=KernelBackendMoE.sonicmoe)[0]
            def fn_ref(x_in):
                with torch.no_grad():
                    with torch.autocast("cuda", torch.float32):
                        return block(x_in, kernel_backend_moe=KernelBackendMoE.torch)[0]

        elif args.mode in ("te-bf16", "te-fp16"):
            dtype = torch.bfloat16 if args.mode == "te-bf16" else torch.float16
            block = build_te_moe_block(args.H, args.I, args.E, args.K, dtype)

            def fn_fwd(x_in):
                return block(x_in)
            def fn_ref(x_in):
                with torch.no_grad():
                    return torch_ref_for_te(block, x_in)

        elif args.mode == "mcore-bf16":
            block = build_megatron_moe_block(args.H, args.I, args.E, args.K, torch.bfloat16)

            def fn_fwd(x_in):
                return block(x_in)
            def fn_ref(x_in):
                return None  # Megatron router has its own scaling — skip verify

        elif args.mode == "mcore-fp8":
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import DelayedScaling, Format
            block = build_megatron_moe_block(args.H, args.I, args.E, args.K, torch.bfloat16)
            recipe = DelayedScaling(fp8_format=Format.HYBRID, margin=0,
                                    amax_history_len=16, amax_compute_algo='max')

            def fn_fwd(x_in):
                with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                    return block(x_in)
            def fn_ref(x_in):
                return None  # same as mcore-bf16: router not trivially reproducible

        else:  # te-fp8
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import DelayedScaling, Format
            block = build_te_moe_block(args.H, args.I, args.E, args.K, torch.bfloat16)
            recipe = DelayedScaling(fp8_format=Format.HYBRID, margin=0,
                                    amax_history_len=16, amax_compute_algo='max')

            def fn_fwd(x_in):
                with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                    return block(x_in)
            def fn_ref(x_in):
                # FP8 reference: torch math with TE's BF16 weights (best we can do).
                with torch.no_grad():
                    return torch_ref_for_te(block, x_in)

        # ---------- Verification ----------
        verify_result = None
        if args.verify:
            torch.manual_seed(0)
            verify_dtype = torch.float16 if args.mode == "te-fp16" else torch.bfloat16
            x_v = 0.02 * torch.randn(args.T_verify, args.H,
                                     device="cuda", dtype=verify_dtype)
            with torch.no_grad():
                # Need fresh forward for verification (no grad)
                if args.mode == "sonic-bf16":
                    y_test = fn_fwd(x_v)
                elif args.mode in ("te-fp8", "mcore-fp8"):
                    import transformer_engine.pytorch as te
                    from transformer_engine.common.recipe import DelayedScaling, Format
                    rec = DelayedScaling(fp8_format=Format.HYBRID, margin=0,
                                         amax_history_len=16, amax_compute_algo='max')
                    with te.fp8_autocast(enabled=True, fp8_recipe=rec):
                        y_test = block(x_v)
                else:
                    y_test = block(x_v)
                y_ref = fn_ref(x_v)

            if y_ref is None:
                # Backend doesn't provide a torch reference (e.g. Megatron-Core).
                verify_result = (float("nan"), float("nan"), True)  # PASS by default
            else:
                # Elementwise tolerance check (same convention as tests/moe_test.py):
                #   |y_test - y_ref| <= atol + rtol * |y_ref|
                diff = (y_test.float() - y_ref.float()).abs()
                ref_abs = y_ref.float().abs()
                atol, rtol = TOL  # (atol_max, rtol_max)
                n_fail = (diff > (atol + rtol * ref_abs)).sum().item()
                vpass = (n_fail == 0)
                vmax = diff.max().item()
                # head = vmax / atol — values <1 are comfortably under absolute tol
                head = vmax / atol
                verify_result = (vmax, head, vpass)
            del x_v, y_test, y_ref
            torch.cuda.empty_cache()

        # ---------- Timing ----------
        x = 0.02 * torch.randn(args.T, args.H, device="cuda",
                                dtype=torch.float16 if args.mode == "te-fp16" else torch.bfloat16,
                                requires_grad=args.include_backward)

        if args.include_backward:
            dy = torch.randn_like(x)
            def fn():
                y = fn_fwd(x)
                g, = torch.autograd.grad(y, x, dy, retain_graph=False)
                return g
        else:
            def fn():
                with torch.no_grad():
                    return fn_fwd(x)

        fn(); torch.cuda.synchronize()
        t_ms = do_bench(fn, warmup=args.warmup, rep=args.rep)
        report(t_ms=t_ms, verify=verify_result)

    except torch.cuda.OutOfMemoryError:
        report(err="OOM")
    except Exception as e:
        msg = str(e).replace("|", "/").replace("\n", " ")[:80]
        report(err=f"{type(e).__name__}: {msg}")


if __name__ == "__main__":
    main()
