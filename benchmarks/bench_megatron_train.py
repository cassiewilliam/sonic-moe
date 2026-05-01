# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""
Single-layer Megatron-Core MoE training comparison: TEGroupedMLP vs SonicMoE.

Builds two `MoELayer` instances with the *same* router + AlltoAllTokenDispatcher
+ unpermute path, swapping only the experts module. Initializes them with
identical weights (TE → SonicMoE conversion), then runs N training steps with
identical inputs and reports:

    * per-step forward / backward / optimizer latency
    * loss curve for each backend
    * step-0 output L2 + grad L2 difference (sanity)

Usage:
    python benchmarks/bench_megatron_train.py \
        --H 2048 --I 768 --E 128 --K 8 --T 8192 --steps 20

Defaults match Qwen3-30B-A3B (one MoE layer's worth of work).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time

import torch
import torch.nn as nn

from sonicmoe._checkpoint_convert import load_te_weights_into_sonic
from sonicmoe.megatron_adapter import SonicMoEExpertsForMcore


def _setup_distributed(ep_size: int | None = None):
    """
    Initialize torch.distributed (honoring torchrun env if present) and
    Megatron's parallel state with `expert_model_parallel_size = ep_size`.
    Returns (world_size, rank, local_rank).
    """
    # When launched via torchrun these are already set; otherwise fall back to
    # single-process defaults so the script remains runnable as `python …`.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    # When nproc_per_node > num_visible_gpus we share GPUs across ranks
    # (used to simulate multi-node EP on a single node — useful for EP-sweep
    # benchmarking, but a2a will be unrealistically fast since it's all NVLink).
    n_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % n_gpus)

    if ep_size is None:
        ep_size = world_size

    from megatron.core import parallel_state
    if hasattr(parallel_state, "is_initialized"):
        already = parallel_state.is_initialized()
    else:
        already = not parallel_state.is_unitialized()
    if not already:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=ep_size,
        )

    return world_size, rank, local_rank


def _build_moe_layer(H, I, E, K, dtype, experts_cls, ep_size: int = 1,
                     dispatcher: str = "alltoall"):
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.moe.moe_layer import MoELayer

    cfg_kwargs = dict(
        num_layers=1,
        hidden_size=H,
        num_attention_heads=8,
        ffn_hidden_size=I,
        num_moe_experts=E,
        moe_router_topk=K,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type=dispatcher,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        bf16=True,
        params_dtype=dtype,
        expert_model_parallel_size=ep_size,
    )
    if dispatcher == "flex":
        cfg_kwargs["moe_flex_dispatcher_backend"] = "deepep"
        cfg_kwargs["moe_router_dtype"] = "fp32"
    cfg = TransformerConfig(**cfg_kwargs)
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=E, moe_grouped_gemm=True)
    moe_submods = spec.submodules.mlp.submodules
    if experts_cls is not None:
        moe_submods = dataclasses.replace(moe_submods, experts=experts_cls)
    moe = MoELayer(cfg, submodules=moe_submods).cuda().to(dtype)
    return moe, cfg


def _copy_te_weights_to_sonic(te_layer, sonic_layer, num_local_experts):
    """
    Copy router + experts weights from a TE-backed MoELayer into a SonicMoE-backed
    MoELayer, so both run on identical parameters.
    """
    # 1. Router weight (and bias if any).
    sonic_layer.router.load_state_dict(te_layer.router.state_dict())

    # 2. Expert weights — go through the SonicMoEExperts converter.
    te_experts = te_layer.experts
    sonic_experts = sonic_layer.experts
    load_te_weights_into_sonic(sonic_experts, te_experts.state_dict())


def _time_event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _train_loop(layer: nn.Module, x, target, steps: int, lr: float, label: str,
                rank: int = 0, world_size: int = 1, warmup: int = 3):
    """
    Run `steps` forward+backward+optimizer iterations. With multi-rank EP, the
    dispatcher's all-to-all is included in the timed forward/backward window;
    we barrier across ranks before / after each step so timings are aligned.
    """
    optim = torch.optim.AdamW(layer.parameters(), lr=lr, betas=(0.9, 0.95))
    losses = []
    fwd_t = bwd_t = step_t = 0.0
    n_timed = 0
    WARMUP = warmup

    layer.train()
    for s in range(steps):
        if world_size > 1:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        t0 = _time_event()

        out, _ = layer(x.unsqueeze(0))
        loss = (out.squeeze(0) - target).pow(2).mean()

        t1 = _time_event()
        optim.zero_grad(set_to_none=True)
        loss.backward()

        t2 = _time_event()
        optim.step()

        t3 = _time_event()
        torch.cuda.synchronize()

        losses.append(loss.item())
        if s >= WARMUP:
            fwd_t += t0.elapsed_time(t1)
            bwd_t += t1.elapsed_time(t2)
            step_t += t2.elapsed_time(t3)
            n_timed += 1
        if rank == 0:
            print(f"  [{label}] step {s:3d}  loss={loss.item():.6f}", flush=True)

    fwd_avg = fwd_t / n_timed
    bwd_avg = bwd_t / n_timed
    step_avg = step_t / n_timed
    if world_size > 1:
        # Use the slowest rank for each phase — accurate end-to-end step time.
        t = torch.tensor([fwd_avg, bwd_avg, step_avg], device="cuda")
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        fwd_avg, bwd_avg, step_avg = t.tolist()
    return losses, fwd_avg, bwd_avg, step_avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=8192)
    p.add_argument("--H", type=int, default=2048)
    p.add_argument("--I", type=int, default=768)
    p.add_argument("--E", type=int, default=128)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", type=str, default="Qwen3-30B-A3B")
    p.add_argument("--ep", type=int, default=None,
                   help="expert_model_parallel_size (default = WORLD_SIZE)")
    p.add_argument("--dispatcher", choices=["alltoall", "flex"], default="alltoall",
                   help="MoE token dispatcher (alltoall = NCCL a2a, flex = DeepEP)")
    p.add_argument("--warmup", type=int, default=3,
                   help="iterations excluded from timing (autotune absorption)")
    args = p.parse_args()

    dtype = torch.bfloat16
    world_size, rank, _ = _setup_distributed(ep_size=args.ep)
    ep_size = args.ep if args.ep is not None else world_size

    def log(msg: str = ""):
        if rank == 0:
            print(msg, flush=True)

    log(f"=== Megatron MoELayer training: {args.name} (single layer) ===")
    log(f"    T={args.T}  H={args.H}  I={args.I}  E={args.E}  K={args.K}  "
        f"steps={args.steps}  lr={args.lr}  EP={ep_size}  world_size={world_size}  "
        f"dispatcher={args.dispatcher}")
    log(f"    num_local_experts per rank = {args.E // ep_size}")

    # All ranks must use the same seed so they generate the same router weights
    # (Megatron initializes them from torch's default RNG) and the same x/target.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Synthetic data — generate on rank 0, broadcast so every rank sees the
    # same input (mirrors data-parallel training where all ranks share micro-batch).
    if rank == 0:
        x = (0.02 * torch.randn(args.T, args.H, dtype=dtype, device="cuda"))
        target = 0.02 * torch.randn(args.T, args.H, dtype=dtype, device="cuda")
    else:
        x = torch.empty(args.T, args.H, dtype=dtype, device="cuda")
        target = torch.empty(args.T, args.H, dtype=dtype, device="cuda")
    if world_size > 1:
        torch.distributed.broadcast(x, src=0)
        torch.distributed.broadcast(target, src=0)
    x.requires_grad_(False)

    # ----------- Baseline: TEGroupedMLP -----------
    log("\n[1/2] building TE-backed MoELayer …")
    te_layer, _ = _build_moe_layer(
        args.H, args.I, args.E, args.K, dtype, experts_cls=None, ep_size=ep_size,
        dispatcher=args.dispatcher)
    n_params_local = sum(p.numel() for p in te_layer.parameters())
    log(f"      params (per rank): {n_params_local/1e6:.1f}M")

    # ----------- SonicMoE-backed MoELayer -----------
    log("\n[2/2] building SonicMoE-backed MoELayer …")
    sonic_layer, _ = _build_moe_layer(
        args.H, args.I, args.E, args.K, dtype,
        experts_cls=SonicMoEExpertsForMcore, ep_size=ep_size,
        dispatcher=args.dispatcher,
    )

    log("    copying TE weights → SonicMoE …")
    with torch.no_grad():
        _copy_te_weights_to_sonic(te_layer, sonic_layer, args.E // ep_size)

    # ---- step-0 sanity check ----
    te_layer.eval(); sonic_layer.eval()
    with torch.no_grad():
        y_te, _ = te_layer(x.unsqueeze(0))
        y_sonic, _ = sonic_layer(x.unsqueeze(0))
        diff = (y_te - y_sonic).abs()
        max_y = y_te.abs().max().item()
        max_diff = diff.max().item()
        rel = max_diff / max(max_y, 1e-6)
        if world_size > 1:
            t = torch.tensor([max_diff, max_y], device="cuda")
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            max_diff, max_y = t.tolist()
            rel = max_diff / max(max_y, 1e-6)
        log(f"\n[sanity] step-0 forward |Δy|_max = {max_diff:.4e}  "
            f"(rel {rel*100:.2f}%)  |y_te|_max = {max_y:.4e}")

    # ---- training loops ----
    log("\n=== TE training loop ===")
    losses_te, fwd_te, bwd_te, step_te = _train_loop(
        te_layer, x, target, args.steps, args.lr, "TE   ", rank, world_size, args.warmup)

    log("\n=== SonicMoE training loop ===")
    losses_s, fwd_s, bwd_s, step_s = _train_loop(
        sonic_layer, x, target, args.steps, args.lr, "Sonic", rank, world_size, args.warmup)

    total_te = fwd_te + bwd_te + step_te
    total_s = fwd_s + bwd_s + step_s
    log("\n" + "=" * 78)
    log(f"{'':18} {'fwd ms':>10} {'bwd ms':>10} {'opt ms':>10} {'total ms':>10}  loss[-1]")
    log(f"{'TEGroupedMLP':18} {fwd_te:10.2f} {bwd_te:10.2f} {step_te:10.2f} "
        f"{total_te:10.2f}  {losses_te[-1]:.6f}")
    log(f"{'SonicMoEExperts':18} {fwd_s:10.2f} {bwd_s:10.2f} {step_s:10.2f} "
        f"{total_s:10.2f}  {losses_s[-1]:.6f}")
    log(f"{'speedup':18} {fwd_te/fwd_s:9.2f}x {bwd_te/bwd_s:9.2f}x "
        f"{step_te/step_s:9.2f}x {total_te/total_s:9.2f}x")
    log("=" * 78)
    log(f"loss diff (TE - Sonic) at step {args.steps - 1}: "
        f"{losses_te[-1] - losses_s[-1]:+.6e}")

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
