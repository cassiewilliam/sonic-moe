# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""
Single-GPU sweep: TEGroupedMLP vs SonicMoEExperts through Megatron's MoELayer,
across multiple model architectures and EP slicings.

EP semantics here
=================
EP is treated as a *pure E-divisor* on a single GPU. Per cell we build a
MoELayer with `num_experts = E_global / EP`, keeping T, H, I, K unchanged.
This is the per-rank kernel workload the layer would face under real
expert-parallel deployment (with the cross-rank a2a omitted).

For real end-to-end numbers (with NCCL or DeepEP a2a), use
`benchmarks/bench_megatron_train.py` + torchrun.

Memory layout (important)
=========================
To make the largest configs (DeepSeek-V3 single-card etc.) fit on one H100,
we keep only ONE MoELayer on GPU at a time:

    1. Build TE layer on GPU
    2. Snapshot its initial state_dict to CPU
    3. Compute y_te(x) once, move to CPU for the later sanity check
    4. Train TE → record fwd/bwd/opt timings
    5. Free TE + AdamW state
    6. Build Sonic layer on GPU
    7. Load CPU snapshot into sonic (router + experts via the converter)
    8. Compute y_sonic(x), compare with cached y_te (sanity)
    9. Train Sonic → record timings
   10. Free Sonic, move on to next cell

CSV output
==========
Every cell appends a row to `--csv` (default: bench_E_EP_<timestamp>.csv).
Columns:

    config,H,I,E_global,K,EP,E_local,
    fwd_te,bwd_te,opt_te,total_te,
    fwd_s,bwd_s,opt_s,total_s,
    total_speedup,fwd_speedup,bwd_speedup,
    sanity_rel_pct,loss_diff,
    status,note,timestamp_iso

`status` is one of:
    OK              cell ran clean, sanity rel < 2%
    SKIP_K_GE_E     skipped because K >= E_local (degenerate routing)
    SANITY_FAIL     forward output diff > 2% — data not trustworthy
    OOM             cuda.OutOfMemoryError on either TE or Sonic train
    ERROR           any other exception (see `note` for the message)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import os
import sys
import traceback

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Sweep matrix — edit here to add / change configs
# ----------------------------------------------------------------------------
# 3 architectural variants of the same ~30.5B/3.3B-active MoE.
# All three have identical total params and identical per-token activated
# params; only the granularity (E × I × K) differs.
# (label,                              H,    I,    E_global, K)
CONFIGS = [
    ("Qwen3-30B-A3B-128E (original)", 2048, 768,  128,      8),
    ("Qwen3-30B-A3B-256E (variant)",  2048, 384,  256,      16),
    ("Qwen3-30B-A3B-384E (variant)",  2048, 256,  384,      24),
]

# EP values to simulate (1 = full single card, no slicing)
EPS = [1, 8, 16]


# ----------------------------------------------------------------------------
# Megatron / distributed bootstrap (single-rank, no torchrun needed)
# ----------------------------------------------------------------------------
def _init_distributed_singleton():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(0)
    from megatron.core import parallel_state
    if hasattr(parallel_state, "is_initialized"):
        already = parallel_state.is_initialized()
    else:
        already = not parallel_state.is_unitialized()
    if not already:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )


def _build_moe_layer(H, I, E, K, dtype, experts_cls):
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec)
    from megatron.core.transformer.moe.moe_layer import MoELayer

    cfg = TransformerConfig(
        num_layers=1, hidden_size=H, num_attention_heads=8,
        ffn_hidden_size=I, num_moe_experts=E, moe_router_topk=K,
        moe_grouped_gemm=True, moe_token_dispatcher_type="alltoall",
        add_bias_linear=False, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        bf16=True, params_dtype=dtype,
        expert_model_parallel_size=1,
    )
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=E,
                                                      moe_grouped_gemm=True)
    moe_submods = spec.submodules.mlp.submodules
    if experts_cls is not None:
        moe_submods = dataclasses.replace(moe_submods, experts=experts_cls)
    layer = MoELayer(cfg, submodules=moe_submods).cuda().to(dtype)
    return layer


def _state_dict_to_cpu(layer) -> dict:
    """Move a MoELayer's full state_dict to CPU (clones). Used to checkpoint
    initial weights before TE training so sonic can start from the same W0."""
    sd_router = {k: v.detach().cpu().clone()
                 for k, v in layer.router.state_dict().items()}
    sd_experts = {k: v.detach().cpu().clone()
                  for k, v in layer.experts.state_dict().items()}
    return {"router": sd_router, "experts": sd_experts}


def _load_te_init_into_sonic(sonic_layer, init_cpu_state):
    """Load CPU TE init state into a fresh sonic layer (router direct, experts
    via the layout converter)."""
    from sonicmoe._checkpoint_convert import load_te_weights_into_sonic
    sonic_layer.router.load_state_dict(init_cpu_state["router"])
    load_te_weights_into_sonic(sonic_layer.experts, init_cpu_state["experts"])


# ----------------------------------------------------------------------------
# Train loop with cuda.Event timing
# ----------------------------------------------------------------------------
def _ev():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _train_loop(layer, x, target, steps: int, warmup: int, lr: float = 1e-3):
    optim = torch.optim.AdamW(layer.parameters(), lr=lr, betas=(0.9, 0.95))
    losses = []
    fwd_t = bwd_t = opt_t = 0.0
    n_timed = 0
    layer.train()
    for s in range(steps):
        torch.cuda.synchronize()
        t0 = _ev()
        out, _ = layer(x.unsqueeze(0))
        loss = (out.squeeze(0) - target).pow(2).mean()
        t1 = _ev()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        t2 = _ev()
        optim.step()
        t3 = _ev()
        torch.cuda.synchronize()
        losses.append(loss.item())
        if s >= warmup:
            fwd_t += t0.elapsed_time(t1)
            bwd_t += t1.elapsed_time(t2)
            opt_t += t2.elapsed_time(t3)
            n_timed += 1
    return losses, fwd_t / n_timed, bwd_t / n_timed, opt_t / n_timed


# ----------------------------------------------------------------------------
# Run one cell of the sweep — memory-efficient (one layer on GPU at a time)
# ----------------------------------------------------------------------------
def _empty_row(label, H, I, E, K, EP, E_local, status, note=""):
    return {
        "config": label, "H": H, "I": I, "E_global": E, "K": K,
        "EP": EP, "E_local": E_local,
        "fwd_te": "", "bwd_te": "", "opt_te": "", "total_te": "",
        "fwd_s": "", "bwd_s": "", "opt_s": "", "total_s": "",
        "total_speedup": "", "fwd_speedup": "", "bwd_speedup": "",
        "sanity_rel_pct": "", "loss_diff": "",
        "status": status, "note": note,
        "timestamp_iso": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def _run_one(label, H, I, E_global, K, EP, T, steps, warmup, seed):
    """Run TE then Sonic, never holding both on GPU simultaneously."""
    E_local = E_global // EP
    dtype = torch.bfloat16

    print(f"\n{'=' * 78}")
    print(f"  [{label}]  EP={EP}  →  num_experts={E_local}  "
          f"(T={T}, H={H}, I={I}, K={K})")
    print(f"{'=' * 78}")

    # ---- skip degenerate routing ----
    if K >= E_local:
        msg = f"K({K}) >= E_local({E_local}) — skipping degenerate cell"
        print(f"  SKIP: {msg}")
        return _empty_row(label, H, I, E_global, K, EP, E_local,
                          status="SKIP_K_GE_E", note=msg)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x = 0.02 * torch.randn(T, H, dtype=dtype, device="cuda")
    target = 0.02 * torch.randn(T, H, dtype=dtype, device="cuda")
    x.requires_grad_(False)

    init_cpu_state = None
    y_te_cpu = None
    losses_te = fwd_te = bwd_te = opt_te = None

    # ---- Phase 1: TE side ----
    try:
        print(f"  [TE]    building MoELayer…")
        te_layer = _build_moe_layer(H, I, E_local, K, dtype, experts_cls=None)
        n_params = sum(p.numel() for p in te_layer.parameters())
        print(f"  [TE]    params: {n_params/1e6:.1f}M    "
              f"(GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB)")

        # Snapshot init weights to CPU so sonic can start from W0.
        print(f"  [TE]    snapshotting initial weights to CPU…")
        init_cpu_state = _state_dict_to_cpu(te_layer)

        # Capture y_te(x) at init for the later sanity check.
        te_layer.eval()
        with torch.no_grad():
            y_te = te_layer(x.unsqueeze(0))[0]
            y_te_cpu = y_te.detach().cpu().clone()
            del y_te
        te_layer.train()

        print(f"  [TE]    training {steps} steps (warmup={warmup})…")
        losses_te, fwd_te, bwd_te, opt_te = _train_loop(te_layer, x, target,
                                                        steps, warmup)
        del te_layer
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        msg = f"OOM during TE phase: {str(e)[:100]}"
        print(f"  [TE]    OOM — {msg}")
        return _empty_row(label, H, I, E_global, K, EP, E_local,
                          status="OOM", note=msg)
    except Exception as e:  # noqa
        torch.cuda.empty_cache()
        msg = f"TE error: {type(e).__name__}: {str(e)[:120]}"
        traceback.print_exc()
        return _empty_row(label, H, I, E_global, K, EP, E_local,
                          status="ERROR", note=msg)

    # ---- Phase 2: Sonic side ----
    try:
        from sonicmoe.megatron_adapter import SonicMoEExpertsForMcore
        print(f"  [Sonic] building MoELayer…    "
              f"(GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB)")
        sonic_layer = _build_moe_layer(H, I, E_local, K, dtype,
                                       experts_cls=SonicMoEExpertsForMcore)
        print(f"  [Sonic] loading TE init weights from CPU…")
        with torch.no_grad():
            _load_te_init_into_sonic(sonic_layer, init_cpu_state)

        # Sanity: forward parity with cached y_te
        sonic_layer.eval()
        with torch.no_grad():
            y_s = sonic_layer(x.unsqueeze(0))[0]
            y_te_dev = y_te_cpu.to(y_s.device, non_blocking=True)
            diff = (y_te_dev - y_s).abs()
            max_diff = diff.max().item()
            max_y = y_te_dev.abs().max().item()
            rel = max_diff / max(max_y, 1e-12)
            del y_s, y_te_dev
        sonic_layer.train()
        print(f"  [Sonic] sanity: |Δy|_max={max_diff:.3e}  |y|_max={max_y:.3e}  "
              f"rel={rel*100:.2f}%  ({'PASS' if rel < 0.02 else 'FAIL'})")
        if rel >= 0.02:
            del sonic_layer
            torch.cuda.empty_cache()
            return _empty_row(label, H, I, E_global, K, EP, E_local,
                              status="SANITY_FAIL",
                              note=f"rel diff {rel*100:.2f}% >= 2%")

        print(f"  [Sonic] training {steps} steps (warmup={warmup})…")
        losses_s, fwd_s, bwd_s, opt_s = _train_loop(sonic_layer, x, target,
                                                    steps, warmup)
        del sonic_layer
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        msg = f"OOM during Sonic phase: {str(e)[:100]}"
        print(f"  [Sonic] OOM — {msg}")
        return _empty_row(label, H, I, E_global, K, EP, E_local,
                          status="OOM", note=msg)
    except Exception as e:  # noqa
        torch.cuda.empty_cache()
        msg = f"Sonic error: {type(e).__name__}: {str(e)[:120]}"
        traceback.print_exc()
        return _empty_row(label, H, I, E_global, K, EP, E_local,
                          status="ERROR", note=msg)

    # ---- Report ----
    total_te = fwd_te + bwd_te + opt_te
    total_s = fwd_s + bwd_s + opt_s
    spd = total_te / total_s
    spd_fwd = fwd_te / fwd_s
    spd_bwd = bwd_te / bwd_s
    loss_diff = losses_te[-1] - losses_s[-1]

    print(f"\n  {'':18} {'fwd ms':>9} {'bwd ms':>9} {'opt ms':>9} "
          f"{'total ms':>10}  loss[-1]")
    print(f"  {'TEGroupedMLP':18} {fwd_te:9.2f} {bwd_te:9.2f} {opt_te:9.2f} "
          f"{total_te:10.2f}  {losses_te[-1]:.6f}")
    print(f"  {'SonicMoEExperts':18} {fwd_s :9.2f} {bwd_s :9.2f} {opt_s :9.2f} "
          f"{total_s :10.2f}  {losses_s[-1]:.6f}")
    print(f"  {'speedup':18} {spd_fwd:8.2f}x {spd_bwd:8.2f}x "
          f"{opt_te/opt_s:8.2f}x {spd:9.2f}x")
    print(f"  loss diff (TE - Sonic): {loss_diff:+.3e}")

    return {
        "config": label, "H": H, "I": I, "E_global": E_global, "K": K,
        "EP": EP, "E_local": E_local,
        "fwd_te": round(fwd_te, 4), "bwd_te": round(bwd_te, 4),
        "opt_te": round(opt_te, 4), "total_te": round(total_te, 4),
        "fwd_s": round(fwd_s, 4), "bwd_s": round(bwd_s, 4),
        "opt_s": round(opt_s, 4), "total_s": round(total_s, 4),
        "total_speedup": round(spd, 4),
        "fwd_speedup": round(spd_fwd, 4),
        "bwd_speedup": round(spd_bwd, 4),
        "sanity_rel_pct": round(rel * 100, 4),
        "loss_diff": f"{loss_diff:.3e}",
        "status": "OK", "note": "",
        "timestamp_iso": _dt.datetime.now().isoformat(timespec="seconds"),
    }


# ----------------------------------------------------------------------------
# CSV writer — appends one row per cell as soon as it's done
# ----------------------------------------------------------------------------
CSV_COLUMNS = [
    "config", "H", "I", "E_global", "K", "EP", "E_local",
    "fwd_te", "bwd_te", "opt_te", "total_te",
    "fwd_s", "bwd_s", "opt_s", "total_s",
    "total_speedup", "fwd_speedup", "bwd_speedup",
    "sanity_rel_pct", "loss_diff",
    "status", "note", "timestamp_iso",
]


def _open_csv(path):
    """Return (writer, file_handle). Writes header if file didn't exist."""
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if new_file:
        w.writeheader()
        f.flush()
    return w, f


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=8192)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", type=str, default=None,
                   help="output CSV path (default: /tmp/bench_E_EP_<ts>.csv)")
    p.add_argument("--skip-ep1", action="store_true",
                   help="skip EP=1 baseline cells")
    p.add_argument("--only", type=str, default="",
                   help="comma-separated config labels to keep (substring match)")
    args = p.parse_args()

    if args.csv is None:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.csv = f"/tmp/bench_E_EP_{ts}.csv"

    eps = [e for e in EPS if not (args.skip_ep1 and e == 1)]
    cfgs = CONFIGS
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        cfgs = [c for c in cfgs if any(k in c[0] for k in keys)]

    print("=" * 78)
    print("  bench_E_EP_single_gpu.py — TE vs SonicMoE through Megatron MoELayer")
    print("=" * 78)
    print(f"  T={args.T}  steps={args.steps}  warmup={args.warmup}")
    print(f"  configs ({len(cfgs)}):")
    for c in cfgs:
        print(f"    {c[0]:38} H={c[1]:>5} I={c[2]:>5} E={c[3]:>4} K={c[4]:>3}")
    print(f"  EP sweep: {eps}")
    print(f"  CSV out : {args.csv}")
    print()

    _init_distributed_singleton()

    csv_w, csv_f = _open_csv(args.csv)

    rows = []
    n_total = len(cfgs) * len(eps)
    n_done = 0
    for cfg in cfgs:
        label, H, I, E_global, K = cfg
        for EP in eps:
            n_done += 1
            print(f"\n###  [{n_done}/{n_total}]  starting cell  ###")
            r = _run_one(label, H, I, E_global, K, EP,
                         T=args.T, steps=args.steps, warmup=args.warmup,
                         seed=args.seed)
            csv_w.writerow(r)
            csv_f.flush()
            rows.append(r)

    csv_f.close()

    # ---- Final stdout summary ----
    print("\n" + "#" * 100)
    print("# Final summary")
    print("#" * 100)
    print(f"\n{'config':40} {'EP':>3} {'E_loc':>6} "
          f"{'fwd_TE':>7} {'fwd_S':>7} "
          f"{'bwd_TE':>7} {'bwd_S':>7} "
          f"{'tot_TE':>7} {'tot_S':>7} "
          f"{'spd':>5}  {'rel%':>5}  {'status':12}")
    for r in rows:
        if r["status"] == "OK":
            print(f"{r['config'][:40]:40} {r['EP']:>3} {r['E_local']:>6} "
                  f"{r['fwd_te']:>7.2f} {r['fwd_s']:>7.2f} "
                  f"{r['bwd_te']:>7.2f} {r['bwd_s']:>7.2f} "
                  f"{r['total_te']:>7.2f} {r['total_s']:>7.2f} "
                  f"{r['total_speedup']:>4.2f}x  "
                  f"{r['sanity_rel_pct']:>4.1f}%  {r['status']:12}")
        else:
            note = r.get("note", "")[:30]
            print(f"{r['config'][:40]:40} {r['EP']:>3} {r['E_local']:>6} "
                  f"{'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7} "
                  f"{'-':>5}   {'-':>4}   {r['status']:12}  {note}")

    print(f"\nCSV → {args.csv}")
    print("Done.")


if __name__ == "__main__":
    main()
