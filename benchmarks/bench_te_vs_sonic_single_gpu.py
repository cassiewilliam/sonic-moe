# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
"""
Single-GPU SonicMoE vs TransformerEngine direct kernel comparison.

What this measures
------------------
We bypass Megatron-Core's MoELayer entirely (no router, no dispatcher, no
unpermute, no optimizer) and compare *only* the per-expert grouped-GEMM stack:

    Input:  permuted_tokens [TK, H]      — already grouped by expert id
            tokens_per_expert [E]        — int32 bincount, sums to TK
            permuted_probs [TK]          — fp32 routing weight per (token, expert)

    Operation:  for each expert e:
                    h = X[block_e] @ W1[e]^T          # gate+up, shape [n, 2I]
                    a = silu(h[:, :I]) * h[:, I:]      # SwiGLU
                    y = a @ W2[e]^T                   # down,  shape [n, H]
                    y *= permuted_probs[block_e]      # scale by router score
                                  (sonic does this inside _DownProjection;
                                   we apply it manually in the TE reference)

    Output: [TK, H], one row per (token, expert) pair

Both backends consume *identical* inputs and *identical* weights. The only
thing that differs is the GEMM kernel implementation:

  - TE side: `transformer_engine.pytorch.GroupedLinear` (per-expert FP8/BF16
    grouped GEMM, NVIDIA production reference)
  - Sonic side: `sonicmoe.megatron_adapter.SonicMoEExperts` which calls
    `_UpProjection` / `_DownProjection` (CuTeDSL/Triton kernels with dS
    contraction reordering)

Methodology
-----------
1.  Initialize TE GroupedLinear weights (per-expert tensors W1[e], W2[e])
2.  Build SonicMoEExperts with the same shapes
3.  Stack TE per-expert weights into sonic's required (out, in, E) layout with
    stride order (2, 0, 1) and copy them in. From this point on, both backends
    hold mathematically identical W1 / W2.
4.  Synthesize a permuted batch (T tokens × K activations each, sorted by expert
    id, balanced) and shared probs.
5.  Verify forward outputs match within BF16 noise (|Δ| / |y|_max < 2%).
6.  Verify backward gradients (dx, dW1, dW2) match within BF16 noise.
7.  Time forward and backward separately with `torch.cuda.Event`, after a
    warmup that absorbs any kernel autotuning. Print fwd/bwd ms and speedup.

How to run
----------
    python benchmarks/bench_te_vs_sonic_single_gpu.py \
        --T 8192 --H 2048 --I 768 --E 128 --K 8

Args (all optional, defaults match Qwen3-30B-A3B):
    --T  num input tokens (per microbatch)
    --H  hidden size
    --I  intermediate size per expert (gate dim — SwiGLU output is 2I wide)
    --E  number of experts
    --K  top-k routing
    --warmup    iterations excluded from timing (default 10)
    --iters     iterations measured (default 30)

What to look at
---------------
    [sanity] forward |Δy|_max / |y|_max  →  must be < ~2% (BF16 noise)
    [sanity] backward |Δgrad|_max / |grad|_max for dx, dW1, dW2  →  same
    [timing] forward  ms          (TE / Sonic / speedup)
    [timing] backward ms          (TE / Sonic / speedup)

If any sanity check is above ~3% the comparison is unfair (weight conversion
or numerics drift). Reject the timing.

Caveats
-------
- Both backends apply probs with the same scalar; differences come purely
  from the grouped-GEMM kernel.
- We use a deterministic, balanced routing (each expert gets exactly TK/E
  tokens). Real workloads have imbalance; that hurts both backends roughly
  equally so the relative numbers are still informative.
- This script does NOT exercise:
    * mcore router (Linear → softmax → topk)
    * mcore AlltoAllTokenDispatcher (permute + a2a + unpermute)
    * AdamW optimizer
  Those are realistic costs but they're identical between the two backends
  (we share router/dispatcher in the Megatron version). For
  end-to-end-with-Megatron numbers see `bench_megatron_train.py`.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1. Build TE side (raw te.GroupedLinear — no Megatron wrapper)
# ----------------------------------------------------------------------------
class TEMoEExperts(torch.nn.Module):
    """
    Reference implementation using TransformerEngine's GroupedLinear directly.
    Mirrors what `TEGroupedMLP.experts.forward(permuted_tokens, tokens_per_expert,
    permuted_probs)` does inside Megatron-Core, but without the Megatron wrapper:

        h = linear_fc1(permuted_tokens, tokens_per_expert)   # [TK, 2I]
        a = silu(h[:, :I]) * h[:, I:]                        # SwiGLU, concat layout
        y = linear_fc2(a, tokens_per_expert)                  # [TK, H]
        y = y * permuted_probs.unsqueeze(-1)                  # scale by routing weight

    Bias is disabled to match SonicMoEExperts.
    """

    def __init__(self, H: int, I: int, E: int, dtype: torch.dtype):
        super().__init__()
        import transformer_engine.pytorch as te
        self.H, self.I, self.E = H, I, E
        # Each expert is a separate GEMM; layout per expert is [2I, H] / [H, I].
        self.linear_fc1 = te.GroupedLinear(num_gemms=E, in_features=H,
                                            out_features=2 * I, bias=False,
                                            params_dtype=dtype)
        self.linear_fc2 = te.GroupedLinear(num_gemms=E, in_features=I,
                                            out_features=H, bias=False,
                                            params_dtype=dtype)

    def forward(self, permuted_tokens: torch.Tensor,
                tokens_per_expert: torch.Tensor,
                permuted_probs: torch.Tensor) -> torch.Tensor:
        # te.GroupedLinear takes a Python list of int counts.
        cnt = tokens_per_expert.tolist()
        h = self.linear_fc1(permuted_tokens, cnt)         # [TK, 2I]
        a = F.silu(h[..., :self.I]) * h[..., self.I:]     # SwiGLU concat layout
        y = self.linear_fc2(a, cnt)                       # [TK, H]
        y = y * permuted_probs.to(y.dtype).unsqueeze(-1)
        return y

    def per_expert_fc1_fc2(self):
        """
        Return (W1_list, W2_list), each a list of E tensors, in standard
        per-expert layout: W1[e] shape [2I, H], W2[e] shape [H, I].
        Used by the weight-copy step.
        """
        W1 = [getattr(self.linear_fc1, f"weight{e}") for e in range(self.E)]
        W2 = [getattr(self.linear_fc2, f"weight{e}") for e in range(self.E)]
        return W1, W2


# ----------------------------------------------------------------------------
# 2. Build sonic side (SonicMoEExperts) and copy TE weights into it
# ----------------------------------------------------------------------------
def build_sonic_experts(H, I, E, dtype):
    from sonicmoe.megatron_adapter import SonicMoEExperts
    return SonicMoEExperts(
        num_local_experts=E,
        hidden_size=H,
        intermediate_size=I,
        bias=False,
        dtype=dtype,
        concat_layout=False,    # interleaved [g0, u0, g1, u1, ...] — matches converter target
    )


def copy_te_to_sonic(te: TEMoEExperts, sonic):
    """
    Stack TE per-expert weights into the (out, in, E) shape with stride order
    (2, 0, 1) that sonic kernels require. Concat layout [gate, up] is converted
    to interleaved layout because we set sonic.concat_layout=False above.
    """
    from sonicmoe._checkpoint_convert import sonic_weights_from_te

    W1_list, W2_list = te.per_expert_fc1_fc2()
    weight1, weight2 = sonic_weights_from_te(
        num_local_experts=te.E,
        hidden_size=te.H,
        intermediate_size=te.I,
        fc1_per_expert=W1_list,
        fc2_per_expert=W2_list,
        concat_layout=False,    # rearrange [gate(I), up(I)] → [g0, u0, g1, u1, ...]
    )
    with torch.no_grad():
        sonic.weight1.copy_(weight1.to(sonic.weight1.dtype))
        sonic.weight2.copy_(weight2.to(sonic.weight2.dtype))


# ----------------------------------------------------------------------------
# 3. Synthesize a fake-but-balanced dispatcher output
# ----------------------------------------------------------------------------
def synth_inputs(T: int, K: int, H: int, E: int, dtype, device, seed: int):
    """
    Build a permuted batch as if Megatron's AlltoAllTokenDispatcher had just
    finished:

      * Each token is randomly assigned K distinct experts (top-K).
      * Tokens are sorted by expert id (so expert 0's rows come first, then
        expert 1, etc.). This is what `permute_with_probs` produces.
      * tokens_per_expert is the bincount, summing to TK = T*K.
      * permuted_probs is the per-row routing score; we softmax random logits
        so each token's K scores sum to 1 (matching mcore's defaults).

    We use a fixed seed and balanced routing (uniform random) so the workload
    is identical across runs. Real workloads have skew but it hits both
    backends equally.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    TK = T * K

    # Pick K distinct experts per token, then sort by expert id.
    expert_idx = torch.empty(T, K, dtype=torch.int64, device=device)
    for t in range(T):
        expert_idx[t] = torch.randperm(E, generator=g, device=device)[:K]
    flat_expert = expert_idx.flatten()                   # [TK]
    sort_order = torch.argsort(flat_expert, stable=True)
    sorted_expert = flat_expert[sort_order]
    tokens_per_expert = torch.bincount(sorted_expert, minlength=E).to(torch.int32)

    # Tokens (one per row of the permuted batch).
    x_unique = 0.02 * torch.randn(T, H, dtype=dtype, device=device, generator=g)
    permuted_tokens = x_unique.repeat_interleave(K, dim=0)[sort_order].contiguous()

    # Probs: softmax over random logits, take topk, then permute to row layout.
    logits = torch.randn(T, E, dtype=torch.float32, device=device, generator=g)
    probs_full = F.softmax(logits, dim=-1)
    topk_scores = probs_full.gather(1, expert_idx)       # [T, K]
    permuted_probs = topk_scores.flatten()[sort_order].contiguous()

    return permuted_tokens, tokens_per_expert, permuted_probs


# ----------------------------------------------------------------------------
# 4. Sanity checks (forward + backward equivalence)
# ----------------------------------------------------------------------------
def sanity(label_a, fwd_a, label_b, fwd_b, x, *, atol_max_rel: float = 0.02):
    """Compare two forward callables on the same input; report abs/rel diffs."""
    y_a = fwd_a(x)
    y_b = fwd_b(x)
    diff = (y_a - y_b).abs()
    ymax = y_a.abs().max().item()
    rel = diff.max().item() / max(ymax, 1e-12)
    print(f"  [sanity fwd] |Δ|_max={diff.max().item():.3e}  "
          f"|y|_max={ymax:.3e}  rel={rel*100:.2f}%   "
          f"({label_a} vs {label_b})")
    assert rel < atol_max_rel, (
        f"forward outputs differ too much (rel={rel*100:.2f}%); "
        f"weights are NOT identical. Reject this run."
    )


def sanity_grad(name, g_a, g_b, *, atol_max_rel: float = 0.05):
    diff = (g_a - g_b).abs()
    gmax = g_a.abs().max().item()
    rel = diff.max().item() / max(gmax, 1e-12)
    print(f"  [sanity bwd] grad {name:5s}: |Δ|_max={diff.max().item():.3e}  "
          f"|g|_max={gmax:.3e}  rel={rel*100:.2f}%")
    assert rel < atol_max_rel, f"backward grad {name} differs too much"


# ----------------------------------------------------------------------------
# 5. Timing
# ----------------------------------------------------------------------------
def time_fwd_bwd(label, layer, args_factory, n_warmup: int, n_iter: int):
    """
    Time forward and backward separately. `args_factory()` returns a fresh
    `(permuted_tokens, tokens_per_expert, permuted_probs)` triple where
    permuted_tokens has requires_grad=True (so we can take a real bwd).

    Returns (avg_fwd_ms, avg_bwd_ms).
    """
    # --- warmup ---
    for _ in range(n_warmup):
        permuted_tokens, tokens_per_expert, permuted_probs = args_factory()
        out = layer(permuted_tokens, tokens_per_expert, permuted_probs)
        # SonicMoEExperts.forward returns (output, output_bias); TE returns a tensor.
        if isinstance(out, tuple):
            out = out[0]
        loss = out.float().sum()
        loss.backward()
    torch.cuda.synchronize()

    # --- measure forward ---
    fwd_evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    saved = []
    for e0, e1 in fwd_evs:
        permuted_tokens, tokens_per_expert, permuted_probs = args_factory()
        e0.record()
        out = layer(permuted_tokens, tokens_per_expert, permuted_probs)
        if isinstance(out, tuple):
            out = out[0]
        e1.record()
        saved.append((permuted_tokens, out))
    torch.cuda.synchronize()
    fwd_ms = sum(e0.elapsed_time(e1) for e0, e1 in fwd_evs) / n_iter

    # --- measure backward (using fresh fwd-bwd pairs, fwd uncounted) ---
    bwd_evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(n_iter)]
    for e0, e1 in bwd_evs:
        permuted_tokens, tokens_per_expert, permuted_probs = args_factory()
        out = layer(permuted_tokens, tokens_per_expert, permuted_probs)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.float().sum()
        torch.cuda.synchronize()
        e0.record()
        loss.backward()
        e1.record()
    torch.cuda.synchronize()
    bwd_ms = sum(e0.elapsed_time(e1) for e0, e1 in bwd_evs) / n_iter

    return fwd_ms, bwd_ms


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=8192)
    p.add_argument("--H", type=int, default=2048)
    p.add_argument("--I", type=int, default=768)
    p.add_argument("--E", type=int, default=128)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dtype = torch.bfloat16
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"=== Bench: TE GroupedLinear vs SonicMoEExperts ===")
    print(f"    T={args.T}  H={args.H}  I={args.I}  E={args.E}  K={args.K}")
    print(f"    TK={args.T*args.K}  per-expert tokens (avg)={args.T*args.K // args.E}")
    print(f"    warmup={args.warmup}  iters={args.iters}  dtype={dtype}")
    print()

    # Build TE side. te.GroupedLinear initializes its own weights randomly.
    print("[1/3] building TE experts (te.GroupedLinear x 2)...")
    te_experts = TEMoEExperts(args.H, args.I, args.E, dtype).to(device)

    # Build sonic side and copy TE weights in.
    print("[2/3] building SonicMoEExperts and copying TE weights in...")
    sonic_experts = build_sonic_experts(args.H, args.I, args.E, dtype).to(device)
    copy_te_to_sonic(te_experts, sonic_experts)

    # Build inputs.
    print("[3/3] synthesizing dispatcher output (balanced random routing)...")
    pt, tpe, pp = synth_inputs(args.T, args.K, args.H, args.E, dtype, device,
                                seed=args.seed)
    print(f"    permuted_tokens: shape={tuple(pt.shape)}, dtype={pt.dtype}")
    print(f"    tokens_per_expert: shape={tuple(tpe.shape)}, sum={tpe.sum().item()}")
    print(f"    permuted_probs:  shape={tuple(pp.shape)}, mean={pp.mean().item():.4f}")
    print()

    # ------------ sanity: forward equivalence ------------
    print("Sanity forward (no_grad):")
    with torch.no_grad():
        y_te = te_experts(pt, tpe, pp)
        y_s_out = sonic_experts(pt, tpe, pp)
        y_s = y_s_out[0] if isinstance(y_s_out, tuple) else y_s_out
    diff = (y_te - y_s).abs()
    ymax = y_te.abs().max().item()
    rel = diff.max().item() / max(ymax, 1e-12)
    print(f"  |Δy|_max={diff.max().item():.3e}  |y|_max={ymax:.3e}  "
          f"rel={rel*100:.2f}%   ({'PASS' if rel < 0.02 else 'FAIL — STOP HERE'})")
    if rel >= 0.02:
        raise SystemExit("Outputs disagree; weight copy or kernel layout bug.")
    print()

    # ------------ sanity: backward equivalence ------------
    print("Sanity backward (real autograd, identical inputs and grad_outputs):")
    pt_te = pt.detach().clone().requires_grad_(True)
    pt_s  = pt.detach().clone().requires_grad_(True)
    # Reset param gradients on both sides.
    for layer in (te_experts, sonic_experts):
        for p_ in layer.parameters():
            p_.grad = None
    y_te = te_experts(pt_te, tpe, pp)
    out_s = sonic_experts(pt_s, tpe, pp)
    y_s = out_s[0] if isinstance(out_s, tuple) else out_s

    # Use a *deterministic* (not random) grad_output so any difference in
    # gradients is purely from kernel implementation, not from random noise.
    dy = torch.full_like(y_te, 1e-3)
    y_te.backward(dy)
    y_s.backward(dy)

    # Compare input-grad
    sanity_grad("dx", pt_te.grad.float(), pt_s.grad.float())

    # Compare W1 grads (need to re-stack TE per-expert grads to (2I, H, E))
    W1_te_list, W2_te_list = te_experts.per_expert_fc1_fc2()
    dW1_te_stack = torch.stack([w.grad.float() for w in W1_te_list], dim=-1)  # [2I, H, E]
    dW2_te_stack = torch.stack([w.grad.float() for w in W2_te_list], dim=-1)  # [H, I, E]
    # The sonic side stores weights in [2I, H, E] interleaved (gate/up rows
    # interleaved across the 2I axis), whereas TE stores [gate(I), up(I)]
    # concat. To compare, we re-stack TE gradients in the sonic layout.
    I_ = args.I
    dW1_te_interleaved = torch.empty_like(dW1_te_stack)
    dW1_te_interleaved[0::2, :, :] = dW1_te_stack[:I_, :, :]    # gate rows
    dW1_te_interleaved[1::2, :, :] = dW1_te_stack[I_:, :, :]    # up rows
    sanity_grad("dW1", dW1_te_interleaved, sonic_experts.weight1.grad.float())
    sanity_grad("dW2", dW2_te_stack,       sonic_experts.weight2.grad.float())
    print()

    # ------------ timing ------------
    print(f"Timing (warmup={args.warmup}, iters={args.iters}):")

    def args_factory():
        # Fresh leaf each iter so backward graphs don't tangle.
        return (pt.detach().clone().requires_grad_(True),
                tpe, pp)

    fwd_te, bwd_te = time_fwd_bwd("TE",    te_experts,    args_factory,
                                   args.warmup, args.iters)
    fwd_s,  bwd_s  = time_fwd_bwd("Sonic", sonic_experts, args_factory,
                                   args.warmup, args.iters)

    print()
    print("=" * 72)
    print(f"{'':18} {'fwd ms':>10} {'bwd ms':>10} {'fwd+bwd ms':>13}")
    print(f"{'TE GroupedLinear':18} {fwd_te:10.3f} {bwd_te:10.3f} {fwd_te + bwd_te:13.3f}")
    print(f"{'SonicMoEExperts':18} {fwd_s :10.3f} {bwd_s :10.3f} {fwd_s + bwd_s :13.3f}")
    print(f"{'speedup (TE/S)':18} {fwd_te/fwd_s:9.2f}x {bwd_te/bwd_s:9.2f}x "
          f"{(fwd_te+bwd_te)/(fwd_s+bwd_s):12.2f}x")
    print("=" * 72)


if __name__ == "__main__":
    main()
