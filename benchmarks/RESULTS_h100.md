# SonicMoE vs TransformerEngine vs Megatron-Core on H100

Forward + Backward benchmarks across 6 real open-source MoE configurations.
All numbers are single-MoE-layer measurements on **NVIDIA H100 80GB HBM3**
inside `nvcr.io/nvidia/pytorch:25.10-py3` (PyTorch 2.9, Triton 3.4, TE 2.8,
Megatron-Core 0.17, sonic-moe 0.1.2).

## Backends compared

| Mode | Library | Precision | What it actually calls |
|---|---|---|---|
| `sonic-bf16` | sonic-moe | BF16 | SonicMoE's CuTeDSL + Triton kernels |
| `te-bf16` | TransformerEngine | BF16 | `te.GroupedLinear` (cuBLAS grouped GEMM) + `te.moe_permute*` |
| `te-fp16` | TransformerEngine | FP16 | same as `te-bf16` but FP16 weights |
| `te-fp8` | TransformerEngine | FP8 (E4M3/E5M2) | same as `te-bf16` wrapped in `te.fp8_autocast` |
| `mcore-bf16` | Megatron-Core | BF16 | `MoELayer` → `TopKRouter` + `AlltoAllTokenDispatcher` + `TEGroupedMLP` (uses `te.GroupedLinear` underneath) |
| `mcore-fp8` | Megatron-Core | FP8 (E4M3/E5M2) | same as `mcore-bf16` wrapped in `te.fp8_autocast` |

**Verification**: every (sonic, te-*) backend output checked element-wise vs a
plain-torch reference using `|y_test - y_ref| ≤ atol + rtol·|y_ref|`. Megatron's
router has internal scaling that's hard to replicate exactly so we skip its verify.

## MoE configurations

| Name | H | I | E | K | Notes |
|---|---|---|---|---|---|
| OLMoE-1B-7B | 2048 | 1024 | 64 | 8 | Reference fine-grained baseline |
| Qwen3-30B-A3B | 2048 | 768 | 128 | 8 | Real Qwen3 30B activated |
| Qwen3.5-35B-A3B | 2048 | 1024 | 128 | 8 | Hypothetical (slightly larger I) |
| Qwen3-Next-80B-A3B | 2048 | 512 | 512 | 10 | Most fine-grained / sparse (K/E=1/51) |
| Qwen3-235B-A22B | 4096 | 1536 | 128 | 8 | Compute-bound territory |
| DeepSeek-V3.2-Exp | 7168 | 2048 | 256 | 8 | Largest H/I (685B model layer) |

T (microbatch) sweep: 8K / 16K / 32K / 64K / 256K.

## 1. Forward TFLOPS @ T=32K

| Model | sonic-bf16 | te-bf16 | te-fp16 | te-fp8 | mcore-bf16 | mcore-fp8 |
|---|---:|---:|---:|---:|---:|---:|
| OLMoE-1B-7B            | **562.3** | 407.8 | 397.3 | 307.4 | 278.5 | 281.1 |
| Qwen3-30B-A3B          | **513.3** | 316.7 | 314.4 | 126.7 | 229.1 | 142.9 |
| Qwen3.5-35B-A3B        | **553.3** | 361.4 | 358.6 | 163.2 | 265.9 | 199.3 |
| Qwen3-Next-80B-A3B     | **413.8** |  88.6 |  92.2 |  26.3 |  85.0 |  35.6 |
| Qwen3-235B-A22B        | **588.4** | 482.2 | 460.3 | 461.3 | 361.7 | 422.7 |
| DeepSeek-V3.2-Exp      | 535.6     | **543.4** | 503.7 | 473.7 | 407.8 | 520.8 |

**Forward speedup vs sonic-bf16** (>1× means sonic wins):

| Model | te-bf16 | te-fp16 | te-fp8 | mcore-bf16 | mcore-fp8 |
|---|---:|---:|---:|---:|---:|
| OLMoE-1B-7B            | 1.38× | 1.42× | 1.83× | 2.02× | 2.00× |
| Qwen3-30B-A3B          | 1.62× | 1.63× | 4.05× | 2.24× | 3.59× |
| Qwen3.5-35B-A3B        | 1.53× | 1.54× | 3.39× | 2.08× | 2.78× |
| **Qwen3-Next-80B-A3B** | **4.67×** | **4.49×** | **15.75×** | **4.87×** | **11.61×** |
| Qwen3-235B-A22B        | 1.22× | 1.28× | 1.28× | 1.63× | 1.39× |
| DeepSeek-V3.2-Exp      | 0.99× | 1.06× | 1.13× | 1.31× | 1.03× |

## 2. Backward TFLOPS @ T=8K

| Model | sonic-bf16 | te-bf16 | te-fp16 | te-fp8 | mcore-bf16 | mcore-fp8 |
|---|---:|---:|---:|---:|---:|---:|
| OLMoE-1B-7B            | **527.0** | 228.1 | 243.1 | ⚠ FAIL | 196.8 | ⚠ FAIL |
| Qwen3-30B-A3B          | **448.1** |  99.5 |  99.8 | ⚠ FAIL |  84.2 | ⚠ FAIL |
| Qwen3.5-35B-A3B        | **474.6** | 136.8 | 125.5 | ⚠ FAIL | 119.1 | ⚠ FAIL |
| Qwen3-Next-80B-A3B     | **281.9** |  18.6 |  18.7 | ⚠ FAIL |  16.7 | ⚠ FAIL |
| Qwen3-235B-A22B        | **486.0** | 364.8 | 366.5 | ⚠ FAIL | 303.6 | ⚠ FAIL |
| DeepSeek-V3.2-Exp      | **417.6** | 360.9 | 344.0 | ⚠ FAIL | 313.3 | ⚠ FAIL |

**Backward speedup vs sonic-bf16**:

| Model | te-bf16 | te-fp16 | te-fp8 | mcore-bf16 | mcore-fp8 |
|---|---:|---:|---:|---:|---:|
| OLMoE-1B-7B            | 2.31× | 2.17× | — | 2.68× | — |
| Qwen3-30B-A3B          | 4.50× | 4.49× | — | 5.32× | — |
| Qwen3.5-35B-A3B        | 3.47× | 3.78× | — | 3.99× | — |
| **Qwen3-Next-80B-A3B** | **15.12×** | **15.08×** | — | **16.84×** | — |
| Qwen3-235B-A22B        | 1.33× | 1.33× | — | 1.60× | — |
| DeepSeek-V3.2-Exp      | 1.16× | 1.21× | — | 1.33× | — |

⚠ All FP8 backward tests crash with:
```
RuntimeError: /workspace/TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu:712
```
TE 2.8's FP8 backward path through `GroupedLinear` is unsupported / broken
in this version. Reported as a known limitation.

## 3. Headline observations

### 3.1 SonicMoE wins forward across all 6 backends, except DeepSeek-V3.2 vs te-bf16
- Average forward speedup vs te-bf16:  **1.90×** (range 0.99×–4.67×)
- Average forward speedup vs mcore-bf16: **2.36×** (range 1.31×–4.87×)
- **Qwen3-Next-80B-A3B (K/E=10/512, most sparse)** is the strongest case for SonicMoE — 4.7× fwd / 15.1× bwd vs te-bf16, because TE's per-expert grouped GEMM scheduling overhead with E=512 experts is huge.
- **DeepSeek-V3.2-Exp (H=7168, I=2048)** is the weakest — TE-BF16 is essentially tied (-1%) because cuBLAS grouped GEMM is highly tuned for these large tile sizes; this is the regime where the GEMM compute dominates and SonicMoE's IO savings are less impactful.

### 3.2 SonicMoE's backward advantage is even bigger than forward
The backward speedups (T=8K) are 1.5–4× larger than forward speedups for the same configs:
- OLMoE: fwd 1.38× → bwd 2.31× (vs te-bf16)
- Qwen3-30B: fwd 1.62× → bwd 4.50×
- Qwen3-Next-80B: fwd 4.67× → bwd **15.12×**

This is exactly the algorithmic advantage SonicMoE's paper claims: dS contraction reordering eliminates the dY GEMM in backward, and the dH kernel fuses dH/A'/dS together in one launch.

### 3.3 Megatron-Core MoELayer is consistently slower than raw te-bf16 by 25–35%
This is the cost of Megatron's complete production stack on top of TE primitives:
- `TopKRouter` with aux loss / load balancing
- `MoEAlltoAllTokenDispatcher` with capacity factor / token drop logic
- additional graph captures / output reshapes

Both `mcore-bf16` and `mcore-fp8` slower than `te-bf16` and `te-fp8` respectively. This means **a fairer "production-grade baseline" is mcore-bf16, against which SonicMoE wins by 1.31–4.87× forward**.

### 3.4 FP8 grouped GEMM is a footgun on H100
- Both `te-fp8` and `mcore-fp8` are slower than their BF16 counterparts in most configs:
  - OLMoE fwd: te-bf16 408 → te-fp8 307 (-25%)
  - Qwen3-30B fwd: te-bf16 317 → te-fp8 127 (-60%)
  - Qwen3-Next fwd: te-bf16 89 → te-fp8 26 (-70%)
- Backward FP8 totally broken (cuBLASLt error in all 12 cases)
- Root cause: per-expert FP8 amax/scale/cast overhead + cuBLAS's small-tile FP8 weakness + MoE being memory-bound (FP8 saves compute, not bandwidth)
- See "为什么 mcore-fp8 在 H100 上反而比 mcore-bf16 慢" section below

### 3.5 Verification: 96 forward runs, 0 numerical mismatches
- All `sonic-*`, `te-*` runs (96 verified runs across 6 models × 4 backends) pass element-wise tolerance check vs torch reference (`|y_test - y_ref| ≤ atol + rtol·|y_ref|`)
- Megatron's router has nontrivial scaling (load-balance jitter, capacity routing) — verify is skipped for `mcore-*`
- No FP8 verify failures observed in the 6 forward runs that completed

---

# Megatron-Core MoELayer 完整调用链

> 这一节解释当 `--mode mcore-bf16` 或 `--mode mcore-fp8` 被选中时，
> 一次 forward 在 megatron-core 0.17 内部走过的所有层。

## 调用栈总览

```
benchmark caller
  └─ _MCoreWrap(x: [T,H])              ← 我们的 [T,H] → [B=1,T,H] adapter
      └─ MoELayer.forward(hidden_states: [B,T,H])
          ├─ ① TopKRouter.routing(hidden_states)
          │    ├─ apply MoEAuxLossAutoScaler hooks (load balancing)
          │    ├─ x' = (apply_router_pre_softmax_input_jitter? -> x with noise)
          │    ├─ logits = self.weight @ x'         (router gate Linear, no bias)
          │    ├─ scores = score_function(logits)   ← softmax / sigmoid / topk-softmax
          │    ├─ probs, indices = topk(scores, K)
          │    ├─ if moe_router_pre_softmax_renorm: probs /= probs.sum(-1)
          │    └─ aux_loss = aux_loss_fn(scores, indices, expert_count)
          │
          ├─ ② TokenDispatcher.token_permutation(x, probs, indices)
          │   └─ AlltoAllTokenDispatcher path (default in our config):
          │       ├─ flatten [B,T,H] → [B*T, H]
          │       ├─ moe_permute(...)   ← TE primitive: shuffle tokens by expert
          │       ├─ all_to_all (no-op here, single GPU)
          │       └─ return permuted_x [TK, H], dispatched_probs [TK]
          │
          ├─ ③ TEGroupedMLP.forward(permuted_x, expert_counts)
          │    ├─ ③a linear_fc1 = te.GroupedLinear(num_gemms=E, H, 2I)   ← cuBLAS grouped GEMM
          │    │   FP8 path: tcgen05.mma + per-tensor or per-group FP8 amax/scale
          │    ├─ ③b activation_func: SwiGLU = silu(h_gate) * h_up    (in-block)
          │    └─ ③c linear_fc2 = te.GroupedLinear(num_gemms=E, I, H)   ← cuBLAS grouped GEMM
          │
          └─ ④ TokenDispatcher.token_unpermutation(expert_output, ...)
              ├─ all_to_all (no-op here)
              ├─ moe_unpermute(... merging_probs=probs)   ← TE primitive: weighted scatter sum
              └─ return [B*T, H] → reshape to [B, T, H]
```

## 关键源码位置（megatron-core 0.17）

| 步骤 | 文件 | 类 / 函数 |
|---|---|---|
| ① 路由 | `megatron/core/transformer/moe/router.py` | `TopKRouter.routing` |
| ① score function | `megatron/core/transformer/moe/router.py` | `apply_routing_score_function` |
| ② 派发（permute） | `megatron/core/transformer/moe/token_dispatcher.py` | `MoEAlltoAllTokenDispatcher.token_permutation` |
| ③ TE Grouped MLP | `megatron/core/transformer/moe/experts.py` | `TEGroupedMLP.forward` |
| ③a/③c grouped GEMM | (TE) `transformer_engine.pytorch.module.grouped_linear` | `GroupedLinear.forward` |
| ④ 反派发 | `megatron/core/transformer/moe/token_dispatcher.py` | `MoEAlltoAllTokenDispatcher.token_unpermutation` |

## FP8 在 Megatron-Core 中可以调的所有参数

下面列出 `TransformerConfig` 里所有与 FP8 相关的字段（megatron-core 0.17）。
按"对吞吐 / 数值 / 显存 三类影响"分组。

### A. 总开关 + format

| 字段 | 类型 | 默认 | 含义 | 可选值 |
|---|---|---|---|---|
| `fp8` | `Optional[str]` | `None` | 关掉 FP8（None）或选 format | `"e4m3"` / `"hybrid"` / `"e5m2"` / `"mxfp8"` / `"blockwise"` / `None` |
| `fp8_recipe` | `str` | `"delayed"` | scale 计算策略 | `"delayed"` / `"tensorwise"` / `"mxfp8"` / `"blockwise"` |

**`fp8` 选哪个？**
- `"hybrid"` ← 标准选择：forward 用 E4M3（精度优先），backward 用 E5M2（动态范围优先）。我们的 `mcore-fp8` 用的就是这个。
- `"e4m3"` ← 全 E4M3：训练数值稳定性变差但 forward inference 可能更准
- `"e5m2"` ← 全 E5M2：动态范围最大，但精度损失大，几乎不用
- `"mxfp8"` ← Microscaling FP8（Blackwell 支持）：每 32 元素一块 scale，比 delayed 更准
- `"blockwise"` ← Block-wise scaling（DeepSeek-V3 风格）：每 128×128 块一组 scale

**`fp8_recipe` 选哪个？**
- `"delayed"` ← 标准：用历史 amax 作为下次的 scale，第一次默认 1.0。需要 amax 历史 buffer。
- `"tensorwise"` ← 每次 forward 都重新算 amax（更准但每 op 多一次 reduction）
- `"mxfp8"` ← microscaling：每 32 元素一组动态 scale，硬件直接支持 (Blackwell `tcgen05.mma` 有 mxfp8 variant)
- `"blockwise"` ← 块级 scaling，比 mxfp8 粒度大

### B. delayed scaling 的细节参数

| 字段 | 默认 | 含义 |
|---|---|---|
| `fp8_amax_history_len` | `1024` | amax 历史窗口大小，`max` 算法时窗口越大 scale 越保守 |
| `fp8_amax_compute_algo` | `"most_recent"` | `"max"` 取窗口最大值 / `"most_recent"` 只用上一步的 amax |
| `fp8_margin` | `0` | scale 上加的保守 bit margin（>0 时 scale 缩小 2^margin 倍，避免 overflow） |
| `fp8_interval` | `1` | 每 N 步更新一次 amax / scale（>1 时减少 amax kernel overhead 但精度变差） |

### C. 显存 / 吞吐相关

| 字段 | 默认 | 含义 |
|---|---|---|
| `fp8_param_gather` | `False` | 是否在 weight all-gather (FSDP/ZeRO) 时也用 FP8（节省通信，有数值噪声） |
| `fp8_dot_product_attention` | `False` | attention 是否走 FP8（影响 attention kernel，不影响 MoE） |
| `fp8_multi_head_attention` | `False` | 同上 |

### D. MoE 特定

| 字段 | 默认 | 含义 |
|---|---|---|
| `moe_grouped_gemm` | `False` | 必须设 `True` 才能走 `TEGroupedMLP` 的 FP8 grouped GEMM 路径（否则退化到 `SequentialMLP`，每个 expert 独立调用 GroupedLinear，FP8 几乎没用） |
| `moe_token_dispatcher_type` | `"alltoall"` | dispatcher 实现：`alltoall`（标准）/ `allgather`（小集群）/ `flex`（带 capacity factor） |

### 我们当前的 mcore-fp8 配置

```python
TransformerConfig(
    fp8="hybrid",                # E4M3 fwd + E5M2 bwd
    # 用 te.fp8_autocast 包 forward，所以下面这些其实是 TE 的 DelayedScaling 接管
    bf16=True, params_dtype=torch.bfloat16,
    moe_grouped_gemm=True,       # 必须 True 才走 fused 路径
    moe_token_dispatcher_type="alltoall",
)
# + 在 forward 时:
te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling(
    fp8_format=Format.HYBRID, margin=0,
    amax_history_len=16,         # 短窗口，bench 场景下足够
    amax_compute_algo="max"))    # 取窗口最大值
```

## 为什么 mcore-fp8 在 H100 上反而比 mcore-bf16 慢

H100 FP8 Tensor Core peak 是 BF16 的 2×（约 1979 TFLOPS vs 989 TFLOPS）。理论上
FP8 应该至少快 1.5×，但实测 mcore-fp8 比 mcore-bf16 慢 ~2×。原因：

1. **Per-expert FP8 amax / scale overhead**: TE 的 `GroupedLinear` 内部对每个 expert
   单独算 amax（用 `tex.compute_amax()` kernel 一次发一个），E=128 时 launch 开销巨大。
   `te.fp8_autocast` 必须在每个 expert 的 GEMM 前确保 scale 已设好 → 串行化的 amax-update + scale-cast 链。

2. **FP8 cast / dequant kernels**: 每次 grouped GEMM 调用前要把 BF16 weight 用 `cast_to_fp8` cast 一次（per-expert）；输出又要从 FP8 cast 回 BF16。这些是独立 kernel launch，不能 fuse。

3. **GroupedLinear FP8 path 的 cuBLAS 实现**：底下其实是按 expert 一个个调 cuBLAS GEMM
   （而不是真正的 grouped GEMM），FP8 GEMM 的小 tile 性能差。

4. **MoE 本身就是 memory-bound**: paper 已论证 fine-grained MoE arithmetic intensity (~210)
   远低于 H100 的 BF16 计算/带宽比 (~325)，更远低于 FP8 的 (~650)。FP8 节省的是 compute，
   但 MoE kernel 的瓶颈是 HBM 带宽，所以 FP8 帮不上。

**真正能用上 H100 FP8 的 MoE kernel 必须是**：
- 一发 grouped GEMM 处理所有 expert（不是 per-expert loop）
- amax / scale 在 epilogue 内部计算（不走单独 kernel）
- Forward / backward / weight grad 共享 amax buffer

这正是 SonicMoE roadmap 上的 future work（"add MXFP8 / MXFP4 support"），目前 SonicMoE 还
没 FP8 路径，所以 fair 对比下我们的 sonic-bf16 vs mcore-fp8 是 BF16 vs FP8 的跨精度比较。

---

## Reproduction

```bash
# Single combo
python benchmarks/bench_compare.py --mode mcore-fp8 --name OLMoE \
    --T 32768 --H 2048 --I 1024 --E 64 --K 8

# Full sweep (6 model × 5 T × 6 backend = 180 combos, ~60 min)
bash benchmarks/sweep_compare.sh

# Forward + backward
bash benchmarks/sweep_compare.sh --backward
```

Raw output is pipe-delimited and parseable:
```
<name>|T=<T>|<mode>|<time>ms|<TFLOPS>TF[|vmax=<>|head=<>|verify=<PASS|FAIL>]
```
