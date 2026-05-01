# SonicMoE vs TransformerEngine 对比 benchmark

`bench_compare.py` + `sweep_compare.sh` 两个脚本配合使用。

## 一、脚本职责

| 文件 | 用途 |
|---|---|
| `bench_compare.py` | 单 combo bench：1 个 (mode, name, T, H, I, E, K) → 1 行 pipe-delimited 输出 |
| `sweep_compare.sh` | 编排脚本：用 bash 循环把多个 combo 用<b>独立子进程</b>跑一遍（OOM/CUDA 错误隔离），最后调用渲染器 |
| `_render_sweep_table.py` | 把 sweep log 渲染成可读表格（per model 一张表，含 verify ✓/✗） |

## 二、4 种 backend mode

```
sonic-bf16   SonicMoE BF16（待测）
te-bf16      TransformerEngine GroupedLinear BF16
te-fp16      TransformerEngine GroupedLinear FP16
te-fp8       TransformerEngine GroupedLinear + fp8_autocast (Hybrid: E4M3 fwd, E5M2 grad)
```

每个 backend 都有<b>对应的 torch 参考实现</b>用于 `--verify` 元素级 atol+rtol 判定：

| mode | torch reference 怎么来 | 容差（atol, rtol） |
|---|---|---|
| sonic-bf16 | `MoE(...).forward(x, kernel_backend_moe=KernelBackendMoE.torch)` —— SonicMoE 自带的 per-expert F.linear loop | (1.4e-2, 2e-2) |
| te-bf16    | 从 `te.GroupedLinear` 抽取 `weight0..weight{E-1}`，用 plain torch 重写 forward | (1.4e-2, 2e-2) |
| te-fp16    | 同上 | (5e-3, 1e-2) |
| te-fp8     | 同上（FP8 forward vs 用 BF16 权重的 torch ref，比较 FP8 数值误差） | (5e-2, 5e-2) |

## 三、单 combo 用法

```bash
python benchmarks/bench_compare.py \
    --mode sonic-bf16 --name OLMoE \
    --T 32768 --H 2048 --I 1024 --E 64 --K 8 \
    --verify
```

输出（一行，pipe-delimited，可直接 grep / awk 解析）：

```
OLMoE|T=32768|sonic-bf16|5.68ms|517.0TF|vmax=9.08e-07|head=6.49e-05|verify=PASS
```

字段说明：
- `vmax`：max abs(y_test − y_ref) over all elements
- `head`：vmax / atol，<1 表示在 atol 范围内（headroom）
- `verify=PASS/FAIL`：元素级 `|diff| <= atol + rtol·|y_ref|` 判据

加 `--include-backward` 同时计 forward+backward 时间，FLOPs 按 3× 计。

## 四、全 sweep 用法

```bash
bash benchmarks/sweep_compare.sh                # forward only + verify (默认)
bash benchmarks/sweep_compare.sh --backward     # forward + backward
bash benchmarks/sweep_compare.sh --no-verify    # 跳过 correctness check
```

默认配置矩阵（编辑 sweep_compare.sh 顶部 CONFIGS / T_LIST / MODES 修改）：

| Configs | T values | Modes | 总组合 |
|---|---|---|---|
| 6 个真实开源 MoE | 5 档（8K/16K/32K/64K/256K） | 4 backend | 120 |

每个组合 isolated subprocess，单 combo timeout 600s（`TIMEOUT_SECS` 改）。最后自动调 renderer 输出表格。

## 五、环境变量

```bash
TORCH_CUDA_ARCH_LIST=9.0       # SM90 = Hopper；Blackwell 用 10.0
PYTORCH_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0         # 选 GPU
TIMEOUT_SECS=600               # per-combo 超时
LOGFILE=/tmp/sweep_compare.log # 原始日志
```

## 六、容器使用建议

NGC PyTorch 镜像自带 TE：
```bash
docker run -d --name moe-bench --gpus '"device=0"' --shm-size=16g --ipc=host \
  -v $(pwd):/workspace/sonic-moe -w /workspace/sonic-moe \
  nvcr.io/nvidia/pytorch:25.10-py3 tail -f /dev/null

docker exec moe-bench bash -lc 'pip install -e . && bash benchmarks/sweep_compare.sh'
```
