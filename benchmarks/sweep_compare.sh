#!/bin/bash
# Sweep SonicMoE vs TransformerEngine across multiple (model, T) combos.
#
# Each combo runs in an isolated python subprocess, so a CUDA error / OOM on
# one combo does NOT poison subsequent ones. The 4-mode comparison
# (sonic-bf16, te-bf16, te-fp16, te-fp8) is run for each combo.
#
# Each backend is also run with --verify against a torch reference (small T).
#
# Usage:
#     bash benchmarks/sweep_compare.sh             # default sweep, fwd-only, with verify
#     bash benchmarks/sweep_compare.sh --backward  # also time forward+backward
#     bash benchmarks/sweep_compare.sh --no-verify # skip correctness check
#
# Env knobs:
#     CUDA_VISIBLE_DEVICES   pick GPU(s)
#     TIMEOUT_SECS           per-combo timeout (default 600)
#     LOGFILE                where to dump raw lines (default /tmp/sweep_compare.log)

set -uo pipefail

# ---- locate the bench script ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY="${SCRIPT_DIR}/bench_compare.py"
RENDER_PY="${SCRIPT_DIR}/_render_sweep_table.py"
if [ ! -f "$BENCH_PY" ]; then
    echo "ERROR: bench_compare.py not found at $BENCH_PY" >&2
    exit 1
fi

LOGFILE="${LOGFILE:-/tmp/sweep_compare.log}"
TIMEOUT_SECS="${TIMEOUT_SECS:-600}"
INCLUDE_BWD=""
VERIFY="--verify"

for arg in "$@"; do
    case "$arg" in
        --backward)  INCLUDE_BWD="--include-backward" ;;
        --no-verify) VERIFY="" ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

if [ -n "$INCLUDE_BWD" ]; then echo "Mode: forward + backward"; else echo "Mode: forward only"; fi
if [ -n "$VERIFY" ];      then echo "Verify: ON  (each backend vs torch ref @ T=1024)"; else echo "Verify: off"; fi

# ---- configs: name|H|I|E|K ----
CONFIGS=(
  "OLMoE-1B-7B|2048|1024|64|8"
  "Qwen3-30B-A3B|2048|768|128|8"
  "Qwen3.5-35B-A3B|2048|1024|128|8"
  "Qwen3-Next-80B-A3B|2048|512|512|10"
  "Qwen3-235B-A22B|4096|1536|128|8"
  "DeepSeek-V3.2-Exp|7168|2048|256|8"
)

# ---- T values (microbatch in tokens) ----
T_LIST=(8192 16384 32768 65536 262144)

# ---- backends to compare ----
# mcore-* = Megatron-Core MoELayer (production NVIDIA, uses TE GroupedLinear underneath)
# Note: mcore needs torch.distributed init each subprocess (~3s extra warmup).
MODES=(sonic-bf16 te-bf16 te-fp16 te-fp8 mcore-bf16 mcore-fp8)

# ---- environment defaults ----
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"          # SM90 = Hopper
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

: > "$LOGFILE"
echo "Logging raw lines to $LOGFILE"
echo "===================================================================="

total=$(( ${#CONFIGS[@]} * ${#T_LIST[@]} * ${#MODES[@]} ))
i=0
start_ts=$(date +%s)

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME H I E K <<< "$entry"
    for T in "${T_LIST[@]}"; do
        for MODE in "${MODES[@]}"; do
            i=$((i+1))
            printf "[%2d/%2d] %-22s T=%6d %-12s ... " "$i" "$total" "$NAME" "$T" "$MODE"
            line=$(timeout "$TIMEOUT_SECS" python "$BENCH_PY" \
                --mode "$MODE" --name "$NAME" \
                --T "$T" --H "$H" --I "$I" --E "$E" --K "$K" \
                $VERIFY $INCLUDE_BWD 2>/dev/null \
                || echo "${NAME}|T=${T}|${MODE}|FAIL|TIMEOUT_OR_CRASH")
            echo "$line"
            echo "$line" >> "$LOGFILE"
        done
    done
done

end_ts=$(date +%s)
echo "===================================================================="
echo "Total time: $(( end_ts - start_ts ))s   ·   Raw log: $LOGFILE"
echo

# ---- pretty-print summary table ----
if [ -f "$RENDER_PY" ]; then
    python "$RENDER_PY" "$LOGFILE" || cat "$LOGFILE"
else
    cat "$LOGFILE"
fi
