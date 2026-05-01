#!/bin/bash
# ==============================================================================
# run_bench_E_EP.sh — wrapper for benchmarks/bench_E_EP_single_gpu.py
# ==============================================================================
#
# Runs a single-GPU sweep: 3 configs (128E / 256E / 384E) × 3 EP slicings
# (1 / 8 / 16). Logs everything to a timestamped file and prints just the
# final summary table at the end.
#
# Where to run
# ------------
#   * Inside the sonic-moe-test container (recommended), or any env that has:
#       - megatron-core 0.17.0
#       - transformer-engine 2.x
#       - quack-kernels 0.4.0
#       - sonic-moe (editable install)
#   * Single GPU is enough; 80 GB H100 fits all 9 cells.
#
# Quick run inside the existing container:
#   docker exec sonic-moe-test bash /workspace/sonic-moe/benchmarks/run_bench_E_EP.sh
#
# Or directly on the host (if your env has the deps):
#   bash benchmarks/run_bench_E_EP.sh
#
# Pass-through args go to the python script:
#   bash benchmarks/run_bench_E_EP.sh --T 4096 --warmup 15
#   bash benchmarks/run_bench_E_EP.sh --skip-ep1
# ==============================================================================

set -uo pipefail

# ---- Locate the bench script (works whether you cd or not) -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_PY="${SCRIPT_DIR}/bench_E_EP_single_gpu.py"

if [ ! -f "${BENCH_PY}" ]; then
    echo "ERROR: ${BENCH_PY} not found." >&2
    exit 1
fi

# ---- Output paths -----------------------------------------------------------
TS="$(date +%Y%m%d-%H%M%S)"
LOGDIR="${LOGDIR:-/tmp}"
LOGFILE="${LOGDIR}/bench_E_EP_${TS}.log"
CSVFILE="${CSVFILE:-${LOGDIR}/bench_E_EP_${TS}.csv}"

# ---- Environment ------------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"     # Hopper SM90
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# Megatron's MoELayer requires torch.distributed even at world_size=1; the
# python script will set these defaults itself, but we set them here too in
# case anyone has stale values exported.
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT 2>/dev/null || true
# Make CUDA errors easier to debug if something goes wrong.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# ---- Pre-flight check -------------------------------------------------------
echo "================================================================"
echo "  bench_E_EP_single_gpu sweep   ($(date))"
echo "================================================================"
echo "  Repo root: ${REPO_ROOT}"
echo "  Bench    : ${BENCH_PY}"
echo "  Log file : ${LOGFILE}"
echo "  CSV out  : ${CSVFILE}"
echo "  GPU(s)   :"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader \
  | sed 's/^/    /'
echo "  Python   : $(python --version 2>&1)"
echo "  Args     : $*"
echo "================================================================"
echo

# Cancel any leftover quack autotune from a previous run.
# (Quack caches results, but a half-written cache file can crash the next run.)
QUACK_CACHE_DIR="${QUACK_CACHE_DIR:-${HOME}/.quack/cache}"
echo "  Quack cache dir: ${QUACK_CACHE_DIR}"
if [ -d "${QUACK_CACHE_DIR}" ]; then
    BAD=$(find "${QUACK_CACHE_DIR}" -name "*.tmp" 2>/dev/null | wc -l)
    if [ "${BAD}" -gt 0 ]; then
        echo "  -> cleaning ${BAD} stale temp files"
        find "${QUACK_CACHE_DIR}" -name "*.tmp" -delete
    fi
fi
echo

# ---- Run ---------------------------------------------------------------------
START=$(date +%s)
python "${BENCH_PY}" --csv "${CSVFILE}" "$@" 2>&1 | tee "${LOGFILE}"
RC=${PIPESTATUS[0]}
END=$(date +%s)

echo
echo "================================================================"
echo "  done in $((END - START))s   exit_code=${RC}"
echo "  log = ${LOGFILE}"
echo "  csv = ${CSVFILE}"
echo "================================================================"

# ---- Extract just the final summary block ------------------------------------
echo
echo "=========== final summary table (parsed from log) ==========="
awk '/^# Final summary/{flag=1} flag' "${LOGFILE}" | sed -n '1,80p'
echo "============================================================="

# ---- Quick CSV preview -------------------------------------------------------
if [ -s "${CSVFILE}" ]; then
    echo
    echo "=========== CSV preview (head) ==========="
    head -1 "${CSVFILE}"
    echo "------------------------------------------"
    tail -n +2 "${CSVFILE}" | column -t -s ','
    echo "==========================================="
fi

exit ${RC}
