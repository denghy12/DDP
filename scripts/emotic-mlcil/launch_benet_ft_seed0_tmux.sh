#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_benet_ft_seed0_val}"
RUN_ID="${RUN_ID:-benet_ft_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
SOURCE_ROOT="${BENET_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/benet_release_b86747e}"
PRETRAINED_WEIGHTS="${BENET_PRETRAINED_WEIGHTS:-${SOURCE_ROOT}/models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth}"
OUTPUT_BASE="${BENET_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/benet_ft_track_b_v0.1}"
TRAIN_BATCH_SIZE="${BENET_TRAIN_BATCH_SIZE:-24}"
EVAL_BATCH_SIZE="${BENET_EVAL_BATCH_SIZE:-8}"
WORKERS="${BENET_WORKERS:-0}"
RUN_GPU_SMOKE="${BENET_RUN_GPU_SMOKE:-1}"
CPU_THREADS="${BENET_CPU_THREADS:-2}"

[[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "BENET_CPU_THREADS must be a positive integer" >&2; exit 2; }
# Export for preflight.  The same values are also embedded in the tmux command
# because an already-running tmux server may retain an older environment.
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"

tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${SOURCE_ROOT}/lib/models/BENet.py" ]] || { echo "Missing fixed BENet source: ${SOURCE_ROOT}" >&2; exit 2; }
[[ -s "${PRETRAINED_WEIGHTS}" ]] || { echo "Missing official HigherHRNet weights: ${PRETRAINED_WEIGHTS}" >&2; exit 2; }
[[ "${TRAIN_BATCH_SIZE}" == "24" ]] || { echo "Registered BENet batch size is 24" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "BENet validation requires a clean Git worktree" >&2; exit 2; }

RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_DIR="${OUTPUT_BASE}/_launcher_logs"
PREFLIGHT_LOG="${LOG_DIR}/${RUN_ID}_preflight.log"
ORACLE_JSON="${LOG_DIR}/${RUN_ID}_upstream_oracle.json"
mkdir -p "${LOG_DIR}"

set +e
(
set -e
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil.registry import method_names
from benchmarks.emotic_mlcil.runner import CORE_RUNTIME_VERSION
if CORE_RUNTIME_VERSION != "0.11.0":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if "benet_ft" not in method_names():
    raise RuntimeError("BENet-FT is not registered")
print({"runtime": CORE_RUNTIME_VERSION, "methods": tuple(method_names())})
PY
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest tests.test_emotic_task_adapter_bank tests.test_ddp_internal_adapter tests.test_ddp_prompt_free_auxiliary
"${PYTHON}" "${SCRIPT_DIR}/compare_benet_upstream_reference.py" --source-root "${SOURCE_ROOT}" --output "${ORACLE_JSON}"
if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT_DIR}/smoke_benet_ft_training.py" \
    --source-root "${SOURCE_ROOT}" --pretrained-weights "${PRETRAINED_WEIGHTS}" --batch-size "${TRAIN_BATCH_SIZE}"
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
[[ "${PREFLIGHT_RC}" -eq 0 ]] || { echo "BENet-FT preflight failed: ${PREFLIGHT_RC}" >&2; exit "${PREFLIGHT_RC}"; }

printf -v command \
  'cd %q && RUN_ID=%q SEED=0 GPU=%q PYTHON=%q DATA_ROOT=%q BENET_SOURCE_ROOT=%q BENET_PRETRAINED_WEIGHTS=%q OUTPUT_ROOT=%q REPORTING_SPLIT=val TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q BENET_CPU_THREADS=%q OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q NUMEXPR_NUM_THREADS=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; package_code=not_run; if [[ "$code" -eq 0 ]]; then %q %q --run-root %q --run-id %q --expected-bundles 1 --extra %q; package_code=$?; fi; echo BENET_FT_EXIT_CODE=$code; echo DOWNLOAD_PACKAGE_EXIT_CODE=$package_code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" "${SOURCE_ROOT}" \
  "${PRETRAINED_WEIGHTS}" "${RUN_ROOT}" "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${WORKERS}" \
  "${CPU_THREADS}" "${CPU_THREADS}" "${CPU_THREADS}" "${CPU_THREADS}" "${CPU_THREADS}" \
  "${SCRIPT_DIR}/run_benet_ft_baseline.sh" "${LOG_DIR}/${RUN_ID}.log" \
  "${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" "${RUN_ROOT}" "${RUN_ID}" "${ORACLE_JSON}"

tmux new-session -d -s "${SESSION}" -n "benet_ft_seed0_g${GPU}" "${command}"
echo "Started BENet-FT Track-B validation session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Download: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
