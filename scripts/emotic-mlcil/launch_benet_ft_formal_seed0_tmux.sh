#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_benet_ft_formal_seed0}"
RUN_ID="${RUN_ID:-benet_ft_formal_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-7}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
SOURCE_ROOT="${BENET_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/benet_official}"
PRETRAINED_WEIGHTS="${BENET_PRETRAINED_WEIGHTS:-${SOURCE_ROOT}/models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
OUTPUT_BASE="${BENET_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/benet_ft_track_b_v0.1}"
CPU_THREADS="${BENET_CPU_THREADS:-2}"
MIN_FREE_GPU_MIB="${BENET_MIN_FREE_GPU_MIB:-15000}"
RUN_GPU_SMOKE="${BENET_RUN_GPU_SMOKE:-0}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "BENET_FT_TRACK_B_V0_1" ]] || {
  echo "Refusing held-out test: invalid BENet-FT configuration lock" >&2
  exit 2
}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
[[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid CPU thread count" >&2; exit 2; }
[[ "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "EXPECTED_GIT_COMMIT must be a full lowercase SHA" >&2; exit 2; }
[[ "${RUN_GPU_SMOKE}" == "0" || "${RUN_GPU_SMOKE}" == "1" ]] || { echo "BENET_RUN_GPU_SMOKE must be 0 or 1" >&2; exit 2; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${SOURCE_ROOT}/lib/models/BENet.py" ]] || { echo "Missing fixed BENet source: ${SOURCE_ROOT}" >&2; exit 2; }
[[ -s "${PRETRAINED_WEIGHTS}" ]] || { echo "Missing BENet initialization: ${PRETRAINED_WEIGHTS}" >&2; exit 2; }

CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "Refusing formal run: HEAD ${CURRENT_COMMIT} != frozen ${EXPECTED_GIT_COMMIT}" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || { echo "Formal run requires a clean Git worktree" >&2; exit 2; }

FREE_GPU_MIB="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "${FREE_GPU_MIB}" =~ ^[0-9]+$ && "${FREE_GPU_MIB}" -ge "${MIN_FREE_GPU_MIB}" ]] || {
  echo "GPU ${GPU} has ${FREE_GPU_MIB:-unknown} MiB free; ${MIN_FREE_GPU_MIB} MiB required" >&2
  exit 2
}

RUN_OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
[[ ! -e "${RUN_OUTPUT_ROOT}" ]] || { echo "Run output already exists: ${RUN_OUTPUT_ROOT}" >&2; exit 2; }
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
LOG_DIR="${OUTPUT_BASE}/_launcher_logs"
PREFLIGHT_LOG="${LOG_DIR}/${RUN_ID}_preflight.log"
LAUNCHER_LOG="${LOG_DIR}/${RUN_ID}.log"
ORACLE_JSON="${PREFLIGHT_DIR}/upstream_oracle.json"
mkdir -p "${PREFLIGHT_DIR}" "${LOG_DIR}"

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"

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
print({"runtime": CORE_RUNTIME_VERSION, "formal_seed": 0, "track": "B"})
PY
"${PYTHON}" -m unittest tests.emotic_mlcil.test_benet_ft tests.emotic_mlcil.test_benet_ft_reference_audit
"${PYTHON}" "${SCRIPT_DIR}/compare_benet_upstream_reference.py" \
  --source-root "${SOURCE_ROOT}" --output "${ORACLE_JSON}"
if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT_DIR}/smoke_benet_ft_training.py" \
    --source-root "${SOURCE_ROOT}" --pretrained-weights "${PRETRAINED_WEIGHTS}" --batch-size 24
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
[[ "${PREFLIGHT_RC}" -eq 0 ]] || { echo "BENet-FT formal preflight failed: ${PREFLIGHT_RC}" >&2; exit "${PREFLIGHT_RC}"; }
cp "${PREFLIGHT_LOG}" "${PREFLIGHT_DIR}/preflight.log"

printf -v command \
  'cd %q && RUN_ID=%q GPU=%q PYTHON=%q DATA_ROOT=%q BENET_SOURCE_ROOT=%q BENET_PRETRAINED_WEIGHTS=%q PROTOCOL=%q RUN_OUTPUT_ROOT=%q EXPECTED_GIT_COMMIT=%q CONFIGURATION_LOCKED_CONFIRMATION=%q BENET_CPU_THREADS=%q OMP_NUM_THREADS=%q MKL_NUM_THREADS=%q OPENBLAS_NUM_THREADS=%q NUMEXPR_NUM_THREADS=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; echo BENET_FT_FORMAL_SEED0_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" "${SOURCE_ROOT}" \
  "${PRETRAINED_WEIGHTS}" "${PROTOCOL}" "${RUN_OUTPUT_ROOT}" "${EXPECTED_GIT_COMMIT}" \
  "${CONFIGURATION_LOCKED_CONFIRMATION}" "${CPU_THREADS}" "${CPU_THREADS}" "${CPU_THREADS}" \
  "${CPU_THREADS}" "${CPU_THREADS}" "${SCRIPT_DIR}/run_benet_ft_formal_seed0.sh" "${LAUNCHER_LOG}"

tmux new-session -d -s "${SESSION}" -n "benet_ft_seed0_g${GPU}" "${command}"
echo "Started locked BENet-FT Track-B formal seed 0: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "GPU: ${GPU}; free GPU memory before launch: ${FREE_GPU_MIB} MiB"
echo "Frozen loader: train batch 24, eval batch 8, workers 0, CPU threads ${CPU_THREADS}"
echo "Runtime log: ${RUN_OUTPUT_ROOT}/runtime_logs/seed0_gpu${GPU}.log"
echo "Archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
