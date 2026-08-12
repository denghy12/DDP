#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
GPU="${GPU:-7}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
SOURCE_ROOT="${BENET_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/benet_official}"
PRETRAINED_WEIGHTS="${BENET_PRETRAINED_WEIGHTS:-${SOURCE_ROOT}/models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"
CPU_THREADS="${BENET_CPU_THREADS:-2}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "BENET_FT_TRACK_B_V0_1" ]] || {
  echo "Invalid BENet-FT configuration-lock confirmation" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "BENet-FT formal worker is not running the frozen commit" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "BENet-FT formal worker requires a clean Git worktree" >&2
  exit 2
}

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"

STATE_DIR="${RUN_OUTPUT_ROOT}/runtime_logs"
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
RUNTIME_LOG="${STATE_DIR}/seed0_gpu${GPU}.log"
mkdir -p "${STATE_DIR}"

set +e
RUN_ID="${RUN_ID}" \
SEED=0 \
GPU="${GPU}" \
PYTHON="${PYTHON}" \
DATA_ROOT="${DATA_ROOT}" \
BENET_SOURCE_ROOT="${SOURCE_ROOT}" \
BENET_PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS}" \
PROTOCOL="${PROTOCOL}" \
OUTPUT_ROOT="${RUN_OUTPUT_ROOT}" \
REPORTING_SPLIT=test \
TRAIN_BATCH_SIZE=24 \
EVAL_BATCH_SIZE=8 \
WORKERS=0 \
BENET_CPU_THREADS="${CPU_THREADS}" \
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION}" \
  bash "${SCRIPT_DIR}/run_benet_ft_baseline.sh" >"${RUNTIME_LOG}" 2>&1
RUN_RC=$?
set -e

if [[ "${RUN_RC}" -ne 0 ]]; then
  printf 'seed=0\nphysical_gpu=%s\nexit_code=%s\nlog=%s\n' \
    "${GPU}" "${RUN_RC}" "${RUNTIME_LOG}" >"${STATE_DIR}/seed0.failed"
  echo "BENet-FT seed 0 failed with exit code ${RUN_RC}; see ${RUNTIME_LOG}" >&2
  exit "${RUN_RC}"
fi
printf 'seed=0\nphysical_gpu=%s\nexit_code=0\nlog=%s\n' \
  "${GPU}" "${RUNTIME_LOG}" >"${STATE_DIR}/seed0.done"

"${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --expected-bundles 1 \
  --extra "${PREFLIGHT_DIR}" \
  --extra "${STATE_DIR}"

echo "Locked BENet-FT seed 0 completed"
echo "Archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
