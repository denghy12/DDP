#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
GPUS="${GPUS:-2,3,4}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
NATIVE_INIT="${EMOT_NET_NATIVE_INIT:-${ROOT}/pretrained/emot_net/emot_net_native_init_v0.1.pth}"
CCIM_DICTIONARY="${CCIM_DICTIONARY:-${ROOT}/pretrained/ccim/emotic_task0_places365_k256_v0.1.pth}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "EMOT_NET_CCIM_FT_TRACK_B_V0_1" ]] || exit 2
[[ "${GPUS}" == "2,3,4" ]] || { echo "Frozen formal execution requires physical GPUs 2,3,4" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || exit 2
[[ -z "$(git status --porcelain)" ]] || { echo "Formal worker requires a clean worktree" >&2; exit 2; }

STATE_DIR="${RUN_OUTPUT_ROOT}/runtime_logs"
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
SUMMARY="${RUN_OUTPUT_ROOT}/formal_seed0_summary.json"
RUNTIME_LOG="${STATE_DIR}/seed0_gpu234.log"
mkdir -p "${STATE_DIR}"

set +e
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
RUN_ID="${RUN_ID}" SEED=0 GPU="${GPUS}" PYTHON="${PYTHON}" \
DATA_ROOT="${DATA_ROOT}" EMOT_NET_NATIVE_INIT="${NATIVE_INIT}" \
CCIM_DICTIONARY="${CCIM_DICTIONARY}" PROTOCOL="${PROTOCOL}" \
OUTPUT_ROOT="${RUN_OUTPUT_ROOT}" REPORTING_SPLIT=test \
TRAIN_BATCH_SIZE=52 EVAL_BATCH_SIZE=16 WORKERS=8 \
EMOT_NET_CCIM_TOWER_MODEL_PARALLEL=1 \
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION}" \
  bash "${SCRIPT_DIR}/run_emot_net_ccim_ft_baseline.sh" >"${RUNTIME_LOG}" 2>&1
RUN_RC=$?
set -e

if [[ "${RUN_RC}" -ne 0 ]]; then
  printf 'seed=0\nphysical_gpus=%s\nexit_code=%s\nlog=%s\n' \
    "${GPUS}" "${RUN_RC}" "${RUNTIME_LOG}" >"${STATE_DIR}/seed0.failed"
  exit "${RUN_RC}"
fi
printf 'seed=0\nphysical_gpus=%s\nexit_code=0\nlog=%s\n' \
  "${GPUS}" "${RUNTIME_LOG}" >"${STATE_DIR}/seed0.done"

"${PYTHON}" "${SCRIPT_DIR}/validate_emot_net_ccim_ft_formal_result.py" \
  --run-root "${RUN_OUTPUT_ROOT}" --run-id "${RUN_ID}" \
  --expected-git-commit "${EXPECTED_GIT_COMMIT}" --output "${SUMMARY}"

"${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  --run-root "${RUN_OUTPUT_ROOT}" --run-id "${RUN_ID}" \
  --expected-bundles 1 --extra "${PREFLIGHT_DIR}" \
  --extra "${STATE_DIR}" --extra "${SUMMARY}"

echo "Locked EMOT-Net+CCIM-FT seed 0 completed and validated"
echo "Archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
