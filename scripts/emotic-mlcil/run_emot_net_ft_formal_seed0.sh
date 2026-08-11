#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
NATIVE_INIT="${EMOT_NET_NATIVE_INIT:-${ROOT}/pretrained/emot_net/emot_net_native_init_v0.1.pth}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "EMOT_NET_FT_TRACK_B_V0_1" ]] || {
  echo "Invalid EMOT-Net-FT configuration-lock confirmation" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "EMOT-Net-FT formal worker is not running the frozen commit" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "EMOT-Net-FT formal worker requires a clean Git worktree" >&2
  exit 2
}

STATE_DIR="${RUN_OUTPUT_ROOT}/runtime_logs"
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
FORMAL_SUMMARY="${RUN_OUTPUT_ROOT}/formal_seed0_summary.json"
RUNTIME_LOG="${STATE_DIR}/seed0_gpu${GPU}.log"
mkdir -p "${STATE_DIR}"

set +e
RUN_ID="${RUN_ID}" \
SEED=0 \
GPU="${GPU}" \
PYTHON="${PYTHON}" \
DATA_ROOT="${DATA_ROOT}" \
EMOT_NET_NATIVE_INIT="${NATIVE_INIT}" \
PROTOCOL="${PROTOCOL}" \
OUTPUT_ROOT="${RUN_OUTPUT_ROOT}" \
REPORTING_SPLIT=test \
TRAIN_BATCH_SIZE=52 \
EVAL_BATCH_SIZE=16 \
WORKERS=0 \
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION}" \
  bash "${SCRIPT_DIR}/run_emot_net_ft_baseline.sh" >"${RUNTIME_LOG}" 2>&1
RUN_RC=$?
set -e

if [[ "${RUN_RC}" -ne 0 ]]; then
  printf 'seed=0\nphysical_gpu=%s\nexit_code=%s\nlog=%s\n' \
    "${GPU}" "${RUN_RC}" "${RUNTIME_LOG}" >"${STATE_DIR}/seed0.failed"
  echo "EMOT-Net-FT seed 0 failed with exit code ${RUN_RC}; see ${RUNTIME_LOG}" >&2
  exit "${RUN_RC}"
fi
printf 'seed=0\nphysical_gpu=%s\nexit_code=0\nlog=%s\n' \
  "${GPU}" "${RUNTIME_LOG}" >"${STATE_DIR}/seed0.done"

"${PYTHON}" "${SCRIPT_DIR}/validate_emot_net_ft_formal_result.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --expected-git-commit "${EXPECTED_GIT_COMMIT}" \
  --output "${FORMAL_SUMMARY}"

"${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --expected-bundles 1 \
  --extra "${PREFLIGHT_DIR}" \
  --extra "${STATE_DIR}" \
  --extra "${FORMAL_SUMMARY}"

echo "Locked EMOT-Net-FT seed 0 completed and validated"
echo "Archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
