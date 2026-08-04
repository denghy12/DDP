#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "REPLAY_20C_TRACK_A_V0_1" ]] || {
  echo "Invalid replay configuration-lock confirmation" >&2
  exit 2
}
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: ${GPU}" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "Replay formal worker is not running the frozen commit" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Replay formal worker requires a clean Git worktree" >&2
  exit 2
}

STATE_DIR="${RUN_OUTPUT_ROOT}/runtime_logs"
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
FORMAL_SUMMARY="${RUN_OUTPUT_ROOT}/formal_seed_summary.json"
mkdir -p "${STATE_DIR}"

run_seed() {
  local method="$1"
  local seed="$2"
  local log_path="${STATE_DIR}/${method}_seed${seed}_gpu${GPU}.log"
  local marker_prefix="${STATE_DIR}/${method}_seed${seed}"
  local code

  echo "Starting locked ${method^^} seed ${seed} on physical GPU ${GPU}"
  set +e
  METHOD="${method}" \
  RUN_ID="${RUN_ID}" \
  SEED="${seed}" \
  GPU="${GPU}" \
  PYTHON="${PYTHON}" \
  DATA_ROOT="${DATA_ROOT}" \
  CLIP_MODEL_PATH="${CLIP_MODEL_PATH}" \
  PROTOCOL="${PROTOCOL}" \
  OUTPUT_ROOT="${RUN_OUTPUT_ROOT}" \
  REPORTING_SPLIT=test \
  TRAIN_BATCH_SIZE=32 \
  EVAL_BATCH_SIZE=64 \
  WORKERS=2 \
  CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION}" \
    bash "${SCRIPT_DIR}/run_replay_baseline.sh" >"${log_path}" 2>&1
  code=$?
  set -e

  if [[ "${code}" -eq 0 ]]; then
    printf 'method=%s\nseed=%s\nphysical_gpu=%s\nphase=three_seed_gpu0_wave\nexit_code=0\nlog=%s\n' \
      "${method}" "${seed}" "${GPU}" "${log_path}" >"${marker_prefix}.done"
    echo "Completed locked ${method^^} seed ${seed} on GPU ${GPU}"
    return 0
  fi
  printf 'method=%s\nseed=%s\nphysical_gpu=%s\nphase=three_seed_gpu0_wave\nexit_code=%s\nlog=%s\n' \
    "${method}" "${seed}" "${GPU}" "${code}" "${log_path}" >"${marker_prefix}.failed"
  echo "${method^^} seed ${seed} failed with exit code ${code}; see ${log_path}" >&2
  return "${code}"
}

run_method_wave() {
  local method="$1"
  local pids=()
  local overall_code=0
  local code

  echo "Starting three-process ${method^^} wave on GPU ${GPU}"
  for seed in 0 1 2; do
    run_seed "${method}" "${seed}" &
    pids+=("$!")
  done

  set +e
  for pid in "${pids[@]}"; do
    wait "${pid}"
    code=$?
    if [[ "${code}" -ne 0 ]]; then
      overall_code=1
    fi
  done
  set -e
  if [[ "${overall_code}" -ne 0 ]]; then
    echo "At least one ${method^^} seed failed; later waves and packaging are blocked" >&2
    return 1
  fi
  for seed in 0 1 2; do
    [[ -f "${STATE_DIR}/${method}_seed${seed}.done" ]] || {
      echo "Missing completion marker for ${method^^} seed ${seed}" >&2
      return 1
    }
  done
  echo "Completed three-process ${method^^} wave"
}

# A single process peaks near 5.8 GiB. Keep concurrency at three on the 24 GiB
# RTX 4090 and hand off automatically between methods instead of starting six.
run_method_wave er
run_method_wave prs

"${PYTHON}" "${SCRIPT_DIR}/validate_replay_formal_results.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --seeds 0 1 2 \
  --expected-git-commit "${EXPECTED_GIT_COMMIT}" \
  --output "${FORMAL_SUMMARY}"

"${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --expected-bundles 6 \
  --extra "${PREFLIGHT_DIR}" \
  --extra "${STATE_DIR}" \
  --extra "${FORMAL_SUMMARY}"

echo "All locked ER/PRS seeds completed and validated"
echo "Archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
