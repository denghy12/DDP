#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
SEEDS="${SEEDS:-0 1 2}"
GPUS="${GPUS:-0 1 2}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "MULTI_LANE_TRACK_A_V0_1" ]] || {
  echo "Invalid MULTI-LANE configuration-lock confirmation" >&2
  exit 2
}
read -r -a seed_values <<< "${SEEDS}"
read -r -a gpu_values <<< "${GPUS}"
[[ "${seed_values[*]}" == "0 1 2" ]] || {
  echo "Formal MULTI-LANE seeds must be exactly: 0 1 2" >&2
  exit 2
}
[[ "${#gpu_values[@]}" -eq 3 ]] || {
  echo "GPUS must provide exactly one physical GPU per seed" >&2
  exit 2
}

STATE_DIR="${RUN_OUTPUT_ROOT}/runtime_logs"
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
FORMAL_SUMMARY="${RUN_OUTPUT_ROOT}/formal_seed_summary.json"
mkdir -p "${STATE_DIR}"

run_seed() {
  local seed="$1"
  local gpu="$2"
  local log_path="${STATE_DIR}/seed${seed}_gpu${gpu}.log"
  local marker_prefix="${STATE_DIR}/seed${seed}"
  local code

  echo "Starting locked MULTI-LANE seed ${seed} on physical GPU ${gpu}"
  set +e
  RUN_ID="${RUN_ID}" \
  SEED="${seed}" \
  GPU="${gpu}" \
  PYTHON="${PYTHON}" \
  DATA_ROOT="${DATA_ROOT}" \
  CLIP_MODEL_PATH="${CLIP_MODEL_PATH}" \
  PROTOCOL="${PROTOCOL}" \
  OUTPUT_ROOT="${RUN_OUTPUT_ROOT}" \
  REPORTING_SPLIT=test \
  TRAIN_BATCH_SIZE=64 \
  EVAL_BATCH_SIZE=64 \
  WORKERS=2 \
  CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION}" \
    bash "${SCRIPT_DIR}/run_multi_lane_baseline.sh" >"${log_path}" 2>&1
  code=$?
  set -e

  if [[ "${code}" -eq 0 ]]; then
    printf 'seed=%s\nphysical_gpu=%s\nexit_code=0\nlog=%s\n' \
      "${seed}" "${gpu}" "${log_path}" >"${marker_prefix}.done"
    echo "Completed locked MULTI-LANE seed ${seed} on physical GPU ${gpu}"
    return 0
  fi
  printf 'seed=%s\nphysical_gpu=%s\nexit_code=%s\nlog=%s\n' \
    "${seed}" "${gpu}" "${code}" "${log_path}" >"${marker_prefix}.failed"
  echo "MULTI-LANE seed ${seed} failed with exit code ${code}; see ${log_path}" >&2
  return "${code}"
}

pids=()
for index in "${!seed_values[@]}"; do
  run_seed "${seed_values[index]}" "${gpu_values[index]}" &
  pids+=("$!")
done

overall_code=0
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
  echo "At least one MULTI-LANE seed failed; validation and packaging blocked" >&2
  exit 1
fi
for seed in "${seed_values[@]}"; do
  [[ -f "${STATE_DIR}/seed${seed}.done" ]] || {
    echo "Missing completion marker for MULTI-LANE seed ${seed}" >&2
    exit 1
  }
done

"${PYTHON}" "${SCRIPT_DIR}/validate_multi_lane_formal_results.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --seeds "${seed_values[@]}" \
  --expected-git-commit "${EXPECTED_GIT_COMMIT}" \
  --output "${FORMAL_SUMMARY}"

"${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --expected-bundles 3 \
  --extra "${PREFLIGHT_DIR}" \
  --extra "${STATE_DIR}" \
  --extra "${FORMAL_SUMMARY}"

echo "All three locked MULTI-LANE seeds completed and validated"
echo "Archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
