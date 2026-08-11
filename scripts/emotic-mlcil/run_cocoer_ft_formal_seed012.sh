#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
GPUS="${GPUS:?GPUS is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
RESNET_INIT="${COCOER_RESNET50_INIT:?COCOER_RESNET50_INIT is required}"
CLIP_RN50="${COCOER_CLIP_RN50:?COCOER_CLIP_RN50 is required}"
HEAD_CACHE="${COCOER_HEAD_CACHE:?COCOER_HEAD_CACHE is required}"
PROTOCOL="${PROTOCOL:?PROTOCOL is required}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?lock is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "COCOER_FT_TRACK_B_V0_1" ]] || {
  echo "Invalid CocoER-FT configuration lock" >&2; exit 2;
}
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "CocoER-FT formal worker is not on the frozen commit" >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "CocoER-FT formal worker requires a clean worktree" >&2; exit 2;
}
read -r -a gpu_values <<< "${GPUS}"
[[ "${#gpu_values[@]}" -eq 3 && "${gpu_values[0]}" != "${gpu_values[1]}" \
  && "${gpu_values[0]}" != "${gpu_values[2]}" \
  && "${gpu_values[1]}" != "${gpu_values[2]}" ]] || {
  echo "CocoER-FT requires three distinct GPUs" >&2; exit 2;
}

STATE_DIR="${RUN_OUTPUT_ROOT}/runtime_logs"
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
FORMAL_SUMMARY="${RUN_OUTPUT_ROOT}/formal_seed_summary.json"
mkdir -p "${STATE_DIR}"

run_seed() {
  local seed="$1" gpu="$2"
  local log="${STATE_DIR}/seed${seed}_gpu${gpu}.log"
  local marker="${STATE_DIR}/seed${seed}"
  local code
  echo "Starting locked CocoER-FT seed ${seed} on GPU ${gpu}"
  set +e
  RUN_ID="${RUN_ID}" SEED="${seed}" GPU="${gpu}" PYTHON="${PYTHON}" \
  DATA_ROOT="${DATA_ROOT}" COCOER_RESNET50_INIT="${RESNET_INIT}" \
  COCOER_CLIP_RN50="${CLIP_RN50}" COCOER_HEAD_CACHE="${HEAD_CACHE}" \
  PROTOCOL="${PROTOCOL}" OUTPUT_ROOT="${RUN_OUTPUT_ROOT}" \
  REPORTING_SPLIT=test TRAIN_BATCH_SIZE=64 EVAL_BATCH_SIZE=16 WORKERS=0 \
  CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION}" \
    bash "${SCRIPT_DIR}/run_cocoer_ft_baseline.sh" >"${log}" 2>&1
  code=$?
  set -e
  if [[ "${code}" -eq 0 ]]; then
    printf 'seed=%s\nphysical_gpu=%s\nexit_code=0\nlog=%s\n' \
      "${seed}" "${gpu}" "${log}" >"${marker}.done"
    echo "Completed locked CocoER-FT seed ${seed} on GPU ${gpu}"
    return 0
  fi
  printf 'seed=%s\nphysical_gpu=%s\nexit_code=%s\nlog=%s\n' \
    "${seed}" "${gpu}" "${code}" "${log}" >"${marker}.failed"
  echo "CocoER-FT seed ${seed} failed; see ${log}" >&2
  return "${code}"
}

pids=()
for seed in 0 1 2; do
  run_seed "${seed}" "${gpu_values[seed]}" &
  pids+=("$!")
done
overall=0
set +e
for pid in "${pids[@]}"; do
  wait "${pid}" || overall=1
done
set -e
[[ "${overall}" -eq 0 ]] || {
  echo "At least one CocoER-FT seed failed; aggregation blocked" >&2; exit 1;
}

"${PYTHON}" "${SCRIPT_DIR}/validate_cocoer_ft_formal_results.py" \
  --run-root "${RUN_OUTPUT_ROOT}" --run-id "${RUN_ID}" \
  --seeds 0 1 2 --gpus "${gpu_values[@]}" \
  --expected-git-commit "${EXPECTED_GIT_COMMIT}" --output "${FORMAL_SUMMARY}"

"${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  --run-root "${RUN_OUTPUT_ROOT}" --run-id "${RUN_ID}" \
  --expected-bundles 3 --extra "${PREFLIGHT_DIR}" \
  --extra "${STATE_DIR}" --extra "${FORMAL_SUMMARY}"

echo "All locked CocoER-FT seeds completed and were packaged"
echo "Archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
