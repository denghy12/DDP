#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
GPUS="${GPUS:-0 0 0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
AGCN_WORD_EMBEDDINGS="${AGCN_WORD_EMBEDDINGS:-${ROOT}/pretrained/agcn/emotic_glove_6b_300d.json}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "AGCN_TRACK_A_V0_1" ]] || {
  echo "Invalid AGCN configuration-lock confirmation" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "AGCN formal worker is not running the frozen commit" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "AGCN formal worker requires a clean Git worktree" >&2
  exit 2
}
read -r -a gpu_values <<< "${GPUS}"
[[ "${#gpu_values[@]}" -eq 3 ]] || {
  echo "GPUS must provide one physical GPU assignment per seed" >&2
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

  echo "Starting locked AGCN seed ${seed} on physical GPU ${gpu}"
  set +e
  RUN_ID="${RUN_ID}" \
  SEED="${seed}" \
  GPU="${gpu}" \
  PYTHON="${PYTHON}" \
  DATA_ROOT="${DATA_ROOT}" \
  CLIP_MODEL_PATH="${CLIP_MODEL_PATH}" \
  AGCN_WORD_EMBEDDINGS="${AGCN_WORD_EMBEDDINGS}" \
  PROTOCOL="${PROTOCOL}" \
  OUTPUT_ROOT="${RUN_OUTPUT_ROOT}" \
  REPORTING_SPLIT=test \
  TRAIN_BATCH_SIZE=8 \
  EVAL_BATCH_SIZE=32 \
  WORKERS=0 \
  CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION}" \
    bash "${SCRIPT_DIR}/run_agcn_baseline.sh" >"${log_path}" 2>&1
  code=$?
  set -e

  if [[ "${code}" -eq 0 ]]; then
    printf 'seed=%s\nphysical_gpu=%s\nphase=three_seed_single_gpu_parallel\nexit_code=0\nlog=%s\n' \
      "${seed}" "${gpu}" "${log_path}" >"${marker_prefix}.done"
    echo "Completed locked AGCN seed ${seed} on GPU ${gpu}"
    return 0
  fi
  printf 'seed=%s\nphysical_gpu=%s\nphase=three_seed_single_gpu_parallel\nexit_code=%s\nlog=%s\n' \
    "${seed}" "${gpu}" "${code}" "${log_path}" >"${marker_prefix}.failed"
  echo "AGCN seed ${seed} failed with exit code ${code}; see ${log_path}" >&2
  return "${code}"
}

pids=()
for seed in 0 1 2; do
  run_seed "${seed}" "${gpu_values[seed]}" &
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
  echo "At least one AGCN seed failed; aggregation and packaging are blocked" >&2
  exit 1
fi
for seed in 0 1 2; do
  [[ -f "${STATE_DIR}/seed${seed}.done" ]] || {
    echo "Missing completion marker for AGCN seed ${seed}" >&2
    exit 1
  }
done

"${PYTHON}" "${SCRIPT_DIR}/validate_agcn_formal_results.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --seeds 0 1 2 \
  --expected-git-commit "${EXPECTED_GIT_COMMIT}" \
  --output "${FORMAL_SUMMARY}"

"${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --expected-bundles 3 \
  --extra "${PREFLIGHT_DIR}" \
  --extra "${STATE_DIR}" \
  --extra "${FORMAL_SUMMARY}"

echo "All three locked AGCN seeds completed and validated"
echo "Archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
