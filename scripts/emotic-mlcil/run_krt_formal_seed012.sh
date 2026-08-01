#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-1}"
AUTO_RETRY_SEQUENTIAL="${AUTO_RETRY_SEQUENTIAL:-1}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

for value in "${GPU}" "${MAX_CONCURRENT_JOBS}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "GPU and MAX_CONCURRENT_JOBS must be non-negative integers" >&2
    exit 2
  }
done
[[ "${MAX_CONCURRENT_JOBS}" -gt 0 ]] || {
  echo "MAX_CONCURRENT_JOBS must be positive" >&2
  exit 2
}
[[ "${AUTO_RETRY_SEQUENTIAL}" == "0" || "${AUTO_RETRY_SEQUENTIAL}" == "1" ]] || {
  echo "AUTO_RETRY_SEQUENTIAL must be 0 or 1" >&2
  exit 2
}

read -r -a seed_values <<< "${SEEDS}"
[[ "${#seed_values[@]}" -gt 0 ]] || { echo "SEEDS is empty" >&2; exit 2; }
seen_seeds=" "
for seed in "${seed_values[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "Invalid seed: ${seed}" >&2; exit 2; }
  if [[ "${seen_seeds}" == *" ${seed} "* ]]; then
    echo "Duplicate seed: ${seed}" >&2
    exit 2
  fi
  seen_seeds+="${seed} "
done

STATE_DIR="${RUN_OUTPUT_ROOT}/runtime_logs"
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
FAILED_ATTEMPT_DIR="${RUN_OUTPUT_ROOT}/failed_attempts"
mkdir -p "${STATE_DIR}"

run_seed() {
  local seed="$1"
  local attempt="$2"
  local log_path="${STATE_DIR}/seed${seed}_${attempt}.log"
  local marker_prefix="${STATE_DIR}/seed${seed}"
  local code

  echo "Starting KRT formal seed ${seed}, attempt ${attempt}, GPU ${GPU}"
  set +e
  RUN_ID="${RUN_ID}" \
  SEED="${seed}" \
  GPU="${GPU}" \
  PYTHON="${PYTHON}" \
  DATA_ROOT="${DATA_ROOT}" \
  CLIP_MODEL_PATH="${CLIP_MODEL_PATH}" \
  PROTOCOL="${PROTOCOL}" \
  OUTPUT_ROOT="${RUN_OUTPUT_ROOT}" \
  REPORTING_SPLIT=test \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  WORKERS="${WORKERS}" \
  CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION}" \
    bash "${SCRIPT_DIR}/run_krt_baseline.sh" >"${log_path}" 2>&1
  code=$?
  set -e

  if [[ "${code}" -eq 0 ]]; then
    printf 'attempt=%s\nexit_code=0\n' "${attempt}" >"${marker_prefix}.done"
    echo "Completed KRT formal seed ${seed}, attempt ${attempt}"
  else
    printf 'attempt=%s\nexit_code=%s\nlog=%s\n' \
      "${attempt}" "${code}" "${log_path}" >"${marker_prefix}.failed"
    echo "KRT formal seed ${seed} failed with exit code ${code}; see ${log_path}" >&2
  fi
}

run_initial_jobs() {
  local -a active_pids=()
  local seed pid
  for seed in "${seed_values[@]}"; do
    run_seed "${seed}" parallel &
    active_pids+=("$!")
    if [[ "${#active_pids[@]}" -ge "${MAX_CONCURRENT_JOBS}" ]]; then
      for pid in "${active_pids[@]}"; do
        wait "${pid}"
      done
      active_pids=()
    fi
  done
  for pid in "${active_pids[@]}"; do
    wait "${pid}"
  done
}

is_memory_failure() {
  local seed="$1"
  local marker="${STATE_DIR}/seed${seed}.failed"
  local log_path exit_code
  log_path="$(sed -n 's/^log=//p' "${marker}")"
  exit_code="$(sed -n 's/^exit_code=//p' "${marker}")"
  [[ "${exit_code}" == "137" ]] && return 0
  grep -Eiq \
    'CUDA out of memory|torch\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory' \
    "${log_path}"
}

archive_failed_attempt() {
  local seed="$1"
  local source="${RUN_OUTPUT_ROOT}/benchmarks/emotic_b5c3_v0.1/A/KRT/seed${seed}"
  local destination="${FAILED_ATTEMPT_DIR}/seed${seed}_parallel"
  mkdir -p "${FAILED_ATTEMPT_DIR}"
  if [[ -e "${source}" ]]; then
    [[ ! -e "${destination}" ]] || {
      echo "Failed-attempt archive already exists: ${destination}" >&2
      return 1
    }
    mv "${source}" "${destination}"
  fi
  mv "${STATE_DIR}/seed${seed}.failed" \
    "${STATE_DIR}/seed${seed}.failed.parallel"
}

run_initial_jobs

failed_seeds=()
for seed in "${seed_values[@]}"; do
  if [[ -f "${STATE_DIR}/seed${seed}.failed" ]]; then
    failed_seeds+=("${seed}")
  fi
done

if [[ "${#failed_seeds[@]}" -gt 0 ]]; then
  retry_allowed=1
  if [[ "${AUTO_RETRY_SEQUENTIAL}" != "1" || "${MAX_CONCURRENT_JOBS}" -le 1 ]]; then
    retry_allowed=0
  fi
  for seed in "${failed_seeds[@]}"; do
    if ! is_memory_failure "${seed}"; then
      retry_allowed=0
    fi
  done
  if [[ "${retry_allowed}" -ne 1 ]]; then
    echo "One or more KRT jobs failed for a non-retryable reason" >&2
    exit 1
  fi

  echo "Parallel CUDA memory failure detected; retrying failed seeds sequentially"
  for seed in "${failed_seeds[@]}"; do
    archive_failed_attempt "${seed}"
    run_seed "${seed}" sequential_retry
    [[ -f "${STATE_DIR}/seed${seed}.done" ]] || {
      echo "Sequential retry failed for seed ${seed}" >&2
      exit 1
    }
  done
fi

for seed in "${seed_values[@]}"; do
  [[ -f "${STATE_DIR}/seed${seed}.done" ]] || {
    echo "Missing completion marker for seed ${seed}" >&2
    exit 1
  }
done

"${PYTHON}" "${SCRIPT_DIR}/collect_clip_continual_results.py" \
  --run-output-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --expected-bundles "${#seed_values[@]}" \
  --extra-dir "${PREFLIGHT_DIR}" \
  --extra-dir "${STATE_DIR}"

DOWNLOAD_DIR="${RUN_OUTPUT_ROOT}/download_ready/${RUN_ID}"
if find "${DOWNLOAD_DIR}" -type f -name '*.pth' -print -quit | grep -q .; then
  echo "Download directory unexpectedly contains .pth checkpoints" >&2
  exit 1
fi
PACKAGE_DIR="${RUN_OUTPUT_ROOT}/download_packages"
ARCHIVE="${PACKAGE_DIR}/${RUN_ID}.tar.gz"
mkdir -p "${PACKAGE_DIR}"
tar -C "${RUN_OUTPUT_ROOT}/download_ready" -czf "${ARCHIVE}" "${RUN_ID}"
sha256sum "${ARCHIVE}" >"${ARCHIVE}.sha256"

echo "All locked KRT formal seeds completed"
echo "Checkpoint-free download directory: ${DOWNLOAD_DIR}"
echo "Checkpoint-free archive: ${ARCHIVE}"
echo "Archive checksum: ${ARCHIVE}.sha256"
