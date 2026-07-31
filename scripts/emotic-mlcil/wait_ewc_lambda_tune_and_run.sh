#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:?SESSION is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
TUNING_STATE_DIR="${TUNING_STATE_DIR:?TUNING_STATE_DIR is required}"
TUNING_JOB_KEYS="${TUNING_JOB_KEYS:?TUNING_JOB_KEYS is required}"
TUNING_ROOT="${TUNING_ROOT:?TUNING_ROOT is required}"
SELECTION_DIR="${SELECTION_DIR:?SELECTION_DIR is required}"
FORMAL_ROOT="${FORMAL_ROOT:?FORMAL_ROOT is required}"
FORMAL_STATE_DIR="${FORMAL_STATE_DIR:?FORMAL_STATE_DIR is required}"
FORMAL_SEEDS="${FORMAL_SEEDS:-0 1 2}"
GPU_SLOTS="${GPU_SLOTS:?GPU_SLOTS is required}"
LAMBDAS="${LAMBDAS:?LAMBDAS is required}"
TUNING_SEED="${TUNING_SEED:-0}"
POLL_SECONDS="${POLL_SECONDS:-20}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"

wait_for_jobs() {
  local state_dir="$1"
  local keys="$2"
  local label="$3"
  local completed failed job
  read -r -a jobs <<< "${keys}"
  while true; do
    completed=0
    failed=0
    for job in "${jobs[@]}"; do
      if [[ -f "${state_dir}/${job}.done" ]]; then
        completed=$((completed + 1))
      elif [[ -f "${state_dir}/${job}.failed" ]]; then
        failed=$((failed + 1))
      fi
    done
    echo "${label}: completed=${completed}/${#jobs[@]} failed=${failed}"
    if [[ "${failed}" -gt 0 ]]; then
      echo "${label} failed; inspect ${state_dir}/*.log" >&2
      return 1
    fi
    if [[ "${completed}" -eq "${#jobs[@]}" ]]; then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

wait_for_jobs "${TUNING_STATE_DIR}" "${TUNING_JOB_KEYS}" "EWC tuning"

read -r -a lambda_values <<< "${LAMBDAS}"
"${PYTHON}" "${SCRIPT_DIR}/select_ewc_lambda.py" \
  --tuning-root "${TUNING_ROOT}" \
  --lambdas "${lambda_values[@]}" \
  --seed "${TUNING_SEED}" \
  --output-dir "${SELECTION_DIR}"

selected_lambda="$("${PYTHON}" - "${SELECTION_DIR}/selection.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(format(float(json.load(stream)["selected_ewc_lambda"]), ".12g"))
PY
)"
echo "Locked EWC lambda from validation: ${selected_lambda}"

mkdir -p "${FORMAL_STATE_DIR}"
read -r -a seeds <<< "${FORMAL_SEEDS}"
read -r -a slots <<< "${GPU_SLOTS}"
[[ "${#slots[@]}" -gt 0 ]] || {
  echo "No GPU slots are available for formal EWC runs" >&2
  exit 2
}

declare -a assignments
for index in "${!slots[@]}"; do
  assignments[index]=""
done
formal_keys=()
for index in "${!seeds[@]}"; do
  slot=$((index % ${#slots[@]}))
  assignments[slot]="${assignments[slot]} ewc:seed${seeds[index]}"
  formal_keys+=("ewc_seed${seeds[index]}")
done

worker_count=0
for slot in "${!slots[@]}"; do
  assignment="${assignments[slot]# }"
  [[ -n "${assignment}" ]] || continue
  gpu="${slots[slot]}"
  printf -v command \
    'cd %q && GPU=%q JOBS=%q STATE_DIR=%q EWC_LAMBDA=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q OUTPUT_ROOT=%q PROTOCOL=%q REPORTING_SPLIT=test TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q RUN_ID=%q bash %q; code=$?; echo FORMAL_WORKER_EXIT_CODE=$code; exec bash' \
    "${ROOT}" "${gpu}" "${assignment}" "${FORMAL_STATE_DIR}" \
    "${selected_lambda}" "${PYTHON}" "${DATA_ROOT}" \
    "${CLIP_MODEL_PATH}" "${FORMAL_ROOT}" "${PROTOCOL}" \
    "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${WORKERS}" \
    "${RUN_ID}" "${SCRIPT_DIR}/run_clip_continual_worker.sh"
  tmux new-window -t "${SESSION}" -n "formal${slot}_g${gpu}" "${command}"
  worker_count=$((worker_count + 1))
done
echo "Started ${worker_count} formal EWC workers for seeds ${FORMAL_SEEDS}"

formal_key_string="${formal_keys[*]}"
wait_for_jobs "${FORMAL_STATE_DIR}" "${formal_key_string}" "Formal EWC"

"${PYTHON}" "${SCRIPT_DIR}/validate_ewc_formal_selection.py" \
  --selection "${SELECTION_DIR}/selection.json" \
  --formal-root "${FORMAL_ROOT}" \
  --seeds "${seeds[@]}" \
  --output "${SELECTION_DIR}/formal_validation.json"

"${PYTHON}" "${SCRIPT_DIR}/collect_clip_continual_results.py" \
  --run-output-root "${FORMAL_ROOT}" \
  --run-id "${RUN_ID}" \
  --expected-bundles "${#formal_keys[@]}" \
  --extra-dir "${SELECTION_DIR}"

echo "EWC tuning and formal 3-seed run completed"
echo "Selected lambda: ${selected_lambda}"
echo "Checkpoints: ${FORMAL_ROOT}/benchmarks/*/A/EWC/seed*/checkpoints"
echo "Download only: ${FORMAL_ROOT}/download_ready/${RUN_ID}"
