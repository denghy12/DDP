#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to the physical GPU index}"
TRAINING_MODE="${TRAINING_MODE:?Set TRAINING_MODE to full or 16shot}"
STATE_DIR="./output/emotic_ddp_task_adapter_bank_pipeline"
mkdir -p "${STATE_DIR}"

record_failure() {
  code=$?
  if (( code != 0 )); then
    echo "${code}" > "${STATE_DIR}/${TRAINING_MODE}.failed"
  fi
}
trap record_failure EXIT

for seed in 0 1 2; do
  GPU="${GPU}" TRAINING_MODE="${TRAINING_MODE}" SEED="${seed}" bash \
    scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_train.sh
done

for seed in 0 1 2; do
  GPU="${GPU}" TRAINING_MODE="${TRAINING_MODE}" SEED="${seed}" bash \
    scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_eval.sh
done

touch "${STATE_DIR}/${TRAINING_MODE}.done"
echo "${TRAINING_MODE} Task Adapter Bank worker complete"
