#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU}"
TRAINING_MODE="${TRAINING_MODE:?Set TRAINING_MODE}"
LOSS_NAME="${LOSS_NAME:?Set LOSS_NAME}"
SEED="${SEED:?Set SEED}"
STATE_DIR="./output/emotic_ddp_task_adapter_bank_loss_pipeline"
KEY="${LOSS_NAME}_${TRAINING_MODE}_seed${SEED}"
mkdir -p "${STATE_DIR}"
rm -f "${STATE_DIR}/${KEY}.failed"

record_failure() {
  code=$?
  if (( code != 0 )); then
    echo "${code}" > "${STATE_DIR}/${KEY}.failed"
  fi
}
trap record_failure EXIT

GPU="${GPU}" TRAINING_MODE="${TRAINING_MODE}" LOSS_NAME="${LOSS_NAME}" \
  SEED="${SEED}" bash \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_train.sh

GPU="${GPU}" TRAINING_MODE="${TRAINING_MODE}" LOSS_NAME="${LOSS_NAME}" \
  SEED="${SEED}" bash \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_eval.sh

touch "${STATE_DIR}/${KEY}.done"
echo "Complete ${KEY}"
