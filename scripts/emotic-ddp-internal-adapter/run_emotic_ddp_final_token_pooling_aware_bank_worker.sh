#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to the physical GPU index}"
SEEDS="${SEEDS:?Set space-separated SEEDS}"
WORKER="${WORKER:?Set WORKER name}"
STATE_DIR="./output/emotic_ddp_final_token_pooling_aware_bank_pipeline"
mkdir -p "${STATE_DIR}"

record_failure() {
  code=$?
  if (( code != 0 )); then
    echo "${code}" > "${STATE_DIR}/${WORKER}.failed"
  fi
}
trap record_failure EXIT

for seed in ${SEEDS}; do
  GPU="${GPU}" SEED="${seed}" bash \
    scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_bank_train.sh
done

touch "${STATE_DIR}/${WORKER}.done"
echo "${WORKER} completed seeds ${SEEDS}"
