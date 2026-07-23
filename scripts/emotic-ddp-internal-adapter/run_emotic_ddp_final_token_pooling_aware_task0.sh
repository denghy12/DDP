#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
RUN_NAME="emotic_ddp_final_token_pooling_aware_task0_seed0"
OUTPUT_DIR="./output/${RUN_NAME}"
BASELINE="./output/emotic_ddp_final_token_c300_multiseed_seed0/training_summary.json"

[[ -s "${BASELINE}" ]] || {
  echo "Missing controlled C300 seed0 baseline: ${BASELINE}" >&2
  exit 1
}
if [[ -s "${OUTPUT_DIR}/training_summary.json" ]]; then
  echo "Refusing to overwrite completed pooling-aware pilot: ${OUTPUT_DIR}" >&2
  exit 1
fi

GPU="${GPU}" \
SEED=0 \
RUN_NAME="${RUN_NAME}" \
LEARNING_RATE="3e-5" \
MAX_OPTIMIZER_STEPS=300 \
HEALTH_EPOCHS=1 \
IDENTITY_WEIGHT=0.1 \
POOLING_WEIGHT=100.0 \
ATTENTION_WEIGHT=100.0 \
MARGIN_WEIGHT=1.0 \
MARGIN_BETA=1.0 \
  bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_training_healthcheck.sh

python summarize_emotic_ddp_final_token_pooling_aware_pilot.py \
  --baseline "${BASELINE}" \
  --candidate "${OUTPUT_DIR}/training_summary.json" \
  --output_dir "${OUTPUT_DIR}"

