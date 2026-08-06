#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

STATE_DIR="${STATE_DIR:-./output/emotic_ddp_transformer_adapter_bank_pipeline}"
MODES="${MODES:-full 16shot}"
SEEDS="${SEEDS:-0 1 2}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-./output/emotic_ddp_transformer_adapter_bank_comparison}"

while true; do
  pending=0
  for mode in ${MODES}; do
    for seed in ${SEEDS}; do
      key="${mode}_seed${seed}"
      if [[ -f "${STATE_DIR}/${key}.failed" ]]; then
        echo "Worker failed: ${key}" >&2
        exit 1
      fi
      [[ -f "${STATE_DIR}/${key}.done" ]] || pending=$((pending + 1))
    done
  done
  (( pending > 0 )) || break
  echo "Waiting for ${pending} Transformer Bank runs..."
  sleep 20
done

python summarize_emotic_ddp_transformer_adapter_bank.py \
  --output_root ./output \
  --training_modes ${MODES} \
  --seeds ${SEEDS} \
  --output_dir "${SUMMARY_OUTPUT_DIR}" \
  2>&1 | tee "${STATE_DIR}/summary.log"

touch "${STATE_DIR}/complete.done"
echo "Comparison: ${SUMMARY_OUTPUT_DIR}/"
