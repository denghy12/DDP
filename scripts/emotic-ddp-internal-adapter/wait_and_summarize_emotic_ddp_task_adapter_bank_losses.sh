#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

STATE_DIR="./output/emotic_ddp_task_adapter_bank_loss_pipeline"
MODES="${MODES:-full}"
LOSSES="${LOSSES:-weighted_bce asl bal_paper}"
SEEDS="${SEEDS:-0 1 2}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-./output/emotic_ddp_task_adapter_bank_loss_comparison}"
mkdir -p "${STATE_DIR}"

while true; do
  pending=0
  for mode in ${MODES}; do
    for loss_name in ${LOSSES}; do
      for seed in ${SEEDS}; do
        key="${loss_name}_${mode}_seed${seed}"
        if [[ -f "${STATE_DIR}/${key}.failed" ]]; then
          echo "Worker failed: ${key}" >&2
          exit 1
        fi
        [[ -f "${STATE_DIR}/${key}.done" ]] || pending=$((pending + 1))
      done
    done
  done
  if (( pending == 0 )); then
    break
  fi
  echo "Waiting for ${pending} loss-bank runs..."
  sleep 20
done

python summarize_emotic_ddp_task_adapter_bank_losses.py \
  --output_root ./output \
  --training_modes ${MODES} \
  --losses ${LOSSES} \
  --seeds ${SEEDS} \
  --output_dir "${SUMMARY_OUTPUT_DIR}" \
  2>&1 | tee "${STATE_DIR}/summary.log"

touch "${STATE_DIR}/complete.done"
echo "Comparison: ${SUMMARY_OUTPUT_DIR}/"
