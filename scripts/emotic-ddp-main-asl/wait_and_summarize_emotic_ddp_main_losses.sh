#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

STATE_DIR="${STATE_DIR:-./output/emotic_ddp_main_loss_pipeline}"
SUMMARY_DIR="${SUMMARY_DIR:-./output/emotic_ddp_main_loss_comparison}"
LOSSES="${LOSSES:-two_way_bce asl}"
SEEDS="${SEEDS:-0 1 2}"
mkdir -p "${STATE_DIR}"

while true; do
  pending=0
  for loss in ${LOSSES}; do
    for seed in ${SEEDS}; do
      key="${loss}_seed${seed}"
      if [[ -f "${STATE_DIR}/${key}.failed" ]]; then
        echo "Worker failed: ${key}" >&2
        exit 1
      fi
      [[ -f "${STATE_DIR}/${key}.done" ]] || pending=$((pending + 1))
    done
  done
  if (( pending == 0 )); then
    break
  fi
  echo "Waiting for ${pending} DDP main-loss runs..."
  sleep 20
done

python summarize_emotic_ddp_main_losses.py \
  --output_root ./output \
  --seeds ${SEEDS} \
  --output_dir "${SUMMARY_DIR}" \
  2>&1 | tee "${STATE_DIR}/summary.log"

touch "${STATE_DIR}/complete.done"
echo "Comparison: ${SUMMARY_DIR}/comparison_summary.html"
