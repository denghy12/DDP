#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

STATE_DIR="./output/emotic_ddp_cls_to_final_token_transfer_pipeline"
OUTPUT_DIR="./output/emotic_ddp_cls_to_final_token_transfer_comparison"

while [[ ! -f "${STATE_DIR}/gpu0.done" || ! -f "${STATE_DIR}/gpu1.done" ]]; do
  if [[ -f "${STATE_DIR}/gpu0.failed" || -f "${STATE_DIR}/gpu1.failed" ]]; then
    echo "A CLS-to-Final-token transfer worker failed" >&2
    exit 1
  fi
  sleep 20
done

python summarize_emotic_ddp_cls_to_final_token_transfer.py \
  --output_root ./output \
  --seeds 0 1 2 \
  --output_dir "${OUTPUT_DIR}"

for path in \
  "${OUTPUT_DIR}/comparison_summary.json" \
  "${OUTPUT_DIR}/comparison_summary.html" \
  "${OUTPUT_DIR}/comparison_summary.csv" \
  "${OUTPUT_DIR}/per_task_comparison.json" \
  "${OUTPUT_DIR}/per_task_comparison.csv" \
  "${OUTPUT_DIR}/per_class_comparison.json" \
  "${OUTPUT_DIR}/per_class_comparison.csv" \
  "${OUTPUT_DIR}/structural_diagnostics.json" \
  "${OUTPUT_DIR}/structural_diagnostics.csv"; do
  [[ -s "${path}" ]] || { echo "Missing summary artifact: ${path}" >&2; exit 1; }
done

touch "${STATE_DIR}/complete.done"
echo "Transfer experiment and summary complete"
echo "Sync result directory: ${OUTPUT_DIR}"
echo "Also sync per-seed provenance/results:"
for seed in 0 1 2; do
  echo "  output/emotic_ddp_cls_to_final_token_transfer_seed${seed}"
done
