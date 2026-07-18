#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

STATE_DIR="./output/emotic_ddp_task_adapter_bank_pipeline"
mkdir -p "${STATE_DIR}"

python audit_emotic_task_adapter_data.py \
  --data_root ./datasets/EMOTIC \
  --output_dir ./output/emotic_b5c3_task_adapter_audit \
  2>&1 | tee "${STATE_DIR}/audit.log"

while [[ ! -f "${STATE_DIR}/full.done" || ! -f "${STATE_DIR}/16shot.done" ]]; do
  if [[ -f "${STATE_DIR}/full.failed" || -f "${STATE_DIR}/16shot.failed" ]]; then
    echo "A Task Adapter Bank worker failed; inspect pipeline logs" >&2
    exit 1
  fi
  echo "Waiting for Full and 16-shot workers..."
  sleep 20
done

python summarize_emotic_ddp_task_adapter_bank.py \
  --output_root ./output \
  --output_dir ./output/emotic_ddp_task_adapter_bank_comparison \
  2>&1 | tee "${STATE_DIR}/summary.log"

echo "Comparison JSON: output/emotic_ddp_task_adapter_bank_comparison/comparison_summary.json"
echo "Comparison HTML: output/emotic_ddp_task_adapter_bank_comparison/comparison_summary.html"
