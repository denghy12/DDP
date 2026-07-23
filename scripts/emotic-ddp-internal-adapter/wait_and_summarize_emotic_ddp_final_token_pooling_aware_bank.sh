#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
STATE_DIR="./output/emotic_ddp_final_token_pooling_aware_bank_pipeline"

while [[ ! -f "${STATE_DIR}/gpu0.done" || ! -f "${STATE_DIR}/gpu1.done" ]]; do
  if [[ -f "${STATE_DIR}/gpu0.failed" || -f "${STATE_DIR}/gpu1.failed" ]]; then
    echo "A pooling-aware training worker failed" >&2
    exit 1
  fi
  sleep 20
done

GPU="${GPU}" bash \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_bank_eval.sh

python summarize_emotic_ddp_final_token_pooling_aware_bank.py \
  --output_root ./output \
  --output_dir ./output/emotic_ddp_final_token_pooling_aware_bank_comparison

touch "${STATE_DIR}/complete.done"
echo "Comparison JSON: output/emotic_ddp_final_token_pooling_aware_bank_comparison/comparison_summary.json"
echo "Comparison HTML: output/emotic_ddp_final_token_pooling_aware_bank_comparison/comparison_summary.html"
