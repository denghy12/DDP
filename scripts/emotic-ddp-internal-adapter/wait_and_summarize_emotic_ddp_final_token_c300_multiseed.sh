#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

STATE_DIR="./output/emotic_ddp_final_token_c300_multiseed_pipeline"
mkdir -p "${STATE_DIR}"

while [[ ! -f "${STATE_DIR}/s02.done" || ! -f "${STATE_DIR}/s1.done" ]]; do
  if [[ -f "${STATE_DIR}/s02.failed" || -f "${STATE_DIR}/s1.failed" ]]; then
    echo "A C300 multi-seed worker failed; inspect the pipeline logs" >&2
    exit 1
  fi
  echo "Waiting for C300 Task-0 seeds 0/1/2..."
  sleep 15
done

python summarize_emotic_ddp_final_token_c300_multiseed.py \
  --output_root ./output \
  --output_dir ./output/emotic_ddp_final_token_c300_multiseed_summary \
  2>&1 | tee "${STATE_DIR}/summary.log"

echo "Summary JSON: output/emotic_ddp_final_token_c300_multiseed_summary/c300_multiseed_summary.json"
echo "Summary HTML: output/emotic_ddp_final_token_c300_multiseed_summary/c300_multiseed_summary.html"

