#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

STATE_DIR="./output/emotic_ddp_final_token_safety_screen_pipeline"
mkdir -p "${STATE_DIR}"

while [[ ! -f "${STATE_DIR}/ac.done" || ! -f "${STATE_DIR}/b.done" ]]; do
  if [[ -f "${STATE_DIR}/ac.failed" || -f "${STATE_DIR}/b.failed" ]]; then
    echo "A Final-token safety worker failed; inspect the pipeline logs" >&2
    exit 1
  fi
  echo "Waiting for Final-token safety configs A/B/C..."
  sleep 15
done

python summarize_emotic_ddp_final_token_safety_screen.py \
  --output_root ./output \
  --output_dir ./output/emotic_ddp_final_token_safety_screen_summary \
  2>&1 | tee "${STATE_DIR}/summary.log"

echo "Summary JSON: output/emotic_ddp_final_token_safety_screen_summary/safety_screen_summary.json"
echo "Summary HTML: output/emotic_ddp_final_token_safety_screen_summary/safety_screen_summary.html"

