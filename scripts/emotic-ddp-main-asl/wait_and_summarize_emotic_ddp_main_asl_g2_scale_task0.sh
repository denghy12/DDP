#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SEEDS="${SEEDS:-0 1 2}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_main_asl_g2_scale_task0_pipeline}"
SUMMARY_DIR="${SUMMARY_DIR:-./output/emotic_ddp_main_asl_g2_scale_task0_summary}"

while true; do
  pending=0
  for seed in ${SEEDS}; do
    key="asl_g2_lw009_seed${seed}"
    if [[ -f "${STATE_DIR}/${key}.failed" ]]; then
      echo "Worker failed: ${key}" >&2
      exit 1
    fi
    [[ -f "${STATE_DIR}/${key}.done" ]] || pending=$((pending + 1))
  done
  (( pending == 0 )) && break
  echo "Waiting for ${pending} scale-matched Task-0 runs..."
  sleep 20
done

python summarize_emotic_ddp_main_asl_g2_scale_task0.py \
  --output_root ./output \
  --seeds ${SEEDS} \
  --output_dir "${SUMMARY_DIR}" \
  2>&1 | tee "${STATE_DIR}/summary.log"

touch "${STATE_DIR}/complete.done"
echo "Summary: ${SUMMARY_DIR}/scale_matched_summary.html"
