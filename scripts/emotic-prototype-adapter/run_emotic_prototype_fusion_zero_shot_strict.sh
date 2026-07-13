#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-emotic_prototype_fusion_strict_zero_shot}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${RUN_NAME}}"
SUMMARY="${OUTPUT_DIR}/fusion_all_tasks_summary.json"

if [[ -s "${SUMMARY}" ]]; then
  echo "Skip completed zero-shot fusion: ${SUMMARY}"
  exit 0
fi

GPU="${GPU}" RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
  bash scripts/emotic-prototype-adapter/run_emotic_prototype_fusion_all_tasks.sh \
  --zero_shot_prototype "$@"
