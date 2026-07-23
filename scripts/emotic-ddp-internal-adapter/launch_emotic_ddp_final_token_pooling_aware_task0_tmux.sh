#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
SESSION="${SESSION:-ddp_final_token_poolaware}"
OUTPUT_DIR="./output/emotic_ddp_final_token_pooling_aware_task0_seed0"
mkdir -p "${OUTPUT_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && GPU='${GPU}' bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_task0.sh 2>&1 | tee '${OUTPUT_DIR}/pipeline.log'; code=\${PIPESTATUS[0]}; echo PIPELINE_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION} on GPU${GPU}"
echo "Attach with: tmux attach -t ${SESSION}"
echo "Log: ${OUTPUT_DIR}/pipeline.log"

