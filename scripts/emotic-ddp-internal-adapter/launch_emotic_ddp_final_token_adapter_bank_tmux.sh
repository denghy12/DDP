#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SESSION="${SESSION:-ddp_final_token_bank}"
STATE_DIR="./output/emotic_ddp_final_token_adapter_bank_pipeline"
mkdir -p "${STATE_DIR}"
rm -f \
  "${STATE_DIR}/full.done" "${STATE_DIR}/16shot.done" \
  "${STATE_DIR}/full.failed" "${STATE_DIR}/16shot.failed"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n full_gpu0 \
  "cd '${ROOT}' && GPU='${GPU0}' TRAINING_MODE=full bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_adapter_bank_worker.sh 2>&1 | tee '${STATE_DIR}/full_gpu0.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"
tmux new-window -t "${SESSION}" -n shot16_gpu1 \
  "cd '${ROOT}' && GPU='${GPU1}' TRAINING_MODE=16shot bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_adapter_bank_worker.sh 2>&1 | tee '${STATE_DIR}/16shot_gpu1.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"
tmux new-window -t "${SESSION}" -n summary \
  "cd '${ROOT}' && bash scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_adapter_bank.sh 2>&1 | tee '${STATE_DIR}/orchestrator.log'; code=\${PIPESTATUS[0]}; echo SUMMARY_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION}"
echo "  Full seeds 0/1/2:    GPU${GPU0}"
echo "  16-shot seeds 0/1/2: GPU${GPU1}"
echo "  Each mode evaluates three seeds in one shared DDP token pass."
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs:   ${STATE_DIR}/"
