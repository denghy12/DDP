#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SESSION="${SESSION:-ddp_final_token_safety}"
STATE_DIR="./output/emotic_ddp_final_token_safety_screen_pipeline"
mkdir -p "${STATE_DIR}"
rm -f \
  "${STATE_DIR}/ac.running" "${STATE_DIR}/ac.done" "${STATE_DIR}/ac.failed" \
  "${STATE_DIR}/b.running" "${STATE_DIR}/b.done" "${STATE_DIR}/b.failed"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n ac_gpu0 \
  "cd '${ROOT}' && GPU='${GPU0}' WORKER=ac bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_safety_worker.sh 2>&1 | tee '${STATE_DIR}/ac_gpu0.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"
tmux new-window -t "${SESSION}" -n b_gpu1 \
  "cd '${ROOT}' && GPU='${GPU1}' WORKER=b bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_safety_worker.sh 2>&1 | tee '${STATE_DIR}/b_gpu1.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"
tmux new-window -t "${SESSION}" -n summary \
  "cd '${ROOT}' && bash scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_safety_screen.sh 2>&1 | tee '${STATE_DIR}/orchestrator.log'; code=\${PIPESTATUS[0]}; echo SUMMARY_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION}"
echo "  Config A then C: GPU${GPU0}"
echo "  Config B:        GPU${GPU1}"
echo "Attach with: tmux attach -t ${SESSION}"
echo "Logs: ${STATE_DIR}/"

