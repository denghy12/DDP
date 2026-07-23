#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SESSION="${SESSION:-ddp_final_token_c300_seeds}"
STATE_DIR="./output/emotic_ddp_final_token_c300_multiseed_pipeline"
mkdir -p "${STATE_DIR}"
rm -f \
  "${STATE_DIR}/s02.running" "${STATE_DIR}/s02.done" "${STATE_DIR}/s02.failed" \
  "${STATE_DIR}/s1.running" "${STATE_DIR}/s1.done" "${STATE_DIR}/s1.failed"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n seeds02_gpu0 \
  "cd '${ROOT}' && GPU='${GPU0}' WORKER=s02 bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_c300_multiseed_worker.sh 2>&1 | tee '${STATE_DIR}/seeds02_gpu0.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"
tmux new-window -t "${SESSION}" -n seed1_gpu1 \
  "cd '${ROOT}' && GPU='${GPU1}' WORKER=s1 bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_c300_multiseed_worker.sh 2>&1 | tee '${STATE_DIR}/seed1_gpu1.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"
tmux new-window -t "${SESSION}" -n summary \
  "cd '${ROOT}' && bash scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_c300_multiseed.sh 2>&1 | tee '${STATE_DIR}/orchestrator.log'; code=\${PIPESTATUS[0]}; echo SUMMARY_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION}"
echo "  Seeds 0 then 2: GPU${GPU0}"
echo "  Seed 1:         GPU${GPU1}"
echo "  No Task1-7 run will be launched automatically."
echo "Attach with: tmux attach -t ${SESSION}"
echo "Logs: ${STATE_DIR}/"

