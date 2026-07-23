#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SESSION="${SESSION:-ddp_cls_to_final_token_transfer}"
STATE_DIR="./output/emotic_ddp_cls_to_final_token_transfer_pipeline"
mkdir -p "${STATE_DIR}"
rm -f \
  "${STATE_DIR}"/{gpu0,gpu1}.{done,failed} \
  "${STATE_DIR}/complete.done"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n seeds02_gpu0 \
  "cd '${ROOT}' && GPU='${GPU0}' SEEDS='0 2' WORKER=gpu0 bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_to_final_token_transfer.sh 2>&1 | tee '${STATE_DIR}/seeds02_gpu0.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"
tmux new-window -t "${SESSION}" -n seed1_gpu1 \
  "cd '${ROOT}' && GPU='${GPU1}' SEEDS='1' WORKER=gpu1 bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_to_final_token_transfer.sh 2>&1 | tee '${STATE_DIR}/seed1_gpu1.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"
tmux new-window -t "${SESSION}" -n summarize \
  "cd '${ROOT}' && bash scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_cls_to_final_token_transfer.sh 2>&1 | tee '${STATE_DIR}/orchestrator.log'; code=\${PIPESTATUS[0]}; echo SUMMARY_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION}"
echo "  GPU${GPU0}: seed0, then seed2"
echo "  GPU${GPU1}: seed1"
echo "  Each seed jointly evaluates DDP, original CLS, all-token, and CLS-only"
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs:   ${STATE_DIR}/"
