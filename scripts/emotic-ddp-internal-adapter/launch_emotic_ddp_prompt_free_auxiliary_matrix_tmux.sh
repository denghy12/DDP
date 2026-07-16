#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SESSION="${SESSION:-ddp_aux_matrix}"
LOG_DIR="./output/emotic_ddp_prompt_free_auxiliary_matrix_pipeline"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" -n gpu0 \
  "cd '${ROOT}' && GPU='${GPU0}' WORKER_ID=0 bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_prompt_free_auxiliary_matrix_worker.sh 2>&1 | tee '${LOG_DIR}/gpu0.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"
tmux new-window -t "${SESSION}" -n gpu1 \
  "cd '${ROOT}' && GPU='${GPU1}' WORKER_ID=1 bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_prompt_free_auxiliary_matrix_worker.sh 2>&1 | tee '${LOG_DIR}/gpu1.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION}: GPU${GPU0} in window gpu0, GPU${GPU1} in window gpu1"
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs:   ${LOG_DIR}/gpu0.log and ${LOG_DIR}/gpu1.log"
