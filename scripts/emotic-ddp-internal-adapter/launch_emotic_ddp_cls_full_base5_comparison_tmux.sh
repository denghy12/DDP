#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${SESSION:-ddp_cls_full_base5}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
ENV_BIN="$(dirname "${PYTHON_BIN}")"
LOG_DIR="${ROOT}/output/emotic_ddp_cls_full_base5_comparison"

mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && export PATH='${ENV_BIN}':\$PATH && GPU='${GPU}' bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_full_base5_comparison.sh 2>&1 | tee '${LOG_DIR}/pipeline.log'"

echo "Started ${SESSION} on GPU${GPU}"
echo "Log: ${LOG_DIR}/pipeline.log"
echo "Attach with: tmux attach -t ${SESSION}"
