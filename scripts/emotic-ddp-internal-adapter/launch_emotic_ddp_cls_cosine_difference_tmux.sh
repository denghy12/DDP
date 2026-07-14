#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${SESSION:-ddp_cls_cosine}"
GPU="${GPU:-0}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && GPU='${GPU}' bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_cosine_difference.sh"

echo "Started ${SESSION} on GPU${GPU}"
echo "Attach with: tmux attach -t ${SESSION}"
