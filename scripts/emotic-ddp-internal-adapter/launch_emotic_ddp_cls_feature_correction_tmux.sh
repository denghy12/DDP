#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${SESSION:-ddp_cls_feature_correction}"
GPU="${GPU:-0}"
ADAPTER_SOURCE="${ADAPTER_SOURCE:-full_base5}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && GPU='${GPU}' ADAPTER_SOURCE='${ADAPTER_SOURCE}' bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_feature_correction.sh"

echo "Started ${SESSION} on GPU${GPU} with ADAPTER_SOURCE=${ADAPTER_SOURCE}"
echo "Attach with: tmux attach -t ${SESSION}"
