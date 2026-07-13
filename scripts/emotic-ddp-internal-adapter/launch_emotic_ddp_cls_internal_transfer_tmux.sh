#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="ddp_cls_internal"

for path in \
  "${ROOT}/output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task0.pth" \
  "${ROOT}/output/emotic_prototype_adapter_base5_16shot_seed0/best_adapter.pth" \
  "${ROOT}/output/emotic_prototype_adapter_base5_16shot_seed1/best_adapter.pth" \
  "${ROOT}/output/emotic_prototype_adapter_base5_16shot_seed2/best_adapter.pth"; do
  [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && GPU=0 bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_internal_transfer.sh"
echo "Started ${SESSION} on GPU0"
