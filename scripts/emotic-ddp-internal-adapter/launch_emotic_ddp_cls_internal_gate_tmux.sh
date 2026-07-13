#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="ddp_cls_gate"

for seed in 0 1 2; do
  path="${ROOT}/output/emotic_ddp_cls_internal_transfer_screen/transferred_adapter_seed${seed}.pth"
  [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
done
for task in 0 1 2 3 4 5 6 7; do
  for split in val test; do
    path="${ROOT}/output/emotic_ddp_cls_internal_feature_cache/task${task}_${split}_cls_path_features.pt"
    [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
  done
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session already exists: ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && GPU=0 bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_internal_gate.sh"
echo "Started ${SESSION} on GPU0"
