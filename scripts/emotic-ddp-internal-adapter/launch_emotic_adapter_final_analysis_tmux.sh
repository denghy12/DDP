#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="adapter_final"

for seed in 0 1 2; do
  for path in \
    "${ROOT}/output/emotic_ddp_cls_internal_transfer_screen/transferred_adapter_seed${seed}.pth" \
    "${ROOT}/output/emotic_prototype_adapter_base5_16shot_seed${seed}/best_adapter.pth"; do
    [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
  done
  for task in 0 1 2 3 4 5 6 7; do
    path="${ROOT}/output/emotic_prototype_fusion_strict_base5_16shot_seed${seed}/task${task}_fusion_scores.pt"
    [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
  done
done
for task in 0 1 2 3 4 5 6 7; do
  [[ -s "${ROOT}/output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task${task}.pth" ]] || {
    echo "Missing DDP task${task} checkpoint" >&2; exit 1;
  }
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
  "cd '${ROOT}' && conda run -n ddp --no-capture-output env GPU=0 bash scripts/emotic-ddp-internal-adapter/run_emotic_adapter_final_analysis.sh"
echo "Started ${SESSION} on GPU0"
echo "Watch with: tmux attach -t ${SESSION}"
