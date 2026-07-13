#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"


missing=0
for task in 0 1 2 3 4 5 6 7; do
  for path in \
    "${ROOT}/output/emotic_b5c3_ddp_semantic_tau2_val_threshold_sweep/task${task}/task_scores.pt" \
    "${ROOT}/output/emotic_b5c3_ddp_semantic_tau2_test_threshold050/task${task}_scores.pt"; do
    if [[ ! -s "${path}" ]]; then
      echo "Missing strict DDP score: ${path}" >&2
      missing=1
    fi
  done
done
for shots in 1 2 4 8 16; do
  for seed in 0 1 2; do
    path="${ROOT}/output/emotic_prototype_adapter_base5_${shots}shot_seed${seed}/best_adapter.pth"
    if [[ ! -s "${path}" ]]; then
      echo "Missing existing few-shot checkpoint: ${path}" >&2
      missing=1
    fi
  done
done
if [[ ! -s "${ROOT}/output/emotic_prototype_adapter_base5_balanced/best_adapter.pth" ]]; then
  echo "Missing full Base5 seed0 checkpoint" >&2
  missing=1
fi
if [[ "${missing}" -ne 0 ]]; then
  echo "Preflight failed; no tmux experiments were started." >&2
  exit 1
fi

launch() {
  local session="$1"
  local command="$2"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Skip existing tmux session: ${session}"
    return
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && ${command}; status=\$?; echo EXIT_CODE=\${status}; exit \${status}"
  echo "Started ${session}: ${command}"
}

gpu0_free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -n1 | tr -d ' ')"
full_seed2_gpu=0
if [[ "${gpu0_free_mb}" -lt 12000 ]]; then
  full_seed2_gpu=1
fi
echo "GPU0 free=${gpu0_free_mb} MiB; full seed2 assigned to GPU${full_seed2_gpu}"

launch proto_zero "GPU=0 bash scripts/emotic-prototype-adapter/run_emotic_prototype_fusion_zero_shot_strict.sh"
launch proto_k12 "GPU=0 SHOTS_LIST='1 2' bash scripts/emotic-prototype-adapter/run_emotic_prototype_fusion_fewshot_sweep.sh"
launch proto_k48 "GPU=0 SHOTS_LIST='4 8' bash scripts/emotic-prototype-adapter/run_emotic_prototype_fusion_fewshot_sweep.sh"
launch proto_k16 "GPU=0 SHOTS_LIST='16' bash scripts/emotic-prototype-adapter/run_emotic_prototype_fusion_fewshot_sweep.sh"
launch proto_full0 "GPU=0 SEED=0 bash scripts/emotic-prototype-adapter/run_emotic_prototype_full_seed_pipeline.sh"
launch proto_full1 "GPU=0 SEED=1 bash scripts/emotic-prototype-adapter/run_emotic_prototype_full_seed_pipeline.sh"
launch proto_full2 "GPU=${full_seed2_gpu} SEED=2 bash scripts/emotic-prototype-adapter/run_emotic_prototype_full_seed_pipeline.sh"

echo
echo "Monitor: tmux list-sessions"
echo "Attach:  tmux attach -t proto_zero"
echo "After all runs finish: python summarize_emotic_prototype_fusion_sweep.py"
