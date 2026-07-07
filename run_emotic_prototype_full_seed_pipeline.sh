#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:?Set SEED to 0, 1, or 2}"

if [[ "${SEED}" == "0" ]]; then
  adapter_run="emotic_prototype_adapter_base5_balanced"
else
  adapter_run="emotic_prototype_adapter_base5_balanced_seed${SEED}"
fi
checkpoint="./output/${adapter_run}/best_adapter.pth"

if [[ -s "${checkpoint}" ]]; then
  echo "Skip completed full-data Adapter training: ${checkpoint}"
else
  echo "Training full-data Base5-balanced Adapter seed=${SEED}"
  GPU="${GPU}" RUN_NAME="${adapter_run}" \
    bash run_emotic_prototype_adapter_base5.sh \
    --class_balanced_bce --seed "${SEED}"
fi

fusion_run="emotic_prototype_fusion_strict_full_seed${SEED}"
summary="./output/${fusion_run}/fusion_all_tasks_summary.json"
if [[ -s "${summary}" ]]; then
  echo "Skip completed full-data fusion: ${summary}"
else
  GPU="${GPU}" RUN_NAME="${fusion_run}" PROTOTYPE_CHECKPOINT="${checkpoint}" \
    bash run_emotic_prototype_fusion_all_tasks.sh
fi
