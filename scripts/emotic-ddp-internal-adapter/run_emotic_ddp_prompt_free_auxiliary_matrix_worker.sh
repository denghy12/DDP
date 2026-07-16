#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to the physical GPU index}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID to 0 or 1}"

if [[ "${WORKER_ID}" == "0" ]]; then
  TRAIN_SEEDS=(0)
elif [[ "${WORKER_ID}" == "1" ]]; then
  TRAIN_SEEDS=(1 2)
else
  echo "WORKER_ID must be 0 or 1" >&2
  exit 2
fi

for seed in "${TRAIN_SEEDS[@]}"; do
  GPU="${GPU}" SEED="${seed}" TRAINING_MODE=16shot bash \
    scripts/emotic-ddp-internal-adapter/run_emotic_ddp_prompt_free_auxiliary_base5.sh
done

wait_for_16shot_adapters() {
  while true; do
    complete="$(python - <<'PY'
import json
import os

complete = True
for seed in (0, 1, 2):
    root = f"output/emotic_ddp_prompt_free_auxiliary_base5_16shot_seed{seed}"
    summary = os.path.join(root, "training_summary.json")
    checkpoint = os.path.join(root, "best_adapter.pth")
    if not os.path.isfile(summary) or not os.path.isfile(checkpoint):
        complete = False
        break
    data = json.load(open(summary, encoding="utf-8"))
    if not (
        data.get("sampling", {}).get("mode") == "fewshot"
        and data.get("sampling", {}).get("shots_per_class") == 16
        and data.get("args", {}).get("epochs") == 200
        and data.get("protocol", {}).get("single_clip_instance") is True
    ):
        complete = False
        break
print("1" if complete else "0")
PY
)"
    [[ "${complete}" == "1" ]] && return 0
    echo "GPU${GPU} worker${WORKER_ID}: waiting for all three 16-shot Adapters"
    sleep 20
  done
}
wait_for_16shot_adapters

run_formula() {
  local training_mode="$1"
  local formula="$2"
  echo "GPU${GPU}: ${training_mode} / ${formula}"
  GPU="${GPU}" TRAINING_MODE="${training_mode}" FORMULA="${formula}" bash \
    scripts/emotic-ddp-internal-adapter/run_emotic_ddp_prompt_free_auxiliary_formula.sh
}

if [[ "${WORKER_ID}" == "0" ]]; then
  run_formula full_base5 feature_difference
  run_formula 16shot feature_difference
  run_formula 16shot feature_correction
else
  run_formula full_base5 cosine_difference
  # This is already complete and will be verified/skipped, not recomputed.
  run_formula full_base5 feature_correction
  run_formula 16shot cosine_difference
fi

if [[ "${WORKER_ID}" == "0" ]]; then
  # The comparison is generated only after both GPU workers have either
  # completed strict test evaluation or recorded a failed validation gate.
  while true; do
    matrix_complete="$(python - <<'PY'
import json
import os

stems = (
    "emotic_ddp_prompt_free_auxiliary_cls_feature_difference",
    "emotic_ddp_prompt_free_auxiliary_cls_cosine_difference",
    "emotic_ddp_prompt_free_auxiliary_cls_feature_correction",
    "emotic_ddp_prompt_free_auxiliary_16shot_cls_feature_difference",
    "emotic_ddp_prompt_free_auxiliary_16shot_cls_cosine_difference",
    "emotic_ddp_prompt_free_auxiliary_16shot_cls_feature_correction",
)
complete = True
for stem in stems:
    screen_path = f"output/{stem}_screen/transfer_screen_summary.json"
    if not os.path.isfile(screen_path):
        complete = False
        break
    screen = json.load(open(screen_path, encoding="utf-8"))
    if screen.get("passes_val_gate"):
        summary_path = f"output/{stem}_summary/summary.json"
        if not os.path.isfile(summary_path):
            complete = False
            break
print("1" if complete else "0")
PY
)"
    [[ "${matrix_complete}" == "1" ]] && break
    echo "GPU${GPU} worker0: waiting for the complete 2x3 matrix"
    sleep 20
  done
  python summarize_emotic_ddp_prompt_free_auxiliary_matrix.py
fi

echo "GPU${GPU} worker${WORKER_ID} complete"
