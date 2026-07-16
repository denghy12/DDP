#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
SEED="${SEED:-0}"
TRAINING_MODE="${TRAINING_MODE:-full_base5}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_ddp_prompt_free_auxiliary_feature_cache}"
DDP_CHECKPOINT="${DDP_CHECKPOINT:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task0.pth}"

case "${TRAINING_MODE}" in
  full_base5)
    DEFAULT_OUTPUT_DIR="./output/emotic_ddp_prompt_free_auxiliary_base5_seed${SEED}"
    EXPECTED_SAMPLING_MODE="full_data"
    EPOCHS=50
    SHOT_ARGS=()
    ;;
  16shot)
    DEFAULT_OUTPUT_DIR="./output/emotic_ddp_prompt_free_auxiliary_base5_16shot_seed${SEED}"
    EXPECTED_SAMPLING_MODE="fewshot"
    EPOCHS=200
    SHOT_ARGS=(--shots_per_class 16)
    ;;
  *)
    echo "Unknown TRAINING_MODE=${TRAINING_MODE}; use full_base5 or 16shot" >&2
    exit 2
    ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

[[ -s "${DDP_CHECKPOINT}" ]] || {
  echo "Missing task0 DDP checkpoint: ${DDP_CHECKPOINT}" >&2
  exit 1
}

COMPLETE="$(python - "${OUTPUT_DIR}" "${EXPECTED_SAMPLING_MODE}" "${EPOCHS}" <<'PY'
import json
import os
import sys

path = os.path.join(sys.argv[1], "training_summary.json")
expected_sampling_mode = sys.argv[2]
expected_epochs = int(sys.argv[3])
complete = False
if os.path.isfile(path):
    data = json.load(open(path, encoding="utf-8"))
    protocol = data.get("protocol", {})
    complete = (
        protocol.get("single_clip_instance") is True
        and protocol.get("visual_prompts_during_auxiliary_training") is False
        and protocol.get("auxiliary_branch_retained_for_inference") is False
        and data.get("sampling", {}).get("mode") == expected_sampling_mode
        and data.get("args", {}).get("epochs") == expected_epochs
        and os.path.isfile(os.path.join(sys.argv[1], "best_adapter.pth"))
    )
print("1" if complete else "0")
PY
)"

if [[ "${COMPLETE}" == "1" ]]; then
  echo "Skip completed ${TRAINING_MODE} prompt-free auxiliary seed ${SEED}: ${OUTPUT_DIR}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" python train_emotic_ddp_prompt_free_auxiliary.py \
  --ddp_checkpoint "${DDP_CHECKPOINT}" \
  --data_root ./datasets/EMOTIC \
  --clip_model_path ./pretrained/clip/ViT-B-16.pt \
  --output_dir "${OUTPUT_DIR}" \
  --feature_cache_dir "${CACHE_DIR}" \
  --protocol base5 \
  --class_balanced_bce \
  "${SHOT_ARGS[@]}" \
  --epochs "${EPOCHS}" \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --adapter_dim 128 \
  --residual_scale 0.1 \
  --identity_weight 0.1 \
  --feature_batch_size 128 \
  --adapter_batch_size 1024 \
  --workers 4 \
  --seed "${SEED}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "Checkpoint: ${OUTPUT_DIR}/best_adapter.pth"
echo "JSON:       ${OUTPUT_DIR}/training_summary.json"
echo "HTML:       ${OUTPUT_DIR}/training_history.html"
