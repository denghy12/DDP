#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SHOTS_LIST="${SHOTS_LIST:-1 2 4 8 16}"
SEEDS_LIST="${SEEDS_LIST:-0 1 2}"
EPOCHS="${EPOCHS:-200}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_clip_feature_cache}"

for SHOTS in ${SHOTS_LIST}; do
  for SEED in ${SEEDS_LIST}; do
    RUN_NAME="emotic_prototype_adapter_all26_${SHOTS}shot_seed${SEED}"
    OUTPUT_DIR="./output/${RUN_NAME}"
    mkdir -p "${OUTPUT_DIR}"

    PYTHONHASHSEED="${SEED}" CUDA_VISIBLE_DEVICES="${GPU}" \
    python train_emotic_prototype_adapter.py \
      --protocol all26 \
      --shots_per_class "${SHOTS}" \
      --class_balanced_bce \
      --datadir ./datasets/EMOTIC \
      --clip_model_path ./pretrained/clip/ViT-B-16.pt \
      --output_dir "${OUTPUT_DIR}" \
      --cache_dir "${CACHE_DIR}" \
      --base_classes 5 \
      --total_classes 26 \
      --epochs "${EPOCHS}" \
      --feature_batch_size 128 \
      --adapter_batch_size 1024 \
      --num_workers 4 \
      --adapter_dim 128 \
      --residual_scale 0.1 \
      --identity_weight 0.1 \
      --threshold 0.5 \
      --seed "${SEED}" \
      "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
  done
done

python summarize_emotic_prototype_fewshot.py \
  --protocol all26 \
  --shots ${SHOTS_LIST} \
  --seeds ${SEEDS_LIST}
