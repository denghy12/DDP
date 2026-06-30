#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-emotic_prototype_adapter_base5}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${RUN_NAME}}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_clip_feature_cache}"

mkdir -p "${OUTPUT_DIR}"

PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="${GPU}" \
python train_emotic_prototype_adapter.py \
  --protocol base5 \
  --datadir ./datasets/EMOTIC \
  --clip_model_path ./pretrained/clip/ViT-B-16.pt \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --base_classes 5 \
  --total_classes 26 \
  --epochs 50 \
  --feature_batch_size 128 \
  --adapter_batch_size 1024 \
  --num_workers 4 \
  --adapter_dim 128 \
  --residual_scale 0.1 \
  --identity_weight 0.1 \
  --threshold 0.5 \
  --seed 0 \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"

