#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-emotic_b5c3_ddp_semantic_tau2}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${RUN_NAME}}"

mkdir -p "${OUTPUT_DIR}"

PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
  --dataset emotic \
  --name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_file configs/models/vitb16_ep50.yaml \
  --dataset_config_file configs/datasets/emotic.yaml \
  --datadir ./datasets/EMOTIC \
  --clip_model_path ./pretrained/clip/ViT-B-16.pt \
  --base_classes 5 \
  --task_size 3 \
  --total_classes 26 \
  --epochs 30 \
  --train_batch_size 8 \
  --effective_batch_size 256 \
  --eval_batch_size 4 \
  --num_workers 4 \
  --emotic_input_mode full \
  --positive_prompt "a photo of a person clearly feeling" \
  --negative_prompt "a photo of a person not feeling" \
  --csc \
  --t_min 1 \
  --t_max 2 \
  --t_gamma 0.7 \
  --thre 0.8 \
  --seed 0 \
  --reset_optimizer_each_task \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
