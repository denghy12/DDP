#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
SEED="${SEED:?Set SEED to 0, 1, or 2}"
RUN_NAME="${RUN_NAME:-emotic_ddp_internal_adapter_16shot_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${RUN_NAME}}"
DDP_OUTPUT="${DDP_OUTPUT:-./output/emotic_b5c3_ddp_semantic_tau2}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-${DDP_OUTPUT}/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_ddp_internal_feature_cache}"

mkdir -p "${OUTPUT_DIR}"

if [[ -s "${OUTPUT_DIR}/best_adapter.pth" ]]; then
  echo "Skip completed Adapter training: ${OUTPUT_DIR}/best_adapter.pth"
else
  CUDA_VISIBLE_DEVICES="${GPU}" python train_emotic_ddp_internal_adapter.py \
    --ddp_checkpoint "${DDP_CHECKPOINT_DIR}/task0.pth" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output_dir "${OUTPUT_DIR}" \
    --shots_per_class 16 \
    --seed "${SEED}" \
    --epochs 200 \
    --batch_size 64 \
    --feature_batch_size 4 \
    --workers 4 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --adapter_dim 128 \
    --residual_scale 0.1 \
    --identity_weight 0.1 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

if [[ -s "${OUTPUT_DIR}/evaluation_summary.json" ]]; then
  echo "Skip completed all-task evaluation: ${OUTPUT_DIR}/evaluation_summary.json"
else
  CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_internal_adapter.py \
    --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
    --adapter_checkpoint "${OUTPUT_DIR}/best_adapter.pth" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output_dir "${OUTPUT_DIR}" \
    --name "${RUN_NAME}" \
    --cache_dir "${CACHE_DIR}" \
    --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
    --feature_batch_size 4 \
    --adapter_batch_size 256 \
    --workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
fi
