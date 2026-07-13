#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
DDP_CHECKPOINT="${DDP_CHECKPOINT:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task0.pth}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-./output/emotic_ddp_internal_adapter_16shot_seed0}"

for LOSS_BALANCE in balanced unweighted; do
  for LR in 1e-5 3e-5 1e-4; do
    RUN_NAME="emotic_ddp_internal_screen_seed0_dim16_scale001_id1_lr${LR}_${LOSS_BALANCE}"
    OUTPUT_DIR="./output/${RUN_NAME}"
    if [[ -s "${OUTPUT_DIR}/training_summary.json" ]]; then
      echo "Skip completed ${RUN_NAME}"
      continue
    fi
    mkdir -p "${OUTPUT_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU}" python train_emotic_ddp_internal_adapter.py \
      --ddp_checkpoint "${DDP_CHECKPOINT}" \
      --data_root ./datasets/EMOTIC \
      --clip_model_path ./pretrained/clip/ViT-B-16.pt \
      --output_dir "${OUTPUT_DIR}" \
      --feature_cache_dir "${FEATURE_CACHE_DIR}" \
      --shots_per_class 16 \
      --seed 0 \
      --epochs 50 \
      --batch_size 64 \
      --feature_batch_size 4 \
      --workers 4 \
      --lr "${LR}" \
      --weight_decay 1e-4 \
      --adapter_dim 16 \
      --residual_scale 0.01 \
      --identity_weight 1.0 \
      --loss_balance "${LOSS_BALANCE}" \
      --eval_every_steps 1 \
      --early_stop_patience 10 \
      2>&1 | tee "${OUTPUT_DIR}/train.log"
  done
done

python summarize_emotic_ddp_internal_adapter_screen.py
