#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
LOSS_NAME="${LOSS_NAME:-asl}"
SEED="${SEED:-0}"
MAX_TASKS="${MAX_TASKS:-}"
EPOCHS="${EPOCHS:-30}"
RUN_NAME="${RUN_NAME:-emotic_b5c3_ddp_main_${LOSS_NAME}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${RUN_NAME}}"
LOSS_WEIGHT=0.03

case "${LOSS_NAME}" in
  two_way_bce)
    LOSS_ARGS=(--ddp_classification_loss two_way_bce)
    ;;
  asl|asl_g9p8)
    LOSS_ARGS=(
      --ddp_classification_loss asl
      --ddp_asl_gamma_neg 9.8
      --ddp_asl_gamma_pos 0.0
      --ddp_asl_clip 0.05
      --ddp_asl_eps 1e-8
    )
    ;;
  asl_g4)
    LOSS_ARGS=(
      --ddp_classification_loss asl
      --ddp_asl_gamma_neg 4.0
      --ddp_asl_gamma_pos 0.0
      --ddp_asl_clip 0.05
      --ddp_asl_eps 1e-8
    )
    ;;
  asl_g2)
    LOSS_ARGS=(
      --ddp_classification_loss asl
      --ddp_asl_gamma_neg 2.0
      --ddp_asl_gamma_pos 0.0
      --ddp_asl_clip 0.05
      --ddp_asl_eps 1e-8
    )
    ;;
  asl_g2_lw009)
    LOSS_WEIGHT=0.09
    LOSS_ARGS=(
      --ddp_classification_loss asl
      --ddp_asl_gamma_neg 2.0
      --ddp_asl_gamma_pos 0.0
      --ddp_asl_clip 0.05
      --ddp_asl_eps 1e-8
    )
    ;;
  *)
    echo "Unsupported LOSS_NAME=${LOSS_NAME}" >&2
    exit 2
    ;;
esac

TASK_ARGS=()
if [[ -n "${MAX_TASKS}" ]]; then
  TASK_ARGS+=(--max_tasks "${MAX_TASKS}")
fi

mkdir -p "${OUTPUT_DIR}"

PYTHONHASHSEED="${SEED}" CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
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
  --epochs "${EPOCHS}" \
  --train_batch_size 8 \
  --effective_batch_size 256 \
  --eval_batch_size 4 \
  --num_workers 4 \
  --emotic_input_mode full \
  --positive_prompt "a photo of a person clearly feeling" \
  --negative_prompt "a photo of a person not feeling" \
  --csc \
  --loss_w "${LOSS_WEIGHT}" \
  --t_min 1 \
  --t_max 2 \
  --t_gamma 0.7 \
  --thre 0.5 \
  --seed "${SEED}" \
  --reset_optimizer_each_task \
  "${LOSS_ARGS[@]}" \
  "${TASK_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
