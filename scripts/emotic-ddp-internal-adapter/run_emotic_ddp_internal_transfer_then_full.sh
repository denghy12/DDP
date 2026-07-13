#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
PATH_CACHE_DIR="${PATH_CACHE_DIR:-./output/emotic_ddp_internal_adapter_16shot_seed0}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-./output/emotic_ddp_internal_feature_cache}"
TRANSFER_DIR="./output/emotic_ddp_internal_transfer_screen"
FULL_DIR="./output/emotic_ddp_internal_full_base5_screen"

mkdir -p "${TRANSFER_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" python screen_emotic_ddp_internal_adapter_transfer.py \
  --val_cache "${PATH_CACHE_DIR}/task0_val_path_features.pt" \
  --output_dir "${TRANSFER_DIR}" \
  2>&1 | tee "${TRANSFER_DIR}/screen.log"

TRANSFER_PASSES="$(python - <<'PY'
import json
data = json.load(open('output/emotic_ddp_internal_transfer_screen/transfer_screen_summary.json'))
print('1' if data['passes_val_gate'] else '0')
PY
)"

if [[ "${TRANSFER_PASSES}" == "1" ]]; then
  echo "Transfer passed pure-val gate; running locked three-seed test evaluation."
  for SEED in 0 1 2; do
    RUN_NAME="emotic_ddp_internal_transfer_16shot_seed${SEED}"
    OUTPUT_DIR="./output/${RUN_NAME}"
    mkdir -p "${OUTPUT_DIR}"
    if [[ ! -s "${OUTPUT_DIR}/evaluation_summary.json" ]]; then
      CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_internal_adapter.py \
        --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
        --adapter_checkpoint "${TRANSFER_DIR}/transferred_adapter_seed${SEED}.pth" \
        --data_root ./datasets/EMOTIC \
        --clip_model_path ./pretrained/clip/ViT-B-16.pt \
        --output_dir "${OUTPUT_DIR}" \
        --name "${RUN_NAME}" \
        --cache_dir "${EVAL_CACHE_DIR}" \
        --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"
    fi
  done
  python summarize_emotic_ddp_internal_adapter.py \
    --run_prefix emotic_ddp_internal_transfer_16shot_seed \
    --output_dir ./output/emotic_ddp_internal_transfer_16shot_summary
  exit 0
fi

echo "Transfer did not pass pure-val gate; running Full Base5 internal upper screen."
mkdir -p "${FULL_DIR}"
if [[ ! -s "${FULL_DIR}/training_summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python train_emotic_ddp_internal_adapter.py \
    --ddp_checkpoint "${DDP_CHECKPOINT_DIR}/task0.pth" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output_dir "${FULL_DIR}" \
    --feature_cache_dir "${PATH_CACHE_DIR}" \
    --full_base5 \
    --seed 0 \
    --epochs 50 \
    --batch_size 256 \
    --feature_batch_size 4 \
    --workers 4 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --adapter_dim 16 \
    --residual_scale 0.01 \
    --identity_weight 1.0 \
    --loss_balance balanced \
    --eval_every_steps 10 \
    --early_stop_patience 10 \
    2>&1 | tee "${FULL_DIR}/train.log"
fi

FULL_PASSES="$(python - <<'PY'
import json
data = json.load(open('output/emotic_ddp_internal_full_base5_screen/training_summary.json'))
print('1' if data['best_val_gain'] > 0.1 else '0')
PY
)"

if [[ "${FULL_PASSES}" == "1" ]]; then
  echo "Full Base5 passed pure-val gate; running one locked all-task test evaluation."
  CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_internal_adapter.py \
    --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
    --adapter_checkpoint "${FULL_DIR}/best_adapter.pth" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output_dir "${FULL_DIR}" \
    --name emotic_ddp_internal_full_base5_screen \
    --cache_dir "${EVAL_CACHE_DIR}" \
    --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
    2>&1 | tee "${FULL_DIR}/eval.log"
else
  echo "Full Base5 did not pass pure-val gate; test evaluation was not run."
fi
