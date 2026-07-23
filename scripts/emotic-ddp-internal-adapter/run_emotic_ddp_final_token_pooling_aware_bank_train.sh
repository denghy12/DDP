#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to the physical GPU index}"
SEED="${SEED:?Set SEED to 0, 1, or 2}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-./output/emotic_ddp_final_token_training_cache}"
BANK_DIR="./output/emotic_ddp_final_token_pooling_aware_bank_full/seed${SEED}"

LR="3e-5"
MAX_STEPS=300
IDENTITY_WEIGHT=0.1
POOLING_WEIGHT=100.0
ATTENTION_WEIGHT=100.0
MARGIN_WEIGHT=1.0
MARGIN_BETA=1.0

mkdir -p "${BANK_DIR}"

is_complete() {
  python - "$1" "$2" "${SEED}" "$3" <<'PY'
import json
import os
import sys

summary_path, checkpoint_path, seed, task = sys.argv[1:]
complete = False
if os.path.isfile(summary_path) and os.path.isfile(checkpoint_path):
    data = json.load(open(summary_path, encoding="utf-8"))
    meta = data.get("final_token_adapter", {})
    protocol = data.get("protocol", {})
    regularization = protocol.get("pooling_aware_regularization", {})
    args = data.get("args", {})
    complete = (
        int(meta.get("schema_version", -1)) == 2
        and meta.get("training_mode") == "full"
        and int(meta.get("seed", -1)) == int(seed)
        and int(meta.get("task_id", -1)) == int(task)
        and int(meta.get("token_count", -1)) == 197
        and protocol.get("checkpoint_selection")
        == "fixed final diagnostic optimizer step"
        and protocol.get("validation_role") == "post-training reporting only"
        and protocol.get("test_dataset_constructed") is False
        and protocol.get("test_labels_used") is False
        and protocol.get("old_labels_supervised") is False
        and protocol.get("future_labels_supervised") is False
        and protocol.get("training_health_check_enabled") is True
        and data.get("training_health_check", {}).get("status") == "passed"
        and int(data.get("optimizer_steps", -1)) == 300
        and int(args.get("max_optimizer_steps", -1)) == 300
        and abs(float(args.get("lr", -1)) - 3e-5) < 1e-12
        and float(regularization.get("pooling_weight", -1)) == 100.0
        and float(regularization.get("attention_weight", -1)) == 100.0
        and regularization.get("attention_kl_implementation")
        == "log_softmax_stable"
        and float(regularization.get("margin_weight", -1)) == 1.0
        and float(regularization.get("margin_beta", -1)) == 1.0
        and float(protocol.get("residual_scale_locked", -1)) == 0.03
        and float(protocol.get("decision_threshold_locked", -1)) == 0.5
        and protocol.get("training_precision") == "float32"
    )
print("1" if complete else "0")
PY
}

for task in 0 1 2 3 4 5 6 7; do
  DDP_CHECKPOINT="${DDP_CHECKPOINT_DIR}/task${task}.pth"
  [[ -s "${DDP_CHECKPOINT}" ]] || {
    echo "Missing task-native DDP checkpoint ${DDP_CHECKPOINT}" >&2
    exit 1
  }
  TASK_DIR="${BANK_DIR}/task${task}"
  SUMMARY="${TASK_DIR}/training_summary.json"
  CHECKPOINT="${TASK_DIR}/final_adapter.pth"
  if [[ "$(is_complete "${SUMMARY}" "${CHECKPOINT}" "${task}")" == "1" ]]; then
    echo "Skip complete pooling-aware Full seed${SEED} task${task}"
    continue
  fi

  mkdir -p "${TASK_DIR}"
  INIT_ARGS=()
  if (( task > 0 )); then
    INIT_CHECKPOINT="${BANK_DIR}/task0/final_adapter.pth"
    [[ -s "${INIT_CHECKPOINT}" ]] || {
      echo "Missing pooling-aware task0 anchor ${INIT_CHECKPOINT}" >&2
      exit 1
    }
    INIT_ARGS=(--init_adapter_checkpoint "${INIT_CHECKPOINT}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" python train_emotic_ddp_final_token_adapter.py \
    --task_id "${task}" \
    --training_mode full \
    --seed "${SEED}" \
    --ddp_checkpoint "${DDP_CHECKPOINT}" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output_dir "${TASK_DIR}" \
    --token_cache_dir "${TOKEN_CACHE_DIR}" \
    --epochs 50 \
    --max_optimizer_steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --weight_decay 1e-4 \
    --adapter_dim 128 \
    --residual_scale 0.03 \
    --identity_weight "${IDENTITY_WEIGHT}" \
    --pooling_weight "${POOLING_WEIGHT}" \
    --attention_weight "${ATTENTION_WEIGHT}" \
    --margin_weight "${MARGIN_WEIGHT}" \
    --margin_beta "${MARGIN_BETA}" \
    --shots_per_class 16 \
    --class_balanced_bce \
    --full_precision_training \
    --training_health_check \
    --health_check_steps 2 \
    --health_min_signal 1e-12 \
    --feature_batch_size 2 \
    --adapter_batch_size 4 \
    --cache_shard_samples 32 \
    --workers 4 \
    "${INIT_ARGS[@]}" \
    2>&1 | tee "${TASK_DIR}/train.log"
done

python build_emotic_ddp_final_token_adapter_bank.py \
  --bank_dir "${BANK_DIR}" \
  --training_mode full \
  --seed "${SEED}" \
  --residual_scale 0.03 \
  --selection fixed_optimizer_steps_no_validation_selection \
  --optimizer_steps "${MAX_STEPS}" \
  --learning_rate "${LR}" \
  --identity_weight "${IDENTITY_WEIGHT}" \
  --pooling_weight "${POOLING_WEIGHT}" \
  --attention_weight "${ATTENTION_WEIGHT}" \
  --attention_kl_implementation log_softmax_stable \
  --margin_weight "${MARGIN_WEIGHT}" \
  --margin_beta "${MARGIN_BETA}" \
  --training_precision float32

echo "Bank manifest: ${BANK_DIR}/final_token_adapter_bank_manifest.json"
