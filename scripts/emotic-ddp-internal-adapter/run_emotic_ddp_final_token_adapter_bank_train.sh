#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to the physical GPU index}"
TRAINING_MODE="${TRAINING_MODE:?Set TRAINING_MODE to full or 16shot}"
SEED="${SEED:?Set SEED to 0, 1, or 2}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-./output/emotic_ddp_final_token_training_cache}"

case "${TRAINING_MODE}" in
  full)
    EPOCHS="${FULL_EPOCHS:-50}"
    ;;
  16shot)
    EPOCHS="${SHOT16_EPOCHS:-200}"
    ;;
  *)
    echo "TRAINING_MODE must be full or 16shot" >&2
    exit 2
    ;;
esac

BANK_DIR="./output/emotic_ddp_final_token_adapter_bank_${TRAINING_MODE}/seed${SEED}"
mkdir -p "${BANK_DIR}"

for task in 0 1 2 3 4 5 6 7; do
  DDP_CHECKPOINT="${DDP_CHECKPOINT_DIR}/task${task}.pth"
  [[ -s "${DDP_CHECKPOINT}" ]] || {
    echo "Missing task-native DDP checkpoint ${DDP_CHECKPOINT}" >&2
    exit 1
  }
  TASK_DIR="${BANK_DIR}/task${task}"
  SUMMARY="${TASK_DIR}/training_summary.json"
  CHECKPOINT="${TASK_DIR}/final_adapter.pth"
  COMPLETE="$(python - "${SUMMARY}" "${CHECKPOINT}" "${TRAINING_MODE}" "${SEED}" "${task}" <<'PY'
import json
import os
import sys

summary_path, checkpoint_path, mode, seed, task = sys.argv[1:]
complete = False
if os.path.isfile(summary_path) and os.path.isfile(checkpoint_path):
    data = json.load(open(summary_path, encoding="utf-8"))
    meta = data.get("final_token_adapter", {})
    protocol = data.get("protocol", {})
    complete = (
        int(meta.get("schema_version", -1)) == 2
        and
        meta.get("training_mode") == mode
        and int(meta.get("seed", -1)) == int(seed)
        and int(meta.get("task_id", -1)) == int(task)
        and int(meta.get("token_count", -1)) == 197
        and protocol.get("checkpoint_selection") == "fixed last epoch"
        and protocol.get("validation_role") == "post-training reporting only"
        and protocol.get("test_dataset_constructed") is False
        and protocol.get("test_labels_used") is False
        and protocol.get("old_labels_supervised") is False
        and protocol.get("future_labels_supervised") is False
        and float(protocol.get("residual_scale_locked", -1)) == 0.03
        and float(protocol.get("decision_threshold_locked", -1)) == 0.5
        and protocol.get("training_health_check_enabled") is True
    )
print("1" if complete else "0")
PY
)"
  if [[ "${COMPLETE}" == "1" ]]; then
    echo "Skip complete Final-token ${TRAINING_MODE} seed${SEED} task${task}"
    continue
  fi

  mkdir -p "${TASK_DIR}"
  INIT_ARGS=()
  if (( task > 0 )); then
    INIT_CHECKPOINT="${BANK_DIR}/task0/final_adapter.pth"
    [[ -s "${INIT_CHECKPOINT}" ]] || {
      echo "Missing Final-token task0 anchor ${INIT_CHECKPOINT}" >&2
      exit 1
    }
    INIT_ARGS=(--init_adapter_checkpoint "${INIT_CHECKPOINT}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python train_emotic_ddp_final_token_adapter.py \
    --task_id "${task}" \
    --training_mode "${TRAINING_MODE}" \
    --seed "${SEED}" \
    --ddp_checkpoint "${DDP_CHECKPOINT}" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output_dir "${TASK_DIR}" \
    --token_cache_dir "${TOKEN_CACHE_DIR}" \
    --epochs "${EPOCHS}" \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --adapter_dim 128 \
    --residual_scale 0.03 \
    --identity_weight 0.1 \
    --shots_per_class 16 \
    --class_balanced_bce \
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
  --training_mode "${TRAINING_MODE}" \
  --seed "${SEED}" \
  --residual_scale 0.03

echo "Bank manifest: ${BANK_DIR}/final_token_adapter_bank_manifest.json"
