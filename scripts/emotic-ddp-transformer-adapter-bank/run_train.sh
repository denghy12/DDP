#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to a physical GPU index}"
TRAINING_MODE="${TRAINING_MODE:?Set TRAINING_MODE to full or 16shot}"
SEED="${SEED:?Set SEED to 0, 1, or 2}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BANK_DIR="./output/emotic_ddp_transformer_adapter_bank_asl_${TRAINING_MODE}/seed${SEED}"

case "${TRAINING_MODE}" in
  full|16shot) ;;
  *) echo "TRAINING_MODE must be full or 16shot" >&2; exit 2 ;;
esac
mkdir -p "${BANK_DIR}"

for task in ${TASKS}; do
  TASK_DIR="${BANK_DIR}/task${task}"
  CHECKPOINT="${TASK_DIR}/last_transformer_adapter.pth"
  SUMMARY="${TASK_DIR}/training_summary.json"
  DDP_CHECKPOINT="${DDP_CHECKPOINT_DIR}/task${task}.pth"
  [[ -s "${DDP_CHECKPOINT}" ]] || {
    echo "Missing DDP checkpoint ${DDP_CHECKPOINT}" >&2
    exit 1
  }
  COMPLETE="$(python - "${SUMMARY}" "${CHECKPOINT}" "${TRAINING_MODE}" "${SEED}" "${task}" <<'PY'
import json
import os
import sys

summary_path, checkpoint_path, mode, seed, task = sys.argv[1:]
complete = False
if os.path.isfile(summary_path) and os.path.isfile(checkpoint_path):
    data = json.load(open(summary_path, encoding="utf-8"))
    meta = data.get("transformer_task_adapter", {})
    protocol = data.get("protocol", {})
    architecture = meta.get("architecture", {})
    loss = meta.get("loss_config", {})
    complete = (
        meta.get("training_mode") == mode
        and int(meta.get("seed", -1)) == int(seed)
        and int(meta.get("task_id", -1)) == int(task)
        and meta.get("classification_loss") == "asl"
        and meta.get("checkpoint_rule") == "last_epoch"
        and architecture.get("layer_numbers") == list(range(4, 13))
        and int(architecture.get("bottleneck_dim", -1)) == 128
        and float(loss.get("gamma_neg", -1)) == 9.8
        and protocol.get("ddp_main_loss") == "two_way_bce_frozen_checkpoint"
        and protocol.get("test_dataset_constructed") is False
        and protocol.get("validation_used_for_checkpoint_selection") is False
    )
print("1" if complete else "0")
PY
)"
  if [[ "${COMPLETE}" == "1" ]]; then
    echo "Skip complete Transformer Adapter ${TRAINING_MODE} seed${SEED} task${task}"
    continue
  fi

  mkdir -p "${TASK_DIR}"
  INIT_ARGS=()
  if (( task > 0 )); then
    ANCHOR="${BANK_DIR}/task0/last_transformer_adapter.pth"
    [[ -s "${ANCHOR}" ]] || { echo "Missing Task 0 anchor ${ANCHOR}" >&2; exit 1; }
    INIT_ARGS=(--init_task0_adapter "${ANCHOR}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" python train_emotic_ddp_transformer_adapter.py \
    --task_id "${task}" \
    --training_mode "${TRAINING_MODE}" \
    --seed "${SEED}" \
    --ddp_checkpoint "${DDP_CHECKPOINT}" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output_dir "${TASK_DIR}" \
    --epochs 20 \
    --lr 4e-4 \
    --train_batch_size 4 \
    --effective_batch_size 64 \
    --eval_batch_size 4 \
    --workers 4 \
    --hidden_dim 768 \
    --bottleneck_dim 128 \
    --shots_per_class 16 \
    --asl_gamma_neg 9.8 \
    --asl_gamma_pos 0.0 \
    --asl_clip 0.05 \
    "${INIT_ARGS[@]}" \
    2>&1 | tee "${TASK_DIR}/train.log"
done

complete_bank=1
for task in 0 1 2 3 4 5 6 7; do
  [[ -s "${BANK_DIR}/task${task}/last_transformer_adapter.pth" ]] \
    || complete_bank=0
done
if (( complete_bank )); then
  python build_emotic_ddp_transformer_adapter_bank.py \
    --bank_dir "${BANK_DIR}" \
    --training_mode "${TRAINING_MODE}" \
    --seed "${SEED}"
  echo "Bank manifest: ${BANK_DIR}/transformer_adapter_bank_manifest.json"
else
  echo "Partial training complete (${TASKS}); Bank manifest waits for tasks 0--7."
fi
