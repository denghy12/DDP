#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-emotic_ddp_final_token_training_healthcheck_full_task0_seed${SEED}_v2}"
OUTPUT_DIR="./output/${RUN_NAME}"
DDP_CHECKPOINT="${DDP_CHECKPOINT:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task0.pth}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-./output/emotic_ddp_final_token_training_cache}"
HEALTH_EPOCHS="${HEALTH_EPOCHS:-1}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
IDENTITY_WEIGHT="${IDENTITY_WEIGHT:-0.1}"
POOLING_WEIGHT="${POOLING_WEIGHT:-0.0}"
ATTENTION_WEIGHT="${ATTENTION_WEIGHT:-0.0}"
MARGIN_WEIGHT="${MARGIN_WEIGHT:-0.0}"
MARGIN_BETA="${MARGIN_BETA:-1.0}"

[[ -s "${DDP_CHECKPOINT}" ]] || {
  echo "Missing task0 DDP checkpoint: ${DDP_CHECKPOINT}" >&2
  exit 1
}

if [[ -e "${OUTPUT_DIR}/training_summary.json" ]]; then
  echo "Refusing to overwrite an existing health run: ${OUTPUT_DIR}" >&2
  echo "Set a new RUN_NAME to launch another diagnostic run." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python train_emotic_ddp_final_token_adapter.py \
  --task_id 0 \
  --training_mode full \
  --seed "${SEED}" \
  --ddp_checkpoint "${DDP_CHECKPOINT}" \
  --data_root ./datasets/EMOTIC \
  --clip_model_path ./pretrained/clip/ViT-B-16.pt \
  --output_dir "${OUTPUT_DIR}" \
  --token_cache_dir "${TOKEN_CACHE_DIR}" \
  --epochs "${HEALTH_EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --adapter_dim 128 \
  --residual_scale 0.03 \
  --identity_weight "${IDENTITY_WEIGHT}" \
  --pooling_weight "${POOLING_WEIGHT}" \
  --attention_weight "${ATTENTION_WEIGHT}" \
  --margin_weight "${MARGIN_WEIGHT}" \
  --margin_beta "${MARGIN_BETA}" \
  --shots_per_class 16 \
  --class_balanced_bce \
  --feature_batch_size 2 \
  --adapter_batch_size 4 \
  --cache_shard_samples 32 \
  --workers 4 \
  --training_health_check \
  --health_check_steps 2 \
  --health_min_signal 1e-12 \
  --max_optimizer_steps "${MAX_OPTIMIZER_STEPS}" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

python - "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
health = json.loads((root / "training_health_check.json").read_text())
summary = json.loads((root / "training_summary.json").read_text())
if health.get("status") != "passed":
    raise SystemExit("TRAINING_HEALTH_CHECK_FAILED")
checks = health.get("checks", {})
if not checks or not all(checks.values()):
    raise SystemExit("TRAINING_HEALTH_CHECK_HAS_FAILED_SUBCHECKS")
print("TRAINING_HEALTH_CHECK_PASSED")
print("optimizer_steps:", summary["optimizer_steps"])
print("learning_rate:", summary["args"]["lr"])
print("initial_val_mAP:", summary["initial_val_mAP"])
print("final_val_mAP:", summary["final_val_mAP"])
print("reporting_val_gain:", summary["reporting_val_gain"])
print("JSON:", root / "training_health_check.json")
print("HTML:", root / "training_health_check.html")
PY
