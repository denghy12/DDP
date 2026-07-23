#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to the physical GPU index}"
TRAINING_MODE="${TRAINING_MODE:?Set TRAINING_MODE to full or 16shot}"
LOSS_NAME="${LOSS_NAME:?Set LOSS_NAME}"
SEED="${SEED:?Set SEED to 0, 1, or 2}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_ddp_cls_feature_correction_cache}"

BANK_DIR="./output/emotic_ddp_task_adapter_bank_loss_${LOSS_NAME}_${TRAINING_MODE}/seed${SEED}"
MANIFEST="${BANK_DIR}/adapter_bank_manifest.json"
RUN_NAME="emotic_ddp_task_adapter_bank_loss_${LOSS_NAME}_${TRAINING_MODE}_feature_difference_seed${SEED}"
OUTPUT_DIR="./output/${RUN_NAME}"

[[ -s "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}" >&2; exit 1; }
for task in 0 1 2 3 4 5 6 7; do
  [[ -s "${DDP_CHECKPOINT_DIR}/task${task}.pth" ]] || exit 1
  [[ -s "${BASELINE_SCORES_DIR}/task${task}_scores.pt" ]] || exit 1
done

COMPLETE="$(python - "${OUTPUT_DIR}/evaluation_summary.json" "${TRAINING_MODE}" "${LOSS_NAME}" "${SEED}" <<'PY'
import json
import os
import sys

path, mode, loss_name, seed = sys.argv[1:]
complete = False
if os.path.isfile(path):
    data = json.load(open(path, encoding="utf-8"))
    protocol = data.get("protocol", {})
    complete = (
        protocol.get("training_mode") == mode
        and protocol.get("classification_loss") == loss_name
        and protocol.get("checkpoint_rule") == "last_epoch"
        and int(protocol.get("seed", -1)) == int(seed)
        and protocol.get("correction_mode") == "feature_difference"
        and float(protocol.get("inference_alpha", -1)) == 0.03
        and float(protocol.get("decision_threshold", -1)) == 0.5
        and protocol.get("test_used_for_selection") is False
        and len(data.get("tasks", [])) == 8
    )
print("1" if complete else "0")
PY
)"
if [[ "${COMPLETE}" == "1" ]]; then
  echo "Skip complete ${RUN_NAME}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_task_adapter_bank.py \
  --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
  --bank_manifest "${MANIFEST}" \
  --data_root ./datasets/EMOTIC \
  --clip_model_path ./pretrained/clip/ViT-B-16.pt \
  --output_dir "${OUTPUT_DIR}" \
  --name "${RUN_NAME}" \
  --cache_dir "${CACHE_DIR}" \
  --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
  --fixed_threshold 0.5 \
  --feature_batch_size 4 \
  --adapter_batch_size 256 \
  --workers 4 \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo "Evaluation JSON: ${OUTPUT_DIR}/evaluation_summary.json"
echo "Evaluation HTML: ${OUTPUT_DIR}/evaluation_summary.html"
