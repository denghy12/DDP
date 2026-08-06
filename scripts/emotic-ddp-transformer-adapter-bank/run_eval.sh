#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to a physical GPU index}"
TRAINING_MODE="${TRAINING_MODE:?Set TRAINING_MODE to full or 16shot}"
SEED="${SEED:?Set SEED to 0, 1, or 2}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
BANK_DIR="./output/emotic_ddp_transformer_adapter_bank_asl_${TRAINING_MODE}/seed${SEED}"
MANIFEST="${BANK_DIR}/transformer_adapter_bank_manifest.json"
RUN_NAME="emotic_ddp_transformer_adapter_bank_asl_${TRAINING_MODE}_seed${SEED}_evaluation"
OUTPUT_DIR="./output/${RUN_NAME}"

[[ -s "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}" >&2; exit 1; }
for task in 0 1 2 3 4 5 6 7; do
  [[ -s "${DDP_CHECKPOINT_DIR}/task${task}.pth" ]] || exit 1
  [[ -s "${BASELINE_SCORES_DIR}/task${task}_scores.pt" ]] || exit 1
done

COMPLETE="$(python - "${OUTPUT_DIR}/evaluation_summary.json" "${TRAINING_MODE}" "${SEED}" <<'PY'
import json
import os
import sys

path, mode, seed = sys.argv[1:]
complete = False
if os.path.isfile(path):
    data = json.load(open(path, encoding="utf-8"))
    protocol = data.get("protocol", {})
    complete = (
        protocol.get("training_mode") == mode
        and int(protocol.get("seed", -1)) == int(seed)
        and protocol.get("adapter_loss") == "asl"
        and protocol.get("checkpoint_rule") == "fixed_last_epoch"
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
CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_transformer_adapter_bank.py \
  --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
  --bank_manifest "${MANIFEST}" \
  --data_root ./datasets/EMOTIC \
  --clip_model_path ./pretrained/clip/ViT-B-16.pt \
  --output_dir "${OUTPUT_DIR}" \
  --name "${RUN_NAME}" \
  --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
  --fixed_threshold 0.5 \
  --batch_size 2 \
  --workers 4 \
  2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo "Evaluation JSON: ${OUTPUT_DIR}/evaluation_summary.json"
echo "Evaluation HTML: ${OUTPUT_DIR}/evaluation_summary.html"
