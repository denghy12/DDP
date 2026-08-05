#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
LOSS_NAME="${LOSS_NAME:-asl}"
SEED="${SEED:-0}"
TRAIN_RUN="${TRAIN_RUN:-emotic_b5c3_ddp_main_${LOSS_NAME}_seed${SEED}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./output/${TRAIN_RUN}/checkpoints}"

for task in 0 1 2 3 4 5 6 7; do
  checkpoint="${CHECKPOINT_DIR}/task${task}.pth"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

for split in val test; do
  EVAL_NAME="${TRAIN_RUN}_${split}_threshold050"
  EVAL_DIR="./output/${EVAL_NAME}"
  if [[ -s "${EVAL_DIR}/evaluation_summary.json" ]]; then
    python - "${EVAL_DIR}/evaluation_summary.json" "${split}" \
      "${LOSS_NAME}" "${SEED}" <<'PY'
import json
import sys

path, expected_split, expected_loss, expected_seed = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
protocol = data.get("training_protocol", {})
if data.get("eval_splits") != [expected_split]:
    raise RuntimeError(f"Existing evaluation split mismatch: {path}")
if protocol.get("ddp_main_classification_loss") != expected_loss:
    raise RuntimeError(f"Existing evaluation loss mismatch: {path}")
if int(protocol.get("seed", -1)) != int(expected_seed):
    raise RuntimeError(f"Existing evaluation seed mismatch: {path}")
if len(data.get("rows", [])) != 8:
    raise RuntimeError(f"Existing evaluation is incomplete: {path}")
print(f"Reusing completed evaluation: {path}")
PY
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_all_tasks.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-root ./datasets/EMOTIC \
    --clip-model-path ./pretrained/clip/ViT-B-16.pt \
    --output-dir "${EVAL_DIR}" \
    --name "${EVAL_NAME}" \
    --threshold 0.50 \
    --eval-splits "${split}" \
    --batch-size 4 \
    --workers 4 \
    --t-min 1 \
    --t-max 2 \
    --t-gamma 0.7 \
    2>&1 | tee "${EVAL_DIR}.log"
done
