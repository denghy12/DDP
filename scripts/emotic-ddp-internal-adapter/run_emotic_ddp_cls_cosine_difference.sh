#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_ddp_cls_internal_feature_cache}"
if [[ -z "${EXTERNAL_PATTERN:-}" ]]; then
  # Keep the literal placeholder outside a ${VAR:-default} expression: the
  # closing brace in "{seed}" would otherwise terminate that expression.
  EXTERNAL_PATTERN='./output/emotic_prototype_adapter_base5_16shot_seed{seed}/best_adapter.pth'
fi
SCREEN_DIR="${SCREEN_DIR:-./output/emotic_ddp_cls_cosine_difference_screen}"
SUMMARY_DIR="${SUMMARY_DIR:-./output/emotic_ddp_cls_cosine_difference_summary}"

mkdir -p "${SCREEN_DIR}"

for task in 0 1 2 3 4 5 6 7; do
  path="${DDP_CHECKPOINT_DIR}/task${task}.pth"
  [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
done

for seed in 0 1 2; do
  adapter_path="${EXTERNAL_PATTERN/\{seed\}/${seed}}"
  [[ -s "${adapter_path}" ]] || {
    echo "Missing external Adapter ${adapter_path}" >&2
    exit 1
  }
done

CUDA_VISIBLE_DEVICES="${GPU}" python cache_emotic_ddp_cls_features.py \
  --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
  --data_root ./datasets/EMOTIC \
  --clip_model_path ./pretrained/clip/ViT-B-16.pt \
  --cache_dir "${CACHE_DIR}" \
  --tasks 0 \
  --splits val \
  --feature_batch_size 4 \
  --workers 4 \
  2>&1 | tee "${SCREEN_DIR}/cache_val.log"

CUDA_VISIBLE_DEVICES="${GPU}" python screen_emotic_ddp_internal_adapter_transfer.py \
  --external_pattern "${EXTERNAL_PATTERN}" \
  --val_cache "${CACHE_DIR}/task0_val_cls_path_features.pt" \
  --feature_key path_features \
  --correction_mode cosine_difference \
  --residual_scales 0 0.001 0.003 0.01 0.03 0.1 \
  --minimum_val_gain 0.1 \
  --output_dir "${SCREEN_DIR}" \
  2>&1 | tee "${SCREEN_DIR}/screen.log"

PASSES="$(python - "${SCREEN_DIR}/transfer_screen_summary.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
if data.get("correction_mode") != "cosine_difference":
    raise RuntimeError("Screen output does not use cosine_difference")
print("1" if data["passes_val_gate"] else "0")
PY
)"

if [[ "${PASSES}" != "1" ]]; then
  echo "Cosine difference did not pass the pre-registered pure-val gate."
  echo "Test evaluation was intentionally not run."
  exit 0
fi

echo "Cosine difference passed pure-val screening; running locked test evaluation."
for SEED in 0 1 2; do
  RUN_NAME="emotic_ddp_cls_cosine_difference_seed${SEED}"
  OUTPUT_DIR="./output/${RUN_NAME}"
  mkdir -p "${OUTPUT_DIR}"
  COMPLETE="$(python - "${OUTPUT_DIR}" <<'PY'
import json
import os
import sys

path = os.path.join(sys.argv[1], "evaluation_summary.json")
complete = False
if os.path.isfile(path):
    data = json.load(open(path, encoding="utf-8"))
    complete = (
        data.get("protocol", {}).get("correction_mode")
        == "cosine_difference"
        and len(data.get("tasks", [])) == 8
    )
print("1" if complete else "0")
PY
)"
  if [[ "${COMPLETE}" == "1" ]]; then
    echo "Skip completed ${RUN_NAME}"
    continue
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_internal_adapter.py \
    --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
    --adapter_checkpoint "${SCREEN_DIR}/transferred_adapter_seed${SEED}.pth" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output_dir "${OUTPUT_DIR}" \
    --name "${RUN_NAME}" \
    --cache_dir "${CACHE_DIR}" \
    --feature_source cls \
    --correction_mode cosine_difference \
    --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
    --feature_batch_size 4 \
    --adapter_batch_size 256 \
    --workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
done

python summarize_emotic_ddp_internal_adapter.py \
  --run_prefix emotic_ddp_cls_cosine_difference_seed \
  --output_dir "${SUMMARY_DIR}" \
  --title "DDP CLS Cosine-Difference Logit Correction"

echo "Screen:  ${SCREEN_DIR}/transfer_screen_summary.html"
echo "Summary: ${SUMMARY_DIR}/summary.html"
