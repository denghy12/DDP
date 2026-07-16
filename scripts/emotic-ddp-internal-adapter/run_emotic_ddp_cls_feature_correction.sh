#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
ADAPTER_SOURCE="${ADAPTER_SOURCE:-full_base5}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_ddp_cls_feature_correction_cache}"

case "${ADAPTER_SOURCE}" in
  ddp_aux_full_base5)
    RUN_STEM="emotic_ddp_prompt_free_auxiliary_cls_feature_correction"
    TITLE="DDP Prompt-Free Auxiliary Base5 → Prompted CLS-to-Pooled Feature Correction"
    ADAPTER_CHECKPOINTS=(
      "./output/emotic_ddp_prompt_free_auxiliary_base5_seed0/best_adapter.pth"
      "./output/emotic_ddp_prompt_free_auxiliary_base5_seed1/best_adapter.pth"
      "./output/emotic_ddp_prompt_free_auxiliary_base5_seed2/best_adapter.pth"
    )
    ;;
  full_base5)
    RUN_STEM="emotic_ddp_cls_full_base5_feature_correction"
    TITLE="Full Base5 → Internal CLS-to-Pooled Feature Correction"
    ADAPTER_CHECKPOINTS=(
      "./output/emotic_prototype_adapter_base5_balanced/best_adapter.pth"
      "./output/emotic_prototype_adapter_base5_balanced_seed1/best_adapter.pth"
      "./output/emotic_prototype_adapter_base5_balanced_seed2/best_adapter.pth"
    )
    ;;
  16shot)
    RUN_STEM="emotic_ddp_cls_feature_correction"
    TITLE="Base5 16-shot → Internal CLS-to-Pooled Feature Correction"
    ADAPTER_CHECKPOINTS=(
      "./output/emotic_prototype_adapter_base5_16shot_seed0/best_adapter.pth"
      "./output/emotic_prototype_adapter_base5_16shot_seed1/best_adapter.pth"
      "./output/emotic_prototype_adapter_base5_16shot_seed2/best_adapter.pth"
    )
    ;;
  *)
    echo "Unknown ADAPTER_SOURCE=${ADAPTER_SOURCE}; use ddp_aux_full_base5, full_base5, or 16shot" >&2
    exit 2
    ;;
esac

SCREEN_DIR="${SCREEN_DIR:-./output/${RUN_STEM}_screen}"
SUMMARY_DIR="${SUMMARY_DIR:-./output/${RUN_STEM}_summary}"
mkdir -p "${SCREEN_DIR}"

for task in 0 1 2 3 4 5 6 7; do
  path="${DDP_CHECKPOINT_DIR}/task${task}.pth"
  [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
  path="${BASELINE_SCORES_DIR}/task${task}_scores.pt"
  [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
done
for path in "${ADAPTER_CHECKPOINTS[@]}"; do
  [[ -s "${path}" ]] || { echo "Missing Adapter ${path}" >&2; exit 1; }
done

# Build a paired validation cache containing both class-specific CLS features
# and the original DDP pooled representations. No test sample is used here.
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

# Use exactly the same task0-validation global-alpha protocol as the previous
# Feature/Cosine Difference experiments. There is no class gate or task alpha.
CUDA_VISIBLE_DEVICES="${GPU}" python screen_emotic_ddp_internal_adapter_transfer.py \
  --adapter_checkpoints "${ADAPTER_CHECKPOINTS[@]}" \
  --val_cache "${CACHE_DIR}/task0_val_cls_path_features.pt" \
  --feature_key path_features \
  --pooled_feature_key pooled_features \
  --correction_mode feature_correction \
  --residual_scales 0 0.001 0.003 0.01 0.03 0.1 \
  --minimum_val_gain 0.1 \
  --output_dir "${SCREEN_DIR}" \
  2>&1 | tee "${SCREEN_DIR}/screen.log"

PASSES="$(python - "${SCREEN_DIR}/transfer_screen_summary.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
if data.get("correction_mode") != "feature_correction":
    raise RuntimeError("Screen output does not use feature_correction")
print("1" if data["passes_val_gate"] else "0")
PY
)"

if [[ "${PASSES}" != "1" ]]; then
  echo "Feature Correction did not pass the pre-registered pure-val gate."
  echo "Test evaluation was intentionally not run."
  echo "Screen JSON: ${SCREEN_DIR}/transfer_screen_summary.json"
  echo "Screen HTML: ${SCREEN_DIR}/transfer_screen_summary.html"
  exit 0
fi

echo "Feature Correction passed pure-val screening; running locked test evaluation."
for SEED in 0 1 2; do
  RUN_NAME="${RUN_STEM}_seed${SEED}"
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
    protocol = data.get("protocol", {})
    complete = (
        protocol.get("correction_mode") == "feature_correction"
        and protocol.get("norm_preserving") is True
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
    --correction_mode feature_correction \
    --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
    --feature_batch_size 4 \
    --adapter_batch_size 256 \
    --workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
done

python summarize_emotic_ddp_internal_adapter.py \
  --run_prefix "${RUN_STEM}_seed" \
  --output_dir "${SUMMARY_DIR}" \
  --title "${TITLE}"

echo "Screen JSON:  ${SCREEN_DIR}/transfer_screen_summary.json"
echo "Screen HTML:  ${SCREEN_DIR}/transfer_screen_summary.html"
echo "Summary JSON: ${SUMMARY_DIR}/summary.json"
echo "Summary HTML: ${SUMMARY_DIR}/summary.html"
