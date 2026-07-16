#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
TRAINING_MODE="${TRAINING_MODE:-full_base5}"
FORMULA="${FORMULA:-feature_correction}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_ddp_cls_feature_correction_cache}"

case "${TRAINING_MODE}" in
  full_base5)
    ADAPTER_STEM="emotic_ddp_prompt_free_auxiliary_base5_seed"
    DATA_TITLE="Full Base5"
    if [[ "${FORMULA}" == "feature_correction" ]]; then
      # Reuse the completed result rather than creating a duplicate directory.
      RUN_STEM="emotic_ddp_prompt_free_auxiliary_cls_feature_correction"
    else
      RUN_STEM="emotic_ddp_prompt_free_auxiliary_cls_${FORMULA}"
    fi
    ;;
  16shot)
    ADAPTER_STEM="emotic_ddp_prompt_free_auxiliary_base5_16shot_seed"
    DATA_TITLE="Base5 16-shot"
    RUN_STEM="emotic_ddp_prompt_free_auxiliary_16shot_cls_${FORMULA}"
    ;;
  *)
    echo "Unknown TRAINING_MODE=${TRAINING_MODE}; use full_base5 or 16shot" >&2
    exit 2
    ;;
esac

case "${FORMULA}" in
  feature_difference)
    CORRECTION_MODE="linear_residual"
    FORMULA_TITLE="Feature Difference"
    ;;
  cosine_difference)
    CORRECTION_MODE="cosine_difference"
    FORMULA_TITLE="Cosine Difference"
    ;;
  feature_correction)
    CORRECTION_MODE="feature_correction"
    FORMULA_TITLE="Feature Correction"
    ;;
  *)
    echo "Unknown FORMULA=${FORMULA}; use feature_difference, cosine_difference, or feature_correction" >&2
    exit 2
    ;;
esac

SCREEN_DIR="${SCREEN_DIR:-./output/${RUN_STEM}_screen}"
SUMMARY_DIR="${SUMMARY_DIR:-./output/${RUN_STEM}_summary}"
ADAPTER_CHECKPOINTS=(
  "./output/${ADAPTER_STEM}0/best_adapter.pth"
  "./output/${ADAPTER_STEM}1/best_adapter.pth"
  "./output/${ADAPTER_STEM}2/best_adapter.pth"
)
mkdir -p "${SCREEN_DIR}"

for task in 0 1 2 3 4 5 6 7; do
  checkpoint="${DDP_CHECKPOINT_DIR}/task${task}.pth"
  baseline="${BASELINE_SCORES_DIR}/task${task}_scores.pt"
  [[ -s "${checkpoint}" ]] || { echo "Missing ${checkpoint}" >&2; exit 1; }
  [[ -s "${baseline}" ]] || { echo "Missing ${baseline}" >&2; exit 1; }
done
for checkpoint in "${ADAPTER_CHECKPOINTS[@]}"; do
  [[ -s "${checkpoint}" ]] || { echo "Missing integrated Adapter ${checkpoint}" >&2; exit 1; }
done

SCREEN_COMPLETE="$(python - "${SCREEN_DIR}" "${CORRECTION_MODE}" "${ADAPTER_STEM}" <<'PY'
import json
import os
import sys

screen_dir, expected_mode, source_stem = sys.argv[1:]
path = os.path.join(screen_dir, "transfer_screen_summary.json")
complete = False
if os.path.isfile(path):
    data = json.load(open(path, encoding="utf-8"))
    sources = data.get("args", {}).get("adapter_checkpoints") or []
    complete = (
        data.get("correction_mode") == expected_mode
        and len(sources) == 3
        and all(source_stem in source for source in sources)
        and (
            data.get("passes_val_gate") is False
            or all(
                os.path.isfile(os.path.join(screen_dir, f"transferred_adapter_seed{seed}.pth"))
                for seed in (0, 1, 2)
            )
        )
    )
print("1" if complete else "0")
PY
)"

if [[ "${SCREEN_COMPLETE}" != "1" ]]; then
  # Cache creation is protected by a filesystem lock and is safe when the two
  # GPU workers reach this step concurrently.
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
    --adapter_checkpoints "${ADAPTER_CHECKPOINTS[@]}" \
    --val_cache "${CACHE_DIR}/task0_val_cls_path_features.pt" \
    --feature_key path_features \
    --pooled_feature_key pooled_features \
    --correction_mode "${CORRECTION_MODE}" \
    --residual_scales 0 0.001 0.003 0.01 0.03 0.1 \
    --minimum_val_gain 0.1 \
    --output_dir "${SCREEN_DIR}" \
    2>&1 | tee "${SCREEN_DIR}/screen.log"
else
  echo "Skip completed validation screen: ${SCREEN_DIR}"
fi

PASSES="$(python - "${SCREEN_DIR}/transfer_screen_summary.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
print("1" if data.get("passes_val_gate") else "0")
PY
)"
if [[ "${PASSES}" != "1" ]]; then
  echo "${DATA_TITLE} ${FORMULA_TITLE} failed the pure-validation gate; test was not run."
  exit 0
fi

for seed in 0 1 2; do
  RUN_NAME="${RUN_STEM}_seed${seed}"
  OUTPUT_DIR="./output/${RUN_NAME}"
  mkdir -p "${OUTPUT_DIR}"
  COMPLETE="$(python - "${OUTPUT_DIR}" "${CORRECTION_MODE}" "${ADAPTER_STEM}" <<'PY'
import json
import os
import sys

output_dir, expected_mode, source_stem = sys.argv[1:]
path = os.path.join(output_dir, "evaluation_summary.json")
complete = False
if os.path.isfile(path):
    data = json.load(open(path, encoding="utf-8"))
    transfer = data.get("inputs", {}).get("transfer_selection") or {}
    complete = (
        data.get("protocol", {}).get("correction_mode") == expected_mode
        and data.get("protocol", {}).get("adapter_training_source")
        == "ddp_owned_prompt_free_auxiliary_branch"
        and source_stem in transfer.get("source", "")
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
    --adapter_checkpoint "${SCREEN_DIR}/transferred_adapter_seed${seed}.pth" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output_dir "${OUTPUT_DIR}" \
    --name "${RUN_NAME}" \
    --cache_dir "${CACHE_DIR}" \
    --feature_source cls \
    --correction_mode "${CORRECTION_MODE}" \
    --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
    --feature_batch_size 4 \
    --adapter_batch_size 256 \
    --workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
done

python summarize_emotic_ddp_internal_adapter.py \
  --run_prefix "${RUN_STEM}_seed" \
  --output_dir "${SUMMARY_DIR}" \
  --title "DDP-owned Auxiliary ${DATA_TITLE}: ${FORMULA_TITLE}"

echo "Screen JSON:  ${SCREEN_DIR}/transfer_screen_summary.json"
echo "Screen HTML:  ${SCREEN_DIR}/transfer_screen_summary.html"
echo "Summary JSON: ${SUMMARY_DIR}/summary.json"
echo "Summary HTML: ${SUMMARY_DIR}/summary.html"
