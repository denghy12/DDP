#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_ddp_cls_internal_feature_cache}"
COMPARISON_DIR="${COMPARISON_DIR:-./output/emotic_ddp_cls_full_base5_comparison}"

FULL_CHECKPOINTS=(
  "./output/emotic_prototype_adapter_base5_balanced/best_adapter.pth"
  "./output/emotic_prototype_adapter_base5_balanced_seed1/best_adapter.pth"
  "./output/emotic_prototype_adapter_base5_balanced_seed2/best_adapter.pth"
)

mkdir -p "${COMPARISON_DIR}"

for task in 0 1 2 3 4 5 6 7; do
  path="${DDP_CHECKPOINT_DIR}/task${task}.pth"
  [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
  path="${BASELINE_SCORES_DIR}/task${task}_scores.pt"
  [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
done
for path in "${FULL_CHECKPOINTS[@]}"; do
  [[ -s "${path}" ]] || { echo "Missing Full Base5 Adapter ${path}" >&2; exit 1; }
done

# The two Adapter formulas are screened on task0 validation before any new
# Adapter test evaluation. This cache is shared and does not contain test data.
CUDA_VISIBLE_DEVICES="${GPU}" python cache_emotic_ddp_cls_features.py \
  --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
  --data_root ./datasets/EMOTIC \
  --clip_model_path ./pretrained/clip/ViT-B-16.pt \
  --cache_dir "${CACHE_DIR}" \
  --tasks 0 \
  --splits val \
  --feature_batch_size 4 \
  --workers 4 \
  2>&1 | tee "${COMPARISON_DIR}/cache_task0_val.log"

screen_method() {
  local correction_mode="$1"
  local slug="$2"
  local screen_dir="./output/emotic_ddp_cls_full_base5_${slug}_screen"
  mkdir -p "${screen_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" python screen_emotic_ddp_internal_adapter_transfer.py \
    --external_checkpoints "${FULL_CHECKPOINTS[@]}" \
    --val_cache "${CACHE_DIR}/task0_val_cls_path_features.pt" \
    --feature_key path_features \
    --correction_mode "${correction_mode}" \
    --residual_scales 0 0.001 0.003 0.01 0.03 0.1 \
    --minimum_val_gain 0.1 \
    --output_dir "${screen_dir}" \
    2>&1 | tee "${screen_dir}/screen.log"
}

screen_method linear_residual feature_difference
screen_method cosine_difference cosine_difference

passes_screen() {
  local slug="$1"
  python - "./output/emotic_ddp_cls_full_base5_${slug}_screen/transfer_screen_summary.json" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
print("1" if data["passes_val_gate"] else "0")
PY
}

FEATURE_PASSES="$(passes_screen feature_difference)"
COSINE_PASSES="$(passes_screen cosine_difference)"
echo "Feature difference passes validation gate: ${FEATURE_PASSES}"
echo "Cosine difference passes validation gate: ${COSINE_PASSES}"

# Produce an explicit original-DDP run with the same caches, temperature
# schedule, task0-val/per-task-val threshold policy, and forgetting definition.
BASELINE_DIR="./output/emotic_ddp_cls_full_base5_ddp_baseline"
BASELINE_COMPLETE="$(python - "${BASELINE_DIR}" <<'PY'
import json
import os
import sys

path = os.path.join(sys.argv[1], "evaluation_summary.json")
complete = False
if os.path.isfile(path):
    data = json.load(open(path, encoding="utf-8"))
    complete = data.get("protocol", {}).get("ddp_only") and len(data.get("tasks", [])) == 8
print("1" if complete else "0")
PY
)"
if [[ "${BASELINE_COMPLETE}" != "1" ]]; then
  mkdir -p "${BASELINE_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_internal_adapter.py \
    --ddp_only \
    --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output_dir "${BASELINE_DIR}" \
    --name emotic_ddp_cls_full_base5_ddp_baseline \
    --cache_dir "${CACHE_DIR}" \
    --feature_source cls \
    --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
    --feature_batch_size 4 \
    --adapter_batch_size 256 \
    --workers 4 \
    2>&1 | tee "${BASELINE_DIR}/eval.log"
else
  echo "Skip completed original DDP baseline"
fi

evaluate_method() {
  local correction_mode="$1"
  local slug="$2"
  local title="$3"
  local screen_dir="./output/emotic_ddp_cls_full_base5_${slug}_screen"
  local run_prefix="emotic_ddp_cls_full_base5_${slug}_seed"
  local summary_dir="./output/emotic_ddp_cls_full_base5_${slug}_summary"
  for seed in 0 1 2; do
    local run_name="${run_prefix}${seed}"
    local output_dir="./output/${run_name}"
    local complete
    complete="$(python - "${output_dir}" "${correction_mode}" <<'PY'
import json
import os
import sys

path = os.path.join(sys.argv[1], "evaluation_summary.json")
complete = False
if os.path.isfile(path):
    data = json.load(open(path, encoding="utf-8"))
    complete = (
        data.get("protocol", {}).get("correction_mode") == sys.argv[2]
        and len(data.get("tasks", [])) == 8
        and "base5_balanced" in data.get("inputs", {}).get("transfer_selection", {}).get("source", "")
    )
print("1" if complete else "0")
PY
)"
    if [[ "${complete}" == "1" ]]; then
      echo "Skip completed ${run_name}"
      continue
    fi
    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_internal_adapter.py \
      --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
      --adapter_checkpoint "${screen_dir}/transferred_adapter_seed${seed}.pth" \
      --data_root ./datasets/EMOTIC \
      --clip_model_path ./pretrained/clip/ViT-B-16.pt \
      --output_dir "${output_dir}" \
      --name "${run_name}" \
      --cache_dir "${CACHE_DIR}" \
      --feature_source cls \
      --correction_mode "${correction_mode}" \
      --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
      --feature_batch_size 4 \
      --adapter_batch_size 256 \
      --workers 4 \
      2>&1 | tee "${output_dir}/eval.log"
  done
  python summarize_emotic_ddp_internal_adapter.py \
    --run_prefix "${run_prefix}" \
    --output_dir "${summary_dir}" \
    --title "${title}"
}

if [[ "${FEATURE_PASSES}" == "1" ]]; then
  evaluate_method \
    linear_residual \
    feature_difference \
    "Full Base5 → Internal CLS Feature Difference"
else
  echo "Feature difference failed the pure-validation gate; test was not run."
fi

if [[ "${COSINE_PASSES}" == "1" ]]; then
  evaluate_method \
    cosine_difference \
    cosine_difference \
    "Full Base5 → Internal CLS Cosine Difference"
else
  echo "Cosine difference failed the pure-validation gate; test was not run."
fi

if [[ "${FEATURE_PASSES}" == "1" && "${COSINE_PASSES}" == "1" ]]; then
  python summarize_emotic_ddp_cls_full_base5_comparison.py \
    --output_dir "${COMPARISON_DIR}"
  echo "Comparison JSON: ${COMPARISON_DIR}/comparison_summary.json"
  echo "Comparison HTML: ${COMPARISON_DIR}/comparison_summary.html"
else
  echo "Combined comparison was not generated because at least one method failed validation."
fi

