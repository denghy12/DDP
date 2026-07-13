#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_ddp_cls_internal_feature_cache}"
SCREEN_DIR="./output/emotic_ddp_cls_internal_transfer_screen"

mkdir -p "${SCREEN_DIR}"

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
  --val_cache "${CACHE_DIR}/task0_val_cls_path_features.pt" \
  --feature_key path_features \
  --output_dir "${SCREEN_DIR}" \
  2>&1 | tee "${SCREEN_DIR}/screen.log"

PASSES="$(python - <<'PY'
import json
data = json.load(open('output/emotic_ddp_cls_internal_transfer_screen/transfer_screen_summary.json'))
print('1' if data['passes_val_gate'] else '0')
PY
)"

if [[ "${PASSES}" != "1" ]]; then
  echo "CLS transfer did not pass pure-val stability gate; test was not run."
  exit 0
fi

echo "CLS transfer passed pure-val gate; running locked three-seed test."
for SEED in 0 1 2; do
  RUN_NAME="emotic_ddp_cls_internal_transfer_16shot_seed${SEED}"
  OUTPUT_DIR="./output/${RUN_NAME}"
  mkdir -p "${OUTPUT_DIR}"
  if [[ -s "${OUTPUT_DIR}/evaluation_summary.json" ]]; then
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
    --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
    --feature_batch_size 4 \
    --adapter_batch_size 256 \
    --workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
done

python summarize_emotic_ddp_internal_adapter.py \
  --run_prefix emotic_ddp_cls_internal_transfer_16shot_seed \
  --output_dir ./output/emotic_ddp_cls_internal_transfer_16shot_summary
