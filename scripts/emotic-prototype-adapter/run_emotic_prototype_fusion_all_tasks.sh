#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-emotic_prototype_fusion_all_tasks}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${RUN_NAME}}"
DDP_VAL_SCORES_ROOT="${DDP_VAL_SCORES_ROOT:-./output/emotic_b5c3_ddp_semantic_tau2_val_threshold_sweep}"
DDP_TEST_SCORES_DIR="${DDP_TEST_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
PROTOTYPE_CHECKPOINT="${PROTOTYPE_CHECKPOINT:-./output/emotic_prototype_adapter_base5_balanced/best_adapter.pth}"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_prototype_fusion_all_tasks.py \
  --ddp_val_scores_root "${DDP_VAL_SCORES_ROOT}" \
  --ddp_test_scores_dir "${DDP_TEST_SCORES_DIR}" \
  --prototype_checkpoint "${PROTOTYPE_CHECKPOINT}" \
  --val_cache ./output/emotic_clip_feature_cache/val_full_224_vitb16.pt \
  --test_cache ./output/emotic_clip_feature_cache/test_full_224_vitb16.pt \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 2048 \
  --beta_step 0.02 \
  --threshold_min 0.05 \
  --threshold_max 0.95 \
  --threshold_step 0.01 \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/run.log"
