#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-emotic_prototype_fusion_task7}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${RUN_NAME}}"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_prototype_fusion.py \
  --ddp_scores ./output/emotic_b5c3_ddp_semantic_tau2_threshold050/task7_scores.pt \
  --prototype_checkpoint ./output/emotic_prototype_adapter_base5_balanced/best_adapter.pth \
  --val_cache ./output/emotic_clip_feature_cache/val_full_224_vitb16.pt \
  --test_cache ./output/emotic_clip_feature_cache/test_full_224_vitb16.pt \
  --output_dir "${OUTPUT_DIR}" \
  --seen_classes 26 \
  --batch_size 2048 \
  --beta_step 0.02 \
  --threshold_min 0.05 \
  --threshold_max 0.95 \
  --threshold_step 0.01 \
  "$@" 2>&1 | tee "${OUTPUT_DIR}/run.log"

