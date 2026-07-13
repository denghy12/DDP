#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
CACHE_DIR="${CACHE_DIR:-./output/emotic_ddp_cls_internal_feature_cache}"
ADAPTER_DIR="${ADAPTER_DIR:-./output/emotic_ddp_cls_internal_transfer_screen}"

for SEED in 0 1 2; do
  RUN_NAME="emotic_ddp_cls_internal_gate_seed${SEED}"
  OUTPUT_DIR="./output/${RUN_NAME}"
  mkdir -p "${OUTPUT_DIR}"
  COMPLETE="$(python - "${OUTPUT_DIR}" <<'PY'
import os, sys, torch
complete = True
for task in range(8):
    path = os.path.join(sys.argv[1], f'task{task}_scores.pt')
    if not os.path.isfile(path):
        complete = False
        break
    payload = torch.load(path, map_location='cpu')
    if 'val_class_gate_scores' not in payload:
        complete = False
        break
print('1' if complete else '0')
PY
)"
  if [[ -s "${OUTPUT_DIR}/evaluation_summary.json" && "${COMPLETE}" == "1" ]]; then
    echo "Skip completed ${RUN_NAME}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_cls_internal_gate.py \
    --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
    --adapter_checkpoint "${ADAPTER_DIR}/transferred_adapter_seed${SEED}.pth" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --name "${RUN_NAME}" \
    --alpha_candidates 0 0.001 0.003 0.01 0.03 \
    --task_val_margin 0.1 \
    --class_val_margin 0.1 \
    --adapter_batch_size 256 \
    --workers 4 \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
done

python summarize_emotic_ddp_internal_adapter.py \
  --run_prefix emotic_ddp_cls_internal_gate_seed \
  --output_dir ./output/emotic_ddp_cls_internal_gate_summary
