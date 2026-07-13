#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
INTERNAL_SCREEN_DIR="${INTERNAL_SCREEN_DIR:-./output/emotic_ddp_cls_internal_transfer_screen}"
INTERNAL_GATE_PREFIX="${INTERNAL_GATE_PREFIX:-./output/emotic_ddp_cls_internal_gate_seed}"
EXTERNAL_FUSION_PREFIX="${EXTERNAL_FUSION_PREFIX:-./output/emotic_prototype_fusion_strict_base5_16shot_seed}"
EXTERNAL_ADAPTER_PREFIX="${EXTERNAL_ADAPTER_PREFIX:-./output/emotic_prototype_adapter_base5_16shot_seed}"
HYBRID_PREFIX="${HYBRID_PREFIX:-./output/emotic_ddp_cls_external_hybrid_seed}"

# Re-export the frozen internal results with val scores needed by the hybrid.
GPU="${GPU}" bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_internal_gate.sh

for SEED in 0 1 2; do
  OUTPUT_DIR="${HYBRID_PREFIX}${SEED}"
  mkdir -p "${OUTPUT_DIR}"
  if [[ -s "${OUTPUT_DIR}/evaluation_summary.json" ]]; then
    echo "Skip completed hybrid seed ${SEED}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_cls_external_hybrid.py \
    --seed "${SEED}" \
    --ddp_checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
    --internal_dir "${INTERNAL_GATE_PREFIX}${SEED}" \
    --external_dir "${EXTERNAL_FUSION_PREFIX}${SEED}" \
    --data_root ./datasets/EMOTIC \
    --output_dir "${OUTPUT_DIR}" \
    --name "emotic_ddp_cls_external_hybrid_seed${SEED}" \
    --beta_step 0.02 \
    --threshold_min 0.05 \
    --threshold_max 0.95 \
    --threshold_step 0.01 \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
done

python summarize_emotic_ddp_internal_adapter.py \
  --run_prefix emotic_ddp_cls_external_hybrid_seed \
  --output_dir ./output/emotic_ddp_cls_external_hybrid_summary

python summarize_emotic_adapter_ablation.py

BENCHMARK_DIR=./output/emotic_adapter_inference_benchmark
mkdir -p "${BENCHMARK_DIR}"
for METHOD in ddp internal external hybrid; do
  CUDA_VISIBLE_DEVICES="${GPU}" python benchmark_emotic_adapter_inference.py \
    --method "${METHOD}" \
    --ddp_checkpoint "${DDP_CHECKPOINT_DIR}/task7.pth" \
    --internal_adapter "${INTERNAL_SCREEN_DIR}/transferred_adapter_seed1.pth" \
    --internal_gate_summary "${INTERNAL_GATE_PREFIX}1/evaluation_summary.json" \
    --external_adapter "${EXTERNAL_ADAPTER_PREFIX}1/best_adapter.pth" \
    --data_root ./datasets/EMOTIC \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --output "${BENCHMARK_DIR}/${METHOD}.json" \
    --warmup 10 \
    --iterations 50
done
python summarize_emotic_adapter_benchmark.py

echo "Final analysis complete."
echo "Ablation:  output/emotic_adapter_final_ablation/ablation.html"
echo "Benchmark: output/emotic_adapter_inference_benchmark/benchmark.html"
echo "Hybrid:    output/emotic_ddp_cls_external_hybrid_summary/summary.html"
