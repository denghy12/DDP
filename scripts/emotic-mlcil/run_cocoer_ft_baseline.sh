#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SEED="${SEED:-0}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
RESNET_INIT="${COCOER_RESNET50_INIT:-${ROOT}/pretrained/cocoer/resnet50_imagenet1k_v1.pth}"
CLIP_RN50="${COCOER_CLIP_RN50:-${ROOT}/pretrained/clip/RN50.pt}"
HEAD_CACHE="${COCOER_HEAD_CACHE:-${ROOT}/pretrained/cocoer/emotic_head_boxes_v0.1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/cocoer_ft_track_b_v0.1/${RUN_ID}}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
REPORTING_SPLIT="${REPORTING_SPLIT:-val}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
WORKERS="${WORKERS:-2}"
EXPORT_SYNC_RESULTS="${EXPORT_SYNC_RESULTS:-1}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:-}"
export OMP_NUM_THREADS="${COCOER_OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${COCOER_MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${COCOER_OPENBLAS_NUM_THREADS:-4}"

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "Invalid SEED" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
[[ "${TRAIN_BATCH_SIZE}" == "64" ]] || {
  echo "Registered source CocoER train batch size is 64" >&2
  exit 2
}
for asset in "${RESNET_INIT}" "${CLIP_RN50}" "${HEAD_CACHE}"; do
  [[ -s "${asset}" ]] || { echo "Missing CocoER asset: ${asset}" >&2; exit 2; }
done
[[ "${REPORTING_SPLIT}" == "val" || "${REPORTING_SPLIT}" == "test" ]] || {
  echo "REPORTING_SPLIT must be val or test" >&2; exit 2;
}

runner_args=(
  --protocol "${PROTOCOL}"
  --method cocoer_ft
  --seed "${SEED}"
  --data-root "${DATA_ROOT}"
  --cocoer-resnet50-init "${RESNET_INIT}"
  --cocoer-clip-rn50 "${CLIP_RN50}"
  --cocoer-head-cache "${HEAD_CACHE}"
  --input-mode cocoer_multilevel
  --output-root "${OUTPUT_ROOT}"
  --reporting-split "${REPORTING_SPLIT}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --workers "${WORKERS}"
  --device cuda
)
if [[ "${REPORTING_SPLIT}" == "test" ]]; then
  [[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "COCOER_FT_TRACK_B_V0_1" ]] || {
    echo "Held-out test requires a frozen CocoER-FT validation configuration" >&2
    exit 2
  }
  runner_args+=(--configuration-locked)
fi
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m benchmarks.emotic_mlcil.runner "${runner_args[@]}"

if [[ "${EXPORT_SYNC_RESULTS}" == "1" ]]; then
  "${PYTHON}" -m benchmarks.emotic_mlcil.runner \
    --protocol "${PROTOCOL}" --method cocoer_ft --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}" --reporting-split "${REPORTING_SPLIT}" \
    --export-sync-results --shard-run-id "${RUN_ID}"
fi
