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
SOURCE_ROOT="${BENET_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/benet_release_b86747e}"
PRETRAINED_WEIGHTS="${BENET_PRETRAINED_WEIGHTS:-${SOURCE_ROOT}/models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/benet_ft_track_b_v0.1/${RUN_ID}}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
REPORTING_SPLIT="${REPORTING_SPLIT:-val}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
WORKERS="${WORKERS:-0}"
EXPORT_SYNC_RESULTS="${EXPORT_SYNC_RESULTS:-1}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:-}"

[[ "${TRAIN_BATCH_SIZE}" == "24" ]] || { echo "Registered BENet train batch size is 24" >&2; exit 2; }
[[ -s "${SOURCE_ROOT}/lib/models/BENet.py" ]] || { echo "Missing fixed BENet source: ${SOURCE_ROOT}" >&2; exit 2; }
[[ -s "${PRETRAINED_WEIGHTS}" ]] || { echo "Missing official BENet HigherHRNet weights: ${PRETRAINED_WEIGHTS}" >&2; exit 2; }
if [[ "${REPORTING_SPLIT}" == "test" ]]; then
  [[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "BENET_FT_TRACK_B_V0_1" ]] || {
    echo "Held-out test requires a frozen BENet-FT configuration" >&2
    exit 2
  }
fi

runner_args=(
  --protocol "${PROTOCOL}"
  --method benet_ft
  --seed "${SEED}"
  --data-root "${DATA_ROOT}"
  --benet-source-root "${SOURCE_ROOT}"
  --benet-pretrained-weights "${PRETRAINED_WEIGHTS}"
  --input-mode benet_views
  --output-root "${OUTPUT_ROOT}"
  --reporting-split "${REPORTING_SPLIT}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --workers "${WORKERS}"
  --device cuda
)
[[ "${REPORTING_SPLIT}" == "test" ]] && runner_args+=(--configuration-locked)
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m benchmarks.emotic_mlcil.runner "${runner_args[@]}"

if [[ "${EXPORT_SYNC_RESULTS}" == "1" ]]; then
  "${PYTHON}" -m benchmarks.emotic_mlcil.runner \
    --protocol "${PROTOCOL}" --method benet_ft --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}" --reporting-split "${REPORTING_SPLIT}" \
    --export-sync-results --shard-run-id "${RUN_ID}"
fi
