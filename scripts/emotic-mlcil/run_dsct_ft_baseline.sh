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
SOURCE_ROOT="${DSCT_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/dsct_release_8b0fe36}"
PRETRAINED="${DSCT_PRETRAINED_WEIGHTS:-${SOURCE_ROOT}/r50_deformable_detr-checkpoint.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/dsct_ft_track_b_v0.1/${RUN_ID}}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
REPORTING_SPLIT="${REPORTING_SPLIT:-val}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
WORKERS="${WORKERS:-0}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:-}"

[[ "${TRAIN_BATCH_SIZE}" == "4" ]] || { echo "Official DSCT train batch size is 4" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+$|^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || { echo "DSCT GPU must contain one or four physical IDs" >&2; exit 2; }
IFS=',' read -r -a DSCT_GPU_IDS <<<"${GPU}"
DSCT_UNIQUE_GPU_COUNT="$(printf '%s\n' "${DSCT_GPU_IDS[@]}" | sort -u | wc -l | tr -d '[:space:]')"
[[ "${DSCT_UNIQUE_GPU_COUNT}" -eq "${#DSCT_GPU_IDS[@]}" ]] || { echo "DSCT physical GPU IDs must be unique" >&2; exit 2; }
[[ -d "${SOURCE_ROOT}/models" ]] || { echo "Missing fixed DSCT source: ${SOURCE_ROOT}" >&2; exit 2; }
[[ -s "${PRETRAINED}" ]] || { echo "Missing official DSCT R50 pretraining: ${PRETRAINED}" >&2; exit 2; }
args=(--protocol "${PROTOCOL}" --method dsct_ft --seed "${SEED}" --data-root "${DATA_ROOT}"
  --dsct-source-root "${SOURCE_ROOT}" --dsct-pretrained-weights "${PRETRAINED}"
  --input-mode dsct_scene --output-root "${OUTPUT_ROOT}" --reporting-split "${REPORTING_SPLIT}"
  --train-batch-size "${TRAIN_BATCH_SIZE}" --eval-batch-size "${EVAL_BATCH_SIZE}"
  --workers "${WORKERS}" --device cuda)
if [[ "${REPORTING_SPLIT}" == "test" ]]; then
  [[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "DSCT_FT_TRACK_B_V0_2" ]] || { echo "Held-out test requires frozen DSCT-FT v0.2 configuration" >&2; exit 2; }
  args+=(--configuration-locked)
fi
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m benchmarks.emotic_mlcil.runner "${args[@]}"
"${PYTHON}" -m benchmarks.emotic_mlcil.runner --protocol "${PROTOCOL}" --method dsct_ft --seed "${SEED}" \
  --output-root "${OUTPUT_ROOT}" --reporting-split "${REPORTING_SPLIT}" --export-sync-results --shard-run-id "${RUN_ID}"
