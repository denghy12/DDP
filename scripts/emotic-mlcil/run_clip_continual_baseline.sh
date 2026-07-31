#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

METHOD="${METHOD:?METHOD is required}"
SEED="${SEED:?SEED is required}"
GPU="${GPU:?GPU is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
STATE_DIR="${STATE_DIR:?STATE_DIR is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/clip_continual_v0.2}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
REPORTING_SPLIT="${REPORTING_SPLIT:-test}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"
EWC_LAMBDA="${EWC_LAMBDA:-}"
EXPORT_SYNC_RESULTS="${EXPORT_SYNC_RESULTS:-1}"
STATE_KEY="${STATE_KEY:-}"

case "${METHOD}" in
  finetune|lwf|ewc) ;;
  *)
    echo "Unsupported CLIP continual baseline: ${METHOD}" >&2
    exit 2
    ;;
esac
[[ "${SEED}" =~ ^[0-9]+$ ]] || {
  echo "SEED must be a non-negative integer" >&2
  exit 2
}
[[ "${GPU}" =~ ^[0-9]+$ ]] || {
  echo "GPU must be a physical GPU index" >&2
  exit 2
}
[[ "${REPORTING_SPLIT}" == "val" || "${REPORTING_SPLIT}" == "test" ]] || {
  echo "REPORTING_SPLIT must be val or test" >&2
  exit 2
}
[[ "${EXPORT_SYNC_RESULTS}" == "0" || "${EXPORT_SYNC_RESULTS}" == "1" ]] || {
  echo "EXPORT_SYNC_RESULTS must be 0 or 1" >&2
  exit 2
}
if [[ -n "${EWC_LAMBDA}" && "${METHOD}" != "ewc" ]]; then
  echo "EWC_LAMBDA is valid only for METHOD=ewc" >&2
  exit 2
fi

mkdir -p "${STATE_DIR}"
key="${STATE_KEY:-${METHOD}_seed${SEED}}"
rm -f "${STATE_DIR}/${key}.done" "${STATE_DIR}/${key}.failed"

runner_args=(
  --protocol "${PROTOCOL}"
  --method "${METHOD}"
  --seed "${SEED}"
  --data-root "${DATA_ROOT}"
  --clip-model-path "${CLIP_MODEL_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --reporting-split "${REPORTING_SPLIT}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --workers "${WORKERS}"
  --device cuda
)
if [[ "${REPORTING_SPLIT}" == "test" ]]; then
  runner_args+=(--configuration-locked)
fi
if [[ -n "${EWC_LAMBDA}" ]]; then
  runner_args+=(--ewc-lambda "${EWC_LAMBDA}")
fi

if CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  -m benchmarks.emotic_mlcil.runner "${runner_args[@]}"; then
  if [[ "${EXPORT_SYNC_RESULTS}" == "0" ]]; then
    touch "${STATE_DIR}/${key}.done"
    echo "Completed ${key} on physical GPU ${GPU}"
    exit 0
  fi
  if "${PYTHON}" -m benchmarks.emotic_mlcil.runner \
    --protocol "${PROTOCOL}" \
    --method "${METHOD}" \
    --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}" \
    --reporting-split "${REPORTING_SPLIT}" \
    --export-sync-results \
    --shard-run-id "${RUN_ID}"; then
    touch "${STATE_DIR}/${key}.done"
    echo "Completed ${key} on physical GPU ${GPU}"
    exit 0
  fi
fi

touch "${STATE_DIR}/${key}.failed"
echo "Failed ${key} on physical GPU ${GPU}" >&2
exit 1
