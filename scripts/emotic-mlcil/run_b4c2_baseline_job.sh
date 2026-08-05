#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

METHOD="${METHOD:?METHOD is required}"
SEED="${SEED:?SEED is required}"
PHYSICAL_GPU="${PHYSICAL_GPU:?PHYSICAL_GPU is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b4c2.yaml}"
REPLAY_CONTRACT="${REPLAY_CONTRACT:-${ROOT}/configs/emotic_mlcil/replay_b4c2_20c_v0.1.yaml}"
DERPP_REPLAY_CONTRACT="${DERPP_REPLAY_CONTRACT:-${ROOT}/configs/emotic_mlcil/replay_derpp_b4c2_20c_v0.1.yaml}"
AGCN_WORD_EMBEDDINGS="${AGCN_WORD_EMBEDDINGS:-${ROOT}/pretrained/agcn/emotic_glove_6b_300d.json}"

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "Invalid seed: ${SEED}" >&2; exit 2; }
[[ "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]] || {
  echo "Invalid physical GPU: ${PHYSICAL_GPU}" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "B4-C2 job is not running the frozen sweep commit" >&2
  exit 2
}

case "${METHOD}" in
  finetune)
    DISPLAY_METHOD="Sequential Fine-Tuning"
    TRAIN_BATCH_SIZE=32
    EVAL_BATCH_SIZE=64
    WORKERS=2
    ;;
  lwf)
    DISPLAY_METHOD="LwF"
    TRAIN_BATCH_SIZE=32
    EVAL_BATCH_SIZE=64
    WORKERS=2
    ;;
  ewc)
    DISPLAY_METHOD="EWC"
    TRAIN_BATCH_SIZE=32
    EVAL_BATCH_SIZE=64
    WORKERS=2
    ;;
  agcn)
    DISPLAY_METHOD="AGCN"
    TRAIN_BATCH_SIZE=8
    EVAL_BATCH_SIZE=32
    WORKERS=0
    ;;
  csc)
    DISPLAY_METHOD="CSC"
    TRAIN_BATCH_SIZE=64
    EVAL_BATCH_SIZE=64
    WORKERS=2
    ;;
  multi_lane)
    DISPLAY_METHOD="MULTI-LANE"
    TRAIN_BATCH_SIZE=64
    EVAL_BATCH_SIZE=64
    WORKERS=2
    ;;
  l3a)
    DISPLAY_METHOD="L3A"
    TRAIN_BATCH_SIZE=64
    EVAL_BATCH_SIZE=64
    WORKERS=2
    ;;
  original_ddp)
    DISPLAY_METHOD="Original-DDP-Tau2"
    TRAIN_BATCH_SIZE=8
    EVAL_BATCH_SIZE=1
    WORKERS=0
    ;;
  er)
    DISPLAY_METHOD="ER"
    TRAIN_BATCH_SIZE=32
    EVAL_BATCH_SIZE=64
    WORKERS=2
    ;;
  prs)
    DISPLAY_METHOD="PRS"
    TRAIN_BATCH_SIZE=32
    EVAL_BATCH_SIZE=64
    WORKERS=2
    ;;
  derpp)
    DISPLAY_METHOD="DER++"
    TRAIN_BATCH_SIZE=32
    EVAL_BATCH_SIZE=64
    WORKERS=2
    ;;
  krt)
    DISPLAY_METHOD="KRT"
    TRAIN_BATCH_SIZE=32
    EVAL_BATCH_SIZE=64
    WORKERS=2
    ;;
  *)
    echo "Unsupported B4-C2 baseline: ${METHOD}" >&2
    exit 2
    ;;
esac

RESULT_ROOT="${RUN_OUTPUT_ROOT}/benchmarks/emotic_b4c2_v0.1/A/${DISPLAY_METHOD}/seed${SEED}"
if [[ -e "${RESULT_ROOT}" ]]; then
  echo "Refusing to overwrite an existing result root: ${RESULT_ROOT}" >&2
  exit 2
fi

runner_args=(
  --protocol "${PROTOCOL}"
  --method "${METHOD}"
  --seed "${SEED}"
  --data-root "${DATA_ROOT}"
  --clip-model-path "${CLIP_MODEL_PATH}"
  --agcn-word-embeddings "${AGCN_WORD_EMBEDDINGS}"
  --output-root "${RUN_OUTPUT_ROOT}"
  --reporting-split test
  --configuration-locked
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --workers "${WORKERS}"
  --device cuda
)

case "${METHOD}" in
  ewc)
    runner_args+=(--ewc-lambda 1000000)
    ;;
  er|prs)
    runner_args+=(--replay-contract "${REPLAY_CONTRACT}")
    ;;
  derpp)
    runner_args+=(--replay-contract "${DERPP_REPLAY_CONTRACT}")
    ;;
esac

echo "Starting ${DISPLAY_METHOD} seed ${SEED} on physical GPU ${PHYSICAL_GPU}"
echo "Protocol: ${PROTOCOL}"
echo "Batches: train=${TRAIN_BATCH_SIZE} eval=${EVAL_BATCH_SIZE}; workers=${WORKERS}"

CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
PYTHONUNBUFFERED=1 \
  "${PYTHON}" -m benchmarks.emotic_mlcil.runner "${runner_args[@]}"

export_args=(
  --protocol "${PROTOCOL}"
  --method "${METHOD}"
  --seed "${SEED}"
  --output-root "${RUN_OUTPUT_ROOT}"
  --reporting-split test
  --export-sync-results
  --shard-run-id "${RUN_ID}"
)
case "${METHOD}" in
  er|prs)
    export_args+=(--replay-contract "${REPLAY_CONTRACT}")
    ;;
  derpp)
    export_args+=(--replay-contract "${DERPP_REPLAY_CONTRACT}")
    ;;
esac
"${PYTHON}" -m benchmarks.emotic_mlcil.runner "${export_args[@]}"

echo "Completed ${DISPLAY_METHOD} seed ${SEED} on physical GPU ${PHYSICAL_GPU}"
