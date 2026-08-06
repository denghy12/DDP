#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

PROTOCOL_KEY="${PROTOCOL_KEY:?Set PROTOCOL_KEY to b10c4 or b4c2}"
SEED="${SEED:?Set SEED to 0, 1, or 2}"
GPU="${GPU:?Set GPU to the physical GPU index}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"

case "${PROTOCOL_KEY}" in
  b10c4)
    PROTOCOL="${ROOT}/configs/emotic_mlcil/protocol_b10c4.yaml"
    PROTOCOL_ID="emotic_b10c4_v0.1"
    TASK_COUNT=5
    BASELINE_RUN_ROOT="${B10C4_BASELINE_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/b10c4_12baseline_v0.1/b10c4_12baseline_2slot_seed012_20260805_163640}"
    ;;
  b4c2)
    PROTOCOL="${ROOT}/configs/emotic_mlcil/protocol_b4c2.yaml"
    PROTOCOL_ID="emotic_b4c2_v0.1"
    TASK_COUNT=12
    BASELINE_RUN_ROOT="${B4C2_BASELINE_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/b4c2_12baseline_v0.1/b4c2_12baseline_2slot_seed012_20260805_225429}"
    ;;
  *)
    echo "PROTOCOL_KEY must be b10c4 or b4c2" >&2
    exit 2
    ;;
esac

CHECKPOINT_DIR="${BASELINE_RUN_ROOT}/benchmarks/${PROTOCOL_ID}/A/Original-DDP-Tau2/seed${SEED}/checkpoints"
BASELINE_SCORES="${BASELINE_RUN_ROOT}/benchmarks/${PROTOCOL_ID}/A/Original-DDP-Tau2/seed${SEED}/scores"
for ((task=0; task<TASK_COUNT; task++)); do
  [[ -s "${CHECKPOINT_DIR}/task${task}.pth" ]] || {
    echo "Missing frozen DDP checkpoint: ${CHECKPOINT_DIR}/task${task}.pth" >&2
    exit 1
  }
  [[ -s "${BASELINE_SCORES}/task${task}_scores.pt" ]] || {
    echo "Missing frozen DDP scores: ${BASELINE_SCORES}/task${task}_scores.pt" >&2
    exit 1
  }
done

RESULT_ROOT="${RUN_ROOT}/benchmarks/${PROTOCOL_ID}/A/DDP-Task-Adapter-Bank-Full/seed${SEED}"
if [[ -s "${RESULT_ROOT}/metrics/summary.json" ]]; then
  echo "Skip complete ${PROTOCOL_KEY} seed${SEED}"
  exit 0
fi
if [[ -e "${RESULT_ROOT}" ]]; then
  echo "Refusing incomplete existing result root: ${RESULT_ROOT}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPU}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
PYTHONUNBUFFERED=1 \
  "${PYTHON}" -m benchmarks.emotic_mlcil.runner \
    --protocol "${PROTOCOL}" \
    --method task_adapter_bank \
    --seed "${SEED}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --output-root "${RUN_ROOT}" \
    --reporting-split test \
    --configuration-locked \
    --train-batch-size 128 \
    --eval-batch-size 4 \
    --workers 4 \
    --device cuda

"${PYTHON}" -m benchmarks.emotic_mlcil.runner \
  --protocol "${PROTOCOL}" \
  --method task_adapter_bank \
  --seed "${SEED}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-root "${RUN_ROOT}" \
  --reporting-split test \
  --export-sync-results \
  --shard-run-id "${RUN_ID}"

echo "Completed ${PROTOCOL_KEY} seed${SEED} on GPU${GPU}"
