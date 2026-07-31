#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

TASK_ID="${TASK_ID:?Set TASK_ID to 0..7}"
GPU="${GPU:?Set GPU to one physical GPU index}"
SHARD_RUN_ID="${SHARD_RUN_ID:?Set SHARD_RUN_ID}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/output}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WORKERS="${WORKERS:-0}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
STATE_DIR="${OUTPUT_ROOT}/benchmarks/emotic_b5c3_v0.1/A/DDP/seed0/shards/${SHARD_RUN_ID}/_state"

[[ "${TASK_ID}" =~ ^[0-7]$ ]] || {
  echo "TASK_ID must be one integer in 0..7" >&2
  exit 2
}
[[ "${SHARD_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid SHARD_RUN_ID: ${SHARD_RUN_ID}" >&2
  exit 2
}
[[ "${GPU}" =~ ^[0-9]+$ ]] || {
  echo "GPU must be a non-negative physical GPU index" >&2
  exit 2
}
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing data root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${CHECKPOINT_DIR}/task${TASK_ID}.pth" ]] || {
  echo "Missing checkpoint: ${CHECKPOINT_DIR}/task${TASK_ID}.pth" >&2
  exit 2
}
[[ -s "${CLIP_MODEL_PATH}" ]] || {
  echo "Missing CLIP model: ${CLIP_MODEL_PATH}" >&2
  exit 2
}
mkdir -p "${STATE_DIR}"
rm -f "${STATE_DIR}/task${TASK_ID}.done" "${STATE_DIR}/task${TASK_ID}.failed"
ulimit -n 65535 2>/dev/null || true

set +e
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${PYTHON}" \
  -m benchmarks.emotic_mlcil.runner \
  --protocol "${PROTOCOL}" \
  --method ddp \
  --data-root "${DATA_ROOT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --clip-model-path "${CLIP_MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --reporting-split test \
  --configuration-locked \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --device cuda:0 \
  --task-id "${TASK_ID}" \
  --shard-run-id "${SHARD_RUN_ID}" \
  2>&1 | tee "${STATE_DIR}/task${TASK_ID}.log"
code="${PIPESTATUS[0]}"
set -e

if [[ "${code}" -eq 0 ]]; then
  touch "${STATE_DIR}/task${TASK_ID}.done"
else
  printf '%s\n' "${code}" > "${STATE_DIR}/task${TASK_ID}.failed"
fi
exit "${code}"
