#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/haoyuan/workspace/CODE_DDP-benchmark-cocoer-ft}"
COCOER_ENV="${COCOER_ENV:-/opt/conda/envs/cocoer-preprocess}"
COCOER_PY="${COCOER_PY:-${COCOER_ENV}/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
INSIGHTFACE_ROOT="${INSIGHTFACE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/cocoer_insightface}"
OUTPUT="${OUTPUT:-/mnt/haoyuan/workspace/baseline_sources/cocoer_head_detections.json}"
GPU="${GPU:-0}"

if [[ ! -x "${COCOER_PY}" ]]; then
  echo "Missing CocoER preprocessing Python: ${COCOER_PY}" >&2
  exit 2
fi
if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "Missing CocoER project worktree: ${PROJECT_DIR}" >&2
  exit 2
fi
if [[ ! -d "${INSIGHTFACE_ROOT}/models/buffalo_l" ]]; then
  echo "Missing fixed buffalo_l model tree: ${INSIGHTFACE_ROOT}/models/buffalo_l" >&2
  exit 2
fi

NVIDIA_LIB_DIRS="$(
  find "${COCOER_ENV}/lib/python3.9/site-packages/nvidia" \
    -type d -name lib -print 2>/dev/null \
  | sort -u \
  | paste -sd: -
)"
TORCH_LIB="$(
  "${COCOER_PY}" -c \
    'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'
)"

export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${TORCH_LIB}:${COCOER_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

cd "${PROJECT_DIR}"
exec "${COCOER_PY}" \
  scripts/emotic-mlcil/generate_cocoer_head_detections.py \
  --data-root "${DATA_ROOT}" \
  --insightface-root "${INSIGHTFACE_ROOT}" \
  --device cuda \
  --output "${OUTPUT}" \
  "$@"
