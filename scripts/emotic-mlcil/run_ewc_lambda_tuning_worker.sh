#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
LAMBDAS="${LAMBDAS:?LAMBDAS is required}"
TUNING_ROOT="${TUNING_ROOT:?TUNING_ROOT is required}"
STATE_DIR="${STATE_DIR:?STATE_DIR is required}"
TUNING_SEED="${TUNING_SEED:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"

slug() {
  "${PYTHON}" - "$1" <<'PY'
import sys

value = float(sys.argv[1])
print(format(value, ".12g").replace("+", "p").replace("-", "m").replace(".", "d"))
PY
}

worker_failed=0
read -r -a candidates <<< "${LAMBDAS}"
for candidate in "${candidates[@]}"; do
  candidate_slug="$(slug "${candidate}")"
  key="ewc_lambda_${candidate_slug}"
  log="${STATE_DIR}/${key}.log"
  candidate_root="${TUNING_ROOT}/lambda_${candidate_slug}"
  echo "Starting ${key} validation tuning on physical GPU ${GPU}"
  set +e
  METHOD=ewc \
    SEED="${TUNING_SEED}" \
    GPU="${GPU}" \
    RUN_ID="${RUN_ID}" \
    STATE_DIR="${STATE_DIR}" \
    STATE_KEY="${key}" \
    EWC_LAMBDA="${candidate}" \
    OUTPUT_ROOT="${candidate_root}" \
    REPORTING_SPLIT=val \
    EXPORT_SYNC_RESULTS=0 \
    PYTHON="${PYTHON}" \
    DATA_ROOT="${DATA_ROOT}" \
    CLIP_MODEL_PATH="${CLIP_MODEL_PATH}" \
    PROTOCOL="${PROTOCOL}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    WORKERS="${WORKERS}" \
    bash "${SCRIPT_DIR}/run_clip_continual_baseline.sh" \
    2>&1 | tee "${log}"
  code=${PIPESTATUS[0]}
  set -e
  if [[ "${code}" -ne 0 ]]; then
    worker_failed=1
    echo "${key} failed with exit code ${code}" >&2
  fi
done

exit "${worker_failed}"
