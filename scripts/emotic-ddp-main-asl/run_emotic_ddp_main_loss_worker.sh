#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
LOSS_NAME="${LOSS_NAME:?LOSS_NAME is required}"
SEED="${SEED:?SEED is required}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_main_loss_pipeline}"
KEY="${LOSS_NAME}_seed${SEED}"
RUN_NAME="emotic_b5c3_ddp_main_${LOSS_NAME}_seed${SEED}"
TRAIN_DIR="./output/${RUN_NAME}"

mkdir -p "${STATE_DIR}"
rm -f "${STATE_DIR}/${KEY}.done" "${STATE_DIR}/${KEY}.failed"
trap 'touch "${STATE_DIR}/${KEY}.failed"' ERR

if [[ -s "${TRAIN_DIR}/checkpoints/task7.pth" ]]; then
  python - "${TRAIN_DIR}/checkpoints/task7.pth" "${LOSS_NAME}" "${SEED}" <<'PY'
import sys
import torch

checkpoint_path, expected_loss, expected_seed = sys.argv[1:]
checkpoint = torch.load(checkpoint_path, map_location="cpu")
saved = checkpoint.get("args", {})
actual_loss = saved.get("ddp_classification_loss", "two_way_bce")
actual_seed = int(saved.get("seed", -1))
if actual_loss != expected_loss or actual_seed != int(expected_seed):
    raise RuntimeError(
        f"Existing checkpoint protocol mismatch: loss={actual_loss}, "
        f"seed={actual_seed}"
    )
print(f"Reusing completed training: {checkpoint_path}")
PY
else
  if find "${TRAIN_DIR}/checkpoints" -type f -name 'task*.pth' \
      -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to overwrite incomplete run: ${TRAIN_DIR}" >&2
    touch "${STATE_DIR}/${KEY}.failed"
    exit 1
  fi
  GPU="${GPU}" LOSS_NAME="${LOSS_NAME}" SEED="${SEED}" \
    bash scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss.sh
fi

GPU="${GPU}" LOSS_NAME="${LOSS_NAME}" SEED="${SEED}" \
  bash scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss_eval.sh

touch "${STATE_DIR}/${KEY}.done"
rm -f "${STATE_DIR}/${KEY}.failed"
trap - ERR
echo "WORKER_COMPLETE ${KEY}"
