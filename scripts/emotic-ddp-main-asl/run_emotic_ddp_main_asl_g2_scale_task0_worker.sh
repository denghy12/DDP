#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
SEED="${SEED:?SEED is required}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_main_asl_g2_scale_task0_pipeline}"
KEY="asl_g2_lw009_seed${SEED}"
RUN_NAME="emotic_ddp_main_task0_asl_g2_lw009_seed${SEED}_v1"
OUTPUT_DIR="./output/${RUN_NAME}"
CHECKPOINT="${OUTPUT_DIR}/checkpoints/task0.pth"

mkdir -p "${STATE_DIR}"
rm -f "${STATE_DIR}/${KEY}.done" "${STATE_DIR}/${KEY}.failed"
trap 'touch "${STATE_DIR}/${KEY}.failed"' ERR

if [[ ! -s "${CHECKPOINT}" ]]; then
  if find "${OUTPUT_DIR}/checkpoints" -type f -name 'task*.pth' \
      -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to overwrite incomplete checkpoint set: ${OUTPUT_DIR}" >&2
    exit 1
  fi
  GPU="${GPU}" LOSS_NAME=asl_g2_lw009 SEED="${SEED}" \
    MAX_TASKS=1 EPOCHS=30 RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
    bash scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss.sh
fi

python - "${CHECKPOINT}" "${SEED}" <<'PY'
import sys
import torch

path, expected_seed = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu")
args = checkpoint.get("args", {})
checks = {
    "objective": args.get("ddp_classification_loss") == "asl",
    "seed": int(args.get("seed", -1)) == int(expected_seed),
    "epochs": int(args.get("epochs", -1)) == 30,
    "gamma_neg": float(args.get("ddp_asl_gamma_neg", -1)) == 2.0,
    "gamma_pos": float(args.get("ddp_asl_gamma_pos", -1)) == 0.0,
    "clip": float(args.get("ddp_asl_clip", -1)) == 0.05,
    "loss_w": float(args.get("loss_w", -1)) == 0.09,
    "threshold": float(args.get("thre", -1)) == 0.5,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError(f"Checkpoint protocol mismatch {failed}: {path}")
print(f"Scale-matched checkpoint audit passed: {path}")
PY

[[ -s "${OUTPUT_DIR}/training_diagnostics.json" ]]
[[ -s "${OUTPUT_DIR}/detail/${RUN_NAME}_per_class_task_table.json" ]]
touch "${STATE_DIR}/${KEY}.done"
rm -f "${STATE_DIR}/${KEY}.failed"
trap - ERR
echo "SCALE_MATCHED_TASK0_COMPLETE seed${SEED}"
