#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
METHOD="${METHOD:?METHOD is required}"
SEED="${SEED:?SEED is required}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_main_asl_task0_ablation_pipeline}"
KEY="${METHOD}_seed${SEED}"
RUN_NAME="emotic_ddp_main_task0_${METHOD}_seed${SEED}_v1"
OUTPUT_DIR="./output/${RUN_NAME}"

case "${METHOD}" in
  two_way_bce) EXPECTED_OBJECTIVE=two_way_bce; EXPECTED_GAMMA=none ;;
  asl_g9p8) EXPECTED_OBJECTIVE=asl; EXPECTED_GAMMA=9.8 ;;
  asl_g4) EXPECTED_OBJECTIVE=asl; EXPECTED_GAMMA=4.0 ;;
  asl_g2) EXPECTED_OBJECTIVE=asl; EXPECTED_GAMMA=2.0 ;;
  *) echo "Unsupported METHOD=${METHOD}" >&2; exit 2 ;;
esac

mkdir -p "${STATE_DIR}"
rm -f "${STATE_DIR}/${KEY}.done" "${STATE_DIR}/${KEY}.failed"
trap 'touch "${STATE_DIR}/${KEY}.failed"' ERR

legacy_dir=""
if [[ "${SEED}" == 0 && "${METHOD}" == two_way_bce ]]; then
  legacy_dir="./output/emotic_ddp_main_healthcheck_two_way_bce_task0_seed0_v2"
elif [[ "${SEED}" == 0 && "${METHOD}" == asl_g9p8 ]]; then
  legacy_dir="./output/emotic_ddp_main_healthcheck_asl_task0_seed0_v2"
fi

if [[ -n "${legacy_dir}" && -s "${legacy_dir}/checkpoints/task0.pth" ]]; then
  OUTPUT_DIR="${legacy_dir}"
  echo "Reusing corrected seed-0 health check: ${legacy_dir}"
fi

checkpoint="${OUTPUT_DIR}/checkpoints/task0.pth"
if [[ ! -s "${checkpoint}" ]]; then
  if find "${OUTPUT_DIR}/checkpoints" -type f -name 'task*.pth' \
      -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to overwrite incomplete run: ${OUTPUT_DIR}" >&2
    exit 1
  fi
  GPU="${GPU}" LOSS_NAME="${METHOD}" SEED="${SEED}" MAX_TASKS=1 EPOCHS=30 \
    RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
    bash scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss.sh
fi

python - "${checkpoint}" "${EXPECTED_OBJECTIVE}" "${EXPECTED_GAMMA}" \
  "${SEED}" <<'PY'
import sys
import torch

path, expected_objective, expected_gamma, expected_seed = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu")
args = checkpoint.get("args", {})
if args.get("ddp_classification_loss", "two_way_bce") != expected_objective:
    raise RuntimeError(f"Objective mismatch in {path}")
if int(args.get("seed", -1)) != int(expected_seed):
    raise RuntimeError(f"Seed mismatch in {path}")
if int(args.get("epochs", -1)) != 30:
    raise RuntimeError(f"Epoch mismatch in {path}")
if float(args.get("loss_w", -1)) != 0.03:
    raise RuntimeError(f"Loss scale mismatch in {path}")
if float(args.get("thre", -1)) != 0.5:
    raise RuntimeError(f"Threshold mismatch in {path}")
if expected_gamma != "none":
    if float(args.get("ddp_asl_gamma_neg", -1)) != float(expected_gamma):
        raise RuntimeError(f"gamma_neg mismatch in {path}")
    if float(args.get("ddp_asl_gamma_pos", -1)) != 0.0:
        raise RuntimeError(f"gamma_pos mismatch in {path}")
    if float(args.get("ddp_asl_clip", -1)) != 0.05:
        raise RuntimeError(f"ASL clip mismatch in {path}")
print(f"Checkpoint protocol audit passed: {path}")
PY

[[ -s "${OUTPUT_DIR}/training_diagnostics.json" ]]
[[ -s "${OUTPUT_DIR}/detail/${OUTPUT_DIR##*/}_per_class_task_table.json" ]]
touch "${STATE_DIR}/${KEY}.done"
rm -f "${STATE_DIR}/${KEY}.failed"
trap - ERR
echo "TASK0_ABLATION_COMPLETE ${KEY}"
