#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
LANE_FILE="${LANE_FILE:?LANE_FILE is required}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_main_asl_g2_scale_task0_pipeline}"

while read -r seed; do
  [[ -n "${seed}" ]] || continue
  echo "[Lane GPU${GPU}] starting ASL-2 loss_w=0.09 seed${seed}"
  GPU="${GPU}" SEED="${seed}" STATE_DIR="${STATE_DIR}" \
    bash scripts/emotic-ddp-main-asl/run_emotic_ddp_main_asl_g2_scale_task0_worker.sh
done < "${LANE_FILE}"

echo "SCALE_LANE_COMPLETE GPU${GPU} ${LANE_FILE}"
