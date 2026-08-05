#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
LANE_FILE="${LANE_FILE:?LANE_FILE is required}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_main_loss_pipeline}"

while read -r loss_name seed; do
  [[ -n "${loss_name}" ]] || continue
  echo "[Lane GPU${GPU}] starting ${loss_name} seed${seed}"
  GPU="${GPU}" LOSS_NAME="${loss_name}" SEED="${seed}" \
    STATE_DIR="${STATE_DIR}" \
    bash scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss_worker.sh
done < "${LANE_FILE}"

echo "LANE_COMPLETE GPU${GPU} ${LANE_FILE}"
