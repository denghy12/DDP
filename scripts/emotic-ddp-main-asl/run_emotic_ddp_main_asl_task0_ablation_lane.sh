#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
LANE_FILE="${LANE_FILE:?LANE_FILE is required}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_main_asl_task0_ablation_pipeline}"

while read -r method seed; do
  [[ -n "${method}" ]] || continue
  echo "[Lane GPU${GPU}] starting ${method} seed${seed}"
  GPU="${GPU}" METHOD="${method}" SEED="${seed}" STATE_DIR="${STATE_DIR}" \
    bash scripts/emotic-ddp-main-asl/run_emotic_ddp_main_asl_task0_ablation_worker.sh
done < "${LANE_FILE}"

echo "LANE_COMPLETE GPU${GPU} ${LANE_FILE}"
