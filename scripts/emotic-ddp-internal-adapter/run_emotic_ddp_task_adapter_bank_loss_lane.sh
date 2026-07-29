#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:?Set GPU}"
LANE_FILE="${LANE_FILE:?Set LANE_FILE}"

while read -r mode loss_name seed; do
  [[ -n "${mode}" ]] || continue
  GPU="${GPU}" TRAINING_MODE="${mode}" LOSS_NAME="${loss_name}" SEED="${seed}" \
    STATE_DIR="${STATE_DIR:-}" \
    bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_worker.sh
done < "${LANE_FILE}"
