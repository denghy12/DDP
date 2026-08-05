#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

while true; do
  pending=0
  for loss in two_way_bce asl; do
    path="./output/emotic_ddp_main_healthcheck_${loss}_task0_seed0_v2/checkpoints/task0.pth"
    [[ -s "${path}" ]] || pending=$((pending + 1))
  done
  if (( pending == 0 )); then
    break
  fi
  echo "Waiting for ${pending} Task-0 health-check runs..."
  sleep 20
done

python summarize_emotic_ddp_main_loss_healthcheck.py
