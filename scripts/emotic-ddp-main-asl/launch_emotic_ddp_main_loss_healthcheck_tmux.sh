#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-ddp_main_loss_health}"
GPU_BCE="${GPU_BCE:-0}"
GPU_ASL="${GPU_ASL:-1}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

for specification in "two_way_bce:${GPU_BCE}" "asl:${GPU_ASL}"; do
  IFS=: read -r loss gpu <<< "${specification}"
  run_name="emotic_ddp_main_healthcheck_${loss}_task0_seed0_v2"
  command="cd '${ROOT}' && GPU='${gpu}' LOSS_NAME='${loss}' SEED=0 MAX_TASKS=1 EPOCHS=30 RUN_NAME='${run_name}' OUTPUT_DIR='./output/${run_name}' bash scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss.sh; code=\$?; echo HEALTH_EXIT_CODE=\${code}; exec bash"
  if [[ "${loss}" == "two_way_bce" ]]; then
    tmux new-session -d -s "${SESSION}" -n bce "${command}"
  else
    tmux new-window -t "${SESSION}" -n asl "${command}"
  fi
done

tmux new-window -t "${SESSION}" -n summary \
  "cd '${ROOT}' && bash scripts/emotic-ddp-main-asl/wait_and_summarize_emotic_ddp_main_loss_healthcheck.sh; code=\$?; echo SUMMARY_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION}: BCE on GPU${GPU_BCE}, ASL on GPU${GPU_ASL}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Outputs: output/emotic_ddp_main_healthcheck_{two_way_bce,asl}_task0_seed0_v2/"
