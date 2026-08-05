#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5}"
LOSSES="${LOSSES:-two_way_bce asl}"
SEEDS="${SEEDS:-0 1 2}"
SESSION="${SESSION:-ddp_main_bce_asl}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_main_loss_pipeline}"
SUMMARY_DIR="${SUMMARY_DIR:-./output/emotic_ddp_main_loss_comparison}"
mkdir -p "${STATE_DIR}"
read -r -a GPUS <<< "${GPU_LIST}"

run_count=0
for loss in ${LOSSES}; do
  for seed in ${SEEDS}; do
    run_count=$((run_count + 1))
  done
done
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_LIST must contain at least one lane" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

rm -f "${STATE_DIR}/complete.done"
for lane in "${!GPUS[@]}"; do
  : > "${STATE_DIR}/lane${lane}.tsv"
done

task_index=0
for loss in ${LOSSES}; do
  for seed in ${SEEDS}; do
    key="${loss}_seed${seed}"
    rm -f "${STATE_DIR}/${key}.done" "${STATE_DIR}/${key}.failed"
    lane=$((task_index % ${#GPUS[@]}))
    printf '%s %s\n' "${loss}" "${seed}" >> "${STATE_DIR}/lane${lane}.tsv"
    task_index=$((task_index + 1))
  done
done


for lane in "${!GPUS[@]}"; do
  gpu="${GPUS[$lane]}"
  lane_file="${STATE_DIR}/lane${lane}.tsv"
  window="gpu${gpu}_lane${lane}"
  command="cd '${ROOT}' && GPU='${gpu}' LANE_FILE='${lane_file}' STATE_DIR='${STATE_DIR}' bash scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss_lane.sh 2>&1 | tee '${STATE_DIR}/${window}.log'; code=\${PIPESTATUS[0]}; echo LANE_EXIT_CODE=\${code}; exec bash"
    if (( lane == 0 )); then
      tmux new-session -d -s "${SESSION}" -n "${window}" "${command}"
    else
      tmux new-window -t "${SESSION}" -n "${window}" "${command}"
    fi
done

tmux new-window -t "${SESSION}" -n summary \
  "cd '${ROOT}' && LOSSES='${LOSSES}' SEEDS='${SEEDS}' STATE_DIR='${STATE_DIR}' SUMMARY_DIR='${SUMMARY_DIR}' bash scripts/emotic-ddp-main-asl/wait_and_summarize_emotic_ddp_main_losses.sh 2>&1 | tee '${STATE_DIR}/orchestrator.log'; code=\${PIPESTATUS[0]}; echo SUMMARY_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION}: ${run_count} runs over ${#GPUS[@]} lanes"
for lane in "${!GPUS[@]}"; do
  echo "  lane${lane} GPU${GPUS[$lane]}: $(tr '\n' ';' < "${STATE_DIR}/lane${lane}.tsv")"
done
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs: ${STATE_DIR}/"
