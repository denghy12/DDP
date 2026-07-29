#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
MODES="${MODES:-full}"
LOSSES="${LOSSES:-weighted_bce asl bal_paper}"
SUMMARY_LOSSES="${SUMMARY_LOSSES:-${LOSSES}}"
SEEDS="${SEEDS:-0 1 2}"
SESSION="${SESSION:-ddp_task_bank_losses}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-./output/emotic_ddp_task_adapter_bank_loss_comparison}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_task_adapter_bank_loss_pipeline}"
mkdir -p "${STATE_DIR}"

read -r -a GPUS <<< "${GPU_LIST}"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_LIST is empty" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

for index in "${!GPUS[@]}"; do
  : > "${STATE_DIR}/lane${index}.tsv"
done

task_index=0
for mode in ${MODES}; do
  for loss_name in ${LOSSES}; do
    for seed in ${SEEDS}; do
      key="${loss_name}_${mode}_seed${seed}"
      rm -f "${STATE_DIR}/${key}.done" "${STATE_DIR}/${key}.failed"
      lane=$((task_index % ${#GPUS[@]}))
      printf '%s %s %s\n' "${mode}" "${loss_name}" "${seed}" \
        >> "${STATE_DIR}/lane${lane}.tsv"
      task_index=$((task_index + 1))
    done
  done
done
rm -f "${STATE_DIR}/complete.done"

for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  lane_file="${STATE_DIR}/lane${index}.tsv"
  window="gpu${gpu}_lane${index}"
  command="cd '${ROOT}' && GPU='${gpu}' LANE_FILE='${lane_file}' STATE_DIR='${STATE_DIR}' bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_lane.sh 2>&1 | tee '${STATE_DIR}/${window}.log'; code=\${PIPESTATUS[0]}; echo LANE_EXIT_CODE=\${code}; exec bash"
  if (( index == 0 )); then
    tmux new-session -d -s "${SESSION}" -n "${window}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${window}" "${command}"
  fi
done

tmux new-window -t "${SESSION}" -n summary \
  "cd '${ROOT}' && MODES='${MODES}' WAIT_LOSSES='${LOSSES}' LOSSES='${SUMMARY_LOSSES}' SEEDS='${SEEDS}' STATE_DIR='${STATE_DIR}' SUMMARY_OUTPUT_DIR='${SUMMARY_OUTPUT_DIR}' bash scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_task_adapter_bank_losses.sh 2>&1 | tee '${STATE_DIR}/orchestrator.log'; code=\${PIPESTATUS[0]}; echo SUMMARY_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION} with ${task_index} runs on ${#GPUS[@]} GPU lanes"
for index in "${!GPUS[@]}"; do
  echo "  GPU${GPUS[$index]}: $(tr '\n' ';' < "${STATE_DIR}/lane${index}.tsv")"
done
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs:   ${STATE_DIR}/"
