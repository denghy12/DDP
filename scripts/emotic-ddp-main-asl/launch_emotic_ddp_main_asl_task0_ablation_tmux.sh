#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU_LIST="${GPU_LIST:-0 0}"
METHODS="${METHODS:-two_way_bce asl_g9p8 asl_g4 asl_g2}"
SEEDS="${SEEDS:-0 1 2}"
SESSION="${SESSION:-ddp_main_asl_task0_ablation}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_main_asl_task0_ablation_pipeline}"
SUMMARY_DIR="${SUMMARY_DIR:-./output/emotic_ddp_main_asl_task0_ablation_summary}"
read -r -a GPUS <<< "${GPU_LIST}"

if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_LIST must contain at least one logical lane" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "${STATE_DIR}"
rm -f "${STATE_DIR}/complete.done"
for lane in "${!GPUS[@]}"; do
  : > "${STATE_DIR}/lane${lane}.tsv"
done

task_index=0
run_count=0
for method in ${METHODS}; do
  for seed in ${SEEDS}; do
    key="${method}_seed${seed}"
    rm -f "${STATE_DIR}/${key}.done" "${STATE_DIR}/${key}.failed"
    lane=$((task_index % ${#GPUS[@]}))
    printf '%s %s\n' "${method}" "${seed}" >> "${STATE_DIR}/lane${lane}.tsv"
    task_index=$((task_index + 1))
    run_count=$((run_count + 1))
  done
done

for lane in "${!GPUS[@]}"; do
  gpu="${GPUS[$lane]}"
  lane_file="${STATE_DIR}/lane${lane}.tsv"
  window="gpu${gpu}_lane${lane}"
  command="cd '${ROOT}' && GPU='${gpu}' LANE_FILE='${lane_file}' STATE_DIR='${STATE_DIR}' bash scripts/emotic-ddp-main-asl/run_emotic_ddp_main_asl_task0_ablation_lane.sh 2>&1 | tee '${STATE_DIR}/${window}.log'; code=\${PIPESTATUS[0]}; echo LANE_EXIT_CODE=\${code}; exec bash"
  if (( lane == 0 )); then
    tmux new-session -d -s "${SESSION}" -n "${window}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${window}" "${command}"
  fi
done

tmux new-window -t "${SESSION}" -n summary \
  "cd '${ROOT}' && METHODS='${METHODS}' SEEDS='${SEEDS}' STATE_DIR='${STATE_DIR}' SUMMARY_DIR='${SUMMARY_DIR}' bash scripts/emotic-ddp-main-asl/wait_and_summarize_emotic_ddp_main_asl_task0_ablation.sh 2>&1 | tee '${STATE_DIR}/orchestrator.log'; code=\${PIPESTATUS[0]}; echo SUMMARY_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION}: ${run_count} protocol entries over ${#GPUS[@]} lanes"
echo "Corrected BCE/ASL-9.8 seed0 health checks are reused when present."
for lane in "${!GPUS[@]}"; do
  echo "  lane${lane} GPU${GPUS[$lane]}: $(tr '\n' ';' < "${STATE_DIR}/lane${lane}.tsv")"
done
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs: ${STATE_DIR}/"
