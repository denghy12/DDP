#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-ddp_task_bank_b10c4_b4c2}"
RUN_ID="${RUN_ID:-task_bank_b10c4_b4c2_seed012_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/task_adapter_bank_multiprotocol_v0.1/${RUN_ID}}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5}"
DATA_WORKERS="${DATA_WORKERS:-0}"
read -r -a GPUS <<< "${GPU_LIST}"
if (( ${#GPUS[@]} < 6 )); then
  echo "GPU_LIST must provide six entries, e.g. '0 1 2 3 4 5'" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Formal test evaluation requires a clean Git worktree" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

STATE_DIR="${RUN_ROOT}/extras/runtime_state"
LOG_DIR="${RUN_ROOT}/extras/job_logs"
mkdir -p "${STATE_DIR}" "${LOG_DIR}"
printf '%s\n' "${RUN_ID}" > "${RUN_ROOT}/RUN_ID.txt"

launch_job() {
  local index="$1" protocol="$2" seed="$3" gpu="$4"
  local name="${protocol}_s${seed}_g${gpu}"
  local command="cd '${ROOT}' && set -o pipefail; PROTOCOL_KEY='${protocol}' SEED='${seed}' GPU='${gpu}' DATA_WORKERS='${DATA_WORKERS}' RUN_ID='${RUN_ID}' RUN_ROOT='${RUN_ROOT}' bash scripts/emotic-mlcil/run_task_adapter_bank_multiprotocol_job.sh 2>&1 | tee '${LOG_DIR}/${name}.log'; code=\${PIPESTATUS[0]}; if (( code == 0 )); then touch '${STATE_DIR}/${name}.done'; else echo \${code} > '${STATE_DIR}/${name}.failed'; fi; echo JOB_EXIT_CODE=\${code}; exec bash"
  if (( index == 0 )); then
    tmux new-session -d -s "${SESSION}" -n "${name}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${name}" "${command}"
  fi
}

index=0
for protocol in b10c4 b4c2; do
  for seed in 0 1 2; do
    launch_job "${index}" "${protocol}" "${seed}" "${GPUS[$index]}"
    index=$((index + 1))
  done
done

tmux new-window -t "${SESSION}" -n summarize \
  "cd '${ROOT}' && RUN_ID='${RUN_ID}' RUN_ROOT='${RUN_ROOT}' bash scripts/emotic-mlcil/wait_and_summarize_task_adapter_bank_multiprotocol.sh 2>&1 | tee '${LOG_DIR}/summarize.log'; code=\${PIPESTATUS[0]}; echo SUMMARY_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION}: ${RUN_ID}"
echo "Run root: ${RUN_ROOT}"
echo "B10-C4 seeds 0/1/2: GPU${GPUS[0]}/GPU${GPUS[1]}/GPU${GPUS[2]}"
echo "B4-C2  seeds 0/1/2: GPU${GPUS[3]}/GPU${GPUS[4]}/GPU${GPUS[5]}"
echo "DataLoader workers per GPU process: ${DATA_WORKERS}"
echo "Attach: tmux attach -t ${SESSION}"
