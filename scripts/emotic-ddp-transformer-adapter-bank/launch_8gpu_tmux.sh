#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
MODES="${MODES:-full 16shot}"
SEEDS="${SEEDS:-0 1 2}"
SESSION="${SESSION:-ddp_transformer_adapter_bank}"
STATE_DIR="${STATE_DIR:-./output/emotic_ddp_transformer_adapter_bank_pipeline}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-./output/emotic_ddp_transformer_adapter_bank_comparison}"
mkdir -p "${STATE_DIR}"
read -r -a GPUS <<< "${GPU_LIST}"

jobs=()
for mode in ${MODES}; do
  for seed in ${SEEDS}; do jobs+=("${mode}:${seed}"); done
done
if (( ${#GPUS[@]} < ${#jobs[@]} )); then
  echo "Need at least ${#jobs[@]} GPUs for one-run-per-GPU parallelism" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
rm -f "${STATE_DIR}/complete.done"

echo "Running zero-identity and backward integration smoke test on GPU${GPUS[0]}..."
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python smoke_emotic_ddp_transformer_adapter.py

for index in "${!jobs[@]}"; do
  IFS=: read -r mode seed <<< "${jobs[$index]}"
  gpu="${GPUS[$index]}"
  key="${mode}_seed${seed}"
  rm -f "${STATE_DIR}/${key}.done" "${STATE_DIR}/${key}.failed"
  window="${mode}_s${seed}_g${gpu}"
  command="cd '${ROOT}' && GPU='${gpu}' TRAINING_MODE='${mode}' SEED='${seed}' STATE_DIR='${STATE_DIR}' bash scripts/emotic-ddp-transformer-adapter-bank/run_worker.sh 2>&1 | tee '${STATE_DIR}/${window}.log'; code=\${PIPESTATUS[0]}; echo WORKER_EXIT_CODE=\${code}; exec bash"
  if (( index == 0 )); then
    tmux new-session -d -s "${SESSION}" -n "${window}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${window}" "${command}"
  fi
done

tmux new-window -t "${SESSION}" -n summary \
  "cd '${ROOT}' && MODES='${MODES}' SEEDS='${SEEDS}' STATE_DIR='${STATE_DIR}' SUMMARY_OUTPUT_DIR='${SUMMARY_OUTPUT_DIR}' bash scripts/emotic-ddp-transformer-adapter-bank/wait_and_summarize.sh 2>&1 | tee '${STATE_DIR}/orchestrator.log'; code=\${PIPESTATUS[0]}; echo SUMMARY_EXIT_CODE=\${code}; exec bash"

echo "Started ${SESSION}: ${#jobs[@]} runs on ${#jobs[@]} GPUs"
for index in "${!jobs[@]}"; do echo "  GPU${GPUS[$index]}: ${jobs[$index]}"; done
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs:   ${STATE_DIR}/"
