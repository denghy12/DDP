#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?Set RUN_ID}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
STATE_DIR="${RUN_ROOT}/extras/runtime_state"
B10C4_BASELINE_ROOT="${B10C4_BASELINE_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/b10c4_12baseline_v0.1/b10c4_12baseline_2slot_seed012_20260805_163640}"
B4C2_BASELINE_ROOT="${B4C2_BASELINE_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/b4c2_12baseline_v0.1/b4c2_12baseline_2slot_seed012_20260805_225429}"

jobs=()
for protocol in b10c4 b4c2; do
  for seed in 0 1 2; do
    jobs+=("${protocol}_s${seed}")
  done
done

while true; do
  complete=0
  for prefix in "${jobs[@]}"; do
    if compgen -G "${STATE_DIR}/${prefix}_g*.failed" >/dev/null; then
      echo "A worker failed: ${prefix}" >&2
      exit 1
    fi
    if compgen -G "${STATE_DIR}/${prefix}_g*.done" >/dev/null; then
      complete=$((complete + 1))
    fi
  done
  if (( complete == 6 )); then
    break
  fi
  echo "Waiting for Task Bank jobs: ${complete}/6 complete"
  sleep 20
done

SUMMARY_DIR="${RUN_ROOT}/summary"
"${PYTHON}" summarize_emotic_ddp_task_bank_multiprotocol.py \
  --run-root "${RUN_ROOT}" \
  --b10c4-baseline-root "${B10C4_BASELINE_ROOT}" \
  --b4c2-baseline-root "${B4C2_BASELINE_ROOT}" \
  --output-dir "${SUMMARY_DIR}"
touch "${RUN_ROOT}/complete.done"
echo "Summary JSON: ${SUMMARY_DIR}/multiprotocol_summary.json"
echo "Summary HTML: ${SUMMARY_DIR}/multiprotocol_summary.html"
