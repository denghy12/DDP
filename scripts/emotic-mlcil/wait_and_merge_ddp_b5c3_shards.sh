#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SHARD_RUN_ID="${SHARD_RUN_ID:?Set SHARD_RUN_ID}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/output}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
POLL_SECONDS="${POLL_SECONDS:-15}"
STATE_DIR="${OUTPUT_ROOT}/benchmarks/emotic_b5c3_v0.1/A/DDP/seed0/shards/${SHARD_RUN_ID}/_state"

[[ "${SHARD_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid SHARD_RUN_ID: ${SHARD_RUN_ID}" >&2
  exit 2
}
mkdir -p "${STATE_DIR}"
echo "Waiting for 8 task shards: ${SHARD_RUN_ID}"
while true; do
  failed=()
  completed=0
  for task_id in 0 1 2 3 4 5 6 7; do
    if [[ -f "${STATE_DIR}/task${task_id}.failed" ]]; then
      failed+=("${task_id}")
    fi
    if [[ -f "${STATE_DIR}/task${task_id}.done" ]]; then
      completed=$((completed + 1))
    fi
  done
  if [[ "${#failed[@]}" -gt 0 ]]; then
    echo "Failed task shards: ${failed[*]}" >&2
    echo "Inspect logs under ${STATE_DIR}" >&2
    exit 1
  fi
  echo "Completed ${completed}/8 task shards"
  if [[ "${completed}" -eq 8 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" -m benchmarks.emotic_mlcil.runner \
  --protocol "${PROTOCOL}" \
  --method ddp \
  --output-root "${OUTPUT_ROOT}" \
  --reporting-split test \
  --configuration-locked \
  --merge-shards \
  --shard-run-id "${SHARD_RUN_ID}" \
  2>&1 | tee "${STATE_DIR}/merge.log"

"${PYTHON}" -m benchmarks.emotic_mlcil.runner \
  --protocol "${PROTOCOL}" \
  --method ddp \
  --output-root "${OUTPUT_ROOT}" \
  --reporting-split test \
  --configuration-locked \
  --export-sync-results \
  --shard-run-id "${SHARD_RUN_ID}" \
  2>&1 | tee -a "${STATE_DIR}/merge.log"

echo "Merged benchmark artifacts:"
echo "  ${OUTPUT_ROOT}/benchmarks/emotic_b5c3_v0.1/A/DDP/seed0"
echo "Download only this checkpoint-free folder:"
echo "  ${OUTPUT_ROOT}/benchmarks/emotic_b5c3_v0.1/A/DDP/seed0/results_to_sync/${SHARD_RUN_ID}"
