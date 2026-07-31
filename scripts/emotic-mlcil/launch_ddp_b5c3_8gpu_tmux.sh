#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_benchmark_ddp8}"
SHARD_RUN_ID="${SHARD_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/output}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WORKERS="${WORKERS:-0}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"

read -r -a gpu_ids <<< "${GPU_LIST}"
if [[ "${#gpu_ids[@]}" -ne 8 ]]; then
  echo "GPU_LIST must contain exactly 8 physical GPU indices" >&2
  exit 2
fi
[[ "${SHARD_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid SHARD_RUN_ID: ${SHARD_RUN_ID}" >&2
  exit 2
}
for gpu in "${gpu_ids[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || {
    echo "GPU_LIST contains an invalid index: ${gpu}" >&2
    exit 2
  }
done
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing data root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${CLIP_MODEL_PATH}" ]] || {
  echo "Missing CLIP model: ${CLIP_MODEL_PATH}" >&2
  exit 2
}
for task_id in 0 1 2 3 4 5 6 7; do
  [[ -s "${CHECKPOINT_DIR}/task${task_id}.pth" ]] || {
    echo "Missing checkpoint: ${CHECKPOINT_DIR}/task${task_id}.pth" >&2
    exit 2
  }
done

echo "Verifying synchronized benchmark runtime..."
"${PYTHON}" - <<'PY'
import json

from benchmarks.emotic_mlcil.artifacts import ArtifactStore
from benchmarks.emotic_mlcil.runner import (
    CORE_RUNTIME_VERSION,
    _current_source_state,
)

expected = "0.3.1"
if CORE_RUNTIME_VERSION != expected:
    raise RuntimeError(
        f"Server benchmark runtime {CORE_RUNTIME_VERSION!r} != {expected!r}"
    )
for method_name in ("load_shard_scores", "export_sync_results"):
    if not hasattr(ArtifactStore, method_name):
        raise RuntimeError(
            f"Server ArtifactStore is missing {method_name}; synchronize source"
        )
state = _current_source_state()
if len(state["source_tree_hash"]) != 64:
    raise RuntimeError("Invalid benchmark source-tree fingerprint")
print(json.dumps({"runtime": CORE_RUNTIME_VERSION, **state}, indent=2))
PY

echo "Running fast benchmark-core preflight tests..."
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .

for task_id in 0 1 2 3 4 5 6 7; do
  gpu="${gpu_ids[task_id]}"
  printf -v command \
    'cd %q && TASK_ID=%q GPU=%q SHARD_RUN_ID=%q PYTHON=%q DATA_ROOT=%q CHECKPOINT_DIR=%q CLIP_MODEL_PATH=%q OUTPUT_ROOT=%q EVAL_BATCH_SIZE=%q WORKERS=%q PROTOCOL=%q bash %q; code=$?; echo TASK_EXIT_CODE=$code; exec bash' \
    "${ROOT}" "${task_id}" "${gpu}" "${SHARD_RUN_ID}" "${PYTHON}" \
    "${DATA_ROOT}" "${CHECKPOINT_DIR}" "${CLIP_MODEL_PATH}" "${OUTPUT_ROOT}" \
    "${EVAL_BATCH_SIZE}" "${WORKERS}" "${PROTOCOL}" \
    "${SCRIPT_DIR}/run_ddp_b5c3_task_shard.sh"
  if [[ "${task_id}" -eq 0 ]]; then
    tmux new-session -d -s "${SESSION}" -n "task0_gpu${gpu}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "task${task_id}_gpu${gpu}" "${command}"
  fi
done

printf -v merge_command \
  'cd %q && SHARD_RUN_ID=%q PYTHON=%q OUTPUT_ROOT=%q PROTOCOL=%q bash %q; code=$?; echo MERGE_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${SHARD_RUN_ID}" "${PYTHON}" "${OUTPUT_ROOT}" "${PROTOCOL}" \
  "${SCRIPT_DIR}/wait_and_merge_ddp_b5c3_shards.sh"
tmux new-window -t "${SESSION}" -n merge "${merge_command}"

echo "Started tmux session: ${SESSION}"
echo "Shard run ID: ${SHARD_RUN_ID}"
echo "Task/GPU map: task0..task7 -> ${GPU_LIST}"
echo "Evaluation batch size: ${EVAL_BATCH_SIZE}; workers per task: ${WORKERS}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Status: tmux list-windows -t ${SESSION}"
echo "Logs: ${OUTPUT_ROOT}/benchmarks/emotic_b5c3_v0.1/A/DDP/seed0/shards/${SHARD_RUN_ID}/_state"
