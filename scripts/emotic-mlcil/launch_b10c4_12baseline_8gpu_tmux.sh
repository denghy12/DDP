#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_b10c4_12baseline}"
RUN_ID="${RUN_ID:-b10c4_12baseline_seed012_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/b10c4_12baseline_v0.1}"
RUN_OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b10c4.yaml}"
REPLAY_CONTRACT="${REPLAY_CONTRACT:-${ROOT}/configs/emotic_mlcil/replay_b10c4_20c_v0.1.yaml}"
DERPP_REPLAY_CONTRACT="${DERPP_REPLAY_CONTRACT:-${ROOT}/configs/emotic_mlcil/replay_derpp_b10c4_20c_v0.1.yaml}"
AGCN_WORD_EMBEDDINGS="${AGCN_WORD_EMBEDDINGS:-${ROOT}/pretrained/agcn/emotic_glove_6b_300d.json}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"
CONFIGURATION_LOCKED_CONFIRMATION="EMOTIC_B10C4_12BASELINE_V0_1"

if [[ ! -f "${CLIP_MODEL_PATH}" ]]; then
  LEGACY_CLIP="/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt"
  if [[ -f "${LEGACY_CLIP}" ]]; then
    CLIP_MODEL_PATH="${LEGACY_CLIP}"
  fi
fi

for required in \
  "${PYTHON}" \
  "${CLIP_MODEL_PATH}" \
  "${PROTOCOL}" \
  "${REPLAY_CONTRACT}" \
  "${DERPP_REPLAY_CONTRACT}" \
  "${AGCN_WORD_EMBEDDINGS}"; do
  [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 2; }
done
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC data root: ${DATA_ROOT}" >&2; exit 2; }
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || {
  echo "B10-C4 formal sweep requires a clean Git worktree" >&2
  git status --short
  exit 2
}
EXPECTED_GIT_COMMIT="$(git rev-parse HEAD)"

read -r -a gpu_values <<< "${GPUS}"
[[ "${#gpu_values[@]}" -eq 8 ]] || {
  echo "GPUS must contain exactly eight physical GPU indices" >&2
  exit 2
}
[[ "$(printf '%s\n' "${gpu_values[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || {
  echo "GPUS must contain eight distinct indices" >&2
  exit 2
}

declare -A free_by_gpu
while IFS=',' read -r index free_mib; do
  index="${index//[[:space:]]/}"
  free_mib="${free_mib//[[:space:]]/}"
  free_by_gpu["${index}"]="${free_mib}"
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
for gpu in "${gpu_values[@]}"; do
  [[ -n "${free_by_gpu[${gpu}]:-}" ]] || { echo "GPU ${gpu} does not exist" >&2; exit 2; }
  if (( free_by_gpu[${gpu}] < MIN_FREE_MIB )); then
    echo "GPU ${gpu} has only ${free_by_gpu[${gpu}]} MiB free; need ${MIN_FREE_MIB}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_BASE}/_launcher_logs"
PREFLIGHT_LOG="${OUTPUT_BASE}/_launcher_logs/${RUN_ID}_preflight.log"
LAUNCHER_LOG="${OUTPUT_BASE}/_launcher_logs/${RUN_ID}.log"

echo "Running B10-C4 protocol and regression preflight..."
{
  "${PYTHON}" - "${PROTOCOL}" "${REPLAY_CONTRACT}" "${DERPP_REPLAY_CONTRACT}" <<'PY'
import sys
from pathlib import Path
from benchmarks.emotic_mlcil.protocol import load_protocol
from benchmarks.emotic_mlcil.replay_memory import ReplayMemoryContract

protocol = load_protocol(Path(sys.argv[1]))
assert protocol.protocol_id == "emotic_b10c4_v0.1"
assert [len(task) for task in protocol.tasks] == [10, 4, 4, 4, 4]
assert list(protocol.class_order) == sorted(protocol.class_order, key=str.casefold)
standard = ReplayMemoryContract.from_yaml(Path(sys.argv[2]), protocol)
derpp = ReplayMemoryContract.from_yaml(Path(sys.argv[3]), protocol)
assert standard.task_capacities == (200, 280, 360, 440, 520)
assert derpp.task_capacities == standard.task_capacities
assert not standard.stores_logits and derpp.stores_logits
print({
    "protocol": protocol.protocol_id,
    "task_sizes": [len(task) for task in protocol.tasks],
    "seen_classes": [len(protocol.seen_class_indices(i)) for i in range(protocol.num_tasks)],
    "replay_capacities": standard.task_capacities,
})
PY
  "${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
  "${PYTHON}" -m unittest \
    tests.test_emotic_task_adapter_bank \
    tests.test_ddp_internal_adapter \
    tests.test_ddp_prompt_free_auxiliary
  echo "git_commit=${EXPECTED_GIT_COMMIT}"
  echo "gpus=${GPUS}"
  echo "gpu_free_mib=$(for gpu in "${gpu_values[@]}"; do printf '%s:%s ' "${gpu}" "${free_by_gpu[${gpu}]}"; done)"
  echo "open_file_limit=$(ulimit -n)"
} 2>&1 | tee "${PREFLIGHT_LOG}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
if [[ -e "${RUN_OUTPUT_ROOT}" ]]; then
  echo "Run output already exists; use a new RUN_ID: ${RUN_OUTPUT_ROOT}" >&2
  exit 2
fi

printf -v tmux_command \
  'cd %q && RUN_ID=%q RUN_OUTPUT_ROOT=%q GPUS=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q PROTOCOL=%q REPLAY_CONTRACT=%q DERPP_REPLAY_CONTRACT=%q AGCN_WORD_EMBEDDINGS=%q EXPECTED_GIT_COMMIT=%q CONFIGURATION_LOCKED_CONFIRMATION=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; echo B10C4_SWEEP_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${RUN_OUTPUT_ROOT}" "${GPUS}" "${PYTHON}" \
  "${DATA_ROOT}" "${CLIP_MODEL_PATH}" "${PROTOCOL}" \
  "${REPLAY_CONTRACT}" "${DERPP_REPLAY_CONTRACT}" \
  "${AGCN_WORD_EMBEDDINGS}" "${EXPECTED_GIT_COMMIT}" \
  "${CONFIGURATION_LOCKED_CONFIRMATION}" \
  "${SCRIPT_DIR}/run_b10c4_12baseline_sweep.sh" "${LAUNCHER_LOG}"

tmux new-session -d -s "${SESSION}" "${tmux_command}"

echo "Started B10-C4 12-baseline sweep: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Commit: ${EXPECTED_GIT_COMMIT}"
echo "Scheduling: 36 jobs, eight GPUs, one process per GPU, dynamic backfill"
echo "Attach: tmux attach -t ${SESSION}"
echo "Progress: watch -n 10 'find ${RUN_OUTPUT_ROOT}/runtime_state -name \"*.done.json\" 2>/dev/null | wc -l'"
echo "Scheduler log: ${LAUNCHER_LOG}"
echo "After success download only: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "And checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
