#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_csc_formal_seed012}"
RUN_ID="${RUN_ID:-csc_formal_seed012_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-0 1 2}"
GPUS="${GPUS:-0 1 2}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/csc_track_a_v0.1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "CSC_TRACK_A_V0_1" ]] || {
  echo "Refusing held-out test: invalid configuration-lock confirmation" >&2
  exit 2
}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid RUN_ID: ${RUN_ID}" >&2
  exit 2
}
for value in "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${WORKERS}" "${MIN_FREE_MIB}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "Batch, worker, and memory settings must be non-negative integers" >&2
    exit 2
  }
done
[[ "${TRAIN_BATCH_SIZE}" -eq 64 && "${EVAL_BATCH_SIZE}" -eq 64 && "${WORKERS}" -eq 2 ]] || {
  echo "Frozen CSC formal runner requires train/eval batch 64 and workers 2" >&2
  exit 2
}
[[ "${RUN_GPU_SMOKE}" == "0" || "${RUN_GPU_SMOKE}" == "1" ]] || {
  echo "RUN_GPU_SMOKE must be 0 or 1" >&2
  exit 2
}

read -r -a seed_values <<< "${SEEDS}"
read -r -a gpu_values <<< "${GPUS}"
[[ "${seed_values[*]}" == "0 1 2" ]] || {
  echo "Formal CSC seeds must be exactly: 0 1 2" >&2
  exit 2
}
[[ "${#gpu_values[@]}" -eq 3 ]] || {
  echo "GPUS must contain three physical GPU IDs" >&2
  exit 2
}
seen_gpus=" "
for gpu in "${gpu_values[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: ${gpu}" >&2; exit 2; }
  if [[ "${seen_gpus}" == *" ${gpu} "* ]]; then
    echo "Each CSC seed requires a distinct GPU; duplicate ${gpu}" >&2
    exit 2
  fi
  seen_gpus+="${gpu} "
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing data root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${CLIP_MODEL_PATH}" ]] || { echo "Missing CLIP model: ${CLIP_MODEL_PATH}" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }

CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "Refusing formal run: HEAD ${CURRENT_COMMIT} != frozen ${EXPECTED_GIT_COMMIT}" >&2
  exit 2
}
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Formal CSC runs require a clean Git worktree" >&2
  exit 2
fi

RUN_OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
[[ ! -e "${RUN_OUTPUT_ROOT}" ]] || {
  echo "Run output already exists: ${RUN_OUTPUT_ROOT}" >&2
  exit 2
}
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
LOG_DIR="${OUTPUT_BASE}/_launcher_logs"
PREFLIGHT_LOG="${LOG_DIR}/${RUN_ID}_preflight.log"
LAUNCHER_LOG="${LOG_DIR}/${RUN_ID}.log"
mkdir -p "${PREFLIGHT_DIR}" "${LOG_DIR}"

set +e
(
set -e
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil import method_names
from benchmarks.emotic_mlcil.runner import CORE_BASE_COMMIT, CORE_RUNTIME_VERSION

if CORE_RUNTIME_VERSION != "0.4.0":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if CORE_BASE_COMMIT != "00f399f13bc7552c254c8f6e6c095a8be4f56146":
    raise RuntimeError(f"Unexpected Core base: {CORE_BASE_COMMIT}")
if "csc" not in method_names():
    raise RuntimeError("CSC is not registered")
print({"runtime": CORE_RUNTIME_VERSION, "core_base_commit": CORE_BASE_COMMIT})
PY

echo "Running frozen CSC/Core tests..."
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary

for index in "${!gpu_values[@]}"; do
  gpu="${gpu_values[index]}"
  free_mib="$(
    nvidia-smi -i "${gpu}" --query-gpu=memory.free \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  [[ "${free_mib}" =~ ^[0-9]+$ ]] || {
    echo "Cannot read free memory for GPU ${gpu}: ${free_mib}" >&2
    exit 2
  }
  [[ "${free_mib}" -ge "${MIN_FREE_MIB}" ]] || {
    echo "GPU ${gpu} has ${free_mib} MiB free; ${MIN_FREE_MIB} MiB required" >&2
    exit 2
  }
  echo "physical_gpu=${gpu} free_mib=${free_mib} assigned_seed=${seed_values[index]}"
done
nvidia-smi

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  smoke_gpu="${gpu_values[0]}"
  echo "Running frozen worst-task CSC memory smoke on physical GPU ${smoke_gpu}..."
  CUDA_VISIBLE_DEVICES="${smoke_gpu}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_csc_visual_training.py" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --batch-size "${TRAIN_BATCH_SIZE}"
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
  echo "CSC formal preflight failed with exit code ${PREFLIGHT_RC}" >&2
  echo "Preflight log: ${PREFLIGHT_LOG}" >&2
  exit "${PREFLIGHT_RC}"
fi
cp "${PREFLIGHT_LOG}" "${PREFLIGHT_DIR}/preflight.log"

printf -v command \
  'cd %q && RUN_ID=%q SEEDS=%q GPUS=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q PROTOCOL=%q RUN_OUTPUT_ROOT=%q TRAIN_BATCH_SIZE=64 EVAL_BATCH_SIZE=64 WORKERS=2 EXPECTED_GIT_COMMIT=%q CONFIGURATION_LOCKED_CONFIRMATION=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; echo CSC_FORMAL_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${SEEDS}" "${GPUS}" "${PYTHON}" \
  "${DATA_ROOT}" "${CLIP_MODEL_PATH}" "${PROTOCOL}" \
  "${RUN_OUTPUT_ROOT}" "${EXPECTED_GIT_COMMIT}" \
  "${CONFIGURATION_LOCKED_CONFIRMATION}" \
  "${SCRIPT_DIR}/run_csc_formal_seed012.sh" "${LAUNCHER_LOG}"

tmux new-session -d -s "${SESSION}" -n "csc_seed012" "${command}"
echo "Started locked CSC three-seed formal session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Assignments: seed0->GPU${gpu_values[0]}, seed1->GPU${gpu_values[1]}, seed2->GPU${gpu_values[2]}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Preflight log: ${PREFLIGHT_LOG}"
echo "Runtime logs: ${RUN_OUTPUT_ROOT}/runtime_logs"
echo "Checkpoints stay server-only: ${RUN_OUTPUT_ROOT}/benchmarks/emotic_b5c3_v0.1/A/CSC/seed*/checkpoints"
echo "Final archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Final checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
