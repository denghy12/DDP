#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_krt_formal_seed012}"
RUN_ID="${RUN_ID:-krt_formal_seed012_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/krt_track_a_v0.1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"
KRT_JOB_MEMORY_MIB="${KRT_JOB_MEMORY_MIB:-12000}"
GPU_RESERVE_MIB="${GPU_RESERVE_MIB:-2048}"
MAX_CONCURRENT_PER_GPU="${MAX_CONCURRENT_PER_GPU:-2}"
AUTO_RETRY_SEQUENTIAL="${AUTO_RETRY_SEQUENTIAL:-1}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "KRT_TRACK_A_V0_1" ]] || {
  echo "Refusing held-out test: invalid configuration-lock confirmation" >&2
  exit 2
}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid RUN_ID: ${RUN_ID}" >&2
  exit 2
}
for value in "${GPU}" "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" \
  "${WORKERS}" "${KRT_JOB_MEMORY_MIB}" "${GPU_RESERVE_MIB}" \
  "${MAX_CONCURRENT_PER_GPU}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "GPU and capacity settings must be non-negative integers" >&2
    exit 2
  }
done
[[ "${TRAIN_BATCH_SIZE}" -gt 0 && "${EVAL_BATCH_SIZE}" -gt 0 ]] || {
  echo "Batch sizes must be positive" >&2
  exit 2
}
[[ "${KRT_JOB_MEMORY_MIB}" -gt 0 && "${MAX_CONCURRENT_PER_GPU}" -gt 0 ]] || {
  echo "KRT_JOB_MEMORY_MIB and MAX_CONCURRENT_PER_GPU must be positive" >&2
  exit 2
}
for flag in "${AUTO_RETRY_SEQUENTIAL}" "${RUN_GPU_SMOKE}"; do
  [[ "${flag}" == "0" || "${flag}" == "1" ]] || {
    echo "AUTO_RETRY_SEQUENTIAL and RUN_GPU_SMOKE must be 0 or 1" >&2
    exit 2
  }
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
  echo "Formal KRT runs require a clean Git worktree" >&2
  exit 2
fi

RUN_OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
[[ ! -e "${RUN_OUTPUT_ROOT}" ]] || {
  echo "Run output already exists: ${RUN_OUTPUT_ROOT}" >&2
  exit 2
}
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
mkdir -p "${PREFLIGHT_DIR}"

echo "Verifying frozen KRT runtime and running CPU tests..."
"${PYTHON}" - <<'PY' 2>&1 | tee "${PREFLIGHT_DIR}/runtime.log"
from benchmarks.emotic_mlcil import method_names
from benchmarks.emotic_mlcil.runner import CORE_BASE_COMMIT, CORE_RUNTIME_VERSION

if CORE_RUNTIME_VERSION != "0.3.1":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if CORE_BASE_COMMIT != "00f399f13bc7552c254c8f6e6c095a8be4f56146":
    raise RuntimeError(f"Unexpected Core base: {CORE_BASE_COMMIT}")
if "krt" not in method_names():
    raise RuntimeError("KRT is not registered")
print({"runtime": CORE_RUNTIME_VERSION, "core_base_commit": CORE_BASE_COMMIT})
PY
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t . \
  2>&1 | tee "${PREFLIGHT_DIR}/core_tests.log"
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary \
  2>&1 | tee "${PREFLIGHT_DIR}/legacy_tests.log"

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running frozen worst-task KRT memory smoke on physical GPU ${GPU}..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_krt_visual_training.py" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    2>&1 | tee "${PREFLIGHT_DIR}/gpu_memory_smoke.log"
fi

FREE_MIB="$(
  nvidia-smi -i "${GPU}" --query-gpu=memory.free \
    --format=csv,noheader,nounits | tr -d '[:space:]'
)"
[[ "${FREE_MIB}" =~ ^[0-9]+$ ]] || {
  echo "Cannot read free memory for GPU ${GPU}: ${FREE_MIB}" >&2
  exit 2
}
USABLE_MIB=$((FREE_MIB - GPU_RESERVE_MIB))
CAPACITY=0
if [[ "${USABLE_MIB}" -gt 0 ]]; then
  CAPACITY=$((USABLE_MIB / KRT_JOB_MEMORY_MIB))
fi
if [[ "${CAPACITY}" -gt "${MAX_CONCURRENT_PER_GPU}" ]]; then
  CAPACITY="${MAX_CONCURRENT_PER_GPU}"
fi
[[ "${CAPACITY}" -gt 0 ]] || {
  echo "GPU ${GPU} lacks safe memory for even one frozen KRT job" >&2
  exit 2
}
{
  echo "physical_gpu=${GPU}"
  echo "free_mib=${FREE_MIB}"
  echo "reserve_mib=${GPU_RESERVE_MIB}"
  echo "estimated_per_job_mib=${KRT_JOB_MEMORY_MIB}"
  echo "max_concurrent_jobs=${CAPACITY}"
  echo "seeds=${SEEDS}"
  echo "frozen_git_commit=${CURRENT_COMMIT}"
} | tee "${PREFLIGHT_DIR}/capacity_plan.log"

printf -v command \
  'cd %q && RUN_ID=%q GPU=%q SEEDS=%q MAX_CONCURRENT_JOBS=%q AUTO_RETRY_SEQUENTIAL=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q PROTOCOL=%q RUN_OUTPUT_ROOT=%q TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q CONFIGURATION_LOCKED_CONFIRMATION=%q bash %q; code=$?; echo KRT_FORMAL_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${SEEDS}" "${CAPACITY}" \
  "${AUTO_RETRY_SEQUENTIAL}" "${PYTHON}" "${DATA_ROOT}" \
  "${CLIP_MODEL_PATH}" "${PROTOCOL}" "${RUN_OUTPUT_ROOT}" \
  "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${WORKERS}" \
  "${CONFIGURATION_LOCKED_CONFIRMATION}" \
  "${SCRIPT_DIR}/run_krt_formal_seed012.sh"

tmux new-session -d -s "${SESSION}" -n "krt_seed012_g${GPU}" "${command}"
echo "Started locked KRT formal session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Physical GPU: ${GPU}; concurrent jobs: ${CAPACITY}; seeds: ${SEEDS}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Runtime logs: ${RUN_OUTPUT_ROOT}/runtime_logs"
echo "Heavy checkpoints: ${RUN_OUTPUT_ROOT}/benchmarks/emotic_b5c3_v0.1/A/KRT/seed*/checkpoints"
echo "Final checkpoint-free archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
