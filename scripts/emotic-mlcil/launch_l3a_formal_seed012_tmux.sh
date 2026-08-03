#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_l3a_formal_seed012}"
RUN_ID="${RUN_ID:-l3a_formal_seed012_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
if [[ ! -s "${CLIP_MODEL_PATH}" && -s "/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt" ]]; then
  CLIP_MODEL_PATH="/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt"
fi
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
UPSTREAM_ROOT="${L3A_UPSTREAM_ROOT:-/mnt/haoyuan/workspace/baseline_sources/l3a_official}"
UPSTREAM_ARCHIVE="${L3A_UPSTREAM_ARCHIVE:-}"
OUTPUT_BASE="${L3A_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/l3a_track_a_v0.1}"
MIN_FREE_MIB="${L3A_MIN_FREE_MIB:-10000}"
RUN_GPU_SMOKE="${L3A_RUN_GPU_SMOKE:-1}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "L3A_TRACK_A_V0_1" ]] || {
  echo "Refusing held-out test: invalid L3A configuration-lock confirmation" >&2
  exit 2
}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid RUN_ID: ${RUN_ID}" >&2
  exit 2
}
[[ "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "EXPECTED_GIT_COMMIT must be a full lowercase SHA" >&2
  exit 2
}
[[ "${MIN_FREE_MIB}" =~ ^[0-9]+$ ]] || {
  echo "L3A_MIN_FREE_MIB must be a non-negative integer" >&2
  exit 2
}
[[ "${RUN_GPU_SMOKE}" == "0" || "${RUN_GPU_SMOKE}" == "1" ]] || {
  echo "L3A_RUN_GPU_SMOKE must be 0 or 1" >&2
  exit 2
}
read -r -a gpu_values <<< "${GPUS}"
[[ "${#gpu_values[@]}" -eq 3 ]] || {
  echo "GPUS must contain three distinct physical GPU IDs" >&2
  exit 2
}
seen_gpus=" "
for gpu in "${gpu_values[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: ${gpu}" >&2; exit 2; }
  if [[ "${seen_gpus}" == *" ${gpu} "* ]]; then
    echo "Each L3A seed requires a distinct GPU; duplicate ${gpu}" >&2
    exit 2
  fi
  seen_gpus+="${gpu} "
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${CLIP_MODEL_PATH}" ]] || { echo "Missing CLIP model: ${CLIP_MODEL_PATH}" >&2; exit 2; }
[[ -s "${UPSTREAM_ROOT}/MultiLabelIncremental_L3A.py" ]] || {
  echo "Missing fixed L3A source: ${UPSTREAM_ROOT}" >&2
  exit 2
}
if [[ -n "${UPSTREAM_ARCHIVE}" && ! -s "${UPSTREAM_ARCHIVE}" ]]; then
  echo "Configured L3A archive is missing: ${UPSTREAM_ARCHIVE}" >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }

CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "Refusing formal run: HEAD ${CURRENT_COMMIT} != frozen ${EXPECTED_GIT_COMMIT}" >&2
  exit 2
}
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Formal L3A runs require a clean Git worktree" >&2
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
ORACLE_JSON="${PREFLIGHT_DIR}/upstream_oracle.json"
mkdir -p "${PREFLIGHT_DIR}" "${LOG_DIR}"

set +e
(
set -e
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil.registry import method_names
from benchmarks.emotic_mlcil.runner import CORE_BASE_COMMIT, CORE_RUNTIME_VERSION

if CORE_RUNTIME_VERSION != "0.6.0":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if CORE_BASE_COMMIT != "00f399f13bc7552c254c8f6e6c095a8be4f56146":
    raise RuntimeError(f"Unexpected Core base: {CORE_BASE_COMMIT}")
if "l3a" not in method_names():
    raise RuntimeError("L3A is not registered")
print({"runtime": CORE_RUNTIME_VERSION, "core_base_commit": CORE_BASE_COMMIT})
PY

echo "Running frozen L3A/Core tests..."
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
    echo "Cannot read GPU ${gpu} free memory" >&2
    exit 2
  }
  [[ "${free_mib}" -ge "${MIN_FREE_MIB}" ]] || {
    echo "GPU ${gpu} has ${free_mib} MiB free; ${MIN_FREE_MIB} MiB required" >&2
    exit 2
  }
  echo "physical_gpu=${gpu} free_mib=${free_mib} assigned_seed=${index}"
done
nvidia-smi

oracle_args=(--upstream-root "${UPSTREAM_ROOT}" --output "${ORACLE_JSON}")
if [[ -n "${UPSTREAM_ARCHIVE}" ]]; then
  oracle_args+=(--upstream-archive "${UPSTREAM_ARCHIVE}")
fi
echo "Running fixed-source L3A operator equivalence..."
"${PYTHON}" "${SCRIPT_DIR}/compare_l3a_upstream_reference.py" "${oracle_args[@]}"

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running frozen L3A Task-0/analytic memory smoke on GPU ${gpu_values[0]}..."
  CUDA_VISIBLE_DEVICES="${gpu_values[0]}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_l3a_training.py" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --batch-size 64 \
    --analytic-batch-size 64
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
  echo "L3A formal preflight failed with exit code ${PREFLIGHT_RC}" >&2
  exit "${PREFLIGHT_RC}"
fi
cp "${PREFLIGHT_LOG}" "${PREFLIGHT_DIR}/preflight.log"

printf -v command \
  'cd %q && RUN_ID=%q GPUS=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q PROTOCOL=%q RUN_OUTPUT_ROOT=%q EXPECTED_GIT_COMMIT=%q CONFIGURATION_LOCKED_CONFIRMATION=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; echo L3A_FORMAL_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPUS}" "${PYTHON}" "${DATA_ROOT}" \
  "${CLIP_MODEL_PATH}" "${PROTOCOL}" "${RUN_OUTPUT_ROOT}" \
  "${EXPECTED_GIT_COMMIT}" "${CONFIGURATION_LOCKED_CONFIRMATION}" \
  "${SCRIPT_DIR}/run_l3a_formal_seed012.sh" "${LAUNCHER_LOG}"

tmux new-session -d -s "${SESSION}" -n "l3a_parallel_seed012" "${command}"
echo "Started locked parallel L3A formal session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Concurrent assignments: seed0->GPU${gpu_values[0]}, seed1->GPU${gpu_values[1]}, seed2->GPU${gpu_values[2]}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Preflight log: ${PREFLIGHT_LOG}"
echo "Runtime logs: ${RUN_OUTPUT_ROOT}/runtime_logs"
echo "Checkpoints stay server-only: ${RUN_OUTPUT_ROOT}/benchmarks/emotic_b5c3_v0.1/A/L3A/seed*/checkpoints"
echo "Final archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Final checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
