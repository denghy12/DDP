#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_replay_formal_seed012}"
RUN_ID="${RUN_ID:-replay_er_prs_formal_seed012_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
if [[ ! -s "${CLIP_MODEL_PATH}" && -s "/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt" ]]; then
  CLIP_MODEL_PATH="/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt"
fi
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
PRS_SOURCE_ROOT="${PRS_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/prs_official}"
OUTPUT_BASE="${REPLAY_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/replay_track_a_v0.1}"
MIN_FREE_MIB="${REPLAY_MIN_FREE_MIB:-21000}"
RUN_GPU_SMOKE="${REPLAY_RUN_GPU_SMOKE:-1}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "REPLAY_20C_TRACK_A_V0_1" ]] || {
  echo "Refusing held-out test: invalid replay configuration-lock confirmation" >&2
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
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: ${GPU}" >&2; exit 2; }
[[ "${MIN_FREE_MIB}" =~ ^[0-9]+$ ]] || {
  echo "REPLAY_MIN_FREE_MIB must be a non-negative integer" >&2
  exit 2
}
[[ "${RUN_GPU_SMOKE}" == "0" || "${RUN_GPU_SMOKE}" == "1" ]] || {
  echo "REPLAY_RUN_GPU_SMOKE must be 0 or 1" >&2
  exit 2
}

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${CLIP_MODEL_PATH}" ]] || { echo "Missing CLIP model: ${CLIP_MODEL_PATH}" >&2; exit 2; }
[[ -s "${PROTOCOL}" ]] || { echo "Missing protocol: ${PROTOCOL}" >&2; exit 2; }
[[ -s "${PRS_SOURCE_ROOT}/code/models/reservoir/mlab_stratified_reservoir.py" ]] || {
  echo "Missing fixed PRS source: ${PRS_SOURCE_ROOT}" >&2
  exit 2
}
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }

CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "Refusing formal run: HEAD ${CURRENT_COMMIT} != frozen ${EXPECTED_GIT_COMMIT}" >&2
  exit 2
}
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Formal replay runs require a clean Git worktree" >&2
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
ORACLE_JSON="${PREFLIGHT_DIR}/prs_upstream_equivalence.json"
mkdir -p "${PREFLIGHT_DIR}" "${LOG_DIR}"

set +e
(
set -e
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil.registry import method_names
from benchmarks.emotic_mlcil.runner import CORE_BASE_COMMIT, CORE_RUNTIME_VERSION

if CORE_RUNTIME_VERSION != "0.9.0":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if CORE_BASE_COMMIT != "00f399f13bc7552c254c8f6e6c095a8be4f56146":
    raise RuntimeError(f"Unexpected Core base: {CORE_BASE_COMMIT}")
if not {"er", "prs"}.issubset(method_names()):
    raise RuntimeError("ER/PRS are not registered")
print({"runtime": CORE_RUNTIME_VERSION, "core_base_commit": CORE_BASE_COMMIT})
PY

echo "Running frozen Replay/Core tests..."
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary

free_mib="$(
  nvidia-smi -i "${GPU}" --query-gpu=memory.free \
    --format=csv,noheader,nounits | tr -d '[:space:]'
)"
[[ "${free_mib}" =~ ^[0-9]+$ ]] || {
  echo "Cannot read GPU ${GPU} free memory" >&2
  exit 2
}
[[ "${free_mib}" -ge "${MIN_FREE_MIB}" ]] || {
  echo "GPU ${GPU} has ${free_mib} MiB free; ${MIN_FREE_MIB} MiB required" >&2
  exit 2
}
echo "physical_gpu=${GPU} free_mib=${free_mib}"
echo "capacity_plan=two_automatic_three_process_waves single_peak_mib=5827.2 estimated_wave_peak_mib=17481.6 required_free_mib=${MIN_FREE_MIB}"
nvidia-smi

echo "Running immutable PRS source/operator equivalence..."
"${PYTHON}" "${SCRIPT_DIR}/compare_prs_upstream_reference.py" \
  --upstream-root "${PRS_SOURCE_ROOT}" >"${ORACLE_JSON}"

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running frozen current+replay memory smoke on GPU ${GPU}..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_replay_visual_training.py" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --batch-size 32
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
  echo "Replay formal preflight failed with exit code ${PREFLIGHT_RC}" >&2
  exit "${PREFLIGHT_RC}"
fi
cp "${PREFLIGHT_LOG}" "${PREFLIGHT_DIR}/preflight.log"

printf -v command \
  'cd %q && RUN_ID=%q GPU=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q PROTOCOL=%q RUN_OUTPUT_ROOT=%q EXPECTED_GIT_COMMIT=%q CONFIGURATION_LOCKED_CONFIRMATION=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; echo REPLAY_FORMAL_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" \
  "${CLIP_MODEL_PATH}" "${PROTOCOL}" "${RUN_OUTPUT_ROOT}" \
  "${EXPECTED_GIT_COMMIT}" "${CONFIGURATION_LOCKED_CONFIRMATION}" \
  "${SCRIPT_DIR}/run_replay_formal_seed012.sh" "${LAUNCHER_LOG}"

tmux new-session -d -s "${SESSION}" -n "replay_seed012_gpu${GPU}" "${command}"
echo "Started locked ER/PRS formal session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Execution: ER seeds 0/1/2 concurrently on GPU${GPU}, then PRS seeds 0/1/2 concurrently on GPU${GPU}"
echo "Maximum concurrent training processes: 3 (never 6)"
echo "Frozen loader: train batch 32, eval batch 64, workers 2"
echo "Attach: tmux attach -t ${SESSION}"
echo "Preflight log: ${PREFLIGHT_LOG}"
echo "Runtime logs: ${RUN_OUTPUT_ROOT}/runtime_logs"
echo "Checkpoints stay server-only under: ${RUN_OUTPUT_ROOT}/benchmarks"
echo "Final archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Final checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
