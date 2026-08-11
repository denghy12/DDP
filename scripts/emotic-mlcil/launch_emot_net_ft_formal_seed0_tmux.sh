#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_emot_net_ft_formal_seed0}"
RUN_ID="${RUN_ID:-emot_net_ft_formal_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
NATIVE_INIT="${EMOT_NET_NATIVE_INIT:-${ROOT}/pretrained/emot_net/emot_net_native_init_v0.1.pth}"
UPSTREAM_ROOT="${EMOT_NET_UPSTREAM_ROOT:-/mnt/haoyuan/workspace/baseline_sources/emot_net_release_69c3a51}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
OUTPUT_BASE="${EMOT_NET_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/emot_net_ft_track_b_v0.1}"
MIN_FREE_GPU_MIB="${EMOT_NET_MIN_FREE_GPU_MIB:-4096}"
MIN_FREE_DISK_MIB="${EMOT_NET_MIN_FREE_DISK_MIB:-8192}"
RUN_GPU_SMOKE="${EMOT_NET_RUN_GPU_SMOKE:-1}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "EMOT_NET_FT_TRACK_B_V0_1" ]] || {
  echo "Refusing held-out test: invalid EMOT-Net-FT configuration lock" >&2
  exit 2
}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
[[ "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "EXPECTED_GIT_COMMIT must be a full lowercase SHA" >&2; exit 2; }
[[ "${MIN_FREE_GPU_MIB}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU memory guard" >&2; exit 2; }
[[ "${MIN_FREE_DISK_MIB}" =~ ^[0-9]+$ ]] || { echo "Invalid disk guard" >&2; exit 2; }
[[ "${RUN_GPU_SMOKE}" == "0" || "${RUN_GPU_SMOKE}" == "1" ]] || { echo "EMOT_NET_RUN_GPU_SMOKE must be 0 or 1" >&2; exit 2; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${NATIVE_INIT}" ]] || { echo "Missing audited EMOT-Net initialization: ${NATIVE_INIT}" >&2; exit 2; }
[[ -s "${UPSTREAM_ROOT}/src/create_model.lua" ]] || { echo "Missing fixed EMOT-Net source: ${UPSTREAM_ROOT}" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }

CURRENT_COMMIT="$(git rev-parse HEAD)"
[[ "${CURRENT_COMMIT}" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "Refusing formal run: HEAD ${CURRENT_COMMIT} != frozen ${EXPECTED_GIT_COMMIT}" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || { echo "Formal run requires a clean Git worktree" >&2; exit 2; }

mkdir -p "${OUTPUT_BASE}"
RUN_OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
[[ ! -e "${RUN_OUTPUT_ROOT}" ]] || { echo "Run output already exists: ${RUN_OUTPUT_ROOT}" >&2; exit 2; }
FREE_GPU_MIB="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "${FREE_GPU_MIB}" =~ ^[0-9]+$ && "${FREE_GPU_MIB}" -ge "${MIN_FREE_GPU_MIB}" ]] || {
  echo "GPU ${GPU} has ${FREE_GPU_MIB:-unknown} MiB free; ${MIN_FREE_GPU_MIB} MiB required" >&2
  exit 2
}
FREE_DISK_MIB="$(( $(df -Pk "${OUTPUT_BASE}" | awk 'NR==2 {print $4}') / 1024 ))"
[[ "${FREE_DISK_MIB}" -ge "${MIN_FREE_DISK_MIB}" ]] || {
  echo "Output filesystem has ${FREE_DISK_MIB} MiB free; ${MIN_FREE_DISK_MIB} MiB required" >&2
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
from benchmarks.emotic_mlcil.runner import CORE_RUNTIME_VERSION

if CORE_RUNTIME_VERSION != "0.10.0":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if "emot_net_ft" not in method_names():
    raise RuntimeError("EMOT-Net-FT is not registered")
print({"runtime": CORE_RUNTIME_VERSION, "formal_seed": 0, "track": "B"})
PY

echo "Running frozen EMOT-Net-FT/Core tests..."
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary

echo "Running immutable EMOT-Net source/operator audit..."
"${PYTHON}" "${SCRIPT_DIR}/compare_emot_net_upstream_reference.py" \
  --upstream-root "${UPSTREAM_ROOT}" \
  --output "${ORACLE_JSON}"

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running frozen native EMOT-Net memory smoke on GPU ${GPU}..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_emot_net_ft_training.py" \
    --native-init "${NATIVE_INIT}" \
    --batch-size 52
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
  echo "EMOT-Net-FT formal preflight failed with exit code ${PREFLIGHT_RC}" >&2
  exit "${PREFLIGHT_RC}"
fi
cp "${PREFLIGHT_LOG}" "${PREFLIGHT_DIR}/preflight.log"

printf -v command \
  'cd %q && RUN_ID=%q GPU=%q PYTHON=%q DATA_ROOT=%q EMOT_NET_NATIVE_INIT=%q PROTOCOL=%q RUN_OUTPUT_ROOT=%q EXPECTED_GIT_COMMIT=%q CONFIGURATION_LOCKED_CONFIRMATION=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; echo EMOT_NET_FT_FORMAL_SEED0_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" \
  "${NATIVE_INIT}" "${PROTOCOL}" "${RUN_OUTPUT_ROOT}" \
  "${EXPECTED_GIT_COMMIT}" "${CONFIGURATION_LOCKED_CONFIRMATION}" \
  "${SCRIPT_DIR}/run_emot_net_ft_formal_seed0.sh" "${LAUNCHER_LOG}"

tmux new-session -d -s "${SESSION}" -n "emot_net_ft_seed0_g${GPU}" "${command}"
echo "Started locked EMOT-Net-FT Track-B formal seed 0: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "GPU: ${GPU}; free GPU memory: ${FREE_GPU_MIB} MiB; free disk: ${FREE_DISK_MIB} MiB"
echo "Frozen loader: train batch 52, eval batch 16, workers 0"
echo "Attach: tmux attach -t ${SESSION}"
echo "Runtime log: ${RUN_OUTPUT_ROOT}/runtime_logs/seed0_gpu${GPU}.log"
echo "Checkpoint-free archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
