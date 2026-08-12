#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_emot_net_ccim_ft_formal_seed0}"
RUN_ID="${RUN_ID:-emot_net_ccim_ft_formal_seed0_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-2,3,4}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
NATIVE_INIT="${EMOT_NET_NATIVE_INIT:-${ROOT}/pretrained/emot_net/emot_net_native_init_v0.1.pth}"
CCIM_DICTIONARY="${CCIM_DICTIONARY:-${ROOT}/pretrained/ccim/emotic_task0_places365_k256_v0.1.pth}"
UPSTREAM_ROOT="${CCIM_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/ccim_official}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
OUTPUT_BASE="${EMOT_NET_CCIM_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/emot_net_ccim_ft_track_b_v0.1}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"

[[ "${GPUS}" == "2,3,4" ]] || { echo "Formal launcher requires GPUs 2,3,4" >&2; exit 2; }
[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "EMOT_NET_CCIM_FT_TRACK_B_V0_1" ]] || exit 2
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || exit 2
[[ -z "$(git status --porcelain)" ]] || { echo "Formal run requires a clean worktree" >&2; exit 2; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Session exists: ${SESSION}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" && -s "${NATIVE_INIT}" && -s "${CCIM_DICTIONARY}" ]] || exit 2
[[ -s "${UPSTREAM_ROOT}/CCIM.py" ]] || exit 2

for gpu in 2 3 4; do
  free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ "${free}" =~ ^[0-9]+$ && "${free}" -ge 20000 ]] || {
    echo "GPU ${gpu} has only ${free:-unknown} MiB free" >&2
    exit 2
  }
done

RUN_OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
[[ ! -e "${RUN_OUTPUT_ROOT}" ]] || exit 2
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
LOG_DIR="${OUTPUT_BASE}/_launcher_logs"
PREFLIGHT_LOG="${LOG_DIR}/${RUN_ID}_preflight.log"
ORACLE_JSON="${PREFLIGHT_DIR}/upstream_oracle.json"
LAUNCHER_LOG="${LOG_DIR}/${RUN_ID}.log"
mkdir -p "${PREFLIGHT_DIR}" "${LOG_DIR}"

set +e
(
set -e
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil.registry import method_names
from benchmarks.emotic_mlcil.runner import CORE_RUNTIME_VERSION
assert CORE_RUNTIME_VERSION == "0.12.1", CORE_RUNTIME_VERSION
assert "emot_net_ccim_ft" in method_names()
print({"runtime": CORE_RUNTIME_VERSION, "formal_seed": 0, "track": "B", "physical_gpus": (2, 3, 4)})
PY
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest tests.test_emotic_task_adapter_bank tests.test_ddp_internal_adapter tests.test_ddp_prompt_free_auxiliary
"${PYTHON}" "${SCRIPT_DIR}/compare_ccim_upstream_reference.py" --upstream-root "${UPSTREAM_ROOT}" --output "${ORACLE_JSON}"
CUDA_VISIBLE_DEVICES="${GPUS}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "${PYTHON}" "${SCRIPT_DIR}/smoke_emot_net_ccim_ft_training.py" \
  --native-init "${NATIVE_INIT}" --ccim-dictionary "${CCIM_DICTIONARY}" \
  --batch-size 52 --tower-model-parallel
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
[[ "${PREFLIGHT_RC}" -eq 0 ]] || exit "${PREFLIGHT_RC}"
cp "${PREFLIGHT_LOG}" "${PREFLIGHT_DIR}/preflight.log"

printf -v command \
  'cd %q && RUN_ID=%q GPUS=%q PYTHON=%q DATA_ROOT=%q EMOT_NET_NATIVE_INIT=%q CCIM_DICTIONARY=%q PROTOCOL=%q RUN_OUTPUT_ROOT=%q EXPECTED_GIT_COMMIT=%q CONFIGURATION_LOCKED_CONFIRMATION=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; echo EMOT_NET_CCIM_FT_FORMAL_SEED0_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPUS}" "${PYTHON}" "${DATA_ROOT}" \
  "${NATIVE_INIT}" "${CCIM_DICTIONARY}" "${PROTOCOL}" "${RUN_OUTPUT_ROOT}" \
  "${EXPECTED_GIT_COMMIT}" "${CONFIGURATION_LOCKED_CONFIRMATION}" \
  "${SCRIPT_DIR}/run_emot_net_ccim_ft_formal_seed0.sh" "${LAUNCHER_LOG}"

tmux new-session -d -s "${SESSION}" -n "emot_net_ccim_ft_seed0_g234" "${command}"
echo "Started locked EMOT-Net+CCIM-FT Track-B formal seed 0: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Physical GPUs: ${GPUS}; primary=2, context=3, body=4"
echo "Frozen loader: train batch 52, eval batch 16, workers 8"
echo "Attach: tmux attach -t ${SESSION}"
echo "Runtime log: ${RUN_OUTPUT_ROOT}/runtime_logs/seed0_gpu234.log"
echo "Checkpoint-free archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
