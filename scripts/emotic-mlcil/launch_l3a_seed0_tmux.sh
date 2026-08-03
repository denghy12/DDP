#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_l3a_seed0_val}"
RUN_ID="${RUN_ID:-l3a_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
if [[ ! -s "${CLIP_MODEL_PATH}" && -s "/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt" ]]; then
  CLIP_MODEL_PATH="/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt"
fi
UPSTREAM_ROOT="${L3A_UPSTREAM_ROOT:-/mnt/haoyuan/workspace/baseline_sources/l3a_official}"
if [[ ! -d "${UPSTREAM_ROOT}" && -d "/mnt/haoyuan/workspace/baseline_sources/L3A-main" ]]; then
  UPSTREAM_ROOT="/mnt/haoyuan/workspace/baseline_sources/L3A-main"
fi
UPSTREAM_ARCHIVE="${L3A_UPSTREAM_ARCHIVE:-}"
OUTPUT_BASE="${L3A_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/l3a_track_a_v0.1}"
TRAIN_BATCH_SIZE="${L3A_TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${L3A_EVAL_BATCH_SIZE:-64}"
WORKERS="${L3A_WORKERS:-2}"
REQUIRE_CLEAN="${L3A_REQUIRE_CLEAN:-1}"
RUN_GPU_SMOKE="${L3A_RUN_GPU_SMOKE:-1}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid RUN_ID" >&2
  exit 2
}
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
[[ "${TRAIN_BATCH_SIZE}" == "64" ]] || {
  echo "Registered L3A Track-A train batch size is 64" >&2
  exit 2
}
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${CLIP_MODEL_PATH}" ]] || {
  echo "Missing CLIP checkpoint: ${CLIP_MODEL_PATH}" >&2
  exit 2
}
[[ -s "${UPSTREAM_ROOT}/MultiLabelIncremental_L3A.py" ]] || {
  echo "Missing fixed L3A source: ${UPSTREAM_ROOT}" >&2
  exit 2
}
if [[ -n "${UPSTREAM_ARCHIVE}" && ! -s "${UPSTREAM_ARCHIVE}" ]]; then
  echo "Configured L3A archive is missing: ${UPSTREAM_ARCHIVE}" >&2
  exit 2
fi
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "L3A validation requires a clean Git worktree" >&2
  exit 2
fi

RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_DIR="${OUTPUT_BASE}/_launcher_logs"
PREFLIGHT_LOG="${LOG_DIR}/${RUN_ID}_preflight.log"
ORACLE_JSON="${LOG_DIR}/${RUN_ID}_upstream_oracle.json"
mkdir -p "${LOG_DIR}"

set +e
(
set -e
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil.registry import method_names
from benchmarks.emotic_mlcil.runner import CORE_RUNTIME_VERSION

if CORE_RUNTIME_VERSION != "0.6.0":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if "l3a" not in method_names():
    raise RuntimeError("L3A is not registered")
print({"runtime": CORE_RUNTIME_VERSION, "methods": tuple(method_names())})
PY

echo "Running L3A/Core preflight tests..."
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary

echo "Running fixed-source L3A operator equivalence..."
oracle_args=(--upstream-root "${UPSTREAM_ROOT}" --output "${ORACLE_JSON}")
if [[ -n "${UPSTREAM_ARCHIVE}" ]]; then
  oracle_args+=(--upstream-archive "${UPSTREAM_ARCHIVE}")
fi
"${PYTHON}" "${SCRIPT_DIR}/compare_l3a_upstream_reference.py" \
  "${oracle_args[@]}"

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running L3A Task-0/analytic memory smoke on GPU ${GPU}..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_l3a_training.py" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --analytic-batch-size "${EVAL_BATCH_SIZE}"
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
  echo "L3A preflight failed with exit code ${PREFLIGHT_RC}" >&2
  echo "Preflight log: ${PREFLIGHT_LOG}" >&2
  exit "${PREFLIGHT_RC}"
fi

printf -v command \
  'cd %q && RUN_ID=%q SEED=0 GPU=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q OUTPUT_ROOT=%q REPORTING_SPLIT=val TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; package_code=not_run; if [[ "$code" -eq 0 ]]; then %q %q --run-root %q --run-id %q --expected-bundles 1 --extra %q; package_code=$?; fi; echo L3A_EXIT_CODE=$code; echo DOWNLOAD_PACKAGE_EXIT_CODE=$package_code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" \
  "${CLIP_MODEL_PATH}" "${RUN_ROOT}" "${TRAIN_BATCH_SIZE}" \
  "${EVAL_BATCH_SIZE}" "${WORKERS}" \
  "${SCRIPT_DIR}/run_l3a_baseline.sh" "${LOG_DIR}/${RUN_ID}.log" \
  "${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  "${RUN_ROOT}" "${RUN_ID}" "${ORACLE_JSON}"

tmux new-session -d -s "${SESSION}" -n "l3a_seed0_g${GPU}" "${command}"
echo "Started L3A validation session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: ${LOG_DIR}/${RUN_ID}.log"
echo "Preflight log: ${PREFLIGHT_LOG}"
echo "After success, download only: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "And checksum: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
