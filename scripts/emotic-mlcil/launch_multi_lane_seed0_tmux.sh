#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_multi_lane_seed0_val}"
RUN_ID="${RUN_ID:-multi_lane_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-/mnt/haoyuan/workspace/multi-lane-main}"
UPSTREAM_ARCHIVE="${UPSTREAM_ARCHIVE:-}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/multi_lane_track_a_v0.1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid RUN_ID" >&2
  exit 2
}
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
[[ "${TRAIN_BATCH_SIZE}" == "64" ]] || {
  echo "Registered MULTI-LANE Track-A train batch size is 64" >&2
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
[[ -s "${UPSTREAM_ROOT}/multi_lane/blocks.py" ]] || {
  echo "Missing MULTI-LANE blocks.py: ${UPSTREAM_ROOT}" >&2
  exit 2
}
[[ -s "${UPSTREAM_ROOT}/multi_lane/vision_transformer.py" ]] || {
  echo "Missing MULTI-LANE vision_transformer.py: ${UPSTREAM_ROOT}" >&2
  exit 2
}
if [[ -n "${UPSTREAM_ARCHIVE}" && ! -s "${UPSTREAM_ARCHIVE}" ]]; then
  echo "Configured MULTI-LANE archive is missing: ${UPSTREAM_ARCHIVE}" >&2
  exit 2
fi
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "MULTI-LANE validation requires a clean Git worktree" >&2
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

if CORE_RUNTIME_VERSION != "0.5.0":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if "multi_lane" not in method_names():
    raise RuntimeError("MULTI-LANE is not registered")
print({"runtime": CORE_RUNTIME_VERSION, "methods": tuple(method_names())})
PY

echo "Running MULTI-LANE/Core preflight tests..."
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary

echo "Running fixed-source MULTI-LANE operator equivalence..."
oracle_args=(
  --upstream-root "${UPSTREAM_ROOT}"
  --output "${ORACLE_JSON}"
)
if [[ -n "${UPSTREAM_ARCHIVE}" ]]; then
  oracle_args+=(--upstream-archive "${UPSTREAM_ARCHIVE}")
fi
"${PYTHON}" "${SCRIPT_DIR}/compare_multi_lane_upstream_reference.py" \
  "${oracle_args[@]}"

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running worst-task MULTI-LANE memory smoke on GPU ${GPU}..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_multi_lane_training.py" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}"
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
  echo "MULTI-LANE preflight failed with exit code ${PREFLIGHT_RC}" >&2
  echo "Preflight log: ${PREFLIGHT_LOG}" >&2
  exit "${PREFLIGHT_RC}"
fi

printf -v command \
  'cd %q && RUN_ID=%q SEED=0 GPU=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q OUTPUT_ROOT=%q REPORTING_SPLIT=val TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; package_code=not_run; if [[ "$code" -eq 0 ]]; then %q %q --run-root %q --run-id %q --expected-bundles 1 --extra %q; package_code=$?; fi; echo MULTI_LANE_EXIT_CODE=$code; echo DOWNLOAD_PACKAGE_EXIT_CODE=$package_code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" \
  "${CLIP_MODEL_PATH}" "${RUN_ROOT}" "${TRAIN_BATCH_SIZE}" \
  "${EVAL_BATCH_SIZE}" "${WORKERS}" \
  "${SCRIPT_DIR}/run_multi_lane_baseline.sh" "${LOG_DIR}/${RUN_ID}.log" \
  "${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  "${RUN_ROOT}" "${RUN_ID}" "${ORACLE_JSON}"

tmux new-session -d -s "${SESSION}" -n "multi_lane_seed0_g${GPU}" "${command}"
echo "Started MULTI-LANE validation session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: ${LOG_DIR}/${RUN_ID}.log"
echo "Preflight log: ${PREFLIGHT_LOG}"
echo "After success, download only: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "And checksum: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
