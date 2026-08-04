#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_agcn_seed0_val}"
RUN_ID="${RUN_ID:-agcn_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
if [[ ! -s "${CLIP_MODEL_PATH}" && -s "/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt" ]]; then
  CLIP_MODEL_PATH="/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt"
fi
AGCN_WORD_EMBEDDINGS="${AGCN_WORD_EMBEDDINGS:-${ROOT}/pretrained/agcn/emotic_glove_6b_300d.json}"
UPSTREAM_ROOT="${AGCN_UPSTREAM_ROOT:-/mnt/haoyuan/workspace/baseline_sources/agcn_release_3afe2ec}"
UPSTREAM_ARCHIVE="${AGCN_UPSTREAM_ARCHIVE:-/mnt/haoyuan/workspace/baseline_sources/agcn_release_3afe2ec.tar.gz}"
OUTPUT_BASE="${AGCN_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/agcn_track_a_v0.1}"
TRAIN_BATCH_SIZE="${AGCN_TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${AGCN_EVAL_BATCH_SIZE:-32}"
WORKERS="${AGCN_WORKERS:-0}"
REQUIRE_CLEAN="${AGCN_REQUIRE_CLEAN:-1}"
RUN_GPU_SMOKE="${AGCN_RUN_GPU_SMOKE:-1}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
[[ "${TRAIN_BATCH_SIZE}" == "8" ]] || { echo "Registered AGCN Track-A train batch size is 8" >&2; exit 2; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${CLIP_MODEL_PATH}" ]] || { echo "Missing CLIP checkpoint: ${CLIP_MODEL_PATH}" >&2; exit 2; }
[[ -s "${AGCN_WORD_EMBEDDINGS}" ]] || {
  echo "Missing AGCN GloVe asset: ${AGCN_WORD_EMBEDDINGS}" >&2
  echo "Prepare it with scripts/emotic-mlcil/prepare_agcn_glove_embeddings.py" >&2
  exit 2
}
[[ -s "${UPSTREAM_ROOT}/AGCN-LML/GCN.py" ]] || { echo "Missing fixed AGCN source: ${UPSTREAM_ROOT}" >&2; exit 2; }
[[ -s "${UPSTREAM_ARCHIVE}" ]] || { echo "Missing fixed AGCN archive: ${UPSTREAM_ARCHIVE}" >&2; exit 2; }
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "AGCN validation requires a clean Git worktree" >&2
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
if CORE_RUNTIME_VERSION != "0.8.0":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if "agcn" not in method_names():
    raise RuntimeError("AGCN is not registered")
print({"runtime": CORE_RUNTIME_VERSION, "methods": tuple(method_names())})
PY

echo "Running AGCN/Core preflight tests..."
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary

echo "Running immutable AGCN source/operator equivalence..."
"${PYTHON}" "${SCRIPT_DIR}/compare_agcn_upstream_reference.py" \
  --upstream-root "${UPSTREAM_ROOT}" \
  --upstream-archive "${UPSTREAM_ARCHIVE}" \
  --output "${ORACLE_JSON}"

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running worst-case AGCN teacher/Adam memory smoke on GPU ${GPU}..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_agcn_training.py" \
    --clip-model-path "${CLIP_MODEL_PATH}" --batch-size "${TRAIN_BATCH_SIZE}"
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
  echo "AGCN preflight failed with exit code ${PREFLIGHT_RC}" >&2
  echo "Preflight log: ${PREFLIGHT_LOG}" >&2
  exit "${PREFLIGHT_RC}"
fi

printf -v command \
  'cd %q && RUN_ID=%q SEED=0 GPU=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q AGCN_WORD_EMBEDDINGS=%q OUTPUT_ROOT=%q REPORTING_SPLIT=val TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; package_code=not_run; if [[ "$code" -eq 0 ]]; then %q %q --run-root %q --run-id %q --expected-bundles 1 --extra %q; package_code=$?; fi; echo AGCN_EXIT_CODE=$code; echo DOWNLOAD_PACKAGE_EXIT_CODE=$package_code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" \
  "${CLIP_MODEL_PATH}" "${AGCN_WORD_EMBEDDINGS}" "${RUN_ROOT}" \
  "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${WORKERS}" \
  "${SCRIPT_DIR}/run_agcn_baseline.sh" "${LOG_DIR}/${RUN_ID}.log" \
  "${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  "${RUN_ROOT}" "${RUN_ID}" "${ORACLE_JSON}"

tmux new-session -d -s "${SESSION}" -n "agcn_seed0_g${GPU}" "${command}"
echo "Started AGCN validation session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: ${LOG_DIR}/${RUN_ID}.log"
echo "Preflight log: ${PREFLIGHT_LOG}"
echo "After success, download only: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "And checksum: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
