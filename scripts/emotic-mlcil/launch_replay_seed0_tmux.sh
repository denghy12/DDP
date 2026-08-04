#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

METHOD="${METHOD:-er}"
SESSION="${SESSION:-emotic_${METHOD}_seed0_val}"
RUN_ID="${RUN_ID:-${METHOD}_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/replay_track_a_v0.1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"
PRS_SOURCE_ROOT="${PRS_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/PRS-master}"
export PRS_SOURCE_ROOT

[[ "${METHOD}" == "er" || "${METHOD}" == "prs" ]] || {
  echo "METHOD must be er or prs" >&2
  exit 2
}
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid RUN_ID" >&2
  exit 2
}
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
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
if [[ "${METHOD}" == "prs" && ! -d "${PRS_SOURCE_ROOT}" ]]; then
  echo "Missing fixed PRS source: ${PRS_SOURCE_ROOT}" >&2
  exit 2
fi
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "Replay validation requires a clean Git worktree" >&2
  exit 2
fi

RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_DIR="${OUTPUT_BASE}/_launcher_logs"
PREFLIGHT_LOG="${LOG_DIR}/${RUN_ID}_preflight.log"
ORACLE_JSON="${RUN_ROOT}/prs_upstream_equivalence.json"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

set +e
(
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil.registry import method_names
from benchmarks.emotic_mlcil.runner import CORE_RUNTIME_VERSION
if CORE_RUNTIME_VERSION != "0.9.0":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
required = {"er", "prs"}
if not required.issubset(set(method_names())):
    raise RuntimeError(f"Missing replay methods: {required - set(method_names())}")
print({"runtime": CORE_RUNTIME_VERSION, "methods": tuple(method_names())})
PY
echo "Running Replay/Core preflight tests..."
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary
if [[ "${METHOD}" == "prs" ]]; then
  echo "Running immutable PRS source/operator equivalence..."
  "${PYTHON}" "${SCRIPT_DIR}/compare_prs_upstream_reference.py" \
    --upstream-root "${PRS_SOURCE_ROOT}" > "${ORACLE_JSON}"
fi
if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running current+replay CLIP memory smoke on GPU ${GPU}..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_replay_visual_training.py" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --batch-size "${TRAIN_BATCH_SIZE}"
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
  echo "Replay preflight failed with exit code ${PREFLIGHT_RC}" >&2
  echo "Preflight log: ${PREFLIGHT_LOG}" >&2
  exit "${PREFLIGHT_RC}"
fi

extra_args=()
if [[ "${METHOD}" == "prs" ]]; then
  extra_args=(--extra "${ORACLE_JSON}")
fi
printf -v command \
  'cd %q && METHOD=%q RUN_ID=%q SEED=0 GPU=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q OUTPUT_ROOT=%q REPORTING_SPLIT=val TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; package_code=not_run; if [[ "$code" -eq 0 ]]; then %q %q --run-root %q --run-id %q --expected-bundles 1' \
  "${ROOT}" "${METHOD}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" \
  "${CLIP_MODEL_PATH}" "${RUN_ROOT}" "${TRAIN_BATCH_SIZE}" \
  "${EVAL_BATCH_SIZE}" "${WORKERS}" "${SCRIPT_DIR}/run_replay_baseline.sh" \
  "${LOG_DIR}/${RUN_ID}.log" "${PYTHON}" \
  "${SCRIPT_DIR}/package_benchmark_download.py" "${RUN_ROOT}" "${RUN_ID}"
for argument in "${extra_args[@]}"; do
  printf -v quoted ' %q' "${argument}"
  command+="${quoted}"
done
command+='; package_code=$?; fi; echo REPLAY_EXIT_CODE=$code; echo DOWNLOAD_PACKAGE_EXIT_CODE=$package_code; exec bash'

tmux new-session -d -s "${SESSION}" -n "${METHOD}_seed0_g${GPU}" "${command}"
echo "Started ${METHOD^^} validation session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: ${LOG_DIR}/${RUN_ID}.log"
echo "Preflight log: ${PREFLIGHT_LOG}"
echo "After success, download only: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "And checksum: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
