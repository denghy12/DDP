#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_emotionclip_ft_seed0_val}"
RUN_ID="${RUN_ID:-emotionclip_ft_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CHECKPOINT="${EMOTIONCLIP_CHECKPOINT:-${ROOT}/pretrained/emotionclip/emotionclip_official.pt}"
UPSTREAM_ROOT="${EMOTIONCLIP_UPSTREAM_ROOT:-/mnt/haoyuan/workspace/baseline_sources/emotionclip_release_ca142dd}"
OUTPUT_BASE="${EMOTIONCLIP_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/emotionclip_ft_track_b_v0.1}"
TRAIN_BATCH_SIZE="${EMOTIONCLIP_TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EMOTIONCLIP_EVAL_BATCH_SIZE:-128}"
WORKERS="${EMOTIONCLIP_WORKERS:-0}"
REQUIRE_CLEAN="${EMOTIONCLIP_REQUIRE_CLEAN:-1}"
RUN_GPU_SMOKE="${EMOTIONCLIP_RUN_GPU_SMOKE:-1}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
[[ "${TRAIN_BATCH_SIZE}" == "64" ]] || { echo "Registered EmotionCLIP-FT batch size is 64" >&2; exit 2; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${CHECKPOINT}" ]] || {
  echo "Missing official EmotionCLIP checkpoint: ${CHECKPOINT}" >&2
  echo "Download the checkpoint linked by the fixed upstream README; do not substitute OpenAI CLIP." >&2
  exit 2
}
[[ -s "${UPSTREAM_ROOT}/src/models/base.py" ]] || { echo "Missing fixed EmotionCLIP source: ${UPSTREAM_ROOT}" >&2; exit 2; }
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "EmotionCLIP-FT validation requires a clean Git worktree" >&2
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
if CORE_RUNTIME_VERSION != "0.11.0":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if "emotionclip_ft" not in method_names():
    raise RuntimeError("EmotionCLIP-FT is not registered")
print({"runtime": CORE_RUNTIME_VERSION, "methods": tuple(method_names())})
PY
echo "Running EmotionCLIP-FT/Core preflight tests..."
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest tests.test_emotic_task_adapter_bank tests.test_ddp_internal_adapter tests.test_ddp_prompt_free_auxiliary
echo "Running immutable EmotionCLIP source/operator audit..."
"${PYTHON}" "${SCRIPT_DIR}/compare_emotionclip_upstream_reference.py" --upstream-root "${UPSTREAM_ROOT}" --output "${ORACLE_JSON}"
if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running EmotionCLIP-FT memory smoke on GPU ${GPU}..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT_DIR}/smoke_emotionclip_ft_training.py" --checkpoint "${CHECKPOINT}" --batch-size "${TRAIN_BATCH_SIZE}"
fi
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
  echo "EmotionCLIP-FT preflight failed with exit code ${PREFLIGHT_RC}" >&2
  exit "${PREFLIGHT_RC}"
fi

printf -v command \
  'cd %q && RUN_ID=%q SEED=0 GPU=%q PYTHON=%q DATA_ROOT=%q EMOTIONCLIP_CHECKPOINT=%q OUTPUT_ROOT=%q REPORTING_SPLIT=val TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; package_code=not_run; if [[ "$code" -eq 0 ]]; then %q %q --run-root %q --run-id %q --expected-bundles 1 --extra %q; package_code=$?; fi; echo EMOTIONCLIP_FT_EXIT_CODE=$code; echo DOWNLOAD_PACKAGE_EXIT_CODE=$package_code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" "${CHECKPOINT}" "${RUN_ROOT}" \
  "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${WORKERS}" \
  "${SCRIPT_DIR}/run_emotionclip_ft_baseline.sh" "${LOG_DIR}/${RUN_ID}.log" \
  "${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" "${RUN_ROOT}" "${RUN_ID}" "${ORACLE_JSON}"

tmux new-session -d -s "${SESSION}" -n "emotionclip_ft_seed0_g${GPU}" "${command}"
echo "Started EmotionCLIP-FT Track-B seed-0 validation: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Download: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
