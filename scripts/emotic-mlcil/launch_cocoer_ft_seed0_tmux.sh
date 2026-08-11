#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_cocoer_ft_seed0_val}"
RUN_ID="${RUN_ID:-cocoer_ft_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
RESNET_INIT="${COCOER_RESNET50_INIT:-${ROOT}/pretrained/cocoer/resnet50_imagenet1k_v1.pth}"
CLIP_RN50="${COCOER_CLIP_RN50:-${ROOT}/pretrained/clip/RN50.pt}"
HEAD_CACHE="${COCOER_HEAD_CACHE:-${ROOT}/pretrained/cocoer/emotic_head_boxes_v0.1.json}"
UPSTREAM_ROOT="${COCOER_UPSTREAM_ROOT:-/mnt/haoyuan/workspace/baseline_sources/cocoer_release_dac8fc1}"
OUTPUT_BASE="${COCOER_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/cocoer_ft_track_b_v0.1}"
TRAIN_BATCH_SIZE="${COCOER_TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${COCOER_EVAL_BATCH_SIZE:-16}"
WORKERS="${COCOER_WORKERS:-0}"
REQUIRE_CLEAN="${COCOER_REQUIRE_CLEAN:-1}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
[[ "${TRAIN_BATCH_SIZE}" == "64" ]] || { echo "Registered source batch size is 64" >&2; exit 2; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session already exists: ${SESSION}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC root: ${DATA_ROOT}" >&2; exit 2; }
for asset in "${RESNET_INIT}" "${CLIP_RN50}" "${HEAD_CACHE}"; do
  [[ -s "${asset}" ]] || { echo "Missing CocoER asset: ${asset}" >&2; exit 2; }
done
[[ -s "${UPSTREAM_ROOT}/models_sw.py" ]] || { echo "Missing fixed CocoER source: ${UPSTREAM_ROOT}" >&2; exit 2; }
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "CocoER-FT validation requires a clean Git worktree" >&2
  exit 2
fi

RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_DIR="${OUTPUT_BASE}/_launcher_logs"
PREFLIGHT_LOG="${LOG_DIR}/${RUN_ID}_preflight.log"
ORACLE_JSON="${LOG_DIR}/${RUN_ID}_upstream_oracle.json"
ASSET_AUDIT_JSON="${LOG_DIR}/${RUN_ID}_asset_audit.json"
MEMORY_SMOKE_JSON="${LOG_DIR}/${RUN_ID}_memory_smoke.json"
mkdir -p "${LOG_DIR}"

set +e
(
set -e
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil.registry import method_names
from benchmarks.emotic_mlcil.runner import CORE_RUNTIME_VERSION
if "cocoer_ft" not in method_names():
    raise RuntimeError("CocoER-FT is not registered")
print({"runtime": CORE_RUNTIME_VERSION, "methods": tuple(method_names())})
PY
echo "Running CocoER-FT/Core preflight tests..."
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary
echo "Running immutable CocoER source/operator audit..."
"${PYTHON}" "${SCRIPT_DIR}/compare_cocoer_upstream_reference.py" \
  --upstream-root "${UPSTREAM_ROOT}" --output "${ORACLE_JSON}"
echo "Running CocoER native-asset and full head-cache audit..."
"${PYTHON}" "${SCRIPT_DIR}/audit_cocoer_assets.py" \
  --data-root "${DATA_ROOT}" \
  --resnet50-init "${RESNET_INIT}" \
  --clip-rn50 "${CLIP_RN50}" \
  --head-cache "${HEAD_CACHE}" \
  --output "${ASSET_AUDIT_JSON}"
echo "Running full-path CocoER-FT batch-${TRAIN_BATCH_SIZE} CUDA memory smoke on GPU ${GPU}..."
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  "${SCRIPT_DIR}/smoke_cocoer_ft_training.py" \
  --protocol "${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml" \
  --resnet50-init "${RESNET_INIT}" \
  --clip-rn50 "${CLIP_RN50}" \
  --head-cache "${HEAD_CACHE}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --output "${MEMORY_SMOKE_JSON}"
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
  echo "CocoER-FT preflight failed with exit code ${PREFLIGHT_RC}" >&2
  exit "${PREFLIGHT_RC}"
fi

printf -v command \
  'cd %q && RUN_ID=%q SEED=0 GPU=%q PYTHON=%q DATA_ROOT=%q COCOER_RESNET50_INIT=%q COCOER_CLIP_RN50=%q COCOER_HEAD_CACHE=%q OUTPUT_ROOT=%q REPORTING_SPLIT=val TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; package_code=not_run; if [[ "$code" -eq 0 ]]; then %q %q --run-root %q --run-id %q --expected-bundles 1 --extra %q --extra %q --extra %q; package_code=$?; fi; echo COCOER_FT_EXIT_CODE=$code; echo DOWNLOAD_PACKAGE_EXIT_CODE=$package_code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" \
  "${RESNET_INIT}" "${CLIP_RN50}" "${HEAD_CACHE}" "${RUN_ROOT}" \
  "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${WORKERS}" \
  "${SCRIPT_DIR}/run_cocoer_ft_baseline.sh" "${LOG_DIR}/${RUN_ID}.log" \
  "${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  "${RUN_ROOT}" "${RUN_ID}" "${ORACLE_JSON}" "${ASSET_AUDIT_JSON}" \
  "${MEMORY_SMOKE_JSON}"

tmux new-session -d -s "${SESSION}" -n "cocoer_ft_seed0_g${GPU}" "${command}"
echo "Started CocoER-FT Track-B validation session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Download: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
echo "Memory smoke: ${MEMORY_SMOKE_JSON}"
