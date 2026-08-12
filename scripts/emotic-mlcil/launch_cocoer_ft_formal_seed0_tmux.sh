#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_cocoer_ft_gpu_v02_seed0}"
RUN_ID="${RUN_ID:-cocoer_ft_gpu_v02_formal_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-2}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
ASSET_ROOT="${COCOER_ASSET_ROOT:-/mnt/haoyuan/workspace/CODE_DDP-benchmark-cocoer-ft}"
RESNET_INIT="${COCOER_RESNET50_INIT:-${ASSET_ROOT}/pretrained/cocoer/resnet50_imagenet1k_v1.pth}"
CLIP_RN50="${COCOER_CLIP_RN50:-${ASSET_ROOT}/pretrained/clip/RN50.pt}"
HEAD_CACHE="${COCOER_HEAD_CACHE:-${ASSET_ROOT}/pretrained/cocoer/emotic_head_boxes_v0.1.json}"
OUTPUT_BASE="${COCOER_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/cocoer_ft_track_b_v0.2}"
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
LOG_DIR="${OUTPUT_BASE}/_launcher_logs"
LOG="${LOG_DIR}/${RUN_ID}.log"
SMOKE_JSON="${COCOER_SMOKE_JSON:-${OUTPUT_BASE}/_preflight/cocoer_gpu_v02_full_batch64_smoke.json}"
BENCHMARK_JSON="${COCOER_BENCHMARK_JSON:-${OUTPUT_BASE}/_preflight/cocoer_gpu_preprocess_benchmark_candidate_v2.json}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing EMOTIC root: ${DATA_ROOT}" >&2; exit 2; }
for asset in "${RESNET_INIT}" "${CLIP_RN50}" "${HEAD_CACHE}" "${SMOKE_JSON}" "${BENCHMARK_JSON}"; do
  [[ -s "${asset}" ]] || { echo "Missing frozen CocoER artifact: ${asset}" >&2; exit 2; }
done
[[ -z "$(git status --porcelain)" ]] || { echo "Formal CocoER run requires a clean worktree" >&2; exit 2; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists: ${RUN_ROOT}" >&2; exit 2; }

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')"
[[ "${free_mib}" =~ ^[0-9]+$ && "${free_mib}" -ge 18000 ]] || {
  echo "GPU ${GPU} has only ${free_mib:-unknown} MiB free; require at least 18000 MiB" >&2
  exit 2
}

mkdir -p "${LOG_DIR}"
printf -v command \
  'cd %q && RUN_ID=%q SEED=0 GPU=%q PYTHON=%q DATA_ROOT=%q COCOER_RESNET50_INIT=%q COCOER_CLIP_RN50=%q COCOER_HEAD_CACHE=%q OUTPUT_ROOT=%q REPORTING_SPLIT=test TRAIN_BATCH_SIZE=64 EVAL_BATCH_SIZE=16 WORKERS=2 CONFIGURATION_LOCKED_CONFIRMATION=COCOER_FT_TRACK_B_V0_2 bash %q 2>&1 | tee %q; run_code=${PIPESTATUS[0]}; package_code=not_run; if [[ "$run_code" -eq 0 ]]; then %q %q --run-root %q --run-id %q --expected-bundles 1 --extra %q --extra %q; package_code=$?; fi; echo COCOER_FORMAL_EXIT_CODE=$run_code; echo DOWNLOAD_PACKAGE_EXIT_CODE=$package_code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" \
  "${RESNET_INIT}" "${CLIP_RN50}" "${HEAD_CACHE}" "${RUN_ROOT}" \
  "${SCRIPT_DIR}/run_cocoer_ft_baseline.sh" "${LOG}" \
  "${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  "${RUN_ROOT}" "${RUN_ID}" "${SMOKE_JSON}" "${BENCHMARK_JSON}"

tmux new-session -d -s "${SESSION}" -n "seed0_gpu${GPU}" "${command}"
echo "Started CocoER-FT GPU-preprocessing v0.2 formal seed 0"
echo "SESSION=${SESSION}"
echo "RUN_ID=${RUN_ID}"
echo "GPU=${GPU}"
echo "LOG=${LOG}"
echo "DOWNLOAD=${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "CHECKSUM=${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
