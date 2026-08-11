#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_cocoer_ft_formal_seed012}"
RUN_ID="${RUN_ID:-cocoer_ft_formal_seed012_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-1 2 3}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
RESNET_INIT="${COCOER_RESNET50_INIT:-${ROOT}/pretrained/cocoer/resnet50_imagenet1k_v1.pth}"
CLIP_RN50="${COCOER_CLIP_RN50:-${ROOT}/pretrained/clip/RN50.pt}"
HEAD_CACHE="${COCOER_HEAD_CACHE:-${ROOT}/pretrained/cocoer/emotic_head_boxes_v0.1.json}"
UPSTREAM_ROOT="${COCOER_UPSTREAM_ROOT:-/mnt/haoyuan/workspace/baseline_sources/cocoer_release_dac8fc1}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
OUTPUT_BASE="${COCOER_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/cocoer_ft_track_b_v0.1}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
LOCK="${CONFIGURATION_LOCKED_CONFIRMATION:?configuration lock is required}"
MIN_FREE_MIB="${COCOER_MIN_FREE_MIB:-18000}"

[[ "${LOCK}" == "COCOER_FT_TRACK_B_V0_1" ]] || { echo "Invalid lock" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "HEAD is not the frozen formal commit" >&2; exit 2;
}
[[ -z "$(git status --porcelain)" ]] || { echo "Formal worktree is dirty" >&2; exit 2; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "Session exists" >&2; exit 2; }
read -r -a gpu_values <<< "${GPUS}"
[[ "${#gpu_values[@]}" -eq 3 && "${gpu_values[0]}" != "${gpu_values[1]}" \
  && "${gpu_values[0]}" != "${gpu_values[2]}" \
  && "${gpu_values[1]}" != "${gpu_values[2]}" ]] || {
  echo "GPUS must contain three distinct IDs" >&2; exit 2;
}
for asset in "${RESNET_INIT}" "${CLIP_RN50}" "${HEAD_CACHE}"; do
  [[ -s "${asset}" ]] || { echo "Missing asset: ${asset}" >&2; exit 2; }
done
[[ -s "${UPSTREAM_ROOT}/models_sw.py" ]] || { echo "Missing upstream source" >&2; exit 2; }

for gpu in "${gpu_values[@]}"; do
  free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ "${free}" =~ ^[0-9]+$ && "${free}" -ge "${MIN_FREE_MIB}" ]] || {
    echo "GPU ${gpu} has ${free:-unknown} MiB free; ${MIN_FREE_MIB} required" >&2; exit 2;
  }
  echo "GPU ${gpu}: ${free} MiB free"
done

RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run output exists: ${RUN_ROOT}" >&2; exit 2; }
PREFLIGHT_DIR="${RUN_ROOT}/preflight_logs"
LOG_DIR="${OUTPUT_BASE}/_launcher_logs"
PREFLIGHT_LOG="${LOG_DIR}/${RUN_ID}_preflight.log"
LAUNCHER_LOG="${LOG_DIR}/${RUN_ID}.log"
mkdir -p "${PREFLIGHT_DIR}" "${LOG_DIR}"

set +e
(
set -e
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter tests.test_ddp_prompt_free_auxiliary
"${PYTHON}" "${SCRIPT_DIR}/compare_cocoer_upstream_reference.py" \
  --upstream-root "${UPSTREAM_ROOT}" --output "${PREFLIGHT_DIR}/upstream_oracle.json"
"${PYTHON}" "${SCRIPT_DIR}/audit_cocoer_assets.py" \
  --data-root "${DATA_ROOT}" --resnet50-init "${RESNET_INIT}" \
  --clip-rn50 "${CLIP_RN50}" --head-cache "${HEAD_CACHE}" \
  --output "${PREFLIGHT_DIR}/asset_audit.json"
CUDA_VISIBLE_DEVICES="${gpu_values[0]}" "${PYTHON}" \
  "${SCRIPT_DIR}/smoke_cocoer_ft_training.py" --protocol "${PROTOCOL}" \
  --resnet50-init "${RESNET_INIT}" --clip-rn50 "${CLIP_RN50}" \
  --head-cache "${HEAD_CACHE}" --batch-size 64 \
  --output "${PREFLIGHT_DIR}/memory_smoke.json"
) 2>&1 | tee "${PREFLIGHT_LOG}"
preflight_rc="${PIPESTATUS[0]}"
set -e
[[ "${preflight_rc}" -eq 0 ]] || { echo "Formal preflight failed" >&2; exit "${preflight_rc}"; }
cp "${PREFLIGHT_LOG}" "${PREFLIGHT_DIR}/preflight.log"

printf -v command \
  'cd %q && RUN_ID=%q GPUS=%q PYTHON=%q DATA_ROOT=%q COCOER_RESNET50_INIT=%q COCOER_CLIP_RN50=%q COCOER_HEAD_CACHE=%q PROTOCOL=%q RUN_OUTPUT_ROOT=%q EXPECTED_GIT_COMMIT=%q CONFIGURATION_LOCKED_CONFIRMATION=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; echo COCOER_FT_FORMAL_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPUS}" "${PYTHON}" "${DATA_ROOT}" \
  "${RESNET_INIT}" "${CLIP_RN50}" "${HEAD_CACHE}" "${PROTOCOL}" \
  "${RUN_ROOT}" "${EXPECTED_GIT_COMMIT}" "${LOCK}" \
  "${SCRIPT_DIR}/run_cocoer_ft_formal_seed012.sh" "${LAUNCHER_LOG}"

tmux new-session -d -s "${SESSION}" -n "cocoer_ft_seed012" "${command}"
echo "Started locked CocoER-FT formal session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Assignments: seed0->GPU${gpu_values[0]}, seed1->GPU${gpu_values[1]}, seed2->GPU${gpu_values[2]}"
echo "Archive: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
