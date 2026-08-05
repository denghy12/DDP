#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:?RUN_OUTPUT_ROOT is required}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
GPU_MEMORY_BUDGET_MIB="${GPU_MEMORY_BUDGET_MIB:-20000}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b4c2.yaml}"
REPLAY_CONTRACT="${REPLAY_CONTRACT:-${ROOT}/configs/emotic_mlcil/replay_b4c2_20c_v0.1.yaml}"
DERPP_REPLAY_CONTRACT="${DERPP_REPLAY_CONTRACT:-${ROOT}/configs/emotic_mlcil/replay_derpp_b4c2_20c_v0.1.yaml}"
AGCN_WORD_EMBEDDINGS="${AGCN_WORD_EMBEDDINGS:-${ROOT}/pretrained/agcn/emotic_glove_6b_300d.json}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "EMOTIC_B4C2_12BASELINE_V0_1" ]] || {
  echo "Invalid B4-C2 configuration-lock confirmation" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || {
  echo "B4-C2 sweep is not running the frozen commit" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "B4-C2 sweep requires a clean Git worktree" >&2
  exit 2
}

read -r -a gpu_values <<< "${GPUS}"
[[ "${#gpu_values[@]}" -eq 8 ]] || {
  echo "GPUS must contain exactly eight physical GPU indices" >&2
  exit 2
}

mkdir -p "${RUN_OUTPUT_ROOT}"
export RUN_ID RUN_OUTPUT_ROOT EXPECTED_GIT_COMMIT
export PYTHON DATA_ROOT CLIP_MODEL_PATH PROTOCOL
export REPLAY_CONTRACT DERPP_REPLAY_CONTRACT AGCN_WORD_EMBEDDINGS

"${PYTHON}" "${SCRIPT_DIR}/schedule_b4c2_12baseline_8gpu.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --job-script "${SCRIPT_DIR}/run_b4c2_baseline_job.sh" \
  --gpus "${gpu_values[@]}" \
  --slots-per-gpu "${SLOTS_PER_GPU}" \
  --gpu-memory-budget-mib "${GPU_MEMORY_BUDGET_MIB}"

SUMMARY_JSON="${RUN_OUTPUT_ROOT}/b4c2_seed012_summary.json"
SUMMARY_MD="${RUN_OUTPUT_ROOT}/b4c2_seed012_summary.md"
"${PYTHON}" "${SCRIPT_DIR}/summarize_b4c2_sweep.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --expected-git-commit "${EXPECTED_GIT_COMMIT}" \
  --output "${SUMMARY_JSON}" \
  --markdown-output "${SUMMARY_MD}"

"${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  --run-root "${RUN_OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --expected-bundles 36 \
  --extra "${RUN_OUTPUT_ROOT}/sweep_plan.json" \
  --extra "${RUN_OUTPUT_ROOT}/scheduler_events.jsonl" \
  --extra "${RUN_OUTPUT_ROOT}/runtime_state" \
  --extra "${RUN_OUTPUT_ROOT}/job_logs" \
  --extra "${SUMMARY_JSON}" \
  --extra "${SUMMARY_MD}"

echo "All 12 B4-C2 baselines × three seeds completed and validated"
echo "Summary: ${SUMMARY_MD}"
echo "Archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
