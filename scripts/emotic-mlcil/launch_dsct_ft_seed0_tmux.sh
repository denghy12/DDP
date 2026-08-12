#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
SESSION="${SESSION:-emotic_dsct_ft_seed0_val}"
RUN_ID="${RUN_ID:-dsct_ft_seed0_val_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-4,5,6,7}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
SOURCE_ROOT="${DSCT_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/dsct_release_8b0fe36}"
PRETRAINED="${DSCT_PRETRAINED_WEIGHTS:-${SOURCE_ROOT}/r50_deformable_detr-checkpoint.pth}"
OUTPUT_BASE="${DSCT_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/dsct_ft_track_b_v0.1}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
RUN_GPU_SMOKE="${DSCT_RUN_GPU_SMOKE:-1}"
CPUSET="${DSCT_CPUSET:-36-47,108-119}"

[[ "${GPU}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]] || {
  echo "Fast DSCT validation requires four physical GPU IDs" >&2
  exit 2
}

tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 2; }
[[ -x "${PYTHON}" && -d "${DATA_ROOT}" && -s "${PRETRAINED}" ]] || { echo "Missing runtime, data, or DSCT pretraining" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "DSCT validation requires a clean worktree" >&2; exit 2; }
RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"; LOG_DIR="${OUTPUT_BASE}/_launcher_logs"; mkdir -p "${LOG_DIR}"
ORACLE="${LOG_DIR}/${RUN_ID}_upstream_oracle.json"; PREFLIGHT="${LOG_DIR}/${RUN_ID}_preflight.log"
(
  set -e
  "${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
  "${PYTHON}" -m unittest tests.test_emotic_task_adapter_bank tests.test_ddp_internal_adapter tests.test_ddp_prompt_free_auxiliary
  "${PYTHON}" "${SCRIPT_DIR}/compare_dsct_upstream_reference.py" --upstream-root "${SOURCE_ROOT}" --output "${ORACLE}"
  if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT_DIR}/smoke_dsct_ft_training.py" \
      --source-root "${SOURCE_ROOT}" --pretrained-weights "${PRETRAINED}" \
      --batch-size 4 --height 800 --width 1333 --warmup-steps 1 --benchmark-steps 2
  fi
) 2>&1 | tee "${PREFLIGHT}"

printf -v command \
  'cd %q && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MALLOC_ARENA_MAX=4 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && RUN_ID=%q SEED=0 GPU=%q PYTHON=%q DATA_ROOT=%q DSCT_SOURCE_ROOT=%q DSCT_PRETRAINED_WEIGHTS=%q OUTPUT_ROOT=%q REPORTING_SPLIT=val TRAIN_BATCH_SIZE=4 EVAL_BATCH_SIZE=4 WORKERS=2 DSCT_CPUSET=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; package_code=not_run; if [[ "$code" -eq 0 ]]; then %q %q --run-root %q --run-id %q --expected-bundles 1 --launcher-log %q --extra %q --extra %q; package_code=$?; fi; echo DSCT_FT_EXIT_CODE=$code; echo DOWNLOAD_PACKAGE_EXIT_CODE=$package_code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" "${SOURCE_ROOT}" "${PRETRAINED}" "${RUN_ROOT}" \
  "${CPUSET}" "${SCRIPT_DIR}/run_dsct_ft_baseline.sh" "${LOG_DIR}/${RUN_ID}.log" "${PYTHON}" "${SCRIPT_DIR}/package_benchmark_download.py" \
  "${RUN_ROOT}" "${RUN_ID}" "${LOG_DIR}/${RUN_ID}.log" "${ORACLE}" "${PREFLIGHT}"
tmux new-session -d -s "${SESSION}" -n "dsct_ft_seed0_g${GPU}" "${command}"
echo "Started DSCT-FT Track-B seed-0 validation: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Physical GPUs: ${GPU}; CPU affinity: ${CPUSET}; workers: 2; eval batch: 4"
echo "Attach: tmux attach -t ${SESSION}"
echo "Download: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
