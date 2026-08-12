#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SESSION="${SESSION:-emotic_dsct_ft_formal_seed0}"
RUN_ID="${RUN_ID:-dsct_ft_formal_seed0_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-5,6}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
SOURCE_ROOT="${DSCT_SOURCE_ROOT:-/mnt/haoyuan/workspace/baseline_sources/dsct_release_8b0fe36}"
PRETRAINED="${DSCT_PRETRAINED_WEIGHTS:-${SOURCE_ROOT}/r50_deformable_detr-checkpoint.pth}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
OUTPUT_BASE="${DSCT_OUTPUT_BASE:-/mnt/haoyuan/workspace/emotic_benchmark_runs/dsct_ft_track_b_v0.1}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?EXPECTED_GIT_COMMIT is required}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:?CONFIGURATION_LOCKED_CONFIRMATION is required}"
MIN_FREE_GPU_MIB="${DSCT_MIN_FREE_GPU_MIB:-12288}"
CPUSET="${DSCT_CPUSET:-36-47,108-119}"

[[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "DSCT_FT_TRACK_B_V0_4_FAST" ]] || { echo "Invalid DSCT-FT v0.4-fast configuration lock" >&2; exit 2; }
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+,[0-9]+$ ]] || { echo "Fast formal DSCT requires two physical GPU IDs" >&2; exit 2; }
[[ "${EXPECTED_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "EXPECTED_GIT_COMMIT must be a full SHA" >&2; exit 2; }
tmux has-session -t "${SESSION}" 2>/dev/null && { echo "tmux session exists: ${SESSION}" >&2; exit 2; }
[[ -x "${PYTHON}" && -d "${DATA_ROOT}" && -d "${SOURCE_ROOT}/models" && -s "${PRETRAINED}" ]] || { echo "Missing DSCT runtime, data, source, or pretraining" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "${EXPECTED_GIT_COMMIT}" ]] || { echo "HEAD is not the frozen commit" >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo "Formal run requires a clean worktree" >&2; exit 2; }

IFS=',' read -r -a GPU_IDS <<<"${GPU}"
[[ "${#GPU_IDS[@]}" -eq 1 || "${#GPU_IDS[@]}" -eq 4 ]] || { echo "DSCT requires one or four GPUs" >&2; exit 2; }
UNIQUE_GPU_COUNT="$(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l | tr -d '[:space:]')"
[[ "${UNIQUE_GPU_COUNT}" -eq "${#GPU_IDS[@]}" ]] || { echo "DSCT physical GPU IDs must be unique" >&2; exit 2; }
FREE_GPU_MIB=()
for gpu_id in "${GPU_IDS[@]}"; do
  free_mib="$(nvidia-smi -i "${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
  [[ "${free_mib}" =~ ^[0-9]+$ && "${free_mib}" -ge "${MIN_FREE_GPU_MIB}" ]] || { echo "GPU ${gpu_id} has ${free_mib:-unknown} MiB free; ${MIN_FREE_GPU_MIB} required" >&2; exit 2; }
  FREE_GPU_MIB+=("${free_mib}")
done

RUN_OUTPUT_ROOT="${OUTPUT_BASE}/${RUN_ID}"
[[ ! -e "${RUN_OUTPUT_ROOT}" ]] || { echo "Run output exists: ${RUN_OUTPUT_ROOT}" >&2; exit 2; }
PREFLIGHT_DIR="${RUN_OUTPUT_ROOT}/preflight_logs"
LOG_DIR="${OUTPUT_BASE}/_launcher_logs"
PREFLIGHT_LOG="${LOG_DIR}/${RUN_ID}_preflight.log"
LAUNCHER_LOG="${LOG_DIR}/${RUN_ID}.log"
ORACLE_JSON="${PREFLIGHT_DIR}/upstream_oracle.json"
mkdir -p "${PREFLIGHT_DIR}" "${LOG_DIR}"

set +e
(
  set -e
  "${PYTHON}" - <<'PY'
import json
import sys

import IPython
import MultiScaleDeformableAttention as msda
import scipy
import sklearn
import torch

print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "ipython": IPython.__version__,
    "scikit_learn": sklearn.__version__,
    "scipy": scipy.__version__,
    "ms_deform_attn_extension": msda.__file__,
}, indent=2))
PY
  "${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
  "${PYTHON}" -m unittest tests.test_emotic_task_adapter_bank tests.test_ddp_internal_adapter tests.test_ddp_prompt_free_auxiliary
  "${PYTHON}" "${SCRIPT_DIR}/compare_dsct_upstream_reference.py" --upstream-root "${SOURCE_ROOT}" --output "${ORACLE_JSON}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${SCRIPT_DIR}/smoke_dsct_ft_training.py" \
    --source-root "${SOURCE_ROOT}" --pretrained-weights "${PRETRAINED}" \
    --batch-size 4 --height 800 --width 1333 --warmup-steps 1 --benchmark-steps 2
) 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_RC="${PIPESTATUS[0]}"
set -e
[[ "${PREFLIGHT_RC}" -eq 0 ]] || { echo "DSCT-FT formal preflight failed: ${PREFLIGHT_RC}" >&2; exit "${PREFLIGHT_RC}"; }
cp "${PREFLIGHT_LOG}" "${PREFLIGHT_DIR}/preflight.log"

printf -v command \
  'cd %q && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MALLOC_ARENA_MAX=4 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && RUN_ID=%q GPU=%q PYTHON=%q DATA_ROOT=%q DSCT_SOURCE_ROOT=%q DSCT_PRETRAINED_WEIGHTS=%q PROTOCOL=%q RUN_OUTPUT_ROOT=%q EXPECTED_GIT_COMMIT=%q CONFIGURATION_LOCKED_CONFIRMATION=%q DSCT_CPUSET=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; echo DSCT_FT_FORMAL_SEED0_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${RUN_ID}" "${GPU}" "${PYTHON}" "${DATA_ROOT}" "${SOURCE_ROOT}" "${PRETRAINED}" "${PROTOCOL}" \
  "${RUN_OUTPUT_ROOT}" "${EXPECTED_GIT_COMMIT}" "${CONFIGURATION_LOCKED_CONFIRMATION}" "${CPUSET}" \
  "${SCRIPT_DIR}/run_dsct_ft_formal_seed0.sh" "${LAUNCHER_LOG}"

tmux new-session -d -s "${SESSION}" -n "dsct_ft_seed0" "${command}"
echo "Started locked DSCT-FT Track-B formal seed 0: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Physical GPUs: ${GPU}; free memory MiB: ${FREE_GPU_MIB[*]}"
echo "CPU affinity: ${CPUSET}; workers: 2; OMP/MKL threads: 4; eval batch: 32"
echo "Attach: tmux attach -t ${SESSION}"
echo "Archive: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz"
echo "Checksum: ${RUN_OUTPUT_ROOT}/download_packages/${RUN_ID}.tar.gz.sha256"
