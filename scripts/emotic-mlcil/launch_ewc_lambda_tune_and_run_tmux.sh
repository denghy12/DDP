#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_ewc_lambda_tune}"
RUN_ID="${RUN_ID:-ewc_lambda_tune_$(date +%Y%m%d_%H%M%S)}"
LAMBDAS="${LAMBDAS:-100 10000 100000 1000000 10000000}"
TUNING_SEED="${TUNING_SEED:-0}"
FORMAL_SEEDS="${FORMAL_SEEDS:-0 1 2}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
GPU_SLOTS="${GPU_SLOTS:-}"
MAX_CONCURRENT_PER_GPU="${MAX_CONCURRENT_PER_GPU:-2}"
EWC_JOB_MEMORY_MIB="${EWC_JOB_MEMORY_MIB:-5200}"
GPU_RESERVE_MIB="${GPU_RESERVE_MIB:-2048}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/ewc_lambda_v0.3}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"
POLL_SECONDS="${POLL_SECONDS:-20}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid RUN_ID: ${RUN_ID}" >&2
  exit 2
}
[[ "${TUNING_SEED}" =~ ^[0-9]+$ ]] || {
  echo "TUNING_SEED must be a non-negative integer" >&2
  exit 2
}
for value in "${MAX_CONCURRENT_PER_GPU}" "${EWC_JOB_MEMORY_MIB}" "${GPU_RESERVE_MIB}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "GPU capacity settings must be non-negative integers" >&2
    exit 2
  }
done
[[ "${MAX_CONCURRENT_PER_GPU}" -gt 0 && "${EWC_JOB_MEMORY_MIB}" -gt 0 ]] || {
  echo "GPU concurrency and per-job memory must be positive" >&2
  exit 2
}
for flag in "${REQUIRE_CLEAN}" "${RUN_GPU_SMOKE}"; do
  [[ "${flag}" == "0" || "${flag}" == "1" ]] || {
    echo "REQUIRE_CLEAN and RUN_GPU_SMOKE must be 0 or 1" >&2
    exit 2
  }
done
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
[[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing data root: ${DATA_ROOT}" >&2; exit 2; }
[[ -s "${CLIP_MODEL_PATH}" ]] || {
  echo "Missing CLIP model: ${CLIP_MODEL_PATH}" >&2
  exit 2
}
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "Formal EWC runs require a clean Git worktree" >&2
  exit 2
fi

read -r -a lambda_values <<< "${LAMBDAS}"
[[ "${#lambda_values[@]}" -gt 0 ]] || {
  echo "LAMBDAS must contain at least one candidate" >&2
  exit 2
}
"${PYTHON}" - "${lambda_values[@]}" <<'PY'
import math
import sys

values = [float(value) for value in sys.argv[1:]]
if any(not math.isfinite(value) or value <= 0 for value in values):
    raise ValueError("Every EWC lambda must be finite and positive")
if len(values) != len(set(values)):
    raise ValueError("EWC lambda candidates must be unique")
PY

read -r -a formal_seeds <<< "${FORMAL_SEEDS}"
[[ "${#formal_seeds[@]}" -gt 0 ]] || {
  echo "FORMAL_SEEDS must contain at least one seed" >&2
  exit 2
}
seen_seeds=" "
for seed in "${formal_seeds[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || {
    echo "Invalid formal seed: ${seed}" >&2
    exit 2
  }
  if [[ "${seen_seeds}" == *" ${seed} "* ]]; then
    echo "Duplicate formal seed: ${seed}" >&2
    exit 2
  fi
  seen_seeds+="${seed} "
done

if [[ -z "${GPU_SLOTS}" ]]; then
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is required for automatic GPU capacity checks" >&2
    exit 2
  }
  read -r -a gpu_ids <<< "${GPU_LIST}"
  slots=()
  seen=" "
  for gpu in "${gpu_ids[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || {
      echo "Invalid GPU index: ${gpu}" >&2
      exit 2
    }
    if [[ "${seen}" == *" ${gpu} "* ]]; then
      echo "Duplicate GPU index: ${gpu}" >&2
      exit 2
    fi
    seen+="${gpu} "
    free_mib="$(
      nvidia-smi -i "${gpu}" --query-gpu=memory.free \
        --format=csv,noheader,nounits | tr -d '[:space:]'
    )"
    [[ "${free_mib}" =~ ^[0-9]+$ ]] || {
      echo "Cannot read free memory for GPU ${gpu}: ${free_mib}" >&2
      exit 2
    }
    usable=$((free_mib - GPU_RESERVE_MIB))
    capacity=0
    if [[ "${usable}" -gt 0 ]]; then
      capacity=$((usable / EWC_JOB_MEMORY_MIB))
    fi
    if [[ "${capacity}" -gt "${MAX_CONCURRENT_PER_GPU}" ]]; then
      capacity="${MAX_CONCURRENT_PER_GPU}"
    fi
    echo "GPU ${gpu}: free=${free_mib} MiB, safe EWC slots=${capacity}"
    for ((index = 0; index < capacity; index++)); do
      slots+=("${gpu}")
    done
  done
  GPU_SLOTS="${slots[*]:-}"
else
  read -r -a slots <<< "${GPU_SLOTS}"
  for gpu in "${slots[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || {
      echo "Invalid GPU slot: ${gpu}" >&2
      exit 2
    }
  done
fi
read -r -a slots <<< "${GPU_SLOTS}"
[[ "${#slots[@]}" -gt 0 ]] || {
  echo "No GPU has enough safe free memory for an EWC job" >&2
  exit 2
}
echo "Selected physical GPU slots: ${GPU_SLOTS}"

echo "Verifying runtime and running CPU tests..."
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil.runner import CORE_BASE_COMMIT, CORE_RUNTIME_VERSION

if CORE_RUNTIME_VERSION != "0.3.1":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if CORE_BASE_COMMIT != "00f399f13bc7552c254c8f6e6c095a8be4f56146":
    raise RuntimeError(f"Unexpected Core base: {CORE_BASE_COMMIT}")
print({"runtime": CORE_RUNTIME_VERSION, "core_base_commit": CORE_BASE_COMMIT})
PY
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running worst-case EWC memory smoke on physical GPU ${slots[0]}..."
  CUDA_VISIBLE_DEVICES="${slots[0]}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_clip_visual_training.py" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --batch-size "${TRAIN_BATCH_SIZE}"
fi

RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
TUNING_ROOT="${RUN_ROOT}/tuning"
TUNING_STATE_DIR="${RUN_ROOT}/runtime_logs/tuning_state"
SELECTION_DIR="${RUN_ROOT}/tuning_selection"
FORMAL_ROOT="${RUN_ROOT}/formal"
FORMAL_STATE_DIR="${RUN_ROOT}/runtime_logs/formal_state"
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "Run output already exists: ${RUN_ROOT}" >&2
  exit 2
}
mkdir -p "${TUNING_STATE_DIR}"

declare -a assignments
for index in "${!slots[@]}"; do
  assignments[index]=""
done
tuning_keys=()
for index in "${!lambda_values[@]}"; do
  slot=$((index % ${#slots[@]}))
  assignments[slot]="${assignments[slot]} ${lambda_values[index]}"
  slug="$("${PYTHON}" - "${lambda_values[index]}" <<'PY'
import sys
value = float(sys.argv[1])
print(format(value, ".12g").replace("+", "p").replace("-", "m").replace(".", "d"))
PY
)"
  tuning_keys+=("ewc_lambda_${slug}")
done

worker_count=0
for slot in "${!slots[@]}"; do
  assignment="${assignments[slot]# }"
  [[ -n "${assignment}" ]] || continue
  gpu="${slots[slot]}"
  printf -v command \
    'cd %q && GPU=%q LAMBDAS=%q TUNING_ROOT=%q STATE_DIR=%q TUNING_SEED=%q RUN_ID=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q PROTOCOL=%q TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q bash %q; code=$?; echo TUNING_WORKER_EXIT_CODE=$code; exec bash' \
    "${ROOT}" "${gpu}" "${assignment}" "${TUNING_ROOT}" \
    "${TUNING_STATE_DIR}" "${TUNING_SEED}" "${RUN_ID}" "${PYTHON}" \
    "${DATA_ROOT}" "${CLIP_MODEL_PATH}" "${PROTOCOL}" \
    "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${WORKERS}" \
    "${SCRIPT_DIR}/run_ewc_lambda_tuning_worker.sh"
  if [[ "${worker_count}" -eq 0 ]]; then
    tmux new-session -d -s "${SESSION}" -n "tune${slot}_g${gpu}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "tune${slot}_g${gpu}" "${command}"
  fi
  worker_count=$((worker_count + 1))
done

tuning_key_string="${tuning_keys[*]}"
printf -v monitor_command \
  'cd %q && SESSION=%q RUN_ID=%q TUNING_STATE_DIR=%q TUNING_JOB_KEYS=%q TUNING_ROOT=%q SELECTION_DIR=%q FORMAL_ROOT=%q FORMAL_STATE_DIR=%q FORMAL_SEEDS=%q GPU_SLOTS=%q LAMBDAS=%q TUNING_SEED=%q POLL_SECONDS=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q PROTOCOL=%q TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q bash %q; code=$?; echo MONITOR_EXIT_CODE=$code; exec bash' \
  "${ROOT}" "${SESSION}" "${RUN_ID}" "${TUNING_STATE_DIR}" \
  "${tuning_key_string}" "${TUNING_ROOT}" "${SELECTION_DIR}" \
  "${FORMAL_ROOT}" "${FORMAL_STATE_DIR}" "${FORMAL_SEEDS}" \
  "${GPU_SLOTS}" "${LAMBDAS}" "${TUNING_SEED}" "${POLL_SECONDS}" \
  "${PYTHON}" "${DATA_ROOT}" "${CLIP_MODEL_PATH}" "${PROTOCOL}" \
  "${TRAIN_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" "${WORKERS}" \
  "${SCRIPT_DIR}/wait_ewc_lambda_tune_and_run.sh"
tmux new-window -t "${SESSION}" -n monitor "${monitor_command}"

echo "Started tmux session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Validation lambda candidates: ${LAMBDAS}"
echo "Tuning workers: ${worker_count} on physical slots ${GPU_SLOTS}"
echo "After validation selection, formal EWC seeds ${FORMAL_SEEDS} start automatically"
echo "Attach: tmux attach -t ${SESSION}"
echo "Heavy tuning/checkpoints: ${RUN_ROOT}"
echo "Final download folder: ${FORMAL_ROOT}/download_ready/${RUN_ID}"
