#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_clip_continual_baselines}"
RUN_ID="${RUN_ID:-clip_continual_$(date +%Y%m%d_%H%M%S)}"
METHODS="${METHODS:-finetune lwf ewc}"
SEEDS="${SEEDS:-0 1 2}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/clip_continual_v0.2}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
REPORTING_SPLIT="${REPORTING_SPLIT:-test}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-2}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"
RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-1}"

read -r -a methods <<< "${METHODS}"
read -r -a seeds <<< "${SEEDS}"
read -r -a gpu_ids <<< "${GPU_LIST}"
[[ "${#gpu_ids[@]}" -gt 0 ]] || {
  echo "GPU_LIST must contain at least one GPU" >&2
  exit 2
}

jobs=()
# Method-major ordering places the ninth default job (EWC seed 2) behind the
# typically faster Fine-Tuning seed 0 job when eight GPUs are supplied.
for method in "${methods[@]}"; do
  case "${method}" in
    finetune|lwf|ewc) ;;
    *)
      echo "Unsupported method: ${method}" >&2
      exit 2
      ;;
  esac
  for seed in "${seeds[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || {
      echo "Invalid seed: ${seed}" >&2
      exit 2
    }
    candidate="${method}:seed${seed}"
    for existing in "${jobs[@]}"; do
      if [[ "${existing}" == "${candidate}" ]]; then
        echo "Duplicate method/seed job: ${candidate}" >&2
        exit 2
      fi
    done
    jobs+=("${candidate}")
  done
done

seen_gpus=" "
for gpu in "${gpu_ids[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || {
    echo "Invalid GPU index: ${gpu}" >&2
    exit 2
  }
  if [[ "${seen_gpus}" == *" ${gpu} "* ]]; then
    echo "Duplicate GPU index: ${gpu}" >&2
    exit 2
  fi
  seen_gpus+="${gpu} "
done
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
  echo "Invalid RUN_ID: ${RUN_ID}" >&2
  exit 2
}
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
  echo "Formal baseline runs require a clean Git worktree" >&2
  exit 2
fi

echo "Verifying runtime and running CPU tests..."
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil import method_names
from benchmarks.emotic_mlcil.runner import CORE_BASE_COMMIT, CORE_RUNTIME_VERSION

if CORE_RUNTIME_VERSION != "0.3.1":
    raise RuntimeError(f"Unexpected runtime: {CORE_RUNTIME_VERSION}")
if CORE_BASE_COMMIT != "00f399f13bc7552c254c8f6e6c095a8be4f56146":
    raise RuntimeError(f"Unexpected Core base: {CORE_BASE_COMMIT}")
for name in ("finetune", "lwf", "ewc"):
    if name not in method_names():
        raise RuntimeError(f"Missing registered baseline: {name}")
print(
    {
        "runtime": CORE_RUNTIME_VERSION,
        "core_base_commit": CORE_BASE_COMMIT,
        "methods": tuple(method_names()),
    }
)
PY
"${PYTHON}" -m unittest discover -s tests/emotic_mlcil -t .
"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary

if [[ "${RUN_GPU_SMOKE}" == "1" ]]; then
  echo "Running worst-case CLIP visual training memory smoke on GPU ${gpu_ids[0]}..."
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${PYTHON}" \
    "${SCRIPT_DIR}/smoke_clip_visual_training.py" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --batch-size "${TRAIN_BATCH_SIZE}"
fi

RUN_OUTPUT_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
STATE_DIR="${OUTPUT_ROOT}/_baseline_state/${RUN_ID}"
mkdir -p "${STATE_DIR}"

declare -a assignments
for index in "${!gpu_ids[@]}"; do
  assignments[index]=""
done
job_keys=()
for index in "${!jobs[@]}"; do
  gpu_slot=$((index % ${#gpu_ids[@]}))
  assignments[gpu_slot]="${assignments[gpu_slot]} ${jobs[index]}"
  method="${jobs[index]%%:*}"
  seed="${jobs[index]#*:seed}"
  job_keys+=("${method}_seed${seed}")
done

worker_count=0
for slot in "${!gpu_ids[@]}"; do
  assignment="${assignments[slot]# }"
  [[ -n "${assignment}" ]] || continue
  gpu="${gpu_ids[slot]}"
  printf -v command \
    'cd %q && GPU=%q JOBS=%q STATE_DIR=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q OUTPUT_ROOT=%q PROTOCOL=%q REPORTING_SPLIT=%q TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q RUN_ID=%q bash %q; code=$?; echo WORKER_EXIT_CODE=$code; exec bash' \
    "${ROOT}" "${gpu}" "${assignment}" "${STATE_DIR}" \
    "${PYTHON}" "${DATA_ROOT}" "${CLIP_MODEL_PATH}" "${RUN_OUTPUT_ROOT}" \
    "${PROTOCOL}" "${REPORTING_SPLIT}" "${TRAIN_BATCH_SIZE}" \
    "${EVAL_BATCH_SIZE}" "${WORKERS}" "${RUN_ID}" \
    "${SCRIPT_DIR}/run_clip_continual_worker.sh"
  window="worker${slot}_g${gpu}"
  if [[ "${worker_count}" -eq 0 ]]; then
    tmux new-session -d -s "${SESSION}" -n "${window}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${window}" "${command}"
  fi
  worker_count=$((worker_count + 1))
done

job_key_string="${job_keys[*]}"
printf -v monitor_command \
  'STATE_DIR=%q JOB_KEYS=%q PYTHON=%q RUN_OUTPUT_ROOT=%q RUN_ID=%q COLLECT_SCRIPT=%q bash %q; code=$?; echo MONITOR_EXIT_CODE=$code; exec bash' \
  "${STATE_DIR}" "${job_key_string}" "${PYTHON}" "${RUN_OUTPUT_ROOT}" \
  "${RUN_ID}" "${SCRIPT_DIR}/collect_clip_continual_results.py" \
  "${SCRIPT_DIR}/wait_clip_continual_baselines.sh"
tmux new-window -t "${SESSION}" -n monitor "${monitor_command}"

echo "Started tmux session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Jobs: ${job_key_string}"
echo "Workers: ${worker_count} across GPUs ${GPU_LIST}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs: ${STATE_DIR}"
echo "Checkpoints: ${RUN_OUTPUT_ROOT}/benchmarks/*/*/*/seed*/checkpoints"
echo "Single download folder after completion: ${RUN_OUTPUT_ROOT}/download_ready/${RUN_ID}"
