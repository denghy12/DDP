#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SESSION="${SESSION:-emotic_frozen_clip_baselines}"
RUN_ID="${RUN_ID:-frozen_clip_$(date +%Y%m%d_%H%M%S)}"
METHODS="${METHODS:-finetune lwf ewc}"
SEEDS="${SEEDS:-0}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${ROOT}/pretrained/clip/ViT-B-16.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/baselines_v0.1}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3.yaml}"
REPORTING_SPLIT="${REPORTING_SPLIT:-test}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
WORKERS="${WORKERS:-0}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"

read -r -a methods <<< "${METHODS}"
read -r -a seeds <<< "${SEEDS}"
read -r -a gpu_ids <<< "${GPU_LIST}"
jobs=()
for seed in "${seeds[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || {
    echo "Invalid seed: ${seed}" >&2
    exit 2
  }
  for method in "${methods[@]}"; do
    case "${method}" in
      finetune|lwf|ewc) ;;
      *)
        echo "Unsupported method: ${method}" >&2
        exit 2
        ;;
    esac
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
if [[ "${#jobs[@]}" -gt "${#gpu_ids[@]}" ]]; then
  echo "This launcher runs one job per GPU. Split ${#jobs[@]} jobs into waves." >&2
  exit 2
fi
for gpu in "${gpu_ids[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || {
    echo "Invalid GPU index: ${gpu}" >&2
    exit 2
  }
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

echo "Verifying baseline runtime and running tests..."
"${PYTHON}" - <<'PY'
from benchmarks.emotic_mlcil import method_names
from benchmarks.emotic_mlcil.runner import CORE_BASE_COMMIT, CORE_RUNTIME_VERSION

if CORE_RUNTIME_VERSION != "0.2.0":
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

STATE_DIR="${OUTPUT_ROOT}/_baseline_state/${RUN_ID}"
mkdir -p "${STATE_DIR}"
job_keys=()
for index in "${!jobs[@]}"; do
  job="${jobs[index]}"
  method="${job%%:*}"
  seed="${job#*:seed}"
  gpu="${gpu_ids[index]}"
  key="${method}_seed${seed}"
  job_keys+=("${key}")
  log="${STATE_DIR}/${key}.log"
  printf -v command \
    'cd %q && METHOD=%q SEED=%q GPU=%q RUN_ID=%q STATE_DIR=%q PYTHON=%q DATA_ROOT=%q CLIP_MODEL_PATH=%q OUTPUT_ROOT=%q PROTOCOL=%q REPORTING_SPLIT=%q TRAIN_BATCH_SIZE=%q EVAL_BATCH_SIZE=%q WORKERS=%q bash %q 2>&1 | tee %q; code=${PIPESTATUS[0]}; echo JOB_EXIT_CODE=$code; exec bash' \
    "${ROOT}" "${method}" "${seed}" "${gpu}" "${RUN_ID}" "${STATE_DIR}" \
    "${PYTHON}" "${DATA_ROOT}" "${CLIP_MODEL_PATH}" "${OUTPUT_ROOT}" \
    "${PROTOCOL}" "${REPORTING_SPLIT}" "${TRAIN_BATCH_SIZE}" \
    "${EVAL_BATCH_SIZE}" "${WORKERS}" \
    "${SCRIPT_DIR}/run_frozen_clip_baseline.sh" "${log}"
  if [[ "${index}" -eq 0 ]]; then
    tmux new-session -d -s "${SESSION}" -n "${key}_g${gpu}" "${command}"
  else
    tmux new-window -t "${SESSION}" -n "${key}_g${gpu}" "${command}"
  fi
done

job_key_string="${job_keys[*]}"
printf -v monitor_command \
  'STATE_DIR=%q JOB_KEYS=%q bash %q; code=$?; echo MONITOR_EXIT_CODE=$code; exec bash' \
  "${STATE_DIR}" "${job_key_string}" \
  "${SCRIPT_DIR}/wait_frozen_clip_baselines.sh"
tmux new-window -t "${SESSION}" -n monitor "${monitor_command}"

echo "Started tmux session: ${SESSION}"
echo "Run ID: ${RUN_ID}"
echo "Jobs: ${job_key_string}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs: ${STATE_DIR}"
echo "Checkpoint roots remain under: ${OUTPUT_ROOT}/benchmarks"
echo "Download bundles will be under each method seed*/results_to_sync/${RUN_ID}"
