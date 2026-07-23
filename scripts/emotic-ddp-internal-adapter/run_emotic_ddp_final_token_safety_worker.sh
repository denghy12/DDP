#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to the physical GPU index}"
WORKER="${WORKER:?Set WORKER to ac or b}"
STATE_DIR="./output/emotic_ddp_final_token_safety_screen_pipeline"
mkdir -p "${STATE_DIR}"

case "${WORKER}" in
  ac) CONFIGS=(A C) ;;
  b) CONFIGS=(B) ;;
  *) echo "WORKER must be ac or b" >&2; exit 2 ;;
esac

touch "${STATE_DIR}/${WORKER}.running"
rm -f "${STATE_DIR}/${WORKER}.done" "${STATE_DIR}/${WORKER}.failed"
trap 'code=$?; rm -f "${STATE_DIR}/${WORKER}.running"; if (( code == 0 )); then touch "${STATE_DIR}/${WORKER}.done"; else touch "${STATE_DIR}/${WORKER}.failed"; fi' EXIT

run_config() {
  local config="$1"
  local run_name learning_rate max_steps
  case "${config}" in
    A)
      run_name="emotic_ddp_final_token_safety_A_lr1e4_steps100_seed0"
      learning_rate="1e-4"
      max_steps="100"
      ;;
    B)
      run_name="emotic_ddp_final_token_safety_B_lr1e4_steps300_seed0"
      learning_rate="1e-4"
      max_steps="300"
      ;;
    C)
      run_name="emotic_ddp_final_token_safety_C_lr3e5_steps300_seed0"
      learning_rate="3e-5"
      max_steps="300"
      ;;
    *) echo "Unknown safety config ${config}" >&2; return 2 ;;
  esac

  local summary="./output/${run_name}/training_summary.json"
  if [[ -s "${summary}" ]]; then
    python - "${summary}" "${learning_rate}" "${max_steps}" <<'PY'
import json
import math
import sys

path, expected_lr, expected_steps = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
valid = (
    data.get("training_health_check", {}).get("status") == "passed"
    and int(data.get("optimizer_steps", -1)) == int(expected_steps)
    and math.isclose(float(data.get("args", {}).get("lr", -1)), float(expected_lr))
    and int(data.get("args", {}).get("task_id", -1)) == 0
    and data.get("args", {}).get("training_mode") == "full"
    and int(data.get("args", {}).get("seed", -1)) == 0
)
if not valid:
    raise SystemExit(f"Existing safety run does not match locked config: {path}")
print(f"Skip complete matching safety run: {path}")
PY
    return
  fi

  echo "Run safety config ${config}: lr=${learning_rate}, steps=${max_steps}, GPU${GPU}"
  GPU="${GPU}" \
  RUN_NAME="${run_name}" \
  LEARNING_RATE="${learning_rate}" \
  MAX_OPTIMIZER_STEPS="${max_steps}" \
  HEALTH_EPOCHS=1 \
  IDENTITY_WEIGHT=0.1 \
    bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_training_healthcheck.sh
}

for config in "${CONFIGS[@]}"; do
  run_config "${config}"
done

echo "Safety worker ${WORKER} completed on GPU${GPU}"

