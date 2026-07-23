#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to the physical GPU index}"
WORKER="${WORKER:?Set WORKER to s02 or s1}"
STATE_DIR="./output/emotic_ddp_final_token_c300_multiseed_pipeline"
mkdir -p "${STATE_DIR}"

case "${WORKER}" in
  s02) SEEDS=(0 2) ;;
  s1) SEEDS=(1) ;;
  *) echo "WORKER must be s02 or s1" >&2; exit 2 ;;
esac

touch "${STATE_DIR}/${WORKER}.running"
rm -f "${STATE_DIR}/${WORKER}.done" "${STATE_DIR}/${WORKER}.failed"
trap 'code=$?; rm -f "${STATE_DIR}/${WORKER}.running"; if (( code == 0 )); then touch "${STATE_DIR}/${WORKER}.done"; else touch "${STATE_DIR}/${WORKER}.failed"; fi' EXIT

for seed in "${SEEDS[@]}"; do
  run_name="emotic_ddp_final_token_c300_multiseed_seed${seed}"
  summary="./output/${run_name}/training_summary.json"
  if [[ -s "${summary}" ]]; then
    python - "${summary}" "${seed}" <<'PY'
import json
import math
import sys

path, seed = sys.argv[1:]
data = json.load(open(path, encoding="utf-8"))
valid = (
    data.get("training_health_check", {}).get("status") == "passed"
    and int(data.get("optimizer_steps", -1)) == 300
    and math.isclose(float(data.get("args", {}).get("lr", -1)), 3e-5)
    and int(data.get("args", {}).get("seed", -1)) == int(seed)
    and "final_validation_diagnostics" in data
)
if not valid:
    raise SystemExit(f"Existing C300 run does not match locked protocol: {path}")
print(f"Skip complete matching C300 run: {path}")
PY
    continue
  fi

  echo "Run locked C300 seed${seed}: lr=3e-5, steps=300, GPU${GPU}"
  GPU="${GPU}" \
  SEED="${seed}" \
  RUN_NAME="${run_name}" \
  LEARNING_RATE="3e-5" \
  MAX_OPTIMIZER_STEPS="300" \
  HEALTH_EPOCHS=1 \
  IDENTITY_WEIGHT=0.1 \
    bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_training_healthcheck.sh
done

echo "C300 worker ${WORKER} completed on GPU${GPU}"

