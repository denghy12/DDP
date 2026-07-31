#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?GPU is required}"
JOBS="${JOBS:?JOBS is required}"
STATE_DIR="${STATE_DIR:?STATE_DIR is required}"

read -r -a jobs <<< "${JOBS}"
worker_failed=0
for job in "${jobs[@]}"; do
  method="${job%%:*}"
  seed="${job#*:seed}"
  key="${method}_seed${seed}"
  log="${STATE_DIR}/${key}.log"
  echo "Starting ${key} on physical GPU ${GPU}"
  set +e
  METHOD="${method}" SEED="${seed}" GPU="${GPU}" \
    EWC_LAMBDA="${EWC_LAMBDA:-}" \
    bash "${SCRIPT_DIR}/run_clip_continual_baseline.sh" \
    2>&1 | tee "${log}"
  code=${PIPESTATUS[0]}
  set -e
  if [[ "${code}" -ne 0 ]]; then
    worker_failed=1
    echo "${key} failed with exit code ${code}" >&2
  fi
done

exit "${worker_failed}"
