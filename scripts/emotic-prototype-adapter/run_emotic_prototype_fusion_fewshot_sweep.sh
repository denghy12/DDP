#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:-0}"
SHOTS_LIST="${SHOTS_LIST:-1 2 4 8 16}"
SEEDS_LIST="${SEEDS_LIST:-0 1 2}"

for shots in ${SHOTS_LIST}; do
  for seed in ${SEEDS_LIST}; do
    checkpoint="./output/emotic_prototype_adapter_base5_${shots}shot_seed${seed}/best_adapter.pth"
    run_name="emotic_prototype_fusion_strict_base5_${shots}shot_seed${seed}"
    summary="./output/${run_name}/fusion_all_tasks_summary.json"
    if [[ -s "${summary}" ]]; then
      echo "Skip completed fusion: K=${shots}, seed=${seed}"
      continue
    fi
    if [[ ! -s "${checkpoint}" ]]; then
      echo "Missing existing few-shot checkpoint: ${checkpoint}" >&2
      exit 1
    fi
    GPU="${GPU}" RUN_NAME="${run_name}" PROTOTYPE_CHECKPOINT="${checkpoint}" \
      bash scripts/emotic-prototype-adapter/run_emotic_prototype_fusion_all_tasks.sh "$@"
  done
done
