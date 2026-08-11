#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

SEED="${SEED:-0}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC}"
NATIVE_INIT="${EMOT_NET_NATIVE_INIT:-${ROOT}/pretrained/emot_net/emot_net_native_init_v0.1.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/haoyuan/workspace/emotic_benchmark_runs/emot_net_ft_track_b_v0.1/${RUN_ID}}"
PROTOCOL="${PROTOCOL:-${ROOT}/configs/emotic_mlcil/protocol_b5c3_track_b.yaml}"
REPORTING_SPLIT="${REPORTING_SPLIT:-val}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-52}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
WORKERS="${WORKERS:-0}"
EXPORT_SYNC_RESULTS="${EXPORT_SYNC_RESULTS:-1}"
CONFIGURATION_LOCKED_CONFIRMATION="${CONFIGURATION_LOCKED_CONFIRMATION:-}"

[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "Invalid SEED" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU" >&2; exit 2; }
[[ "${TRAIN_BATCH_SIZE}" == "52" ]] || {
  echo "Registered source EMOT-Net train batch size is 52" >&2
  exit 2
}
[[ -s "${NATIVE_INIT}" ]] || {
  echo "Missing audited native EMOT-Net initialization: ${NATIVE_INIT}" >&2
  exit 2
}
[[ "${REPORTING_SPLIT}" == "val" || "${REPORTING_SPLIT}" == "test" ]] || {
  echo "REPORTING_SPLIT must be val or test" >&2
  exit 2
}

runner_args=(
  --protocol "${PROTOCOL}"
  --method emot_net_ft
  --seed "${SEED}"
  --data-root "${DATA_ROOT}"
  --emot-net-native-init "${NATIVE_INIT}"
  --input-mode body_context
  --output-root "${OUTPUT_ROOT}"
  --reporting-split "${REPORTING_SPLIT}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --workers "${WORKERS}"
  --device cuda
)
if [[ "${REPORTING_SPLIT}" == "test" ]]; then
  [[ "${CONFIGURATION_LOCKED_CONFIRMATION}" == "EMOT_NET_FT_TRACK_B_V0_1" ]] || {
    echo "Held-out test requires a frozen EMOT-Net-FT configuration" >&2
    exit 2
  }
  runner_args+=(--configuration-locked)
fi
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  -m benchmarks.emotic_mlcil.runner "${runner_args[@]}"

if [[ "${EXPORT_SYNC_RESULTS}" == "1" ]]; then
  "${PYTHON}" -m benchmarks.emotic_mlcil.runner \
    --protocol "${PROTOCOL}" \
    --method emot_net_ft \
    --seed "${SEED}" \
    --output-root "${OUTPUT_ROOT}" \
    --reporting-split "${REPORTING_SPLIT}" \
    --export-sync-results \
    --shard-run-id "${RUN_ID}"
fi
