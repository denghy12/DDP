#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"


for task in 0 1 2 3 4 5 6 7; do
  path="${ROOT}/output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task${task}.pth"
  [[ -s "${path}" ]] || { echo "Missing ${path}" >&2; exit 1; }
done

SMOKE_OUTPUT="${ROOT}/output/emotic_ddp_internal_adapter_identity_smoke.json"
if [[ ! -s "${SMOKE_OUTPUT}" ]]; then
  CUDA_VISIBLE_DEVICES=0 python "${ROOT}/smoke_emotic_ddp_internal_adapter.py" \
    --ddp_checkpoint "${ROOT}/output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task0.pth" \
    --data_root "${ROOT}/datasets/EMOTIC" \
    --clip_model_path "${ROOT}/pretrained/clip/ViT-B-16.pt" \
    --output "${SMOKE_OUTPUT}"
fi

launch() {
  local session="$1"
  local gpu="$2"
  local seed="$3"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Skip existing session ${session}"
    return
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && GPU=${gpu} SEED=${seed} bash scripts/emotic-ddp-internal-adapter/run_emotic_ddp_internal_adapter_16shot.sh"
  echo "Started ${session} on GPU${gpu}"
}

gpu0_free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')"
seed2_gpu=0
if [[ "${gpu0_free}" -lt 12000 ]]; then
  seed2_gpu=1
fi

launch ddp_internal_s0 0 0
launch ddp_internal_s1 0 1
launch ddp_internal_s2 "${seed2_gpu}" 2

echo "After all sessions finish: python summarize_emotic_ddp_internal_adapter.py"
