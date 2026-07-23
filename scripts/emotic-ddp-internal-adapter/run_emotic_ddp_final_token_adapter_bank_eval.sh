#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to the physical GPU index}"
TRAINING_MODE="${TRAINING_MODE:?Set TRAINING_MODE to full or 16shot}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"

MANIFEST_ARGS=()
RUN_NAME_ARGS=()
for seed in 0 1 2; do
  MANIFEST="./output/emotic_ddp_final_token_adapter_bank_${TRAINING_MODE}/seed${seed}/final_token_adapter_bank_manifest.json"
  [[ -s "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}" >&2; exit 1; }
  MANIFEST_ARGS+=("${MANIFEST}")
  RUN_NAME_ARGS+=("emotic_ddp_final_token_adapter_bank_${TRAINING_MODE}_seed${seed}")
done
for task in 0 1 2 3 4 5 6 7; do
  [[ -s "${DDP_CHECKPOINT_DIR}/task${task}.pth" ]] || exit 1
  [[ -s "${BASELINE_SCORES_DIR}/task${task}_scores.pt" ]] || exit 1
done

ALL_COMPLETE=1
for seed in 0 1 2; do
  SUMMARY="./output/emotic_ddp_final_token_adapter_bank_${TRAINING_MODE}_seed${seed}/evaluation_summary.json"
  COMPLETE="$(python - "${SUMMARY}" "${TRAINING_MODE}" "${seed}" <<'PY'
import json
import os
import sys

path, mode, seed = sys.argv[1:]
complete = False
if os.path.isfile(path):
    data = json.load(open(path, encoding="utf-8"))
    protocol = data.get("protocol", {})
    complete = (
        protocol.get("training_mode") == mode
        and int(protocol.get("seed", -1)) == int(seed)
        and protocol.get("feature_source") == "197 final projected prompted ViT tokens"
        and protocol.get("checkpoint_selection") == "fixed last epoch"
        and float(protocol.get("decision_threshold", -1)) == 0.5
        and protocol.get("test_used_for_selection") is False
        and len(data.get("tasks", [])) == 8
    )
print("1" if complete else "0")
PY
)"
  if [[ "${COMPLETE}" != "1" ]]; then
    ALL_COMPLETE=0
  fi
done
if [[ "${ALL_COMPLETE}" == "1" ]]; then
  echo "Skip complete Final-token ${TRAINING_MODE} shared evaluation"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" python eval_emotic_ddp_final_token_adapter_bank.py \
  --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
  --bank_manifests "${MANIFEST_ARGS[@]}" \
  --run_names "${RUN_NAME_ARGS[@]}" \
  --output_root ./output \
  --data_root ./datasets/EMOTIC \
  --clip_model_path ./pretrained/clip/ViT-B-16.pt \
  --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
  --batch_size 2 \
  --workers 4

echo "Completed shared ${TRAINING_MODE} evaluation for seeds 0, 1, 2"
