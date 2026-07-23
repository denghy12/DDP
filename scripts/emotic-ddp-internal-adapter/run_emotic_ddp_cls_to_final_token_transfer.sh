#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

GPU="${GPU:?Set GPU to the physical GPU index}"
SEEDS="${SEEDS:?Set SEEDS, for example '0 2'}"
WORKER="${WORKER:-worker}"
DDP_CHECKPOINT_DIR="${DDP_CHECKPOINT_DIR:-./output/emotic_b5c3_ddp_semantic_tau2/checkpoints}"
BASELINE_SCORES_DIR="${BASELINE_SCORES_DIR:-./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050}"
DATA_ROOT="${DATA_ROOT:-./datasets/EMOTIC}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-./pretrained/clip/ViT-B-16.pt}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WORKERS="${WORKERS:-4}"
STATE_DIR="./output/emotic_ddp_cls_to_final_token_transfer_pipeline"
mkdir -p "${STATE_DIR}"

record_failure() {
  code=$?
  if (( code != 0 )); then
    echo "${code}" > "${STATE_DIR}/${WORKER}.failed"
  fi
}
trap record_failure EXIT

for task in 0 1 2 3 4 5 6 7; do
  [[ -s "${DDP_CHECKPOINT_DIR}/task${task}.pth" ]] || {
    echo "Missing DDP checkpoint: ${DDP_CHECKPOINT_DIR}/task${task}.pth" >&2
    exit 1
  }
  [[ -s "${BASELINE_SCORES_DIR}/task${task}_scores.pt" ]] || {
    echo "Missing baseline scores: ${BASELINE_SCORES_DIR}/task${task}_scores.pt" >&2
    exit 1
  }
done
[[ -d "${DATA_ROOT}" ]] || { echo "Missing ${DATA_ROOT}" >&2; exit 1; }
[[ -s "${CLIP_MODEL_PATH}" ]] || {
  echo "Missing ${CLIP_MODEL_PATH}" >&2
  exit 1
}

for seed in ${SEEDS}; do
  if [[ ! "${seed}" =~ ^[012]$ ]]; then
    echo "Seed must be 0, 1, or 2; found ${seed}" >&2
    exit 1
  fi
  BANK_DIR="./output/emotic_ddp_task_adapter_bank_full/seed${seed}"
  MANIFEST="${BANK_DIR}/adapter_bank_manifest.json"
  OUTPUT_DIR="./output/emotic_ddp_cls_to_final_token_transfer_seed${seed}"
  RUN_NAME="emotic_ddp_cls_to_final_token_transfer_seed${seed}"
  [[ -s "${MANIFEST}" ]] || { echo "Missing ${MANIFEST}" >&2; exit 1; }

  python - "${MANIFEST}" "${seed}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, expected_seed = Path(sys.argv[1]), int(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest.get("training_mode") == "full"
assert int(manifest.get("seed", -1)) == expected_seed
assert float(manifest.get("inference_alpha", -1)) == 0.03
assert manifest.get("class_specific_gate") is False
assert manifest.get("task_specific_alpha") is False
assert manifest.get("test_used_for_selection") is False
adapters = manifest.get("adapters", {})
assert sorted(map(int, adapters)) == list(range(8))
for task_id in range(8):
    item = adapters[str(task_id)]
    checkpoint = manifest_path.parent / item["checkpoint"]
    assert checkpoint.is_file(), checkpoint
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == item["sha256"], checkpoint
print(f"Source bank seed{expected_seed}: manifest and checkpoint hashes OK")
PY

  COMPLETE="$(python - "${OUTPUT_DIR}/evaluation_summary.json" "${seed}" <<'PY'
import json
import os
import sys

path, seed = sys.argv[1], int(sys.argv[2])
complete = False
if os.path.isfile(path):
    data = json.load(open(path, encoding="utf-8"))
    protocol = data.get("protocol", {})
    methods = data.get("methods", data.get("results", {}))
    method_aliases = (
        ("ddp", "ddp_baseline", "baseline_ddp"),
        (
            "old_cls_feature_difference",
            "cls_feature_difference",
            "original_cls_feature_difference",
        ),
        ("all_tokens", "all_token_transfer"),
        ("cls_only", "cls_token_only"),
    )
    scale = protocol.get(
        "residual_scale",
        protocol.get("inference_alpha", protocol.get("adapter_residual_scale", -1)),
    )
    threshold = protocol.get(
        "decision_threshold", protocol.get("threshold", -1)
    )
    diagnostics_complete = all(
        any(
            alias in methods
            and all(
                isinstance(
                    row.get("structural_diagnostics", {}).get("test"),
                    dict,
                )
                for row in methods[alias].get("tasks", [])
            )
            for alias in aliases
        )
        for aliases in (
            ("all_tokens", "all_token_transfer"),
            ("cls_only", "cls_token_only"),
        )
    )
    complete = (
        int(protocol.get("seed", -1)) == seed
        and float(scale) == 0.03
        and float(threshold) == 0.5
        and protocol.get("test_used_for_selection") is False
        and protocol.get("adapter_fine_tuned", False) is False
        and diagnostics_complete
        and os.path.isfile(
            os.path.join(os.path.dirname(path), "transfer_manifest.json")
        )
        and all(
            any(
                alias in methods
                and len(methods[alias].get("tasks", [])) == 8
                and isinstance(methods[alias].get("aggregate"), dict)
                for alias in aliases
            )
            for aliases in method_aliases
        )
    )
print("1" if complete else "0")
PY
)"
  if [[ "${COMPLETE}" == "1" ]]; then
    echo "Skip complete ${RUN_NAME}"
    continue
  fi

  mkdir -p "${OUTPUT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" python \
    eval_emotic_ddp_cls_to_final_token_transfer.py \
    --checkpoint_dir "${DDP_CHECKPOINT_DIR}" \
    --bank_manifest "${MANIFEST}" \
    --data_root "${DATA_ROOT}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --name "${RUN_NAME}" \
    --baseline_scores_dir "${BASELINE_SCORES_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"

  python - "${OUTPUT_DIR}/evaluation_summary.json" "${seed}" <<'PY'
import json
import sys

path, seed = sys.argv[1], int(sys.argv[2])
data = json.load(open(path, encoding="utf-8"))
assert int(data["protocol"]["seed"]) == seed
methods = data.get("methods", data.get("results", {}))
for aliases in (
    ("ddp", "ddp_baseline", "baseline_ddp"),
    (
        "old_cls_feature_difference",
        "cls_feature_difference",
        "original_cls_feature_difference",
    ),
    ("all_tokens", "all_token_transfer"),
    ("cls_only", "cls_token_only"),
):
    payload = next((methods[key] for key in aliases if key in methods), None)
    assert payload is not None, aliases
    assert len(payload["tasks"]) == 8
for method in ("all_tokens", "cls_only"):
    for row in methods[method]["tasks"]:
        diagnostics = row["structural_diagnostics"]["test"]
        assert "attention_kl" in diagnostics
        assert "pooled_feature_cosine_drift" in diagnostics
        assert "path_logit_absolute_drift" in diagnostics
        assert "cls_token_delta_l2" in diagnostics
        assert "patch_token_delta_l2" in diagnostics
print(f"Completed and validated transfer seed{seed}")
PY
done

touch "${STATE_DIR}/${WORKER}.done"
rm -f "${STATE_DIR}/${WORKER}.failed"
echo "${WORKER} complete on GPU${GPU}: seeds ${SEEDS}"
