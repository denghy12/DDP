#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

CHECK_EXPERIMENT_INPUTS="${CHECK_EXPERIMENT_INPUTS:-0}"
REQUIRED_FILES=(
  emotic_cls_to_final_token_transfer.py
  eval_emotic_ddp_cls_to_final_token_transfer.py
  summarize_emotic_ddp_cls_to_final_token_transfer.py
  tests/test_emotic_cls_to_final_token_transfer.py
  tests/test_emotic_cls_to_final_token_transfer_summary.py
  docs/ddp_cls_to_final_token_transfer.md
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_to_final_token_transfer.sh
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_cls_to_final_token_transfer_tmux.sh
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_cls_to_final_token_transfer.sh
  scripts/emotic-ddp-internal-adapter/verify_emotic_ddp_cls_to_final_token_transfer_sync.sh
)
for path in "${REQUIRED_FILES[@]}"; do
  [[ -s "${path}" ]] || {
    echo "Missing synchronized file: ${path}" >&2
    exit 1
  }
done

python -m py_compile \
  emotic_cls_to_final_token_transfer.py \
  eval_emotic_ddp_cls_to_final_token_transfer.py \
  summarize_emotic_ddp_cls_to_final_token_transfer.py \
  tests/test_emotic_cls_to_final_token_transfer.py \
  tests/test_emotic_cls_to_final_token_transfer_summary.py

python -m unittest \
  tests.test_emotic_cls_to_final_token_transfer \
  tests.test_emotic_cls_to_final_token_transfer_summary

for script in \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_cls_to_final_token_transfer.sh \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_cls_to_final_token_transfer_tmux.sh \
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_cls_to_final_token_transfer.sh \
  scripts/emotic-ddp-internal-adapter/verify_emotic_ddp_cls_to_final_token_transfer_sync.sh; do
  bash -n "${script}"
done

python - <<'PY'
from pathlib import Path

runner = Path(
    "scripts/emotic-ddp-internal-adapter/"
    "run_emotic_ddp_cls_to_final_token_transfer.sh"
).read_text(encoding="utf-8")
launcher = Path(
    "scripts/emotic-ddp-internal-adapter/"
    "launch_emotic_ddp_cls_to_final_token_transfer_tmux.sh"
).read_text(encoding="utf-8")
summary = Path(
    "summarize_emotic_ddp_cls_to_final_token_transfer.py"
).read_text(encoding="utf-8")

assert "SEEDS='0 2'" in launcher
assert "SEEDS='1'" in launcher
assert "--bank_manifest" in runner
assert "emotic_ddp_task_adapter_bank_full/seed${seed}" in runner
assert "float(manifest.get(\"inference_alpha\", -1)) == 0.03" in runner
assert '"all_tokens"' in summary
assert '"cls_only"' in summary
assert '"old_cls_feature_difference"' in summary
assert '"decision_threshold": 0.5' in summary
assert '"adapter_fine_tuned": False' in summary
assert "collect_structural_diagnostics" in summary
assert '"structural_diagnostics.json"' in summary
print("Locked CLS-to-Final-token transfer protocol audit: OK")
PY

if [[ "${CHECK_EXPERIMENT_INPUTS}" == "1" ]]; then
  for seed in 0 1 2; do
    manifest="./output/emotic_ddp_task_adapter_bank_full/seed${seed}/adapter_bank_manifest.json"
    [[ -s "${manifest}" ]] || {
      echo "Missing source bank manifest: ${manifest}" >&2
      exit 1
    }
  done
  for task in 0 1 2 3 4 5 6 7; do
    [[ -s "./output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task${task}.pth" ]] || {
      echo "Missing DDP task${task} checkpoint" >&2
      exit 1
    }
    [[ -s "./output/emotic_b5c3_ddp_semantic_tau2_test_threshold050/task${task}_scores.pt" ]] || {
      echo "Missing DDP task${task} baseline scores" >&2
      exit 1
    }
  done
  [[ -d ./datasets/EMOTIC ]] || { echo "Missing EMOTIC data" >&2; exit 1; }
  [[ -s ./pretrained/clip/ViT-B-16.pt ]] || {
    echo "Missing CLIP checkpoint" >&2
    exit 1
  }
  echo "Runtime experiment inputs: OK"
fi

git --no-pager diff --check
echo "SYNC_CHECK_OK: CLS-to-Final-token transfer code is synchronized and validated"
