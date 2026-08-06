#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

EXPECTED_BRANCH="codex/emotic-ddp-transformer-adapter-bank"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Expected branch ${EXPECTED_BRANCH}, found ${CURRENT_BRANCH}" >&2
  exit 1
fi

REQUIRED_FILES=(
  emotic_transformer_adapter_bank.py
  train_emotic_ddp_transformer_adapter.py
  build_emotic_ddp_transformer_adapter_bank.py
  eval_emotic_ddp_transformer_adapter_bank.py
  summarize_emotic_ddp_transformer_adapter_bank.py
  smoke_emotic_ddp_transformer_adapter.py
  tests/test_emotic_transformer_adapter_bank.py
  tests/test_emotic_transformer_adapter_protocol.py
  scripts/emotic-ddp-transformer-adapter-bank/run_train.sh
  scripts/emotic-ddp-transformer-adapter-bank/run_eval.sh
  scripts/emotic-ddp-transformer-adapter-bank/run_worker.sh
  scripts/emotic-ddp-transformer-adapter-bank/wait_and_summarize.sh
  scripts/emotic-ddp-transformer-adapter-bank/launch_8gpu_tmux.sh
  scripts/emotic-ddp-transformer-adapter-bank/verify_sync.sh
)
for path in "${REQUIRED_FILES[@]}"; do
  [[ -s "${path}" ]] || { echo "Missing synchronized file: ${path}" >&2; exit 1; }
done

python -m py_compile \
  clip/model.py \
  models/ddp.py \
  emotic_transformer_adapter_bank.py \
  train_emotic_ddp_transformer_adapter.py \
  build_emotic_ddp_transformer_adapter_bank.py \
  eval_emotic_ddp_transformer_adapter_bank.py \
  summarize_emotic_ddp_transformer_adapter_bank.py \
  smoke_emotic_ddp_transformer_adapter.py \
  tests/test_emotic_transformer_adapter_bank.py \
  tests/test_emotic_transformer_adapter_protocol.py

python -m unittest \
  tests.test_emotic_transformer_adapter_bank \
  tests.test_emotic_transformer_adapter_protocol \
  tests.test_emotic_task_adapter_bank \
  tests.test_emotic_multilabel_losses \
  tests.test_ddp_internal_adapter

for script in "${REQUIRED_FILES[@]}"; do
  [[ "${script}" == *.sh ]] && bash -n "${script}"
done

python - <<'PY'
from pathlib import Path

trainer = Path("train_emotic_ddp_transformer_adapter.py").read_text(encoding="utf-8")
evaluator = Path("eval_emotic_ddp_transformer_adapter_bank.py").read_text(encoding="utf-8")
runner = Path("scripts/emotic-ddp-transformer-adapter-bank/run_train.sh").read_text(encoding="utf-8")
module = Path("emotic_transformer_adapter_bank.py").read_text(encoding="utf-8")

assert 'eval_splits=("test",)' not in trainer
assert '"old_labels_supervised": False' in trainer
assert '"future_labels_supervised": False' in trainer
assert '"validation_used_for_checkpoint_selection": False' in trainer
assert "select_threshold" not in evaluator
assert 'default=0.5' in evaluator
assert "P2L_CA_LAYER_NUMBERS" in module
assert "parallel_to_vit_mlp" in module
assert "--epochs 20" in runner
assert "--lr 4e-4" in runner
assert "--effective_batch_size 64" in runner
assert "--asl_gamma_neg 9.8" in runner
print("Locked Transformer Adapter Bank protocol audit: OK")
PY

git --no-pager diff --check
echo "SYNC_CHECK_OK: Transformer Adapter Bank code is synchronized and validated"
