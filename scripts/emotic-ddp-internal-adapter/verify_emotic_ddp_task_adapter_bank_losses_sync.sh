#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

EXPECTED_BRANCH="codex/emotic-ddp-task-adapter-bank-loss"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Expected branch ${EXPECTED_BRANCH}, found ${CURRENT_BRANCH}" >&2
  exit 1
fi

REQUIRED_FILES=(
  emotic_multilabel_losses.py
  emotic_task_adapter_bank.py
  train_emotic_ddp_task_adapter.py
  build_emotic_ddp_task_adapter_bank.py
  eval_emotic_ddp_task_adapter_bank.py
  summarize_emotic_ddp_task_adapter_bank_losses.py
  tests/test_emotic_multilabel_losses.py
  tests/test_emotic_task_adapter_bank_loss_protocol.py
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_train.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_eval.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_worker.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_lane.sh
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_task_adapter_bank_losses.sh
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_losses_8gpu_tmux.sh
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_bal_ablation_8gpu_tmux.sh
)
for path in "${REQUIRED_FILES[@]}"; do
  [[ -s "${path}" ]] || { echo "Missing synchronized file: ${path}" >&2; exit 1; }
done

python -m py_compile \
  emotic_multilabel_losses.py \
  emotic_task_adapter_bank.py \
  train_emotic_ddp_task_adapter.py \
  build_emotic_ddp_task_adapter_bank.py \
  eval_emotic_ddp_task_adapter_bank.py \
  summarize_emotic_ddp_task_adapter_bank_losses.py \
  models/ddp.py \
  tests/test_emotic_multilabel_losses.py \
  tests/test_emotic_task_adapter_bank_loss_protocol.py

python -m unittest \
  tests.test_emotic_multilabel_losses \
  tests.test_emotic_task_adapter_bank_loss_protocol \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary

for script in \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_train.sh \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_eval.sh \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_worker.sh \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_loss_lane.sh \
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_task_adapter_bank_losses.sh \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_losses_8gpu_tmux.sh \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_bal_ablation_8gpu_tmux.sh; do
  bash -n "${script}"
done

python - <<'PY'
from pathlib import Path

trainer = Path("train_emotic_ddp_task_adapter.py").read_text(encoding="utf-8")
evaluator = Path("eval_emotic_ddp_task_adapter_bank.py").read_text(encoding="utf-8")
runner = Path(
    "scripts/emotic-ddp-internal-adapter/"
    "run_emotic_ddp_task_adapter_bank_loss_train.sh"
).read_text(encoding="utf-8")
eval_runner = Path(
    "scripts/emotic-ddp-internal-adapter/"
    "run_emotic_ddp_task_adapter_bank_loss_eval.sh"
).read_text(encoding="utf-8")
ablation_runner = Path(
    "scripts/emotic-ddp-internal-adapter/"
    "launch_emotic_ddp_task_adapter_bank_bal_ablation_8gpu_tmux.sh"
).read_text(encoding="utf-8")

assert 'eval_splits=("test",)' not in trainer
assert '"future_labels_supervised": False' in trainer
assert '"validation_used_for_checkpoint_selection"' in trainer
assert "--checkpoint_rule last_epoch" in runner
assert "--fixed_threshold 0.5" in eval_runner
assert "feature_difference" in evaluator
assert "asl_smoothing asl_positive_weight" in ablation_runner
assert "asl asl_smoothing asl_positive_weight bal_paper" in ablation_runner
print("Locked ASL/BAL Task Bank protocol audit: OK")
PY

git --no-pager diff --check
echo "SYNC_CHECK_OK: ASL/BAL Task Adapter Bank code is synchronized and validated"
