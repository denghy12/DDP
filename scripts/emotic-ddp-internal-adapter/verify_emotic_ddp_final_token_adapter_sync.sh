#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

EXPECTED_BRANCH="codex/emotic-ddp-final-token-adapter"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Expected branch ${EXPECTED_BRANCH}, found ${CURRENT_BRANCH}" >&2
  exit 1
fi

REQUIRED_FILES=(
  emotic_final_token_adapter.py
  train_emotic_ddp_final_token_adapter.py
  build_emotic_ddp_final_token_adapter_bank.py
  eval_emotic_ddp_final_token_adapter_bank.py
  summarize_emotic_ddp_final_token_adapter_bank.py
  summarize_emotic_ddp_final_token_safety_screen.py
  summarize_emotic_ddp_final_token_c300_multiseed.py
  summarize_emotic_ddp_final_token_pooling_aware_pilot.py
  summarize_emotic_ddp_final_token_pooling_aware_bank.py
  docs/ddp_final_token_adapter.md
  tests/test_emotic_final_token_adapter.py
  tests/test_emotic_final_token_safety_screen.py
  tests/test_emotic_final_token_c300_multiseed.py
  tests/test_emotic_final_token_pooling_aware_pilot.py \
  tests/test_emotic_final_token_pooling_aware_bank.py
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_adapter_bank_train.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_adapter_bank_eval.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_adapter_bank_worker.sh
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_adapter_bank.sh
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_adapter_bank_tmux.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_training_healthcheck.sh
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_training_healthcheck_tmux.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_safety_worker.sh
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_safety_screen.sh
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_safety_screen_tmux.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_c300_multiseed_worker.sh
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_c300_multiseed.sh
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_c300_multiseed_tmux.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_task0.sh
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_pooling_aware_task0_tmux.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_bank_train.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_bank_worker.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_bank_eval.sh
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_pooling_aware_bank.sh
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_pooling_aware_bank_tmux.sh
  scripts/emotic-ddp-internal-adapter/verify_emotic_ddp_final_token_adapter_sync.sh
)
for path in "${REQUIRED_FILES[@]}"; do
  [[ -s "${path}" ]] || { echo "Missing synchronized file: ${path}" >&2; exit 1; }
done

python -m py_compile \
  emotic_final_token_adapter.py \
  train_emotic_ddp_final_token_adapter.py \
  build_emotic_ddp_final_token_adapter_bank.py \
  eval_emotic_ddp_final_token_adapter_bank.py \
  summarize_emotic_ddp_final_token_adapter_bank.py \
  summarize_emotic_ddp_final_token_safety_screen.py \
  summarize_emotic_ddp_final_token_c300_multiseed.py \
  summarize_emotic_ddp_final_token_pooling_aware_pilot.py \
  summarize_emotic_ddp_final_token_pooling_aware_bank.py \
  models/ddp.py \
  tests/test_emotic_final_token_adapter.py \
  tests/test_emotic_final_token_safety_screen.py \
  tests/test_emotic_final_token_c300_multiseed.py \
  tests/test_emotic_final_token_pooling_aware_pilot.py \
  tests/test_emotic_final_token_pooling_aware_bank.py

python -m unittest \
  tests.test_emotic_final_token_adapter \
  tests.test_emotic_final_token_safety_screen \
  tests.test_emotic_final_token_c300_multiseed \
  tests.test_emotic_final_token_pooling_aware_pilot \
  tests.test_emotic_final_token_pooling_aware_bank \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter

for script in \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_adapter_bank_train.sh \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_adapter_bank_eval.sh \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_adapter_bank_worker.sh \
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_adapter_bank.sh \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_adapter_bank_tmux.sh; do
  bash -n "${script}"
done

for script in \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_task0.sh \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_pooling_aware_task0_tmux.sh; do
  bash -n "${script}"
done

for script in \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_bank_train.sh \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_bank_worker.sh \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_pooling_aware_bank_eval.sh \
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_pooling_aware_bank.sh \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_pooling_aware_bank_tmux.sh; do
  bash -n "${script}"
done

for script in \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_training_healthcheck.sh \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_training_healthcheck_tmux.sh \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_safety_worker.sh \
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_safety_screen.sh \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_safety_screen_tmux.sh; do
  bash -n "${script}"
done

for script in \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_final_token_c300_multiseed_worker.sh \
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_final_token_c300_multiseed.sh \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_c300_multiseed_tmux.sh; do
  bash -n "${script}"
done

python - <<'PY'
from pathlib import Path

from emotic_final_token_adapter import FINAL_TOKEN_FORMULA

train = Path("train_emotic_ddp_final_token_adapter.py").read_text(encoding="utf-8")
evaluate = Path("eval_emotic_ddp_final_token_adapter_bank.py").read_text(encoding="utf-8")
runner = Path(
    "scripts/emotic-ddp-internal-adapter/"
    "run_emotic_ddp_final_token_adapter_bank_train.sh"
).read_text(encoding="utf-8")

assert FINAL_TOKEN_FORMULA == "adapt_197_tokens_then_original_ddp_pooling"
assert 'eval_splits=("test",)' not in train
assert '"test_dataset_constructed": False' in train
assert '"checkpoint_selection": (' in train
assert '"fixed last epoch"' in train
assert '"fixed final diagnostic optimizer step"' in train
assert '"validation_role": "post-training reporting only"' in train
assert 'float(args.residual_scale) != 0.03' in train
assert "FIXED_THRESHOLD = 0.5" in evaluate
assert 'task${task}.pth' in runner
assert "--residual_scale 0.03" in runner
assert "adapted = torch.where" not in Path(
    "emotic_final_token_adapter.py"
).read_text(encoding="utf-8")
assert "--training_health_check" in train
assert "automatic_winner_selection" in Path(
    "summarize_emotic_ddp_final_token_safety_screen.py"
).read_text(encoding="utf-8")
assert "final_token_drift_tensors" in train
assert "automatic_full_bank_launch" in Path(
    "summarize_emotic_ddp_final_token_c300_multiseed.py"
).read_text(encoding="utf-8")
assert "final_token_pooling_aware_losses" in train
final_token = Path("emotic_final_token_adapter.py").read_text(encoding="utf-8")
assert "F.log_softmax" in final_token
assert "log(clamped softmax)" in final_token
assert "--pooling_weight" in train
assert "--attention_weight" in train
assert "--margin_weight" in train
assert "--full_precision_training" in train
assert "_cuda_amp_enabled(device, args.full_precision_training)" in train
assert "not args.full_precision_training" not in train
pooling_bank = Path(
    "scripts/emotic-ddp-internal-adapter/"
    "run_emotic_ddp_final_token_pooling_aware_bank_train.sh"
).read_text(encoding="utf-8")
assert "MAX_STEPS=300" in pooling_bank
assert "POOLING_WEIGHT=100.0" in pooling_bank
assert "ATTENTION_WEIGHT=100.0" in pooling_bank
assert "MARGIN_WEIGHT=1.0" in pooling_bank
assert "fixed_optimizer_steps_no_validation_selection" in pooling_bank
assert "log_softmax_stable" in pooling_bank
assert "--full_precision_training" in pooling_bank
assert "--training_precision float32" in pooling_bank
print("Locked Final-token protocol audit: OK")
PY

git --no-pager diff --check
echo "SYNC_CHECK_OK: Final-token Adapter code is synchronized and validated"
