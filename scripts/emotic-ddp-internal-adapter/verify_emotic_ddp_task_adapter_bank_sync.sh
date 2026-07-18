#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

EXPECTED_BRANCH="codex/emotic-ddp-task-adapter-bank"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Expected branch ${EXPECTED_BRANCH}, found ${CURRENT_BRANCH}" >&2
  exit 1
fi

REQUIRED_FILES=(
  emotic_task_adapter_bank.py
  train_emotic_ddp_task_adapter.py
  build_emotic_ddp_task_adapter_bank.py
  eval_emotic_ddp_task_adapter_bank.py
  audit_emotic_task_adapter_data.py
  summarize_emotic_ddp_task_adapter_bank.py
  tests/test_emotic_task_adapter_bank.py
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_train.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_eval.sh
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_worker.sh
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_task_adapter_bank.sh
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_tmux.sh
)
for path in "${REQUIRED_FILES[@]}"; do
  [[ -s "${path}" ]] || { echo "Missing synchronized file: ${path}" >&2; exit 1; }
done

python -m py_compile \
  emotic_task_adapter_bank.py \
  train_emotic_ddp_task_adapter.py \
  build_emotic_ddp_task_adapter_bank.py \
  eval_emotic_ddp_task_adapter_bank.py \
  audit_emotic_task_adapter_data.py \
  summarize_emotic_ddp_task_adapter_bank.py \
  train_emotic_ddp_prompt_free_auxiliary.py \
  models/ddp.py \
  src/helper_functions/emotic_loader.py \
  tests/test_emotic_task_adapter_bank.py

python -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary

for script in \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_train.sh \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_eval.sh \
  scripts/emotic-ddp-internal-adapter/run_emotic_ddp_task_adapter_bank_worker.sh \
  scripts/emotic-ddp-internal-adapter/wait_and_summarize_emotic_ddp_task_adapter_bank.sh \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_tmux.sh; do
  bash -n "${script}"
done

python - <<'PY'
from pathlib import Path

from emotic_task_adapter_bank import TASK_CLASS_RANGES, class_to_task_map

assert TASK_CLASS_RANGES == (
    (0, 5), (5, 8), (8, 11), (11, 14),
    (14, 17), (17, 20), (20, 23), (23, 26),
)
assert class_to_task_map() == (
    0, 0, 0, 0, 0,
    1, 1, 1,
    2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5, 5, 5,
    6, 6, 6,
    7, 7, 7,
)
source = Path("train_emotic_ddp_task_adapter.py").read_text(encoding="utf-8")
assert 'eval_splits=("test",)' not in source
assert '"test_labels_used": False' in source
assert '"inference_alpha_locked": args.inference_alpha' in source
print("Locked protocol audit: OK")
PY

git --no-pager diff --check
echo "SYNC_CHECK_OK: Task-routed Adapter Bank code is synchronized and validated"

