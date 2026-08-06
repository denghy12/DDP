#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/opt/conda/envs/ddp/bin/python}"

expected_branch="codex/emotic-ddp-task-bank-b10c4-b4c2"
[[ "$(git branch --show-current)" == "${expected_branch}" ]] || {
  echo "Expected branch ${expected_branch}" >&2
  exit 1
}

"${PYTHON}" -m py_compile \
  emotic_task_adapter_bank.py \
  benchmarks/emotic_mlcil/methods/task_adapter_bank/method.py \
  benchmarks/emotic_mlcil/runner.py \
  summarize_emotic_ddp_task_bank_multiprotocol.py

"${PYTHON}" -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.emotic_mlcil.test_task_adapter_bank_method

for script in \
  scripts/emotic-mlcil/run_task_adapter_bank_multiprotocol_job.sh \
  scripts/emotic-mlcil/launch_task_adapter_bank_b10c4_b4c2_tmux.sh \
  scripts/emotic-mlcil/wait_and_summarize_task_adapter_bank_multiprotocol.sh; do
  bash -n "${script}"
done

git diff --check
echo "SYNC_CHECK_OK: B10-C4/B4-C2 Task Bank code is synchronized and validated"
