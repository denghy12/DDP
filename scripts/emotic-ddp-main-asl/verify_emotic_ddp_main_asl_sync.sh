#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

EXPECTED_BRANCH="codex/emotic-ddp-main-asl"
actual_branch="$(git branch --show-current)"
if [[ "${actual_branch}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Expected branch ${EXPECTED_BRANCH}, found ${actual_branch}" >&2
  exit 1
fi

required_files=(
  bce_loss.py
  DDP.py
  opts.py
  evaluation_metrics.py
  eval_emotic_all_tasks.py
  summarize_emotic_ddp_main_losses.py
  summarize_emotic_ddp_main_loss_healthcheck.py
  summarize_emotic_ddp_main_asl_task0_ablation.py
  summarize_emotic_ddp_main_asl_g2_scale_task0.py
  tests/test_ddp_main_asl_loss.py
  tests/test_ddp_main_asl_protocol.py
  tests/test_ddp_main_asl_summary.py
  tests/test_ddp_main_asl_task0_ablation.py
  tests/test_ddp_main_asl_g2_scale_task0.py
  scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss.sh
  scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss_eval.sh
  scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss_worker.sh
  scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss_lane.sh
  scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_loss_healthcheck_tmux.sh
  scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_losses_6gpu_tmux.sh
  scripts/emotic-ddp-main-asl/wait_and_summarize_emotic_ddp_main_loss_healthcheck.sh
  scripts/emotic-ddp-main-asl/wait_and_summarize_emotic_ddp_main_losses.sh
  scripts/emotic-ddp-main-asl/run_emotic_ddp_main_asl_task0_ablation_worker.sh
  scripts/emotic-ddp-main-asl/run_emotic_ddp_main_asl_task0_ablation_lane.sh
  scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_asl_task0_ablation_tmux.sh
  scripts/emotic-ddp-main-asl/wait_and_summarize_emotic_ddp_main_asl_task0_ablation.sh
  scripts/emotic-ddp-main-asl/run_emotic_ddp_main_asl_g2_scale_task0_worker.sh
  scripts/emotic-ddp-main-asl/run_emotic_ddp_main_asl_g2_scale_task0_lane.sh
  scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_asl_g2_scale_task0_tmux.sh
  scripts/emotic-ddp-main-asl/wait_and_summarize_emotic_ddp_main_asl_g2_scale_task0.sh
)
for file in "${required_files[@]}"; do
  [[ -f "${file}" ]] || {
    echo "Missing synchronized file: ${file}" >&2
    exit 1
  }
done

python -m py_compile \
  bce_loss.py \
  DDP.py \
  opts.py \
  evaluation_metrics.py \
  eval_emotic_all_tasks.py \
  summarize_emotic_ddp_main_losses.py \
  summarize_emotic_ddp_main_loss_healthcheck.py \
  summarize_emotic_ddp_main_asl_task0_ablation.py \
  summarize_emotic_ddp_main_asl_g2_scale_task0.py \
  tests/test_ddp_main_asl_loss.py \
  tests/test_ddp_main_asl_protocol.py \
  tests/test_ddp_main_asl_summary.py \
  tests/test_ddp_main_asl_task0_ablation.py \
  tests/test_ddp_main_asl_g2_scale_task0.py

python -m unittest \
  tests.test_ddp_main_asl_loss \
  tests.test_ddp_main_asl_protocol \
  tests.test_ddp_main_asl_summary \
  tests.test_ddp_main_asl_task0_ablation \
  tests.test_ddp_main_asl_g2_scale_task0 \
  tests.test_emotic_multilabel_losses

find scripts/emotic-ddp-main-asl -type f -name '*.sh' -print0 \
  | xargs -0 -n1 bash -n

python - <<'PY'
from pathlib import Path

from opts import arg_parser

args = arg_parser().parse_args(["--ddp_classification_loss", "asl"])
assert args.ddp_asl_gamma_neg == 9.8
assert args.ddp_asl_gamma_pos == 0.0
assert args.ddp_asl_clip == 0.05
assert args.ddp_asl_eps == 1e-8
assert args.loss_w == 0.03

source = Path("DDP.py").read_text(encoding="utf-8")
assert "labels[:, low_range:high_range]" in source
assert '"checkpoint_rule": "fixed_last_epoch"' in source
assert '"validation_role": "reporting_only"' in source
assert 'eval_splits=("val",)' in source
assert '"test_used_during_training": False' in source
runner = Path(
    "scripts/emotic-ddp-main-asl/run_emotic_ddp_main_loss.sh"
).read_text(encoding="utf-8")
for locked in (
    "--epochs \"${EPOCHS}\"",
    "LOSS_WEIGHT=0.03",
    '--loss_w "${LOSS_WEIGHT}"',
    "--ddp_asl_gamma_neg 9.8",
    "--ddp_asl_gamma_pos 0.0",
    "--ddp_asl_clip 0.05",
    "--thre 0.5",
):
    assert locked in runner, locked
for locked_mild in (
    "asl_g9p8)",
    "asl_g4)",
    "--ddp_asl_gamma_neg 4.0",
    "asl_g2)",
    "--ddp_asl_gamma_neg 2.0",
    "asl_g2_lw009)",
    "LOSS_WEIGHT=0.09",
):
    assert locked_mild in runner, locked_mild

launcher = Path(
    "scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_asl_task0_ablation_tmux.sh"
).read_text(encoding="utf-8")
for locked_ablation in (
    'GPU_LIST="${GPU_LIST:-0 0}"',
    'METHODS="${METHODS:-two_way_bce asl_g9p8 asl_g4 asl_g2}"',
    'SEEDS="${SEEDS:-0 1 2}"',
):
    assert locked_ablation in launcher, locked_ablation

scale_launcher = Path(
    "scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_asl_g2_scale_task0_tmux.sh"
).read_text(encoding="utf-8")
for locked_scale in (
    'GPU_LIST="${GPU_LIST:-0 0}"',
    'SEEDS="${SEEDS:-0 1 2}"',
    'SESSION="${SESSION:-ddp_main_asl_g2_scale_task0}"',
):
    assert locked_scale in scale_launcher, locked_scale
print("Locked DDP main ASL protocol audit: OK")
PY

git diff --check
echo "SYNC_CHECK_OK: DDP main BCE/ASL code is synchronized and validated"
