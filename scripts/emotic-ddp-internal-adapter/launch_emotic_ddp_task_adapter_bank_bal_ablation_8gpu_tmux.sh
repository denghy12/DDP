#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

# ASL and BAL-paper were completed by the primary loss comparison.  Only the
# two missing component ablations are trained here; all four methods are
# included in the final summary.
export GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
export MODES="${MODES:-full}"
export LOSSES="${LOSSES:-asl_smoothing asl_positive_weight}"
export SUMMARY_LOSSES="${SUMMARY_LOSSES:-asl asl_smoothing asl_positive_weight bal_paper}"
export SEEDS="${SEEDS:-0 1 2}"
export SESSION="${SESSION:-ddp_task_bank_bal_ablation}"
export STATE_DIR="${STATE_DIR:-./output/emotic_ddp_task_adapter_bank_bal_component_ablation_pipeline}"
export SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-./output/emotic_ddp_task_adapter_bank_bal_component_ablation}"

for existing_loss in asl bal_paper; do
  for seed in 0 1 2; do
    existing_eval="./output/emotic_ddp_task_adapter_bank_loss_${existing_loss}_full_feature_difference_seed${seed}/evaluation_summary.json"
    [[ -s "${existing_eval}" ]] || {
      echo "Missing reusable result: ${existing_eval}" >&2
      exit 1
    }
  done
  for task in 0 1 2 3 4 5 6 7; do
    existing_distribution="./output/emotic_ddp_task_adapter_bank_loss_${existing_loss}_full/seed0/task${task}/class_distribution.json"
    [[ -s "${existing_distribution}" ]] || {
      echo "Missing reusable class counts: ${existing_distribution}" >&2
      exit 1
    }
  done
done

bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_losses_8gpu_tmux.sh
