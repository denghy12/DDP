# ASL/BAL Task-routed Adapter Bank

This branch keeps the strongest frozen-DDP inference structure unchanged:

```text
prompted CLS -> class-introduction-task Adapter
             -> Feature Difference
             -> original DDP path logits + correction
```

Only the prompt-free auxiliary classification loss used to train each task
Adapter changes. The primary comparison is:

- `weighted_bce`: the existing masked positive-weighted BCE;
- `asl`: gamma_pos=0, gamma_neg=9.8, negative clip=0.05;
- `bal_paper`: the same ASL negative term plus train-only positive class
  weights (power=1.6) and label smoothing=0.1 over the fixed 26-class label
  space.

Old and future labels remain masked. BAL counts are computed only from visible
current-task training entries. The existing BCE `pos_weight` is never combined
with ASL/BAL.

The formal comparison uses last-epoch checkpoints, fixed inference alpha=0.03,
and fixed decision threshold=0.5. Validation metrics are reporting-only.

## Verify

```bash
bash \
  scripts/emotic-ddp-internal-adapter/verify_emotic_ddp_task_adapter_bank_losses_sync.sh
```

## Run Full on eight GPUs

```bash
GPU_LIST="0 1 2 3 4 5 6 7" \
MODES="full" \
bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_losses_8gpu_tmux.sh
```

This launches nine Full runs (`3 losses x 3 seeds`) over eight GPU lanes. One
lane runs two jobs sequentially.

To include the diagnostic strict 16-shot comparison:

```bash
SESSION=ddp_task_bank_losses_all \
GPU_LIST="0 1 2 3 4 5 6 7" \
MODES="full 16shot" \
bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_losses_8gpu_tmux.sh
```

BAL class weights are constant in strict 16-shot because each active class has
exactly 16 supervised positives. Therefore Full is the primary BAL experiment.

## Outputs

- task checkpoints and train diagnostics:
  `output/emotic_ddp_task_adapter_bank_loss_<loss>_<mode>/seed*/`;
- all-task test evaluation:
  `output/emotic_ddp_task_adapter_bank_loss_<loss>_<mode>_feature_difference_seed*/`;
- JSON/CSV/HTML comparison:
  `output/emotic_ddp_task_adapter_bank_loss_comparison/`;
- orchestration logs:
  `output/emotic_ddp_task_adapter_bank_loss_pipeline/`.
