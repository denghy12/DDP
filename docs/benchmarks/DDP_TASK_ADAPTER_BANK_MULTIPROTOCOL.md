# DDP Task Adapter Bank: B10-C4 and B4-C2

This extension evaluates the strongest full-data, task-routed Feature
Difference Adapter Bank under the frozen EMOTIC Track-A protocols:

- `emotic_b10c4_v0.1`: task sizes `[10, 4, 4, 4, 4]`;
- `emotic_b4c2_v0.1`: task sizes `[4, 2, ..., 2]` (12 tasks).

The class order is the same frozen alphabetical 26-class order used by B5-C3.
Only the task boundaries change. Routing is derived from the protocol file;
there are no B5-C3 constants in the formal method.

## Locked method

For each task, the method loads the corresponding frozen
`Original-DDP-Tau2` checkpoint from the completed 12-baseline run. It never
reuses predictions from another protocol and never retrains or changes the DDP
prompt parameters.

The task Adapter is trained through DDP's prompt-free global CLIP CLS route
using all train persons that contain at least one current-task positive.
Only current-task target columns are exposed by the benchmark data contract.
The Adapter objective is positive-weighted BCE plus identity regularization.
The checkpoint is selected by current-task validation mAP, with the earliest
epoch winning ties.

Locked Adapter settings are transferred from the best B5-C3 Full Task Bank:

| Setting | Value |
|---|---:|
| Adapter | 512 → 128 → 512 residual MLP |
| Epochs | 50 |
| AdamW learning rate | 0.001 |
| Weight decay | 0.0001 |
| Training residual scale | 0.1 |
| Identity-loss weight | 0.1 |
| Positive-weight cap | 20 |
| Inference alpha | 0.03 |
| Inference route | prompted CLS Feature Difference |
| Class routing | class introduction task |

Task 0 starts from the identity Adapter initialization. Every later Adapter
starts from the frozen task-0 anchor, matching the B5-C3 reference. Previously
learned Adapters remain frozen.

## Evaluation contract

- Track A; train exposes current labels only.
- Evaluation includes seen classes only and samples intersecting seen classes.
- One fixed global threshold `0.5`; no task-wise or class-wise threshold scan.
- Adapter checkpoint selection uses validation mAP only.
- Held-out test is used once after configuration lock and never for selection.
- Formal results are reported over seeds `0, 1, 2` with sample standard
  deviation.

## Server workflow

Run the synchronization audit first:

```bash
bash scripts/emotic-mlcil/verify_task_adapter_bank_multiprotocol_sync.sh
```

Then launch six independent protocol/seed jobs:

```bash
GPU_LIST="0 1 2 3 4 5" bash \
  scripts/emotic-mlcil/launch_task_adapter_bank_b10c4_b4c2_tmux.sh
```

The launcher reuses the server-only DDP checkpoints stored under the completed
B10-C4 and B4-C2 12-baseline run roots. It writes a new run below
`/mnt/haoyuan/workspace/emotic_benchmark_runs/task_adapter_bank_multiprotocol_v0.1/`.
Checkpoint-free per-seed bundles and the combined JSON/HTML summary are the
only artifacts that need to be synchronized to the local machine.
