# EMOTIC B10-C4 12-Baseline Sweep

## Protocol

The sweep is a separate Track-A protocol named `emotic_b10c4_v0.1`. It keeps
the same alphabetical 26-class order as B5-C3 and changes only the task
boundaries:

| Task | New classes | Seen classes |
|---:|---|---:|
| 0 | Affection, Anger, Annoyance, Anticipation, Aversion, Confidence, Disapproval, Disconnection, Disquietment, Doubt/Confusion | 10 |
| 1 | Embarrassment, Engagement, Esteem, Excitement | 14 |
| 2 | Fatigue, Fear, Happiness, Pain | 18 |
| 3 | Peace, Pleasure, Sadness, Sensitivity | 22 |
| 4 | Suffering, Surprise, Sympathy, Yearning | 26 |

Training sees current labels only. Validation selects checkpoints from current
labels only. Held-out test evaluates seen labels and cannot change any method
configuration. cF1 and oF1 retain the global threshold 0.5.

## Methods and configuration policy

The sweep runs seeds 0, 1, and 2 for these 12 registered methods:

1. Sequential Fine-Tuning
2. LwF
3. EWC with validation-frozen `lambda=1e6`
4. AGCN
5. CSC
6. MULTI-LANE
7. L3A
8. Original-DDP-Tau2 with `T=1→2, gamma=0.7`
9. ER
10. PRS
11. DER++ with `alpha=beta=0.5`
12. KRT

All architecture, optimizer, learning-rate, epoch, early-stopping, loss,
pseudo-label, and replay settings are transferred unchanged from the frozen
B5-C3 configurations. There is no B10-C4 test tuning. ER, PRS, DER++, and KRT
retain the 20-per-seen-class memory scale; the five task capacities are
`200/280/360/440/520`.

## Eight-GPU scheduling

There are 36 independent method/seed jobs. The scheduler uses eight physical
GPU slots with exactly one training process per GPU. It starts estimated
long-running methods first and immediately backfills a GPU when its current
job completes. This avoids the unsafe three-process DER++/KRT packing used in
some earlier single-GPU experiments while keeping all eight GPUs occupied for
most of the sweep.

The initial priority is Original-DDP, KRT, MULTI-LANE, CSC, LwF, DER++, EWC,
ER, PRS, Fine-Tuning, AGCN, and L3A. Priority affects only scheduling, never
method configuration or result aggregation. Dataloader workers are fixed at
two per process except AGCN and Original-DDP, whose frozen settings use zero.
AGCN retains its frozen `train=8/eval=32` batch sizes; Original-DDP retains
`train=8/eval=1`. Each GPU runs one process, so the measured DER++ peak does not
need a multi-process memory estimate.

## One-command execution

From a clean server checkout of the sweep branch:

```bash
RUN_ID="b10c4_12baseline_seed012_$(date +%Y%m%d_%H%M%S)" \
SESSION=emotic_b10c4_12baseline \
GPUS="0 1 2 3 4 5 6 7" \
bash scripts/emotic-mlcil/launch_b10c4_12baseline_8gpu_tmux.sh
```

The launcher verifies the clean commit, all eight free GPUs, the alphabetic
B10-C4 task split, replay capacities, Core tests, and legacy regressions before
creating tmux. The tmux worker runs all 36 jobs, validates five task artifacts
per bundle, aggregates mean/sample-standard-deviation results, and creates one
checkpoint-free package containing all results and logs.

Progress is the number of completed jobs out of 36:

```bash
RUN_ID=<the-run-id-printed-by-the-launcher>
watch -n 10 "find /mnt/haoyuan/workspace/emotic_benchmark_runs/b10c4_12baseline_v0.1/$RUN_ID/runtime_state -name '*.done.json' | wc -l"
```

The only files that need downloading after success are:

```text
$RUN_OUTPUT_ROOT/download_packages/$RUN_ID.tar.gz
$RUN_OUTPUT_ROOT/download_packages/$RUN_ID.tar.gz.sha256
```

The archive includes all 36 canonical result bundles, the aggregate JSON and
Markdown table, scheduler events, per-job logs, and manifests. It explicitly
excludes every `.pth` checkpoint.
