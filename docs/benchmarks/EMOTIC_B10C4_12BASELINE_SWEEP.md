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

There are 36 independent method/seed jobs. The registered execution used eight
physical GPUs with two scheduler slots per GPU, for at most 16 concurrent jobs.
It started estimated long-running methods first and immediately backfilled a
slot when its current job completed. The largest registered single-process
memory smoke was DER++ at `8434.2 MiB`; however, the sum of isolated smoke
peaks did not bound all overlapping allocator and dataloader transients.

The initial priority is Original-DDP, KRT, MULTI-LANE, CSC, LwF, DER++, EWC,
ER, PRS, Fine-Tuning, AGCN, and L3A. Priority affects only scheduling, never
method configuration or result aggregation. Dataloader workers are fixed at
two per process except AGCN and Original-DDP, whose frozen settings use zero.
AGCN retains its frozen `train=8/eval=32` batch sizes; Original-DDP retains
`train=8/eval=1`.

The formal run demonstrated that a generic two-slot rule is not universally
safe: ER seed 0 and PRS seed 0 exhausted GPU memory when colocated on GPU 6
with DER++ seed 1. Both jobs subsequently completed in isolation without any
algorithm or hyperparameter change. Future schedules must treat DER++ and
ER/PRS as mutually incompatible colocations on a 24 GiB card, or run one of
those methods per GPU. The historical launcher and command below document the
registered execution; they are not a claim that every two-job pairing is safe.

## One-command execution

From a clean server checkout of the sweep branch:

```bash
RUN_ID="b10c4_12baseline_seed012_$(date +%Y%m%d_%H%M%S)" \
SESSION=emotic_b10c4_12baseline \
GPUS="0 1 2 3 4 5 6 7" \
SLOTS_PER_GPU=2 \
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

## Registered formal result

The configuration-locked held-out run
`b10c4_12baseline_2slot_seed012_20260805_163640` completed all 36 method/seed
jobs from clean commit `c759b3c0662dd121a8204ec65fa925dc9176550e`.
Every run manifest is eligible for the main table, records the held-out test
reporting split, and confirms that test labels were not used for selection.
The aggregation is the mean and sample standard deviation over seeds 0--2.

| Rank by Final mAP | Method | Final mAP | Average mAP | Forgetting |
|---:|---|---:|---:|---:|
| 1 | MULTI-LANE | `33.6902 ± 0.1599` | `37.4430 ± 0.1702` | `2.1909 ± 0.0593` |
| 2 | Original-DDP-Tau2 | `31.9185 ± 0.3279` | `35.6599 ± 0.3993` | `2.1929 ± 0.0254` |
| 3 | L3A | `28.7621 ± 0.3590` | `32.6125 ± 0.3756` | `2.3856 ± 0.1918` |
| 4 | KRT | `25.6310 ± 0.5615` | `31.7631 ± 0.1576` | `6.4666 ± 1.0907` |

The complete 12-method table is registered in
[`results/B10C4_TRACK_A_BASELINE_SUMMARY_V0.1.md`](results/B10C4_TRACK_A_BASELINE_SUMMARY_V0.1.md),
with exact metrics and provenance in
[`results/b10c4_12baseline_seed012_formal_v0.1.json`](results/b10c4_12baseline_seed012_formal_v0.1.json).
The checkpoint-free archive contains 36 bundles, 180 canonical task-score
files, and 516 manifest-tracked files. Its SHA-256 is
`7d905ece7a7c8838c33edd80b15507956c1d7c1dd1998e694153cd086eb9e205`;
the outer checksum, all outer manifest entries, and all nested sync manifests
passed independent verification, and no `.pth` is present.

The retained retry audit identifies only two scheduling failures: ER seed 0
and PRS seed 0 initially OOMed while sharing GPU 6 with DER++ seed 1. ER was
rerun alone on GPU 6 and PRS alone on GPU 7. Both completed with the original
clean commit and frozen configuration, so no result was selected or tuned from
held-out behavior. B10-C4 Track A v0.1 is frozen and must not be tuned from
these test results.
