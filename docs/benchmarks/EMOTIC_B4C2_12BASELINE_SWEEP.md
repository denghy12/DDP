# EMOTIC B4-C2 12-Baseline Sweep

## Protocol

`emotic_b4c2_v0.1` is a separate Track-A protocol over the same alphabetical
26-class EMOTIC order used by B5-C3 and B10-C4. Task 0 contains four classes;
each of the eleven incremental tasks adds two classes:

| Task | New classes | Seen classes |
|---:|---|---:|
| 0 | Affection, Anger, Annoyance, Anticipation | 4 |
| 1 | Aversion, Confidence | 6 |
| 2 | Disapproval, Disconnection | 8 |
| 3 | Disquietment, Doubt/Confusion | 10 |
| 4 | Embarrassment, Engagement | 12 |
| 5 | Esteem, Excitement | 14 |
| 6 | Fatigue, Fear | 16 |
| 7 | Happiness, Pain | 18 |
| 8 | Peace, Pleasure | 20 |
| 9 | Sadness, Sensitivity | 22 |
| 10 | Suffering, Surprise | 24 |
| 11 | Sympathy, Yearning | 26 |

Training sees current labels only. Validation selects checkpoints from current
labels only. Held-out test evaluates seen labels and cannot modify a method
configuration. cF1 and oF1 use the fixed global threshold 0.5. Scores and
predictions from B5-C3 or B10-C4 may not be reused because the task trajectory
and protocol hash differ.

## Methods and frozen configuration policy

The sweep runs seeds 0, 1, and 2 for the same 12 registered baselines:

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

Architecture, optimizer, learning rate, epoch, early stopping, loss,
pseudo-label, and replay settings are transferred unchanged from their frozen
B5-C3 registrations. There is no B4-C2 held-out-test tuning. ER, PRS, DER++,
and KRT keep the shared 20-per-seen-class scale; task capacities are
`80/120/160/200/240/280/320/360/400/440/480/520`.

## OOM-safe eight-GPU dual-slot scheduling

There are 36 independent method/seed jobs. The scheduler uses all eight GPUs
and at most two processes per GPU, so the maximum global concurrency remains
16. Longest expected jobs start first, using the measured B10-C4 duration order
as the initial estimate: Original-DDP, KRT, DER++, CSC, MULTI-LANE, PRS, EWC,
LwF, ER, Fine-Tuning, AGCN, and L3A. A completed slot is immediately backfilled
with the earliest compatible pending job.

Slot count alone is not a memory guarantee. Every method therefore receives a
conservative memory reservation, and the combined reservation on one GPU may
not exceed `20000 MiB`:

| Method | Reservation (MiB) | Method | Reservation (MiB) |
|---|---:|---|---:|
| Original-DDP-Tau2 | 6000 | KRT | 12000 |
| MULTI-LANE | 3500 | CSC | 9000 |
| LwF | 6500 | DER++ | 14000 |
| EWC | 8000 | ER | 9500 |
| PRS | 9500 | Fine-Tuning | 6500 |
| AGCN | 4500 | L3A | 8000 |

This leaves more than 4 GiB physical headroom on a 24 GiB RTX 4090. In
addition, DER++ is explicitly forbidden from sharing a GPU with ER or PRS,
matching the B10-C4 retry audit. Over-budget pairs such as DER++ + LwF or
KRT+ER are also rejected automatically. The scheduler scans later pending jobs
for a compatible backfill rather than leaving the second slot unused whenever
a safe pairing exists.

The launcher requires all eight requested cards to have at least `23000 MiB`
free before preflight. It also requires a clean Git checkout and validates the
12-task split, both replay contracts, Core tests, and legacy regressions before
creating tmux.

## One-command execution

From a clean server checkout of `codex/emotic-b4c2-12baseline-sweep`:

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

RUN_ID="b4c2_12baseline_2slot_seed012_$(date +%Y%m%d_%H%M%S)" \
SESSION=emotic_b4c2_12baseline \
GPUS="0 1 2 3 4 5 6 7" \
SLOTS_PER_GPU=2 \
GPU_MEMORY_BUDGET_MIB=20000 \
bash scripts/emotic-mlcil/launch_b4c2_12baseline_8gpu_tmux.sh
```

The launcher prints the exact run ID and paths. Progress is the number of
successful jobs out of 36:

```bash
watch -n 10 'find /mnt/haoyuan/workspace/emotic_benchmark_runs/b4c2_12baseline_v0.1/<RUN_ID>/runtime_state -name "*.done.json" | wc -l'
```

The scheduler and workers remain inside one tmux session:

```bash
tmux attach -t emotic_b4c2_12baseline
```

After all jobs and aggregation succeed, download only:

```text
/mnt/haoyuan/workspace/emotic_benchmark_runs/b4c2_12baseline_v0.1/<RUN_ID>/download_packages/<RUN_ID>.tar.gz
/mnt/haoyuan/workspace/emotic_benchmark_runs/b4c2_12baseline_v0.1/<RUN_ID>/download_packages/<RUN_ID>.tar.gz.sha256
```

The package contains all 36 canonical bundles, 432 task score files, aggregate
JSON/Markdown, scheduler events, manifests, and logs. The universal download
standard excludes every `.pth` checkpoint.
