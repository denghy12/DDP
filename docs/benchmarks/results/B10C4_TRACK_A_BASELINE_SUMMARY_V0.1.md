# EMOTIC B10-C4 Track-A Baseline Summary v0.1

The layout follows the paper-style `Method / Memory / Last / Avg.` table. All
numbers are percentages on the held-out test split. Every row reports the mean
and sample standard deviation over seeds 0, 1, and 2. Hyperparameters were
transferred from the frozen B5-C3 configurations without B10-C4 test tuning.

| Method | Memory | Last mAP | Last cF1 | Last oF1 | Avg. mAP |
|---|---:|---:|---:|---:|---:|
| Sequential Fine-Tuning | 0 | 18.0 ± 0.4 | 8.6 ± 1.0 | 39.8 ± 2.5 | 22.3 ± 0.4 |
| LwF (Li & Hoiem, 2016) | 0 | 20.0 ± 2.7 | 7.4 ± 0.1 | 40.0 ± 0.4 | 24.1 ± 2.3 |
| EWC (Kirkpatrick et al., 2017) | 0 | 18.3 ± 0.8 | 8.8 ± 1.6 | 39.8 ± 11.4 | 22.7 ± 0.7 |
| AGCN (Du et al., 2022) | 0 | 18.1 ± 0.2 | 8.4 ± 1.2 | 43.2 ± 2.8 | 21.4 ± 0.3 |
| CSC (Du et al., 2024) | 0 | 21.9 ± 0.9 | 6.9 ± 1.3 | 16.3 ± 3.0 | 28.7 ± 0.3 |
| **MULTI-LANE (Min et al., 2024)** | **0** | **33.7 ± 0.2** | 27.6 ± 0.1 | 50.4 ± 0.2 | **37.4 ± 0.2** |
| L3A (Zhang et al., 2025) | 0 | 28.8 ± 0.4 | 26.4 ± 0.1 | 34.9 ± 0.3 | 32.6 ± 0.4 |
| Original-DDP-Tau2 | 0 | 31.9 ± 0.3 | 27.3 ± 1.1 | **51.1 ± 1.6** | 35.7 ± 0.4 |
| ER (Rolnick et al., 2019) | 20/class | 18.6 ± 2.0 | 8.6 ± 2.4 | 42.0 ± 4.5 | 23.0 ± 0.7 |
| PRS (Kim et al., 2020) | 20/class | 18.7 ± 0.7 | 1.1 ± 1.0 | 4.8 ± 5.1 | 23.3 ± 0.2 |
| DER++ (Buzzega et al., 2020) | 20/class | 22.4 ± 1.7 | 7.5 ± 0.0 | 39.9 ± 0.3 | 26.5 ± 1.9 |
| **KRT (Dong et al., 2023)** | **20/class** | 25.6 ± 0.6 | **28.2 ± 0.9** | 42.9 ± 0.8 | 31.8 ± 0.2 |

## Interpretation and reporting notes

- MULTI-LANE ranks first in Final mAP and Average mAP and has the lowest mean
  forgetting (`2.1909`). Original-DDP-Tau2 ranks second in both mAP measures,
  has the highest oF1, and is statistically tied with MULTI-LANE on forgetting
  at the displayed precision.
- KRT has the highest fixed-threshold cF1 but consumes a 20-per-seen-class
  replay memory. L3A is the third-ranked exemplar-free method by both mAP
  measures.
- PRS and CSC retain non-trivial ranking performance in mAP while their fixed
  threshold F1 values expose score-calibration limitations. No threshold was
  selected from held-out test labels.
- `Memory` means replay image-sample capacity, not parameter, optimizer, graph,
  or analytic-state memory. ER, PRS, DER++, and KRT use the common 20-per-seen-
  class scale, ending at a target capacity of 520 samples.
- Average mAP and forgetting depend on the number and boundaries of tasks.
  They must not be directly compared with B5-C3 values as if both protocols
  shared an identical training trajectory.
- The exact unrounded seed metrics, per-task curves, ranking, artifact hash,
  and retry audit are stored in
  `b10c4_12baseline_seed012_formal_v0.1.json`.

## Execution integrity

The 36 configuration-locked jobs used clean commit `c759b3c`. All run
manifests are eligible for the main table, use the held-out test reporting
split, and record no test-label selection or prediction reuse. The
checkpoint-free archive contains 36 canonical result bundles and 180 task
score files, excludes `.pth`, and has SHA-256
`7d905ece7a7c8838c33edd80b15507956c1d7c1dd1998e694153cd086eb9e205`.

ER seed 0 and PRS seed 0 initially exhausted GPU memory while sharing GPU 6
with DER++ seed 1. Both jobs were restarted in isolation on GPUs 6 and 7 with
the same commit, method configuration, and hyperparameters. The retry changed
only scheduling and does not invalidate the registered results. Future sweep
schedules must not colocate DER++ with ER or PRS on one 24 GiB GPU.
