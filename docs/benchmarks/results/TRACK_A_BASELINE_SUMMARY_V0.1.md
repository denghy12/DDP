# EMOTIC B5-C3 Track-A Baseline Summary v0.1

The layout follows the paper-style `Method / Memory / Last / Avg.` table. All
numbers are percentages on the held-out test split. Registered methods report
the mean and sample standard deviation over seeds 0, 1, and 2.

| Method | Memory | Last mAP | Last cF1 | Last oF1 | Avg. mAP |
|---|---:|---:|---:|---:|---:|
| Sequential Fine-Tuning | 0 | 18.2 ± 0.8 | 13.9 ± 1.4 | 36.9 ± 6.2 | 23.8 ± 0.4 |
| LwF (Li & Hoiem, 2016) | 0 | 23.3 ± 1.4 | 16.8 ± 2.1 | 46.5 ± 0.7 | 28.6 ± 1.1 |
| EWC (Kirkpatrick et al., 2017) | 0 | 20.4 ± 1.3 | 16.5 ± 2.3 | 42.7 ± 4.2 | 25.0 ± 0.4 |
| AGCN (Du et al., 2022) | 0 | 17.9 ± 0.4 | 13.7 ± 0.4 | 47.9 ± 3.5 | 21.8 ± 0.1 |
| CSC (Du et al., 2024) | 0 | 20.6 ± 0.5 | 2.3 ± 0.7 | 3.7 ± 0.7 | 30.4 ± 0.2 |
| **MULTI-LANE (Min et al., 2024)** | **0** | **31.3 ± 0.1** | **31.8 ± 0.2** | **49.1 ± 0.1** | **38.0 ± 0.5** |
| L3A (Zhang et al., 2025) | 0 | 28.8 ± 0.6 | 26.5 ± 0.1 | 34.8 ± 0.1 | 33.8 ± 1.3 |
| Original-DDP-Tau2 | 0 | 30.0 ± 0.2 | 29.5 ± 0.7 | 48.3 ± 0.5 | 37.2 ± 0.5 |
| ER (Rolnick et al., 2019) | 20/class | 20.3 ± 1.5 | 21.2 ± 3.6 | 45.4 ± 0.5 | 25.9 ± 1.5 |
| PRS (Kim et al., 2020) | 20/class | 20.6 ± 0.4 | 20.9 ± 1.5 | 31.4 ± 2.5 | 26.3 ± 1.6 |
| DER++ (Buzzega et al., 2020) | 20/class | 23.1 ± 1.2 | 20.0 ± 1.1 | 44.9 ± 0.6 | 30.4 ± 0.4 |
| KRT (Dong et al., 2023) | 20/class | 22.2 ± 2.8 | 24.5 ± 2.4 | 36.7 ± 2.0 | 30.1 ± 2.4 |

## Provisional project result

| Method | Memory | Last mAP | Last cF1 | Last oF1 | Avg. mAP |
|---|---:|---:|---:|---:|---:|
| Modified DDP (seed 0 only) | 0 | 30.8 | 31.5 | 49.3 | 37.7 |

The provisional DDP row is not used for ranking or boldface because the method
is still changing and does not yet have a frozen three-seed result.

## Reporting notes

- `Memory` means replay image-sample capacity, not parameter, optimizer, or
  analytic-state memory. A value of 0 therefore means exemplar-free.
- ER, PRS, and DER++ end at exactly 520 samples. KRT uses the same 20/class
  target but stable-ID deduplication yields `509.7 ± 5.9` final samples.
- DER++ additionally stores capture-time logits and masks; equal sample counts
  do not imply equal byte budgets.
- cF1 and oF1 use the protocol-wide fixed threshold 0.5. No held-out threshold
  calibration is applied.
- Joint Training, OCDM, and EmoGrowth/AESL are omitted because they do not have
  eligible registered results.
- The exact unrounded data and per-row provenance are stored in
  `track_a_baseline_summary_v0.1.json`. A paper-ready LaTeX fragment is stored
  in `track_a_baseline_summary_v0.1.tex`.
