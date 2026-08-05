# ER and PRS Track-A Contract

ER and PRS are the first methods governed by the shared
[Replay Memory Contract](REPLAY_MEMORY_CONTRACT.md). They use the same
trainable OpenAI CLIP ViT-B/16 visual encoder and expanding linear multi-label
head as Sequential Fine-Tuning. Neither method adds an Adapter or uses CLIP
text features.

## ER controlled baseline

ER is a repository-native control rather than a claim to reproduce one
specific external implementation. It uses conventional reservoir sampling:
the first items fill available memory, and a later unique stream observation
uniformly draws a replacement position from the observations seen so far.
Stable-ID duplicates merge label visibility and do not consume a second slot.

All other behavior comes from the shared Track-A contract: current-label BCE,
masked replay BCE, 1:1 replay/current samples, equal per-sample weight,
validation-mAP checkpoint selection, and a single task-end memory update pass.
Its purpose is to isolate whether PRS's partition-aware buffer improves over a
plain replay policy at exactly the same memory and training exposure.

## PRS source audit

- paper: *Imbalanced Continual Learning with Partitioning Reservoir Sampling*,
  ECCV 2020;
- official repository: <https://github.com/cdjkim/PRS>;
- fixed commit: `136cee1863af03cc914dc05dfd41bda8b7bc0bf2`;
- fixed Git tree: `8e08f4ea1ffcce30c73f9453d6db522d7cba7c77`;
- license: MIT;
- original multi-label datasets: COCOseq and NUS-WIDEseq;
- released COCO backbone: fine-tuned ImageNet-pretrained ResNet-101;
- released multi-label loss: `MultiLabelSoftMarginLoss`;
- released COCO memory: 2,000 samples;
- released replay/current ratio: 1:1;
- released COCO partition allocation power: `q=-0.03`.

The fixed source remains outside this repository under
`baseline_sources/PRS-master`. The benchmark contains an independent policy
implementation and a fixed-source executable oracle. Preflight verifies the
license, COCO config, and PRS policy hashes, then compares retained sample IDs,
positive-label observation counts, and target partition proportions on a
deterministic multi-label stream.

The registered SHA-256 values are:

- `LICENSE`: `116db92b1b611a171ebe1a30c2290b91b42936a7cc6ce33b43b666b5fd07755a`;
- `code/configs/mlab_prs-coco.yaml`:
  `23150ec815627f96725c947be30e56a9279ed24156e646942404da2dd9d0074a`;
- `code/models/reservoir/mlab_stratified_reservoir.py`:
  `d1e4c97e7c21b28e6ce63a441b358185db40a148a95feddfe155771bd942adfb`.

## Track-A mapping

The port replaces ResNet-101 only with the common trainable CLIP visual tower.
It retains PRS's multi-label partition logic:

- each positive label is a substream;
- observed positive frequency `n_i` produces target allocation proportional to
  `n_i^q`;
- admission probability favors underallocated and infrequent positive labels;
- eviction first selects an overrepresented substream, protects labels whose
  partitions are not over target, and minimizes remaining partition error.

The benchmark changes three surrounding source conventions, all explicitly:

1. capacity is the shared `20 × seen classes` schedule rather than 2,000;
2. task-level partial label visibility replaces the source's complete
   multi-hot COCO target; stored masks ensure loss uses only known columns; and
3. buffer update occurs in one pass at task end rather than online after each
   source optimizer step.

ER and PRS otherwise share the exact optimizer and training defaults inherited
from the registered CLIP classifier: AdamW, visual learning rate `1e-5`, head
learning rate `1e-4`, weight decay `1e-4`, ten epochs maximum, validation-only
early stopping patience three, AMP, TF32, and gradient-norm clipping at 1.0.

## Development gates

Before seed-0 validation, the server must pass:

- the complete Core/baseline unit suite and selected 17 legacy regressions;
- replay contract, hidden-label, deterministic-buffer, checkpoint, and method
  interface tests;
- PRS fixed-source operator equivalence; and
- a current-plus-replay CLIP training memory smoke at the intended batch size.

Validation freezes optimizer settings and all replay policy values. Only then
may a clean commit run with `configuration_locked=true`. After the completed
seed-0 validation/artifact review, the user authorized seeds 0--2 to run
together under the same frozen configuration; test output cannot change it.

## Frozen seed-0 validation

Seed-0 validation passed from clean commit `9a716a5`. ER reached Final mAP
`26.6088`, Average mAP `32.1604`, and Forgetting `6.0568`; PRS reached Final
mAP `28.4171`, Average mAP `32.9586`, and Forgetting `5.8944`. Both finished
with exactly 520 unique replay samples. Their serialized float32 image buffers
occupied `313,206,300` and `313,206,184` bytes respectively.

ER attempted 6,477 optimizer updates and guarded one AMP overflow. PRS
attempted 8,384 and guarded two. Neither run reported NaN, OOM, or a training
traceback. The shared current-32 plus replay-32 CUDA smoke peaked at
`5827.2 MiB`. PRS's external-source oracle retained exactly the same sample
IDs as the fixed source, had zero observation-count error, and differed in
target partition proportions by at most `1.58e-8`.

This evidence freezes the following values before held-out access:

- memory capacity `20 × seen classes`, ending at 520 samples;
- replay/current ratio `1:1` and equal sample weight;
- PRS allocation power `q=-0.03`;
- AdamW, visual/head learning rates `1e-5/1e-4`, weight decay `1e-4`, and
  gradient clipping at 1.0;
- ten epochs maximum with validation-only early stopping patience three; and
- the global F1 threshold at 0.5, despite PRS's lower validation oF1.

The machine-readable evidence is registered in
[`results/er_prs_seed0_validation_v0.1.json`](results/er_prs_seed0_validation_v0.1.json).
Formal test execution must use confirmation `REPLAY_20C_TRACK_A_V0_1` and a
clean, explicitly pinned Git commit.

## Locked formal execution

The single-GPU formal launcher runs three isolated seeds concurrently on
physical GPU 0. Because one process peaked at `5827.2 MiB`, it never starts all
six ER/PRS workers together. It runs ER seeds 0--2 as the first three-process
wave, waits for all three to succeed, then runs PRS seeds 0--2 as the second
wave. A failed worker blocks aggregation and packaging; it is not silently
retried. The result validator requires eight task metrics and canonical score
files for every method/seed, verifies the locked configuration and memory
schedule, aggregates with sample standard deviation, and emits one six-bundle
checkpoint-free archive.

## Registered formal result

Locked held-out seeds 0--2 completed from clean commit `84c79eb` in the
planned two GPU-0 waves. All six workers exited successfully without OOM,
training rerun, NaN, or traceback. Formal preflight passed 133 Core/baseline
tests (two skips), 17 legacy regressions, the fixed-source PRS oracle, and the
current-32 plus replay-32 memory smoke (`5827.4 MiB` peak).

The three-seed results are:

| Method | Final mAP | Average mAP | Forgetting | Final cF1 | Final oF1 |
|---|---:|---:|---:|---:|---:|
| ER | `20.3230 ± 1.4593` | `25.9259 ± 1.4637` | `8.6986 ± 0.6386` | `21.2202 ± 3.6461` | `45.4025 ± 0.4687` |
| PRS | `20.5603 ± 0.4442` | `26.3254 ± 1.5723` | `8.7479 ± 1.6186` | `20.8966 ± 1.5407` | `31.3758 ± 2.5214` |

Values are mean ± sample standard deviation. PRS minus ER is only `+0.2373`
Final mAP and `+0.3995` Average mAP; paired Final-mAP differences are
`+1.6620/-0.7072/-0.2428`. The result supports a small average gain and lower
PRS Final-mAP variance, not a claim of consistent superiority. Forgetting is
effectively unchanged. PRS's `-14.0267` fixed-0.5 oF1 difference is reported
as a score-calibration limitation; the held-out result did not trigger a
threshold scan.

Every run retained exactly 520 samples. ER used
`313,206,297 ± 33.6` bytes and PRS `313,206,120.3 ± 62.1` bytes, so their
sample and byte budgets are effectively identical. Task-0 score tensors are
identical between ER and PRS for each seed, as expected before replay begins.
Across all tasks, ER recorded two guarded AMP overflows in 22,764 optimizer
attempts and PRS recorded two in 24,266; no training instability followed.

The full per-seed metrics, task curves, class forgetting, memory accounting,
paired comparison, execution record, oracle result, and archive SHA-256 are in
[`results/er_prs_seed012_formal_v0.1.json`](results/er_prs_seed012_formal_v0.1.json).
The registered archive SHA-256 is
`ac7a7c38ad797743424aa6b14a670decef03317999902bef0066381378e9ad66`.
ER and PRS Track A are frozen; neither method may be rerun or tuned from these
held-out metrics.

Server entry point after synchronizing the frozen commit:

```bash
RUN_ID="replay_er_prs_formal_seed012_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
SESSION=emotic_replay_formal_seed012 \
EXPECTED_GIT_COMMIT="$(git rev-parse HEAD)" \
CONFIGURATION_LOCKED_CONFIRMATION=REPLAY_20C_TRACK_A_V0_1 \
bash scripts/emotic-mlcil/launch_replay_formal_seed012_tmux.sh
```

The universal checkpoint-free download standard applies: synchronize only the
generated `.tar.gz` and adjacent `.tar.gz.sha256`; `.pth` files remain in the
separate server checkpoint tree.
