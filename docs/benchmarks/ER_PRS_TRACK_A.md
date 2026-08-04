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
may a clean commit run `configuration_locked=true` seed-0 held-out test. Seeds
1 and 2 follow only after seed 0 and artifact memory accounting are reviewed.

The universal checkpoint-free download standard applies: synchronize only the
generated `.tar.gz` and adjacent `.tar.gz.sha256`; `.pth` files remain in the
separate server checkpoint tree.
