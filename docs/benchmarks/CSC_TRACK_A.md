# CSC Track-A Source Audit and Porting Contract

## Scope and immutable source

This document fixed the source and protocol-safe Track-A interpretation of
Confidence Self-Calibration (CSC) before the adapter was implemented and
remains its audit/design contract. It is not a claim that CSC results already
exist.

Development branch: `codex/emotic-baseline-csc`, created from frozen KRT
Track-A commit `029eda42269f053b6306eed5522f8c967be44edb`.

- Paper: *Confidence Self-Calibration for Multi-Label Class-Incremental
  Learning*, ECCV 2024, arXiv `2403.12559`.
- Official author repository: <https://github.com/Kaile-Du/CSC>
- Fixed upstream commit:
  `0bab38a00d6e0555f2df855ae2fe8db1fea68b12` (`main`).
- Fixed GitHub source archive SHA-256:
  `588a098a1c7f3d813dee7df777c283fd27768a08a125d4e60b11b2d6ebdb7faa`.
- External read-only extraction:
  `/Users/denghaoyuan/workspace/MyCode/baseline_sources/csc_release_0bab38a/`.
- Earlier local snapshot:
  `/Users/denghaoyuan/workspace/MyCode/baseline_sources/CSC-main/`.

The earlier snapshot is byte-identical to the fixed archive extraction. It has
no `.git` metadata, so the fixed archive and its digest, rather than the old
directory name, provide its immutable provenance.

No `LICENSE`, `COPYING`, or `NOTICE` file is present at the fixed commit. The
upstream source is therefore reference-only: no upstream CSC source file is to
be copied into this repository unless the authors later publish compatible
license terms. The benchmark port must be an independently written thin
implementation of the documented operations.

## Audited method

CSC addresses task-level partial labels with two method-specific components:

1. A dynamically expanding class-incremental graph convolutional network
   (CI-GCN). Class activation maps produce per-label visual nodes. A learned
   general class-relation matrix and an image-specific relation matrix update
   those nodes before a graph classification branch.
2. Max-entropy confidence calibration over sigmoid outputs. This is combined
   with current-label classification and old-model sigmoid distillation.

The official release trains only on the current task loader. Although it
constructs buffer-related objects and exposes replay arguments, its published
`CSC_MLCIL.train_test()` path neither selects exemplars nor samples the buffer.
CSC is therefore registered with replay memory `0 samples / 0 bytes`. Adding
replay would define a different method.

The release uses the mean of a class-activation classification branch and a
CI-GCN branch as its logits. At task boundaries it expands the class activation
head, general relationship matrix, image-specific relationship generator,
graph output head, and identity mask to the complete seen-label width. A frozen
copy of the previous model supplies independent sigmoid targets for old labels.

## Track-A mapping

Track A changes only the visual feature extractor and dataset plumbing:

- replace ImageNet-21k TResNet-M spatial features with final OpenAI CLIP
  ViT-B/16 patch tokens initialized from the same checkpoint as the other
  Track-A methods;
- keep the CLIP visual encoder trainable, matching CSC's trainable TResNet;
- retain class activation masks, general and image-specific graph relations,
  dynamically expanding seen-class modules, current-label BCE, old-model
  sigmoid distillation, and max-entropy regularization;
- do not add an Adapter, prompt, CLIP text feature, replay buffer, or access to
  old/future ground-truth labels;
- use the Core B5-C3 class order, train/validation/test splits, evaluator,
  global threshold `0.5`, checkpoint schema, and result artifacts.

The CLIP patch sequence is the spatial axis used by the source CAM operations.
The source `1x1` convolution that maps TResNet feature channels into graph
features becomes a token-wise linear projection; the CI-GCN operations and
their seen-class axes remain unchanged.

## Frozen optimization interpretation

The paper reports batch size 64, 20 epochs, Adam, OneCycleLR, weight decay
`1e-4`, and learning rate `4e-5` for every PASCAL VOC task. Because B5-C3 is
explicitly among the VOC scenarios and the release configuration also defaults
to 20 epochs, batch size 64, and `4e-5`, the initial EMOTIC Track-A port will
use these values. They are implementation defaults written to every manifest,
not additions to the frozen protocol YAML.

The released entry point differs from the paper in several executable details:

- it constructs a single Adam optimizer before later modules are replaced and
  does not rebuild it after class expansion;
- it imports a scheduler but never constructs or steps it;
- its CLI defaults weight decay to zero even though the paper reports `1e-4`;
- it copies only selected weights during expansion and leaves some replacement
  biases reinitialized;
- it calculates entropy over all seen classes, while the paper defines it for
  old classes and notes that using all seen classes performs similarly;
- it averages the two output branches, while the paper writes their sum.

The port follows the paper-reported optimizer and schedule and rebuilds the
optimizer after each expansion so every currently registered parameter can be
updated. It preserves the released forward/loss behavior where it is explicit:
the two logits are averaged, entropy covers all seen classes, current and
distillation terms use multi-label sigmoid BCE, `alpha=0.5`, and the released
signed max-entropy coefficient corresponds to an entropy strength of `0.04`.
All copied old sub-blocks, including biases, are preserved across expansion;
new rows/columns are initialized by the same PyTorch layer defaults. These
lifecycle corrections implement the paper's stated inheritance and continual
optimization rather than reproducing dangling optimizer references.

## Protocol and leakage checks

For task `t`, the method may consume only:

- images and ground truth for the current task columns during training;
- previous-model sigmoid predictions for old columns;
- current-column validation labels for epoch selection, if selection is used;
- seen-column ground truth only inside the evaluator after training.

It must reject any training batch whose visibility mask exposes old or future
columns. The teacher is a detached deep copy taken before task expansion. Test
data cannot choose epochs, coefficients, or architecture settings.

## Planned implementation and gates

The independent adapter will live under:

```text
benchmarks/emotic_mlcil/methods/csc/
├── __init__.py
├── method.py
└── model.py
```

Before any GPU experiment, the branch must provide tests for:

- registry/source/config metadata and absence of Adapter/text/replay;
- exact dynamic expansion and preservation of old parameter blocks;
- current-only visibility enforcement;
- detached old-model sigmoid targets and non-zero incremental distillation;
- CI-GCN tensor shapes and sample-specific graph construction;
- max-entropy loss sign and task-0 behavior;
- prediction/target/sample-ID alignment;
- checkpoint round-trip and parameter/memory statistics.

After CPU tests and legacy regressions pass, run a worst-case GPU memory smoke,
then a validation-only seed-0 experiment. Hyperparameters may be frozen only
from validation. Held-out test remains prohibited until the configuration is
locked in a committed, clean source tree.

The validation launcher is
`scripts/emotic-mlcil/launch_csc_seed0_tmux.sh`. It requires a clean worktree,
runs the complete Core/CSC and selected legacy regression suites, performs a
worst-task teacher + forward/backward + Adam-state memory smoke, and only then
starts seed 0 on the validation split in tmux. It does not authorize held-out
test evaluation.
