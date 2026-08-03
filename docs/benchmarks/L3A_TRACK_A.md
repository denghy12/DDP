# L3A Track-A Source Audit and Porting Contract

## Scope and immutable source

This document fixes the source and protocol-safe Track-A interpretation of
L3A before validation or held-out results are produced.

Development branch: `codex/emotic-baseline-l3a`, created from the registered
MULTI-LANE Track-A commit
`20e645dae3e23e9f7860760f5800bdbf49d8f8ca`.

- Paper: *L3A: Label-Augmented Analytic Adaptation for Multi-Label Class
  Incremental Learning*, ICML 2025, PMLR 267:74938--74949.
- Official proceedings: <https://proceedings.mlr.press/v267/zhang25y.html>.
- Official repository: <https://github.com/scut-zx/L3A>.
- Fixed upstream commit:
  `1067bbd6124a7aa96136baa555080ba9183ab2ac` (`main`).
- Fixed upstream tree:
  `9e6ac470600328f2a393f26e30ea764c3b50102b`.
- Fixed `git archive` SHA-256:
  `307d721d76069a7fe95a10346be1323a842cd9624f8d55200f4cf7cd45462409`.
- Clean read-only clone:
  `/Users/denghaoyuan/workspace/MyCode/baseline_sources/l3a_official/`.
- Portable archive:
  `/Users/denghaoyuan/workspace/MyCode/baseline_sources/l3a_release_1067bbd.tar.gz`.
- Earlier snapshot:
  `/Users/denghaoyuan/workspace/MyCode/baseline_sources/L3A-main/`.

The earlier 27-file snapshot is byte-identical to the fixed official Git tree.
No `LICENSE`, `LICENCE`, `COPYING`, or `NOTICE` file and no GitHub license
declaration was found at the fixed commit. The upstream source therefore stays
outside this repository as a read-only reference. The benchmark implementation
is independently written against the paper and audited behavior; no upstream
file is vendored.

The primary implementation file is
`MultiLabelIncremental_L3A.py` (SHA-256
`3f6ccdad5fa4c45627f414d7f942d5c544c847cb5e03f100a618230a335eb689`).
The fixed commit added the official timm ViT-B/16 path and
`configs/l3a_vit_coco.yaml`, which supplies the closest released architecture
mapping for Track A.

## Audited released method

L3A is exemplar-free and has two distinct training phases.

1. On the base task, the pretrained visual backbone and a temporary linear
   head are trained for one epoch with ASL (`gamma_neg=4`, `gamma_pos=0`,
   negative clip `0.05`) using Adam and OneCycleLR.
2. The temporary head is discarded. A newly initialized bias-free random
   projection, ReLU, and bias-free analytic classifier replace it.
3. Base features and labels form cumulative matrices `A = ZᵀΩZ` and
   `C = ZᵀΩY`. Class weights are inverse-square-root frequencies normalized to
   have mean one; each sample weight is the mean weight of its positive labels.
4. The classifier is the closed-form solution
   `W = inverse(A + ridge I) C`.
5. At each incremental task, the visual backbone and random projection stay
   frozen. Old-class sigmoid scores above `0.7` become positive pseudo labels,
   true current labels are concatenated, `A/C` are accumulated, and the
   analytic solution is recomputed.

The active path contains no replay buffer. `sample_proto.py` contains herding
helpers but is not imported or called by `MultiLabelIncremental_L3A.py`.
Likewise, the repository's generic distillation helper is not part of L3A's
active training path.

The released incremental routine assigns the old classifier and inverse to
local variables but does not use them in its update. The persistent state that
actually determines later tasks is the cumulative `A`, cumulative expanding
`C`, per-class frequency counts, frozen feature mapping, and current analytic
weights. The independent port retains that effective behavior without the two
redundant assignments.

## Track-A mapping

Track A replaces only the official pretrained timm ViT-B/16 visual encoder
with the benchmark's OpenAI CLIP ViT-B/16 visual encoder:

- Task 0 fine-tunes the CLIP visual tower and temporary linear head exactly
  once, then freezes and retains the resulting visual weights;
- raw projected CLIP image features replace timm's unnormalized output; no
  additional feature normalization is introduced;
- the random bias-free projection, ReLU, weighted analytic matrices,
  closed-form solve, class-frequency weights, and positive pseudo-label rule
  are retained;
- no Adapter, CLIP text feature, replay buffer, distillation objective, or task
  oracle is added; and
- method code sees only `targets_current`. Old labels are produced exclusively
  from the method's own prior scores, while old/future ground truth remains
  inaccessible.

The official ViT configuration is used as the immutable initial mapping:

- hidden analytic width `4096`;
- ridge coefficient `1`;
- Task-0 Adam learning rate `4e-5`, true weight decay `1e-4`, one epoch;
- OneCycleLR `pct_start=0.2`;
- one analytic pass per task;
- weighted analytic classifier enabled; and
- positive pseudo-label threshold `0.7`.

The official TResNet VOC/COCO ridge value `1000` is not mixed into this Track-A
ViT mapping. Validation may check numerical stability but may not tune these
values after held-out test access.

## Benchmark-only protocol differences

The upstream evaluator reports F1 at `0.525`. The benchmark evaluator instead
uses its already-frozen global threshold `0.5` for every method, task, seed,
and class. This metric-only mapping does not affect AP/mAP or training.

The released code evaluates base, seen, and new validation views after a fixed
training schedule. The Core method-facing validation loader exposes only the
current classes. L3A does not select an epoch or coefficient from that loader;
its current-class validation mAP is monitoring evidence only.

The analytic matrices are float64, as in the released implementation. With
hidden width 4096, `A` and final `C` are method state rather than replay memory.
Their element and byte counts are recorded in the run manifest. Checkpoints
retain them because they are required to continue the closed-form sequence;
the universal downloadable results package still excludes every `.pth`.

## Required gates

Before any held-out test run, this branch must pass:

1. registry/runner and sequential task-lifecycle tests;
2. current-label visibility and no-replay checks;
3. same-state verification of the weighted closed-form solution;
4. pseudo-label, checkpoint continuation, prediction-alignment, and parameter
   growth tests;
5. full Core and selected legacy regressions on the configured server;
6. an executable fixed-source operator audit and worst-task GPU memory smoke;
7. one seed-0 validation-only run; and
8. a committed configuration freeze before `--configuration-locked` test.

Formal artifacts follow [the universal checkpoint-free download
standard](DOWNLOAD_STANDARD.md). Only after seed-0 held-out review may seeds 1
and 2 be launched.
