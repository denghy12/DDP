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
standard](DOWNLOAD_STANDARD.md). Because configuration is frozen before any
held-out access, seeds 0--2 may run concurrently; no held-out seed may change
the configuration or determine whether another seed is reported.

The seed-0 launcher deliberately reads method-specific controls named
`L3A_OUTPUT_BASE`, `L3A_TRAIN_BATCH_SIZE`, `L3A_EVAL_BATCH_SIZE`,
`L3A_WORKERS`, `L3A_RUN_GPU_SMOKE`, `L3A_UPSTREAM_ROOT`, and
`L3A_UPSTREAM_ARCHIVE`. Generic variables left exported by an earlier KRT,
CSC, or MULTI-LANE shell are ignored, preventing cross-method output paths or
silently skipped preflight stages.

## Frozen validation configuration

The clean seed-0 validation run `l3a_seed0_val_clean_20260803_112320` passed all
gates without changing any registered hyperparameter. It produced Final mAP
`37.0597`, Average mAP `42.6542`, Final cF1 `33.4700`, Final oF1 `44.6241`, and
Forgetting `1.7803`. The complete immutable evidence is stored in
[`results/l3a_seed0_validation_v0.1.json`](results/l3a_seed0_validation_v0.1.json).

Task-0 attempted 84 AMP updates: 76 optimizer updates succeeded and 8 overflow
attempts were skipped. Every logged value and both analytic solutions were
finite, and OneCycleLR advanced only on an applied optimizer update. This
guarded behavior is part of the frozen execution record; it is not a reason to
change AMP scaling after observing validation. The fixed evaluator threshold
remains `0.5` even though the final validation predictions favor recall over
precision.

The CUDA gate measured `5926.9473 MiB` peak allocation for Task-0 training and
`1110.9395 MiB` for the full-width float64 analytic solve at batch 64. Formal
execution therefore assigns one seed to one physical GPU. The orchestrator
runs locked seeds 0, 1, and 2 concurrently on three distinct GPUs. All three
results must pass provenance, configuration, artifact, and numerical-stability
validation before they are aggregated and packaged by the universal
checkpoint-free download standard. A failure in any seed blocks the package.

The frozen held-out confirmation is `L3A_TRACK_A_V0_1`. Held-out runs must use
the exact freeze commit supplied to the formal launcher, a clean worktree,
train/eval batch size 64, two workers, and `configuration_locked=true`.
The registered entry point is
`scripts/emotic-mlcil/launch_l3a_formal_seed012_tmux.sh`; its single tmux job
owns all three concurrent workers, final validation, and checkpoint-free
packaging.

## Registered three-seed formal result

The locked held-out run `l3a_parallel_seed012_20260803_121031` completed from
clean commit `d88620fd5c3650e9d1f361fbcb3174c29b21e006`. Seeds 0, 1, and 2 ran
concurrently on physical GPUs 0, 1, and 2. Every manifest reports the test
split, `configuration_locked=true`, no reused predictions, no test-label
selection, and main-table eligibility. No hyperparameter changed after the
validation freeze.

The registered mean ± sample standard deviation is:

| Metric | L3A Track A |
|---|---:|
| Final mAP | `28.8296 ± 0.6470` |
| Average mAP | `33.7746 ± 1.3091` |
| Forgetting | `4.8687 ± 0.3780` |
| Final cF1 | `26.5487 ± 0.0916` |
| Final oF1 | `34.8493 ± 0.0847` |

The mean per-task mAP curve is `44.3229`, `36.9742`, `29.5139`, `33.5338`,
`33.2423`, `32.3917`, `31.3889`, and `28.8296`; its sample standard
deviations are `2.5066`, `1.9713`, `1.2164`, `1.3296`, `1.1849`, `0.9234`,
`0.8289`, and `0.6470`. The lower Task-2 result followed by the Task-3 rebound
is present in every seed and is retained as observed behavior.

Each seed attempted 84 Task-0 AMP updates, applied 76, and skipped 8 overflows
under the already-frozen guarded scheduler policy. All logged values and
analytic solutions were finite; no seed log contains an OOM, NaN, or training
traceback. Old-class pseudo positives were sparse and task dependent: their
three-seed means for Tasks 0--7 were `0.0`, `1891.7`, `98.7`, `1930.7`,
`199.3`, `91.0`, `5.3`, and `22.3`. These counts come only from prior model
scores and do not expose old or future ground truth.

The complete per-class audit identifies Anticipation (`34.1185 ± 0.5784` AP),
Affection (`20.3143 ± 1.5249`), Annoyance (`9.7606 ± 0.8997`), Disconnection
(`8.9922 ± 0.0803`), and Aversion (`6.6034 ± 3.7346`) as the five largest
mean class-level forgetting values. This diagnosis was made after the locked
test run and is reporting evidence only; it does not authorize tuning.

The checkpoint-free archive contains all three bundles and 24 canonical task
score files. All 57 manifest records and their byte counts/SHA-256 values were
verified; no `.pth`, checkpoint directory, or symlink is present. The archive
SHA-256 is
`970d64ad492dacba1fed2d344872f4d33103f0bdc46a1ebc69db615243af704b`.
The immutable machine-readable record is
[`results/l3a_seed012_formal_v0.1.json`](results/l3a_seed012_formal_v0.1.json).
No training rerun or prediction recomputation is required; this result freezes
L3A Track A v0.1.
