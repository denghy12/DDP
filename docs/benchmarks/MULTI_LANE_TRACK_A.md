# MULTI-LANE Track-A Source Audit and Porting Contract

## Scope and immutable source

This document fixes the source and protocol-safe Track-A interpretation of
MULTI-LANE before implementation or validation results are produced.

Development branch: `codex/emotic-baseline-multi-lane`, created from the
registered CSC Track-A commit
`e13cac75ea7d49c5270e5d2e37a884d86e0ed40c`.

- Paper: *Less is more: Summarizing Patch Tokens for efficient Multi-Label
  Class-Incremental Learning*, CoLLAs 2024.
- Official proceedings: <https://proceedings.mlr.press/v274/min25a.html>
- Official repository: <https://github.com/tdemin16/multi-lane>
- Fixed upstream commit:
  `5ee982c9298d4cfd6af471d9bb2ef3c0aad05373` (`main`).
- Fixed source archive SHA-256:
  `dfe84ea31f6d7e51877c2661791c716aed7d888b8d783d16ce067cb8ce022d49`.
- Read-only extraction:
  `/Users/denghaoyuan/workspace/MyCode/baseline_sources/multi_lane_release_5ee982c/`.
- Earlier local snapshot:
  `/Users/denghaoyuan/workspace/MyCode/baseline_sources/multi-lane-main/`.

The two local trees contain the same 26 files with no byte differences. The
fixed archive and digest provide immutable provenance because the earlier
snapshot has no `.git` metadata.

The upstream `LICENCE` applies Creative Commons Attribution-NonCommercial
4.0. Any reused or adapted material must preserve attribution, identify
modifications, and remain non-commercial. The benchmark will use an
independently written thin implementation and will not vendor the upstream
repository.

## Audited released method

The official MLCIL path uses an ImageNet-pretrained ViT-B/16 whose ordinary
image-token pathway is frozen. MULTI-LANE adds a parallel task pathway inside
every transformer block:

1. task-specific Patch Selectors attend to detached image tokens and summarize
   them into a small task-token set;
2. task-specific key/value prompts are inserted in the first five transformer
   layers;
3. drop-and-replace retains the task CLS update while restoring selector
   tokens for the next layer;
4. a shared linear classifier maps every task-lane CLS token to the complete
   class width;
5. training masks logits and targets to the current task classes; and
6. concat inference evaluates every seen task lane, keeps only that lane's
   class block, and sums the disjoint blocks into one seen-class vector.

The released training loop has no replay, old-model teacher, old-label loss,
or future-label loss. At each boundary the next selector and prompt slices are
copied from the preceding task. A new Adam optimizer and cosine schedule are
created per task, clearing optimizer state. Old/future slices remain in the
preallocated tensors but are absent from the current computation and receive
no gradients. The released concat mode does not use a task oracle at test
time; all seen lanes are evaluated.

## Track-A mapping

Track A replaces only the released frozen ImageNet ViT-B/16 with the same
OpenAI CLIP ViT-B/16 visual checkpoint used by the benchmark:

- retain the frozen convolutional patch embedding, positional/class tokens,
  layer norms, attention weights, MLP weights, and final projection;
- reuse each frozen CLIP residual block for both the image pathway and the
  parallel task pathway;
- retain task-specific selectors, first-five-layer key/value prompts,
  orthogonal initialization, drop-and-replace, previous-task initialization,
  pre-head normalization, shared classifier, and concat inference;
- expose only current-task columns to BCE training and use the unified Core
  evaluator for all seen columns;
- add no residual Adapter, CLIP text feature, replay buffer, distillation loss,
  task identifier, or trainable backbone projection.

OpenAI CLIP's transformer width is 768 and its projected visual width is 512.
Selectors and prompts operate at the internal 768-wide block representation;
the shared classifier consumes the frozen 512-wide projected lane CLS token.
This uses the complete shared visual initialization rather than introducing an
uninitialized feature projection.

The protocol's fixed eight-task and 26-class shape may be used to allocate
selector/prompt slices and classifier rows, matching the released code's
preallocation. This is architectural capacity only: current `TrainBatch`
values must expose exactly the current B5-C3 columns, and old/future ground
truth remains inaccessible.

## Initial optimization mapping

The initial EMOTIC configuration follows the released PASCAL VOC MLCIL setup,
which is closer to EMOTIC's class count than the COCO setup:

- 30 epochs per task;
- Adam, zero weight decay, cosine annealing, optimizer reset each task;
- source base learning rate `0.05` at reference batch size 256, scaled linearly
  to the actual training batch size as in the released entry point;
- 10 selectors, 10 K/V prompts, prompts in the first five layers;
- orthogonal selector/prompt initialization;
- drop-and-replace enabled, token merging disabled;
- pre-head L2 normalization, concat inference, temperature 1.0;
- no replay, no text encoder, and frozen CLIP visual weights.

AMP/TF32 may be used as execution accelerators if same-input smoke tests show
finite forward/backward behavior; they do not change the registered method.
Validation may confirm training stability, but held-out test cannot select or
change these values.

## Required implementation gates

Before validation training, the branch must provide:

1. registry and runner integration under
   `benchmarks/emotic_mlcil/methods/multi_lane/`;
2. unit tests for task-slice copying, frozen backbone weights, current-label
   visibility, concat lane masking, checkpoint round-trip, and zero replay;
3. an executable comparison against the fixed external source for selector
   aggregation, prompt attention, drop-and-replace, and concat masking;
4. full Core and selected legacy regression success;
5. a worst-task forward/backward/Adam-state GPU memory smoke; and
6. one seed-0 validation-only run before any configuration-locked held-out
   execution.

Formal results must follow the universal checkpoint-free download standard and
use mean ± sample standard deviation across seeds 0, 1, and 2.

## Implementation candidate

The independent candidate is registered as runner method `multi_lane` under
`benchmarks/emotic_mlcil/methods/multi_lane/`. Runtime `0.5.0` now provides:

- a frozen OpenAI CLIP visual stream and a differentiable parallel task stream;
- selectors and K/V prompt tensors preallocated for the fixed eight tasks;
- exact first-LayerNorm selector aggregation, first-five-layer prompts, and
  released drop-and-replace residual behavior;
- the released full-26-column BCE reduction, with non-current logits and
  targets filled with zero so only current labels have gradients;
- previous-task slice copying, task-local gradients, Adam/cosine reset per
  task, and zero architectural parameter growth after initialization; and
- evaluation over all seen lanes followed by disjoint class-block concat,
  without a task identifier or task oracle.

`compare_multi_lane_upstream_reference.py` verifies the external files it will
execute and compiles the exact upstream `PreT_Attention` and `forward_head`
bodies at runtime. When a clean archive is supplied it additionally verifies
the archive and whole extraction. It compares same-input/same-weight selector
aggregation, prompt attention, drop-and-replace, and concat masking, rejecting
maximum absolute error of `1e-6` or larger. The upstream source remains outside
this repository.

The server's existing `/mnt/haoyuan/workspace/multi-lane-main` tree contains
local data and unrelated modifications, so its whole-tree hash is intentionally
not treated as an upstream identity. The two files actually compiled by the
oracle, `multi_lane/blocks.py` and `multi_lane/vision_transformer.py`, are
byte-identical to the fixed extraction, with SHA-256 values `79f1d549...24af7`
and `6d4777e7...a3e95`. In this critical-file mode the oracle rejects either
file before compiling any code if its digest differs. The locally retained
clean archive and full-tree digests remain the immutable provenance record;
none of the modified tree's other files are imported or executed.

Six focused unit cases cover source identity, frozen visual weights, task-slice
copying and isolation, current-label visibility, concat masking, checkpoint
round-trip, aligned prediction, zero replay, and preallocated parameter
statistics. Full Torch execution is intentionally left to the configured
server because the local macOS system Python has no PyTorch installation.

## Seed-0 validation entry point

After the candidate commit is present on the server, launch the validation-only
gate with the already-present `/mnt/haoyuan/workspace/multi-lane-main` tree:

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

GPU=0 \
SESSION=emotic_multi_lane_seed0_val \
bash scripts/emotic-mlcil/launch_multi_lane_seed0_tmux.sh
```

The launcher refuses a dirty tree, runs the complete Core and selected legacy
tests, executes the fixed-source oracle, measures the worst-task train/eval
GPU peak including Adam state, and only then starts seed-0 validation in tmux.
On success it creates exactly one checkpoint-free download archive and adjacent
checksum:

```text
<run-root>/download_packages/<run-id>.tar.gz
<run-root>/download_packages/<run-id>.tar.gz.sha256
```

## Frozen validation decision and formal execution

The seed-0 validation gate completed from clean commit `35ca2de` with Final
mAP `41.9641`, Average mAP `48.3236`, Forgetting `0.8908`, Final cF1
`37.2893`, and Final oF1 `58.5866`. All 240 epochs and 13,950 optimizer
updates completed with finite values, zero AMP overflow skips, no OOM, and no
traceback. The fixed-source critical-operator comparison remained exactly
equal (`max_abs_error=0.0`). No hyperparameter is changed after this review.

The physical parameter-growth field is zero because all eight task slices are
allocated before task 0, matching the released implementation. This must not
be described as zero task-specific capacity. The frozen ViT-B/16 instance has
`675,840` task-lane parameters (`84,480` per task) plus a `13,338`-parameter
shared classifier, for `689,178` optimizer parameters in total. Future lanes
are preallocated architecture only: they are absent from current computation,
receive no gradient, and cannot access future labels.

The reviewed evidence is registered in
[`results/multi_lane_seed0_validation_v0.1.json`](results/multi_lane_seed0_validation_v0.1.json).
The configuration is now locked for held-out test seeds 0, 1, and 2. They run
concurrently on three distinct GPUs because each continual seed must preserve
its own task sequence; spreading a single seed over the remaining GPUs would
change the implementation and add synchronization overhead.

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

EXPECTED_GIT_COMMIT="$(git rev-parse HEAD)" \
CONFIGURATION_LOCKED_CONFIRMATION=MULTI_LANE_TRACK_A_V0_1 \
CLIP_MODEL_PATH=/mnt/haoyuan/workspace/CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt \
GPUS="0 1 2" \
bash scripts/emotic-mlcil/launch_multi_lane_formal_seed012_tmux.sh
```

The formal launcher rejects a dirty or unexpected Git commit, reruns Core,
legacy, fixed-source, and memory preflights, launches one seed per GPU, checks
all three manifests/configurations/training logs, aggregates with sample
standard deviation, and creates one checkpoint-free archive:

```text
<run-root>/download_packages/<run-id>.tar.gz
<run-root>/download_packages/<run-id>.tar.gz.sha256
```

Any formal test metric is report-only and must not change the frozen
configuration.
