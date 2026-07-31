# Frozen-CLIP Continual Classifier Baselines

## Scope

This development line adds three repository-native Track-A controls:

- Sequential Fine-Tuning;
- Learning without Forgetting (LwF);
- Elastic Weight Consolidation (EWC).

All three start from Benchmark Core v0.1 commit
`00f399f13bc7552c254c8f6e6c095a8be4f56146`. They share exactly the same
model, data boundary, optimizer, validation selection, and checkpoint format.
Only the continual-learning loss changes.

## Shared model and preprocessing

The OpenAI CLIP ViT-B/16 checkpoint is loaded once and completely frozen.
Only its image encoder is used by these classifier controls. EMOTIC full-image
preprocessing remains the frozen DDP/Core path: random resized crop and
horizontal flip for training, resize and center crop for validation/test, then
`ToTensor`. No new method-specific normalization or input path is introduced.

Each task's frozen image features are materialized once in CPU memory. Training
the small continual module for multiple epochs therefore does not repeat the
CLIP forward pass. The cache is discarded at the task boundary and is not a
replay buffer.

The trainable module contains:

1. a shared `512 -> 128 -> 512` residual feature adapter;
2. one protocol-ordered linear sigmoid head block for each introduced task.

The adapter's final projection is zero-initialized. Old task heads are frozen
after their task. New task supervision updates only the shared adapter and the
new head. Consequently Sequential Fine-Tuning can forget old classes through
representation drift without ever reading old-class ground truth.

## Objectives

Every method minimizes binary cross entropy on only `targets_current`.

LwF follows [Li and Hoiem, ECCV
2016](https://arxiv.org/abs/1606.09282): before a new head is added, a frozen
copy of the old continual module becomes the teacher. On current-task images,
the student matches the teacher's independent old-class sigmoid probabilities
at temperature `2.0`. The multi-label distillation term is multiplied by
`T^2`; its configured weight is `1.0`. No old labels or stored old samples are
used. The authors' [reference
repository](https://github.com/lizhitwo/LearningWithoutForgetting) is an
audited algorithm reference, not vendored source.

EWC follows [Kirkpatrick et al., PNAS
2017](https://doi.org/10.1073/pnas.1611835114). At each task boundary, the
implementation estimates a diagonal empirical Fisher from the current
multi-label BCE gradients. Online Fisher values are accumulated with decay
`1.0`, and the next task adds a Fisher-weighted quadratic penalty anchored at
the consolidated parameters. The configured coefficient is `100.0`.

## Frozen optimization settings

The frozen defaults live in `FrozenCLIPOptions`. They are deliberately kept out
of `protocol_b5c3.yaml` so the registered Core v0.1 protocol hash remains
unchanged. Every run writes the fully resolved values into both
`config_resolved.json` and `run_manifest.json`:

| Setting | Value |
|---|---:|
| Feature dimension | 512 |
| Adapter bottleneck | 128 |
| Residual scale | 0.1 |
| Epochs per task | 20 |
| Cached-feature batch size | 256 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| LwF temperature / weight | 2.0 / 1.0 |
| EWC coefficient / online decay | 100.0 / 1.0 |

Validation checkpoint selection uses current-label validation mAP only. Strict
improvement replaces the best state, so ties retain the earliest epoch. Test is
never used for epoch or hyperparameter selection.

## Execution

One method must process Task 0--7 sequentially. Independent task shards are
rejected because they would discard the continual state. Different methods or
seeds can run on different GPUs.

Example validation run:

```bash
python -m benchmarks.emotic_mlcil.runner \
  --protocol configs/emotic_mlcil/protocol_b5c3.yaml \
  --method finetune \
  --data-root /mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC \
  --clip-model-path ./pretrained/clip/ViT-B-16.pt \
  --output-root ./output \
  --reporting-split val \
  --train-batch-size 64 \
  --eval-batch-size 64 \
  --workers 0 \
  --device cuda
```

Replace `finetune` with `lwf` or `ewc`. A held-out formal run additionally
requires `--reporting-split test --configuration-locked`.

After server tests pass, launch the three seed-0 methods concurrently on three
GPUs:

```bash
SEEDS="0" GPU_LIST="0 1 2" \
SESSION=emotic_baselines_seed0 \
RUN_ID=baseline_seed0_<timestamp> \
bash scripts/emotic-mlcil/launch_frozen_clip_baselines_tmux.sh
```

Review seed 0 before launching the remaining registered seeds. Seeds 1 and 2
use six GPUs in a second wave:

```bash
SEEDS="1 2" GPU_LIST="0 1 2 3 4 5" \
SESSION=emotic_baselines_seed12 \
RUN_ID=baseline_seed12_<timestamp> \
bash scripts/emotic-mlcil/launch_frozen_clip_baselines_tmux.sh
```

The launcher refuses duplicate jobs, more jobs than supplied GPUs, dirty formal
source trees, missing data/CLIP weights, and pre-existing tmux session names.
It runs the full Core/baseline and selected legacy regression suites before
creating GPU windows.

Canonical checkpoints remain under `checkpoints/`. After a completed run,
`--export-sync-results --shard-run-id <safe_run_id>` creates a separate
`results_to_sync/<safe_run_id>/` download bundle containing metrics, scores,
logs, manifests, and reports but no `.pth` files.
