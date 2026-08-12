# DSCT-FT Track-B Contract

## Identity and scope

`DSCT-FT` converts the static EMOTIC model **Dual-Stream Contextual
Transformer (DSCT), ACM MM 2024** into the benchmark's no-anti-forgetting
control. It is Track B because it retains the official ResNet-50,
Deformable-DETR/DSCT transformer, preprocessing and optimization rather than
substituting the Track-A CLIP visual encoder.

- official repository: `https://github.com/Sampson-Lee/DSCT`
- immutable commit: `8b0fe36199693ebcd58a4cce8751c166550fc7a5`
- paper R101 EMOTIC result: `37.81` mAP
- repository's reproducible R50 setting: `36.21` mAP
- license caveat: source headers and README state Apache-2.0, but no `LICENSE`
  file is present at the fixed commit. Source remains external and read-only.

The static numbers are context only and are not continual B5-C3 results.

## Static-to-incremental mapping

The official model is retained: ResNet-50, four-scale Deformable DETR, 300
context queries, four subject queries, six encoder/decoder layers, hidden
width 256, subject box regression, sigmoid focal classification and auxiliary
decoder losses. Only the static emotion classifier changes. One current-class
linear head is added at each task and outputs are concatenated in protocol
order; subjectness remains shared.

At task `t`, matching and focal loss use only task-`t` emotion labels plus
subjectness. Old/future logits receive no target or loss. All model parameters
and seen heads remain trainable, so forgetting is ordinary sequential fine
tuning. There is no Adapter, replay, teacher, distillation, EWC, CLIP visual
tower, or CLIP text feature.

Each benchmark row supplies the full scene and its target-person body box.
Geometry is not an emotion label. Evaluation follows official
`face_matching`: select the subject query with maximum IoU to that box and
return its seen-class sigmoid scores. Other persons' emotion annotations are
never exposed.

## Registered validation configuration

| Item | Value |
|---|---:|
| Backbone | official DSCT ResNet-50 |
| Subject/context queries | 4 / 300 |
| Encoder/decoder layers | 6 / 6 |
| Hidden width | 256 |
| Epochs / patience | 50 / 50 |
| Effective train/eval batch | 4 / 16 |
| Per-GPU micro-batch | 2 |
| Multi-GPU execution | two replicas; two samples/replica; one synchronized optimizer step |
| Single-GPU fallback | two sequential micro-batches of 2; one optimizer step |
| Optimizer | AdamW |
| Main/backbone LR | `2e-4` / `2e-5` |
| Projection LR | `2e-5` |
| Weight decay | `1e-4` |
| LR drop | epoch 40, factor 0.1 |
| Gradient clip | 0.1 |
| Matcher costs class/box/GIoU | 2 / 5 / 2 |
| Loss weights class/box/GIoU | 5 / 5 / 2 |
| Focal alpha/gamma | 0.25 / 2 |
| Selection | current-label val mAP, earliest tie |
| Main-table F1 | fixed 0.5 |
| Numerical execution | AMP + channels-last visual/transformer path; FP32 deformable attention and loss/matching |
| Loader / CPU contract | 2 persistent workers, prefetch 2, pinned transfer, 4 BLAS/OpenMP threads |

Training retains official horizontal flip and multi-scale short sides 480--800
with maximum side 1333. Source random crop can remove the benchmark's single
target person, so it is disabled and recorded as a mapping difference.

Initialization must be the official linked
`r50_deformable_detr-checkpoint.pth`. A final 26-class DSCT EMOTIC checkpoint
is forbidden because it has learned future classes. Actual checkpoint SHA-256
is recorded and random/CLIP fallback is rejected.

## Execution entry points

The source remains at
`/mnt/haoyuan/workspace/baseline_sources/dsct_release_8b0fe36`; its custom CUDA
operators must be compiled in the `ddp` environment. The fixed source's
declared `ipython` and `scikit-learn` imports are pinned to `8.18.1` and
`1.3.2` for the benchmark's Python 3.9 runtime. Their observed versions,
along with SciPy, are recorded by formal preflight.

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-dsct-ft

RUN_ID="dsct_ft_seed0_val_$(date +%Y%m%d_%H%M%S)" \
GPU=5,6 \
SESSION=emotic_dsct_ft_seed0_val \
DSCT_SOURCE_ROOT=/mnt/haoyuan/workspace/baseline_sources/dsct_release_8b0fe36 \
DSCT_PRETRAINED_WEIGHTS=/mnt/haoyuan/workspace/baseline_sources/dsct_release_8b0fe36/r50_deformable_detr-checkpoint.pth \
bash scripts/emotic-mlcil/launch_dsct_ft_seed0_tmux.sh
```

The registered configuration above is frozen a priori from the official R50
recipe before any held-out-test access. A formal seed-0 run must use a clean,
exact Git commit, `configuration_locked=true`, GPU memory preflight, the fixed
source and initialization hashes, and the dedicated
`launch_dsct_ft_formal_seed0_tmux.sh` entry point. The source operator test
passes forward equivalence and numerical gradients through width 1025; the
official widths 2048/3096 are synthetic stress cases that exceed a 24 GiB
RTX 4090, while the registered DSCT width 256 passes numerical gradients.
The first single-GPU batch-4 attempt failed inside task 0 when a real
high-resolution batch reached 22.34 GiB allocated with only 74 MiB free; it
produced no checkpoint, metric, manifest, or held-out-test result. Version
`v0.2` therefore preserves the official effective batch and learning rates but
uses per-GPU micro-batch 1. Four visible GPUs process the four samples in
parallel and synchronize once per effective batch; a one-GPU fallback
accumulates the same four weighted micro-losses before its single optimizer
step. Formal smoke uses the true maximum `800x1333` geometry, and every epoch
flushes a `DSCT_PROGRESS` JSON record with loss, validation mAP and ETA.
The subsequent v0.2 four-GPU FP32 attempt was deliberately stopped after its
measured epoch time projected roughly 55--65 hours for the complete eight-task
pipeline; it produced no registered result. Execution contract `v0.3-fast`
does not change the model, optimizer, effective batch, learning rate, epoch
count, losses, label visibility, or checkpoint selection. It uses AMP around
the expensive visual/transformer path while retaining the legacy deformable
attention extension and all matching/loss calculations in FP32. Validation
and test now use effective batch 4 across the same four replicas, transfers
are pinned/non-blocking, and two persistent loader workers are bounded to four
OpenMP/BLAS threads each. Formal placement is physical GPUs `4,5,6,7`, all on
NUMA node 1; CPU affinity is limited to `36-47,108-119` to prevent thread
oversubscription. Before launching a long run, the true `800x1333` smoke
measures optimizer-step time and memory. A full-run estimate above nine hours
requires seed-0 validation first; an estimate at or below nine hours permits
the user-authorized single-seed locked test directly.

The topology gate measured the same worst-case batch and optimizer update:
single-GPU direct batch 4 was `0.3137 s`, two GPUs with 2 samples each was
`0.2939 s`, and four GPUs with 1 sample each was `0.3633--0.4024 s`. Therefore
`v0.4-fast` freezes the empirically fastest two-GPU topology on physical GPUs
`5,6`; allocating four GPUs is slower for this small effective batch because
DataParallel replication/synchronization dominates. Channels-last is an
execution-only memory layout applied consistently to model and images.
The pure worst-case gate initially favored evaluation batch 32: batch 24/32/40
reached `73.13/78.39/79.08 samples/s`, while batch 40 reserved `22.7 GiB` and
offered less than one percent gain over 32. A short real-loader run then showed
that variable image shapes made the allocator cache grow to about `24.0 GiB`
at batch 32. It was stopped before any epoch, checkpoint or metric. The formal
setting is therefore the measured-safe batch 16, which reached `67.91
samples/s` and reserved at most about `7.7 GiB` in the same post-training
process.
Download packages follow `DOWNLOAD_STANDARD.md` and exclude all `.pth` files.
