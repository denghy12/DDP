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
| Train/eval batch | 4 / 1 |
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
operators must be compiled in the `ddp` environment.

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

RUN_ID="dsct_ft_seed0_val_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
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
Download packages follow `DOWNLOAD_STANDARD.md` and exclude all `.pth` files.
