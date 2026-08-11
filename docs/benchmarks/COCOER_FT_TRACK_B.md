# CocoER-FT Track-B Conversion Contract

## Identity

`CocoER-FT` is the no-anti-forgetting conversion of the static CVPR 2025
CocoER model. The immutable reference is
[`bisno/CocoER`](https://github.com/bisno/CocoER) at commit
`dac8fc139e61b87f1bf0b27c581798df2a5a9d38` under MIT. The source stays
outside this repository in `baseline_sources/cocoer_release_dac8fc1/`.

This is **Track B**. It deliberately does not use the benchmark-wide CLIP
ViT-B/16. It retains CocoER's native three independent ImageNet ResNet-50
towers for head, body, and context, plus the frozen OpenAI CLIP RN50 image
encoder used by CocoER's vocabulary-informed branch. It shares the B5-C3
tasks, samples, label firewall, val-only selection, held-out test policy,
metrics, and fixed F1 threshold `0.5` with Track A, but belongs in a separate
native-backbone table.

## Frozen conversion interface

The `CocoER-FT-v0.1` interface retains:

- context/body/head 224×224 views and their body/head coordinates;
- three trainable ImageNet ResNet-50 towers and 2048→256 projections;
- three 3-block, 4-head cross-level attention streams;
- 49×256→256 head/body/context feature classifiers;
- frozen CLIP RN50 image features, trainable 1024→1024 mapping, and the
  released repeated-feature VI MLP;
- threshold-`0.3` VI pseudo labels, feature competition/refinement, and
  four-level final fusion;
- source dynamic class-weight BCE, AdamW, schedule, and loss weights.

Only the static class interface changes: each of head/body/context/VI/global
gets protocol-ordered expanding linear task heads. Task `t` supervises only
`targets_current`. Previous logits remain available for evaluation but receive
no truth, teacher, replay, Fisher penalty, or other anti-forgetting signal.
There is no Adapter or prompt. This makes the result a method-specific
sequential fine-tuning lower bound, not a new continual-learning algorithm.

The released gradient competition is generalized from a fixed 26-way output
layer to the currently seen progressive heads at the feature boundary. It
keeps the source VI-guided three-branch competition and `inside_lr=0.1`
without requiring nonexistent future output rows. This protocol mapping is
recorded explicitly and is not described as byte-for-byte static inference.

## Full-class asset firewall

The released GWT checkpoint and `checkpoints/VI_weights/w.pth` were trained on
the full static EMOTIC label space. Loading either before Task 7 would carry
future-class information across the benchmark firewall. Both are forbidden.

Allowed initialization is limited to the same generic sources selected by the
official code:

```text
three visual towers: torchvision ResNet-50 ImageNet initialization
VI visual encoder:   official OpenAI CLIP RN50 checkpoint
GWT/VI EMOTIC state: forbidden
```

The runner takes explicit files and records their SHA-256:

```text
pretrained/cocoer/resnet50_imagenet1k_v1.pth
pretrained/clip/RN50.pt
```

## Head-box cache

The benchmark EMOTIC annotations contain body boxes but not CocoER's required
head boxes. Silent top-of-body crops are forbidden. `cocoer_multilevel` instead
requires a complete sample-ID keyed cache:

```text
pretrained/cocoer/emotic_head_boxes_v0.1.json
```

Every selected sample must have one finite nondegenerate box or loading stops.
The cache records detector name/model hash and source JSON hash. Freeze an
externally generated mapping with:

```bash
/opt/conda/envs/ddp/bin/python \
  scripts/emotic-mlcil/prepare_cocoer_head_cache.py \
  --detections /path/to/cocoer_head_detections.json \
  --detector-name insightface-fixed-release \
  --detector-model-sha256 <64-hex-sha256> \
  --output pretrained/cocoer/emotic_head_boxes_v0.1.json
```

Train preprocessing ports the source coordinated crop/flip; all three views
use ImageNet mean/std. Validation and test are deterministic resize-only.

## Source settings registered for seed-0 validation

| Setting | Value |
|---|---:|
| epochs | 20 |
| train batch | 64 |
| optimizer | AdamW |
| learning rate | 0.00006 |
| betas | 0.9, 0.96 |
| weight decay | 0.01 |
| LR schedule | StepLR every 3 epochs, ×0.1 |
| cross-level blocks | 3 |
| attention heads / width | 4 / 256 |
| pseudo threshold | 0.3 |
| internal refinement LR | 0.1 |
| global/head/body/context/VI coefficient | 0.2 each |
| competition-distance coefficient | 0.1 |
| replay | 0 |
| F1 threshold | 0.5 |

These are validation-stage registered settings, not a held-out result freeze.
CUDA memory smoke must determine whether source batch 64 fits one 4090 before
the first run. A batch-size change requires an explicit validation decision
and documentation; test data cannot make that decision.

## Validation launch

After syncing the branch and the three assets:

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

RUN_ID="cocoer_ft_seed0_val_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
SESSION=emotic_cocoer_ft_seed0_val \
bash scripts/emotic-mlcil/launch_cocoer_ft_seed0_tmux.sh
```

The launcher runs the complete Core tests and immutable source audit before
starting validation. Its universal download package contains logs, scores,
metrics, manifests, and the source audit but excludes every `.pth` file.
