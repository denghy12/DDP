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

Prepare both without importing any EMOTIC-trained CocoER state:

```bash
/opt/conda/envs/ddp/bin/python \
  scripts/emotic-mlcil/prepare_cocoer_native_assets.py
```

If the server network is slow, download the two official files locally, copy
them to the server, and use the offline path without changing their names or
contents:

```bash
/opt/conda/envs/ddp/bin/python \
  scripts/emotic-mlcil/prepare_cocoer_native_assets.py \
  --resnet50-source /path/to/resnet50-0676ba61.pth \
  --clip-rn50-source /path/to/RN50.pt
```

The script asks torchvision for exactly
`ResNet50_Weights.IMAGENET1K_V1`, requires its official
`resnet50-0676ba61.pth` URL and SHA prefix, and requires OpenAI RN50 SHA-256
`afeb0e10...b6762`. It writes `pretrained/cocoer/native_assets_manifest.json`.

## Head-box cache

The benchmark EMOTIC annotations contain body boxes but not CocoER's required
head boxes. Silent top-of-body crops are forbidden. `cocoer_multilevel` instead
requires a complete sample-ID keyed cache:

```text
pretrained/cocoer/emotic_head_boxes_v0.1.json
```

The official repository and CVPR supplementary material do not publish the
training head-box preprocessing cache or a dataset-preparation entry point.
The released inference path does specify InsightFace `0.7.3`, `buffalo_l`,
640×640 detection and its face/person x-containment rule. The benchmark freezes
that executable rule and the following approved sample-preserving conversion:

1. run the `buffalo_l` SCRFD detector and apply the released strict person
   matching rule;
2. estimate one componentwise-median relative head box from **native-resolved
   train samples only**;
3. project that fixed train-only geometry onto every native-unresolved body box.

The fallback uses neither labels nor validation/test statistics. It preserves
all benchmark person samples instead of changing the data split according to
detector success. Every native/fallback sample ID, split count, median geometry,
runtime provider, detector tree hash, and source-artifact hash is recorded.

The fixed detector pack is InsightFace v0.7 `buffalo_l` from its official
GitHub release. Its archive is exactly `288621354` bytes with SHA-256
`80ffe37d...b0ca2f`; the extracted five-file model tree is
`50fa1383...9d9d3d`. InsightFace model weights are restricted to
non-commercial research use. Prepare the uploaded archive outside the Git
repository with:

```bash
/opt/conda/envs/ddp/bin/python \
  scripts/emotic-mlcil/prepare_cocoer_insightface_assets.py \
  --archive /mnt/haoyuan/workspace/baseline_sources/cocoer_assets/buffalo_l.zip \
  --output-root /mnt/haoyuan/workspace/baseline_sources/cocoer_insightface
```

Generation uses the isolated `cocoer-preprocess` environment. The launcher must
expose its CUDA 11 runtime libraries; declaring CUDA is insufficient unless the
artifact records `CUDAExecutionProvider` as the first actual provider. The
generator also checks a small fixed set of images through the full official
`FaceAnalysis` path and requires exact integer-box equality with the faster
SCRFD-only path before processing the dataset:

```bash
GPU=0 bash scripts/emotic-mlcil/run_cocoer_head_preprocess.sh
```

The wrapper constructs `LD_LIBRARY_PATH` from the isolated environment and
then delegates to `generate_cocoer_head_detections.py`. Its `DATA_ROOT`,
`INSIGHTFACE_ROOT`, `OUTPUT`, and `COCOER_ENV` defaults may be overridden with
environment variables without changing the registered algorithm.

The generator hashes the complete `models/buffalo_l` tree, reuses detections
per image, aligns them to stable person IDs, computes the train-only calibration
after native detection, and then fills only native-unresolved samples. It exits
`3` if any sample remains unresolved after the registered conversion. Only a
complete sample-preserving artifact can be frozen:

```bash
/opt/conda/envs/ddp/bin/python \
  scripts/emotic-mlcil/prepare_cocoer_head_cache.py \
  --detections /mnt/haoyuan/workspace/baseline_sources/cocoer_head_detections.json \
  --output pretrained/cocoer/emotic_head_boxes_v0.1.json
```

Finally audit all three assets against every train/val/test sample:

```bash
/opt/conda/envs/ddp/bin/python scripts/emotic-mlcil/audit_cocoer_assets.py \
  --data-root /mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC \
  --resnet50-init pretrained/cocoer/resnet50_imagenet1k_v1.pth \
  --clip-rn50 pretrained/clip/RN50.pt \
  --head-cache pretrained/cocoer/emotic_head_boxes_v0.1.json \
  --output pretrained/cocoer/cocoer_assets_audit.json
```

The joint audit independently reconstructs every fallback box from the frozen
train median, checks the original CocoER matching rule for every native box,
and verifies exact coverage of the 16,001 train, 2,397 validation, and 5,368
test person samples. The full CUDA-generated cache statistics remain pending;
the earlier 256-sample diagnostic (224 native, 32 fallback candidates) is not a
formal dataset statistic and must not be copied into a result table.

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
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-cocoer-ft

RUN_ID="cocoer_ft_seed0_val_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
SESSION=emotic_cocoer_ft_seed0_val \
bash scripts/emotic-mlcil/launch_cocoer_ft_seed0_tmux.sh
```

The launcher runs the complete Core tests, immutable source audit, and full
asset/sample-coverage audit before starting validation. Its universal download
package contains logs, scores, metrics, manifests, source audit, and asset
audit but excludes every `.pth` file.
