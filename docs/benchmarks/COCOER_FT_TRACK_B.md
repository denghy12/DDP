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

The model-side CocoER interface retains:

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
test person samples. The completed CUDA artifact contains 20,611 native boxes
and 3,155 fallbacks (13.2753%), with zero unresolved samples. The fallback
geometry was calibrated from 13,770 native-resolved train samples. Its complete
statistics and hashes are frozen in
`results/cocoer_head_preprocess_v0.1.json`.

Future preprocessing downloads must use the repository packager rather than a
hand-written `find | sha256sum` manifest. It records only relative paths,
excludes the manifest itself from its file records, and rejects both `.pth` and
`.onnx` payloads:

```bash
/opt/conda/envs/ddp/bin/python \
  scripts/emotic-mlcil/package_cocoer_head_preprocess.py \
  --output-base /mnt/haoyuan/workspace/emotic_benchmark_runs/cocoer_ft_track_b_v0.1 \
  --package-name cocoer_head_preprocess_v0.1 \
  --detections /mnt/haoyuan/workspace/baseline_sources/cocoer_head_detections.json \
  --head-cache pretrained/cocoer/emotic_head_boxes_v0.1.json \
  --asset-audit pretrained/cocoer/cocoer_assets_audit.json \
  --native-manifest pretrained/cocoer/native_assets_manifest.json
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
The launcher's mandatory CUDA smoke constructs the full final-task graph with
all eight expanding heads, executes the head/body/context/VI/global paths,
performs backward and one AdamW step, verifies optimizer-state creation and the
frozen CLIP RN50, and records peak allocated/reserved MiB. It must determine
whether source batch 64 fits one 4090 before the first run. A batch-size change
requires an explicit smoke/validation decision and documentation; test data
cannot make that decision.

## Validation launch

After syncing the branch and the three assets:

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-cocoer-ft

RUN_ID="cocoer_ft_seed0_val_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
SESSION=emotic_cocoer_ft_seed0_val \
bash scripts/emotic-mlcil/launch_cocoer_ft_seed0_tmux.sh
```

The launcher runs the complete Core tests, immutable source audit, full
asset/sample-coverage audit, and mandatory full-path batch-64 CUDA memory smoke
before starting validation. An OOM or missing optimizer step stops before tmux
training is created. Its universal download package contains logs, scores,
metrics, manifests, source audit, asset audit, and memory-smoke JSON but excludes
every `.pth` file.

## Aborted CPU-preprocessing v0.1 run

The registered batch-64 smoke completed before held-out access with peak
allocated/reserved memory of `10951.1/14936.0 MiB`, a successful AdamW update,
and `2013` initialized optimizer-state tensors. The later CPU-preprocessing
execution exposed a severe throughput problem: each process repeatedly decoded
one image and then performed coordinated crop/flip plus three independent PIL
resize/jitter/normalize paths with `WORKERS=0`. A process consumed roughly 48
logical CPU cores while its GPU spent most samples idle; concurrent validation
and formal seeds also contended for disk and memory bandwidth. The user first
reduced the run to formal seed 1 and paused it, then explicitly terminated it
to replace the preprocessing backend. GPU2 returned to 18 MiB after cleanup.

No v0.1 validation or held-out result is registered. Partial task checkpoints
must not be resumed under v0.2 because preprocessing randomness and tensor
implementation identity changed.

## CUDA-preprocessing v0.2

`CocoER-FT-v0.2` leaves the model, loss, label firewall, batch size, optimizer,
epoch schedule, and source augmentation distributions unchanged. Its execution
boundary changes as follows:

1. CPU decodes each JPEG once into its variable-resolution contiguous RGB
   `uint8` tensor and returns original body/head boxes plus height/width;
2. two DataLoader workers prefetch raw samples, while OpenMP/MKL/OpenBLAS are
   limited to four threads per training process;
3. each raw image is transferred once to CUDA without padding the batch to the
   largest image;
4. CUDA tensor operations crop context/body/head at their native sizes, apply
   the coordinated horizontal flip, apply three independent
   brightness/contrast/saturation jitters, and only then resize each view to
   224x224, normalize it, and derive final geometry;
5. validation/test use the same CUDA path with stochastic augmentation disabled.

The raw-image list avoids a high-resolution outlier expanding the whole batch.
The GPU augmentation generator is seed-specific and its complete state is
stored in every task checkpoint. This makes task-boundary recovery reproducible.

Before v0.2 validation, the following server gates are mandatory:

- exact sample-ID equality between CPU v0.1 and CUDA v0.2 eval loaders;
- zero geometry error (up to float rounding tolerance);
- reported max/mean normalized-tensor difference for PIL versus CUDA tensor
  antialiased bilinear resize;
- measured CPU-v0.1 and CUDA-v0.2 samples/second and speedup;
- full batch-64 forward/backward/AdamW smoke including preprocessing peak memory;
- CPU-process audit showing bounded thread use and no starvation of unrelated jobs.

Run the preprocessing benchmark with:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/ddp/bin/python \
  scripts/emotic-mlcil/benchmark_cocoer_gpu_preprocess.py \
  --data-root /mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC \
  --head-cache pretrained/cocoer/emotic_head_boxes_v0.1.json \
  --batch-size 64 --workers 2 --batches 4 \
  --output /tmp/cocoer_gpu_preprocess_benchmark.json
```

The first server candidate gate at commit `283e7bf` passed on GPU 2 while
other jobs were active. For two measured batch-64 iterations it reported
`188.75` versus `130.53` samples/s for CUDA-v0.2 versus CPU-v0.1 evaluation
(`1.446x`) and `64.55` versus `61.31` samples/s for their respective training
augmentation paths (`1.053x`). Sample IDs matched exactly; geometry maximum
absolute error was `1.526e-5`, within the registered `2e-5` float32 tolerance.
The normalized tensor mean/max absolute differences were `0.00485/0.01751`,
as expected from PIL versus CUDA-tensor antialiased bilinear interpolation.
Preprocessing alone peaked at `1375.9/3556.0 MiB` allocated/reserved. The
larger operational benefit is bounded CPU contention: two loader workers and
four math threads replace the observed roughly 48 logical cores per v0.1
process. The full-model batch-64 v0.2 gate subsequently passed on GPU 2 at
commit `27670de`: forward/backward and one AdamW step completed with all five
model paths, `2013` optimizer-state tensors, and `10953.9/14942.0 MiB` peak
allocated/reserved memory. The registered held-out lock token is
`COCOER_FT_TRACK_B_V0_2`.

The v0.1 formal launcher is intentionally absent from the v0.2 branch. A new
configuration-locked formal runner may be added only after v0.2 seed-0
validation completes and the execution configuration is frozen.

At the user's explicit deadline instruction, the frozen v0.2 configuration is
run as one held-out seed only. `launch_cocoer_ft_formal_seed0_tmux.sh` enforces
seed 0, one GPU, batch 64, two loader workers, the v0.2 lock token, at least
18,000 MiB free memory, a clean worktree, and one checkpoint-free result
bundle. One continual seed cannot be sharded over tasks because task `t`
depends on the trained checkpoint from task `t-1`; unvalidated DataParallel is
not used merely to occupy extra GPUs.
