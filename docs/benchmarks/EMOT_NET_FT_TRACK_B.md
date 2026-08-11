# EMOT-Net-FT Track-B Contract

## Identity and scope

`EMOT-Net-FT` converts the classic static EMOTIC model into a deliberately
plain class-incremental lower bound. The immutable reference is the official
[`rkosti/emotic`](https://github.com/rkosti/emotic) repository at commit
`69c3a5106aed08121cd12f6a5b359c745136931e` under MIT. The reference remains
outside this repository in `baseline_sources/emot_net_release_69c3a51/`. Native
weights and the matching executable variant come from the official Dropbox
`emotic_pami_git.zip` release, fixed at SHA-256
`ce6096c1af5a3e91badbc06752e2dbbd04fc63f67f24acc95d76a68e1f7e339b`.

This run is **Track B**, not Track A. It retains EMOT-Net's native visual
architecture and therefore must not be ranked in the unified-CLIP Track-A
table. It uses the same B5-C3 class order, samples, visibility firewall,
validation selection, held-out test policy, metrics, and fixed F1 threshold as
Track A; only the declared model track and model-specific preprocessing differ.

## Frozen static-to-incremental conversion interface

The conversion name is `EMOT-Net-FT-v0.1`:

- input is the full scene plus the annotated person crop;
- the native context branch is the official 640-D Places-style factorized CNN;
- the native body branch is the official release's 256-D grouped AlexNet CNN;
- their 896-D concatenation feeds the official 256-D fusion layer, BatchNorm,
  ReLU, and dropout `0.5`;
- the static 26-class categorical layer is replaced by protocol-ordered,
  expanding linear task heads;
- at task `t`, only `targets_current` enter the source weighted sigmoid-MSE;
- the two towers, fusion module, previous heads, and new head remain in the
  sequential fine-tuning model, but previous heads receive no direct loss;
- no teacher, distillation, replay, Fisher penalty, Adapter, prompt, CLIP
  visual tower, or CLIP text feature is added;
- continuous valence/arousal/dominance prediction is omitted because this
  MLCIL contract evaluates only the 26 categorical labels; the source joint
  objective's categorical coefficient `Wdisc=1/6` is retained.

The release sets `reWeight=1`, so the categorical source class weight is
recomputed for every training mini-batch from only its visible current-label
tensor:

```text
w_c = 0.0001                          when positive_count_c < 1
w_c = 1 / log(1.2 + positive_count_c) otherwise
```

Using the original all-26-class training label matrix would expose future
labels and is forbidden. Validation checkpoint selection uses current-class
mAP only; held-out test never selects an epoch or hyperparameter.

The release's precomputed `class_sampling.t7` indexes samples using the static
26-class label space. Reusing it during early tasks would make future-label
membership influence training. The port therefore uses the benchmark's
uniform shuffled current-task sample view and records this necessary protocol
difference; it does not claim byte-for-byte reproduction of the static sampler.

## Native initialization gate

The official Git repository does not bundle pretrained files. Its instructions
link the official Dropbox package. That package contains a matching source
variant whose defaults and assets are:

```text
context: model_myVDavg_640_Places.t7
body:    alexnet_features.t7
```

The benchmark requires an audited deterministic PyTorch conversion at:

```text
pretrained/emot_net/emot_net_native_init_v0.1.pth
```

It contains only the two native encoder states plus schema and upstream
provenance. The runner records its SHA-256. Missing or mismatched initialization
is a hard error: there is no silent CLIP substitution or random fallback. The
initialization stays server-side and is excluded from result downloads.

Convert the official ZIP directly once (the converter requires the pure-Python
`torchfile` reader):

```bash
/opt/conda/envs/ddp/bin/python scripts/emotic-mlcil/prepare_emot_net_native_initialization.py \
  --release-archive /mnt/haoyuan/workspace/baseline_sources/emot_net_pretrained/emotic_pami_git_official.zip \
  --output pretrained/emot_net/emot_net_native_init_v0.1.pth

sha256sum pretrained/emot_net/emot_net_native_init_v0.1.pth
```

The converter verifies the complete ZIP SHA, four release-source member hashes,
and both asset hashes before loading anything. It then locates the context and
body towers by their complete layer-shape signatures and selects saved
DataParallel replica 0 exactly as the upstream `features:get(1)` path does.
All matching-replica hashes and the selected index are recorded; structurally
different content is refused. The registered asset hashes are:

```text
model_myVDavg_640_Places.t7  bbf8a09edb1a17338f8004b78cf2e83f3cccc3a7f0bf7c3705368b482cab1e7c
alexnet_features.t7          0abdbce4910f4c242433d614287448d110a81e7d26562ab291364762cf2dae87
```

The fixed Git commit defaults to a DecomposeMe body, but its linked Dropbox no
longer distributes `myVD_ImgNet_66_old.t7`. Silently mixing that unavailable
default with another checkpoint is forbidden. The registered benchmark instead
uses the internally consistent, executable official Dropbox release variant:
Places context + bundled AlexNet body. This difference is explicit provenance,
not a benchmark-added backbone substitution.

## Frozen source hyperparameters

| Setting | Registered value | Source relation |
|---|---:|---|
| epochs | 21 | Dropbox release `OptsEmotionModel.lua` default |
| train batch | 52 | `26*2` default |
| optimizer | SGD | source default |
| learning rate | 0.01 | source default |
| LR drop | epoch 7, ×0.1 | source default |
| momentum | 0.9 | source default |
| weight decay | 5e-4 | source default |
| dropout | 0.5 | source default |
| class norm factor | 1.2 | source default |
| categorical loss coefficient | 1/6 | Dropbox release joint `Wdisc` default |
| class reweighting | each training mini-batch | Dropbox release `reWeight=1` |
| augmentation | none | source default `dataAugment=0` |
| input size | context 224×224; body 128×128 | Dropbox release `common_variables.lua` |
| normalization | mean `(0.4709,0.4409,0.4062)`, std `(0.2817,0.2741,0.2810)` | `GetImagePatches` |
| F1 threshold | 0.5 | benchmark-wide fixed policy |

Seed-0 validation completed without changing these settings. They are frozen
for the held-out test; test metrics must not alter the model, optimizer,
schedule, selection rule, loader, or fixed F1 threshold.

The release source explicitly uses body size 128 and applies fusion
BatchNorm+ReLU. The body tensor is zero-padded only while crossing the
benchmark's tensor batch boundary and is cropped back before the first body
convolution.

## Verification and launch

The source audit checks the fixed Git hashes, registered release ZIP/source and
asset hashes, dual-stream fusion, 896→256 width, grouped AlexNet body, dropout,
sigmoid, weighted MSE, schedule, and numerical class-weight/loss formulas. Unit
tests cover registration, Track-B enforcement, body/context shape, label
firewall, lifecycle, checkpointing, parameter growth, deterministic Torch7
replica-0 handling, and zero replay.

After preparing the initialization on the server:

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

RUN_ID="emot_net_ft_seed0_val_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
SESSION=emotic_emot_net_ft_seed0_val \
EMOT_NET_NATIVE_INIT="$PWD/pretrained/emot_net/emot_net_native_init_v0.1.pth" \
bash scripts/emotic-mlcil/launch_emot_net_ft_seed0_tmux.sh
```

On success the launcher creates the universal `.tar.gz` and `.sha256` with
scores, metrics, manifests, audit, and logs. All `.pth` files are excluded.

## Seed-0 validation freeze

The audited validation run `emot_net_ft_seed0_val_20260811_133117` used clean
commit `7851386be204485b78765e184d48b811d150b7e4`. It completed all eight tasks,
168 epochs, and 12,012 optimizer updates with no skipped update, NaN, OOM, or
traceback. Its validation-only metrics were:

| Metric | Value |
|---|---:|
| Final mAP | 27.3062 |
| Average mAP | 33.5043 |
| Forgetting | 5.9747 |
| Final cF1 | 24.6012 |
| Final oF1 | 53.5006 |

The result is deliberately not main-table eligible because it reports `val`
and predates the configuration lock. Complete provenance, per-task curves,
best epochs, parameter counts, native hashes, memory smoke, and archive SHA are
stored in
[`results/emot_net_ft_seed0_validation_v0.1.json`](results/emot_net_ft_seed0_validation_v0.1.json).
No hyperparameter was changed after reviewing validation.

## Locked held-out scope and entry point

The formal scope is one seed only: seed 0. Seeds 1 and 2 were not launched,
and the final report presents a single value rather than a mean or standard
deviation. The reproducibility entry point requires a clean frozen commit, an explicit
commit match, the confirmation string `EMOT_NET_FT_TRACK_B_V0_1`, at least
4 GiB free GPU memory, and at least 8 GiB free output-filesystem space. It
forces `reporting_split=test`, `configuration_locked=true`, train/eval batches
`52/16`, and `workers=0`:

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

FROZEN_COMMIT="$(git rev-parse HEAD)"

RUN_ID="emot_net_ft_formal_seed0_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
SESSION=emotic_emot_net_ft_formal_seed0 \
EXPECTED_GIT_COMMIT="$FROZEN_COMMIT" \
CONFIGURATION_LOCKED_CONFIRMATION=EMOT_NET_FT_TRACK_B_V0_1 \
EMOT_NET_NATIVE_INIT="$PWD/pretrained/emot_net/emot_net_native_init_v0.1.pth" \
bash scripts/emotic-mlcil/launch_emot_net_ft_formal_seed0_tmux.sh
```

The worker validates the eight task artifacts and locked configuration before
creating one checkpoint-free download archive. A formal result with a dirty
tree, validation reporting, missing scores, changed source settings, or more
than seed 0 is rejected.

## Frozen formal result

The requested one-seed formal run `emot_net_ft_formal_seed0_20260811_163003`
completed from clean configuration-freeze commit
`4e44cd135b73e4a61fb01baf86257b246e11eb29`. Its held-out result is:

| Metric | Seed 0 test |
|---|---:|
| Final mAP | 20.2518 |
| Average mAP | 26.0717 |
| Forgetting | 7.2697 |
| Final cF1 | 20.4463 |
| Final oF1 | 43.7496 |

Per-task mAP was `36.1286, 29.7385, 24.3067, 27.9606, 25.0059,
23.3014, 21.8804, 20.2518`. The run completed 12,012 optimizer updates with no
skip, NaN, OOM, traceback, or fallback. The training records are identical to
the validation run, confirming that held-out behavior did not alter training.
The largest class forgetting occurred for Anticipation (`45.8498`), Affection
(`30.8843`), Annoyance (`14.8863`), and Confidence (`14.3318`), which is the
intended behavior of this no-anti-forgetting sequential-FT lower bound.

The final task contains 5,368 unique score rows of shape `5368×26`; every ID
has the `emotic:test:` prefix and none has a validation prefix. The checkpoint-
free archive contains all eight canonical score files, no `.pth`, and passed
all nested manifest hashes. Its SHA-256 is:

```text
d3d717a56f2ff192e88b160c6b024ac34e2bb584d045b6955497442c72c8217d
```

Complete metrics, all 26 class-forgetting rows, configuration, source hashes,
test-ID audit, and validation-to-test context are frozen in
[`results/emot_net_ft_seed0_formal_v0.1.json`](results/emot_net_ft_seed0_formal_v0.1.json).
No additional seed, rerun, threshold calibration, or post-test tuning is
required. Paper tables must report the seed-0 values directly without `±`.
