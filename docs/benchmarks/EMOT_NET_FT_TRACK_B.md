# EMOT-Net-FT Track-B Contract

## Identity and scope

`EMOT-Net-FT` converts the classic static EMOTIC model into a deliberately
plain class-incremental lower bound. The immutable reference is the official
[`rkosti/emotic`](https://github.com/rkosti/emotic) repository at commit
`69c3a5106aed08121cd12f6a5b359c745136931e` under MIT. The reference remains
outside this repository in `baseline_sources/emot_net_release_69c3a51/`.

This run is **Track B**, not Track A. It retains EMOT-Net's native visual
architecture and therefore must not be ranked in the unified-CLIP Track-A
table. It uses the same B5-C3 class order, samples, visibility firewall,
validation selection, held-out test policy, metrics, and fixed F1 threshold as
Track A; only the declared model track and model-specific preprocessing differ.

## Frozen static-to-incremental conversion interface

The conversion name is `EMOT-Net-FT-v0.1`:

- input is the full scene plus the annotated person crop;
- the native context branch is the official 640-D Places-style factorized CNN;
- the native body branch is the official 128-D DecomposeMe-style factorized CNN;
- their 768-D concatenation feeds the official 256-D fusion layer, BatchNorm,
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
  objective's categorical coefficient `Wdisc=0.5` is retained.

The categorical source class weight is recomputed at every task from only the
visible current-label tensor:

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

The official Git repository does not bundle its default pretrained files. Its
training instructions download them separately, and `opts.lua` registers:

```text
context: model_myVDavg_640_Places.t7
body:    myVD_ImgNet_66_old.t7
```

The benchmark requires an audited deterministic PyTorch conversion at:

```text
pretrained/emot_net/emot_net_native_init_v0.1.pth
```

It contains only the two native encoder states plus schema and upstream
provenance. The runner records its SHA-256. Missing or mismatched initialization
is a hard error: there is no silent CLIP substitution or random fallback. The
initialization stays server-side and is excluded from result downloads.

After downloading the two official files, convert them once (the converter
requires the pure-Python `torchfile` reader):

```bash
/opt/conda/envs/ddp/bin/python scripts/emotic-mlcil/prepare_emot_net_native_initialization.py \
  --context-t7 /mnt/haoyuan/workspace/baseline_sources/emot_net_pretrained/model_myVDavg_640_Places.t7 \
  --body-t7 /mnt/haoyuan/workspace/baseline_sources/emot_net_pretrained/myVD_ImgNet_66_old.t7 \
  --output pretrained/emot_net/emot_net_native_init_v0.1.pth

sha256sum pretrained/emot_net/emot_net_native_init_v0.1.pth
```

The converter locates one exact conv/BatchNorm tower by its complete layer
shape signature and refuses ambiguous or structurally different `.t7` files.

## Source hyperparameters retained for validation

| Setting | Registered value | Source relation |
|---|---:|---|
| epochs | 14 | `opts.lua` default |
| train batch | 52 | `26*2` default |
| optimizer | SGD | source default |
| learning rate | 0.01 | source default |
| LR drop | epoch 7, ×0.1 | source default |
| momentum | 0.9 | source default |
| weight decay | 5e-4 | source default |
| dropout | 0.5 | source default |
| class norm factor | 1.2 | source default |
| categorical loss coefficient | 0.5 | source joint `Wdisc` default |
| augmentation | none | source default `dataAugment=0` |
| input size | context 224×224; body 128×128 | published structure and inference path |
| normalization | mean `(0.4709,0.4409,0.4062)`, std `(0.2817,0.2741,0.2810)` | `GetImagePatches` |
| F1 threshold | 0.5 | benchmark-wide fixed policy |

These are validation-stage registered settings, not a held-out result freeze.
After seed-0 validation, a separate commit must freeze the reviewed setting
before any test access.

The fixed repository contains one documented discrepancy: the published
`emotic_cnn_model_structure.txt` has fusion BatchNorm+ReLU, while those two
lines are commented in `CreateEmotionModel_BI`. This port follows the official
published model-structure artifact and records the discrepancy in the source
oracle rather than concealing it.

It also contains a body-size inconsistency: `vars.lua` assigns 224, whereas
the published body tower's final `3×3/stride-16` pooling, `main.lua` comment,
and `single_image_inference.lua` correspond to 128. The registered port uses
128 so the published body tower ends in the stated 128-D descriptor. The body
tensor is zero-padded only while crossing the benchmark's tensor batch
boundary and is cropped back before the first body convolution.

## Verification and launch

The source audit checks fixed file hashes, native model names, dual-stream
fusion, 768→256 width, dropout, sigmoid, weighted MSE, schedule, and numerical
class-weight/loss formulas. Unit tests cover registration, Track-B enforcement,
body/context shape, label firewall, lifecycle, checkpointing, parameter growth,
and zero replay.

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
