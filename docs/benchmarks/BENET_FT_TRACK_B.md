# BENet-FT Track-B contract

## Identity and scope

BENet-FT is the static BENet architecture converted into a sequential
fine-tuning lower bound. The immutable external source is
`https://github.com/TristanCladiere/BENet` at
`b86747e0e259b1ec70fc84ca76efd7ea3bb3728e`. No license file or GitHub license
declaration exists at that fixed commit, so upstream source is not copied into
this repository. Runtime preflight verifies the external files by SHA-256.

This is Track B because it retains BENet's native HigherHRNet-W32 backbone and
official COCO-pose initialization. It must not be merged into the Track-A CLIP
ranking.

The W32/512 initialization is fixed at SHA-256
`ef4c99ab8341b5d8c0437b3ea44aec6bd13b82f49b3b2c5e7696ce06c04777f5`.
The runtime rejects every other payload rather than silently using a different
backbone state or random initialization.

## Static-to-incremental mapping

The source no-fusion EMOTIC configuration has bottom-up (BU), person-crop
(PC), masked-context (BG), and class-independent person-detection paths.
BENet-FT retains those paths and replaces each static 26-class classifier with
protocol-ordered task heads. At inference it reports the source-style average
of the BU target-center, PC, and BG sigmoid probabilities.

For Task `t`:

- only Task `t` categorical labels enter the classification loss;
- all previous heads remain present, but no old-label truth is supplied;
- the HigherHRNet trunk and all existing model parameters remain trainable;
- the target bounding box may supervise the class-independent detector;
- there is no teacher, distillation, replay, EWC, Adapter, CLIP image encoder,
  or CLIP text encoder.

The training loader cycles equally through the source's four input roles:
detection scene, BU scene, extracted person, and masked context. Validation
selects the earliest best epoch using current-label validation mAP only. Test
is inaccessible unless the configuration-locked runner is used after the
validation configuration is frozen.

## Data and source differences

Each benchmark row represents one annotated target person. The tensor-only
method boundary transports full-scene RGB, target-person RGB, masked-context
RGB, the target-box mask, and a valid-image mask. The target box is geometry,
not an emotion label. The benchmark therefore supervises the target person,
whereas the original full-image detector can receive annotations for all
people in an image.

The main-table mapping uses no HECO extra data. BENet's reported `28.75` mAP
uses its emotion-detection extension, while the no-extra-data three-path result
is `27.73`; neither number is presented as the incremental benchmark result.
The source's optional periodic test evaluation is prohibited.

## Initial validation configuration

| Setting | Registered value |
|---|---:|
| Input | `512 × 512`, BU + person + masked context |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Maximum epochs/task | `25` |
| Early stopping patience | `5` |
| Train batch | `24` |
| Branch allocation | equal cyclic det/BU/PC/BG |
| Heatmap loss weight | `1.0` |
| width/height L1 weight | `0.1` |
| Classification loss | source `FocalTagLoss` |
| Main-table F1 threshold | fixed `0.5` |

Server CUDA smoke at batch 24 completed an Adam update on an RTX 4090 with
`12,866.6 MiB` peak allocated and `14,426 MiB` peak reserved. This validates
one BENet-FT process per 24 GiB GPU; multiple BENet workers must not share a
single 4090.

The source static schedule is 250 epochs on the complete 26-class dataset.
Repeating 250 epochs for each of eight incremental tasks would be a different
compute budget; the 25-epoch cap is an explicit benchmark adaptation and must
be frozen from validation before held-out test access.

## Validation entry point and downloads

Use `scripts/emotic-mlcil/launch_benet_ft_seed0_tmux.sh`. Preflight requires
the fixed external source and official
`pose_higher_hrnet_w32_512.pth`, runs the source/operator gate, tests, and a
batch-24 GPU memory smoke before starting seed-0 validation.

The launcher defaults `BENET_CPU_THREADS=2` and explicitly propagates the
corresponding OpenMP, MKL, OpenBLAS, and NumExpr limits into the tmux command.
This prevents an existing tmux server from silently retaining a high-thread
environment while other baselines are running. The training log emits one
JSON record per epoch (`benet_epoch_complete`) with the task/epoch identifiers,
validation mAP, epoch duration, cumulative task duration, optimizer-step
counts, and early-stopping state. It additionally emits approximately ten
`benet_batch_progress` records per epoch with completed/total batches, seconds
per batch, and an estimated remaining duration for that epoch. Task 0 must
finish before benchmark artifact directories appear; the launcher log is
therefore the authoritative live progress source during the first task.

Every completed run follows `DOWNLOAD_STANDARD.md`: the launcher creates one
`download_packages/<RUN_ID>.tar.gz` and adjacent `.sha256`. The archive includes
metrics, score artifacts, manifests, reports, logs, and source-equivalence
evidence, and explicitly excludes all `.pth` checkpoints and other large
training state.
