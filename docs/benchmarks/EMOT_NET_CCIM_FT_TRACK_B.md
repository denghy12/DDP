# EMOT-Net+CCIM-FT Track-B Contract

## Identity

`EMOT-Net+CCIM-FT` is the static EMOT-Net+CCIM model converted into the same
plain sequential fine-tuning lifecycle as `EMOT-Net-FT`. It is **Track B**:
the host keeps native EMOT-Net's Places-context and AlexNet-body towers rather
than substituting the unified Track-A CLIP visual encoder. It must therefore
be compared with `EMOT-Net-FT` as a paired Track-B ablation, not inserted into
the Track-A ranking.

The immutable CCIM reference is
[`ydk122024/CCIM`](https://github.com/ydk122024/CCIM) at
`d6a651f91d1c1c91faca862ddeea915df9314919`, under MIT. The release contains
only the core module and dictionary recipe, not a complete EMOT-Net training
repository or a downloadable dictionary. `CCIM.py` is fixed at SHA-256
`2e1b2f1178fcdc556aee7efed743c624020d43c34ec2272f7415dee4ce776d26`.
The native host and initialization retain the complete source/asset contract
in [EMOT_NET_FT_TRACK_B.md](EMOT_NET_FT_TRACK_B.md).

## Architecture retained

The input, native towers and `896→256` joint feature are identical to
`EMOT-Net-FT`. CCIM is inserted immediately before the expanding task heads:

- dot-product backdoor intervention (`dp_cause`), the paper default;
- query/key width `256`;
- joint projection `W_h: 256→128`;
- confounder projection `W_g: 2048→128`;
- source BatchNorm, `128→512→128` residual classifier, dropout `0.5`, and
  residual multiplier `0.3`;
- one expanding linear head block `128→current classes` per task.

The executable external-source audit copies official weights into the port and
requires exact dot-product/additive intervention and EMOTIC-logit equivalence.
No upstream source file is copied into this repository.

## Protocol-safe confounder dictionary

The static paper builds the EMOTIC dictionary from every training image. That
is invalid in an online class-incremental run because it would inspect images
assigned to future tasks. The registered mapping therefore builds the resource
from **Task-0-accessible training samples only**, then freezes it for all tasks
and all run seeds:

1. Select only train persons whose visible label set intersects Task 0.
2. Mask the target-person bounding box with black pixels.
3. Extract the last-pool `2048`-D feature using ResNet-152 Places365.
4. Run deterministic K-Means++ with `K=256`, `n_init=10`, and seed `0`.
5. Store centroids and cluster-frequency priors, plus the ordered sample-ID,
   feature-checkpoint, source, protocol and construction hashes.

The runner rejects random dictionaries, ImageNet dictionaries, full-training
dictionaries, mismatched protocols, altered source provenance and wrong tensor
dimensions. The fixed dictionary is an architectural auxiliary resource—not
stored replay examples—so replay memory remains zero; its exact persistent
byte count is reported separately in `config_resolved.json`.

The preparation step requires an audited PyTorch-compatible ResNet-152
Places365 checkpoint. The official Places365 release primarily publishes the
paper's ResNet-152 as Torch/Caffe assets, so an arbitrary torchvision/ImageNet
checkpoint must not be substituted. Any conversion must first be independently
hash-registered and then supplied explicitly:

```bash
bash scripts/emotic-mlcil/prepare_ccim_source.sh
```

```bash
CCIM_PLACES365_EXPECTED_SHA256="<verified-converted-checkpoint-sha256>" \
CCIM_PLACES365_SOURCE="<official-asset-url + deterministic-conversion-id>" \
/opt/conda/envs/ddp/bin/python \
  scripts/emotic-mlcil/prepare_ccim_task0_dictionary.py \
  --protocol configs/emotic_mlcil/protocol_b5c3_track_b.yaml \
  --data-root /mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC \
  --places365-checkpoint /mnt/haoyuan/workspace/baseline_sources/places365/resnet152_places365.pth.tar \
  --ccim-source-root /mnt/haoyuan/workspace/baseline_sources/ccim_official \
  --output pretrained/ccim/emotic_task0_places365_k256_v0.1.pth \
  --device cuda:0
```

## Method-FT lifecycle

At task `t`, the model adds only the classifier rows for the current classes.
Training receives current-class labels only and updates the complete native
EMOT-Net host, CCIM, previous heads and new head. It uses the same source
weighted sigmoid-MSE and optimizer schedule as the paired `EMOT-Net-FT` run:

| Setting | Frozen value |
|---|---:|
| Epochs per task | `21` |
| Batch size | `52` |
| Optimizer | SGD |
| Learning rate | `0.01` |
| Momentum | `0.9` |
| Weight decay | `5e-4` |
| LR step / multiplier | epoch `7` / `0.1` |
| Early stopping patience | `21` |
| Gradient clip | `10` |
| AMP / TF32 | false / false |
| Replay / distillation / EWC / Adapter | none |
| F1 threshold | fixed `0.5` |

Validation selects an epoch only through current-label validation mAP. Old and
future train truth are never exposed. Held-out test remains inaccessible until
the configuration is frozen and the explicit lock token is supplied.

## Validation and download

After the dictionary and native initialization exist, launch one validation:

```bash
RUN_ID="emot_net_ccim_ft_seed0_val_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
SESSION=emotic_emot_net_ccim_ft_seed0_val \
bash scripts/emotic-mlcil/launch_emot_net_ccim_ft_seed0_tmux.sh
```

The launcher runs all Core/legacy tests, immutable-source equivalence and CUDA
memory smoke before creating the training tmux session. On success it produces
one universal `download_packages/$RUN_ID.tar.gz` and adjacent `.sha256`, with
all standard metrics, score tensors and logs but no `.pth` checkpoints.
