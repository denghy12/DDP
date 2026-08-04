# AGCN Track-A source audit and porting contract

## Immutable source identity

- Paper: *AGCN: Augmented Graph Convolutional Network for Lifelong Multi-label
  Image Recognition*, ICME 2022, arXiv:2203.05534.
- Official repository: `https://github.com/Kaile-Du/AGCN`.
- Fixed commit: `3afe2ecbbef0051c6e841a97c369885011a683f0`.
- Fixed tree: `6c25689b81d0807523108a782ee59630d25a9b40`.
- Fixed codeload archive SHA-256:
  `b5843d3ee0964767b48c49ba4b71fdf93c4bf954460669ed2b1cd05f9504f301`.
- License: the root `LICENSE` is Apache-2.0 at the fixed commit. `GCN.py` also
  contains a bare `License: BSD` header comment without separate license text;
  no upstream source is copied into the benchmark.
- The fixed extraction is byte-identical to the previously collected
  `AGCN-main/` snapshot after excluding `.DS_Store`.

The official source stays outside this repository at
`baseline_sources/agcn_release_3afe2ec/`. Only an independent implementation,
tests, hashes, and audit scripts enter `benchmarks/emotic_mlcil/`.

## Audited original algorithm

The paper uses an ImageNet-pretrained trainable ResNet-101 and 300-dimensional
GloVe label embeddings. A two-layer GCN maps `300 → 1024 → 2048`, making one
classifier vector per seen class. Images and graph-generated classifier vectors
are trained together for one epoch per task.

Task 0 builds a hard-hard label correlation graph. Later tasks retain the old
graph and add hard-hard current blocks plus soft-hard and Bayes-reversed
hard-soft cross-task blocks. The frozen previous CNN/GNN supplies old logits and
old graph nodes. Training uses current-class sigmoid BCE, old-class sigmoid BCE
distillation, and MSE between old graph nodes. It is rehearsal-free: no old
image sample or replay buffer is read.

The release and paper are not completely self-contained:

- `GCN.py` uses undefined variables `a`, `b`, and `c`; paper Table 3 resolves
  these as classification `0.07`, distillation `0.93`, relationship `1e5`.
- Distillation uses teacher sigmoid logits, while the online ACM is fed teacher
  softmax probabilities. The port preserves this released behavior.
- Released Adam calls inherit PyTorch epsilon `1e-8`; the paper says `1e-4`.
  Track A preserves executable released behavior and records the discrepancy.
- Task-0 graph scale/exponent are `0.28/-0.8`; incremental current/cross scale
  and exponent are `0.25/-0.5`. Thresholds are Task-0 `0`, current `0.4`, and
  cross-task `0.3`; reverse Bayes edges include the released factor `1/2`.
- Paths point to per-task `glove_wordEmbedding.pkl` files that are absent from
  the repository and both released dataset ZIP files.

## Track-A mapping

Only the visual feature extractor changes:

| Component | Original | Track A |
|---|---|---|
| Visual backbone | Trainable ResNet-101, 2048-d | Trainable OpenAI CLIP ViT-B/16 visual tower, 512-d |
| Label semantics | GloVe 300d | Same GloVe family and dimension |
| GCN | `300 → 1024 → 2048` | `300 → 1024 → 512` to match visual width |
| ACM/KD/relation loss | Released AGCN | Retained |
| Replay | None | None |
| Adapter | None | None added |
| CLIP text encoder | None | Not used |

The training label firewall exposes only current-class targets. Old labels come
only from the frozen teacher; future labels are inaccessible. Validation reports
current-class mAP after the fixed one epoch but does not select an epoch or
threshold. Held-out test requires `configuration_locked=true`.

## GloVe asset gate

The release does not include its referenced embedding pickles. Track A fixes a
machine-readable 26-by-300 mapping from the standard
[GloVe 6B 300d](https://nlp.stanford.edu/projects/glove/) vectors, whose
pretrained data are released under PDDL 1.0. To avoid the 822MB four-width ZIP,
the preparation gate accepts Gensim's official data-registry repack: the same
400K×300 Wikipedia-2014/Gigaword-5 vectors converted to word2vec text and gzip.
The fixed gzip is 394,362,229 bytes with registry MD5
`29e9329ac2241937d55b852e8284e89b` and observed SHA-256
`0a7aebbe49097dc6e5ffff7a25e9aa20181a6e862050ca68bd2f36e056739e00`.

Every single-word EMOTIC class maps to its lowercase token except
`Disquietment`: the exact word is absent from the fixed 400K vocabulary, so its
same-root token `disquiet` is frozen before validation. `Doubt/Confusion` is the
arithmetic mean of `doubt` and `confusion`. The JSON records the compressed
source SHA-256/MD5, distribution/conversion, token mapping, protocol class
order, and all vectors. The benchmark refuses a wrong class order, width,
non-finite vector, source MD5, or checkpoint/asset hash mismatch.

The recommended path is to prepare the small JSON locally, then upload only
that asset:

```bash
cd /Users/denghaoyuan/workspace/MyCode/CODE_DDP-benchmark

python3 \
  scripts/emotic-mlcil/prepare_agcn_glove_embeddings.py \
  --glove /Users/denghaoyuan/workspace/MyCode/baseline_sources/glove6b/glove-wiki-gigaword-300.gz \
  --output pretrained/agcn/emotic_glove_6b_300d.json

scp -P 9205 pretrained/agcn/emotic_glove_6b_300d.json \
  root@172.31.214.226:/mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1/pretrained/agcn/
```

The generator has a standard-library-only parser for the frozen class list so
the local preparation does not require PyTorch or PyYAML. The server method
still validates the complete protocol and checks that the asset class order is
identical.

## Validation entry point

After the branch is clean and the fixed source/archive/GloVe asset are present:

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

RUN_ID="agcn_seed0_val_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
SESSION=emotic_agcn_seed0_val \
bash scripts/emotic-mlcil/launch_agcn_seed0_tmux.sh
```

The launcher runs Core and legacy regressions, fixed-source hashes and numerical
operator equivalence, and a conservative final-task teacher/two-Adam memory
smoke before starting seed 0. On success, download only:

```text
<run-root>/download_packages/<run-id>.tar.gz
<run-root>/download_packages/<run-id>.tar.gz.sha256
```

The universal packager includes results, canonical scores, manifests, logs,
the upstream-oracle JSON, and hashes while rejecting every nested `.pth`.

## Frozen validation result

Seed 0 completed the validation-only run `agcn_seed0_val_20260804_130529`
from clean commit `c8ff114`. Final mAP was `25.3571`, Average mAP was
`28.9310`, fixed-0.5 Final cF1/oF1 were `16.9357/55.8194`, and Forgetting was
`1.1844`. These are selection-split diagnostics and are not eligible for the
held-out main table.

All eight tasks completed one epoch for a total of 3,701 optimizer updates with
zero skipped steps, NaN, OOM, or traceback. The graph remained finite and
became progressively sparse, from density `1.0` at Task 0 to `0.1420` at Task
7. The immutable-source oracle's largest numerical difference was `4.47e-8`.
The conservative final-task teacher/two-Adam smoke allocated `2347.7 MiB` and
reserved `2640 MiB` at batch 8. The checkpoint-free validation archive has
SHA-256 `568387f0674be605d89e8ce9316a49a049c06fe10deeaee62100b13927a56f7f`.
The complete machine-readable evidence is in
`results/agcn_seed0_validation_v0.1.json`.

The following identity is frozen before any held-out result is read:

- one epoch per task, visual LR `1e-4`, and graph LR `3e-5`;
- classification/distillation/relationship weights `0.07/0.93/1e5`;
- Task-0/current/cross thresholds `0/0.4/0.3`;
- graph scales `0.28/0.25`, reverse-Bayes factor `0.5`, and degree exponents
  `-0.8/-0.5`;
- the 26-by-300 GloVe JSON at SHA-256
  `e926b8672cd761169586668c54e4676093b152e2b02a218747f55bf72e1d1bee`;
- train/eval batches `8/32`, workers `0`, FP32 training, and global F1
  threshold `0.5`.

No held-out metric may change an embedding token, loss, threshold, optimizer,
graph constant, epoch count, or batch setting.

## Locked three-seed formal entry point

The formal launcher requires the full frozen repository commit and the literal
confirmation `AGCN_TRACK_A_V0_1`. By default it starts three isolated seed
processes concurrently on physical GPU 0. The measured aggregate reserved
memory estimate is about `7.9 GiB`; the launcher requires at least `12 GiB`
free before starting and blocks aggregation if any seed fails.

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

FROZEN_COMMIT="$(git rev-parse HEAD)"

RUN_ID="agcn_formal_seed012_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
SESSION=emotic_agcn_formal_seed012 \
EXPECTED_GIT_COMMIT="$FROZEN_COMMIT" \
CONFIGURATION_LOCKED_CONFIRMATION=AGCN_TRACK_A_V0_1 \
bash scripts/emotic-mlcil/launch_agcn_formal_seed012_tmux.sh
```

The launcher reruns Core/legacy tests, immutable-source equivalence, the CUDA
memory smoke, and the single-GPU capacity check before creating tmux. After all
three workers finish, `validate_agcn_formal_results.py` rejects configuration,
asset, provenance, task-count, score-file, or training-log drift and reports
mean plus sample standard deviation. The universal packager then produces:

```text
<run-root>/download_packages/<run-id>.tar.gz
<run-root>/download_packages/<run-id>.tar.gz.sha256
```

All `.pth` checkpoints remain server-only; canonical `.pt` scores, metrics,
manifests, preflight/runtime logs, source oracle, and the three-seed aggregate
are included in the download package.
