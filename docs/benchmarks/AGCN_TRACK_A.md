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

## Freeze gate

Seed-0 validation must be inspected for task curves, loss stability, ACM
density, optimizer steps/skips, CUDA peak, source-equivalence error, and archive
SHA. Only then may the validation snapshot and exact GloVe/optimizer/config
identity be committed. A separate configuration-locked runner can then launch
held-out seeds 0--2, preferably on independent GPUs. No test-based hyperparameter,
threshold, embedding, or loss change is allowed.
