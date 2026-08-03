# Original-DDP-Tau2 Track-A Source Audit and Porting Contract

## Scope and name

This baseline answers a narrow comparison question: how the collected original
DDP implementation performs on the frozen EMOTIC B5-C3 protocol when its PCD
temperature is matched to the repository-local modified DDP result.

The registered name is **Original-DDP-Tau2**, not simply `DDP`, because one
evaluation rule is intentionally changed at the user's request:

| Setting | Collected source | Registered baseline |
|---|---:|---:|
| PCD minimum temperature | 1.0 | 1.0 |
| PCD maximum temperature | 7.0 | **2.0** |
| PCD exponent γ | 0.2 | **0.7** |

The exact registered schedule is
`T(C_t) = 1 + ((C_t - 5) / (26 - 5))^0.7`. No test result may be
reported as the source-original `T=1→7, γ=0.2` configuration.

Development branch: `codex/emotic-baseline-original-ddp`, created from the
frozen L3A Track-A result commit
`0ace49894c33095bdb91dc63d71e52f87b151463`.

## Fixed external source

The read-only source is kept outside this repository at:

`/Users/denghaoyuan/workspace/MyCode/baseline_sources/CODE_DDP/`

It contains 30 audited files and 1,470,709 bytes after excluding `.DS_Store`,
Git metadata, bytecode, and cache files. Its canonical sorted-record tree
SHA-256 is
`e0b9963e1d891ce95fc0c0d2444389f388fe5a1c2bd04f6ece7dc0617306e5a4`.
The adjacent portable upload archive is
`original_ddp_snapshot_e0b9963.tar.gz` with SHA-256
`b99892e1c12c942b4a8506b89049f8f35933001e0b8018e928883ee60db847a5`;
extracting it reproduces the same 30-file tree hash.
Critical file hashes are:

| File | SHA-256 |
|---|---|
| `models/ddp.py` | `d7dfb6bd463b387db3224a8fb6631e25435bae5c174e65f22f328899fc42b309` |
| `DDP.py` | `76d0767f9de062cf7939dc718f5ed2a6c2c4c3f98c2dd680a47652cea8501fca` |
| `bce_loss.py` | `eaedae20e4614874134565468d359a4ec2795726078b08eb96e74043e29d28c0` |
| `clip/model.py` | `b04d0d4f8e7d0dcab3dc9493a97e9eee7c5e0c58f85379292ec3ebf5a398c952` |
| `configs/models/vitb16_ep50.yaml` | `93034d5a382da508624d689cbc95c8f6ef544d5d8d9290f8e6bfe5f1b3158b8a` |

The snapshot has no `.git` metadata, verifiable remote, commit identifier, or
license declaration. It is therefore an immutable user-collected reference,
not a verified official Git checkout. No source file from it is copied into
the benchmark repository. The executable audit rejects any change to the
complete snapshot before a run starts.

## Audited released behavior

The source preallocates class-specific learnable positive and negative text
contexts for the complete class vocabulary. Both contexts contain 16 tokens.
It also preallocates two 16-token visual prompts per class and injects them into
CLIP ViT-B/16 transformer layers 7--11. Training updates only positive
contexts, negative contexts, and visual prompts; the CLIP visual/text towers
are excluded from Adam.

For each current class, DDP constructs negative and positive visual paths. It
compares every prompted image token with the corresponding text feature,
derives token attention from the negative path, shares those attention weights
between both paths, and produces two logits per class. The objective is the
sum over samples and current classes of the two-path softmax BCE, multiplied by
`0.03`.

The retained optimizer and schedule are:

- one Adam instance across all eight tasks;
- learning rate `5.9e-3`, betas `(0.9, 0.999)`, epsilon `1e-8`, and no weight
  decay;
- one MultiStepLR across all tasks, milestones `[0, 20]`, gamma `0.1`;
- 20 epochs per task and train batch size 8; and
- one AMP GradScaler across the complete continual sequence.

The method has no replay, classifier Adapter, feature Adapter, distillation
teacher, or old/future ground-truth access. Its complete prompt capacity is
known at initialization, so physical parameter growth is reported as zero and
the preallocated prompt capacity is reported separately.

## Track-A and benchmark mappings

The following mappings are required to put the source on the same EMOTIC
contract as the other Track-A methods:

- source VOC B4-C2 class/task definitions are replaced by frozen EMOTIC B5-C3;
- the shared benchmark EMOTIC sample views, class order, image preprocessing,
  and sample IDs replace the source VOC loader/transforms;
- training receives only current-class targets and evaluation receives only
  seen-class targets through the benchmark interfaces;
- the benchmark keeps incomplete final mini-batches instead of source
  `drop_last=True`, so every protocol-selected person sample is used;
- validation exposes current labels only and is monitoring evidence because
  the fixed 20-epoch schedule performs no epoch selection;
- cF1/oF1 use the frozen global threshold 0.5; and
- PCD uses the explicit benchmark-matched `T=1→2, γ=0.7` schedule above.

Freezing CLIP tensors with `requires_grad=False` is a gradient-elision
correction: the source optimizer excludes the same tensors. It changes neither
the forward values nor updates to the three prompt tensors.

The benchmark calls the repository's refactored DDP model only with both
`feature_adapter` and `feature_adapter_bank` absent. Preflight dynamically
extracts the external source `DDP.forward` and `BCELoss` with AST, executes
them on controlled tensors, and compares forward values, loss, and gradients.
It also compares normalized AST hashes for all ten CLIP definitions used by
the prompt-aware ViT builder plus the complete prompt learner, and executes a
separate text-encoder forward check. This proves the adapter-free head, text,
and visual-prompt backbone paths rather than merely checking similarly named
configuration fields.

## Required gates and execution order

Before held-out test access, this branch must pass:

1. all Core/baseline and selected legacy regression tests;
2. complete external snapshot identity verification;
3. zero-error/tolerance executable BCE and DDP-forward audit;
4. current-label visibility, no-Adapter, no-replay, checkpoint continuation,
   optimizer/scheduler lifecycle, and score-alignment unit tests;
5. Task-0 train plus final 26-class inference CUDA memory smoke; and
6. one seed-0 validation-only run from a clean commit.

Only after the seed-0 validation result is reviewed may the configuration be
frozen and the confirmation `ORIGINAL_DDP_TAU2_TRACK_A_V0_1` be used for
held-out test. Seeds 0--2 can then run concurrently on distinct GPUs without
using one held-out seed to tune another.

The seed-0 validation entry point is:

```bash
RUN_ID="original_ddp_tau2_seed0_val_$(date +%Y%m%d_%H%M%S)" \
GPU=0 \
SESSION=emotic_original_ddp_tau2_seed0_val \
bash scripts/emotic-mlcil/launch_original_ddp_seed0_tmux.sh
```

The launcher uses method-specific `ORIGINAL_DDP_*` environment variables, so
stale KRT/CSC/MULTI-LANE/L3A exports cannot redirect output or alter batch
sizes. Default evaluation batch size is deliberately 1 because final-task DDP
expands one image into 52 visual-prompt paths; increase it only after the CUDA
smoke demonstrates sufficient memory.

On success, download only
`download_packages/<run-id>.tar.gz` and its adjacent `.sha256`. The universal
[checkpoint-free download standard](DOWNLOAD_STANDARD.md) includes metrics,
canonical score tensors, manifests, source-oracle JSON, and logs, while
rejecting every `.pth` and checkpoint directory.

## Current status

The independent adapter, source oracle, tests, GPU smoke, runner, packaging,
and seed-0 tmux launcher are implemented. No validation or held-out result has
yet been registered. Hyperparameters remain source-fixed except for the
explicit user-requested PCD mapping; they must not be changed after observing
held-out test data.
