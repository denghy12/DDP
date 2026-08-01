# KRT Track-A Integration

## Fixed source

- Paper: [Knowledge Restore and Transfer for Multi-Label Class-Incremental
  Learning, ICCV 2023](https://arxiv.org/abs/2302.13334)
- Official repository: <https://github.com/witdsl/KRT-MLCIL>
- Fixed commit: `3f79044001edfe9ef94b729cd905a535fe8dd478`
- Official archive at that commit: `KRT-ICCV2023-updated.rar`
- Archive SHA-256:
  `2eec8351568358efca50bdaea5d06034dd28bd35d7d41c12e78cc0b1c915daf1`

The ordinary `MLCIL-ICCV2023/` directory at that commit contains the older
POD/replay baseline. The KRT implementation used for this audit is inside the
RAR added by the fixed commit. The release contains an MIT license text
credited to Alibaba-MIIL, but it does not explicitly state ownership or license
coverage for every KRT-authored addition. For that reason the benchmark
reimplements the audited operations and does not copy upstream files.

Read-only source locations are outside the benchmark repository:

```text
/Users/denghaoyuan/workspace/MyCode/baseline_sources/krt_official/
/Users/denghaoyuan/workspace/MyCode/baseline_sources/krt_release_3f79044/
```

## Retained algorithm

The Track-A method retains these KRT components:

- one learned task token and one LayerNorm/linear classifier per task;
- the shared ClassAttention block used by the official ICA model;
- freezing of old task tokens and old classifier heads;
- ASL with the official training settings (`gamma_neg=4`, `gamma_pos=0`,
  negative clipping `0.05`);
- dynamic pseudo labels for old classes;
- cosine distillation between old task-token representations;
- iCaRL-style herding with the official COCO budget of 20 exemplars per class
  and an expanding, non-fixed total budget;
- Adam, bias/norm-exempt weight-decay grouping, and a OneCycle learning-rate
  schedule;
- under AMP, OneCycle advances only when `GradScaler` actually applies the
  optimizer update; overflow-skipped updates are counted in `train.log`;
- task-0 learning rate `4e-5`, incremental learning rate `1e-4`, 20 epochs,
  embedding width 384, and eight ClassAttention heads.

No residual Adapter and no CLIP text feature is added.

## Track-A substitutions and protocol safeguards

The original TResNet-M ImageNet-21k spatial map is replaced by final projected
OpenAI CLIP ViT-B/16 patch tokens. KRT's own linear spatial projection maps the
512-dimensional CLIP tokens to its 384-dimensional ICA representation. The
CLIP visual tower remains trainable because the official KRT configuration
does not freeze its backbone.

The B5-C3 training API exposes only current-class labels. Consequently:

- old targets on a current-task sample come only from the frozen old KRT model;
- the EMOTIC dynamic-pseudo-label target density is estimated from visible
  current-task training labels and scaled to the number of old classes;
- threshold search retains the upstream `0.005` step and 100-iteration limit;
  if discrete EMOTIC predictions cannot enter the `0.1` tolerance band, the
  closest visited threshold is used and its realized count is logged instead
  of aborting the complete eight-task run;
- no validation/test target participates in threshold calibration or training;
- replay retains only the target columns visible or pseudo-labelled at capture;
- if the same retained person enters a later task, only that later task's newly
  visible columns are updated on the existing record, without adding a second
  buffer entry;
- future columns are never stored and are zero-padded only when that replay
  sample is used in a later task;
- replay samples are deduplicated by stable person-sample ID;
- both retained sample count and the bytes of stored image/target/ID payloads
  are emitted in the standard artifact.

The current Core interface does not expose the underlying dataset to a method.
The replay buffer therefore stores the post-transform float32 image tensor
rather than an upstream dataset index. This deviation is explicit in
`run_manifest.json` and must be considered when comparing Track A with a future
original-backbone Track B run. Replay-only minibatches are distributed across
each epoch; unlike the upstream concatenated dataset, an individual minibatch
does not mix current and replay samples. This avoids materializing the complete
transformed training set while keeping every retained exemplar in each epoch.

Validation checkpoint selection follows the frozen benchmark rule: current
label validation mAP, with higher score and then earlier epoch. This is a
benchmark-wide selection rule rather than a KRT-specific hyperparameter search.

## Acceptance gates

Before a KRT result can enter the table:

1. all Core/KRT and selected legacy tests pass on the server;
2. Task 0 and Task 1 smoke confirms token/head expansion and replay restore;
3. hidden-label rejection, unique replay IDs, sample/byte reporting, checkpoint
   round-trip, and seen-score alignment pass;
4. a GPU memory smoke fixes a safe batch size;
5. seed 0 validation artifacts are reviewed before the configuration is locked;
6. only then is a held-out seed-0 test run allowed.
