# DER++ Track-A Porting Contract

This document fixes the independent EMOTIC B5-C3 Track-A mapping of Dark
Experience Replay++ (DER++). It does not claim that the NeurIPS source natively
supports multi-label class-incremental learning.

## Immutable source

- Paper: *Dark Experience for General Continual Learning: a Strong, Simple
  Baseline*, NeurIPS 2020.
- Official repository: `https://github.com/aimagelab/mammoth`.
- Source tag: annotated tag `neurips2020`, object `1f3978bd58a7f873675f5245d3e6a1ad46bee28a`.
- Dereferenced commit: `cb9a36d788d6ad051c9eee0da358b25421d909f5`.
- Local archive: `baseline_sources/derpp_neurips2020_cb9a36d.tar.gz`, kept
  outside this repository.
- Archive SHA-256:
  `d7cdffefdb7d77939a1055984cb586ad83af0220439cd76e4a106ea201c1695b`.
- Critical hashes: `models/derpp.py`
  `d38736e8d8c888300a8e5ddcac7cab1ac12e2eb1a0aa4c28f2387bdd79fc973a`;
  `utils/buffer.py`
  `3f4c9b416e22241bd6417d1cd2d6ec0625d7309c1d9e308757635debe545d5d3`.
- License: MIT at the fixed source.

The fixed implementation computes the current supervised loss, samples the
buffer once for logit MSE, samples it independently a second time for replay
supervision, and then inserts the current unaugmented images, labels, and
pre-update logits through reservoir sampling. The paper notes that
`alpha=beta=0.5` is stable; those values are the registered validation starting
configuration, not a test-selected result.

## Track-A mapping

The visual model is the same trainable OpenAI CLIP ViT-B/16 visual encoder and
protocol-ordered growing linear classifier used by the controlled ER/PRS
baselines. DER++ adds no Adapter and uses no CLIP text feature.

For a current batch, one dark replay draw, and an independent labelled replay
draw, the registered objective is:

`BCE_current + 0.5 * masked_MSE_dark + 0.5 * masked_BCE_replay`.

- Current BCE sees only the current task's label columns.
- Replay BCE sees only binary columns that were visible when a retained
  observation entered the stream.
- Logit MSE sees every model column that existed at capture time; unavailable
  future columns are zero and physically excluded by `logit_mask`.
- Two replay draws are sampled independently without replacement within each
  draw. They may overlap with one another, as in the fixed source.
- Each draw is capped at the current mini-batch size, giving a 1:1 draw/current
  ratio when memory is large enough.

The buffer is updated online after every optimizer attempt using logits
computed before that attempt. This is intentionally different from ER/PRS's
task-end insertion: changing DER++ to task-end logits would remove its defining
optimization-trajectory signal. Since the benchmark selects an epoch using
current-label validation mAP, the buffer and its RNG state are snapshotted and
restored together with the selected model.

Stable EMOTIC sample IDs remain unique. If one person-sample reappears, its
post-transform image and logits are replaced together by the latest coherent
capture over all then-seen model columns; this avoids pairing first-capture
logits with a different randomly augmented image. Target visibility is merged
independently and remains limited to the corresponding current-label
observations. Logit masks may only grow, and conflicting already-visible truth
is rejected.

## Memory and fairness

The machine-readable authority is
`configs/emotic_mlcil/replay_derpp_20c_v0.1.yaml`. It uses the same primary
capacity schedule as ER/PRS: 20 unique person-samples per seen class, from 100
samples after Task 0 to 520 after Task 7. The source paper reports 200, 500, and
5120-sample studies; none silently replaces the registered Track-A budget.

DER++ byte accounting additionally charges the float32 logit vector and boolean
logit mask. Artifacts must therefore report both sample count and actual bytes;
sample-count equality with ER does not imply byte equality.

## Required validation

Before seed-0 validation, the server must:

1. verify the fixed source hashes and MIT license;
2. confirm two independent `get_data` calls, logit MSE, label replay, and
   `(image, label, logit)` insertion in the immutable source;
3. obtain zero numerical error for dense logit MSE and weighted objective
   composition;
4. pass hidden-target and hidden-logit tests, online lifecycle tests, stable-ID
   merge tests, checkpoint/RNG restoration, and all legacy regressions; and
5. pass the current batch plus two replay batches CUDA memory smoke.

Seed-0 reports validation only. Held-out test remains blocked until the
validation result is reviewed, the configuration is frozen, and
`configuration_locked=true` is enforced by a formal runner.

## Frozen validation result

Seed 0 completed validation from clean commit
`f4887886752e4c6a17ba8e7bec1f9adf0ff64934`. The server preflight passed 139
Core/baseline tests with three intentional skips, 17 selected legacy
regressions, and the immutable-source oracle with zero dense-logit-MSE and
weighted-objective error. The run reported Final mAP `33.4671`, Average mAP
`40.1956`, Forgetting `5.9787`, Final cF1 `25.5183`, and Final oF1 `55.2067`.
It ended at 520 unique samples and 313,273,870 bytes (`298.7612 MiB`), including
the float32 logit payload and mask.

The eight task mAP values were
`46.9458/47.2586/38.6057/40.1672/39.6223/38.6264/36.8715/33.4671`.
Two guarded AMP overflow skips occurred in 8,524 optimizer attempts, one each
in Tasks 3 and 4. All logged values were finite, and no OOM or training
traceback occurred. The checkpoint-free archive contains all eight canonical
score tensors, excludes every `.pth`, and passed its outer and 19 internal
SHA-256 checks. The complete machine-readable evidence is
`results/derpp_seed0_validation_v0.1.json`.

## Frozen formal configuration

Validation review freezes `alpha=0.5`, `beta=0.5`, 20 samples per seen class,
two independent 1:1 replay draws, online insertion of pre-update logits, the
AdamW optimizer, backbone/head learning rates `1e-5/1e-4`, weight decay
`1e-4`, gradient clipping at 1.0, ten-epoch maximum, patience 3, train/eval
batch sizes `32/64`, two workers, AMP/TF32, and the global F1 threshold 0.5.
Held-out metrics may not change any of these values.

Formal execution requires confirmation `DERPP_20C_TRACK_A_V0_1`, a clean
frozen Git commit, `reporting_split=test`, and `configuration_locked=true`.
The measured current-32 plus two replay-32 peak is `8434.2 MiB`; therefore the
three formal seeds run concurrently as isolated processes on three distinct
physical GPUs. The formal validator rejects missing seeds, unlocked/test-
ineligible manifests, configuration drift, incomplete task scores, invalid
online-buffer statistics, or non-finite training logs before packaging.

On the server, `scripts/emotic-mlcil/prepare_derpp_source.sh` downloads or
accepts the adjacent fixed archive, with an automatic fixed-tag GitHub SSH
clone fallback when `codeload.github.com` is unavailable. It verifies the
resolved commit and three critical files, and prepares only
`/mnt/haoyuan/workspace/baseline_sources/derpp_official`. It never copies the
external repository into `benchmarks/emotic_mlcil/methods/`.
Set `DERPP_SOURCE_TRANSPORT=ssh` to skip the HTTPS attempts immediately.
