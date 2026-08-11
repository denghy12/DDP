# EMOTIC MLCIL Benchmark

## Research goal

This benchmark provides one protocol-driven core for comparing multi-label
class-incremental learning (MLCIL) methods on EMOTIC. It keeps the existing DDP
implementation in place and adds adapters around the existing dataset, model,
checkpoint, and metric paths. External baseline implementations are explicitly
out of scope for Core v0.1.

The benchmark is frozen from:

- base branch: `codex/emotic-ddp-task-adapter-bank`
- base commit: `f9459d0769f4ef3ee93e51db31df6ec509a933ad`
- development branch: `codex/emotic-mlcil-benchmark-core`

The authoritative local repository is:

`/Users/denghaoyuan/workspace/MyCode/CODE_DDP-benchmark`

The server development mirror is:

`/mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1`

The old `/Users/denghaoyuan/workspace/MyCode/CODE_DDP` tree is not part of this
work.

## Comparison tracks

- **Track A — unified visual initialization:** every method starts from the
  same OpenAI CLIP ViT-B/16 visual architecture, checkpoint, and EMOTIC
  preprocessing. Backbone trainability follows the audited method: DDP freezes
  it, while Fine-Tuning/LwF/EWC update it. Text features are used only by
  methods whose audited algorithm requires them.
- **Track B — original backbone:** a method keeps the backbone and preprocessing
  required by its audited upstream implementation. Backbone differences must be
  reported and Track A and Track B numbers must not be mixed in one ranking.

The first registered method, legacy DDP, belongs to Track A. Native static
EMOTIC conversions such as EMOT-Net-FT and CocoER-FT belong to Track B.

## Documentation

- [Frozen B5-C3 contract](EMOTIC_B5C3_BENCHMARK_CONTRACT.md)
- [B10-C4 12-baseline sweep contract and launcher](EMOTIC_B10C4_12BASELINE_SWEEP.md)
- [B4-C2 12-baseline sweep contract and OOM-safe launcher](EMOTIC_B4C2_12BASELINE_SWEEP.md)
- [Baseline source audit](BASELINE_SOURCE_AUDIT.md)
- [Baseline porting checklist](BASELINE_PORTING_CHECKLIST.md)
- [Implementation status](BASELINE_STATUS.md)
- [CLIP-visual Fine-Tuning/LwF/EWC design](CLIP_CONTINUAL_BASELINES.md)
- [KRT source audit and Track-A design](KRT_TRACK_A.md)
- [CSC source audit and Track-A porting contract](CSC_TRACK_A.md)
- [MULTI-LANE source audit and Track-A porting contract](MULTI_LANE_TRACK_A.md)
- [L3A source audit and Track-A porting contract](L3A_TRACK_A.md)
- [Original-DDP-Tau2 source audit and Track-A contract](ORIGINAL_DDP_TRACK_A.md)
- [AGCN source audit and Track-A porting contract](AGCN_TRACK_A.md)
- [Shared replay memory contract](REPLAY_MEMORY_CONTRACT.md)
- [ER control and PRS Track-A porting contract](ER_PRS_TRACK_A.md)
- [DER++ source audit and Track-A porting contract](DERPP_TRACK_A.md)
- [EMOT-Net native-backbone sequential FT Track-B contract](EMOT_NET_FT_TRACK_B.md)
- [CocoER native-backbone sequential FT Track-B contract](COCOER_FT_TRACK_B.md)
- [Universal checkpoint-free download standard](DOWNLOAD_STANDARD.md)
- [Registered clean DDP seed-0 result](results/ddp_seed0_core_v0.1.json)
- [Registered Fine-Tuning/LwF and EWC λ=100 diagnostic](results/clip_continual_seed012_lambda100_v0.2.json)
- [Registered validation-selected EWC λ=1e6 result](results/ewc_lambda1e6_seed012_v0.3.json)
- [Frozen KRT seed-0 validation snapshot](results/krt_seed0_validation_v0.1.json)
- [Registered KRT three-seed formal result](results/krt_seed012_formal_v0.1.json)
- [CSC independent-port/upstream-oracle equivalence audit](results/csc_upstream_equivalence_v0.1.json)
- [Frozen CSC seed-0 validation snapshot](results/csc_seed0_validation_v0.1.json)
- [Registered CSC three-seed formal result](results/csc_seed012_formal_v0.1.json)
- [Frozen MULTI-LANE seed-0 validation snapshot](results/multi_lane_seed0_validation_v0.1.json)
- [Registered MULTI-LANE three-seed formal result](results/multi_lane_seed012_formal_v0.1.json)
- [Frozen L3A seed-0 validation snapshot](results/l3a_seed0_validation_v0.1.json)
- [Registered L3A three-seed formal result](results/l3a_seed012_formal_v0.1.json)
- [Frozen Original-DDP-Tau2 seed-0 validation snapshot](results/original_ddp_tau2_seed0_validation_v0.1.json)
- [Registered Original-DDP-Tau2 three-seed formal result](results/original_ddp_tau2_seed012_formal_v0.1.json)
- [Frozen AGCN seed-0 validation snapshot](results/agcn_seed0_validation_v0.1.json)
- [Registered AGCN three-seed formal result](results/agcn_seed012_formal_v0.1.json)
- [Frozen ER/PRS seed-0 validation snapshot](results/er_prs_seed0_validation_v0.1.json)
- [Registered ER/PRS three-seed formal result](results/er_prs_seed012_formal_v0.1.json)
- [Frozen DER++ seed-0 validation snapshot](results/derpp_seed0_validation_v0.1.json)
- [Registered DER++ three-seed formal result](results/derpp_seed012_formal_v0.1.json)
- [EMOTIC B5-C3 Track-A baseline summary table](results/TRACK_A_BASELINE_SUMMARY_V0.1.md)
- [Paper-ready Track-A summary LaTeX](results/track_a_baseline_summary_v0.1.tex)
- [Machine-readable Track-A summary data](results/track_a_baseline_summary_v0.1.json)
- [EMOTIC B10-C4 Track-A baseline summary table](results/B10C4_TRACK_A_BASELINE_SUMMARY_V0.1.md)
- [Registered B10-C4 12-baseline three-seed result](results/b10c4_12baseline_seed012_formal_v0.1.json)
- [Registered B4-C2 12-baseline three-seed result](results/b4c2_12baseline_seed012_formal_v0.1.json)
- [Machine-readable protocol guide](../../configs/emotic_mlcil/README.md)

## Current phase

Core v0.1 is frozen at commit
`00f399f13bc7552c254c8f6e6c095a8be4f56146`. The first baseline development
line branched from that commit as `codex/emotic-baseline-finetune-lwf-ewc` and
implemented repository-native Sequential Fine-Tuning, LwF, and EWC controls
without vendoring external repositories. KRT then received its own frozen
branch. CSC Track A is complete on `codex/emotic-baseline-csc`: its seed-0
validation configuration was frozen before the locked held-out seeds 0--2
were run, and the three-seed result is now registered.

The separate B10-C4 Track-A sweep is also complete. It evaluated the same 12
registered baselines over the alphabetic `10/4/4/4/4` task split, using seeds
0--2 and unchanged frozen B5-C3 hyperparameters. All 36 held-out jobs used
clean commit `c759b3c`, were configuration-locked, and are eligible for the
main table. MULTI-LANE ranks first in Final mAP (`33.6902 ± 0.1599`) and
Average mAP (`37.4430 ± 0.1702`); Original-DDP-Tau2 ranks second and has the
highest fixed-threshold oF1 (`51.0957 ± 1.6497`). The checkpoint-free archive
SHA-256 is `7d905ece7a7c8838c33edd80b15507956c1d7c1dd1998e694153cd086eb9e205`.
Two initial OOMs were isolated scheduling failures for ER seed 0 and PRS seed
0 while sharing a GPU with DER++ seed 1; both retries used unchanged code and
configuration. The registered result is frozen without B10-C4 test tuning.

The separate B4-C2 Track-A sweep is complete and frozen as well. It evaluated
the same 12 registered baselines over the alphabetic `4/2×11` split with seeds
0--2 and unchanged frozen B5-C3 hyperparameters. All 36 configuration-locked
held-out jobs used clean commit `a6d6937`, completed under the memory-guarded
eight-GPU dual-slot scheduler in about `3.039` hours, and are eligible for the
main table. MULTI-LANE ranks first in Final mAP (`29.6138 ± 0.3312`) and
Average mAP (`36.9964 ± 0.2495`), followed by Original-DDP-Tau2 and L3A. No
job OOMed or required a retry. The checkpoint-free archive SHA-256 is
`76d1b799e7d8e02b3ed02f4f922de530713fa2f41695902ac4e78645bcb60d4a`.
All outer and nested manifest hashes passed independent verification. The
result must not be tuned from B4-C2 held-out behavior.

The baseline branch removes the earlier benchmark-added residual Adapter.
Fine-Tuning and LwF have completed three-seed execution. EWC coefficient
selection used only validation data, selected `λ=1e6`, and its locked
three-seed formal test run is complete. The initial `λ=100` EWC run remains
diagnostic evidence only. Details and the tmux entry point are recorded in
`CLIP_CONTINUAL_BASELINES.md`.

The registered repository-local DDP result is a modified DDP-family variant,
not an evaluation of the unmodified original DDP method. Its seed-0 result is
retained as project evidence, while additional seeds are intentionally deferred
until the local modifications and upstream baseline are separately audited.

EMOT-Net-FT development is active on
`codex/emotic-baseline-emot-net-ft`. At the user's direction it does not use
the unified CLIP tower. It is therefore Track B, with the official
Places-context and official-release AlexNet-body architecture, full-scene plus
person-crop input, source weighted sigmoid-MSE, and current-label-only
expanding-head sequential fine-tuning. The Git repository does not bundle its
Torch7 assets; seed-0 validation remains gated on hash-verifying and converting
the separately distributed official Dropbox ZIP. CLIP and random fallback are
explicitly rejected.

CocoER-FT development is active on `codex/emotic-baseline-cocoer-ft`. It is a
second Track-B static-to-incremental lower bound and deliberately retains the
official method's three ImageNet ResNet-50 towers plus native frozen OpenAI
CLIP RN50 vocabulary-informed image branch; it does not use the Track-A CLIP
ViT-B/16. The conversion adds only current-label expanding heads and protocol
visibility. Full-26-class CocoER GWT/VI weights are rejected as future-label
leakage. The initial server suite passed 185 Core/baseline tests (4 skips) and
17 legacy regressions at `881b0e6`. Because EMOTIC publishes no head boxes, the
approved sample-preserving conversion now uses strict `buffalo_l` matches first
and a componentwise-median relative box calibrated only on native-resolved
training samples for the remainder. Labels and validation/test statistics are
forbidden. The cache records actual ONNX providers, SCRFD/FaceAnalysis
equivalence, native/fallback IDs and counts, geometry, and hashes. Seed-0
validation remains gated on full CUDA cache generation, joint asset audit, and
CUDA memory smoke. See
`COCOER_FT_TRACK_B.md`.

Original-DDP-Tau2 is complete on `codex/emotic-baseline-original-ddp`. It
retains the collected original DDP
model, random text/visual prompting, loss, optimizer, and cross-task scheduler
without an Adapter or replay. For direct comparison with the modified DDP, its
registered PCD is the user-requested `T=1→2, γ=0.7`; the collected source's
`T=1→7, γ=0.2` is preserved in the audit as a deliberate difference. Source
identity/operator gates passed with zero numerical error. Seed-0 validation
reached Final mAP `39.3939`, Average mAP `46.8123`, and Forgetting `0.7607`.
The `T=1→2, γ=0.7` configuration, source optimizer/schedule, batch `8/1`, and
zero-worker loader were frozen before held-out access. Locked test seeds 0--2
then completed concurrently as isolated processes on GPU 0 from clean commit
`e9d3945`. The registered formal result is Final mAP
`29.9795 ± 0.2237`, Average mAP `37.2071 ± 0.4779`, and Forgetting
`4.8062 ± 0.1337` (mean ± sample standard deviation). Canonical final-task
scores contain 5,368 `emotic:test:` IDs and no validation IDs, so the raw
source's default `val+test` behavior was not used by the benchmark run.

AGCN Track-A is complete on `codex/emotic-baseline-agcn`, branched
from the frozen Original-DDP-Tau2 result. Official source is fixed at
`3afe2ecbbef0051c6e841a97c369885011a683f0` under Apache-2.0 and remains outside
the benchmark tree as an executable oracle. The independent port replaces only
ResNet-101 with a trainable CLIP ViT-B/16 visual tower and retains GloVe label
nodes, the two-layer GCN, online augmented correlation matrix, sigmoid old-class
distillation, graph relationship MSE, and zero replay. It adds no Adapter or
CLIP text feature. Server preflight passed with a largest fixed-source operator
error of `4.47e-8`; the final-task teacher/two-Adam smoke reserved `2640 MiB`.
Seed-0 validation reached Final mAP `25.3571`, Average mAP `28.9310`, and
Forgetting `1.1844` with 3,701 optimizer steps, zero skipped updates, and no
NaN/OOM. The one-epoch schedule, both learning rates, all three loss weights,
GloVe mapping, graph construction constants, and global threshold were frozen
before held-out access. Locked test seeds 0--2 then completed concurrently as
three isolated processes on GPU 0 from clean commit `d763c66`, with no OOM,
NaN, traceback, or skipped update. The registered formal result is Final mAP
`17.9090 ± 0.3755`, Average mAP `21.7826 ± 0.1163`, and Forgetting
`4.0622 ± 0.9285` (mean ± sample standard deviation). The checkpoint-free
archive and all 57 manifest entries passed independent SHA-256 verification.
AGCN Track A is now frozen and must not be rerun or tuned from held-out test.

ER/PRS replay-baseline development completed on `codex/emotic-baseline-er-prs`,
branched from the frozen AGCN result. A machine-readable shared memory contract
now fixes 20 unique person-samples per seen class (task capacities
`100/160/220/280/340/400/460/520`), 1:1 replay/current exposure, masked
capture-time-visible labels, stable-ID deduplication, and task-end updates. ER
is the conventional reservoir control. PRS is an independent Track-A port of
the official ECCV 2020 source fixed at `136cee1`, retaining allocation power
`q=-0.03` while replacing only ResNet-101 with the common trainable CLIP visual
tower. Both seed-0 validation runs passed from clean commit `9a716a5`. ER
reached Final mAP `26.6088` and PRS reached `28.4171`; both finished with
exactly 520 stored samples and no NaN/OOM/training traceback. PRS's
fixed-source oracle passed with exact retained IDs and maximum target-
proportion error `1.58e-8`. The 20-per-seen-class capacity, 1:1 replay,
`q=-0.03`, optimizer, early stopping, and fixed F1 threshold 0.5 are now
frozen. Locked test seeds 0--2 completed from clean commit `84c79eb` on GPU 0
in two automatic three-process waves: ER first, then PRS. No worker required an
OOM fallback or rerun. ER is registered at Final mAP `20.3230 ± 1.4593`,
Average mAP `25.9259 ± 1.4637`, and Forgetting `8.6986 ± 0.6386`; PRS is
registered at `20.5603 ± 0.4442`, `26.3254 ± 1.5723`, and
`8.7479 ± 1.6186`. PRS therefore has only a small paired Final-mAP gain
(`+0.2373 ± 1.2554`) under the shared budget. Its fixed-0.5 Final oF1 is
`14.0267` points lower than ER, which is retained as a calibration limitation
without post-test threshold tuning. Every run ended at exactly 520 replay
samples, and the six-bundle checkpoint-free archive passed all 108 internal
SHA-256 checks. ER and PRS Track A are now frozen.

DER++ development is active on `codex/emotic-baseline-derpp`, branched from
the frozen ER/PRS commit `895f895`. The official Mammoth `neurips2020` tag is
fixed at dereferenced commit `cb9a36d` under MIT, with the complete source
archive retained outside this repository. The independent Track-A port keeps
the defining online pre-update trajectory logits, reservoir sampling, two
independent replay draws, and paper-stable `alpha=beta=0.5`. Softmax CE is
mapped to visible-mask sigmoid BCE and logit MSE is limited to columns that
existed at capture time. It uses the same `20 × seen classes` sample scale as
ER/PRS but separately charges stored logits and their mask in byte accounting.
Task-end insertion is not used because it would change the method into
boundary-logit replay. Seed-0 validation completed from clean commit `f488788`
after 139 Core/baseline tests (3 skips), 17 legacy regressions, and zero-error
fixed-source objective checks. It reached Final mAP `33.4671`, Average mAP
`40.1956`, and Forgetting `5.9787`, ending at exactly 520 samples and
`298.7612 MiB` after charging stored logits and masks. Two guarded AMP overflow
skips occurred in 8,524 optimizer attempts; no NaN, OOM, or traceback was
observed. The checkpoint-free archive and all 19 manifest entries passed
SHA-256 verification. `alpha=beta=0.5`, 20 samples per seen class, both 1:1
replay draws, online pre-update logits, optimizer, learning rates, early
stopping, and threshold 0.5 are frozen before held-out access. The formal
runner requires `configuration_locked=true` and assigns seeds 0--2 to three
distinct GPUs because the measured single-process peak is `8434.2 MiB`.
Locked held-out seeds 0--2 then completed concurrently on GPUs 0/1/2 from
clean commit `dfb3957`, without OOM fallback or rerun. DER++ is registered at
Final mAP `23.0844 ± 1.2413`, Average mAP `30.4140 ± 0.4120`, and Forgetting
`9.1229 ± 1.3788`. Its paired Final-mAP gains are `+2.7614 ± 2.4241` over ER
and `+2.5241 ± 1.6853` over PRS, with a positive difference in every seed.
Those gains do not extend to all metrics: forgetting is `0.4243` worse than ER
and `0.3750` worse than PRS, while fixed-threshold cF1 does not improve. All
three runs ended at 520 samples, and six guarded AMP overflows occurred in
24,325 optimizer attempts without NaN, OOM, or traceback. The checkpoint-free
archive passed its outer checksum and all 57 internal SHA-256 checks. DER++
Track A is frozen without held-out threshold or hyperparameter tuning.

KRT Track A is frozen on `codex/emotic-baseline-krt` at commit `029eda4`. Its
official source is fixed at
`3f79044001edfe9ef94b729cd905a535fe8dd478`; the Track-A port retains KRT's
dynamic pseudo labels, task-token ClassAttention, old-token distillation,
per-task heads, and herding replay while replacing only the TResNet spatial
features with CLIP ViT-B/16 patch tokens. Its frozen held-out test seeds 0--2
are complete and registered at Final mAP `22.1726 ± 2.8439`. The result reports
its expanding replay sample/byte budget. A three-process single-GPU attempt
triggered the guarded OOM fallback; the affected seeds restarted cleanly and
the future memory smoke/capacity estimate now includes Adam state and replay
batch 64 without changing the registered algorithm or result.

CSC Track A is complete. Its official source is fixed at
`0bab38a00d6e0555f2df855ae2fe8db1fea68b12`, and the exact source archive has
SHA-256 `588a098a1c7f3d813dee7df777c283fd27768a08a125d4e60b11b2d6ebdb7faa`.
No upstream license file was found, so the source remains outside this
repository as a read-only reference. The Track-A contract keeps CSC's
dynamically expanding CI-GCN, current-label loss, old-model sigmoid
distillation, and max-entropy calibration while replacing only TResNet spatial
features with trainable CLIP patch tokens. It explicitly adds no Adapter, text
features, or replay.

The locked CSC test result is Final mAP `20.5614 ± 0.4785`, Average mAP
`30.3967 ± 0.2411`, and Forgetting `9.0112 ± 0.6309` (mean ± sample standard
deviation, three seeds). Its fixed-0.5 Final cF1 is only
`2.3391 ± 0.7219`: held-out scores reproduce the validation-time calibration
failure and newest-class bias. This limitation is registered without any
post-test threshold or hyperparameter change.

MULTI-LANE Track A is complete on
`codex/emotic-baseline-multi-lane`, branched from the registered CSC commit
`e13cac7`. Its official source is fixed at `5ee982c` with archive SHA-256
`dfe84ea3...22d49`; the fixed extraction is byte-identical to the previously
collected snapshot. The implementation contract freezes the shared CLIP visual
tower and trains only MULTI-LANE's task selectors, prompt slices, and
classifier under current-label visibility. Its locked held-out seeds 0--2 were
trained from clean commit `3fe1121` and are registered at Final mAP
`31.2995 ± 0.1410`, Average mAP `37.9986 ± 0.4825`, and Forgetting
`4.7885 ± 0.0199` (mean ± sample standard deviation). Every seed completed
13,950 optimizer updates with zero AMP skips, NaN, or OOM. Aggregator fix
`7206f58` only corrected the postprocessing task-metric path after training;
the retained first traceback did not trigger retraining or prediction changes.

The independent CSC CI-GCN operators have also been checked against the exact
external upstream implementation with mapped inputs and parameters. Float32
combined-logit error is `5.96e-8`; the material discrepancy is confined to the
published task-expansion lifecycle, which reinitializes old biases and leaves
new parameters outside the old optimizer. The comparison is a synthetic
implementation audit, not an EMOTIC performance result.

L3A Track A is complete on `codex/emotic-baseline-l3a`, branched from the
frozen MULTI-LANE commit `20e645d`. Its official source is fixed at `1067bbd`;
because that source has no license declaration, it remains outside this
repository and only an independent implementation enters `methods/l3a/`.
The port retains L3A's one-epoch Task-0 CLIP+ASL stage, frozen random ReLU
features, cumulative weighted analytic updates, and threshold-0.7 old-class
pseudo labels. It adds no Adapter, text feature, or replay.

The configuration was frozen on validation before held-out access. Locked test
seeds 0--2 then ran concurrently on distinct GPUs from clean commit `d88620f`.
The registered result is Final mAP `28.8296 ± 0.6470`, Average mAP
`33.7746 ± 1.3091`, and Forgetting `4.8687 ± 0.3780` (mean ± sample standard
deviation). Every seed attempted 84 Task-0 updates: 76 were applied and 8 were
guarded AMP overflow skips; there was no NaN, OOM, or training traceback. The
fixed-source oracle's largest error remained `2.22e-16`. The result snapshot
also records all pseudo-label counts, per-task curves, and the complete
26-class forgetting audit.

## Standard run and output

Validation-only DDP evaluation uses:

```bash
python -m benchmarks.emotic_mlcil.runner \
  --protocol configs/emotic_mlcil/protocol_b5c3.yaml \
  --method ddp \
  --data-root ./datasets/EMOTIC \
  --checkpoint-dir ./output/emotic_b5c3_ddp_semantic_tau2/checkpoints \
  --clip-model-path ./pretrained/clip/ViT-B-16.pt \
  --output-root ./output \
  --reporting-split val
```

After all checkpoint and configuration decisions are frozen, the held-out test
run changes the final option to
`--reporting-split test --configuration-locked`. Without that explicit lock,
the runner refuses to touch test. Validation is the only selection split.

For the frozen eight-task DDP checkpoints on an eight-GPU server, use the
parallel tmux launcher:

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

EVAL_BATCH_SIZE=4 WORKERS=0 \
  bash scripts/emotic-mlcil/launch_ddp_b5c3_8gpu_tmux.sh
```

The launcher first runs the fast Core test suite, then maps Task 0–7 to physical
GPU 0–7. Each process sees exactly one GPU, writes to an isolated timestamped
shard directory, and evaluates the held-out test split with the frozen global
threshold. A ninth tmux window waits for all shards, rejects missing or
inconsistent outputs, and publishes the canonical artifacts. `WORKERS=0`
avoids the server's file-descriptor failure while the eight task processes
still provide data-loading concurrency.

If post-processing fails after a task has already written `scores.pt`,
relaunching with the same `SHARD_RUN_ID` validates and reuses those exact
scores instead of repeating image inference. Shard metadata and the final
manifest record every reused task ID.

Every shard records the Git commit, dirty-tree flag, and a SHA-256 fingerprint
of the complete Core source/config/test/script tree. All eight fingerprints
must match. A run from an uncommitted tree remains reproducible by fingerprint
but is marked `eligible_for_main_table: false` until the Core is committed and
the formal run is repeated.

`trainable_parameters` means the number of unique parameters actually
registered with the audited optimizer, not every tensor that happens to have
`requires_grad=True`. For legacy DDP this is exactly `ctx_pos`, `ctx_neg`, and
`visual_prompts`, matching `DDP.build_optimizer_scheduler`.

Useful monitoring commands are:

```bash
tmux attach -t emotic_benchmark_ddp8
tmux list-windows -t emotic_benchmark_ddp8
nvidia-smi
```

Artifacts use this stable layout:

```text
output/benchmarks/<protocol_id>/<track>/<method>/seed<seed>/
├── config_resolved.json
├── run_manifest.json
├── checkpoints/
├── scores/task<N>_scores.pt
├── metrics/task_metrics.json
├── metrics/summary.json
├── report.html
├── results_to_sync/<shard_run_id>/
└── train.log
```

`results_to_sync/<shard_run_id>/` is the canonical checkpoint-free method
bundle. The universal packager then combines all expected bundles and available
launcher/preflight logs into `download_packages/<run-id>.tar.gz` plus an
adjacent `.sha256` file. Download only those two files. The package contains
resolved configuration, manifests, metrics, canonical `.pt` task scores,
reports, logs, and hashes; every nested `.pth` is forbidden. Large checkpoint
copies remain in their server-side checkpoint directories. The complete
mandatory policy is [Benchmark Download Package Standard](DOWNLOAD_STANDARD.md).

The summary main-table schema is:

| Method | Type | Replay Memory | Backbone | Final mAP ↑ | Final cF1 ↑ | Final oF1 ↑ | Avg. mAP ↑ | Forgetting ↓ | Parameter Growth |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|

## Environment smoke-test record

The server development image was verified before Core v0.1:

- all 17 pre-existing unit tests passed;
- built-in prompt-free CLIP versus independent OpenAI CLIP:
  image max absolute error `4.2915e-6`, text max absolute error `0`;
- a zero-initialized internal Adapter preserved DDP logits exactly:
  max absolute error `0`;
- CLIP weights, DDP checkpoints, EMOTIC data, and GPU were available.

Core v0.1 validation completed on 2026-07-30 and the freeze audit was
repeated locally on 2026-07-31:

- all 36 final Core tests passed locally; the real-checkpoint integration test
  was skipped
  during ordinary discovery because its environment variables are intentionally
  opt-in;
- all 17 selected pre-existing regression tests passed locally and on the
  server;
- the opt-in task-7 integration test evaluated all 5,368 pure-test
  person-samples twice on `cuda:6` and passed;
- ordered sample IDs and targets were identical;
- score max absolute error was exactly `0.0`;
- per-class AP was identical;
- legacy and benchmark mAP were both
  `30.815213805446934`.

The eight-task standard-artifact smoke also completed on 2026-07-30. It
validated all Task 0--7 shard scores and published the canonical manifest,
metrics, report, logs, and SHA-256 file manifest. Its main summary was:

- Final mAP: `30.805144470640787`;
- Average mAP: `37.737386721952674`;
- Final cF1: `31.492723345851537`;
- Final oF1: `49.31224209078405`;
- Forgetting: `4.899014227592711`.

The formal eight-task DDP run completed on 2026-07-31 from the clean frozen
commit `00f399f13bc7552c254c8f6e6c095a8be4f56146`. Its manifest records
`git_dirty: false`, no reused predictions, corrected trainable parameter count
`825344`, and `eligible_for_main_table: true`. It reproduced the same summary
values listed above and is the registered seed-0 DDP result. The earlier
uncommitted/reused-prediction run remains smoke evidence only.

Legacy reference values (`Final mAP ≈ 30.82`, `Average mAP ≈ 37.71`,
`Final cF1 ≈ 32.4708`, `Final oF1 ≈ 46.6963`) are anomaly-detection hints only.
They are never returned or embedded by the evaluator.

The historical DDP main flow combined `val+test`, and historical score files do
not contain sample IDs. Core v0.1 therefore does not claim file-to-file sample
alignment with those artifacts. Strict equivalence is checked by running the
legacy model path and benchmark wrapper on the same deterministic dataloader.

## Server verification

After PyCharm synchronizes the local tree, run:

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

find benchmarks/emotic_mlcil tests/emotic_mlcil -name '*.py' -print0 \
  | xargs -0 /opt/conda/envs/ddp/bin/python -m py_compile

/opt/conda/envs/ddp/bin/python -m unittest discover \
  -s tests/emotic_mlcil -t .

/opt/conda/envs/ddp/bin/python -m unittest \
  tests.test_emotic_task_adapter_bank \
  tests.test_ddp_internal_adapter \
  tests.test_ddp_prompt_free_auxiliary
```

The strict real-checkpoint task-7 comparison is:

```bash
cd /mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1

EMOTIC_DDP_CHECKPOINT="$PWD/output/emotic_b5c3_ddp_semantic_tau2/checkpoints/task7.pth" \
EMOTIC_DATA_ROOT="/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC" \
EMOTIC_CLIP_MODEL_PATH="$PWD/pretrained/clip/ViT-B-16.pt" \
EMOTIC_DDP_TASK_ID=7 \
EMOTIC_DDP_SPLIT=test \
EMOTIC_DDP_BATCH_SIZE=1 \
EMOTIC_DDP_WORKERS=0 \
EMOTIC_DDP_DEVICE=cuda \
/opt/conda/envs/ddp/bin/python -m unittest \
  tests.emotic_mlcil.test_ddp_wrapper_equivalence.RealDDPWrapperIntegrationTest
```

This command fails unless ordered sample IDs and targets are identical, score
max absolute error is strictly below `1e-7`, and per-class AP/mAP are identical.
When `EMOTIC_DDP_DEVICE=cuda` and multiple GPUs are visible, the integration
test selects the GPU with the most free memory. An explicit value such as
`cuda:7` overrides automatic selection.
