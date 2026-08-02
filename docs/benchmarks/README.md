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

The first registered method, legacy DDP, belongs to Track A. Track B adapters
remain a reserved extension point.

## Documentation

- [Frozen B5-C3 contract](EMOTIC_B5C3_BENCHMARK_CONTRACT.md)
- [Baseline source audit](BASELINE_SOURCE_AUDIT.md)
- [Baseline porting checklist](BASELINE_PORTING_CHECKLIST.md)
- [Implementation status](BASELINE_STATUS.md)
- [CLIP-visual Fine-Tuning/LwF/EWC design](CLIP_CONTINUAL_BASELINES.md)
- [KRT source audit and Track-A design](KRT_TRACK_A.md)
- [Registered clean DDP seed-0 result](results/ddp_seed0_core_v0.1.json)
- [Registered Fine-Tuning/LwF and EWC λ=100 diagnostic](results/clip_continual_seed012_lambda100_v0.2.json)
- [Registered validation-selected EWC λ=1e6 result](results/ewc_lambda1e6_seed012_v0.3.json)
- [Frozen KRT seed-0 validation snapshot](results/krt_seed0_validation_v0.1.json)
- [Registered KRT three-seed formal result](results/krt_seed012_formal_v0.1.json)
- [Machine-readable protocol guide](../../configs/emotic_mlcil/README.md)

## Current phase

Core v0.1 is frozen at commit
`00f399f13bc7552c254c8f6e6c095a8be4f56146`. Baseline development continues
from that exact commit on `codex/emotic-baseline-finetune-lwf-ewc`. The first
development line implements repository-native Sequential Fine-Tuning, LwF,
and EWC controls without vendoring external repositories.

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

KRT is the active integration line. Its official source is fixed at
`3f79044001edfe9ef94b729cd905a535fe8dd478`; the Track-A port retains KRT's
dynamic pseudo labels, task-token ClassAttention, old-token distillation,
per-task heads, and herding replay while replacing only the TResNet spatial
features with CLIP ViT-B/16 patch tokens. Its frozen held-out test seeds 0--2
are complete and registered at Final mAP `22.1726 ± 2.8439`. The result reports
its expanding replay sample/byte budget. A three-process single-GPU attempt
triggered the guarded OOM fallback; the affected seeds restarted cleanly and
the future memory smoke/capacity estimate now includes Adam state and replay
batch 64 without changing the registered algorithm or result.

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

After a successful merge, download only
`results_to_sync/<shard_run_id>/`. It contains the resolved configuration,
manifest, metrics, canonical task scores, report, console logs, shard metadata,
and a SHA-256 file manifest. It contains no `.pth` files. Large checkpoint
copies remain in their server-side checkpoint directories.

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
