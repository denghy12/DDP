# CLIP-Visual Continual Classifier Baselines

## Scope

This development line adds three repository-native Track-A controls:

- Sequential Fine-Tuning;
- Learning without Forgetting (LwF);
- Elastic Weight Consolidation (EWC).

All three start from Benchmark Core v0.1 commit
`00f399f13bc7552c254c8f6e6c095a8be4f56146`. They share exactly the same
CLIP initialization, classifier, data boundary, optimizer family, validation
selection, and checkpoint format. Only the continual-learning objective
changes.

## Shared model and preprocessing

All three methods start from the exact OpenAI CLIP ViT-B/16 checkpoint used by
DDP. Only the visual encoder is retained; the CLIP text tower is discarded and
no text embedding, prompt, or prototype is used. No benchmark-specific Adapter
or hidden MLP is inserted.

The visual encoder and protocol-ordered linear sigmoid heads are trainable.
Sequential Fine-Tuning can therefore forget old classes through genuine visual
representation drift. LwF and EWC protect that same shared representation with
only their method-specific objectives. DDP keeps its own audited frozen-CLIP
training policy; Track A unifies the visual architecture, initialization, and
preprocessing rather than forcing every method to freeze the same tensors.

EMOTIC full-image preprocessing remains the DDP/Core path: random resized crop
and horizontal flip for training, resize and center crop for validation/test,
then `ToTensor`. No method-specific normalization or input path is introduced.

## Objectives

Every method minimizes binary cross entropy on only `targets_current`. Heads
are appended in protocol order; old heads remain in the model but old ground
truth is never exposed to the method.

LwF follows [Li and Hoiem, ECCV
2016](https://arxiv.org/abs/1606.09282): before a new head is added, a frozen
copy of the previous visual encoder and old classifier heads becomes the
teacher. On current-task images, the student matches the teacher's independent
old-class sigmoid probabilities at temperature `2.0`. The multi-label
distillation term is multiplied by `T^2`; its configured weight is `1.0`. No
old labels or stored old samples are used. The authors'
[reference repository](https://github.com/lizhitwo/LearningWithoutForgetting)
is an audited algorithm reference, not vendored source.

EWC follows [Kirkpatrick et al., PNAS
2017](https://doi.org/10.1073/pnas.1611835114). At each task boundary, the
implementation estimates a diagonal empirical Fisher over the trainable visual
encoder and existing heads from current-task multi-label BCE gradients. Online
Fisher values are accumulated with decay `1.0`, and the next task adds a
Fisher-weighted quadratic penalty anchored at the consolidated parameters. New
head parameters are not penalized until they have completed a task. The first
formal diagnostic used coefficient `100.0`; its weighted penalty was only
about `1e-6` of classification loss, so that run is retained as a diagnostic
rather than presented as the final EWC baseline.

## Registered optimization settings

The defaults live in `CLIPClassifierOptions`. They are deliberately kept out of
`protocol_b5c3.yaml` so the registered Core v0.1 protocol hash remains
unchanged. Every run writes the fully resolved values into both
`config_resolved.json` and `run_manifest.json`:

| Setting | Value |
|---|---:|
| Feature dimension | 512 |
| Benchmark-added Adapter | None |
| CLIP text tower | Not used |
| Visual encoder | Fully trainable |
| Maximum epochs per task | 10 |
| Early-stopping patience | 3 epochs |
| Optimizer | AdamW |
| Visual encoder learning rate | 0.00001 |
| Linear-head learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Gradient-norm clipping | 1.0 |
| CUDA AMP / TF32 | Enabled |
| LwF temperature / weight | 2.0 / 1.0 |
| EWC coefficient | Validation-selected from a locked logarithmic grid |
| EWC online decay | 1.0 |

Validation checkpoint selection uses current-label validation mAP only. Strict
improvement replaces the best state, so ties retain the earliest epoch. Test is
never used for epoch or hyperparameter selection.

## Execution

One method must process Task 0--7 sequentially. Independent task shards are
rejected because they would discard the continual state. Different methods and
seeds run on different GPUs.

Before formal jobs, the launcher runs the full unit/regression suites and a real
one-batch GPU memory smoke covering both LwF teacher memory and EWC state. With
eight GPUs, all nine registered runs are launched in one session:

```bash
SEEDS="0 1 2" GPU_LIST="0 1 2 3 4 5 6 7" \
SESSION=emotic_clip_3seed \
RUN_ID=clip_3seed_<timestamp> \
bash scripts/emotic-mlcil/launch_clip_continual_baselines_tmux.sh
```

Eight worker windows run one job per GPU. With the default nine jobs, EWC seed
2 queues behind the typically faster Fine-Tuning seed 0 job on GPU 0; all other
jobs start immediately.

Each run ID gets an isolated output root. Canonical `.pth` files remain under
`benchmarks/.../checkpoints/`. After all jobs complete, the monitor collects
every metrics/scores/logs/manifests/report bundle into the single
`download_ready/<run-id>/` directory and writes a SHA-256 download manifest.
That directory is validated to contain no `.pth` file.

The server development mirror is:

`/mnt/haoyuan/workspace/CODE_DDP-benchmark-v0.1`

## EWC validation tuning and formal rerun

EWC coefficient selection is a two-stage protocol-safe workflow. Candidate
coefficients `1e2`, `1e4`, `1e5`, `1e6`, and `1e7` run with seed 0 and
`reporting_split=val`. These artifacts explicitly have
`configuration_locked=false` and `eligible_for_main_table=false`. The selector
maximizes final validation mAP and breaks an exact tie in favor of the smaller
coefficient. It verifies that every candidate has identical source, protocol,
class-order, and validation-split provenance and that no test metric was used.

After selection, the launcher freezes the selected coefficient and
automatically starts formal test runs for seeds 0, 1, and 2. GPU slots are
derived from currently free memory. The defaults reserve 2 GiB per GPU, assume
5.2 GiB per EWC job, and permit at most two concurrent jobs per physical GPU:

```bash
SESSION=emotic_ewc_lambda_v03 \
RUN_ID=ewc_lambda_v03_<timestamp> \
GPU_LIST="0 1 2 3 4 5 6 7" \
bash scripts/emotic-mlcil/launch_ewc_lambda_tune_and_run_tmux.sh
```

With the observed server state where only GPU 0 has sufficient free memory,
the launcher creates two GPU-0 tuning workers and distributes the five
candidates between them. GPUs with less than the safety threshold are reported
and skipped. An explicit `GPU_SLOTS="0 0"` override is available, but automatic
capacity checks are preferred.

Heavy tuning state and `.pth` checkpoints remain under the run root. Only the
formal metrics, scores, logs, manifests, reports, and `tuning_selection` record
are copied into `formal/download_ready/<run-id>/`. The download directory is
validated to contain no `.pth` file.

## Registered EWC v0.3 result

The validation-only run `ewc_lambda_v03_20260801_011056` evaluated the locked
grid below with seed 0. Selection maximized Final validation mAP; no test metric
participated in this decision.

| EWC coefficient | Final validation mAP | Average validation mAP | Validation forgetting |
|---:|---:|---:|---:|
| `1e2` | 24.5881 | 30.4982 | 6.0811 |
| `1e4` | 24.8897 | 29.2073 | 6.8493 |
| `1e5` | 28.9342 | 31.8689 | 3.9307 |
| **`1e6`** | **29.7755** | **33.0151** | **3.4223** |
| `1e7` | 27.0644 | 32.3118 | 8.0463 |

The selected coefficient `1e6` is an interior grid point. It was frozen before
the held-out test was opened, then used unchanged for formal seeds 0, 1, and 2.
All three clean runs came from commit
`dea022e6b59caf445eacb5e481a511fcd7ace1f4`, completed Task 0--7 without score
reuse, and are eligible for the main table.

| Metric | Mean ± sample standard deviation |
|---|---:|
| Final mAP | **20.4155 ± 1.2887** |
| Average mAP | **24.9688 ± 0.4050** |
| Final cF1 | **16.5073 ± 2.2618** |
| Final oF1 | **42.6844 ± 4.2293** |
| Forgetting | **6.9760 ± 1.6373** |

The registered machine-readable record is
[`results/ewc_lambda1e6_seed012_v0.3.json`](results/ewc_lambda1e6_seed012_v0.3.json).
The earlier `λ=100` three-seed run remains a diagnostic and must not replace or
be averaged with this formal result.
