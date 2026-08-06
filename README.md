# Paper 

PyTorch code for: DDP: Dual-Decoupled Prompting for Multi-Label Class-Incremental Learning

# Project organization

EMOTIC experiment launchers are grouped by branch under `scripts/`:

- `scripts/emotic/`: baseline EMOTIC DDP and upper-bound runs.
- `scripts/emotic-prototype-adapter/`: external CLIP Prototype Adapter and score fusion.
- `scripts/emotic-ddp-internal-adapter/`: DDP-internal Feature Adapter / CLS gate / final ablation.

See `scripts/README.md` and `docs/emotic_experiment_code_map.md` for the full
map. Result directories keep their original `output/emotic_*` paths for
backward compatibility; run `python tools/organize_emotic_artifacts.py` to
regenerate the branch-oriented index at `output/by_branch/`.

# Setup

To set up the environment and install the necessary dependencies, follow the steps below:

1. Install Anaconda from [here](https://www.anaconda.com/distribution/).
2. Create a conda environment with Python 3.9. Example: `conda create --name ddp python=3.9` .
3. Activate the conda environment: `conda activate ddp` .
4. Install the required packages from `requirements.txt` and Dassl:
   `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118`
   `pip install --no-deps --no-build-isolation git+https://github.com/KaiyangZhou/Dassl.pytorch.git`


# Datasets

Ensure you have the following datasets available:

- **PASCAL VOC 2007**: Place it so that the project contains:
  `./datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages`,
  `./datasets/VOC2007/VOCdevkit/VOC2007/Annotations`, and
  `./datasets/VOC2007/VOCdevkit/VOC2007/ImageSets/Main`.
- **MS-COCO 2014**: Download the dataset and place it in `./datasets/MSCOCO`.

# CLIP weights

Place the OpenAI CLIP ViT-B/16 weight at:
`./pretrained/clip/ViT-B-16.pt`.

You can override this location with `--clip_model_path`.

# Data Partitioning

`./src/helper_functions/IncrementalDataset.py`

# Training

For the convenience of reviewers, simply run the following simple command to execute with our default configuration.
The current default incremental protocol is VOC B0-C4 (`--base_classes 0 --task_size 4 --total_classes 20`).
For paper-style DDP calibration, you can also pass `--seed 0 --lr 0.0059 --reset_optimizer_each_task --t_min 1.0 --t_max 7.0 --t_gamma 0.2`.



```
CUDA_VISIBLE_DEVICES=0 python main.py --config_file configs/models/vitb16_ep50.yaml\
    --datadir ./datasets/VOC2007/VOCdevkit/VOC2007 \
    --clip_model_path ./pretrained/clip/ViT-B-16.pt \
    --dataset_config_file configs/datasets/voc2007.yaml \
    --input_size 224 \
    --base_classes 0 \
    --task_size 4 \
    --total_classes 20
```

# EMOTIC Prototype Adapter

This branch includes an isolated Prototype Adapter experiment. It freezes the
official CLIP ViT-B/16 encoders, caches one global image feature per EMOTIC
sample, and learns a zero-initialized residual adapter against fixed ensembles
of positive/negative emotion text prototypes. Existing DDP prompts and
checkpoints are not modified.

Run the supervised 26-class feasibility upper bound first:

```
bash scripts/emotic-prototype-adapter/run_emotic_prototype_adapter_all26.sh
```

Then run the class-incremental-safe transfer experiment:

```
bash scripts/emotic-prototype-adapter/run_emotic_prototype_adapter_base5.sh
```

The Base5 variant uses only samples intersecting the first five alphabetical
EMOTIC classes and computes its training loss only on those labels. Both
protocols select checkpoints on `val` only and report `test` once after all
selection is locked. Combining `val` and `test` for metrics, visualization, or
paper reporting is forbidden; legacy `val+test` artifacts are diagnostic only.

The two runs reuse deterministic CLIP features in
`./output/emotic_clip_feature_cache`. Each output directory contains
`best_adapter.pth`, `last_adapter.pth`, `train.log`, and
`evaluation_summary.json`. Use `--force_recache` after changing CLIP weights,
input mode, or preprocessing.

## EMOTIC Prototype Adapter Few-Shot Ablation

The CLIP-Adapter-inspired sample-efficiency experiment uses
`K = 1, 2, 4, 8, 16` positive training anchors per active class and seeds
`0, 1, 2`. EMOTIC is multi-label, unlike the single-label datasets used by
CLIP-Adapter. To keep the meaning of K exact, each class receives exactly K
supervised positive labels. Positive co-labels introduced by samples selected
for another class are masked for that class, while genuine negatives in the
selected union remain supervised. The sampler records the exact source indices
and per-class supervised/ignored counts in every summary.

Run the non-incremental 26-class sample-efficiency upper bound:

```
bash scripts/emotic-prototype-adapter/run_emotic_prototype_adapter_fewshot_all26.sh
```

Then run the class-incremental-safe Base5 transfer curve:

```
bash scripts/emotic-prototype-adapter/run_emotic_prototype_adapter_fewshot_base5.sh
```

Both launchers use class-balanced masked BCE, 200 epochs, the unchanged frozen
CLIP/text prototypes/`512 -> 128 -> 512` Adapter, and the shared feature cache.
Each K is repeated over three seeds. Aggregated held-out test mean and standard
deviation are written to `output/emotic_prototype_fewshot_summary/` as JSON and
CSV. Each run also writes `fewshot_sampling.json` before optimization with the
exact source indices and supervision counts. All26 few-shot sees future class
labels and is only a feasibility curve; Base5 few-shot is the valid
incremental-transfer experiment.

After the standalone Base5-balanced adapter has been validated, task7 can be
evaluated by offline score fusion without retraining DDP:

```
bash scripts/emotic-prototype-adapter/run_emotic_prototype_fusion_task7.sh
```

The fusion script verifies exact target ordering between the saved DDP scores
and the cached Prototype features. It uses `val` only to fit one global
temperature and bias, select a global fusion weight beta by mAP, and select
global decision thresholds by mean cF1/oF1. The `test` split is evaluated only
after all selections are fixed. Results are saved to `fusion_summary.json` and
`fusion_scores.pt`.

To evaluate the incrementally safe binary-gated fusion over all eight B5-C3
tasks, run:

```
bash scripts/emotic-prototype-adapter/run_emotic_prototype_fusion_all_tasks.sh
```

For every task, the script reconstructs exactly the same seen-class sample
subset as CODE_DDP and asserts target-by-target alignment. It uses validation
data to calibrate the frozen Prototype Adapter, choose one global beta, choose
per-class gates from `{0, beta}`, and select decision thresholds. It then
reports held-out test metrics, average task mAP, and peak-to-final old-class
forgetting. It must not produce or use combined `val+test` metrics. This is
offline score fusion; it does not retrain or modify DDP checkpoints.

## EMOTIC DDP Internal Shared Feature Adapter

The internal experiment freezes every original DDP parameter and inserts one
shared `512 -> 128 -> 512` residual Adapter after DDP's token attention has
formed each class-specific positive/negative visual feature. The Adapter is
zero-initialized, so enabling it before training must preserve DDP logits
exactly. It is trained only on 16-shot Base5 task0 supervision, selected on
pure validation data, then frozen and attached to the original task0-task7 DDP
checkpoints. It does not use external Prototype scores, beta fusion, future
labels, or test data during selection.

Run the identity smoke test and three seeds in tmux with:

```
bash scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_internal_adapter_tmux.sh
```

Each seed writes training/evaluation JSON, training and per-class HTML,
`best_adapter.pth`, and eight task score files. After all seeds finish:

```
python summarize_emotic_ddp_internal_adapter.py
```

The aggregate JSON/CSV/HTML is written to
`output/emotic_ddp_internal_adapter_16shot_summary/`. Shared deterministic
task feature caches use a file lock so parallel seed evaluation cannot corrupt
them.

The first 16-shot internal run selected epoch 0 even though every trained
checkpoint was worse than the untouched identity mapping on task0 validation.
The corrected trainer includes the identity state as epoch `-1`, evaluates
after optimizer steps, and can early-stop without forcing adaptation. Run the
validation-only conservative screen with:

```
bash scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_internal_adapter_screen_tmux.sh
```

It compares bottleneck 16, residual scale 0.01, identity weight 1.0, learning
rates `1e-5/3e-5/1e-4`, and balanced versus unweighted BCE. It reuses the
existing seed0 path-feature cache, never reads test data, and writes ranked
JSON/CSV/HTML to `output/emotic_ddp_internal_adapter_screen_summary/`. A
configuration must improve task0 validation mAP by more than 0.1 before any
new three-seed test evaluation is allowed.

The next conditional stage first transfers the three already trained external
Base5 16-shot Prototype Adapter `down/up` weights into the internal pooled DDP
feature path and selects a residual scale from `0/0.001/0.003/0.01/0.03/0.1`
using task0 validation only:

```
bash scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_internal_transfer_then_full_tmux.sh
```

Transfer proceeds to locked three-seed test evaluation only when its mean val
mAP gain exceeds 0.1 and all three seeds improve. Otherwise the same pipeline
trains one Full Base5 internal upper-bound run. The Full run also evaluates
test only after exceeding +0.1 val mAP. Transfer screening writes JSON/CSV/HTML
under `output/emotic_ddp_internal_transfer_screen/`; the fallback Full run
writes its training JSON/HTML and optional strict evaluation under
`output/emotic_ddp_internal_full_base5_screen/`.

The class-token internalization experiment keeps the original DDP attention
pooled logits intact and applies the transferred external 16-shot Adapter only
to each class-specific positive/negative visual path's normalized CLS token.
The resulting text-similarity residual is added back to the original DDP
logits; no vanilla-CLIP branch, fixed Prototype prediction, beta fusion, or
second image encoding is used at inference. Run:

```
bash scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_cls_internal_transfer_tmux.sh
```

Task0 validation first selects a stable residual scale across all three
external Adapter seeds. Test evaluation is launched only when mean val gain is
greater than 0.1 and every seed improves. CLS caches and screen JSON/CSV/HTML
are written under `output/emotic_ddp_cls_internal_feature_cache/` and
`output/emotic_ddp_cls_internal_transfer_screen/`; accepted three-seed results
are summarized under `output/emotic_ddp_cls_internal_transfer_16shot_summary/`.

The cosine-difference readout is an isolated follow-up on the same frozen
three Base5 16-shot Adapter weights. Instead of the historical linear residual
projection, it adds the change between normalized adapted and original CLS
similarities to the untouched DDP path logits:

```
delta_logit = 100 * (cos(text, adapted_cls) - cos(text, original_cls))
```

Run the pre-registered validation screen and conditional three-seed test in one
tmux session with:

```
bash scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_cls_cosine_difference_tmux.sh
```

Only one global residual scale is selected on pure task0 validation. No
task-specific alpha, per-class gate, external score fusion, or test-based
selection is used. The test stage runs only if every seed improves and the
mean validation gain exceeds 0.1 mAP. Screen artifacts are written to
`output/emotic_ddp_cls_cosine_difference_screen/`; accepted per-seed results
and aggregate JSON/CSV/HTML are written to
`output/emotic_ddp_cls_cosine_difference_seed*/` and
`output/emotic_ddp_cls_cosine_difference_summary/`.

The norm-preserving Feature Correction route keeps the same frozen CLS
Adapter, but interprets its output as an EMOTIC semantic offset applied to the
original DDP pooled representation:

```
delta_f = adapted_cls - original_cls
corrected_pool = norm(pool) * normalize(pool + delta_f)
delta_logit = 100 * text.T @ (corrected_pool - pool)
final_logit = cached_ddp_logit + delta_logit
```

The last two lines are the numerically baseline-anchored implementation of
applying the original DDP dot-product head to `corrected_pool`. The Adapter
weights and DDP checkpoints remain frozen. Run Full Base5 (default) or 16-shot
transfer with:

```
GPU=0 ADAPTER_SOURCE=full_base5 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_cls_feature_correction_tmux.sh

GPU=0 ADAPTER_SOURCE=16shot SESSION=ddp_cls_feature_correction_16shot bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_cls_feature_correction_tmux.sh
```

This experiment uses a paired cache containing both class-specific CLS and DDP
pooled features. JSON/CSV/HTML artifacts use the
`emotic_ddp_cls_*feature_correction*` output names.

To compare Full Base5 external Adapter transfer under the same internal CLS
protocol, run:

```
bash scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_cls_full_base5_comparison_tmux.sh
```

The three class-balanced Full Base5 checkpoints (seeds 0/1/2) are reused; DDP
and the external Adapters are not retrained. The pipeline first screens one
global alpha on task0 validation for Feature difference, Cosine difference,
and norm-preserving Feature Correction.
Each accepted formula is then frozen and evaluated over all eight tasks. It
also produces an explicit DDP-only run using the same caches, temperature
schedule, validation-threshold policy, and forgetting implementation. No
per-class gate, task-specific alpha, external score fusion, or test-based
selection is used. The unified JSON/CSV/HTML comparison is written under
`output/emotic_ddp_cls_full_base5_comparison/`.

After the ungated CLS transfer is established, task-wise residual strength and
per-class binary residual gates can be selected without retraining:

```
bash scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_cls_internal_gate_tmux.sh
```

For each task and Adapter seed, validation selects alpha from
`0/0.001/0.003/0.01/0.03`, requiring more than +0.1 task mAP over identity.
Each seen class then enables the selected CLS correction only when its val AP
gain exceeds +0.1. Held-out test is evaluated after these choices are locked.
Outputs contain DDP-only, task-alpha, and class-gate metrics plus per-class gate
decisions in JSON and the main class-gated DetailReport in HTML/JSON. The three
seed aggregate is written to `output/emotic_ddp_cls_internal_gate_summary/`.

The internal model is frozen after this stage: alpha candidates, the `+0.1`
task margin, and the `+0.1 AP` class-gate margin must not be changed after
looking at test results. Generate the final ablation, isolated-process
batch-one GPU benchmark, and a separately labelled internal-plus-external
upper bound with:

```
bash scripts/emotic-ddp-internal-adapter/launch_emotic_adapter_final_analysis_tmux.sh
```

The upper bound combines the frozen internal CLS class-gated scores with the
external Base5 16-shot Prototype branch. Global beta, external class gates,
and thresholds are selected on val independently for each task and seed;
test is used only after selection. It is not a replacement for the standalone
internal result because it restores a second vanilla-CLIP image encoding.

Final artifacts are written to:

- `output/emotic_adapter_final_ablation/ablation.{json,csv,html}`;
- `output/emotic_adapter_inference_benchmark/benchmark.{json,csv,html}`;
- `output/emotic_ddp_cls_external_hybrid_summary/summary.{json,csv,html}`.

The latency benchmark uses batch size one, GPU forward only, ten warmups and
fifty timed iterations. Preprocessing and host-to-device transfer are excluded;
each method runs in a fresh process so peak allocated CUDA memory is comparable.

### DDP-owned prompt-free auxiliary Adapter

The external vanilla-CLIP training route can now be reproduced inside one DDP
model. During Adapter training, the frozen DDP image encoder is called with
`visual_prompts=None` and its global CLS is supervised by fixed positive and
negative text prototypes encoded by DDP's own frozen text tower. Only the
shared Adapter `W1/W2` is kept. At inference, the prompt-free auxiliary head is
discarded; the same `W1/W2` is applied to class-specific prompted CLS features
and its semantic offset corrects the original DDP pooled features.

Run the equivalence audit, three Full-Base5 seeds, pure-validation global-scale
selection, and strict eight-task evaluation in one tmux session:

```bash
GPU=0 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_prompt_free_auxiliary_tmux.sh
```

This pipeline does not read `output/emotic_prototype_adapter_*` checkpoints.
Training/evaluation JSON and HTML locations are documented in
`docs/ddp_prompt_free_auxiliary_branch.md`.

To run the complete DDP-owned Auxiliary `16-shot / Full Base5` × `Feature
difference / Cosine difference / Feature Correction` matrix on two GPUs, use:

```bash
GPU0=0 GPU1=1 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_prompt_free_auxiliary_matrix_tmux.sh
```

The launcher skips the already completed Full Base5 Feature Correction run,
trains only missing 16-shot seeds, and assigns the remaining formula runs to
GPU0/GPU1. A unified comparison against the legacy external Adapter results is
written under `output/emotic_ddp_prompt_free_auxiliary_matrix_comparison/`.

### Task-routed Adapter Bank

The incremental Adapter Bank trains one prompt-free Adapter for every B5-C3
task and routes each positive/negative prompted CLS path to the Adapter from
that class's introduction task. Old and future labels are explicitly masked;
the training scripts never construct or evaluate a test dataset. Inference uses only Feature
Difference with one pre-registered global `alpha=0.03`—there is no per-task
scale, per-class gate, or external score fusion.

Run Full-data and exact 16-shot Banks in parallel on GPU0/GPU1 with:

```bash
GPU0=0 GPU1=1 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_tmux.sh
```

The launcher audits cross-task multi-label sample overlap, trains three seeds
for both Banks, performs strict all-task evaluation, and writes the combined
JSON/CSV/HTML report to
`output/emotic_ddp_task_adapter_bank_comparison/`.

The fixed-last ASL/BAL comparison that preserves the same deterministic routing
and Feature Difference inference path is documented in
`docs/ddp_task_adapter_bank_asl_bal.md`.

## Transformer-block Task Adapter Bank

The `codex/emotic-ddp-transformer-adapter-bank` experiment keeps the original
BCE-trained DDP checkpoints frozen and trains a class-routed bank of parallel
ViT-MLP Adapters with ASL. The locked first experiment uses blocks 4--12,
`768→128→768`, Adam `4e-4`, cosine scheduling, effective batch 64, 20 epochs,
fixed-last checkpoints, and a fixed test threshold of 0.5. See
`docs/ddp_transformer_adapter_bank.md` for the protocol and server commands.
