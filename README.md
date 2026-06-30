# Paper 

PyTorch code for: DDP: Dual-Decoupled Prompting for Multi-Label Class-Incremental Learning

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
bash run_emotic_prototype_adapter_all26.sh
```

Then run the class-incremental-safe transfer experiment:

```
bash run_emotic_prototype_adapter_base5.sh
```

The Base5 variant uses only samples intersecting the first five alphabetical
EMOTIC classes and computes its training loss only on those labels. Both
protocols select checkpoints on `val` only, report `test` once after selection,
and additionally report `val+test` for comparison with the existing CODE_DDP
evaluation convention.

The two runs reuse deterministic CLIP features in
`./output/emotic_clip_feature_cache`. Each output directory contains
`best_adapter.pth`, `last_adapter.pth`, `train.log`, and
`evaluation_summary.json`. Use `--force_recache` after changing CLIP weights,
input mode, or preprocessing.

After the standalone Base5-balanced adapter has been validated, task7 can be
evaluated by offline score fusion without retraining DDP:

```
bash run_emotic_prototype_fusion_task7.sh
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
bash run_emotic_prototype_fusion_all_tasks.sh
```

For every task, the script reconstructs exactly the same seen-class sample
subset as CODE_DDP and asserts target-by-target alignment. It uses validation
data to calibrate the frozen Prototype Adapter, choose one global beta, choose
per-class gates from `{0, beta}`, and select decision thresholds. It then
reports held-out test and legacy val+test metrics, average task mAP, and
peak-to-final old-class forgetting. This is offline score fusion; it does not
retrain or modify DDP checkpoints.
