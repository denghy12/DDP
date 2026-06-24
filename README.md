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
