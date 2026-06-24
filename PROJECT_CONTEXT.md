# PROJECT_CONTEXT

## 当前运行环境

- 远程 Conda 环境名：`ddp`
- 环境路径：`/opt/conda/envs/ddp`
- Python：3.9.25
- PyTorch：2.0.1+cu118
- torchvision：0.15.2+cu118
- torchaudio：2.0.2+cu118
- NumPy：1.23.5
- GPU：NVIDIA GeForce RTX 4090

## VOC2007

项目默认数据目录：

`./datasets/VOC2007/VOCdevkit/VOC2007`

该目录下必须包含：

- `JPEGImages/`
- `Annotations/`
- `ImageSets/Main/trainval.txt`
- `ImageSets/Main/test.txt`

服务器目前可只读复用：

`/mnt/haoyuan/workspace/multi-lane-main/datasets/VOC2007/VOCdevkit/VOC2007`

## CLIP

- Backbone：OpenAI CLIP ViT-B/16
- 默认权重：`./pretrained/clip/ViT-B-16.pt`
- 官方 SHA-256：
  `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`

可以通过 `--datadir` 和 `--clip_model_path` 覆盖默认路径。

## 当前复现实验状态

- 当前代码默认增量协议已改为 VOC B0-C4：`--base_classes 0 --task_size 4 --total_classes 20`。
- 如需复现实验记录中的 VOC B4-C2，需要显式传入：`--base_classes 4 --task_size 2 --total_classes 20`。
- 已补充对照所需的关键开关：`--seed`、`--reset_optimizer_each_task`、`--t_min/--t_max/--t_gamma`，以及 `--lr` 可直接控制 DDP optimizer 学习率。
- VOC B4-C2 1 epoch debug 已完成：final/average mAP 为 `70.7949/77.3016`。
- VOC B4-C2 20 epoch 已完成并同步日志：`./logs/voc_B4-C2_ddp_vitb16_20ep.log`。
- 20 epoch 当前结果：final mAP/CF1/OF1 为 `81.4849/69.6920/70.8030`，average/mean mAP 为 `88.9816`。
- 项目自带参考 `./results/VOC-B4C2.log`：final mAP/CF1/OF1 为 `83.5802/72.4685/74.4435`，mean mAP 为 `90.7454`。
- 当前 20 epoch 低于参考，主要差距在增量后期 recall 和 OF1；下一步优先核对协议、随机性和评估阈值。

## VOC B0-C4 对照结果

- A：CODE_DDP 纯官方默认 random + tau7/gamma0.2，日志
  `./logs/voc_B0-C4_ddp_vitb16_20ep_seed0_gpu1.log`：
  final mAP/mean_mAP/OF1/CF1 = `88.2210/92.9532/56.2860/47.3973`。
- B：CODE_DDP random + tau2/gamma0.7，日志
  `./logs/voc_B0-C4_ddp_vitb16_20ep_random_tau2_seed0_gpu1.log`：
  final mAP/mean_mAP/OF1/CF1 = `84.8075/84.4384/78.7446/76.2101`。
- C：CODE_DDP 原生 semantic prompt + tau2/gamma0.7，日志
  `./logs/voc_B0-C4_ddp_vitb16_20ep_semantic_tau2_seed0_gpu2.log`：
  final mAP/mean_mAP/OF1/CF1 = `90.6430/95.0634/82.4081/81.8312`。
- 当前 CODE_DDP 最强 B0-C4 配置为 C 组。semantic prompt 在 CODE_DDP 中会把 context
  长度改成短语词数（本次 positive 为 4 tokens），不是 `multi-lane-main` 的 16-token
  semantic seed。
- `multi-lane-main` official semantic tau2 20ep 已同步，final mAP/amAP/OF1/CF1 =
  `88.5226/93.4123/72.7760/76.3618`；3ep 为 `88.0259/93.3177/79.4418/78.6858`。
  该实现 20ep mAP 上升但 F1 下降，CODE_DDP 20ep semantic tau2 当前整体更强。
