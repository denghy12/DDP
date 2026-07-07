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

## EMOTIC 数据与协议

- 服务器源数据：`/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC`
- CODE_DDP 使用路径：`./datasets/EMOTIC`
- 标注：`CVPR17_Annotations.mat`
- 图片：`cvpr_emotic/`
- 26 类按字母序排列；train 用于训练，val 只用于模型/阈值选择，test 只用于锁定配置后的
  最终报告。禁止将 val 与 test 合并用于正式评估。
- 人物样本数：train `16001`，val `2397`，test `5368`。
- B5-C3 共 8 个任务，每任务新类数为 `5,3,3,3,3,3,3,3`。
- 纯 test seen-class 评估样本数依次为
  `3142,4600,4691,5232,5342,5354,5367,5368`。
- 默认采用 full-image。由于 EMOTIC 是人物级标注，多人物图可能产生相同完整图像、
  不同人物标签的冲突；历史聚合统计中受影响样本占 train `31.44%`、合并后的
  val+test `51.78%`，后者只能用于数据诊断，不能作为评估集口径。

## EMOTIC CODE_DDP 配置

- Backbone：冻结的 OpenAI CLIP ViT-B/16。
- 正 prompt：`a photo of a person clearly feeling`。
- 负 prompt：`a photo of a person not feeling`。
- 两者均为 7 个空格词和 7 个 CLIP BPE token；CSC 开启。
- B5-C3：30 epochs/task，物理 batch 8，累积 32，有效 batch 256，
  PCD `T=1→2,gamma=0.7`，seed 0，full-image 224。
- Joint upper：26 类单任务，30 epochs，物理 batch 2，累积 128，有效 batch 256，
  `T=1`，seed 0。
- Adam 名义 LR 为 `5.9e-3`，但 `MultiStepLR([0,20],gamma=0.1)` 会在初始化时
  立即衰减；实际 epoch 0–19 为 `5.9e-4`，epoch 20–29 为 `5.9e-5`。
- loss weight `0.03`；训练 transform 和 val transform 与 multi-lane EMOTIC 对齐。

## EMOTIC 结果摘要（严格纯 test）

- B5-C3 threshold 0.80：final mAP/amAP/oF1/cF1 =
  `30.8241/37.7120/29.9534/11.9843`。
- B5-C3 threshold 0.50：final mAP/amAP/oF1/cF1 =
  `30.8241/37.7120/49.3122/31.4927`；八任务平均 oF1/cF1 =
  `59.3235/36.5039`。该阈值此前参考过混合 val+test 诊断，因此 F1 仅作阶段性结果。
- B5-C3 已完成纯 val 全八任务统一阈值选择：扫描 `0.05–0.95`（步长 `0.01`），
  按八任务平均 `(oF1+cF1)/2` 最大化选得 threshold `0.36`；对应纯 val 平均
  oF1/cF1/综合值为 `66.2677/46.2670/56.2674`。该阈值已锁定，下一次正式操作只能
  在纯 test 上评估 `0.36`，不得再根据 test 结果修改。
- 最接近的 multi-lane CLIP ViT-B/16 patch B5-C3 参考：final
  mAP/amAP/oF1/cF1 = `32.8635/39.8831/47.0667/20.2515`。
- CODE_DDP joint upper：只在纯 val 上选得折中 threshold `0.04`，锁定后纯 test
  mAP/oF1/cF1 = `35.8955/55.2235/37.2848`。
- Joint upper 相对 B5-C3 threshold 0.50 的纯 test final 差距为
  mAP/oF1/cF1 `+5.07/+5.91/+5.79`；其中 mAP 比较严格有效，F1 需等 B5-C3
  完成纯 val 阈值选择后再作为最终公平比较。
- multi-lane frozen CLIP patch upper mAP 为 `37.1691`；其原始 F1 使用未校准
  threshold 0.8，不作为主要对照。
- Upper bound 只有一个任务，`amAP == mAP`；不能与 B5-C3 跨任务 amAP 直接比较。
- 历史 `val+test` 结果全部降级为诊断记录，不得用于论文主表、结论或 upper bound。

## 正式评估硬规则

- 论文指标只能来自官方纯 `test` split，任何 `val+test` 拼接评估均视为无效。
- `val` 仅负责 checkpoint、threshold 及其他超参数选择；锁定全部配置后才能运行 test。
- test 不参与任何调参，正式输出目录、日志、HTML 和 JSON 必须明确记录
  `split=test`、样本数和已锁定的验证集配置。

## EMOTIC 输出与入口

- B5-C3 训练：`run_emotic_b5c3_semantic_tau2.sh`
- B5-C3 原始输出：`output/emotic_b5c3_ddp_semantic_tau2/`
- B5-C3 threshold 0.50 输出：
  `output/emotic_b5c3_ddp_semantic_tau2_threshold050/`
- Joint 训练：`run_emotic_upper_bound_semantic_threshold050.sh`
- Joint 输出：`output/emotic_upper_bound_ddp_semantic_threshold050/`
- 阈值扫描：`eval_emotic_threshold_sweep.py`
- 全任务 checkpoint 复评：`eval_emotic_all_tasks.py`
- 所有主实验均输出 class-order JSON、逐类 task-table HTML/JSON 和独立 checkpoint。

## EMOTIC 已知实现注意事项

- 阈值只影响二值 precision/recall/F1，不影响由连续分数排序得到的 AP/mAP。
- `text_feature_cache` 未保存在 checkpoint 中；复评脚本会显式重建缓存。
- `evaluation_metrics.prf_cal()` 的控制台阈值仍硬编码为 0.8；正式比较应读取
  detail HTML/JSON，而不是混用控制台 F1。
- 类别 support 最大相差约 39 倍；后续需要评估 class-balanced loss。
- 正负 prompt 平均分离增大，但不是所有类别都单调拉开，不能仅以 prompt cosine
  判断分类质量。

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
