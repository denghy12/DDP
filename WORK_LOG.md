# WORK_LOG

## 2026-06-30

- Preserved the clean `emotic` baseline at commit `7878f58` and created the
  `emotic-prototype-adapter` branch for the new experiment.
- Added a standalone EMOTIC Prototype Adapter without changing the DDP prompt
  learner or existing checkpoints.
- The experiment uses frozen official CLIP ViT-B/16 global features, fixed
  positive/negative text-prototype ensembles, and a zero-initialized
  `512 -> 128 -> 512` residual adapter.
- Added `all26` supervised feasibility and strict `base5` transfer protocols.
  Base5 excludes later classes from its loss and selects its checkpoint only by
  base-class mAP on base-positive validation samples.
- Separated `val` checkpoint selection from final `test` reporting. A combined
  `val+test` result is reported only for comparison with existing CODE_DDP runs.
- Added deterministic feature caching, zero-shot/best/per-class metrics,
  identity preservation regularization, optional class-balanced BCE, launcher
  scripts, and adapter unit tests.

## 2026-06-24

- 创建独立 Python 3.9 Conda 环境 `ddp`，路径为 `/opt/conda/envs/ddp`，未修改服务器其他环境。
- 将最初的路径环境 `/mnt/haoyuan/ddp` 完整克隆并验证后移除。
- 安装 PyTorch 2.0.1+cu118、项目依赖和官方 Dassl。
- 修正 VOC 路径硬编码，使 `--datadir` 生效。
- 增加 `--clip_model_path`，并优先读取 `./pretrained/clip/ViT-B-16.pt`。
- 修正 `--train_batch_size`、`--num_workers` 和 `--epochs` 参数的使用。
- 初始化累计 CF1/OF1，并确保 checkpoint 目录保存前创建。
- 下载并校验 OpenAI CLIP ViT-B/16 官方权重。
- 使用真实 VOC2007 图片完成：
  - 数据加载；
  - DDP/CLIP 模型构建；
  - 单类前向；
  - 基础任务 4 类的一次反向传播和 optimizer step。
- 训练 smoke test：batch size 1，loss `0.1036628410`，峰值显存约 `1.23 GiB`。
- 在 tmux 会话 `ddp_voc_1ep` 中启动 VOC B4-C2 全任务 1 epoch debug experiment。
- 运行参数：ViT-B/16、batch size 8、8 个 DataLoader workers、loss weight 0.03、GPU 0。
- 日志：`./logs/voc_B4-C2_ddp_vitb16_1ep.log`。
- 1 epoch debug experiment 完成：final/average mAP 为 `70.79/77.30`，明显低于项目自带
  20 epoch 结果的 `83.58/90.75`，主要对照差异是训练轮数。
- 已在 tmux 会话 `ddp_voc_20ep` 启动补充代码完整 20 epoch VOC B4-C2 实验。
- 完整实验日志：`./logs/voc_B4-C2_ddp_vitb16_20ep.log`。
- 完整配置：ViT-B/16、B4-C2、batch 8、20 epoch/task、Adam 5.9e-3、
  MultiStepLR `[0,20]`、loss weight 0.03、PCD tau 1→7/gamma 0.2、threshold 0.8。
- VOC B4-C2 20 epoch 实验已同步并完成分析：
  - 日志：`./logs/voc_B4-C2_ddp_vitb16_20ep.log`
  - final task：mAP `81.4849`，CF1 `69.6920`，OF1 `70.8030`
  - average/mean mAP：`88.9816`
  - 对比项目自带 `results/VOC-B4C2.log` final task `83.5802/72.4685/74.4435`
    （mAP/CF1/OF1），当前分别低 `2.0953/2.7765/3.6405`。
  - 对比 1 epoch debug final task `70.7949/38.7009/47.6621`
    （mAP/CF1/OF1），20 epoch 分别提升 `10.6900/30.9911/23.1409`。
  - 主要差距集中在增量后期 recall/OF1；final task CP 比参考略高
    （`77.7882` vs `76.9439`），CR 明显更低（`71.3117` vs `76.4232`）。
- 按用户要求将默认 VOC 协议改为 B0-C4：
  - 新增命令行参数 `--base_classes`、`--task_size`、`--total_classes`。
  - 默认值为 `0/4/20`，即 VOC B0-C4。
  - `base_classes=0` 时不再生成空的 `(0, 0)` 任务，任务划分为
    `(0,4),(4,8),(8,12),(12,16),(16,20)`。
  - 如需复现旧 B4-C2，需要显式传入 `--base_classes 4 --task_size 2 --total_classes 20`。
- 为对照 `multi-lane-main` 的 paper-style 配置，补充了：
  - `--seed`
  - `--reset_optimizer_each_task`
  - `--t_min/--t_max/--t_gamma`
  - `--lr` 直接控制 DDP optimizer 学习率
- VOC B0-C4 三组 CODE_DDP 对照结果已同步并分析：
  - A 纯官方默认 random + tau7/gamma0.2：
    `./logs/voc_B0-C4_ddp_vitb16_20ep_seed0_gpu1.log`
    final mAP/mean_mAP/OF1/CF1 = `88.2210/92.9532/56.2860/47.3973`。
    该组 precision 高但 recall 严重偏低，final CP/CR = `93.1767/37.9526`。
  - B 官方代码 random + tau2/gamma0.7：
    `./logs/voc_B0-C4_ddp_vitb16_20ep_random_tau2_seed0_gpu1.log`
    final mAP/mean_mAP/OF1/CF1 = `84.8075/84.4384/78.7446/76.2101`。
    相对 A，OF1/CF1 分别提升 `22.4586/28.8128`，但 mAP/mean_mAP 分别下降 `3.4135/8.5148`。
  - C 官方代码原生 semantic prompt + tau2/gamma0.7：
    `./logs/voc_B0-C4_ddp_vitb16_20ep_semantic_tau2_seed0_gpu2.log`
    final mAP/mean_mAP/OF1/CF1 = `90.6430/95.0634/82.4081/81.8312`。
    相对 B，mAP/mean_mAP/OF1/CF1 分别提升 `5.8355/10.6250/3.6636/5.6210`。
    相对 A，mAP/mean_mAP/OF1/CF1 分别提升 `2.4220/2.1101/26.1221/34.4339`。
  - C 组使用 `--positive_prompt "a photo containing a"` 与
    `--negative_prompt "a photo without a"`，CODE_DDP 会将 context token 数改为 4，
    与 `multi-lane-main` 的 16-token semantic seed 不是完全等价。
  - 结论：在当前 CODE_DDP 官方代码路径下，semantic + tau2 是目前最强 B0-C4 配置。
- 最新结果复核发现两项复现风险：
  - 三组 B0-C4 均保存到 `./checkpoint/b0c4/epoch20/`，并行运行时互相覆盖；
    当前服务器该目录中的 task checkpoint 不能视为任一单组实验的完整连续产物。
  - `VOC.get_filenames()` 将 `set` 直接转换成 `list`，样本顺序受 Python hash
    随机化影响；当前仅设置运行时 seed，没有在 Python 启动前固定
    `PYTHONHASHSEED`。因此两个 random seed0 实验仍出现明显不同的训练轨迹。
- B0-C4 semantic + tau2 与论文主表对比：
  - 当前 last mAP/CF1/OF1、average mAP：
    `90.6430/81.8312/82.4081/95.0634`
  - 论文：`90.2/76.9/80.8/94.8`
  - 当前分别为 `+0.44/+4.93/+1.61/+0.26`。
  - 但该组将语义短语直接作为 context，使 prompt 长度从 16 变为 4，并使用
    `tau_max=2,gamma=0.7`，因此属于达到论文水准的调优配置，而不是严格默认复现。
