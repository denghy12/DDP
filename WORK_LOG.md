# WORK_LOG

## 2026-07-07

- 根据 CLIP-Adapter 的 few-shot 设定新增 Prototype Adapter 样本效率实验，shot
  取 `1/2/4/8/16`，每个配置运行 seed `0/1/2`。
- 针对 EMOTIC 多标签共现设计严格 K-shot 采样：每个 active class 恰好使用 K 个
  supervised positive anchors；由其他类别抽样带入的额外正共标签对该类进行 mask，
  真实负标签继续参与 BCE，避免 nominal K-shot 实际获得超过 K 个正监督。
- 新增 masked class-balanced BCE、精确采样索引/逐类监督统计、All26/Base5 批量启动
  脚本及跨 seed JSON/CSV 汇总。All26 曲线仅为非增量样本效率上限，Base5 曲线用于
  检验少量 Task0 标注学到的共享域映射能否迁移到未来21类。

## 2026-07-01

- 将论文评估协议升级为项目级硬规则：正式指标只能在官方纯 test split 上计算；val
  只用于 checkpoint、阈值和超参数选择，严禁再次拼接 val+test 用于正式评估、可视化
  或论文报告。所有既有 val+test 数值均标记为历史诊断结果，不再作为论文结果或上界。
- 完成 EMOTIC B5-C3 两组 checkpoint 的纯 test 复评（5368 个人物样本）：
  - threshold 0.50：final mAP/amAP/oF1/cF1 =
    `30.8241/37.7120/49.3122/31.4927`；
  - threshold 0.80：final mAP/amAP/oF1/cF1 =
    `30.8241/37.7120/29.9534/11.9843`。
  两组 mAP 相同符合排序指标定义；0.50 的 F1 更高，但因该阈值曾参考混合
  val+test 诊断，最终论文 F1 仍需通过纯 val 重新选择阈值。
- 完成 joint upper 的严格协议评估：在纯 val（2397 样本）选择折中 threshold `0.04`，
  锁定后在纯 test（5368 样本）得到 mAP/oF1/cF1 =
  `35.8955/55.2235/37.2848`。该结果取代旧的 val+test upper
  `38.3274/62.7937/38.7754`，成为当前可用于论文的 joint upper。
- 严格纯 test 下，joint upper 相对 B5-C3 threshold 0.50 final 提升
  mAP/oF1/cF1 `+5.07/+5.91/+5.79`。mAP 差距可直接报告；F1 差距需待 B5-C3
  完成纯 val 阈值选择后最终确认。
- 完成 B5-C3 全部 8 个 checkpoint 的纯 val 阈值扫描。每个 Task 仅使用官方 val，
  seen-class 样本数为 `2285,2375,2381,2397,2397,2397,2397,2397`；基于缓存的纯 val
  分数在 `0.05–0.95`、步长 `0.01` 上统一汇总。预先固定的选择标准为八任务平均
  `(oF1+cF1)/2`，最佳统一 threshold 为 `0.36`，对应平均 oF1/cF1/综合值
  `66.2677/46.2670/56.2674`。`0.37` 的综合值为 `56.2672`，说明峰值较平坦，仍按
  确定性最大值锁定 `0.36`。输出位于
  `output/emotic_b5c3_ddp_semantic_tau2_val_threshold_sweep/`。
- 纯 val 下，最大平均 oF1 的 threshold 为 `0.49`（平均 oF1 `67.3821`），最大平均
  cF1 的 threshold 为 `0.27`（平均 cF1 `46.9982`）；论文主协议不分别挑选，而是统一
  使用已预先定义的综合标准所选 `0.36`。下一步只允许在纯 test 上用锁定的 `0.36`
  评估一次，不能用 test 再调阈值。

- 完成 CODE_DDP EMOTIC joint-training upper bound：26 类在单一任务中联合训练，
  train 和历史合并评估集分别为 `16001/7765` 个人物样本，30 epochs，物理 batch 2、
  梯度累积 128、有效 batch 256，full-image、seed 0、semantic 正负 prompt、
  `T=1`。启动脚本为 `run_emotic_upper_bound_semantic_threshold050.sh`。
- 为 joint upper bound 增加 `--upper_bound` 协议支持：允许
  `base_classes == total_classes == 26`，只生成 `(0,26)` 一个任务，并输出与
  B5-C3 相同格式的 class-order JSON、逐类 HTML/JSON 和 checkpoint。
- 历史 val+test 口径下，Joint upper bound 在初始 threshold `0.50` 下结果为：mAP `38.3323`、
  oF1 `29.4237`、cF1 `13.9217`。其排序能力形成了有效上界，但概率明显偏保守，
  全局 precision/recall 为 `87.57%/17.68%`。
- 修正 `eval_emotic_threshold_sweep.py` 对 upper-bound checkpoint 的识别：task 0
  应包含全部 26 类而非 B5-C3 的前 5 类，且 joint 评估固定 `T=1`。
- 历史 val+test 阈值扫描范围为 `0.05–0.80`（仅保留作诊断，不得用于论文）：
  - 最佳 oF1：threshold `0.11`，mAP/oF1/cF1 = `38.3274/63.0375/37.0217`；
  - 最佳 cF1：threshold `0.05`，mAP/oF1/cF1 = `38.3274/60.6964/39.9529`；
  - 最佳 oF1/cF1 折中：threshold `0.08`，mAP/oF1/cF1 =
    `38.3274/62.7937/38.7754`。
- 历史 val+test 口径下，以各自折中阈值比较，joint upper 相对 B5-C3 continual final 提升
  mAP/oF1/cF1 `+4.85/+10.15/+5.27`；26 类中有 23 类 AP 提升。最大 AP 提升来自
  Suffering `+27.87`、Pleasure `+19.31`、Sadness `+13.48`。
- `multi-lane-main` 的 frozen CLIP ViT-B/16 patch upper bound 位于
  `output/emotic_upper_bound_clip_vit_b16_patch/`，mAP 为 `37.1691`。其原始
  threshold 0.8 下 F1 严重失准，因此当前只将 mAP 作为可靠对照；CODE_DDP joint
  upper 的 mAP 高 `1.16`。
- 注意：以下 `38.33` 与 `33.48` 均为历史 val+test 数值，不可用于论文；此外
  upper-bound 只有一个任务，其 `amAP == mAP`，不能与 B5-C3 的跨任务
  amAP `40.5820` 直接比较；正确比较是 joint mAP `38.33` 对 continual final mAP
  `33.48`。

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
- At that time, `val` checkpoint selection was separated from final `test`
  reporting, but a combined `val+test` result was still emitted for legacy
  comparison. That legacy output is now prohibited and is not a paper result.
- Added deterministic feature caching, zero-shot/best/per-class metrics,
  identity preservation regularization, optional class-balanced BCE, launcher
  scripts, and adapter unit tests.
- Switched the standalone feature extractor from the legacy JIT execution path
  to an eager vanilla CLIP model because PyTorch 2.0.1 no longer supports the
  old JIT Node subscript API. The DDP prompt-aware `CLIP_conv_proj` path remains
  unchanged.
- Added task7 offline fusion evaluation for the saved DDP scores and the
  Base5-balanced Prototype Adapter. The script asserts exact target alignment,
  fits a global Prototype temperature/bias on `val`, selects beta and decision
  thresholds on `val`, and reports `test` only after selection.
- Extended offline fusion to all eight EMOTIC B5-C3 tasks. Each task rebuilds
  CODE_DDP's seen-class sample mask from cached labels, asserts exact score
  target order, and performs all calibration and selection on `val` only.
- Added the safer per-class binary gate as the main fusion: every seen class
  chooses either DDP-only (`beta=0`) or the task's globally selected mixture.
  The original summary emitted both test and combined val+test metrics; the
  combined metrics are historical diagnostics and must no longer be produced
  or reported. The test metrics remain subject to the strict protocol above.

## 2026-06-24 至 2026-06-25（EMOTIC）

- 从 `main` 创建 `emotic` 分支并加入原生 EMOTIC 支持（以下为当时的历史实现，
  其中 val+test 评估已废止）：
  - 新增 `src/helper_functions/emotic_loader.py`，读取
    `CVPR17_Annotations.mat` 和 `cvpr_emotic/`；
  - 类别采用 26 类字母序；训练使用 train，当时错误地使用 val+test 评估；
  - 服务器数据通过 `datasets/EMOTIC` 只读链接复用
    `/mnt/haoyuan/workspace/multi-lane-main/datasets/EMOTIC`；
  - train/val+test 人物样本数为 `16001/7765`。
- 当时基于废止的 val+test 口径实现 EMOTIC B5-C3 协议，共 8 个任务：
  `(0,5),(5,8),(8,11),(11,14),(14,17),(17,20),(20,23),(23,26)`；
  每个任务的 seen-class 评估样本数为
  `5427,6975,7072,7629,7739,7751,7764,7765`，与 `multi-lane-main` 对齐。
- 训练增强与 `multi-lane-main` 对齐：RandomResizedCrop 224
  (`scale=0.05–1.0`,`ratio=3/4–4/3`) + horizontal flip + ToTensor；评估为
  Resize 256 + CenterCrop 224 + ToTensor；默认 full-image。
- B5-C3 semantic tau2 正式配置：30 epochs/task、物理 batch 8、梯度累积 32、
  有效 batch 256、Adam、loss weight 0.03、seed 0、每任务重建 optimizer、
  PCD `T=1→2,gamma=0.7`。正负 prompt 分别为
  `a photo of a person clearly feeling` / `a photo of a person not feeling`，
  均为 7 个空格词和 7 个 CLIP BPE token。
- 新增 `src/helper_functions/detail_report.py`，输出与 `multi-lane-main` 对齐的：
  - `{run}_class_order.json`；
  - `{run}_per_class_task_table.html`；
  - `{run}_per_class_task_table.json`；
  - 每类 AP/F1/precision/recall/support/TP/FP/FN/TN 及每任务
    mAP/amAP/oF1/cF1/loss。
- 新增独立输出/checkpoint 目录，避免 VOC 时期不同实验互相覆盖；增加有效 batch
  梯度累积、独立 eval batch、smoke task/batch 限制等参数。
- 历史 val+test 下 B5-C3 原始 threshold 0.8 结果（非论文结果）：
  final mAP/amAP/oF1/cF1 =
  `33.4675/40.5761/29.8385/11.8034`。mAP 略高于最接近的 multi-lane
  CLIP ViT-B/16 patch 参考 `32.8635/39.8831/47.0667/20.2515`，但 recall 因
  `T=2 + threshold=0.8` 严重塌缩。
- 曾在 val+test 上对 Task 7 checkpoint 完成 threshold `0.30–0.85` 扫描；该选择存在
  test 泄漏，仅保留以下数值用于追溯，不得复用：
  - 最佳 oF1 threshold `0.55`：oF1/cF1 `53.2209/31.2239`；
  - 最佳 cF1 threshold `0.42`：oF1/cF1 `50.2717/35.1711`；
  - 最佳折中 threshold `0.50`：oF1/cF1 `52.6416/33.5044`。
- 曾使用统一 threshold `0.50` 在 val+test 上重新评估全部 8 个 checkpoint，生成的
  以下报告属于历史诊断，不得作为论文结果：
  `output/emotic_b5c3_ddp_semantic_tau2_threshold050/`。最终
  mAP/amAP/oF1/cF1 = `33.4788/40.5820/52.6416/33.5044`；八任务平均
  oF1/cF1 = `62.1423/37.8072`。相对 CLIP patch 参考 final 提升
  mAP/oF1/cF1 `+0.62/+5.57/+13.25`。
- 阈值只改变二值 TP/FP/FN/TN、precision/recall/F1，不改变连续分数排序，
  因而不改变 AP/mAP/amAP；两次重推理约 `0.01` 的 mAP 差异来自浮点和缓存重建。
- 正负 prompt 的分离度得到量化：
  - 原始 context pos/neg cosine `0.854→0.763`，L2 `0.371→0.512`；
  - CLIP 编码后同类 pos/neg cosine `0.972→0.926`，L2 `0.233→0.362`；
  - 平均上确实拉开，但 Engagement、Excitement 等类别并非单调分离。
- 复核发现的 EMOTIC 风险：
  - full-image 是人物级标签，同一多人图会出现相同输入、不同标签；受冲突影响的
    人物样本占 train `31.44%`、val+test `51.78%`；
  - 类别 support 最大/最小相差约 `39×`，不同 B5-C3 task 的 optimizer step 数
    差异很大；
  - `MultiStepLR([0,20])` 在构造时即衰减，名义 LR `5.9e-3` 的实际 LR 为
    epoch 0–19 `5.9e-4`、epoch 20–29 `5.9e-5`；
  - `text_feature_cache` 不在 checkpoint state_dict 中，离线评估必须重建；
  - 控制台 `prf_cal()` 仍硬编码 threshold 0.8，HTML/JSON 则使用 `--thre`；
  - 当前 threshold 是在 val+test 上诊断/选择，不能作为无偏 test 结果，正式报告
    必须在 val 选阈值并在 test 上报告。
- 新增工具和入口：
  - `run_emotic_b5c3_semantic_tau2.sh`；
  - `eval_emotic_threshold_sweep.py`；
  - `eval_emotic_all_tasks.py`；
  - `run_emotic_upper_bound_semantic_threshold050.sh`。

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
