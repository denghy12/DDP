# DDP 内置无 Prompt 辅助 Adapter

这条实现把原先“外部原始 CLIP 上训练 Adapter，再把权重迁移到 DDP”重构为一个 DDP 模型内部的训练期辅助分支。视觉主干、文本主干和 Adapter 权重都只保留一份。

## 训练期三条表示路线

1. **无 Prompt 辅助路线（仅训练期）**
   - 输入采用与原外部 OpenAI CLIP 实验完全相同的 resize、center crop 和 CLIP normalization。
   - 调用 `DDP.image_encoder(image, visual_prompts=None)`。
   - 取最终投影并归一化的全局 CLS，使用固定正/负文本 prototype 和 Base5 标签训练共享 Adapter。

2. **原始 DDP 主路线**
   - 类别正/负 visual prompt 进入冻结视觉编码器。
   - 全 token 序列经过 DDP 的 token–text attention pooling，得到每条正/负路径的 pooled feature 和原始 logits。

3. **Prompted CLS Adapter 路线**
   - 从与原始 DDP 主路线相同的一次 prompted 编码中取每条正/负路径的 CLS。
   - 将训练期辅助路线学到的同一份 `W1/W2` 用于 prompted CLS，估计 EMOTIC 语义偏移。
   - 偏移加到 DDP pooled feature，重新归一化到原 pooled norm，并沿用原始 DDP 文本相似度头。

## 推理期

无 Prompt 辅助路线和 prototype head 都被丢弃。推理只做原来的一次 class-specific prompted 编码，其输出同时提供 full tokens 和 CLS。因此相较当前内部 Feature Correction，不会增加第二次 CLIP 图像编码。

## 输出

每个训练 seed：

- `output/emotic_ddp_prompt_free_auxiliary_base5_seed{seed}/best_adapter.pth`
- `output/emotic_ddp_prompt_free_auxiliary_base5_seed{seed}/training_summary.json`
- `output/emotic_ddp_prompt_free_auxiliary_base5_seed{seed}/evaluation_summary.json`
- `output/emotic_ddp_prompt_free_auxiliary_base5_seed{seed}/training_history.html`

严格 DDP all-task 评估：

- `output/emotic_ddp_prompt_free_auxiliary_cls_feature_correction_screen/`
- `output/emotic_ddp_prompt_free_auxiliary_cls_feature_correction_seed{seed}/`
- `output/emotic_ddp_prompt_free_auxiliary_cls_feature_correction_summary/`

等价性审计：

- `output/emotic_ddp_prompt_free_auxiliary_equivalence/equivalence.json`

流水线启动时会临时加载一次 vanilla CLIP，只用于证明 DDP 的无 prompt CLS 与固定文本编码在数值上等价。该进程结束后才开始 Adapter 训练，因此训练与推理都没有第二套 CLIP。

## 服务器运行

```bash
cd /mnt/haoyuan/workspace/CODE_DDP
conda activate ddp
GPU=0 bash scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_prompt_free_auxiliary_tmux.sh
```

训练和评估均通过 JSON 记录：单一 CLIP 实例、辅助训练时无 visual prompt、只在 validation 上选择 checkpoint/全局残差比例、test 不参与选择，以及推理时删除辅助路线。

## 16-shot / Full Base5 × 三种公式

```bash
GPU0=0 GPU1=1 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_prompt_free_auxiliary_matrix_tmux.sh
```

该 tmux session 包含 `gpu0`、`gpu1` 两个窗口。Full Base5 使用 50 epochs；16-shot 与旧外部 few-shot 设置一致，使用每类 16 个正样本锚点、class-balanced masked BCE、200 epochs、三个 seed。已完成且协议匹配的实验会被跳过。

最终统一对照表：

- `output/emotic_ddp_prompt_free_auxiliary_matrix_comparison/comparison_summary.json`
- `output/emotic_ddp_prompt_free_auxiliary_matrix_comparison/comparison_summary.csv`
- `output/emotic_ddp_prompt_free_auxiliary_matrix_comparison/comparison_summary.html`
