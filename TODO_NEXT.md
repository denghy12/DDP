# TODO_NEXT

## EMOTIC 主实验

1. B5-C3 已在纯 val 全八任务上选定并锁定统一 threshold `0.36`。下一步只在纯 test
   上复评一次，生成最终论文 HTML/JSON；不得根据 test 结果反向调整阈值。
2. Joint upper 的严格评估已完成：纯 val 选择 threshold `0.04`，纯 test 得到
   mAP/oF1/cF1 `35.8955/55.2235/37.2848`。后续只维护这一严格结果；旧 val+test
   upper 仅作历史诊断，不再进入论文比较。
3. 做 full-image 与 `person_crop` 的单变量对照。full-image 中受多人冲突标签影响的样本
   占 train `31.44%`、历史合并 val+test `51.78%`，这是当前最重要的数据协议风险；
   后续应分别统计纯 val 与纯 test，但不得拼接二者进行指标评估。
4. 修正并消融学习率调度：当前 `[0,20]` 会把名义 `5.9e-3` 立即变成 `5.9e-4`；至少
   比较保持历史行为与去掉 milestone 0 的版本，并记录每 epoch 实际 LR。
5. 针对约 39 倍类别不平衡，比较 unweighted BCE、class-balanced BCE/pos-weight 和
   task-balanced optimizer-step 方案；阈值必须在独立 val 上重选。
6. 将 `text_feature_cache` 重建做成正式 checkpoint evaluation 路径，并让控制台
   `prf_cal()` 使用 `--thre`，避免控制台 0.8 与 HTML/JSON 指定阈值不一致。
7. 增加每 epoch 或固定间隔的 val mAP/F1 与 checkpoint，确认 30 epochs 是否必要，并
   对 joint upper 做学习曲线/早停诊断。
8. 至少增加 seed 1/2，对 B5-C3 final mAP、joint gap 和阈值稳定性报告均值与方差。

## Prototype Adapter / 融合

9. 运行 CLIP-Adapter 风格的 `K={1,2,4,8,16}` few-shot 曲线，每个 K 使用
   seed 0/1/2。先完成 All26 样本效率上限，再完成 Base5 向未来21类迁移的严格实验；
   报告纯 test 均值±标准差和真实 unique sample 数。
10. 核对全八任务二值类别门控融合的 target alignment、平均 mAP 与 old-class
   forgetting；所有 beta、gate 和 threshold 只允许在 val 选择。
11. 将全局 beta、二值类别门控和 DDP-only 逐任务并列；连续逐类 beta 仅作为高自由度
    诊断，不作为主结果。
12. 保留 All26-balanced 作为监督可行性上限、Base5-balanced 作为增量安全主版本、
    Base5 普通 BCE 作为排序消融；在离线融合确认稳定收益前不修改 DDP 训练目标。
13. 离线融合严格实验已完成：Full Base5 三 seed 的 final mAP 为
    `33.8449±0.0510`，16-shot 三 seed为 `32.4573±0.1350`。内部 CLS class-gate
    final mAP 为 `31.8938±0.1041`；其 alpha 与门控规则已冻结，不再根据 test 调整。
14. 生成最终消融表和独立进程 GPU 效率表；另运行“内部 CLS gate + 外部 16-shot”
    组合上限。组合的 beta、类别门控和 threshold 只能在纯 val 选择，且必须与不含外部
    编码器的内部主结果分开报告。
15. Full Base5/16-shot × Feature difference/Cosine difference/Feature Correction
    的 DDP-owned auxiliary 三 seed 对照已完成。六组均在 task0 val 选得全局
    `alpha=0.03`，三 seed 全正后才运行 test；当前最强为 Full Base5 + Feature
    difference `31.4980±0.0723`。
16. 为“外部训练”与“DDP 内置训练”补一次严格数值等价审计：两路加载同一份
    Adapter `W`、同一份 cached DDP features 和同一个 `alpha`，检查 logits 最大误差
    与 mAP 完全一致。如需对训练轨迹做更强证明，两个训练器必须共用一份
    序列化初始 Adapter、feature cache 和 DataLoader 顺序。

## VOC 与工程复现

17. 补充 CODE_DDP semantic tau2 与 `multi-lane-main` official semantic tau2 的逐
    task/per-class 对齐分析，并检查 4-token semantic context 是否是 VOC 优势来源。
18. 如继续复现 VOC B4-C2，显式传入
    `--base_classes 4 --task_size 2 --total_classes 20`。
19. 所有实验保持独立 checkpoint/output 目录；启动前固定 `PYTHONHASHSEED`，并将 VOC
    的 set 结果排序后构建 dataset，确保同 seed 单变量复验可信。
