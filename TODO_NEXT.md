# TODO_NEXT

1. 在服务器运行全八任务的二值类别门控融合，核对每个任务的 target alignment、平均 mAP 与 old-class forgetting。
2. 将全局 beta、二值类别门控和 DDP-only 的逐任务结果并列；连续逐类 beta 仅保留为高自由度诊断，不作为主结果。
3. 保留 All26-balanced 作为监督上限、Base5-balanced 作为增量安全主版本、Base5 普通 BCE 作为排序消融。
4. 在离线融合证明确有收益前，不修改 DDP 训练目标。
5. 基于 CODE_DDP semantic + tau2 20ep 最强结果，补充与 `multi-lane-main` official semantic tau2 的逐 task/per-class 对齐分析。
6. 检查 CODE_DDP semantic prompt context 长度变为 4 是否是优势来源；如需严谨对照，可新增 16-token semantic seed 初始化。
7. 若继续优化 CODE_DDP，优先做 semantic tau2 的 threshold sweep / per-class AP-F1 诊断，确认 0.8 阈值是否仍最优。
8. 若需要继续复现旧 VOC B4-C2，运行时显式传入 `--base_classes 4 --task_size 2 --total_classes 20`。
9. 为每组实验增加独立 checkpoint 输出目录，避免并行实验覆盖。
10. 在启动命令中固定 `PYTHONHASHSEED`，并将 VOC `set` 结果排序后再构建 dataset，
   才能进行可信的同 seed 单变量复验。
