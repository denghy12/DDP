# TODO_NEXT

1. 在服务器运行 task7 离线 Prototype/DDP 校准融合，核对 target alignment、val 选择的 beta 与 test 增益。
2. 若全局 beta 融合有效，再扩展到逐 task 评估；若无效，先分析逐类 AP 互补性，不直接引入 soft teacher。
3. 保留 All26-balanced 作为监督上限、Base5-balanced 作为增量安全主版本、Base5 普通 BCE 作为排序消融。
4. 在离线融合证明确有收益前，不修改 DDP 训练目标。
5. 基于 CODE_DDP semantic + tau2 20ep 最强结果，补充与 `multi-lane-main` official semantic tau2 的逐 task/per-class 对齐分析。
6. 检查 CODE_DDP semantic prompt context 长度变为 4 是否是优势来源；如需严谨对照，可新增 16-token semantic seed 初始化。
7. 若继续优化 CODE_DDP，优先做 semantic tau2 的 threshold sweep / per-class AP-F1 诊断，确认 0.8 阈值是否仍最优。
8. 若需要继续复现旧 VOC B4-C2，运行时显式传入 `--base_classes 4 --task_size 2 --total_classes 20`。
9. 为每组实验增加独立 checkpoint 输出目录，避免并行实验覆盖。
10. 在启动命令中固定 `PYTHONHASHSEED`，并将 VOC `set` 结果排序后再构建 dataset，
   才能进行可信的同 seed 单变量复验。
