# TODO_NEXT

1. 基于 CODE_DDP semantic + tau2 20ep 最强结果，补充与 `multi-lane-main` official semantic tau2 的逐 task/per-class 对齐分析。
2. 检查 CODE_DDP semantic prompt context 长度变为 4 是否是优势来源；如需严谨对照，可新增 16-token semantic seed 初始化。
3. 若继续优化 CODE_DDP，优先做 semantic tau2 的 threshold sweep / per-class AP-F1 诊断，确认 0.8 阈值是否仍最优。
4. 若需要继续复现旧 VOC B4-C2，运行时显式传入 `--base_classes 4 --task_size 2 --total_classes 20`。
5. 为每组实验增加独立 checkpoint 输出目录，避免并行实验覆盖。
6. 在启动命令中固定 `PYTHONHASHSEED`，并将 VOC `set` 结果排序后再构建 dataset，
   才能进行可信的同 seed 单变量复验。
