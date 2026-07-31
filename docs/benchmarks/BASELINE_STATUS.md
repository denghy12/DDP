# Baseline Status

Statuses describe Benchmark Core v0.1 only. `Not started` means no algorithm
implementation or external source import has occurred.

| Method | Paper audit | Unified interface | Unit tests | Smoke test | 3 seeds | Results summary | Current blocker |
|---|---|---|---|---|---|---|---|
| Joint Training | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit |
| Sequential Fine-Tuning | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit |
| LwF | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit |
| EWC | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit |
| ER | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit and fair memory budget |
| DER++ | Not started | Reserved | Not started | Not started | Not started | Not started | Multi-label source/protocol audit |
| PRS | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit |
| OCDM | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit |
| KRT | Not started | Reserved | Not started | Not started | Not started | Not started | Official source/license audit |
| CSC | Not started | Reserved | Not started | Not started | Not started | Not started | Official source/license audit |
| MULTI-LANE | Not started | Reserved | Not started | Not started | Not started | Not started | Official source/license audit |
| L3A | Not started | Reserved | Not started | Not started | Not started | Not started | Official source/license audit |
| AGCN | Not started | Reserved | Not started | Not started | Not started | Not started | Static-to-incremental design after source audit |
| EmoGrowth/AESL | Not started | Reserved | Not started | Not started | Not started | Not started | Identity/source audit |
| DDP | Local implementation noted; external audit pending | Core v0.1 wrapper | 36 Core tests and 17 selected legacy regressions passed | Strict task-7 pure-test equivalence passed: 5,368 samples, score max error 0, targets/IDs/AP/mAP identical; eight-task artifact smoke passed | Existing legacy runs only | Smoke: Final mAP 30.8051, Average mAP 37.7374, Forgetting 4.8990; uncommitted/reused-prediction run is not main-table eligible | Freeze Core, then run clean formal seeds; external paper/source/license audit remains |
| Task-routed Adapter Bank | Local implementation noted; external audit pending | Reserved | Existing repository tests | Existing server smoke | Existing work, not benchmark-frozen | Existing work, not benchmark-frozen | Port through core after Core v0.1 freeze |
