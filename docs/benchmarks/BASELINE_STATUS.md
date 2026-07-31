# Baseline Status

Core v0.1 is frozen at `00f399f`. Statuses below include work on the first
baseline development line. `Not started` means no algorithm implementation or
approved external source port has occurred.

| Method | Paper audit | Unified interface | Unit tests | Smoke test | 3 seeds | Results summary | Current blocker |
|---|---|---|---|---|---|---|---|
| Joint Training | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit |
| Sequential Fine-Tuning | Repository-native control specified | Implemented on shared frozen-CLIP classifier | 47 Core/baseline tests passed locally; server validation pending | Pending | Not started | Pending | Run server tests, then one seed-0 smoke |
| LwF | ECCV 2016 paper and author repository audited; adaptation differences documented | Implemented with old-model sigmoid distillation | 47 Core/baseline tests passed locally; server validation pending | Pending | Not started | Pending | Run server tests, then one seed-0 smoke |
| EWC | PNAS 2017 paper audited; supervised multi-label Fisher adaptation documented | Implemented with online diagonal Fisher | 47 Core/baseline tests passed locally; server validation pending | Pending | Not started | Pending | Run server tests, then one seed-0 smoke |
| ER | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit and fair memory budget |
| DER++ | Not started | Reserved | Not started | Not started | Not started | Not started | Multi-label source/protocol audit |
| PRS | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit |
| OCDM | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit |
| KRT | Not started | Reserved | Not started | Not started | Not started | Not started | Official source/license audit |
| CSC | Not started | Reserved | Not started | Not started | Not started | Not started | Official source/license audit |
| MULTI-LANE | Not started | Reserved | Not started | Not started | Not started | Not started | Official source/license audit |
| L3A | Not started | Reserved | Not started | Not started | Not started | Not started | Official source/license audit |
| AGCN | Local snapshot identifies native lifelong multi-label method; immutable source audit pending | Reserved | Not started | Not started | Not started | Not started | Official fixed commit/license audit and Track-A design |
| EmoGrowth/AESL | Not started | Reserved | Not started | Not started | Not started | Not started | Identity/source audit |
| DDP | Local implementation noted; external audit pending | Core v0.1 wrapper | 36 Core tests and 17 selected legacy regressions passed | Strict task-7 pure-test equivalence passed: 5,368 samples, score max error 0, targets/IDs/AP/mAP identical; clean eight-task formal run passed | Seed 0 complete; seeds 1--2 pending | Eligible seed-0 result at `00f399f`: Final mAP 30.8051, Average mAP 37.7374, Forgetting 4.8990 | Run remaining registered seeds; external paper/source/license audit remains |
| Task-routed Adapter Bank | Local implementation noted; external audit pending | Reserved | Existing repository tests | Existing server smoke | Existing work, not benchmark-frozen | Existing work, not benchmark-frozen | Port through core after Core v0.1 freeze |
