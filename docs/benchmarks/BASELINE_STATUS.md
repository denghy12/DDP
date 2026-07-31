# Baseline Status

Core v0.1 is frozen at `00f399f`. Statuses below include work on the first
baseline development line. `Not started` means no algorithm implementation or
approved external source port has occurred.

| Method | Paper audit | Unified interface | Unit tests | Smoke test | 3 seeds | Results summary | Current blocker |
|---|---|---|---|---|---|---|---|
| Joint Training | Not started | Reserved | Not started | Not started | Not started | Not started | Source/protocol audit |
| Sequential Fine-Tuning | Repository-native control specified | Trainable CLIP visual encoder + expanding linear heads; no common Adapter | Current branch: 51 Core/baseline tests and 17 legacy regressions passed; formal run used the preceding 47-test commit | Passed; worst-case suite covered by EWC/LwF smoke | Seeds 0--2 complete | Final mAP `18.1788 ± 0.8222`; Average mAP `23.8012 ± 0.4440` | None; preserve registered result |
| LwF | ECCV 2016 paper and author repository audited; adaptation differences documented | Old-model sigmoid distillation over the same trainable CLIP visual model | Current branch: 51 Core/baseline tests and 17 legacy regressions passed; formal run used the preceding 47-test commit | Passed; measured peak `3591.7 MiB` at batch 32 | Seeds 0--2 complete | Final mAP `23.3289 ± 1.3574`; Average mAP `28.6328 ± 1.1302` | None; preserve registered result |
| EWC | PNAS 2017 paper audited; supervised multi-label Fisher adaptation documented | Online diagonal Fisher over trainable visual encoder and existing heads; validation-only coefficient selection added | Current branch: 51 Core/baseline tests and 17 legacy regressions passed | Passed; measured peak `4871.7 MiB` at batch 32 | Seeds 0--2 at λ=100 retained as diagnostic; tuned rerun pending | λ=100 Final mAP `18.0109 ± 1.0736`; weighted penalty was effectively zero | Run validation-only logarithmic λ sweep, lock selection, then automatic 3-seed test rerun |
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
