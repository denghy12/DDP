# Baseline Source Audit

Core v0.1 creates the audit surface only. External source snapshots are kept
outside this repository under
`/Users/denghaoyuan/workspace/MyCode/baseline_sources/`; none has been copied
into the benchmark package. Unverified bibliographic, repository, license, and
algorithm details are deliberately marked `TBD`.

## Local source collection status

The snapshots below were inspected on 2026-07-31. None contains a `.git`
directory, so none provides an immutable upstream commit or verifiable remote.
They are read-only audit inputs, not yet approved porting sources.

| Method | Local snapshot | Git provenance | License file observed in snapshot | Required next action |
|---|---|---|---|---|
| AGCN | `AGCN-main/` | Unavailable | Top-level Apache-2.0 text | Re-clone official repository and verify license at a fixed commit |
| CSC | `CSC-main/` | Unavailable | None found | Identify official repository, fixed commit, and license |
| EmoGrowth | `EmoGrowth-master/` | Unavailable | None found | Resolve method identity, official repository, fixed commit, and license |
| KRT | `KRT-MLCIL-main/` | Unavailable | Nested MIT text credited to Alibaba-MIIL | Verify that the license covers KRT additions at the official fixed commit |
| L3A | `L3A-main/` | Unavailable | None found | Identify official fixed commit and license |
| MULTI-LANE | `multi-lane-main/` | Unavailable | Top-level CC BY-NC 4.0 text | Verify official commit and record attribution/non-commercial constraints |

Independent authoritative snapshots for PRS, OCDM, and DER++ have not been
collected. Implementations found inside another baseline repository remain
secondary references until their original paper and official source are
audited.

| Method | Type | Paper | Official repository | Fixed upstream commit | License | Original dataset | Original backbone | Core loss | Replay memory | Parameters grow by task? | Track A adaptation | Track B adaptation | Unverified questions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Joint Training | Upper bound | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Use frozen CLIP ViT-B/16; all classes jointly visible | Preserve audited implementation | Exact training/data convention |
| Sequential Fine-Tuning | Lower bound | Repository-native control; no claimed source paper | N/A | N/A | Repository code license | EMOTIC | Frozen OpenAI CLIP ViT-B/16 | Current-label BCE; shared adapter and new task head update | None | Linear head block per task after Task 0 | Implemented shared frozen-CLIP classifier | N/A | Formal smoke and 3-seed results |
| LwF | Distillation | [Learning without Forgetting, ECCV 2016](https://arxiv.org/abs/1606.09282) | [Author repository](https://github.com/lizhitwo/LearningWithoutForgetting) | `795176d49017663ae76455e6536b4c2b7caf35fe` (`release`) | Repository-specific BSD-like research license; source is reference-only | Original: ImageNet/Places2/PASCAL VOC/MIT67/CUB/MNIST; adaptation: EMOTIC | Original: AlexNet/VGG via MatConvNet; adaptation: frozen CLIP ViT-B/16 | Current-label BCE plus temperature-scaled old-model distillation | None | Linear head block per task after Task 0 | Implemented independent sigmoid distillation on old classes | Preserve audited original only in Track B | Multi-label sigmoid replaces original task softmax; no original code copied |
| EWC | Regularization | [Overcoming catastrophic forgetting in neural networks, PNAS 2017](https://doi.org/10.1073/pnas.1611835114) | No implementation vendored; paper-driven repository-native adaptation | N/A | Repository code license applies to this implementation | Original: random patterns/MNIST/Atari; adaptation: EMOTIC | Original: experiment-specific networks; adaptation: frozen CLIP ViT-B/16 | Current-label BCE plus online diagonal-Fisher quadratic penalty | None | Linear head block per task after Task 0 | Implemented current-task empirical Fisher on Track-A trainables | Preserve audited original only in Track B | Multi-label BCE Fisher, full current-task cache, decay 1.0, coefficient 100 |
| ER | Replay | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Replay with frozen CLIP and fixed budgets | Preserve audited backbone | Buffer policy and source |
| DER++ | Replay/distillation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Unified CLIP plus audited replay logits | Preserve audited backbone | Official multi-label variant |
| PRS | Replay | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Unified CLIP with audited sampling | Preserve audited backbone | Official source and budget units |
| OCDM | Replay | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Unified CLIP with audited memory | Preserve audited backbone | Official source and label access |
| KRT | Native MLCIL | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Port core loss to frozen CLIP | Preserve audited backbone | Source, license, protocol |
| CSC | Native MLCIL | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Port core loss to frozen CLIP | Preserve audited backbone | Source, license, protocol |
| MULTI-LANE | Native MLCIL | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Port core loss to frozen CLIP | Preserve audited backbone | Source, license, protocol |
| L3A | Native MLCIL | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Port core loss to frozen CLIP | Preserve audited backbone | Source, license, protocol |
| AGCN | Native lifelong multi-label | AGCN: Augmented Graph Convolutional Network for Lifelong Multi-label Image Recognition (ICME 2022; local README cites arXiv:2203.05534) | Local snapshot remote not verifiable | TBD | Apache-2.0 text observed; verify at fixed commit | Split-COCO and Split-NUS | TBD | Augmented correlation matrix, graph convolution, relationship-preserving loss | TBD | TBD | Port graph/loss to frozen CLIP after audit | Preserve audited original | Official fixed commit, exact backbone, buffer behavior, and label access |
| EmoGrowth/AESL | Static/incremental EMOTIC | TBD | TBD | TBD | TBD | EMOTIC (TBD) | TBD | TBD | TBD | TBD | Incremental wrapper on frozen CLIP | Preserve audited backbone | Naming, source, and method relation |
| DDP | Native MLCIL | TBD | TBD | Local benchmark base: `f9459d0`; upstream TBD | TBD | Repository-local runs: VOC/COCO/EMOTIC; paper scope TBD | Repository-local EMOTIC path: CLIP ViT-B/16; paper backbone TBD | Repository-local dual-decoupled prompting objective; paper audit TBD | Repository-local EMOTIC path: none; paper audit TBD | Repository-local model preallocates configured prompts; paper audit TBD | Existing EMOTIC DDP wrapper | TBD | Upstream paper/repository/commit/license audit |
| Task-routed Adapter Bank | Adapter | TBD | TBD | Local benchmark base: `f9459d0`; upstream TBD | TBD | Repository-local run: EMOTIC; upstream status TBD | Repository-local path: frozen DDP-owned CLIP ViT-B/16 | Repository-local task Adapter objective; upstream status TBD | Repository-local path: none | Repository-local path: one Adapter per task | Existing class-routed Adapter Bank | TBD | Formal upstream status, provenance, and final 3-seed results |

Before any row leaves `TBD`, record the paper version, official repository URL,
immutable commit, repository license at that commit, and the exact source files
used for the port.
