# EMOTIC experiment code map

This map groups the experiment code by the branch where each line of work
logically belongs. Python modules are intentionally kept in the repository root
unless moving them would be low-risk; many scripts and notebooks refer to these
files directly.

## `emotic`

Baseline DDP / EMOTIC protocol.

### Training and evaluation entry points

- `main.py`
- `DDP.py`
- `eval_emotic_all_tasks.py`
- `eval_emotic_threshold_sweep.py`
- `eval_emotic_upper_bound.py`
- `select_emotic_global_threshold.py`

### Model/config support

- `models/ddp.py`
- `models/model_builder.py`
- `configs/models/vitb16_ep50.yaml`
- `configs/datasets/emotic.yaml`

### Shell scripts

- `scripts/emotic/run_emotic_b5c3_semantic_tau2.sh`
- `scripts/emotic/run_emotic_upper_bound_semantic_threshold050.sh`

### Output index

- `output/by_branch/emotic/`

## `emotic-prototype-adapter`

External vanilla-CLIP Prototype Adapter and score fusion. This line uses a
separate global CLIP image feature, learns/evaluates a residual adapter, and then
fuses the prototype score with DDP scores.

### Adapter / few-shot training

- `prototype_adapter.py`
- `prototype_fewshot.py`
- `train_emotic_prototype_adapter.py`
- `summarize_emotic_prototype_fewshot.py`

### DDP + prototype score fusion

- `eval_emotic_prototype_fusion.py`
- `eval_emotic_prototype_fusion_all_tasks.py`
- `summarize_emotic_prototype_fusion_sweep.py`
- `export_emotic_external_fusion_html.py`

### Tests

- `tests/test_prototype_adapter.py`
- `tests/test_prototype_fewshot.py`
- `tests/test_prototype_fusion.py`
- `tests/test_prototype_fusion_all_tasks.py`

### Shell scripts

- `scripts/emotic-prototype-adapter/`

### Output index

- `output/by_branch/emotic-prototype-adapter/`

## `emotic-ddp-internal-adapter`

DDP-internal shared Feature Adapter / CLS transfer / class gate / final ablation.
This line keeps DDP as the main model and inserts or transfers a small shared
adapter inside DDP feature/logit flow.

### Internal adapter model and training/evaluation

- `ddp_internal_adapter.py`
- `train_emotic_ddp_internal_adapter.py`
- `train_emotic_ddp_prompt_free_auxiliary.py`
- `train_emotic_ddp_task_adapter.py`
- `emotic_task_adapter_bank.py`
- `build_emotic_ddp_task_adapter_bank.py`
- `eval_emotic_ddp_internal_adapter.py`
- `eval_emotic_ddp_task_adapter_bank.py`
- `audit_emotic_task_adapter_data.py`
- `smoke_emotic_ddp_internal_adapter.py`
- `smoke_emotic_ddp_prompt_free_equivalence.py`
- `screen_emotic_ddp_internal_adapter_transfer.py`
- `summarize_emotic_ddp_internal_adapter.py`
- `summarize_emotic_ddp_internal_adapter_screen.py`
- `summarize_emotic_ddp_task_adapter_bank.py`
- `summarize_emotic_ddp_cls_full_base5_comparison.py`

### CLS internal branch and final diagnostics

- `cache_emotic_ddp_cls_features.py`
- `eval_emotic_ddp_cls_internal_gate.py`
- `eval_emotic_ddp_cls_external_hybrid.py`
- `summarize_emotic_adapter_ablation.py`
- `benchmark_emotic_adapter_inference.py`
- `summarize_emotic_adapter_benchmark.py`

### Tests

- `tests/test_ddp_internal_adapter.py`
- `tests/test_ddp_prompt_free_auxiliary.py`
- `tests/test_ddp_internal_transfer.py`
- `tests/test_ddp_cosine_difference.py`
- `tests/test_ddp_cls_internal_gate.py`
- `tests/test_ddp_cls_external_hybrid.py`

### Shell scripts

- `scripts/emotic-ddp-internal-adapter/`

### Output index

- `output/by_branch/emotic-ddp-internal-adapter/`

## Regenerating the output index

After syncing new server results back to this local workspace:

```bash
python tools/organize_emotic_artifacts.py
```

This updates:

- `output/MANIFEST.md`
- `output/by_branch/emotic/`
- `output/by_branch/emotic-prototype-adapter/`
- `output/by_branch/emotic-ddp-internal-adapter/`
