# EMOTIC experiment scripts

Shell launchers are grouped by the branch where the experiment logically belongs.
All scripts first `cd` to the repository root, so they can be launched either from
the root or by absolute path.

## `scripts/emotic/`

Baseline EMOTIC / DDP experiments:

- `run_emotic_b5c3_semantic_tau2.sh`: strict B5-C3 DDP training.
- `run_emotic_upper_bound_semantic_threshold050.sh`: joint 26-class upper-bound DDP training.

## `scripts/emotic-prototype-adapter/`

External vanilla-CLIP Prototype Adapter experiments:

- `run_emotic_prototype_adapter_all26.sh`
- `run_emotic_prototype_adapter_base5.sh`
- `run_emotic_prototype_adapter_fewshot_all26.sh`
- `run_emotic_prototype_adapter_fewshot_base5.sh`
- `run_emotic_prototype_fusion_task7.sh`
- `run_emotic_prototype_fusion_all_tasks.sh`
- `run_emotic_prototype_fusion_fewshot_sweep.sh`
- `run_emotic_prototype_fusion_zero_shot_strict.sh`
- `run_emotic_prototype_full_seed_pipeline.sh`
- `launch_emotic_prototype_fusion_strict_tmux.sh`

## `scripts/emotic-ddp-internal-adapter/`

DDP-internal shared Feature Adapter / CLS gate / final ablation experiments:

- `run_emotic_ddp_internal_adapter_16shot.sh`
- `run_emotic_ddp_internal_adapter_screen.sh`
- `run_emotic_ddp_internal_transfer_then_full.sh`
- `run_emotic_ddp_cls_internal_transfer.sh`
- `run_emotic_ddp_cls_internal_gate.sh`
- `run_emotic_adapter_final_analysis.sh`
- `launch_emotic_ddp_internal_adapter_tmux.sh`
- `launch_emotic_ddp_internal_adapter_screen_tmux.sh`
- `launch_emotic_ddp_internal_transfer_then_full_tmux.sh`
- `launch_emotic_ddp_cls_internal_transfer_tmux.sh`
- `launch_emotic_ddp_cls_internal_gate_tmux.sh`
- `launch_emotic_adapter_final_analysis_tmux.sh`

## Output index

Experiment outputs still live at their original `output/emotic_*` paths to avoid
breaking old evaluation code. For a clean branch-oriented view, regenerate:

```bash
python tools/organize_emotic_artifacts.py
```

Then browse:

- `output/MANIFEST.md`
- `output/by_branch/emotic/`
- `output/by_branch/emotic-prototype-adapter/`
- `output/by_branch/emotic-ddp-internal-adapter/`
