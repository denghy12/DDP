# EMOTIC B5-C3 Final-token Adapter Bank

## Locked method

For every DDP class-specific negative/positive visual path, the frozen CLIP
ViT-B/16 returns 197 projected tokens.  One task-routed Adapter is applied
point-wise with shared weights:

\[
\mathbf r_n^{c\pm}=W_{2,t}\,\mathrm{GELU}
(W_{1,t}\,\mathrm{norm}(\mathbf f_n^{c\pm})),
\]

\[
\widetilde{\mathbf f}_n^{c\pm}=\mathrm{norm}
(\mathbf f_n^{c\pm}+0.03\,\mathbf r_n^{c\pm}).
\]

The original DDP equations are then recomputed from the adapted tokens:

\[
\widetilde q_n^{c\pm}=20(\mathbf t^{c\pm})^\top
\widetilde{\mathbf f}_n^{c\pm},
\qquad
\widetilde\omega_n^c=\mathrm{softmax}_n(\widetilde q_{n,\mathrm{ref}}^c),
\]

\[
\widetilde\ell_{\mathrm{DDP}}^{c\pm}
=5\sum_n\widetilde\omega_n^c\widetilde q_n^{c\pm}.
\]

There is no feature/logit correction head, class gate, score fusion, or
validation-selected alpha.  Every class is routed to the Adapter trained when
that class was introduced, and both +/- paths share that Adapter.

## Strict training protocol

- Task `t` uses frozen `checkpoints/taskt.pth`, not task0 for every task.
- Only current-task labels are supervised; old/future labels are masked.
- Full and exact 16-shot banks are trained independently for seeds 0/1/2.
- Adapter architecture is 512→128→512 with zero-initialized up projection.
- Training and inference use the same fixed residual scale `0.03`.
- Checkpoint is the fixed last epoch; validation is reporting-only.
- Test data are never constructed by the training program.
- Evaluation uses the fixed decision threshold `0.5` for cF1/oF1.
- mAP, average mAP, final mAP, current-task mAP, and forgetting are reported.

## Parallel execution

The launcher assigns Full seeds to GPU0 and 16-shot seeds to GPU1.  Within a
mode, all three banks share each frozen DDP token extraction during evaluation,
so the costly visual encoder is run once rather than once per seed.

```bash
GPU0=0 GPU1=1 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_adapter_bank_tmux.sh
```

Outputs:

- `output/emotic_ddp_final_token_adapter_bank_{full,16shot}/seed*/`
- `output/emotic_ddp_final_token_adapter_bank_{full,16shot}_seed*/`
- `output/emotic_ddp_final_token_adapter_bank_comparison/`

## Training-health smoke test

Before rerunning the bank, run the Task-0 Full seed-0 health audit.  It checks
that the zero-initialized up projection receives a gradient on the first
optimizer step, that the residual becomes non-zero, and that the down
projection receives a gradient after the up projection changes.  The default
run trains one complete epoch and uses validation only for reporting.

```bash
GPU=0 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_training_healthcheck_tmux.sh
```

Reports are written to:

- `output/emotic_ddp_final_token_training_healthcheck_full_task0_seed0_v2/training_health_check.json`
- `output/emotic_ddp_final_token_training_healthcheck_full_task0_seed0_v2/training_health_check.html`
- `output/emotic_ddp_final_token_training_healthcheck_full_task0_seed0_v2/training_summary.json`

### Locked safe-strength screen

After the gradient audit passes, three Task-0 Full seed-0 configurations test
global training strength without test data or per-class selection:

- A: learning rate `1e-4`, 100 optimizer steps;
- B: learning rate `1e-4`, 300 optimizer steps;
- C: learning rate `3e-5`, 300 optimizer steps.

Configs A and C run sequentially on GPU0 while B runs on GPU1.  The summary
reports all three configurations and deliberately does not select a winner.

```bash
GPU0=0 GPU1=1 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_safety_screen_tmux.sh
```

The combined reports are:

- `output/emotic_ddp_final_token_safety_screen_summary/safety_screen_summary.json`
- `output/emotic_ddp_final_token_safety_screen_summary/safety_screen_summary.html`

### Locked C300 Task-0 three-seed gate

The final low-cost decision run fixes learning rate `3e-5`, 300 optimizer
steps, Full Task-0 data, alpha `0.03`, and identity weight `0.1` for seeds
0/1/2.  Seeds 0 and 2 run sequentially on GPU0; seed 1 runs on GPU1.

Validation evaluation additionally reports, without adding them to the loss:

- `KL(original attention || adapted attention)`;
- pooled-feature cosine drift;
- pre-softmax DDP path-logit drift;
- separate CLS and patch-token L2 drift;
- all-token L2 p95 and maximum.

The pre-registered gate is: continue to a full Task0-7 Bank only when the
three-seed mean Task-0 validation mAP gain is strictly positive.  The script
never launches the full Bank automatically.

```bash
GPU0=0 GPU1=1 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_c300_multiseed_tmux.sh
```

Outputs:

- `output/emotic_ddp_final_token_c300_multiseed_seed{0,1,2}/`
- `output/emotic_ddp_final_token_c300_multiseed_summary/`
- `output/emotic_ddp_final_token_c300_multiseed_pipeline/`

### Pooling-aware regularization pilot

The first controlled pilot keeps the C300 seed-0 configuration unchanged and
adds three differentiable teacher-preservation terms from the frozen original
DDP prompted-token route:

\[
L = L_{\mathrm{BCE}} + 0.1L_{\mathrm{token-id}}
  + 100L_{\mathrm{pool}} + 100L_{\mathrm{attention}}
  + L_{\mathrm{margin}}.
\]

Here `pool` is pooled-feature cosine drift, `attention` is
`KL(original || adapted)` on positive-path attention, and `margin` is Smooth
L1 on the change of `positive path logit - negative path logit`.  The weights
are scale matched from the prior C300 diagnostics and are locked before this
pilot.  No test split is constructed and no full Bank is launched.

```bash
GPU=0 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_pooling_aware_task0_tmux.sh
```

Outputs:

- `output/emotic_ddp_final_token_pooling_aware_task0_seed0/training_summary.json`
- `output/emotic_ddp_final_token_pooling_aware_task0_seed0/pooling_aware_pilot_summary.json`
- `output/emotic_ddp_final_token_pooling_aware_task0_seed0/pooling_aware_pilot_summary.html`

The training cache is FP16 and sharded because a single Full token cache can
be many gigabytes.  Full caches are shared across seeds; 16-shot caches remain
seed-specific because their sampled persons differ.

### Complete pooling-aware Full Bank

For the requested complete comparison, the pilot configuration is frozen and
extended to Task 0–7 with Full task-native training data and seeds 0/1/2.  This
is a confirmatory run despite the negative Task-0 pilot; no hyperparameter is
changed from the pilot except for the mathematically equivalent numerically
stable log-softmax implementation of attention KL.  All Task-0 runs are
retrained under that implementation for seed consistency.  Adapter training
uses FP32 because inherited non-zero task anchors can overflow the first AMP
gradient-scaling steps even when every FP32 loss term is finite.  GPU0 runs
seeds 0 and 2, GPU1 runs seed 1, and the three banks share the expensive DDP
test feature pass.

```bash
GPU0=0 GPU1=1 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_final_token_pooling_aware_bank_tmux.sh
```

The comparison contains DDP, the previous unregularized Full Final-token Bank,
and the new pooling-aware Full Bank:

- `output/emotic_ddp_final_token_pooling_aware_bank_full/seed*/`
- `output/emotic_ddp_final_token_pooling_aware_bank_full_seed*/`
- `output/emotic_ddp_final_token_pooling_aware_bank_comparison/comparison_summary.json`
- `output/emotic_ddp_final_token_pooling_aware_bank_comparison/comparison_summary.html`
