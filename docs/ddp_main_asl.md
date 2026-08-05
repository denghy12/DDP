# DDP main-loss ASL experiment

This branch changes the classification loss used to train the original DDP
text/visual prompts. It does **not** change the Adapter auxiliary loss, DDP
architecture, inference graph, class order, or visible-label protocol.

For every sample and current-task class, DDP returns the ordered path-logit
pair

\[
\boldsymbol\ell_{ic}=(\ell^-_{ic},\ell^+_{ic}).
\]

The ASL binary logit is the exact two-way softmax margin

\[
d_{ic}=\ell^+_{ic}-\ell^-_{ic},\qquad p_{ic}=\sigma(d_{ic}).
\]

With shifted negative probability

\[
p^-_{m,ic}=\min(1,1-p_{ic}+m),
\]

the primary loss is

\[
\mathcal L_{\mathrm{ASL}}
=-\sum_{i,c\in\mathcal C^t}
\left[
y_{ic}(1-p_{ic})^{\gamma_+}\log p_{ic}
+(1-y_{ic})(1-p^-_{m,ic})^{\gamma_-}\log p^-_{m,ic}
\right].
\]

The locked primary configuration is

- `gamma_pos = 0.0`
- `gamma_neg = 9.8`
- `clip = 0.05`
- `eps = 1e-8`
- sum reduction
- outer DDP `loss_w = 0.03`
- fixed-last checkpoint after 30 epochs per task
- training-time evaluation uses `val` only and is reporting-only
- `test` is evaluated only after all fixed-last checkpoints are saved
- fixed evaluation threshold 0.50 and the existing Tau2 schedule

Only labels in the current class interval `labels[:, low_range:high_range]`
enter the main loss. Old and future labels remain invisible.

The controlled full comparison is

\[
\{\text{two-way BCE},\text{ASL}\}\times\{0,1,2\}\text{ seeds}.
\]

Each run writes `ddp_main_loss_protocol.json`,
`training_diagnostics.json`, fixed-last Task 0--7 checkpoints, validation and
test JSON/HTML reports. The comparison summarizer produces aggregate,
per-task, per-class, and head/middle/tail CSV files.

After synchronization, validate the server copy with

```bash
bash scripts/emotic-ddp-main-asl/verify_emotic_ddp_main_asl_sync.sh
```

Run the Task-0 health check first:

```bash
GPU_BCE=0 GPU_ASL=1 bash \
  scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_loss_healthcheck_tmux.sh
```

The corrected health-check outputs use the `_v2` suffix and record the first
finite **unscaled optimizer-step** prompt gradient, rather than inspecting AMP
scaled gradients before `GradScaler.unscale_`.

The primary `gamma_neg=9.8` health check suppresses negative supervision too
strongly for DDP prompt training. Before any full Task 0--7 run, the follow-up
experiment therefore pre-registers this Task-0-only matrix:

| Method | Main loss | `gamma_neg` | Seeds |
|---|---|---:|---|
| `two_way_bce` | original two-way BCE | -- | 0, 1, 2 |
| `asl_g9p8` | ASL negative reference | 9.8 | 0, 1, 2 |
| `asl_g4` | mild ASL candidate | 4.0 | 0, 1, 2 |
| `asl_g2` | mild ASL candidate | 2.0 | 0, 1, 2 |

All runs use 30 fixed epochs, fixed-last checkpoints, threshold 0.50, the same
`loss_w=0.03`, and validation only. Test data are not evaluated. Corrected
seed-0 BCE and ASL-9.8 health-check artifacts are reused, so ten new Task-0
runs remain. The full-experiment gate is fixed before seeing the new results:

\[
\Delta\mathrm{mAP}\geq -0.1,\qquad
\Delta\mathrm{cF1}\geq -2.0,\qquad
\Delta\mathrm{oF1}\geq -2.0,
\]

where every delta is the three-seed paired mean relative to BCE. Candidate
priority is fixed as `asl_g4`, then `asl_g2`; the summarizer does not select the
largest observed validation score.

With only GPU0 available and enough memory for two training processes, launch
two logical lanes on that device:

```bash
GPU_LIST="0 0" bash \
  scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_asl_task0_ablation_tmux.sh
```

The final JSON, CSV, and HTML report is written to
`output/emotic_ddp_main_asl_task0_ablation_summary/`.

## ASL-2 training-scale alignment

The mild-ASL screen showed that `gamma_neg=2` restores fixed-threshold F1 but
has an initial prompt-gradient norm of only about 31.8% of BCE under the same
outer loss weight. A final Task-0 diagnostic therefore keeps the ASL shape
fixed and changes only

\[
\texttt{loss\_w}: 0.03 \rightarrow 0.09.
\]

The value is the rounded training-gradient ratio, not a value selected from
validation or test performance. The experiment compares three paired seeds of
BCE, unscaled ASL-2, and scaled ASL-2. It remains validation-only, fixed-last,
and uses threshold 0.50. Launch two logical lanes on GPU0 with:

```bash
GPU_LIST="0 0" bash \
  scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_asl_g2_scale_task0_tmux.sh
```

Results are written to
`output/emotic_ddp_main_asl_g2_scale_task0_summary/`. The same pre-registered
mAP/cF1/oF1 gate determines whether a full Task 0--7 experiment is justified.

Only after the health report is accepted, launch the controlled six-run
comparison:

```bash
GPU_LIST="0 1 2 3 4 5" bash \
  scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_losses_6gpu_tmux.sh
```

When only GPU0 is available but two DDP processes fit in memory, use two
logical lanes on the same device. Each lane runs three experiments
sequentially, so there are never more than two simultaneous training jobs:

```bash
GPU_LIST="0 0" bash \
  scripts/emotic-ddp-main-asl/launch_emotic_ddp_main_losses_6gpu_tmux.sh
```
