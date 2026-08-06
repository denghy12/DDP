# EMOTIC DDP Transformer-block Task Adapter Bank

This branch tests whether the class-routed Adapter Bank becomes more effective
when its Adapters operate inside the frozen CLIP ViT instead of correcting the
final prompted CLS representation.

## Locked model

- The original B5-C3 DDP checkpoints are reused without retraining. Their main
  loss remains the original two-way BCE.
- Every incremental task owns one frozen Adapter after it has been trained.
- Each task Adapter contains nine bottleneck branches, inserted in ViT blocks
  4--12 (paper numbering, Python indices 3--11).
- A branch is parallel to the frozen block MLP:

  `x <- x + MLP(LN2(x)) + Up(ReLU(Down(LN2(x))))`.

- ViT-B/16 has token width 768. The first controlled experiment locks the
  bottleneck at 128, giving `768 -> 128 -> 768` in every selected block.
- The up projection is zero initialized, so a newly created Adapter preserves
  the exact DDP path before optimization.
- Both the positive and negative visual-prompt paths for class `c` use the
  Adapter owned by the task that introduced `c`. Routing uses class identity,
  never oracle sample/task identity.

P2L-CA reports the best placement as nine Adapters in layers 4--12 and uses
Adam, learning rate `4e-4`, cosine annealing, batch size 64, 20 epochs, and ASL.
The paper does not state a bottleneck width or exact ASL gamma values in its
implementation-details table. Therefore width 128 and the already audited
EMOTIC ASL setting (`gamma_neg=9.8`, `gamma_pos=0`, `clip=0.05`) are explicit
project choices, not attributed to P2L-CA.

## Strict incremental supervision

For task `t`, only train persons with at least one positive label in task `t`.
The loss mask exposes only current-task labels. Old and future labels attached
to the same multi-label person remain invisible. The two data regimes are:

- `full`: all eligible training persons;
- `16shot`: the existing deterministic multi-label 16-shot sampler.

Adapter classification uses ASL on the DDP positive-minus-negative margin.
Validation is reporting-only. Every checkpoint is the fixed last epoch. Test
evaluation uses the fixed threshold 0.5 and cannot select epochs, thresholds,
layers, routing, or per-class gates.

## Main commands

Server synchronization check:

```bash
bash scripts/emotic-ddp-transformer-adapter-bank/verify_sync.sh
```

Optional low-cost first gate (paper configuration, Full Task 0, seed 0 only):

```bash
GPU=0 TRAINING_MODE=full SEED=0 TASKS="0" bash \
  scripts/emotic-ddp-transformer-adapter-bank/run_train.sh
```

This writes the same Task-0 checkpoint consumed by the complete launcher, so
passing the gate does not duplicate training.

Six independent jobs (`full/16shot x seeds 0/1/2`) use GPUs 0--5 by default:

```bash
GPU_LIST="0 1 2 3 4 5 6 7" bash \
  scripts/emotic-ddp-transformer-adapter-bank/launch_8gpu_tmux.sh
```

The two unused GPUs are intentionally left free because tasks within one seed
must wait for that seed's Task-0 initialization anchor.

## Outputs to synchronize

```text
output/emotic_ddp_transformer_adapter_bank_asl_full/
output/emotic_ddp_transformer_adapter_bank_asl_16shot/
output/emotic_ddp_transformer_adapter_bank_asl_full_seed{0,1,2}_evaluation/
output/emotic_ddp_transformer_adapter_bank_asl_16shot_seed{0,1,2}_evaluation/
output/emotic_ddp_transformer_adapter_bank_comparison/
output/emotic_ddp_transformer_adapter_bank_pipeline/
```

The formal result files are the two bank manifests per seed, eight task score
files per evaluation, `evaluation_summary.{json,html}`, and
`comparison_summary.{json,html}` plus the CSV tables.
