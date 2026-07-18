# EMOTIC B5-C3 Task-routed Adapter Bank

This experiment maintains two independent frozen banks:

- exact 16-shot: `A0^16 ... A7^16`;
- all current-task persons: `A0^Full ... A7^Full`.

For class `c`, inference deterministically selects `A_tau(c)`, where `tau(c)`
is the class-introduction task. A multi-label sample can therefore use several
Adapters in one forward pass without an oracle sample-level task ID.

Training uses the DDP-owned frozen prompt-free CLIP CLS and only current-task
positive/negative text prototypes. The supervision mask is one only for the
current class block. Old/future labels never enter BCE. Full mode keeps every
unique train person with a current positive; 16-shot mode supervises exactly
16 positive anchors per current class and masks unselected positive co-labels.

At inference, the prompt-free auxiliary head is discarded. Each prompted CLS
path receives the fixed Feature Difference correction

```text
delta_f(c,+/-) = 0.03 * W2_tau(c) GELU(W1_tau(c) normalize(f_cls(c,+/-)))
delta_l(c,+/-) = 100 * text(c,+/-)^T delta_f(c,+/-)
l_final(c,+/-) = l_DDP(c,+/-) + delta_l(c,+/-)
```

The DDP path order is `[all negative classes, all positive classes]`. Routing
therefore applies the same task Adapter to indices `[low:high]` and
`[K+low:K+high]`.

## Run

```bash
GPU0=0 GPU1=1 bash \
  scripts/emotic-ddp-internal-adapter/launch_emotic_ddp_task_adapter_bank_tmux.sh
```

GPU0 runs Full and GPU1 runs 16-shot. Both reuse the locked prompt-free feature
cache; cache creation is file-locked and atomic. Existing complete task/seed
outputs are validated and skipped.

Main artifacts:

- `output/emotic_b5c3_task_adapter_audit/`;
- `output/emotic_ddp_task_adapter_bank_{full,16shot}/seed*/`;
- `output/emotic_ddp_task_adapter_bank_{full,16shot}_feature_difference_seed*/`;
- `output/emotic_ddp_task_adapter_bank_comparison/`.

Training constructs only train and val datasets and never accesses test labels.
Test is accessed by the final bank evaluator after the architecture, Feature
Difference formula, and global alpha are fixed.
