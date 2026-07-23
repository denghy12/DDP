# Prompt-free CLS Adapter weights at the DDP final-token stage

## Question

The Task-routed Adapter Bank was trained from the frozen, prompt-free CLIP
global CLS feature. This experiment asks whether those existing weights can be
used before original DDP token-attention pooling, without training a new
Final-token Adapter.

The source is the existing three-seed Full Task Adapter Bank:

```text
output/emotic_ddp_task_adapter_bank_full/seed{0,1,2}/
```

Each class is routed to the Adapter learned when that class was introduced.
Negative and positive DDP paths for one class therefore share the same frozen
Adapter.

## Locked comparison

All four methods share the same prompted DDP visual encoding for each batch:

1. Original DDP.
2. Original CLS Feature Difference.
3. Apply the CLS-trained Adapter point-wise to all 197 projected final tokens,
   then run the exact original DDP token-attention pooling.
4. Apply the CLS-trained Adapter only to final token 0 (CLS); all patch tokens
   remain bitwise unchanged, then run the exact original DDP pooling.

For a token \(\mathbf f_n^{c\pm}\), the transferred mapping is

\[
\widetilde{\mathbf f}_n^{c\pm}
=
\operatorname{norm}
\left(
\mathbf f_n^{c\pm}
+\alpha W_{2,t(c)}
\sigma\!\left(W_{1,t(c)}
\operatorname{norm}(\mathbf f_n^{c\pm})\right)
\right),
\qquad \alpha=0.03 .
\]

The all-token route applies this expression for \(n=0,\ldots,196\). The
CLS-only route applies it only for \(n=0\). Original DDP then recomputes
token–text logits, positive-path attention, shared attention pooling, and
positive/negative path logits.

## Experimental controls

- Full source Bank, seeds 0, 1, and 2.
- Tasks 0–7 under EMOTIC B5-C3.
- No Adapter training or fine-tuning.
- No class gate, task-specific scale, or score fusion.
- Fixed residual scale \(\alpha=0.03\).
- Fixed decision threshold 0.5.
- Validation is reporting-only; validation and test do not choose a method,
  checkpoint, scale, gate, or threshold.
- GPU0 evaluates seeds 0 and 2; GPU1 evaluates seed 1.

Alongside mAP, cF1, oF1, current-task performance, and forgetting, the
evaluation records attention KL divergence, pooled-feature cosine drift,
absolute and signed path-logit drift, and CLS/patch token delta norms.

## Outputs

Per seed:

```text
output/emotic_ddp_cls_to_final_token_transfer_seed{0,1,2}/
```

Three-seed comparison:

```text
output/emotic_ddp_cls_to_final_token_transfer_comparison/
```

Execution logs and completion marker:

```text
output/emotic_ddp_cls_to_final_token_transfer_pipeline/
```
