# EMOTIC B5-C3 Benchmark Contract

Status: frozen for Benchmark Core v0.1.

## Dataset and splits

The dataset is EMOTIC using `CVPR17_Annotations.mat` and the existing
`src.helper_functions.emotic_loader.EMOTIC` implementation.

- training split: `train`
- validation split: `val`
- test split: `test`

Validation is used for checkpoint selection and any pre-registered global
configuration selection. Test is held out until the configuration,
checkpoint-selection rule, seed list, temperature policy, and all method
hyperparameters are locked. The historical combined `val+test` route is a
legacy diagnostic and is not a benchmark split.

## Class order and tasks

The class order is the alphabetical order produced by the existing EMOTIC
loader:

| ID | Class | Task |
|---:|---|---:|
| 0 | Affection | 0 |
| 1 | Anger | 0 |
| 2 | Annoyance | 0 |
| 3 | Anticipation | 0 |
| 4 | Aversion | 0 |
| 5 | Confidence | 1 |
| 6 | Disapproval | 1 |
| 7 | Disconnection | 1 |
| 8 | Disquietment | 2 |
| 9 | Doubt/Confusion | 2 |
| 10 | Embarrassment | 2 |
| 11 | Engagement | 3 |
| 12 | Esteem | 3 |
| 13 | Excitement | 3 |
| 14 | Fatigue | 4 |
| 15 | Fear | 4 |
| 16 | Happiness | 4 |
| 17 | Pain | 5 |
| 18 | Peace | 5 |
| 19 | Pleasure | 5 |
| 20 | Sadness | 6 |
| 21 | Sensitivity | 6 |
| 22 | Suffering | 6 |
| 23 | Surprise | 7 |
| 24 | Sympathy | 7 |
| 25 | Yearning | 7 |

Thus Task 0 is IDs `[0,5)`, and Tasks 1–7 are `[5,8)`, `[8,11)`,
`[11,14)`, `[14,17)`, `[17,20)`, `[20,23)`, and `[23,26)`.

This agrees with the existing loader's sorted class discovery, the legacy DDP
B5-C3 stages, and `emotic_task_adapter_bank.TASK_CLASS_RANGES`. The values live
in `configs/emotic_mlcil/protocol_b5c3.yaml`; runner, data module, and evaluator
must derive them from the loaded protocol rather than repeat them.

## Training membership and label visibility

For task \(t\), let \(C_t\) be the current classes, \(C_{\le t}\) the seen
classes, and \(C_{>t}\) the future classes.

A training person-sample enters task \(t\) exactly when its full annotation has
at least one positive label in \(C_t\). Membership is computed inside the data
boundary before labels are hidden.

The method-facing training batch contains only:

- image;
- stable sample ID;
- `targets_current`, the target columns for \(C_t\);
- `visible_mask`, a protocol mask identifying \(C_t\).

Current-class labels, including current-class negatives, are visible. Old-class
truth and future-class truth are not present in a normal training or selection
batch. Their values must not be inferable from padding, sentinels, or a retained
full target tensor. Replay methods may retain only information permitted by
their audited memory contract; this does not grant access to hidden truth.

The complete `targets_seen` tensor is available only from evaluator-designated
validation/test loaders. It contains exactly \(C_{\le t}\), never future
columns. Methods must not read evaluator-only targets during optimization.

## Validation and test scope

At task \(t\), validation/test include every person-sample with at least one
positive annotation in \(C_{\le t}\). Scores and targets contain exactly the
seen columns in protocol order.

Validation may select the checkpoint using validation mAP according to the
pre-registered tie rule: higher mAP, then earlier epoch. It must not select a
model using test results. Test is run once after the configuration is locked.

## Identity and alignment

A sample ID is derived without label content as:

`emotic:<split>:<relative-image-path>:person=<zero-based-person-ordinal>`

The ordinal follows annotation/dataset order within an image. For any scored
split:

1. sample IDs must be unique;
2. score row \(i\), target row \(i\), and sample ID \(i\) refer to the same
   person-sample;
3. an expected-ID sequence, when supplied, must match exactly, not as a set;
4. scores and targets must both be `[N, C_seen]`;
5. class-order hash and split hash must match the protocol and loader metadata.

Scores are positive-class probabilities in `[0,1]`, not raw logits. Targets are
binary values. NaN/Inf, an empty evaluation subset, duplicate/empty IDs, or any
out-of-range probability invalidates the run.

The split hash is SHA-256 over the ordered split name and ordered sample IDs.
The class-order hash is SHA-256 over the canonical JSON class-name list.

Legacy score files without sample IDs cannot establish identity equivalence.
Their limitation must be reported. DDP equivalence instead uses one common
deterministic dataloader for both inference paths and verifies IDs and targets.

## Metrics

All reported values are percentages in `[0,100]`. Let \(y_{ic}\in\{0,1\}\),
score \(s_{ic}\), fixed threshold \(\tau=0.5\), and
\(\hat y_{ic}=\mathbb{1}[s_{ic}>\tau]\).

Per-class AP and mAP directly call the repository's legacy
`evaluation_metrics.mAP`/`average_precision`: positives are sorted by descending
score and precision is averaged at positive ranks using the existing
`1e-8` denominator epsilon. A class with no positives has AP 0. This preserves
legacy DDP values exactly instead of introducing a near-equivalent rewrite.

`evaluation_metrics.prf_cal` is not reused because it hardcodes threshold
`0.8`, which conflicts with this frozen contract. Its TP/FP/FN semantics were
checked against `detail_report.binary_counts`; the benchmark reuses the latter
with the protocol's global threshold `0.5`.

\[
\mathrm{mAP}_t = \frac{100}{|C_{\le t}|}
 \sum_{c\in C_{\le t}}\mathrm{AP}_{t,c}.
\]

For each class, precision and recall use zero when their denominator is zero,
and \(F1_c=2P_cR_c/(P_c+R_c)\), also zero for a zero denominator.

\[
\mathrm{cPrecision}=100\,\mathrm{mean}_c(P_c),\quad
\mathrm{cRecall}=100\,\mathrm{mean}_c(R_c),\quad
\mathrm{cF1}=100\,\mathrm{mean}_c(F1_c).
\]

Overall precision/recall pool TP, FP, and FN over all evaluated samples and
seen classes:

\[
\mathrm{oPrecision}=100\frac{\sum TP}{\sum(TP+FP)},\quad
\mathrm{oRecall}=100\frac{\sum TP}{\sum(TP+FN)},\quad
\mathrm{oF1}=\frac{2\,\mathrm{oPrecision}\,\mathrm{oRecall}}
{\mathrm{oPrecision}+\mathrm{oRecall}}.
\]

For \(T\) evaluated tasks:

\[
\mathrm{Average\ mAP}=\frac{1}{T}\sum_{t=0}^{T-1}\mathrm{mAP}_t,\qquad
\mathrm{Final\ mAP}=\mathrm{mAP}_{T-1}.
\]

If \(a_{t,c}=100\,\mathrm{AP}_{t,c}\) and \(i(c)\) is the introduction task,
per-class forgetting at the final task is:

\[
F_c=\max_{t\in[i(c),T-1]}a_{t,c}-a_{T-1,c}.
\]

Overall Forgetting is the arithmetic mean of \(F_c\) over classes introduced
before the final task. It is 0 when no such class exists. Negative forgetting
is impossible by construction because the final value is included in the max.

## Threshold and selection policy

The primary metric is threshold-independent mAP. Main-table cF1/oF1 use one
fixed global threshold, exactly `0.5`, for every task, class, method, and seed.
No validation threshold scan is performed. Per-class or per-task threshold
selection is forbidden.

Test labels or metrics must never select a model, epoch, temperature, alpha,
beta, gate, threshold, replay policy, or any other hyperparameter. A run
manifest must state `test_labels_used_for_selection=false`; the artifact writer
rejects any other value.

## Main-table fields

`Method`, `Type`, `Replay Memory`, `Backbone`, `Final mAP ↑`, `Final cF1 ↑`,
`Final oF1 ↑`, `Avg. mAP ↑`, `Forgetting ↓`, and `Parameter Growth`.
