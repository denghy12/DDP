# EMOTIC MLCIL protocol configurations

`protocol_b5c3.yaml` is the frozen Core v0.1 protocol. The loader validates
configuration content and never infers a protocol from the filename.

`protocol_b10c4.yaml` is the registered five-task extension over the identical
alphabetical 26-class order. Task sizes are `10/4/4/4/4`, seen-class counts are
`10/14/18/22/26`, and all label-visibility, validation-selection, held-out test,
and fixed-threshold rules are unchanged. It is a distinct protocol with its own
hash and must never reuse B5-C3 predictions. The adjacent
`replay_b10c4_20c_v0.1.yaml` and
`replay_derpp_b10c4_20c_v0.1.yaml` retain 20 samples per seen class with task
capacities `200/280/360/440/520`.

`protocol_b4c2.yaml` is the registered twelve-task extension over that same
alphabetical order. Task sizes are `4/2/2/2/2/2/2/2/2/2/2/2`, seen-class
counts are `4/6/8/10/12/14/16/18/20/22/24/26`, and all visibility,
selection, held-out test, and fixed-threshold rules remain unchanged. Its
standard and DER++ replay contracts retain 20 samples per seen class with
capacities `80/120/160/200/240/280/320/360/400/440/480/520`.

To add B10-C2, B2-C2, or a custom class order, add another YAML file with the
same schema and change `class_order` plus `tasks`. Every class must occur exactly
once, and flattening `tasks` must equal `class_order`. No runner, evaluator, or
data-module edit should be necessary.

Required top-level keys are:

`protocol_id`, `dataset`, `class_order`, `tasks`, `train_split`,
`validation_split`, `test_split`, `label_visibility`, `evaluation_scope`,
`threshold_policy`, `primary_metric`, `auxiliary_metrics`,
`checkpoint_selection`, `track`, `seed`, and `output_schema_version`.

The main-table policy is intentionally strict: threshold-independent mAP and a
single global threshold of `0.5` for cF1/oF1. Test-based selection and
per-task/per-class threshold scans are rejected during protocol validation.

`method_options` is part of the protocol hash. Core v0.1 records only the
already-frozen DDP temperature schedule there. New baseline settings must not be
added to this frozen protocol file because doing so would invalidate comparison
with the registered DDP protocol hash. The runner instead writes each
baseline's registered architecture, optimizer, LwF, EWC, and KRT settings to
`config_resolved.json` and `run_manifest.json` as method configuration.

KRT likewise uses immutable code defaults rather than adding fields to the
frozen YAML. Its manifest records the fixed upstream commit/archive hash,
CLIP-token substitution, DPL/ICA/token-loss settings, 20-exemplar-per-class
herding policy, pseudo-density source, and actual replay sample/byte counts.

CSC also uses audited immutable code defaults outside the frozen YAML. Its
manifest records the fixed official commit/archive hash, CLIP patch-token
substitution, dynamic CI-GCN expansion, current-label/old-model loss weights,
max-entropy scope, paper-reported Adam/OneCycle settings, and explicit zero
replay memory. The absence of Adapter and CLIP text features is recorded.

MULTI-LANE follows the same immutable-default rule. Its manifest must record
the fixed official commit/archive hash, frozen CLIP-block substitution,
preallocated task selectors and K/V prompts, concat inference, current-class
BCE mask and full-width released reduction, zero replay memory, and the source
VOC optimizer/schedule mapping. The registered train batch is 64, yielding
effective Adam learning rate `0.0125` from source rule `0.05 * batch / 256`.

L3A also uses immutable audited defaults outside the frozen YAML. Its manifest
records the fixed official commit/tree/archive, Task-0-only CLIP visual
fine-tuning with ASL, the official ViT hidden width/ridge mapping, random ReLU
expansion, float64 cumulative analytic state, inverse-square-root class
weighting, threshold-0.7 pseudo labels, and zero replay. No L3A field is added
to `protocol_b5c3.yaml`, so the frozen dataset/protocol hash remains unchanged.

Original-DDP-Tau2 likewise keeps immutable method defaults outside the frozen
YAML. Its manifest records both the collected source PCD (`T=1→7, γ=0.2`) and
the deliberately registered, user-requested comparison mapping
(`T=1→2, γ=0.7`). The latter changes method inference only; it does not alter
the shared dataset protocol hash.

AGCN follows the same rule: its GCN dimensions, one-epoch optimizer, ACM
thresholds/scales, and paper-resolved `0.07 / 0.93 / 1e5` loss weights are
immutable code defaults. The runner option `--agcn-word-embeddings` identifies
an audited 26-by-300 GloVe JSON prepared by
`prepare_agcn_glove_embeddings.py`. This is a method asset, not a protocol
field; AGCN does not use CLIP text features. The registered mapping averages
`doubt` and `confusion` and uses same-root `disquiet` because `disquietment` is
absent from the fixed GloVe 6B vocabulary.

Replay methods use the separate machine-readable
`replay_20c_v0.1.yaml` contract. It fixes a 20-sample-per-seen-class capacity
schedule, visible-column-only payload, stable-ID deduplication, task-end update,
and 1:1 replay/current exposure. The replay file is validated against the
protocol ID but is not inserted into the frozen Core protocol object, so the
registered B5-C3 protocol hash remains unchanged. ER and PRS record the full
resolved replay contract and actual sample/byte usage in their method
artifacts.

DER++ uses the adjacent `replay_derpp_20c_v0.1.yaml`. It keeps the identical
20-per-seen-class capacity schedule but registers the method-defining online
pre-update logit payload, two independent 1:1 replay draws, and source-style
weighted objective instead of ER/PRS task-end insertion. Logits and a logit
mask are included in actual replay byte accounting. These method details do
not change the frozen B5-C3 protocol hash.

The only registered tuning override is runner option `--ewc-lambda`. It is
kept outside the frozen protocol object, validated as finite and positive, and
written into method configuration. The official tuning launcher permits this
option only during validation candidates, records the selection decision, and
then reuses the locked value for formal test runs.
