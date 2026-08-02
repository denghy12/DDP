# EMOTIC MLCIL protocol configurations

`protocol_b5c3.yaml` is the frozen Core v0.1 protocol. The loader validates
configuration content and never infers a protocol from the filename.

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

The only registered tuning override is runner option `--ewc-lambda`. It is
kept outside the frozen protocol object, validated as finite and positive, and
written into method configuration. The official tuning launcher permits this
option only during validation candidates, records the selection decision, and
then reuses the locked value for formal test runs.
