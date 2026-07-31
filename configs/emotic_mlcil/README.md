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
baseline's frozen architecture, optimizer, LwF, and EWC settings to
`config_resolved.json` and `run_manifest.json` as method configuration.
