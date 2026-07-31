# Baseline Porting Checklist

Complete one copy per method and track before a run is eligible for the main
table.

## Provenance

- [ ] Paper title/version and citation are recorded.
- [ ] Official repository URL, immutable upstream commit, and license are
      recorded.
- [ ] Local port commit and all modifications from upstream are auditable.
- [ ] Original dataset, class order, backbone, preprocessing, and core losses
      are recorded.
- [ ] The original paper's essential loss/optimization behavior is preserved or
      every deviation is explicitly justified.

## Protocol safety

- [ ] Training sample membership follows the loaded protocol.
- [ ] Future-class labels are never exposed to the method.
- [ ] Old-class ground truth is not used unless the method's explicitly audited
      replay contract permits retained labels.
- [ ] Training never reads evaluator-only `targets_seen`.
- [ ] Validation alone selects checkpoint/epoch and allowed hyperparameters.
- [ ] Test does not select epoch, temperature, alpha, beta, gate, threshold,
      replay policy, or any hyperparameter.
- [ ] The unified evaluator is used without method-specific metric changes.
- [ ] One fixed global threshold `0.5` is used for main-table cF1/oF1.
- [ ] No per-class or per-task threshold scan occurs.

## Reproducibility and outputs

- [ ] All random seeds are fixed and recorded.
- [ ] Resolved config, benchmark commit, upstream commit, protocol hash, class
      order hash, and data split hash are recorded.
- [ ] Standard JSON, PT, and HTML outputs are produced.
- [ ] Score rows, target rows, and ordered sample IDs have exact alignment.
- [ ] NaN/Inf and shape/hash checks pass.
- [ ] Unit tests and one method-specific smoke test pass.
- [ ] Three registered seeds complete before aggregate reporting.

## Fairness and resource accounting

- [ ] Track A uses the unified frozen CLIP ViT-B/16 backbone.
- [ ] Track B preserves and reports the audited original backbone.
- [ ] Replay sample count and replay byte count are both reported.
- [ ] Replay budget is compared under both sample and byte views.
- [ ] Total, optimizer-updated, and incremental parameter counts are reported;
      optimizer-updated tensors are deduplicated and must not be inferred only
      from `requires_grad=True`.
- [ ] Per-task parameter growth is recorded.
- [ ] Any task oracle, routing signal, or test-time side information is declared.
