# Replay Memory Contract v0.1

This contract fixes the memory and training exposure used by controlled
replay baselines in EMOTIC B5-C3 Track A. It exists because a replay method's
reported accuracy is not comparable unless its capacity, payload, update
timing, and replay ratio are all fixed. The machine-readable authority is
`configs/emotic_mlcil/replay_20c_v0.1.yaml`.

## Registered budget

The primary replay budget is **20 unique EMOTIC person-samples per seen
class**. It is a class-normalized expanding budget, not a fraction of the
training set and not each method's original paper budget.

| Task | Seen classes | Maximum unique samples |
|---:|---:|---:|
| 0 | 5 | 100 |
| 1 | 8 | 160 |
| 2 | 11 | 220 |
| 3 | 14 | 280 |
| 4 | 17 | 340 |
| 5 | 20 | 400 |
| 6 | 23 | 460 |
| 7 | 26 | 520 |

The maximum grows when new classes become visible and never shrinks. A method
may temporarily retain fewer samples because stable sample-ID deduplication or
its audited admission policy rejected items. The artifact must report both the
actual retained count and actual payload bytes after every task; the nominal
capacity alone is insufficient.

The `20/class` choice is shared with the registered KRT memory scale, but it
does not make the policies identical. KRT retains its audited herding policy,
ER uses conventional reservoir replacement, and PRS uses partition-aware
replacement. Their results must therefore name both the policy and budget.

## Storage unit and label visibility

One memory unit is one unique benchmark `sample_id`, corresponding to an
EMOTIC person-sample. Duplicate IDs are merged rather than charged twice.
Memory stores:

- the post-transform contiguous float32 image tensor;
- the stable sample ID;
- binary targets only for columns visible when that observation entered the
  stream; and
- a boolean mask identifying those visible columns.

Hidden columns are physically zeroed and excluded from replay loss. A sample
seen again in a later task may add only the newly visible columns; conflicting
values in an already visible column are rejected. This prevents replay from
becoming an indirect path to old-class or future-class ground truth that the
protocol-safe `TrainBatch` did not expose.

`replay_memory_bytes` is the exact serialized payload view: image, target,
mask, and UTF-8 sample-ID bytes. It deliberately excludes Python container,
allocator, and checkpoint-format overhead. Checkpoints stay server-side and
the standard download package still excludes every `.pth` file.

## Training exposure

The registered lifecycle is task-based:

1. Train the current task using the buffer frozen at the preceding task end.
2. For each current mini-batch, draw replay without replacement up to a 1:1
   replay/current sample ratio. The draw is capped only when memory is smaller
   than the current batch.
3. Give every current and replay sample equal weight. Replay BCE is masked to
   that record's stored visible columns.
4. Select the epoch only by current validation mAP.
5. Restore the selected checkpoint, then make one deterministic pass over the
   current task's training loader to update memory.

Task 0 has no replay because no prior memory exists. The update pass must never
read validation or test. ER and PRS share this lifecycle, optimizer, visual
model, classifier, batch sizes, label visibility, and random-seed convention;
only their admission/eviction policy differs.

This task-end mapping is a deliberate Track-A adaptation of PRS's original
online stream, which updates memory after every optimizer step. It keeps the
benchmark's task-level B5-C3 training contract consistent and is recorded in
every resolved method configuration.

## Policy registrations

- **ER:** repository-native control using uniform reservoir replacement. It
  provides the policy-neutral reference under the shared budget.
- **PRS:** independent implementation of Partitioning Reservoir Sampling.
  Positive-label stream frequencies define target partitions with the
  official COCO allocation power `q=-0.03`; admission and eviction follow the
  fixed upstream operators.

No method may silently substitute a paper-specific memory size in the primary
table. Original-budget runs, including PRS's released COCO capacity of 2,000,
belong in a separately labelled diagnostic table and may not replace the
shared-budget result.

## Eligibility checks

A replay result is eligible only when all of the following hold:

- the YAML contract and B5-C3 protocol IDs match;
- the task capacity schedule is exact;
- IDs are unique and retained samples never exceed capacity;
- hidden target values are zero and replay loss uses the visible mask;
- the replay/current ratio and equal-sample weighting are recorded;
- memory state and RNG state survive checkpoint round trips;
- retained sample count and bytes appear in standard artifacts; and
- held-out test is accessed only with `configuration_locked=true`.
