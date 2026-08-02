# Benchmark Download Package Standard

## Purpose

Every EMOTIC MLCIL training or evaluation run must finish with one small,
verifiable, checkpoint-free download package. Result transfer must not require
recursively downloading a run directory whose nested `checkpoints/` folders
contain large `.pth` files.

This standard applies to every method, track, split, seed, tuning run, smoke
result retained as evidence, and future baseline added to the benchmark.

## Server-side separation

Training artifacts remain under the canonical method directory:

```text
<run-root>/benchmarks/<protocol>/<track>/<method>/seed<seed>/
├── checkpoints/                         # server only; never in download package
├── scores/                              # canonical .pt score tensors
├── metrics/
├── config_resolved.json
├── run_manifest.json
├── report.html
├── train.log
└── results_to_sync/<run-id>/            # checkpoint-free method bundle
```

The universal packager publishes a physically separate download surface:

```text
<run-root>/
├── download_ready/<run-id>/
│   ├── bundles/<method>/seed<seed>/
│   ├── logs/
│   ├── extras/                          # optional preflight/coordinator evidence
│   ├── DOWNLOAD_INFO.txt
│   └── download_manifest.json
└── download_packages/
    ├── <run-id>.tar.gz                  # download this
    └── <run-id>.tar.gz.sha256           # download this too
```

Checkpoint files remain recoverable on the server in their original method
directories. This standard does not delete or move them.

## Required package contents

Each seed/method bundle must contain:

- `config_resolved.json`;
- `run_manifest.json`;
- `metrics/task_metrics.json`;
- `metrics/summary.json`;
- `report.html`;
- `train.log`;
- every canonical `scores/task<N>_scores.pt` file;
- `sync_manifest.json` with byte counts and SHA-256 for every exported file.

The package additionally includes every available launcher/preflight log and
explicitly requested coordinator metadata. Canonical `.pt` score tensors are
required: they support metric recomputation, target/sample-ID alignment audit,
and integrity checks, and are not model checkpoints.

## Mandatory exclusions and validation

- Every file whose suffix is `.pth` is forbidden at any nesting depth.
- Symlinks are forbidden, preventing a package from indirectly referencing a
  checkpoint or file outside the run.
- The exact expected bundle count is required; a missing seed or method blocks
  packaging.
- Every inner `sync_manifest.json` hash and byte count is revalidated.
- The score-file count must equal the protocol task count.
- The assembled directory and final tar archive are scanned again for `.pth`.
- The final archive receives an adjacent SHA-256 checksum file.

Other large training state, optimizer state, caches, raw datasets, pretrained
models, and source repositories are not download results and must remain
outside the package. If checkpoints are intentionally requested later, create
a separate, explicitly named checkpoint-only package; never weaken this
result-package standard.

## Universal command

Run after every expected `results_to_sync/<run-id>/` bundle is complete:

```bash
python scripts/emotic-mlcil/package_benchmark_download.py \
  --run-root <absolute-run-root> \
  --run-id <run-id> \
  --expected-bundles <exact-count>
```

Optional evidence can be added with repeated `--launcher-log <file>` and
`--extra <file-or-directory>` arguments. The packager automatically discovers
`<run-root-parent>/_launcher_logs/<run-id>.log` and
`<run-id>_preflight.log` when present.

For a single-seed CSC validation run:

```bash
python scripts/emotic-mlcil/package_benchmark_download.py \
  --run-root "/mnt/haoyuan/workspace/emotic_benchmark_runs/csc_track_a_v0.1/${RUN_ID}" \
  --run-id "${RUN_ID}" \
  --expected-bundles 1
```

Download only the printed `.tar.gz` and `.tar.gz.sha256` paths. On the server,
verify before transfer with:

```bash
cd <run-root>/download_packages
sha256sum -c <run-id>.tar.gz.sha256
tar -tzf <run-id>.tar.gz | grep -Ei '\.pth$'
```

The checksum command must report `OK`; the archive scan must print nothing.
After transfer to macOS, verify with:

```bash
shasum -a 256 <run-id>.tar.gz
```

and compare it with the first field in `<run-id>.tar.gz.sha256`.

## Launcher requirement

Every new method launcher/coordinator must:

1. create canonical benchmark artifacts;
2. call `--export-sync-results` for every completed bundle;
3. wait for the exact expected seed/method count;
4. invoke the universal packager only after successful completion;
5. print the archive and checksum paths;
6. leave checkpoints in the server-only canonical directory.

Packaging failure does not convert a failed or incomplete run into a result.
Fix the missing artifact/log and rerun only the packaging step; do not retrain
when the canonical run itself is already complete.
