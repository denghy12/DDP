#!/usr/bin/env python
"""Select an EWC coefficient using validation artifacts only."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


TRAINING_RECORD = re.compile(r"^task=(\d+) training=(\{.*\})$")


def lambda_slug(value: float) -> str:
    """Return the directory-safe canonical name used by the tmux launcher."""

    if not math.isfinite(value) or value <= 0:
        raise ValueError("EWC lambda candidates must be finite and positive")
    return format(value, ".12g").replace("+", "p").replace("-", "m").replace(
        ".", "d"
    )


def _single_path(paths: Iterable[Path], description: str) -> Path:
    values = list(paths)
    if len(values) != 1:
        raise RuntimeError(
            f"Expected one {description}, found {len(values)}: {values}"
        )
    return values[0]


def _load_candidate(tuning_root: Path, value: float, seed: int) -> Dict[str, Any]:
    candidate_root = tuning_root / f"lambda_{lambda_slug(value)}"
    summary_path = _single_path(
        candidate_root.glob(
            f"benchmarks/*/A/EWC/seed{seed}/metrics/summary.json"
        ),
        f"summary for EWC lambda {value:g}",
    )
    run_root = summary_path.parents[1]
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if summary.get("seed") != seed or manifest.get("seed") != seed:
        raise ValueError("Tuning artifact seed does not match requested seed")
    if manifest.get("method") != "EWC":
        raise ValueError("Tuning artifacts must come from EWC")
    if manifest.get("reporting_split") != "val":
        raise ValueError("EWC lambda selection must use reporting_split=val")
    if manifest.get("configuration_locked") is not False:
        raise ValueError("Validation tuning must not claim a locked configuration")
    if manifest.get("eligible_for_main_table") is not False:
        raise ValueError("Validation tuning must not be eligible for the main table")
    if manifest.get("test_labels_used_for_selection") is not False:
        raise ValueError("Test labels cannot be used for EWC lambda selection")

    actual_lambda = float(
        manifest.get("method_configuration", {}).get("ewc_lambda", math.nan)
    )
    if not math.isclose(actual_lambda, value, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(
            f"Manifest EWC lambda {actual_lambda} != candidate {value}"
        )

    main = summary.get("main_table", {})
    final_map = float(main.get("final_mAP", math.nan))
    if not math.isfinite(final_map):
        raise ValueError("Candidate final validation mAP must be finite")

    ratios: List[float] = []
    log_path = run_root / "train.log"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = TRAINING_RECORD.match(line)
        if not match or int(match.group(1)) == 0:
            continue
        record = json.loads(match.group(2))
        ratio = float(record.get("ewc_to_classification_ratio", math.nan))
        if math.isfinite(ratio):
            ratios.append(ratio)

    return {
        "ewc_lambda": value,
        "final_validation_mAP": final_map,
        "average_validation_mAP": float(main["average_mAP"]),
        "forgetting_on_validation": float(main["forgetting"]),
        "mean_ewc_to_classification_ratio": (
            sum(ratios) / len(ratios) if ratios else None
        ),
        "git_commit": manifest.get("git_commit"),
        "source_tree_hash": manifest.get("source_tree_hash"),
        "core_runtime_version": manifest.get("core_runtime_version"),
        "protocol_hash": manifest.get("protocol_hash"),
        "class_order_hash": manifest.get("class_order_hash"),
        "data_split_hash": manifest.get("data_split_hash"),
        "summary_path": str(summary_path.relative_to(tuning_root)),
    }


def select_lambda(
    tuning_root: Path,
    candidates: Iterable[float],
    seed: int,
) -> Dict[str, Any]:
    rows = [_load_candidate(tuning_root, float(value), seed) for value in candidates]
    if not rows:
        raise ValueError("At least one EWC lambda candidate is required")

    provenance_fields = (
        "git_commit",
        "source_tree_hash",
        "core_runtime_version",
        "protocol_hash",
        "class_order_hash",
        "data_split_hash",
    )
    for field in provenance_fields:
        canonical = json.dumps(rows[0][field], sort_keys=True)
        if any(json.dumps(row[field], sort_keys=True) != canonical for row in rows[1:]):
            raise ValueError(f"Candidate provenance differs for {field}")

    # Primary protocol metric first; a numerical tie chooses the smaller
    # coefficient to avoid unnecessary regularization.
    ranked = sorted(
        rows,
        key=lambda row: (-row["final_validation_mAP"], row["ewc_lambda"]),
    )
    selected = ranked[0]
    return {
        "selection_schema_version": 1,
        "method": "EWC",
        "tuning_seed": seed,
        "selection_split": "val",
        "selection_metric": "final_mAP",
        "selection_mode": "max",
        "tie_break": "lower_ewc_lambda",
        "test_metrics_used": False,
        "candidate_count": len(rows),
        "candidates": sorted(rows, key=lambda row: row["ewc_lambda"]),
        "selection_provenance": {
            field: rows[0][field]
            for field in provenance_fields
        },
        "selected_ewc_lambda": selected["ewc_lambda"],
        "selected_final_validation_mAP": selected["final_validation_mAP"],
        "configuration_locked_after_selection": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-root", required=True)
    parser.add_argument("--lambdas", nargs="+", required=True, type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    payload = select_lambda(
        Path(args.tuning_root).resolve(),
        args.lambdas,
        args.seed,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir / "selection.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
