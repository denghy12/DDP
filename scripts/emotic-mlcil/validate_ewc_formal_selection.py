#!/usr/bin/env python
"""Validate that formal EWC runs use the locked validation selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _single_path(paths: Iterable[Path], description: str) -> Path:
    values = list(paths)
    if len(values) != 1:
        raise RuntimeError(
            f"Expected one {description}, found {len(values)}: {values}"
        )
    return values[0]


def validate_formal_runs(
    selection_path: Path,
    formal_root: Path,
    seeds: Iterable[int],
) -> Dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selection_split") != "val":
        raise ValueError("EWC selection must have been made on validation")
    if selection.get("test_metrics_used") is not False:
        raise ValueError("EWC selection cannot use test metrics")
    selected_lambda = float(selection["selected_ewc_lambda"])
    provenance = selection.get("selection_provenance", {})
    expected_seeds = [int(seed) for seed in seeds]
    if not expected_seeds or len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("Formal seeds must be a non-empty unique list")

    records: List[Dict[str, Any]] = []
    formal_data_split_hash: Any = None
    for seed in expected_seeds:
        manifest_path = _single_path(
            formal_root.glob(
                f"benchmarks/*/A/EWC/seed{seed}/run_manifest.json"
            ),
            f"formal EWC seed {seed} manifest",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "method": "EWC",
            "seed": seed,
            "reporting_split": "test",
            "configuration_locked": True,
            "eligible_for_main_table": True,
            "test_labels_used_for_selection": False,
        }
        for key, expected in required.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"Formal seed {seed} manifest {key}={manifest.get(key)!r}; "
                    f"expected {expected!r}"
                )
        if manifest.get("prediction_reused_task_ids") != []:
            raise ValueError("Formal EWC runs cannot reuse predictions")
        actual_lambda = float(
            manifest.get("method_configuration", {}).get(
                "ewc_lambda", math.nan
            )
        )
        if not math.isclose(
            actual_lambda,
            selected_lambda,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"Formal seed {seed} lambda {actual_lambda} != locked "
                f"{selected_lambda}"
            )
        for key in (
            "git_commit",
            "source_tree_hash",
            "core_runtime_version",
            "class_order_hash",
        ):
            if manifest.get(key) != provenance.get(key):
                raise ValueError(
                    f"Formal seed {seed} provenance differs for {key}"
                )
        if formal_data_split_hash is None:
            formal_data_split_hash = manifest.get("data_split_hash")
        elif manifest.get("data_split_hash") != formal_data_split_hash:
            raise ValueError(
                f"Formal seed {seed} test data split hashes differ"
            )
        if seed == int(selection["tuning_seed"]):
            if manifest.get("protocol_hash") != provenance.get("protocol_hash"):
                raise ValueError(
                    "Formal tuning-seed protocol hash differs from selection"
                )
        records.append(
            {
                "seed": seed,
                "ewc_lambda": actual_lambda,
                "eligible_for_main_table": True,
                "manifest_path": str(manifest_path.relative_to(formal_root)),
            }
        )

    return {
        "formal_validation_schema_version": 1,
        "method": "EWC",
        "selected_ewc_lambda": selected_lambda,
        "validated_seeds": expected_seeds,
        "all_eligible_for_main_table": True,
        "test_metrics_used_for_selection": False,
        "formal_data_split_hash": formal_data_split_hash,
        "runs": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_formal_runs(
        Path(args.selection).resolve(),
        Path(args.formal_root).resolve(),
        args.seeds,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
