"""Tests for generic 12-method B4-C2 formal aggregation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "summarize_b4c2_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_b4c2_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class B4C2SweepSummaryTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _build_bundle(
        self,
        root: Path,
        run_id: str,
        commit: str,
        method_key: str,
        display_name: str,
        seed: int,
    ) -> None:
        target = (
            root
            / "benchmarks"
            / "emotic_b4c2_v0.1"
            / "A"
            / display_name
            / f"seed{seed}"
        )
        class_order = [f"Class{index:02d}" for index in range(26)]
        tasks = [class_order[:4]] + [
            class_order[start : start + 2] for start in range(4, 26, 2)
        ]
        replay_samples = (
            520 if method_key in {"er", "prs", "derpp"} else
            510 if method_key == "krt" else 0
        )
        self._write_json(
            target / "config_resolved.json",
            {"protocol": {"class_order": class_order, "tasks": tasks}},
        )
        self._write_json(
            target / "run_manifest.json",
            {
                "method": display_name,
                "protocol_id": "emotic_b4c2_v0.1",
                "track": "A",
                "seed": seed,
                "git_commit": commit,
                "git_dirty": False,
                "source_tree_hash": "tree",
                "class_order_hash": "classes",
                "protocol_hash": f"protocol-seed-{seed}",
                "reporting_split": "test",
                "configuration_locked": True,
                "test_labels_used_for_selection": False,
                "eligible_for_main_table": True,
                "prediction_reused_task_ids": [],
            },
        )
        offset = float(seed)
        task_rows = [
            {"task_id": task, "mAP": 10.0 + task + offset}
            for task in range(12)
        ]
        main_table = {
            "final_mAP": 20.0 + offset,
            "final_cF1": 21.0 + offset,
            "final_oF1": 22.0 + offset,
            "average_mAP": 23.0 + offset,
            "forgetting": 4.0 + offset,
            "replay_memory": {
                "samples": replay_samples,
                "bytes": replay_samples * 100,
            },
            "parameter_growth": 10,
        }
        self._write_json(
            target / "metrics" / "summary.json",
            {"main_table": main_table, "summary": {"task_metrics": task_rows}},
        )
        self._write_json(
            target / "metrics" / "task_metrics.json",
            {"tasks": task_rows},
        )
        for task in range(12):
            score = target / "scores" / f"task{task}_scores.pt"
            score.parent.mkdir(parents=True, exist_ok=True)
            score.write_bytes(b"score")
        self._write_json(
            target / "results_to_sync" / run_id / "sync_manifest.json",
            {"contains_pth": False},
        )

    def test_all_36_bundles_aggregate_with_sample_std(self) -> None:
        run_id = "b4c2-test"
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method_key, display_name, _memory in MODULE.METHODS:
                for seed in MODULE.SEEDS:
                    self._build_bundle(
                        root,
                        run_id,
                        commit,
                        method_key,
                        display_name,
                        seed,
                    )
            output = root / "summary.json"
            markdown = root / "summary.md"
            argv = [
                str(SCRIPT),
                "--run-root", str(root),
                "--run-id", run_id,
                "--expected-git-commit", commit,
                "--output", str(output),
                "--markdown-output", str(markdown),
            ]
            with mock.patch.object(sys, "argv", argv):
                MODULE.main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["bundle_count"], 36)
            self.assertEqual(len(payload["methods"]), 12)
            self.assertEqual(
                payload["methods"][0]["aggregate"]["final_mAP"],
                {"mean": 21.0, "std": 1.0},
            )
            self.assertIn("Original-DDP-Tau2", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
