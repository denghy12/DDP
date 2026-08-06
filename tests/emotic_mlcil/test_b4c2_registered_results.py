"""Integrity checks for the frozen B4-C2 12-baseline result snapshot."""

from __future__ import annotations

import json
import statistics
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = (
    ROOT
    / "docs"
    / "benchmarks"
    / "results"
    / "b4c2_12baseline_seed012_formal_v0.1.json"
)
PROTOCOL_DOC = ROOT / "docs" / "benchmarks" / "EMOTIC_B4C2_12BASELINE_SWEEP.md"
STATUS_DOC = ROOT / "docs" / "benchmarks" / "BASELINE_STATUS.md"
BENCHMARK_README = ROOT / "docs" / "benchmarks" / "README.md"


class B4C2RegisteredResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    def test_protocol_and_execution_are_frozen(self) -> None:
        benchmark = self.payload["benchmark"]
        execution = self.payload["execution"]
        self.assertEqual(self.payload["status"], "frozen")
        self.assertEqual(benchmark["protocol_id"], "emotic_b4c2_v0.1")
        self.assertEqual(benchmark["track"], "A")
        self.assertEqual(benchmark["task_sizes"], [4] + [2] * 11)
        self.assertEqual(
            [len(task) for task in benchmark["tasks"]],
            [4] + [2] * 11,
        )
        self.assertEqual(
            [label for task in benchmark["tasks"] for label in task],
            benchmark["class_order"],
        )
        self.assertEqual(execution["seeds"], [0, 1, 2])
        self.assertEqual(execution["reporting_split"], "test")
        self.assertTrue(execution["configuration_locked"])
        self.assertFalse(execution["git_dirty"])
        self.assertEqual(execution["bundle_count"], 36)

    def test_all_methods_have_three_seeds_and_twelve_task_curve_points(self) -> None:
        methods = self.payload["methods"]
        self.assertEqual(len(methods), 12)
        for method in methods:
            with self.subTest(method=method["display_name"]):
                self.assertEqual(
                    [row["seed"] for row in method["per_seed"]],
                    [0, 1, 2],
                )
                self.assertEqual(
                    [row["task"] for row in method["per_task_mAP"]],
                    list(range(12)),
                )
                final_values = [row["final_mAP"] for row in method["per_seed"]]
                aggregate = method["aggregate"]["final_mAP"]
                self.assertAlmostEqual(
                    aggregate["mean"],
                    statistics.mean(final_values),
                )
                self.assertAlmostEqual(
                    aggregate["std"],
                    statistics.stdev(final_values),
                )

    def test_rankings_are_derived_from_registered_aggregates(self) -> None:
        methods = self.payload["methods"]
        for metric in ("final_mAP", "final_cF1", "final_oF1", "average_mAP"):
            expected = [
                row["display_name"]
                for row in sorted(
                    methods,
                    key=lambda item: item["aggregate"][metric]["mean"],
                    reverse=True,
                )
            ]
            self.assertEqual(self.payload["ranking"][metric], expected)
        expected_forgetting = [
            row["display_name"]
            for row in sorted(
                methods,
                key=lambda item: item["aggregate"]["forgetting"]["mean"],
            )
        ]
        self.assertEqual(self.payload["ranking"]["forgetting"], expected_forgetting)

    def test_artifact_and_scheduler_audits_are_preserved(self) -> None:
        preflight = self.payload["preflight_audit"]
        integrity = self.payload["integrity"]
        scheduler = self.payload["scheduler_audit"]
        self.assertEqual(preflight["core_and_baseline_tests"]["ran"], 160)
        self.assertEqual(preflight["core_and_baseline_tests"]["successful"], 157)
        self.assertEqual(preflight["selected_legacy_regressions"]["successful"], 17)
        self.assertTrue(integrity["outer_checksum_verified"])
        self.assertTrue(integrity["canonical_scores_included"])
        self.assertFalse(integrity["contains_pth"])
        self.assertEqual(integrity["canonical_score_file_count"], 432)
        self.assertEqual(integrity["outer_manifest_verified_file_count"], 763)
        self.assertEqual(integrity["nested_sync_manifest_count"], 36)
        self.assertEqual(integrity["nested_sync_manifest_verified_file_count"], 648)
        self.assertEqual(len(integrity["archive_sha256"]), 64)
        self.assertEqual(scheduler["jobs_started"], 36)
        self.assertEqual(scheduler["jobs_completed"], 36)
        self.assertEqual(scheduler["jobs_failed"], 0)
        self.assertEqual(scheduler["oom_failures"], [])
        self.assertEqual(scheduler["retries"], [])

    def test_amp_skip_audit_and_leading_result_are_preserved(self) -> None:
        skips = self.payload["training_stability"][
            "skipped_updates_by_method_and_seed"
        ]
        self.assertEqual(skips["krt"]["total"], 258)
        self.assertEqual(skips["original_ddp"]["total"], 104)
        self.assertEqual(skips["l3a"]["total"], 25)
        self.assertEqual(skips["multi_lane"]["total"], 0)
        self.assertEqual(self.payload["ranking"]["final_mAP"][0], "MULTI-LANE")
        top = next(
            row
            for row in self.payload["methods"]
            if row["display_name"] == "MULTI-LANE"
        )
        self.assertAlmostEqual(
            top["aggregate"]["final_mAP"]["mean"],
            29.613776596509812,
        )

    def test_documentation_references_the_frozen_snapshot(self) -> None:
        snapshot_name = SNAPSHOT.name
        protocol_doc = PROTOCOL_DOC.read_text(encoding="utf-8")
        status_doc = STATUS_DOC.read_text(encoding="utf-8")
        readme = BENCHMARK_README.read_text(encoding="utf-8")
        self.assertIn("B4-C2 Track A v0.1 is frozen", protocol_doc)
        self.assertIn(snapshot_name, protocol_doc)
        self.assertIn(snapshot_name, status_doc)
        self.assertIn(snapshot_name, readme)
        self.assertNotIn("Execution has not started", status_doc)


if __name__ == "__main__":
    unittest.main()
