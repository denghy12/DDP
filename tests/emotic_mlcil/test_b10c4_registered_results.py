"""Integrity checks for the frozen B10-C4 12-baseline result snapshot."""

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
    / "b10c4_12baseline_seed012_formal_v0.1.json"
)
MARKDOWN = SNAPSHOT.with_name("B10C4_TRACK_A_BASELINE_SUMMARY_V0.1.md")


class B10C4RegisteredResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    def test_protocol_and_execution_are_frozen(self) -> None:
        benchmark = self.payload["benchmark"]
        execution = self.payload["execution"]
        self.assertEqual(self.payload["status"], "frozen")
        self.assertEqual(benchmark["protocol_id"], "emotic_b10c4_v0.1")
        self.assertEqual(benchmark["track"], "A")
        self.assertEqual(benchmark["task_sizes"], [10, 4, 4, 4, 4])
        self.assertEqual([len(task) for task in benchmark["tasks"]], [10, 4, 4, 4, 4])
        self.assertEqual(
            [label for task in benchmark["tasks"] for label in task],
            benchmark["class_order"],
        )
        self.assertEqual(execution["seeds"], [0, 1, 2])
        self.assertEqual(execution["reporting_split"], "test")
        self.assertTrue(execution["configuration_locked"])
        self.assertFalse(execution["git_dirty"])
        self.assertEqual(execution["bundle_count"], 36)

    def test_all_methods_have_three_seeds_and_five_task_curve_points(self) -> None:
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
                    [0, 1, 2, 3, 4],
                )
                final_values = [row["final_mAP"] for row in method["per_seed"]]
                aggregate = method["aggregate"]["final_mAP"]
                self.assertAlmostEqual(aggregate["mean"], statistics.mean(final_values))
                self.assertAlmostEqual(aggregate["std"], statistics.stdev(final_values))

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

    def test_artifact_and_retry_audits_are_preserved(self) -> None:
        integrity = self.payload["integrity"]
        retry = self.payload["retry_audit"]
        self.assertTrue(integrity["outer_checksum_verified"])
        self.assertTrue(integrity["canonical_scores_included"])
        self.assertFalse(integrity["contains_pth"])
        self.assertEqual(integrity["canonical_score_file_count"], 180)
        self.assertEqual(len(integrity["archive_sha256"]), 64)
        self.assertEqual(retry["original_failures"], ["er_seed0", "prs_seed0"])
        self.assertTrue(retry["recovery"]["isolated"])
        self.assertFalse(
            retry["recovery"]["algorithm_or_hyperparameters_changed"]
        )

    def test_paper_table_references_the_registered_result(self) -> None:
        markdown = MARKDOWN.read_text(encoding="utf-8")
        self.assertIn("33.7 ± 0.2", markdown)
        self.assertIn("b10c4_12baseline_seed012_formal_v0.1.json", markdown)


if __name__ == "__main__":
    unittest.main()
