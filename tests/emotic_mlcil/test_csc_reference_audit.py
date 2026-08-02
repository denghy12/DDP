import json
import unittest
from pathlib import Path


class CSCReferenceAuditSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "benchmarks"
            / "results"
            / "csc_upstream_equivalence_v0.1.json"
        )
        cls.payload = json.loads(path.read_text(encoding="utf-8"))

    def test_fixed_clean_provenance_and_no_vendored_source(self):
        self.assertEqual(
            self.payload["upstream"]["commit"],
            "0bab38a00d6e0555f2df855ae2fe8db1fea68b12",
        )
        self.assertFalse(self.payload["upstream"]["source_copied_into_repository"])
        self.assertFalse(self.payload["port"]["git_dirty"])
        self.assertEqual(
            self.payload["port"]["git_commit"],
            "6dc9cf25801f35826f3084246a87ae7248c268c0",
        )

    def test_operator_errors_are_numerical_precision_only(self):
        float64 = self.payload["base_operator_equivalence"]["float64"]
        float32 = self.payload["base_operator_equivalence"]["float32"]
        self.assertLess(float64["combined_logits_max_abs_error"], 1.0e-12)
        self.assertLess(float64["mapped_parameter_gradient_max_abs_error"], 1.0e-12)
        self.assertLess(float32["combined_logits_max_abs_error"], 1.0e-6)
        self.assertLess(float32["mapped_parameter_gradient_max_abs_error"], 1.0e-5)

    def test_expansion_difference_is_isolated_and_documented(self):
        audit = self.payload["incremental_expansion_audit"]
        self.assertGreater(
            audit["controlled_logit_delta_from_reinitialized_old_biases"],
            1.0e-3,
        )
        self.assertLess(
            audit["full_expanded_parameter_remap_max_abs_error"],
            1.0e-6,
        )
        official = audit["optimizer_coverage_after_expansion"][
            "official_optimizer_created_before_expansion"
        ]
        port = audit["optimizer_coverage_after_expansion"][
            "port_optimizer_rebuilt_after_expansion"
        ]
        self.assertGreater(official["missing_live_parameter_tensors"], 0)
        self.assertGreater(official["stale_parameter_tensors"], 0)
        self.assertEqual(port["missing_live_parameter_tensors"], 0)
        self.assertEqual(port["stale_parameter_tensors"], 0)


if __name__ == "__main__":
    unittest.main()
