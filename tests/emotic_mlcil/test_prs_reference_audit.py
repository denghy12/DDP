import importlib.util
import os
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "compare_prs_upstream_reference.py"
)
SPEC = importlib.util.spec_from_file_location("compare_prs_upstream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PRSReferenceAuditTest(unittest.TestCase):
    def test_fixed_external_source_operator_equivalence(self):
        configured = os.environ.get("PRS_SOURCE_ROOT")
        candidates = [
            Path(configured) if configured else None,
            Path("/Users/denghaoyuan/workspace/MyCode/baseline_sources/PRS-master"),
            Path("/mnt/haoyuan/workspace/baseline_sources/PRS-master"),
        ]
        source = next(
            (candidate for candidate in candidates if candidate and candidate.is_dir()),
            Path("/nonexistent/fixed-prs-source"),
        )
        if not source.is_dir():
            self.skipTest("fixed external PRS source is not available")
        payload = MODULE.compare(source)
        self.assertEqual(payload["upstream"]["commit"], MODULE.COMMIT)
        self.assertEqual(payload["upstream"]["license"], "MIT")
        self.assertFalse(payload["upstream"]["source_copied_into_repository"])
        self.assertTrue(
            payload["operator_equivalence"]["retained_id_exact_match"]
        )
        self.assertLessEqual(
            payload["operator_equivalence"]["target_proportion_max_abs_error"],
            1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
