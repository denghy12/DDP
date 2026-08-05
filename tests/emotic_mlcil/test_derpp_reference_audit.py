import importlib.util
import os
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "compare_derpp_upstream_reference.py"
)
SPEC = importlib.util.spec_from_file_location("compare_derpp_upstream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DERPPReferenceAuditTest(unittest.TestCase):
    def test_fixed_external_source_operator_equivalence(self):
        configured = os.environ.get("DERPP_SOURCE_ROOT")
        candidates = [
            Path(configured) if configured else None,
            Path(
                "/Users/denghaoyuan/workspace/MyCode/"
                "baseline_sources/derpp_official"
            ),
            Path("/mnt/haoyuan/workspace/baseline_sources/derpp_official"),
        ]
        source = next(
            (candidate for candidate in candidates if candidate and candidate.is_dir()),
            Path("/nonexistent/fixed-derpp-source"),
        )
        if not source.is_dir():
            self.skipTest("fixed external DER++ source is not available")
        payload = MODULE.compare(source)
        self.assertEqual(payload["upstream"]["commit"], MODULE.COMMIT)
        self.assertEqual(payload["upstream"]["license"], "MIT")
        self.assertEqual(payload["source_structure"]["independent_get_data_calls"], 2)
        self.assertFalse(payload["upstream"]["source_copied_into_repository"])
        self.assertLessEqual(
            max(payload["operator_equivalence"].values()),
            1.0e-7,
        )


if __name__ == "__main__":
    unittest.main()
