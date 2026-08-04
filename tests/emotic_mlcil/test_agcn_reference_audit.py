import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "emotic-mlcil" / "compare_agcn_upstream_reference.py"
SPEC = importlib.util.spec_from_file_location("compare_agcn_upstream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AGCNReferenceAuditTest(unittest.TestCase):
    def test_fixed_external_source_operator_equivalence(self):
        source = Path("/Users/denghaoyuan/workspace/MyCode/baseline_sources/agcn_release_3afe2ec")
        if not source.is_dir():
            self.skipTest("fixed external AGCN source is not available")
        payload = MODULE.compare(source)
        self.assertEqual(payload["upstream"]["commit"], MODULE.COMMIT)
        self.assertEqual(payload["upstream"]["license"], "Apache-2.0")
        self.assertFalse(payload["upstream"]["source_copied_into_repository"])
        self.assertLess(max(payload["operator_equivalence"].values()), 1e-6)


if __name__ == "__main__":
    unittest.main()
