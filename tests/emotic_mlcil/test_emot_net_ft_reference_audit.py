import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "emotic-mlcil" / "compare_emot_net_upstream_reference.py"
SPEC = importlib.util.spec_from_file_location("compare_emot_net_upstream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EMOTNetFTReferenceAuditTest(unittest.TestCase):
    def test_fixed_external_source_and_operator_contract(self):
        source = Path(
            "/Users/denghaoyuan/workspace/MyCode/baseline_sources/"
            "emot_net_release_69c3a51"
        )
        if not source.is_dir():
            self.skipTest("fixed external EMOT-Net source is not available")
        payload = MODULE.compare(source)
        self.assertEqual(payload["upstream"]["commit"], MODULE.COMMIT)
        self.assertEqual(payload["upstream"]["license"], "MIT")
        self.assertFalse(payload["upstream"]["source_copied_into_repository"])
        self.assertFalse(payload["conversion"]["clip_used"])
        self.assertLess(max(payload["operator_equivalence"].values()), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
