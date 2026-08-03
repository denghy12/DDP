import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "compare_l3a_upstream_reference.py"
)
SPEC = importlib.util.spec_from_file_location("compare_l3a_upstream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class L3AReferenceAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (
            Path(__file__).resolve().parents[3]
            / "baseline_sources"
            / "l3a_official"
        )
        if not cls.root.is_dir():
            raise unittest.SkipTest("fixed external L3A source is unavailable")

    def test_fixed_source_identity(self):
        payload = MODULE._verify_upstream(self.root, archive=None)
        self.assertEqual(payload["commit"], MODULE.UPSTREAM_COMMIT)
        self.assertEqual(payload["git_tree"], MODULE.UPSTREAM_GIT_TREE)
        self.assertFalse(payload["source_copied_into_repository"])
        self.assertEqual(len(payload["verified_file_sha256"]), 5)

    def test_critical_operator_equivalence(self):
        payload = MODULE._operator_audit(self.root)
        for key, value in payload.items():
            if key.endswith("_error"):
                self.assertLess(value, 1.0e-12, key)
        self.assertTrue(payload["pseudo_threshold_exact_match"])
        self.assertTrue(payload["pseudo_threshold_is_strict_greater_than"])


if __name__ == "__main__":
    unittest.main()
