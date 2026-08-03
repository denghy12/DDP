import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "compare_original_ddp_source_reference.py"
)
SPEC = importlib.util.spec_from_file_location("compare_original_ddp_source", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OriginalDDPReferenceAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (
            Path(__file__).resolve().parents[3]
            / "baseline_sources"
            / "CODE_DDP"
        )
        if not cls.root.is_dir():
            raise unittest.SkipTest("fixed external Original DDP source is unavailable")

    def test_fixed_source_identity(self):
        payload = MODULE._verify_source(self.root)
        self.assertEqual(
            payload["snapshot_tree_sha256"],
            MODULE.SOURCE_SNAPSHOT_TREE_SHA256,
        )
        self.assertEqual(payload["file_count"], 30)
        self.assertEqual(payload["total_bytes"], 1_470_709)
        self.assertEqual(
            payload["registered_archive_sha256"],
            MODULE.SOURCE_ARCHIVE_SHA256,
        )
        self.assertEqual(payload["license_files"], [])
        self.assertFalse(payload["source_copied_into_repository"])

    def test_released_operators_and_registered_pcd(self):
        payload = MODULE._operator_audit(self.root)
        for key, value in payload.items():
            if key.endswith("_error"):
                self.assertLessEqual(value, 1.0e-12, key)
        self.assertEqual(payload["source_pcd_temperature"]["maximum"], 7.0)
        self.assertEqual(payload["source_pcd_temperature"]["gamma"], 0.2)
        self.assertEqual(payload["registered_pcd_temperature"]["maximum"], 2.0)
        self.assertEqual(payload["registered_pcd_temperature"]["gamma"], 0.7)
        self.assertTrue(payload["pcd_is_deliberate_benchmark_deviation"])
        self.assertTrue(payload["prompt_aware_clip_ast_exact_match"])
        self.assertEqual(payload["verified_definition_count"], 11)


if __name__ == "__main__":
    unittest.main()
