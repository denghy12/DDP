import tempfile
import unittest
from pathlib import Path

from summarize_emotic_ddp_main_asl_task0_ablation import (
    eligibility,
    legacy_seed0_name,
    mean_std,
    resolve_run_root,
)


class DDPMainASLTask0AblationTest(unittest.TestCase):
    def test_seed0_reuses_corrected_healthcheck_only_for_references(self):
        self.assertEqual(
            legacy_seed0_name("two_way_bce", 0),
            "emotic_ddp_main_healthcheck_two_way_bce_task0_seed0_v2",
        )
        self.assertEqual(
            legacy_seed0_name("asl_g9p8", 0),
            "emotic_ddp_main_healthcheck_asl_task0_seed0_v2",
        )
        self.assertIsNone(legacy_seed0_name("asl_g4", 0))
        self.assertIsNone(legacy_seed0_name("two_way_bce", 1))

    def test_primary_run_takes_precedence_over_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "emotic_ddp_main_task0_two_way_bce_seed0_v1"
            legacy = root / "emotic_ddp_main_healthcheck_two_way_bce_task0_seed0_v2"
            primary.mkdir()
            legacy.mkdir()
            self.assertEqual(
                resolve_run_root(root, "two_way_bce", 0), primary
            )

    def test_gate_requires_all_three_pre_registered_checks(self):
        passing = {
            "val_mAP": {"mean": -0.05},
            "val_cF1": {"mean": -1.0},
            "val_oF1": {"mean": 1.0},
        }
        failing = {
            **passing,
            "val_oF1": {"mean": -2.01},
        }
        self.assertTrue(eligibility(passing)["passes"])
        self.assertFalse(eligibility(failing)["passes"])

    def test_std_is_population_std(self):
        result = mean_std([1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["std"], (2.0 / 3.0) ** 0.5)


if __name__ == "__main__":
    unittest.main()
