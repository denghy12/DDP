import unittest

from summarize_emotic_ddp_main_asl_g2_scale_task0 import (
    METHODS,
    paired_delta,
)


class DDPMainASLG2ScaleTask0Test(unittest.TestCase):
    def test_scale_matched_configuration_is_locked(self):
        self.assertEqual(METHODS["asl_g2_lw009"]["objective"], "asl")
        self.assertEqual(METHODS["asl_g2_lw009"]["gamma_neg"], 2.0)
        self.assertEqual(METHODS["asl_g2_lw009"]["loss_w"], 0.09)

    def test_paired_delta_keeps_seed_alignment(self):
        baseline = [
            {"seed": 0, "val_mAP": 10.0, "val_cF1": 20.0, "val_oF1": 30.0},
            {"seed": 1, "val_mAP": 11.0, "val_cF1": 21.0, "val_oF1": 31.0},
        ]
        candidate = [
            {"seed": 0, "val_mAP": 10.5, "val_cF1": 22.0, "val_oF1": 29.0},
            {"seed": 1, "val_mAP": 11.5, "val_cF1": 23.0, "val_oF1": 30.0},
        ]
        result = paired_delta(candidate, baseline)
        self.assertEqual(result["val_mAP"]["mean"], 0.5)
        self.assertEqual(result["val_cF1"]["mean"], 2.0)
        self.assertEqual(result["val_oF1"]["mean"], -1.0)
        self.assertEqual(result["val_mAP"]["per_seed"], {"0": 0.5, "1": 0.5})


if __name__ == "__main__":
    unittest.main()
