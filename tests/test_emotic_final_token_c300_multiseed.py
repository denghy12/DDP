import json
import tempfile
import unittest
from pathlib import Path

from summarize_emotic_ddp_final_token_c300_multiseed import (
    RUN_TEMPLATE,
    SEEDS,
    build_summary,
)


def diagnostics(value):
    distribution = {"mean": value, "p95": value, "max": value}
    return {
        "attention_kl": dict(distribution),
        "pooled_feature_cosine_drift": dict(distribution),
        "path_logit_absolute_drift": dict(distribution),
        "path_logit_rms_drift": value,
        "path_logit_signed_mean_drift": value,
        "all_token_delta_l2": dict(distribution),
        "cls_token_delta_l2": dict(distribution),
        "patch_token_delta_l2": dict(distribution),
    }


class FinalTokenC300MultiSeedTest(unittest.TestCase):
    def test_non_positive_mean_gain_stops_full_bank(self):
        gains = (0.1, -0.2, 0.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed, gain in zip(SEEDS, gains):
                run = root / RUN_TEMPLATE.format(seed=seed)
                run.mkdir()
                payload = {
                    "protocol": {
                        "test_dataset_constructed": False,
                        "test_labels_used": False,
                        "validation_role": "post-training reporting only",
                    },
                    "args": {
                        "task_id": 0,
                        "training_mode": "full",
                        "seed": seed,
                        "lr": 3e-5,
                        "residual_scale": 0.03,
                        "identity_weight": 0.1,
                    },
                    "training_health_check": {
                        "status": "passed",
                        "checks": {"gradient": True},
                    },
                    "optimizer_steps": 300,
                    "initial_val_mAP": 50.0,
                    "final_val_mAP": 50.0 + gain,
                    "initial_per_class_ap": {"A": 40.0},
                    "final_per_class_ap": {"A": 40.0 + gain},
                    "initial_validation_diagnostics": diagnostics(0.0),
                    "final_validation_diagnostics": diagnostics(0.1),
                    "history": [
                        {
                            "loss": 0.5,
                            "identity_loss": 0.01,
                            "up_weight_delta_norm": 1.0,
                            "down_weight_delta_norm": 1.0,
                        }
                    ],
                }
                (run / "training_summary.json").write_text(json.dumps(payload))
            summary = build_summary(root)
        self.assertFalse(summary["aggregate"]["passes_task0_mean_gain_gate"])
        self.assertFalse(summary["automatic_full_bank_launch"])
        self.assertIn("stop_final_token", summary["aggregate"]["decision"])


if __name__ == "__main__":
    unittest.main()

