import json
import tempfile
import unittest
from pathlib import Path

from summarize_emotic_ddp_final_token_safety_screen import CONFIGS, build_summary


class FinalTokenSafetyScreenTest(unittest.TestCase):
    def test_summary_audits_protocol_without_selecting_a_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, config in enumerate(CONFIGS):
                run = root / config["run_name"]
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
                        "seed": 0,
                        "lr": config["learning_rate"],
                        "residual_scale": 0.03,
                        "identity_weight": 0.1,
                    },
                    "training_health_check": {
                        "status": "passed",
                        "checks": {"gradient": True},
                    },
                    "optimizer_steps": config["max_optimizer_steps"],
                    "initial_val_mAP": 50.0,
                    "final_val_mAP": 50.0 + index,
                    "initial_per_class_ap": {"A": 40.0},
                    "final_per_class_ap": {"A": 40.0 + index},
                    "history": [
                        {
                            "loss": 0.5,
                            "classification_loss": 0.49,
                            "identity_loss": 0.01,
                            "up_weight_delta_norm": 1.0,
                            "down_weight_delta_norm": 1.0,
                            "last_raw_residual_rms": 0.1,
                            "last_adapted_feature_delta_rms": 0.01,
                        }
                    ],
                }
                (run / "training_summary.json").write_text(json.dumps(payload))
            summary = build_summary(root)
        self.assertIsNone(summary["selection"])
        self.assertFalse(summary["protocol"]["automatic_winner_selection"])
        self.assertEqual([row["config"] for row in summary["runs"]], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()

