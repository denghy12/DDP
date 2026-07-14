import json
import tempfile
import unittest
from pathlib import Path

from summarize_emotic_ddp_cls_full_base5_comparison import adapter_summary
from summarize_emotic_ddp_cls_full_base5_comparison import baseline_summary


def task_row(task, value):
    return {
        "task": task,
        "seen_classes": (5, 8, 11, 14, 17, 20, 23, 26)[task],
        "test": {"mAP": value, "cF1": value + 1.0, "oF1": value + 2.0},
        "test_mAP_gain": value - 30.0,
    }


class FullBase5ComparisonTest(unittest.TestCase):
    def test_baseline_summary_keeps_ddp_f1_and_forgetting(self):
        data = {
            "tasks": [task_row(task, 30.0 + task) for task in range(8)],
            "aggregate": {
                "average_mAP": 33.5,
                "final_mAP": 37.0,
                "forgetting": {"average_forgetting_old_classes": 4.9},
            },
        }
        result = baseline_summary(data)
        self.assertEqual(result["display_name"], "Original DDP")
        self.assertEqual(result["final_cF1"]["mean"], 38.0)
        self.assertEqual(result["final_oF1"]["mean"], 39.0)
        self.assertEqual(result["forgetting"]["mean"], 4.9)

    def test_adapter_summary_uses_full_base5_sources_and_three_seed_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = "full_feature_seed"
            for seed, offset in enumerate((0.0, 1.0, 2.0)):
                run_dir = root / f"{prefix}{seed}"
                run_dir.mkdir()
                tasks = [task_row(task, 31.0 + task + offset) for task in range(8)]
                data = {
                    "protocol": {"correction_mode": "linear_residual"},
                    "inputs": {
                        "transfer_selection": {
                            "source": f"base5_balanced_seed{seed}/best_adapter.pth"
                        }
                    },
                    "tasks": tasks,
                    "aggregate": {
                        "average_mAP": 34.5 + offset,
                        "final_mAP": 38.0 + offset,
                        "average_mAP_gain": 1.0 + offset,
                        "final_mAP_gain": 1.5 + offset,
                        "forgetting": {
                            "average_forgetting_old_classes": 4.0 + offset
                        },
                    },
                }
                (run_dir / "evaluation_summary.json").write_text(
                    json.dumps(data), encoding="utf-8"
                )
            screen_dir = root / "screen"
            screen_dir.mkdir()
            screen = {
                "selection_split": "val",
                "test_used": False,
                "passes_val_gate": True,
                "identity_val_mAP": 56.4,
                "best": {
                    "residual_scale": 0.03,
                    "mean_val_mAP": 59.0,
                    "mean_val_gain": 2.6,
                    "minimum_seed_gain": 2.0,
                    "all_seeds_positive": True,
                },
            }
            (screen_dir / "transfer_screen_summary.json").write_text(
                json.dumps(screen), encoding="utf-8"
            )
            result = adapter_summary(
                root,
                (0, 1, 2),
                "feature_difference",
                "Feature difference",
                prefix,
                "screen",
            )
            self.assertEqual(result["run_count"], 3)
            self.assertEqual(result["selection"]["global_alpha"], 0.03)
            self.assertAlmostEqual(result["final_mAP"]["mean"], 39.0)
            self.assertAlmostEqual(result["final_mAP"]["std"], 0.8164965809)


if __name__ == "__main__":
    unittest.main()
