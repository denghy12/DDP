import json
import tempfile
import unittest
from pathlib import Path

from summarize_emotic_ddp_final_token_pooling_aware_bank import collect_method


def evaluation_payload(seed, offset=0.0):
    tasks = []
    for task in range(8):
        baseline = 40.0 - task
        value = baseline + offset
        tasks.append(
            {
                "task": task,
                "seen_classes": 5 + 3 * task,
                "baseline_ddp_test_mAP": baseline,
                "test_mAP_gain": offset,
                "test": {"mAP": value},
                "current_test": {"mAP": value},
            }
        )
    return {
        "protocol": {
            "seed": seed,
            "decision_threshold": 0.5,
            "test_used_for_selection": False,
            "checkpoint_selection": "fixed 300 optimizer steps",
            "pooling_aware_regularization": {
                "optimizer_steps": 300,
                "pooling_weight": 100.0,
                "attention_weight": 100.0,
                "attention_kl_implementation": "log_softmax_stable",
                "training_precision": "float32",
                "margin_weight": 1.0,
                "margin_beta": 1.0,
            },
        },
        "tasks": tasks,
        "aggregate": {
            "average_mAP": 36.5 + offset,
            "final_mAP": 33.0 + offset,
            "average_mAP_gain": offset,
            "final_mAP_gain": offset,
            "final_cF1": 30.0,
            "final_oF1": 45.0,
            "forgetting": {"average_forgetting_old_classes": 4.0},
        },
    }


class PoolingAwareBankSummaryTest(unittest.TestCase):
    def test_collects_three_locked_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, offset in enumerate((0.1, 0.2, 0.3)):
                output = root / (
                    f"emotic_ddp_final_token_pooling_aware_bank_full_seed{seed}"
                )
                output.mkdir()
                (output / "evaluation_summary.json").write_text(
                    json.dumps(evaluation_payload(seed, offset)), encoding="utf-8"
                )
            runs, aggregate, tasks = collect_method(
                root, "pooling_aware_full", (0, 1, 2)
            )
            self.assertEqual(len(runs), 3)
            self.assertEqual(len(tasks), 8)
            self.assertAlmostEqual(aggregate["final_mAP_gain"]["mean"], 0.2)
            self.assertAlmostEqual(tasks[0]["gain_mean"], 0.2)

    def test_rejects_unlocked_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "emotic_ddp_final_token_pooling_aware_bank_full_seed0"
            output.mkdir()
            payload = evaluation_payload(0)
            payload["protocol"]["decision_threshold"] = 0.37
            (output / "evaluation_summary.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                collect_method(root, "pooling_aware_full", (0,))


if __name__ == "__main__":
    unittest.main()
