import unittest

from summarize_emotic_ddp_final_token_pooling_aware_pilot import build_summary


def diagnostics(value):
    return {
        "attention_kl": {"mean": value, "p95": value, "max": value},
        "pooled_feature_cosine_drift": {
            "mean": value,
            "p95": value,
            "max": value,
        },
    }


def payload(gain, regularization):
    return {
        "args": {
            "task_id": 0,
            "training_mode": "full",
            "seed": 0,
            "lr": 3e-5,
            "residual_scale": 0.03,
            "identity_weight": 0.1,
            "max_optimizer_steps": 300,
        },
        "protocol": {
            "test_dataset_constructed": False,
            "test_labels_used": False,
        },
        "training_health_check": {"status": "passed"},
        "regularization": regularization,
        "initial_val_mAP": 50.0,
        "final_val_mAP": 50.0 + gain,
        "reporting_val_gain": gain,
        "final_validation_diagnostics": diagnostics(0.1),
        "history": [{"loss": 0.5}],
    }


class PoolingAwarePilotSummaryTest(unittest.TestCase):
    def test_controlled_comparison_keeps_full_bank_disabled(self):
        baseline = payload(
            -0.1,
            {
                "pooling_weight": 0.0,
                "attention_weight": 0.0,
                "margin_weight": 0.0,
                "margin_beta": 1.0,
            },
        )
        candidate = payload(
            0.2,
            {
                "pooling_weight": 100.0,
                "attention_weight": 100.0,
                "margin_weight": 1.0,
                "margin_beta": 1.0,
            },
        )
        summary = build_summary(baseline, candidate)
        self.assertTrue(summary["pilot_passes_positive_gain"])
        self.assertFalse(summary["automatic_full_bank_launch"])
        self.assertAlmostEqual(
            summary["pooling_aware_candidate"][
                "gain_relative_to_unregularized_c300"
            ],
            0.3,
        )


if __name__ == "__main__":
    unittest.main()

