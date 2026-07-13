import unittest

import torch

from screen_emotic_ddp_internal_adapter_transfer import validation_map
from screen_emotic_ddp_internal_adapter_transfer import select_stable_scale


class IdentityAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_scale = 0.0

    def forward(self, features):
        return features, features


class InternalAdapterTransferSelectionTest(unittest.TestCase):
    def test_validation_accepts_eval_cache_targets_key(self):
        payload = {
            "path_features": torch.randn(4, 2, 3),
            "base_path_logits": torch.tensor(
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
            ),
            "text_features": torch.randn(2, 3),
            "targets": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        }
        score, ratio_mean, ratio_max = validation_map(
            IdentityAdapter(), payload, "path_features", 0.0, 4, "cpu"
        )
        self.assertAlmostEqual(score, 100.0)
        self.assertEqual(ratio_mean, 0.0)
        self.assertEqual(ratio_max, 0.0)

    def test_selects_best_stable_scale_not_unstable_mean_winner(self):
        rows = [
            {
                "residual_scale": 0.0,
                "mean_val_mAP": 56.4,
                "mean_val_gain": 0.0,
                "all_seeds_positive": False,
            },
            {
                "residual_scale": 0.01,
                "mean_val_mAP": 56.8,
                "mean_val_gain": 0.4,
                "all_seeds_positive": True,
            },
            {
                "residual_scale": 0.03,
                "mean_val_mAP": 57.0,
                "mean_val_gain": 0.6,
                "all_seeds_positive": False,
            },
        ]
        selected, unconstrained, passes = select_stable_scale(rows, 0.1)
        self.assertTrue(passes)
        self.assertEqual(selected["residual_scale"], 0.01)
        self.assertEqual(unconstrained["residual_scale"], 0.03)

    def test_returns_identity_when_no_scale_passes(self):
        rows = [
            {
                "residual_scale": 0.0,
                "mean_val_mAP": 56.4,
                "mean_val_gain": 0.0,
                "all_seeds_positive": False,
            },
            {
                "residual_scale": 0.01,
                "mean_val_mAP": 56.45,
                "mean_val_gain": 0.05,
                "all_seeds_positive": True,
            },
        ]
        selected, _, passes = select_stable_scale(rows, 0.1)
        self.assertFalse(passes)
        self.assertEqual(selected["residual_scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
