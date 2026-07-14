import unittest
from types import SimpleNamespace

import torch

from eval_emotic_ddp_internal_adapter import predict_from_cache
from screen_emotic_ddp_internal_adapter_transfer import validation_map
from screen_emotic_ddp_internal_adapter_transfer import resolve_external_paths
from screen_emotic_ddp_internal_adapter_transfer import select_stable_scale


class IdentityAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_scale = 0.0

    def forward(self, features):
        return features, features


class AdapterMustNotRun:
    def logits_from_path_features(self, *args, **kwargs):
        raise AssertionError("DDP-only evaluation must not call the Adapter path")


class InternalAdapterTransferSelectionTest(unittest.TestCase):
    def test_ddp_only_prediction_uses_cached_original_logits(self):
        base_logits = torch.tensor(
            [
                [2.0, 0.0, 1.0, -1.0, 0.5, 0.0, 2.0, -1.0, 1.0, -0.5],
                [0.0, 1.0, -1.0, 2.0, 0.5, 2.0, 0.0, 1.0, -1.0, -0.5],
            ]
        )
        payload = {
            "path_features": torch.randn(2, 10, 3),
            "base_path_logits": base_logits,
            "text_features": torch.randn(10, 3),
        }
        args = SimpleNamespace(
            adapter_batch_size=2,
            device="cpu",
            ddp_only=True,
            t_min=1.0,
            t_max=2.0,
            t_gamma=0.7,
            feature_source="cls",
        )
        scores, temperature = predict_from_cache(
            AdapterMustNotRun(), payload, 5, args
        )
        expected = torch.softmax(
            base_logits.reshape(2, 2, 5) / temperature, dim=1
        )[:, 1, :]
        self.assertTrue(torch.equal(scores, expected))

    def test_explicit_external_paths_support_special_seed_zero_name(self):
        paths = resolve_external_paths(
            (0, 1, 2),
            "unused_seed{seed}.pth",
            ("full_base5.pth", "full_base5_seed1.pth", "full_base5_seed2.pth"),
        )
        self.assertEqual(paths[0], "full_base5.pth")
        self.assertEqual(paths[1], "full_base5_seed1.pth")
        self.assertEqual(paths[2], "full_base5_seed2.pth")

    def test_explicit_external_path_count_must_match_seeds(self):
        with self.assertRaisesRegex(ValueError, "one path per seed"):
            resolve_external_paths((0, 1, 2), "unused", ("a.pth", "b.pth"))

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
        self.assertAlmostEqual(score, 100.0, delta=1e-5)
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
