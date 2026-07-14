import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddp_internal_adapter import feature_logit_correction
from models.ddp import DDP


class FixedAdapter(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", delta)

    def forward(self, features):
        original = features.float()
        return original + self.delta, original


class DDPCosineDifferenceTest(unittest.TestCase):
    def test_matches_explicit_difference_of_cosine_logits(self):
        torch.manual_seed(3)
        original = torch.randn(2, 6, 8)
        adapted = original + 0.2 * torch.randn_like(original)
        text = torch.randn(6, 8)

        actual = feature_logit_correction(
            adapted,
            original,
            text,
            mode="cosine_difference",
        )
        normalized_text = F.normalize(text, dim=-1)
        expected = 100.0 * (
            torch.einsum(
                "bkd,kd->bk", F.normalize(adapted, dim=-1), normalized_text
            )
            - torch.einsum(
                "bkd,kd->bk", F.normalize(original, dim=-1), normalized_text
            )
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_ignores_pure_positive_radial_scaling(self):
        torch.manual_seed(5)
        original = torch.randn(2, 4, 7)
        text = torch.randn(4, 7)
        correction = feature_logit_correction(
            2.5 * original,
            original,
            text,
            mode="cosine_difference",
        )
        self.assertTrue(
            torch.allclose(
                correction, torch.zeros_like(correction), atol=1e-4, rtol=0.0
            )
        )

    def test_identity_is_exact_zero(self):
        features = torch.randn(2, 4, 7)
        text = torch.randn(4, 7)
        correction = feature_logit_correction(
            features,
            features,
            text,
            mode="cosine_difference",
        )
        self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))

    def test_linear_mode_reproduces_legacy_correction(self):
        torch.manual_seed(7)
        original = torch.randn(2, 6, 8)
        adapted = original + torch.randn_like(original)
        text = torch.randn(6, 8)
        actual = feature_logit_correction(
            adapted,
            original,
            text,
            mode="linear_residual",
        )
        expected = 100.0 * torch.einsum(
            "bkd,kd->bk", adapted - original, text
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_model_adds_cosine_difference_before_reshape(self):
        torch.manual_seed(11)
        model = DDP.__new__(DDP)
        nn.Module.__init__(model)
        delta = torch.randn(2, 10, 8) * 0.1
        model.feature_adapter = FixedAdapter(delta)
        model.feature_adapter_correction = "cosine_difference"
        features = torch.randn(2, 10, 8)
        text = torch.randn(10, 8)
        base = torch.randn(2, 10)

        logits, aux = model.logits_from_path_features(
            features,
            base,
            text,
            return_adapter_aux=True,
        )
        expected = base + feature_logit_correction(
            features + delta,
            features,
            text,
            mode="cosine_difference",
        )
        self.assertTrue(torch.allclose(logits, expected.reshape(2, 2, 5)))
        self.assertEqual(aux["correction_mode"], "cosine_difference")


if __name__ == "__main__":
    unittest.main()
