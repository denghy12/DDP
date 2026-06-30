import unittest

import torch
import torch.nn.functional as F

from prototype_adapter import ResidualPrototypeAdapter, protocol_class_indices


class ResidualPrototypeAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.positive = F.normalize(torch.randn(4, 8), dim=-1)
        self.negative = F.normalize(torch.randn(4, 8), dim=-1)

    def test_zero_initialized_adapter_preserves_features(self):
        model = ResidualPrototypeAdapter(
            self.positive,
            self.negative,
            bottleneck_dim=3,
            residual_scale=0.1,
        )
        features = torch.randn(5, 8)
        _, adapted, original = model(features)
        self.assertTrue(torch.allclose(adapted, original, atol=1e-6))

    def test_logits_use_positive_minus_negative_similarity(self):
        model = ResidualPrototypeAdapter(
            self.positive,
            self.negative,
            bottleneck_dim=3,
            residual_scale=0.1,
            initial_logit_scale=7.0,
        )
        features = torch.randn(5, 8)
        logits, adapted, _ = model(features)
        expected = 7.0 * (
            adapted @ self.positive.t() - adapted @ self.negative.t()
        )
        self.assertEqual(logits.shape, (5, 4))
        self.assertTrue(torch.allclose(logits, expected, atol=1e-5))

    def test_protocol_indices(self):
        self.assertEqual(
            protocol_class_indices("all26", 26, 5).tolist(), list(range(26))
        )
        self.assertEqual(
            protocol_class_indices("base5", 26, 5).tolist(), list(range(5))
        )


if __name__ == "__main__":
    unittest.main()

