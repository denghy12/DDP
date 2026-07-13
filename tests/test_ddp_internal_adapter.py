import unittest

import torch
import torch.nn as nn

from ddp_internal_adapter import (
    SharedResidualFeatureAdapter,
    feature_identity_loss,
    masked_ddp_bce,
)
from models.ddp import DDP


class DDPInternalAdapterTest(unittest.TestCase):
    def test_zero_init_preserves_features_exactly(self):
        adapter = SharedResidualFeatureAdapter(8, 3, 0.1)
        features = torch.randn(2, 10, 8)
        adapted, original = adapter(features)
        self.assertTrue(torch.equal(adapted, original))
        self.assertLess(abs(feature_identity_loss(adapted, original).item()), 1e-6)

    def test_one_adapter_is_shared_across_arbitrary_path_count(self):
        adapter = SharedResidualFeatureAdapter(8, 3, 0.1)
        with torch.no_grad():
            adapter.up.weight.fill_(0.05)
        five_class, _ = adapter(torch.randn(2, 10, 8))
        eight_class, _ = adapter(torch.randn(2, 16, 8))
        self.assertEqual(five_class.shape, (2, 10, 8))
        self.assertEqual(eight_class.shape, (2, 16, 8))

    def test_zero_adapter_preserves_exact_ddp_path_logits(self):
        model = DDP.__new__(DDP)
        nn.Module.__init__(model)
        model.feature_adapter = SharedResidualFeatureAdapter(8, 3, 0.1)
        pooled = torch.randn(2, 10, 8)
        base = torch.randn(2, 10)
        text = torch.randn(10, 8)
        logits = model.logits_from_path_features(pooled, base, text)
        self.assertTrue(torch.equal(logits, base.reshape(2, 2, 5)))

    def test_masked_loss_ignores_unsupervised_positive_colabel(self):
        logits = torch.zeros(1, 2, 2)
        targets = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[True, False]])
        loss = masked_ddp_bce(logits, targets, mask)
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.0)).item())

    def test_masked_loss_rejects_full_26_class_mask_for_base5_logits(self):
        logits = torch.zeros(2, 2, 5)
        targets = torch.zeros(2, 5)
        mask = torch.ones(2, 26, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "must match"):
            masked_ddp_bce(logits, targets, mask)


if __name__ == "__main__":
    unittest.main()
