import unittest

import torch

from bce_loss import (
    BCELoss,
    TwoWayAsymmetricLoss,
    build_ddp_classification_loss,
    two_way_margin,
    two_way_prediction_diagnostics,
)
from evaluation_metrics import prf_cal


class DDPMainASLLossTest(unittest.TestCase):
    def test_two_way_margin_uses_positive_minus_negative(self):
        logits = torch.tensor(
            [
                [
                    [[0.0]],
                    [[0.0]],
                ]
            ]
        ).reshape(1, 2, 1)
        logits[:, 0, :] = -2.0
        logits[:, 1, :] = 3.5
        self.assertTrue(torch.equal(two_way_margin(logits), torch.tensor([[5.5]])))

    def test_zero_gamma_zero_clip_matches_original_two_way_bce(self):
        logits = torch.tensor(
            [
                [[-0.5, 1.1, -1.2], [0.2, -0.4, 0.7]],
                [[1.3, -0.8, 0.0], [-0.2, 0.9, -0.3]],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        expected = BCELoss()(logits, targets)
        actual = TwoWayAsymmetricLoss(
            gamma_neg=0,
            gamma_pos=0,
            clip=0,
            reduction="sum",
        )(logits, targets)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_loss_is_invariant_to_common_logit_shift(self):
        logits = torch.tensor(
            [[[0.1, -0.3], [0.8, 0.4]]], requires_grad=True
        )
        targets = torch.tensor([[1.0, 0.0]])
        loss_fn = TwoWayAsymmetricLoss()
        original = loss_fn(logits, targets)
        shifted = loss_fn(logits + 17.0, targets)
        self.assertTrue(torch.allclose(original, shifted, atol=1e-7))

    def test_asl_suppresses_an_easy_negative_margin(self):
        logits = torch.tensor([[[0.0], [-6.0]]], requires_grad=True)
        targets = torch.zeros(1, 1)
        bce = BCELoss()(logits, targets)
        bce.backward(retain_graph=True)
        bce_gradient = logits.grad.detach().clone().abs().sum()
        logits.grad.zero_()

        asl = TwoWayAsymmetricLoss(
            gamma_neg=9.8,
            gamma_pos=0.0,
            clip=0.05,
        )(logits, targets)
        asl.backward()
        asl_gradient = logits.grad.detach().abs().sum()
        self.assertLess(float(asl_gradient), float(bce_gradient))

    def test_shape_validation_rejects_single_logit_tensor(self):
        with self.assertRaises(ValueError):
            two_way_margin(torch.zeros(2, 3))

    def test_factory_keeps_bce_as_the_default_compatible_path(self):
        self.assertIsInstance(
            build_ddp_classification_loss("two_way_bce"), BCELoss
        )
        self.assertIsInstance(
            build_ddp_classification_loss("asl"), TwoWayAsymmetricLoss
        )

    def test_large_logits_remain_finite(self):
        logits = torch.tensor(
            [[[-100.0, 100.0], [100.0, -100.0]]],
            requires_grad=True,
        )
        targets = torch.tensor([[1.0, 0.0]])
        loss = TwoWayAsymmetricLoss()(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_diagnostics_report_easy_negative_fraction(self):
        logits = torch.tensor(
            [[[0.0, 0.0], [-6.0, 1.0]]], dtype=torch.float32
        )
        targets = torch.zeros(1, 2)
        row = two_way_prediction_diagnostics(
            logits, targets, easy_negative_cutoff=0.05
        )
        self.assertEqual(row["negative_labels"], 2)
        self.assertEqual(row["easy_negative_labels"], 1)
        self.assertAlmostEqual(row["easy_negative_fraction"], 0.5)

    def test_prf_uses_the_explicit_protocol_threshold(self):
        scores = torch.tensor([[0.6], [0.9]])
        targets = torch.tensor([[1.0], [0.0]])
        at_half = prf_cal(scores, targets, scores, threshold=0.5)
        at_eight = prf_cal(scores, targets, scores, threshold=0.8)
        self.assertAlmostEqual(at_half[-1], 100.0 * 2.0 / 3.0, places=5)
        self.assertEqual(at_eight[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
