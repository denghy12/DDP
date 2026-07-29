import unittest

import torch
import torch.nn.functional as F

from emotic_multilabel_losses import (
    MaskedAsymmetricLoss,
    balanced_positive_class_weights,
    build_asymmetric_loss,
)


class MaskedAsymmetricLossTest(unittest.TestCase):
    def test_zero_gamma_zero_clip_equals_masked_bce(self):
        logits = torch.tensor([[0.2, -1.0], [2.0, 0.5]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mask = torch.tensor([[True, False], [True, True]])
        actual = MaskedAsymmetricLoss(
            gamma_neg=0, gamma_pos=0, clip=0
        )(logits, targets, mask)
        elements = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        expected = elements[mask].mean()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_masked_entries_have_zero_gradient(self):
        logits = torch.tensor([[0.1, -0.2]], requires_grad=True)
        targets = torch.tensor([[1.0, 1.0]])
        mask = torch.tensor([[True, False]])
        loss = MaskedAsymmetricLoss()(logits, targets, mask)
        loss.backward()
        self.assertNotEqual(float(logits.grad[0, 0]), 0.0)
        self.assertEqual(float(logits.grad[0, 1]), 0.0)

    def test_asl_suppresses_an_easy_negative(self):
        bce_logits = torch.tensor([[-5.0]], requires_grad=True)
        target = torch.zeros_like(bce_logits)
        mask = torch.ones_like(bce_logits, dtype=torch.bool)
        bce = F.binary_cross_entropy_with_logits(bce_logits, target)
        bce.backward()
        bce_gradient = bce_logits.grad.abs().item()

        asl_logits = torch.tensor([[-5.0]], requires_grad=True)
        asl = MaskedAsymmetricLoss(
            gamma_neg=4.0, gamma_pos=0.0, clip=0.05
        )(asl_logits, target, mask)
        asl.backward()
        self.assertLess(asl_logits.grad.abs().item(), bce_gradient)

    def test_bal_gives_the_rarest_positive_the_largest_weight(self):
        targets = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        mask = torch.ones_like(targets, dtype=torch.bool)
        weights = balanced_positive_class_weights(targets, mask, power=1.6)
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[1]), float(weights[2]))
        self.assertAlmostEqual(float(weights[2]), 1.0)

    def test_bal_with_unit_weights_and_no_smoothing_equals_asl(self):
        logits = torch.tensor([[0.3, -0.6], [-0.2, 0.8]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        asl = MaskedAsymmetricLoss(
            gamma_neg=4, gamma_pos=0, clip=0.05
        )(logits, targets, mask)
        bal = MaskedAsymmetricLoss(
            gamma_neg=4,
            gamma_pos=0,
            clip=0.05,
            class_weights=torch.ones(2),
            label_smoothing=0,
            positive_weight_mode="paper",
        )(logits, targets, mask)
        self.assertTrue(torch.allclose(asl, bal, atol=1e-7))

    def test_strict_16shot_bal_weights_are_constant(self):
        targets = torch.zeros(48, 3)
        targets[:16, 0] = 1
        targets[16:32, 1] = 1
        targets[32:, 2] = 1
        mask = torch.ones_like(targets, dtype=torch.bool)
        weights = balanced_positive_class_weights(targets, mask, power=1.6)
        self.assertTrue(torch.equal(weights, torch.ones(3)))

    def test_build_loss_records_visible_train_counts(self):
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        mask = torch.tensor([[True, True], [False, True], [True, True]])
        _, diagnostics = build_asymmetric_loss(
            "bal_paper",
            targets,
            mask,
            gamma_neg=9.8,
            gamma_pos=0,
            clip=0.05,
            bal_weight_power=1.6,
            bal_label_smoothing=0.1,
            smoothing_num_classes=26,
        )
        self.assertEqual(diagnostics["visible_positives"], [1.0, 1.0])
        self.assertEqual(diagnostics["visible_negatives"], [1.0, 2.0])
        self.assertEqual(diagnostics["smoothing_num_classes"], 26)

    def test_bal_component_ablation_activates_one_component_at_a_time(self):
        targets = torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        )
        mask = torch.ones_like(targets, dtype=torch.bool)
        common = {
            "targets": targets,
            "supervision_mask": mask,
            "gamma_neg": 9.8,
            "gamma_pos": 0.0,
            "clip": 0.05,
            "bal_weight_power": 1.6,
            "bal_label_smoothing": 0.1,
            "smoothing_num_classes": 26,
        }
        expected = {
            "asl": (False, False),
            "asl_smoothing": (False, True),
            "asl_positive_weight": (True, False),
            "bal_paper": (True, True),
        }
        for loss_name, components in expected.items():
            with self.subTest(loss_name=loss_name):
                _, diagnostics = build_asymmetric_loss(
                    loss_name, **common
                )
                uses_positive_weight, uses_smoothing = components
                self.assertEqual(
                    diagnostics["uses_positive_class_weights"],
                    uses_positive_weight,
                )
                self.assertEqual(
                    diagnostics["uses_label_smoothing"],
                    uses_smoothing,
                )
                self.assertEqual(
                    diagnostics["class_weights"] is not None,
                    uses_positive_weight,
                )
                self.assertEqual(
                    diagnostics["label_smoothing"] > 0,
                    uses_smoothing,
                )


if __name__ == "__main__":
    unittest.main()
