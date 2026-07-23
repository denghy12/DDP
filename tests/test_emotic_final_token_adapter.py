import unittest

import torch
from torch import nn

from emotic_final_token_adapter import (
    FINAL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
    FINAL_TOKEN_FORMULA,
    SharedResidualFinalTokenAdapter,
    TaskRoutedFinalTokenAdapterBank,
    ddp_pool_final_tokens,
    final_token_drift_tensors,
    final_token_pooling_aware_losses,
    validate_final_token_checkpoint,
)
from models.ddp import DDP
from train_emotic_ddp_final_token_adapter import (
    _cuda_amp_enabled,
    assess_training_health,
)


class MarkerTokenAdapter(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, tokens):
        original = tokens.float()
        adapted = original.clone()
        adapted[..., 0] = self.value
        residual = adapted - original
        return adapted, original, residual


class FinalTokenPoolingTest(unittest.TestCase):
    def test_amp_policy_is_explicit_for_fp32_training(self):
        self.assertTrue(_cuda_amp_enabled("cuda", full_precision=False))
        self.assertFalse(_cuda_amp_enabled("cuda", full_precision=True))
        self.assertFalse(_cuda_amp_enabled("cpu", full_precision=False))

    def test_central_pooling_matches_original_ddp_equations(self):
        tokens = torch.randn(2, 10, 8, 197)
        tokens = torch.nn.functional.normalize(tokens, dim=2)
        text = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
        pooled, logits, token_logits, weights = ddp_pool_final_tokens(tokens, text)

        expected_token_logits = 20 * torch.einsum(
            "bkdn,kd->bkn", tokens, text
        )
        reference = torch.softmax(expected_token_logits[:, 5:, :], dim=-1)
        expected_weights = torch.cat([reference, reference], dim=1)
        expected_pooled = torch.einsum(
            "bkdn,bkn->bkd", tokens, expected_weights
        )
        expected_logits = 5 * (expected_token_logits * expected_weights).sum(-1)
        self.assertTrue(torch.equal(token_logits, expected_token_logits))
        self.assertTrue(torch.equal(weights, expected_weights))
        self.assertTrue(torch.equal(pooled, expected_pooled))
        self.assertTrue(torch.equal(logits, expected_logits))

    def test_zero_initialized_adapter_is_numerical_identity(self):
        adapter = SharedResidualFinalTokenAdapter(8, 3, residual_scale=0.03)
        tokens = torch.nn.functional.normalize(
            torch.randn(2, 10, 197, 8), dim=-1
        )
        adapted, original, residual = adapter(tokens)
        self.assertLess((adapted - original).abs().max().item(), 1e-6)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_zero_adapter_preserves_ddp_logits_within_tolerance(self):
        adapter = SharedResidualFinalTokenAdapter(8, 3, residual_scale=0.03)
        tokens = torch.nn.functional.normalize(
            torch.randn(2, 10, 8, 197), dim=2
        )
        text = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
        _, baseline, _, _ = ddp_pool_final_tokens(tokens, text)
        adapted, _, _ = adapter(tokens.permute(0, 1, 3, 2))
        _, candidate, _, _ = ddp_pool_final_tokens(
            adapted.permute(0, 1, 3, 2), text
        )
        self.assertTrue(torch.allclose(candidate, baseline, atol=1e-4, rtol=1e-6))

    def test_zero_initialized_adapter_receives_gradient_and_leaves_identity(self):
        torch.manual_seed(7)
        adapter = SharedResidualFinalTokenAdapter(8, 3, residual_scale=0.03)
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.5)
        tokens = torch.nn.functional.normalize(
            torch.randn(2, 4, 11, 8), dim=-1
        )
        target = torch.randn_like(tokens)

        adapted, _, residual = adapter(tokens)
        self.assertEqual(residual.abs().max().item(), 0.0)
        loss = (adapted * target).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNotNone(adapter.up.weight.grad)
        self.assertGreater(adapter.up.weight.grad.norm().item(), 0.0)
        optimizer.step()

        adapted, original, residual = adapter(tokens)
        self.assertGreater(residual.norm().item(), 0.0)
        self.assertGreater((adapted - original).norm().item(), 0.0)
        loss = (adapted * target).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNotNone(adapter.down.weight.grad)
        self.assertGreater(adapter.down.weight.grad.norm().item(), 0.0)

    def test_ddp_token_entry_returns_two_way_logits(self):
        model = DDP.__new__(DDP)
        nn.Module.__init__(model)
        model.final_token_adapter_bank = None
        tokens = torch.nn.functional.normalize(
            torch.randn(2, 10, 8, 197), dim=2
        )
        text = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
        logits = model.logits_from_token_features(tokens, text)
        self.assertEqual(tuple(logits.shape), (2, 2, 5))

    def test_parameter_count_does_not_depend_on_token_count(self):
        adapter = SharedResidualFinalTokenAdapter(512, 128, residual_scale=0.03)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 131072)

    def test_drift_diagnostics_are_zero_for_identical_tokens(self):
        tokens = torch.nn.functional.normalize(torch.randn(2, 4, 5, 8), dim=-1)
        text = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        drift = final_token_drift_tensors(tokens, tokens, text)
        self.assertLess(
            drift["attention_kl_original_to_adapted"].abs().max().item(), 1e-7
        )
        self.assertLess(
            drift["pooled_feature_cosine_drift"].abs().max().item(), 1e-6
        )
        self.assertEqual(drift["path_logit_delta"].abs().max().item(), 0.0)
        self.assertEqual(drift["token_delta_l2"].abs().max().item(), 0.0)

    def test_drift_diagnostics_separate_cls_and_patch_tokens(self):
        original = torch.nn.functional.normalize(torch.randn(2, 4, 5, 8), dim=-1)
        adapted = original.clone()
        adapted[:, :, 0, :] = torch.nn.functional.normalize(
            adapted[:, :, 0, :] + 0.1, dim=-1
        )
        text = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        drift = final_token_drift_tensors(adapted, original, text)
        token_delta = drift["token_delta_l2"]
        self.assertGreater(token_delta[:, :, 0].mean().item(), 0.0)
        self.assertEqual(token_delta[:, :, 1:].abs().max().item(), 0.0)
        self.assertGreater(drift["path_logit_delta"].abs().max().item(), 0.0)

    def test_pooling_aware_losses_are_zero_at_identity(self):
        tokens = torch.nn.functional.normalize(torch.randn(2, 4, 5, 8), dim=-1)
        text = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        losses = final_token_pooling_aware_losses(tokens, tokens, text)
        self.assertLess(losses["pooling_loss"].abs().item(), 1e-6)
        self.assertLess(losses["attention_loss"].abs().item(), 1e-7)
        self.assertEqual(losses["margin_loss"].item(), 0.0)

    def test_attention_kl_has_finite_gradient_for_nonzero_adapter(self):
        torch.manual_seed(19)
        original = torch.nn.functional.normalize(
            torch.randn(2, 6, 197, 8), dim=-1
        )
        adapted = torch.nn.functional.normalize(
            original + 0.05 * torch.randn_like(original), dim=-1
        ).requires_grad_(True)
        text = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
        losses = final_token_pooling_aware_losses(adapted, original, text)
        losses["attention_loss"].backward()
        self.assertTrue(torch.isfinite(losses["attention_loss"]))
        self.assertIsNotNone(adapted.grad)
        self.assertTrue(torch.isfinite(adapted.grad).all())

    def test_pooling_aware_losses_backpropagate_to_adapted_tokens(self):
        original = torch.nn.functional.normalize(torch.randn(2, 4, 5, 8), dim=-1)
        adapted = torch.nn.functional.normalize(original + 0.03, dim=-1)
        adapted = adapted.detach().requires_grad_(True)
        text = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        losses = final_token_pooling_aware_losses(adapted, original, text)
        total = (
            100.0 * losses["pooling_loss"]
            + 100.0 * losses["attention_loss"]
            + losses["margin_loss"]
        )
        total.backward()
        self.assertIsNotNone(adapted.grad)
        self.assertGreater(adapted.grad.norm().item(), 0.0)


class FinalTokenRoutingTest(unittest.TestCase):
    def test_negative_and_positive_paths_share_introduction_task_adapter(self):
        bank = TaskRoutedFinalTokenAdapterBank(
            {0: MarkerTokenAdapter(1.0), 1: MarkerTokenAdapter(2.0)}
        )
        tokens = torch.zeros(1, 16, 3, 5)
        adapted = bank.adapt_token_features(tokens, seen_classes=8)
        task0 = list(range(0, 5)) + list(range(8, 13))
        task1 = list(range(5, 8)) + list(range(13, 16))
        self.assertTrue(
            torch.equal(adapted[0, task0, 0], torch.ones(10, 5))
        )
        self.assertTrue(
            torch.equal(adapted[0, task1, 0], torch.full((6, 5), 2.0))
        )

    def test_future_adapter_does_not_change_old_class_tokens(self):
        source = torch.zeros(1, 10, 3, 5)
        bank0 = TaskRoutedFinalTokenAdapterBank({0: MarkerTokenAdapter(1.0)})
        bank1 = TaskRoutedFinalTokenAdapterBank(
            {0: MarkerTokenAdapter(1.0), 1: MarkerTokenAdapter(2.0)}
        )
        self.assertTrue(
            torch.equal(
                bank0.adapt_token_features(source, seen_classes=5),
                bank1.adapt_token_features(source, seen_classes=5),
            )
        )

    def test_missing_current_task_adapter_is_rejected(self):
        bank = TaskRoutedFinalTokenAdapterBank({0: MarkerTokenAdapter(1.0)})
        with self.assertRaises(ValueError):
            bank.adapt_token_features(torch.zeros(1, 16, 3, 5), seen_classes=8)


class FinalTokenCheckpointTest(unittest.TestCase):
    def test_checkpoint_requires_locked_final_token_formula(self):
        checkpoint = {
            "model": {"down.weight": torch.zeros(2, 2)},
            "classnames": [f"class{index}" for index in range(26)],
            "final_token_adapter": {
                "schema_version": FINAL_TOKEN_CHECKPOINT_SCHEMA_VERSION,
                "task_id": 2,
                "class_range": [8, 11],
                "training_mode": "full",
                "seed": 1,
                "formula": FINAL_TOKEN_FORMULA,
            },
        }
        metadata = validate_final_token_checkpoint(
            checkpoint, task_id=2, training_mode="full", seed=1
        )
        self.assertEqual(metadata["task_id"], 2)
        checkpoint["final_token_adapter"]["formula"] = "logit_correction"
        with self.assertRaises(ValueError):
            validate_final_token_checkpoint(checkpoint)


class FinalTokenTrainingHealthTest(unittest.TestCase):
    def test_health_audit_accepts_expected_zero_init_training_sequence(self):
        first = {
            "loss": 1.0,
            "up_grad_norm": 0.2,
            "down_grad_norm": 0.0,
            "up_weight_delta_norm": 0.01,
            "raw_residual_rms": 0.0,
            "adapted_feature_delta_rms": 0.0,
        }
        second = {
            "loss": 0.9,
            "up_grad_norm": 0.1,
            "down_grad_norm": 0.03,
            "up_weight_delta_norm": 0.02,
            "raw_residual_rms": 0.04,
            "adapted_feature_delta_rms": 0.001,
        }
        report = assess_training_health([first, second])
        self.assertEqual(report["status"], "passed")
        self.assertTrue(all(report["checks"].values()))

    def test_health_audit_rejects_dead_gradient_sequence(self):
        dead = {
            "loss": 1.0,
            "up_grad_norm": 0.0,
            "down_grad_norm": 0.0,
            "up_weight_delta_norm": 0.0,
            "raw_residual_rms": 0.0,
            "adapted_feature_delta_rms": 0.0,
        }
        report = assess_training_health([dead, dead])
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["up_projection_receives_gradient"])


if __name__ == "__main__":
    unittest.main()
