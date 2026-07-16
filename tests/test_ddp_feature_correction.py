import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from ddp_internal_adapter import (
    feature_logit_correction,
    norm_preserving_feature_correction,
)
from models.ddp import DDP
from eval_emotic_ddp_internal_adapter import predict_from_cache
from screen_emotic_ddp_internal_adapter_transfer import validation_map


class FixedAdapter(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", delta)

    def forward(self, features):
        original = features.float()
        return original + self.delta, original


class IdentityAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_scale = 0.0

    def forward(self, features):
        return features, features


class CaptureFeatureCorrectionModel:
    feature_adapter_correction = "feature_correction"

    def __init__(self):
        self.received_pooled = None

    def logits_from_path_features(
        self,
        features,
        base_logits,
        text_features,
        ddp_pooled_features=None,
    ):
        del features, text_features
        self.received_pooled = ddp_pooled_features.detach().cpu()
        return base_logits.reshape(base_logits.shape[0], 2, -1)


class DDPFeatureCorrectionTest(unittest.TestCase):
    def test_preserves_each_pooled_feature_norm(self):
        torch.manual_seed(13)
        pooled = torch.randn(2, 6, 8)
        delta = 0.2 * torch.randn_like(pooled)
        corrected = norm_preserving_feature_correction(pooled, delta)
        self.assertTrue(
            torch.allclose(
                corrected.norm(dim=-1),
                pooled.norm(dim=-1),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_zero_delta_is_exact_identity_and_zero_logit_change(self):
        pooled = torch.randn(2, 4, 7)
        text = torch.randn(4, 7)
        corrected = norm_preserving_feature_correction(
            pooled, torch.zeros_like(pooled)
        )
        self.assertTrue(torch.equal(corrected, pooled.float()))
        correction = feature_logit_correction(
            pooled,
            pooled,
            text,
            mode="feature_correction",
            pooled_features=pooled,
        )
        self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))

    def test_matches_explicit_norm_preserving_formula(self):
        torch.manual_seed(17)
        cls = torch.randn(2, 6, 8)
        delta = 0.1 * torch.randn_like(cls)
        pooled = torch.randn(2, 6, 8)
        text = torch.randn(6, 8)
        adapted = cls + delta
        # The model receives only the Adapter output and its original input,
        # so its effective offset is the float32 subtraction below.  Reuse
        # exactly that offset in the explicit formula instead of the
        # pre-addition ``delta`` tensor, which can differ by a few ulps after
        # (cls + delta) - cls cancellation and the 100x logit scale.
        effective_delta = adapted - cls
        corrected = norm_preserving_feature_correction(
            pooled, effective_delta
        )
        expected = 100.0 * torch.einsum(
            "bkd,kd->bk", corrected - pooled, text
        )
        actual = feature_logit_correction(
            adapted,
            cls,
            text,
            mode="feature_correction",
            pooled_features=pooled,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_model_uses_cls_for_delta_and_pooled_for_feature_update(self):
        torch.manual_seed(19)
        model = DDP.__new__(DDP)
        nn.Module.__init__(model)
        cls = torch.randn(2, 10, 8)
        pooled = torch.randn(2, 10, 8)
        delta = 0.05 * torch.randn_like(cls)
        text = torch.randn(10, 8)
        base = torch.randn(2, 10)
        model.feature_adapter = FixedAdapter(delta)
        model.feature_adapter_correction = "feature_correction"

        logits, aux = model.logits_from_path_features(
            cls,
            base,
            text,
            return_adapter_aux=True,
            ddp_pooled_features=pooled,
        )
        corrected = norm_preserving_feature_correction(pooled, delta)
        expected = base + 100.0 * torch.einsum(
            "bkd,kd->bk", corrected - pooled, text
        )
        self.assertTrue(torch.allclose(logits, expected.reshape(2, 2, 5)))
        self.assertTrue(torch.allclose(aux["corrected_pooled"], corrected))
        self.assertEqual(aux["correction_mode"], "feature_correction")

    def test_requires_original_ddp_pooled_features(self):
        features = torch.randn(2, 4, 7)
        text = torch.randn(4, 7)
        with self.assertRaisesRegex(ValueError, "requires.*pooled_features"):
            feature_logit_correction(
                features,
                features,
                text,
                mode="feature_correction",
            )

    def test_validation_rejects_legacy_cls_cache(self):
        payload = {
            "path_features": torch.randn(4, 2, 3),
            "base_path_logits": torch.randn(4, 2),
            "text_features": torch.randn(2, 3),
            "targets": torch.zeros(4, 1),
        }
        with self.assertRaisesRegex(KeyError, "requires cache key"):
            validation_map(
                IdentityAdapter(),
                payload,
                "path_features",
                0.03,
                4,
                "cpu",
                correction_mode="feature_correction",
            )

    def test_cached_prediction_passes_paired_pooled_features_to_model(self):
        cls = torch.randn(3, 2, 4)
        pooled = torch.randn(3, 2, 4)
        base = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]])
        payload = {
            "path_features": cls,
            "pooled_features": pooled,
            "base_path_logits": base,
            "text_features": torch.randn(2, 4),
        }
        args = SimpleNamespace(
            adapter_batch_size=3,
            device="cpu",
            ddp_only=False,
            t_min=1.0,
            t_max=2.0,
            t_gamma=0.7,
            feature_source="cls",
        )
        model = CaptureFeatureCorrectionModel()
        scores, _ = predict_from_cache(model, payload, 1, args)
        self.assertTrue(torch.equal(model.received_pooled, pooled))
        self.assertEqual(scores.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
