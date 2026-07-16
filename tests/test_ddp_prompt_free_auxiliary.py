import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddp_internal_adapter import (
    PromptFreePrototypeObjective,
    SharedResidualFeatureAdapter,
)
from models.ddp import DDP, TextEncoder
from prototype_adapter import ResidualPrototypeAdapter


class RecordingImageEncoder(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.register_buffer("tokens", tokens)
        self.last_visual_prompts = "not-called"

    def forward(self, images, visual_prompts):
        self.last_visual_prompts = visual_prompts
        return self.tokens[: images.shape[0]].clone()


class FakeClipTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Identity()
        self.positional_embedding = nn.Parameter(torch.zeros(4, 3))
        self.ln_final = nn.Identity()
        self.text_projection = nn.Parameter(torch.eye(3))
        self.token_embedding = nn.Embedding(8, 3)
        self.dtype = torch.float32


class DDPPromptFreeAuxiliaryTest(unittest.TestCase):
    def test_prompt_free_image_route_uses_same_encoder_without_prompts(self):
        model = DDP.__new__(DDP)
        nn.Module.__init__(model)
        tokens = torch.tensor(
            [
                [[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]],
            ]
        )
        model.image_encoder = RecordingImageEncoder(tokens)
        model.dtype = torch.float32

        features = model.encode_prompt_free_image(torch.randn(2, 3, 4, 4))

        self.assertIsNone(model.image_encoder.last_visual_prompts)
        self.assertTrue(
            torch.allclose(
                features,
                F.normalize(tokens[:, 0, :], dim=-1),
            )
        )

    def test_fixed_token_embedding_does_not_break_old_checkpoint_keys(self):
        encoder = TextEncoder(FakeClipTextModel())
        self.assertNotIn("token_embedding_weight", encoder.state_dict())
        self.assertIn("text_projection", encoder.state_dict())

    def test_integrated_objective_is_equivalent_to_external_adapter_formula(self):
        torch.manual_seed(3)
        positive = F.normalize(torch.randn(5, 8), dim=-1)
        negative = F.normalize(torch.randn(5, 8), dim=-1)
        features = F.normalize(torch.randn(7, 8), dim=-1)

        external = ResidualPrototypeAdapter(
            positive,
            negative,
            bottleneck_dim=3,
            residual_scale=0.1,
            initial_logit_scale=10.0,
        )
        integrated_adapter = SharedResidualFeatureAdapter(
            feature_dim=8,
            bottleneck_dim=3,
            residual_scale=0.1,
        )
        integrated = PromptFreePrototypeObjective(
            integrated_adapter,
            positive,
            negative,
            initial_logit_scale=10.0,
        )
        with torch.no_grad():
            external.down.weight.copy_(torch.randn_like(external.down.weight))
            external.up.weight.copy_(torch.randn_like(external.up.weight))
            integrated.adapter.down.weight.copy_(external.down.weight)
            integrated.adapter.up.weight.copy_(external.up.weight)
            integrated.logit_scale.copy_(external.logit_scale)

        old_logits, old_adapted, old_original = external(features)
        new_logits, new_adapted, new_original = integrated(features)

        self.assertTrue(torch.allclose(new_logits, old_logits, atol=1e-6))
        self.assertTrue(torch.allclose(new_adapted, old_adapted, atol=1e-6))
        self.assertTrue(torch.allclose(new_original, old_original, atol=1e-6))

    def test_transfer_checkpoint_keeps_existing_down_up_schema(self):
        adapter = SharedResidualFeatureAdapter(8, 3, 0.1)
        state = adapter.state_dict()
        self.assertEqual(set(state), {"down.weight", "up.weight"})


if __name__ == "__main__":
    unittest.main()
