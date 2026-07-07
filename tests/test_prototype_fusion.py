import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path

import torch

from eval_emotic_prototype_fusion import (
    load_prototype_model,
    select_beta,
    select_threshold,
    split_metrics,
)
from prototype_adapter import ResidualPrototypeAdapter


class PrototypeFusionSelectionTest(unittest.TestCase):
    def test_beta_selection_prefers_better_prototype_ranking(self):
        targets = torch.tensor(
            [[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32
        )
        ddp_scores = torch.tensor([[0.4], [0.6], [0.3], [0.7]])
        prototype_scores = torch.tensor([[0.9], [0.1], [0.8], [0.2]])
        best, _ = select_beta(
            targets, ddp_scores, prototype_scores, step=0.1
        )
        # beta=0.5 already yields the optimal ranking in this example. The
        # selector intentionally keeps the smallest beta when mAP is tied.
        self.assertGreaterEqual(best["beta"], 0.5)

    def test_threshold_selection_uses_validation_f1(self):
        targets = torch.tensor(
            [[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32
        )
        scores = torch.tensor([[0.8], [0.7], [0.3], [0.2]])
        args = SimpleNamespace(
            threshold_min=0.1,
            threshold_max=0.9,
            threshold_step=0.1,
        )
        best, _ = select_threshold(targets, scores, args)
        self.assertGreater(best["cF1"], 99.9)
        self.assertGreater(best["oF1"], 99.9)

    def test_strict_split_does_not_report_val_test(self):
        scores = torch.tensor([[0.9], [0.1], [0.8], [0.2]])
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        result = split_metrics(scores, targets, 2, 0.5, ["emotion"])
        self.assertEqual(set(result), {"val", "test"})

    def test_zero_shot_loader_ignores_learned_adapter_weights(self):
        positive = torch.eye(2)
        negative = -torch.eye(2)
        learned = ResidualPrototypeAdapter(positive, negative, bottleneck_dim=1)
        with torch.no_grad():
            learned.up.weight.fill_(4.0)
        checkpoint = {
            "model": learned.state_dict(),
            "classnames": ["a", "b"],
            "args": {"residual_scale": 0.1, "initial_logit_scale": 10.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pth"
            torch.save(checkpoint, path)
            zero_shot, _, _ = load_prototype_model(path, torch.device("cpu"), True)
        features = torch.tensor([[0.6, 0.8]])
        adapted, original = zero_shot.adapt(features)
        self.assertTrue(torch.allclose(adapted, original))


if __name__ == "__main__":
    unittest.main()
