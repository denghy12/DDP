import unittest
from types import SimpleNamespace

import torch

from eval_emotic_prototype_fusion import select_beta, select_threshold


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


if __name__ == "__main__":
    unittest.main()
