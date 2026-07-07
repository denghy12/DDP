import unittest

import torch
import torch.nn.functional as F

from prototype_fewshot import (
    masked_bce_with_logits,
    masked_pos_weight,
    sample_multilabel_kshot,
)


class PrototypeFewShotTest(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor(
            [
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=torch.float32,
        )

    def test_sampler_has_exact_positive_supervision_per_class(self):
        selected, mask, rows = sample_multilabel_kshot(
            self.labels, [0, 1, 2], shots_per_class=2, seed=0
        )
        subset = self.labels[selected]
        for class_id in range(3):
            supervised_positive = (
                subset[:, class_id].bool() & mask[:, class_id]
            ).sum()
            self.assertEqual(supervised_positive.item(), 2)
            # Every true negative in the union remains supervised.
            self.assertTrue(
                mask[subset[:, class_id].eq(0), class_id].all().item()
            )
        self.assertEqual(
            [row["supervised_positives"] for row in rows], [2, 2, 2]
        )

    def test_sampler_is_deterministic_for_one_seed(self):
        first = sample_multilabel_kshot(self.labels, [0, 1, 2], 2, 7)
        second = sample_multilabel_kshot(self.labels, [0, 1, 2], 2, 7)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_masked_bce_matches_standard_bce_for_full_mask(self):
        logits = torch.tensor([[0.2, -0.3], [0.8, -1.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        expected = F.binary_cross_entropy_with_logits(logits, targets)
        actual = masked_bce_with_logits(logits, targets, mask)
        self.assertTrue(torch.allclose(actual, expected))

    def test_masked_pos_weight_uses_only_supervised_entries(self):
        labels = torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0]])
        mask = torch.tensor([[True], [False], [True], [True], [False]])
        weight = masked_pos_weight(labels, mask, [0])
        self.assertEqual(weight.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
