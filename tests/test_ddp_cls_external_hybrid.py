import unittest

import torch

from eval_emotic_ddp_cls_external_hybrid import require_aligned_targets


class HybridAlignmentTest(unittest.TestCase):
    def test_accepts_exact_val_test_concatenation(self):
        val = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        test = torch.tensor([[1.0, 1.0]])
        require_aligned_targets(0, val, test, torch.cat([val, test]), 2)

    def test_rejects_mismatched_test_order(self):
        val = torch.tensor([[1.0, 0.0]])
        test = torch.tensor([[0.0, 1.0]])
        external = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        with self.assertRaisesRegex(RuntimeError, "test target alignment"):
            require_aligned_targets(7, val, test, external, 1)


if __name__ == "__main__":
    unittest.main()
