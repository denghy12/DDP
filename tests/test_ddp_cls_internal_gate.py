import unittest

import torch

from eval_emotic_ddp_cls_internal_gate import (
    apply_class_gates,
    select_class_gates,
    select_task_alpha,
)


class DDPCLSInternalGateTest(unittest.TestCase):
    def test_task_alpha_requires_validation_margin(self):
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        identity = torch.tensor([[0.9], [0.1], [0.8], [0.2]])
        tied = identity + 1e-5
        selected, _, _ = select_task_alpha(
            targets,
            {0.0: identity, 0.01: tied},
            [0.0, 0.01],
            minimum_gain=0.1,
        )
        self.assertEqual(selected["alpha"], 0.0)

    def test_class_gate_keeps_ddp_for_hurt_class(self):
        targets = torch.tensor(
            [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
        )
        baseline = torch.tensor(
            [[0.9, 0.4], [0.1, 0.6], [0.8, 0.3], [0.2, 0.7]]
        )
        adapted = torch.tensor(
            [[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]]
        )
        gates, _ = select_class_gates(
            targets,
            baseline,
            adapted,
            ["strong_ddp", "strong_cls"],
            minimum_gain=0.1,
        )
        self.assertEqual(gates.tolist(), [False, True])
        fused = apply_class_gates(baseline, adapted, gates)
        self.assertTrue(torch.equal(fused[:, 0], baseline[:, 0]))
        self.assertTrue(torch.equal(fused[:, 1], adapted[:, 1]))


if __name__ == "__main__":
    unittest.main()
