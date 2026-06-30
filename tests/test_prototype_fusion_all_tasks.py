import unittest

import torch

from eval_emotic_prototype_fusion_all_tasks import (
    TASK_SEEN_CLASSES,
    forgetting_summary,
    fuse_classwise,
    introduction_tasks,
    seen_sample_mask,
    select_binary_class_gates,
)


class AllTaskPrototypeFusionTest(unittest.TestCase):
    def test_seen_sample_mask_matches_incremental_filter(self):
        labels = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        self.assertEqual(
            seen_sample_mask(labels, 5).tolist(), [True, False, False]
        )
        self.assertEqual(
            seen_sample_mask(labels, 8).tolist(), [True, True, True]
        )

    def test_binary_gate_keeps_ddp_for_one_class_and_fuses_another(self):
        targets = torch.tensor(
            [[1, 1], [0, 0], [1, 1], [0, 0]], dtype=torch.float32
        )
        ddp = torch.tensor(
            [[0.9, 0.4], [0.1, 0.6], [0.8, 0.3], [0.2, 0.7]]
        )
        prototype = torch.tensor(
            [[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]]
        )
        betas, _ = select_binary_class_gates(
            targets, ddp, prototype, global_beta=0.8
        )
        self.assertEqual(betas[0].item(), 0.0)
        self.assertAlmostEqual(betas[1].item(), 0.8)
        fused = fuse_classwise(ddp, prototype, betas)
        self.assertTrue(torch.equal(fused[:, 0], ddp[:, 0]))
        self.assertFalse(torch.equal(fused[:, 1], ddp[:, 1]))

    def test_introduction_tasks_follow_b5c3(self):
        tasks = introduction_tasks(26)
        counts = [tasks.count(task_id) for task_id in range(8)]
        starts = (0,) + TASK_SEEN_CLASSES[:-1]
        expected = tuple(
            end - start for start, end in zip(starts, TASK_SEEN_CLASSES)
        )
        self.assertEqual(tuple(counts), expected)

    def test_forgetting_is_peak_minus_final_for_old_classes(self):
        task_summaries = []
        classnames = ["a", "b"]
        for values in ((80.0, 20.0), (70.0, 50.0)):
            task_summaries.append(
                {
                    "results": {
                        "ddp": {
                            "test": {
                                "per_class_ap": dict(zip(classnames, values))
                            }
                        }
                    }
                }
            )
        summary = forgetting_summary(
            task_summaries, "ddp", "test", classnames
        )
        self.assertAlmostEqual(summary["per_class"][0]["forgetting"], 10.0)
        self.assertAlmostEqual(
            summary["average_forgetting_old_classes"], 5.0
        )


if __name__ == "__main__":
    unittest.main()
