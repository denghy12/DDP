import unittest

import torch

from benchmarks.emotic_mlcil.evaluator import BenchmarkEvaluator
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.types import PredictionOutput, TaskMetrics
from tests.emotic_mlcil import protocol_config


def small_protocol():
    classes = ["a", "b", "c", "d"]
    return BenchmarkProtocol.from_dict(
        protocol_config(classes, [classes[:2], classes[2:]])
    )


def prediction(scores, targets, protocol, split_hash="split-hash"):
    rows = len(scores)
    return PredictionOutput(
        scores=torch.tensor(scores, dtype=torch.float32),
        targets=torch.tensor(targets, dtype=torch.float32),
        sample_ids=[f"sample-{index}" for index in range(rows)],
        class_order_hash=protocol.class_order_hash,
        split_hash=split_hash,
    )


class EvaluatorTest(unittest.TestCase):
    def test_fixed_threshold_metrics_match_repository_definitions(self):
        protocol = small_protocol()
        evaluator = BenchmarkEvaluator(protocol)
        output = prediction(
            scores=[[0.9, 0.2], [0.6, 0.8], [0.1, 0.7]],
            targets=[[1, 0], [0, 1], [0, 1]],
            protocol=protocol,
        )
        row = evaluator.evaluate_task(
            0,
            output,
            expected_sample_ids=output.sample_ids,
            expected_split_hash="split-hash",
        )
        self.assertEqual(row.threshold, 0.5)
        self.assertAlmostEqual(row.per_class_ap[0], 100.0, places=5)
        self.assertAlmostEqual(row.per_class_ap[1], 100.0, places=5)
        self.assertAlmostEqual(row.mAP, 100.0, places=5)
        self.assertAlmostEqual(row.cPrecision, 75.0)
        self.assertAlmostEqual(row.cRecall, 100.0)
        self.assertAlmostEqual(row.cF1, 83.3333333333)
        self.assertAlmostEqual(row.oF1, 85.7142857142)

    def test_map_is_threshold_independent_while_f1_uses_fixed_half(self):
        protocol = small_protocol()
        evaluator = BenchmarkEvaluator(protocol)
        targets = [[1, 0], [0, 1]]
        first = prediction(
            [[0.6, 0.55], [0.55, 0.6]],
            targets,
            protocol,
        )
        second = prediction(
            [[0.9, 0.4], [0.4, 0.9]],
            targets,
            protocol,
        )
        first_row = evaluator.evaluate_task(
            0, first, first.sample_ids, first.split_hash
        )
        second_row = evaluator.evaluate_task(
            0, second, second.sample_ids, second.split_hash
        )
        self.assertEqual(first_row.mAP, second_row.mAP)
        self.assertNotEqual(first_row.cF1, second_row.cF1)
        self.assertEqual(second_row.cF1, 100.0)

    def test_summary_average_final_and_forgetting(self):
        protocol = small_protocol()
        evaluator = BenchmarkEvaluator(protocol)
        rows = [
            TaskMetrics(
                task_id=0,
                seen_classes=2,
                samples=3,
                threshold=0.5,
                per_class_ap=[90.0, 70.0],
                mAP=80.0,
                cPrecision=1.0,
                cRecall=2.0,
                cF1=3.0,
                oPrecision=4.0,
                oRecall=5.0,
                oF1=6.0,
                class_order_hash=protocol.class_order_hash,
                split_hash="task0",
            ),
            TaskMetrics(
                task_id=1,
                seen_classes=4,
                samples=4,
                threshold=0.5,
                per_class_ap=[60.0, 65.0, 80.0, 90.0],
                mAP=73.75,
                cPrecision=10.0,
                cRecall=20.0,
                cF1=30.0,
                oPrecision=40.0,
                oRecall=50.0,
                oF1=60.0,
                class_order_hash=protocol.class_order_hash,
                split_hash="task1",
            ),
        ]
        summary = evaluator.summarize(rows)
        self.assertEqual(summary.final_mAP, 73.75)
        self.assertEqual(summary.average_mAP, 76.875)
        self.assertEqual(summary.final_cF1, 30.0)
        self.assertEqual(summary.final_oF1, 60.0)
        self.assertEqual(summary.per_class_forgetting, {"a": 30.0, "b": 5.0})
        self.assertEqual(summary.forgetting, 17.5)

    def test_nan_scores_are_rejected(self):
        protocol = small_protocol()
        evaluator = BenchmarkEvaluator(protocol)
        output = prediction(
            [[float("nan"), 0.2]],
            [[1, 0]],
            protocol,
        )
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            evaluator.evaluate_task(
                0, output, output.sample_ids, output.split_hash
            )


if __name__ == "__main__":
    unittest.main()
