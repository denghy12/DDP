import unittest

import torch

from benchmarks.emotic_mlcil.evaluator import BenchmarkEvaluator
from benchmarks.emotic_mlcil.types import PredictionOutput
from tests.emotic_mlcil import make_protocol


def valid_prediction(protocol):
    return PredictionOutput(
        scores=torch.rand(3, 5),
        targets=torch.zeros(3, 5),
        sample_ids=["a", "b", "c"],
        class_order_hash=protocol.class_order_hash,
        split_hash="expected-split",
    )


class ScoreAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.protocol = make_protocol()
        self.evaluator = BenchmarkEvaluator(self.protocol)

    def validate(self, output, ids=("a", "b", "c"), split_hash="expected-split"):
        self.evaluator.validate_prediction(
            0,
            output,
            expected_sample_ids=ids,
            expected_split_hash=split_hash,
        )

    def test_valid_alignment_is_accepted(self):
        self.validate(valid_prediction(self.protocol))

    def test_sample_id_reordering_is_rejected(self):
        output = valid_prediction(self.protocol)
        with self.assertRaisesRegex(ValueError, "sample ID order"):
            self.validate(output, ids=("b", "a", "c"))

    def test_duplicate_sample_ids_are_rejected(self):
        output = valid_prediction(self.protocol)
        output.sample_ids = ["a", "a", "c"]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.validate(output, ids=output.sample_ids)

    def test_class_order_hash_mismatch_is_rejected(self):
        output = valid_prediction(self.protocol)
        output.class_order_hash = "wrong"
        with self.assertRaisesRegex(ValueError, "class_order_hash"):
            self.validate(output)

    def test_split_hash_mismatch_is_rejected(self):
        output = valid_prediction(self.protocol)
        output.split_hash = "wrong"
        with self.assertRaisesRegex(ValueError, "split_hash"):
            self.validate(output)

    def test_seen_class_shape_mismatch_is_rejected(self):
        output = valid_prediction(self.protocol)
        output.scores = torch.rand(3, 6)
        output.targets = torch.zeros(3, 6)
        with self.assertRaisesRegex(ValueError, "seen classes"):
            self.validate(output)

    def test_non_binary_targets_are_rejected(self):
        output = valid_prediction(self.protocol)
        output.targets[0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            self.validate(output)


if __name__ == "__main__":
    unittest.main()
