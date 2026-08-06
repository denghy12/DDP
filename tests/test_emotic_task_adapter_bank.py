import unittest

import torch
from torch import nn

from emotic_task_adapter_bank import (
    TASK_CLASS_RANGES,
    TaskRoutedAdapterBank,
    class_task_id,
    class_to_task_map,
    normalize_task_class_ranges,
    prepare_task_training_subset,
    task_class_range,
    task_ranges_from_sizes,
    validate_task_adapter_checkpoint,
)


class ConstantResidualAdapter(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)
        self.residual_scale = 1.0

    def forward(self, features):
        original = features.float()
        adapted = original + self.residual_scale * self.value
        return adapted, original


class TaskDefinitionTest(unittest.TestCase):
    def test_b5c3_ranges_and_class_routes(self):
        self.assertEqual(task_class_range(0), (0, 5))
        self.assertEqual(task_class_range(7), (23, 26))
        self.assertEqual(class_task_id(0), 0)
        self.assertEqual(class_task_id(4), 0)
        self.assertEqual(class_task_id(5), 1)
        self.assertEqual(class_task_id(25), 7)
        mapping = class_to_task_map()
        self.assertEqual(len(mapping), 26)
        for task_id, (low, high) in enumerate(TASK_CLASS_RANGES):
            self.assertTrue(all(mapping[index] == task_id for index in range(low, high)))

    def test_invalid_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            task_class_range(8)
        with self.assertRaises(ValueError):
            class_task_id(26)

    def test_b10c4_and_b4c2_ranges_are_protocol_driven(self):
        b10c4 = task_ranges_from_sizes([10, 4, 4, 4, 4])
        self.assertEqual(b10c4, ((0, 10), (10, 14), (14, 18), (18, 22), (22, 26)))
        self.assertEqual(class_task_id(9, b10c4), 0)
        self.assertEqual(class_task_id(10, b10c4), 1)
        self.assertEqual(class_task_id(25, b10c4), 4)
        self.assertEqual(class_to_task_map(None, b10c4).count(0), 10)

        b4c2 = task_ranges_from_sizes([4] + [2] * 11)
        self.assertEqual(len(b4c2), 12)
        self.assertEqual(task_class_range(11, b4c2), (24, 26))
        self.assertEqual(class_task_id(4, b4c2), 1)
        with self.assertRaises(ValueError):
            normalize_task_class_ranges(((0, 4), (5, 7)))


class TaskSubsetTest(unittest.TestCase):
    def test_full_subset_masks_old_and_future_labels(self):
        labels = torch.zeros(5, 26)
        labels[0, [1, 5, 15]] = 1
        labels[1, [6, 7]] = 1
        labels[2, [0]] = 1
        labels[3, [10]] = 1
        labels[4, [5]] = 1
        selected, mask, sampling = prepare_task_training_subset(
            labels, task_id=1, training_mode="full", seed=0
        )
        self.assertEqual(selected.tolist(), [0, 1, 4])
        self.assertTrue(mask[:, 5:8].all())
        self.assertFalse(mask[:, :5].any())
        self.assertFalse(mask[:, 8:].any())
        self.assertEqual(sampling["ignored_old_positive_labels"], 1)
        self.assertEqual(sampling["ignored_future_positive_labels"], 1)

    def test_16shot_supervises_exactly_16_positive_anchors(self):
        labels = torch.zeros(70, 26)
        labels[:30, 5] = 1
        labels[20:50, 6] = 1
        labels[40:70, 7] = 1
        labels[:10, 2] = 1
        labels[60:, 20] = 1
        selected, mask, sampling = prepare_task_training_subset(
            labels, task_id=1, training_mode="16shot", seed=2
        )
        self.assertLessEqual(selected.numel(), 48)
        self.assertFalse(mask[:, :5].any())
        self.assertFalse(mask[:, 8:].any())
        for row in sampling["sampling"]:
            self.assertEqual(row["selected_positive_anchors"], 16)
            self.assertEqual(row["supervised_positives"], 16)


class TaskRoutingTest(unittest.TestCase):
    def test_negative_and_positive_paths_use_the_same_task_adapter(self):
        bank = TaskRoutedAdapterBank(
            {0: ConstantResidualAdapter(1.0), 1: ConstantResidualAdapter(2.0)},
            inference_alpha=0.5,
        )
        features = torch.zeros(1, 16, 2)
        text = torch.ones(16, 2)
        correction = bank.feature_difference_correction(features, text, seen_classes=8)
        expected_task0 = 100.0
        expected_task1 = 200.0
        task0_paths = list(range(0, 5)) + list(range(8, 13))
        task1_paths = list(range(5, 8)) + list(range(13, 16))
        self.assertTrue(
            torch.allclose(correction[0, task0_paths], torch.full((10,), expected_task0))
        )
        self.assertTrue(
            torch.allclose(correction[0, task1_paths], torch.full((6,), expected_task1))
        )

    def test_adding_new_adapter_does_not_change_old_class_logits(self):
        features = torch.randn(2, 10, 3)
        text = torch.randn(10, 3)
        base = torch.randn(2, 10)
        bank0 = TaskRoutedAdapterBank(
            {0: ConstantResidualAdapter(1.0)}, inference_alpha=0.03
        )
        bank1 = TaskRoutedAdapterBank(
            {0: ConstantResidualAdapter(1.0), 1: ConstantResidualAdapter(2.0)},
            inference_alpha=0.03,
        )
        logits0 = bank0.logits_from_features(features, base, text, seen_classes=5)
        logits1 = bank1.logits_from_features(features, base, text, seen_classes=5)
        self.assertTrue(torch.equal(logits0, logits1))

    def test_missing_current_task_adapter_is_rejected(self):
        bank = TaskRoutedAdapterBank(
            {0: ConstantResidualAdapter(1.0)}, inference_alpha=0.03
        )
        with self.assertRaises(ValueError):
            bank.feature_difference_correction(
                torch.zeros(1, 16, 2), torch.ones(16, 2), seen_classes=8
            )

    def test_b10c4_routes_both_paths_with_protocol_ranges(self):
        ranges = task_ranges_from_sizes([10, 4, 4, 4, 4])
        bank = TaskRoutedAdapterBank(
            {0: ConstantResidualAdapter(1.0), 1: ConstantResidualAdapter(2.0)},
            inference_alpha=0.5,
            task_class_ranges=ranges,
        )
        features = torch.zeros(1, 28, 2)
        text = torch.ones(28, 2)
        correction = bank.feature_difference_correction(
            features, text, seen_classes=14
        )
        task0 = list(range(0, 10)) + list(range(14, 24))
        task1 = list(range(10, 14)) + list(range(24, 28))
        self.assertTrue(torch.allclose(correction[0, task0], torch.full((20,), 100.0)))
        self.assertTrue(torch.allclose(correction[0, task1], torch.full((8,), 200.0)))

    def test_autocast_correction_is_assembled_as_float32(self):
        bank = TaskRoutedAdapterBank(
            {0: ConstantResidualAdapter(1.0)}, inference_alpha=0.5
        )
        features = torch.zeros(1, 10, 2)
        text = torch.ones(10, 2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            correction = bank.feature_difference_correction(
                features,
                text,
                seen_classes=5,
            )
        self.assertEqual(correction.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(correction, torch.full((1, 10), 100.0))
        )


class CheckpointProtocolTest(unittest.TestCase):
    def test_checkpoint_metadata_validation(self):
        checkpoint = {
            "model": {"down.weight": torch.zeros(2, 2)},
            "classnames": [f"class{index}" for index in range(26)],
            "task_adapter": {
                "task_id": 2,
                "class_range": [8, 11],
                "training_mode": "16shot",
                "seed": 1,
            },
        }
        metadata = validate_task_adapter_checkpoint(
            checkpoint, task_id=2, training_mode="16shot", seed=1
        )
        self.assertEqual(metadata["task_id"], 2)
        with self.assertRaises(ValueError):
            validate_task_adapter_checkpoint(checkpoint, task_id=3)


if __name__ == "__main__":
    unittest.main()
