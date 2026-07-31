import unittest

import torch
from torch.utils.data import Dataset

from benchmarks.emotic_mlcil.data_module import EMOTICMLCILDataModule
from benchmarks.emotic_mlcil.evaluator import BenchmarkEvaluator
from tests.emotic_mlcil import make_protocol


class FakeEMOTIC(Dataset):
    def __init__(self, targets):
        self.targets = [list(target) for target in targets]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = torch.zeros(26)
        target[self.targets[index]] = 1.0
        return torch.full((3, 2, 2), float(index)), target


def fake_data_module():
    protocol = make_protocol()
    source = FakeEMOTIC(
        [
            [0, 6, 20],
            [5, 7],
            [2],
            [9],
            [5],
        ]
    )
    module = EMOTICMLCILDataModule.__new__(EMOTICMLCILDataModule)
    module.protocol = protocol
    module._sources = {"train": source, "val": source, "test": source}
    ids = tuple(f"emotic:split:image.jpg:person={index}" for index in range(5))
    module._sample_ids = {"train": ids, "val": ids, "test": ids}
    return module


class LabelVisibilityTest(unittest.TestCase):
    def test_membership_uses_current_labels_but_batch_hides_old_and_future_truth(self):
        module = fake_data_module()
        dataset = module.method_dataset(task_id=1, split="train")
        self.assertEqual(dataset.indices, (0, 1, 4))
        row = dataset[0]
        self.assertEqual(
            set(row),
            {
                "image",
                "sample_id",
                "targets_current",
                "visible_mask",
                "class_order_hash",
                "split_hash",
            },
        )
        self.assertNotIn("targets_seen", row)
        self.assertEqual(tuple(row["targets_current"].shape), (3,))
        self.assertEqual(row["targets_current"].tolist(), [0.0, 1.0, 0.0])
        self.assertFalse(bool(row["visible_mask"][:5].any()))
        self.assertTrue(bool(row["visible_mask"][5:8].all()))
        self.assertFalse(bool(row["visible_mask"][8:].any()))

    def test_validation_selection_batch_is_current_only(self):
        module = fake_data_module()
        loader = module.method_loader(
            1,
            batch_size=3,
            num_workers=0,
            split="val",
            shuffle=False,
        )
        batch = next(iter(loader))
        self.assertEqual(tuple(batch.targets_current.shape), (3, 3))
        self.assertFalse(hasattr(batch, "targets_seen"))

    def test_evaluator_capability_reveals_seen_but_never_future_columns(self):
        module = fake_data_module()
        evaluator = BenchmarkEvaluator(module.protocol)
        dataset = module.evaluator_dataset(
            1,
            split="test",
            access=evaluator.access_token,
        )
        self.assertEqual(dataset.indices, (0, 1, 2, 4))
        row = dataset[0]
        self.assertIn("targets_seen", row)
        self.assertNotIn("targets_current", row)
        self.assertEqual(tuple(row["targets_seen"].shape), (8,))

    def test_evaluator_targets_require_capability(self):
        module = fake_data_module()
        with self.assertRaises(PermissionError):
            module.evaluator_dataset(0, "val", access=None)


if __name__ == "__main__":
    unittest.main()
