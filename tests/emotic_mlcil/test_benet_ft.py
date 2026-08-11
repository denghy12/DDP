import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from benchmarks.emotic_mlcil.methods.benet_ft import BENetFTBenchmarkMethod, BENetFTOptions, focal_tag_loss
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from src.helper_functions.emotic_loader import BENetViewsTransform, EMOTIC
from tests.emotic_mlcil import protocol_config, task_context


class TinyBENetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 8)
        self.detector = nn.Parameter(torch.tensor(0.1))
        self.heads = nn.ModuleList()
        self._head_sizes = []
        self.source_provenance = {"fixture": "tiny"}

    @property
    def head_sizes(self):
        return tuple(self._head_sizes)

    @property
    def num_classes(self):
        return sum(self._head_sizes)

    def add_head(self, classes):
        self.heads.append(nn.Linear(8, classes))
        self._head_sizes.append(int(classes))

    def restore_heads(self, sizes):
        if self.heads:
            raise RuntimeError("already expanded")
        for size in sizes:
            self.add_head(size)

    def _features(self, images, branch):
        offset = {"bu": 0, "pc": 3, "context": 6}[branch]
        return self.encoder(images[:, offset:offset + 3].mean(dim=(-2, -1)))

    def current_logits(self, images, branch):
        return self.heads[-1](self._features(images, branch))

    def forward(self, images):
        values = []
        for branch in ("bu", "pc", "context"):
            features = self._features(images, branch)
            values.append(torch.sigmoid(torch.cat([head(features) for head in self.heads], dim=1)))
        return torch.stack(values).mean(0)

    def detection_outputs(self, images):
        batch = images.shape[0]
        heatmap = self.detector.expand(batch, 1, 4, 4)
        size = self.detector.expand(batch, 2, 4, 4)
        return [heatmap], [size]


def tiny_protocol():
    config = protocol_config(class_order=("a", "b", "c", "d"), tasks=(("a", "b"), ("c", "d")))
    config["track"] = "B"
    return BenchmarkProtocol.from_dict(config)


def task_batch(protocol, task_id):
    images = torch.randn(4, 11, 16, 16)
    images[:, 9].zero_()
    images[:, 9, 4:12, 5:11] = 1
    images[:, 10].fill_(1)
    visible = torch.zeros(4, protocol.num_classes, dtype=torch.bool)
    visible[:, list(protocol.current_class_indices(task_id))] = True
    return TrainBatch(
        images=images,
        sample_ids=[f"task{task_id}-{index}" for index in range(4)],
        targets_current=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
        visible_mask=visible,
    )


class BENetFTTest(unittest.TestCase):
    def make_method(self):
        return BENetFTBenchmarkMethod(
            tiny_protocol(),
            device="cpu",
            model=TinyBENetModel(),
            option_overrides={
                "epochs": 4,
                "early_stopping_patience": 4,
                "amp": False,
                "tf32": False,
            },
        )

    def test_registration_and_contract(self):
        self.assertIn("benet_ft", method_names())
        self.assertIs(method_class("benet_ft"), BENetFTBenchmarkMethod)
        method = self.make_method()
        config = method.resolved_method_config()
        self.assertEqual(config["input_mode"], "benet_views")
        self.assertTrue(config["current_label_only"])
        self.assertFalse(config["distillation_enabled"])
        self.assertFalse(config["replay_enabled"])
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertFalse(config["extra_heco_data_used"])
        self.assertEqual(BENetFTOptions().learning_rate, 1.0e-3)

    def test_focal_tag_formula(self):
        logits = torch.tensor([[0.2, -0.4, 0.7], [-0.8, 0.3, 0.1]], dtype=torch.float64)
        targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        probability = torch.sigmoid(logits)
        direct = -(
            targets.eq(1) * torch.log(probability + 1.0e-6) * (1 - probability).pow(2)
            + targets.lt(1) * torch.log(1 - probability + 1.0e-6) * probability.pow(2)
        ).sum(1).mean()
        self.assertTrue(torch.allclose(focal_tag_loss(logits, targets), direct))

    def test_lifecycle_label_firewall_prediction_and_checkpoint(self):
        method = self.make_method()
        protocol = method.protocol
        method.begin_task(task_context(protocol, 0))
        batch0 = task_batch(protocol, 0)
        method.train_task([batch0], [batch0])
        method.end_task()
        method.begin_task(task_context(protocol, 1))
        batch1 = task_batch(protocol, 1)
        batch1.visible_mask[:, 0] = True
        with self.assertRaisesRegex(ValueError, "outside current classes"):
            method.train_task([batch1], [batch1])
        batch1.visible_mask[:, 0] = False
        method.train_task([batch1], [batch1])
        evaluation = EvaluationBatch(
            images=batch1.images,
            sample_ids=list(batch1.sample_ids),
            targets_seen=torch.cat((batch0.targets_current, batch1.targets_current), dim=1),
            class_order_hash=protocol.class_order_hash,
            split_hash="benet-eval",
        )
        output = method.predict_scores([evaluation])
        self.assertEqual(tuple(output.scores.shape), (4, 4))
        self.assertTrue(torch.all((0 <= output.scores) & (output.scores <= 1)))
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task1.pth"
            method.save_checkpoint(path)
            restored = self.make_method()
            restored.load_checkpoint(path)
            self.assertEqual(restored.model.head_sizes, (2, 2))

    def test_rejects_wrong_transport(self):
        method = self.make_method()
        method.begin_task(task_context(method.protocol, 0))
        batch = task_batch(method.protocol, 0)
        batch.images = batch.images[:, :3]
        with self.assertRaisesRegex(ValueError, "benet_views"):
            method.train_task([batch], [batch])

    def test_loader_builds_three_views_and_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "sample.png"
            Image.new("RGB", (20, 10), color=(10, 20, 30)).save(image_path)
            dataset = EMOTIC.__new__(EMOTIC)
            dataset.file_paths = [str(image_path)]
            dataset.body_bboxes = [[2, 1, 12, 9]]
            dataset.classes = ["a"]
            dataset.targets = [[0]]
            dataset.input_mode = "benet_views"
            dataset.transform = BENetViewsTransform(train=False, size=32)
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (11, 32, 32))
            self.assertGreater(float(image[9].sum()), 0)
            self.assertTrue(torch.equal(image[10], torch.ones(32, 32)))
            self.assertEqual(target.tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
