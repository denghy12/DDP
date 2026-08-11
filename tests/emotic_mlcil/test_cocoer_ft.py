import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from benchmarks.emotic_mlcil.methods.cocoer_ft import (
    CocoERFTBenchmarkMethod,
    CocoERFTModel,
    CocoERFTOptions,
    dynamic_bce,
)
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from src.helper_functions.emotic_loader import CocoERTransforms, EMOTIC
from tests.emotic_mlcil import protocol_config, task_context


class TinyGridEncoder(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.projection = nn.Linear(3, width)

    def forward(self, images):
        pooled = images.mean(dim=(-2, -1))
        return self.projection(pooled).unsqueeze(1).expand(-1, 49, -1)


class TinyClipEncoder(nn.Module):
    def __init__(self, width=2):
        super().__init__()
        self.projection = nn.Linear(3, width)

    def forward(self, images):
        return self.projection(images.mean(dim=(-2, -1)))


def tiny_protocol():
    config = protocol_config(
        class_order=("a", "b", "c", "d"),
        tasks=(("a", "b"), ("c", "d")),
    )
    config["track"] = "B"
    return BenchmarkProtocol.from_dict(config)


def tiny_model():
    return CocoERFTModel(
        total_classes=4,
        head_encoder=TinyGridEncoder(),
        body_encoder=TinyGridEncoder(),
        context_encoder=TinyGridEncoder(),
        clip_image_encoder=TinyClipEncoder(),
        width=4,
        clip_width=2,
        vi_hidden_width=4,
        encoder_blocks=1,
        inside_lr=0.1,
        pseudo_threshold=0.3,
    )


def task_batch(protocol, task_id):
    images = torch.rand(2, 3, 3, 224, 224)
    geometry = torch.tensor(
        [[[20, 20, 190, 220], [60, 25, 135, 100]]] * 2,
        dtype=torch.float32,
    )
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    visible = torch.zeros(2, protocol.num_classes, dtype=torch.bool)
    visible[:, list(protocol.current_class_indices(task_id))] = True
    return TrainBatch(
        images=images,
        sample_ids=[f"task{task_id}-{index}" for index in range(2)],
        targets_current=targets,
        visible_mask=visible,
        geometry=geometry,
    )


class CocoERFTTest(unittest.TestCase):
    def make_method(self):
        return CocoERFTBenchmarkMethod(
            tiny_protocol(),
            device="cpu",
            model=tiny_model(),
            option_overrides={
                "epochs": 1,
                "early_stopping_patience": 1,
                "lr_step_epochs": 1,
                "learning_rate": 1.0e-3,
                "encoder_blocks": 1,
                "amp": False,
            },
        )

    def test_registration_track_and_native_conversion_contract(self):
        self.assertIn("cocoer_ft", method_names())
        self.assertIs(method_class("cocoer_ft"), CocoERFTBenchmarkMethod)
        method = self.make_method()
        config = method.resolved_method_config()
        self.assertEqual(config["conversion_interface"], "CocoER-FT-v0.1")
        self.assertEqual(config["input_mode"], "cocoer_multilevel")
        self.assertIn("ResNet-50 x3", config["native_backbone"])
        self.assertEqual(config["clip_visual_backbone"], "OpenAI CLIP RN50 (native CocoER component)")
        self.assertFalse(config["full_26_class_gwt_checkpoint_used"])
        self.assertFalse(config["full_26_class_vi_mapping_checkpoint_used"])
        self.assertFalse(config["distillation_enabled"])
        self.assertFalse(config["replay_enabled"])
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertEqual(CocoERFTOptions().epochs, 20)
        track_a = protocol_config(
            class_order=("a", "b", "c", "d"), tasks=(("a", "b"), ("c", "d"))
        )
        with self.assertRaisesRegex(ValueError, "Track B"):
            CocoERFTBenchmarkMethod(BenchmarkProtocol.from_dict(track_a), model=tiny_model())

    def test_dynamic_bce_matches_released_visible_column_formula(self):
        logits = torch.tensor([[0.2, -0.4], [0.8, 0.3]])
        targets = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        counts = targets.sum(0)
        weights = 1.0 / torch.log(counts + 1.2)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=weights.expand_as(logits), reduction="none"
        ).sum(-1).mean()
        self.assertTrue(torch.allclose(dynamic_bce(logits, targets), expected))

    def test_lifecycle_firewall_prediction_and_checkpoint(self):
        method = self.make_method()
        method.begin_task(task_context(method.protocol, 0))
        batch0 = task_batch(method.protocol, 0)
        method.train_task([batch0], [batch0])
        self.assertEqual(method.model.head_sizes, (2,))
        method.end_task()
        method.begin_task(task_context(method.protocol, 1))
        batch1 = task_batch(method.protocol, 1)
        malformed = task_batch(method.protocol, 1)
        malformed.visible_mask[:, 0] = True
        with self.assertRaisesRegex(ValueError, "outside current"):
            method.train_task([malformed], [malformed])
        method.train_task([batch1], [batch1])
        evaluation = EvaluationBatch(
            images=batch1.images,
            geometry=batch1.geometry,
            sample_ids=batch1.sample_ids,
            targets_seen=torch.cat((batch0.targets_current, batch1.targets_current), 1),
            class_order_hash=method.protocol.class_order_hash,
            split_hash="cocoer-eval",
        )
        output = method.predict_scores([evaluation])
        self.assertEqual(tuple(output.scores.shape), (2, 4))
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)
        self.assertEqual(method.parameter_statistics().incremental_parameters, 2 * 5 * 257)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "task1.pth"
            method.save_checkpoint(checkpoint)
            restored = self.make_method()
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.model.head_sizes, (2, 2))

    def test_requires_three_views_and_geometry(self):
        method = self.make_method()
        method.begin_task(task_context(method.protocol, 0))
        batch = task_batch(method.protocol, 0)
        batch.geometry = None
        with self.assertRaisesRegex(ValueError, "geometry"):
            method.train_task([batch], [batch])

    def test_loader_uses_exact_cached_head_box_and_returns_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.png"
            Image.new("RGB", (100, 80), color=(20, 30, 40)).save(path)
            key = "emotic:test:sample.png:person=0"
            dataset = EMOTIC.__new__(EMOTIC)
            dataset.file_paths = [str(path)]
            dataset.body_bboxes = [[10, 10, 90, 78]]
            dataset.sample_keys = [key]
            dataset.classes = ["a"]
            dataset.targets = [[0]]
            dataset.input_mode = "cocoer_multilevel"
            dataset.transform = CocoERTransforms(
                transform=lambda image: torch.ones(3, 224, 224),
                head_boxes={key: (30, 10, 65, 40)},
                train=False,
            )
            images, target, geometry = dataset[0]
            self.assertEqual(tuple(images.shape), (3, 3, 224, 224))
            self.assertEqual(tuple(geometry.shape), (2, 4))
            self.assertTrue(torch.allclose(geometry[1], torch.tensor([67.2, 28.0, 145.6, 112.0])))
            self.assertEqual(target.tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
