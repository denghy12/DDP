import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from PIL import Image

from benchmarks.emotic_mlcil.methods.emot_net_ft import (
    EMOTNetBodyEncoder,
    EMOTNetFTBenchmarkMethod,
    EMOTNetFTModel,
    EMOTNetFTOptions,
    class_weights_from_current_targets,
    weighted_sigmoid_mse,
)
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from tests.emotic_mlcil import protocol_config, task_context
from src.helper_functions.emotic_loader import BodyContextTransforms, EMOTIC


class TinyEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.output_dim = width
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


def task_batch(protocol, task_id):
    context = torch.tensor(
        [
            [1.0, 0.2, 0.4],
            [0.3, 1.0, 0.7],
            [0.8, 0.6, 1.0],
            [0.5, 0.9, 0.3],
        ]
    ).view(4, 3, 1, 1).expand(4, 3, 224, 224).clone()
    body = torch.flip(context, dims=(-1,)).clone()
    body[:, :, 128:, :] = 0
    body[:, :, :, 128:] = 0
    images = torch.stack((context, body), dim=1)
    targets = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    )
    visible = torch.zeros(4, protocol.num_classes, dtype=torch.bool)
    visible[:, list(protocol.current_class_indices(task_id))] = True
    return TrainBatch(
        images=images,
        sample_ids=[f"task{task_id}-{index}" for index in range(4)],
        targets_current=targets,
        visible_mask=visible,
    )


class EMOTNetFTTest(unittest.TestCase):
    def make_method(self):
        torch.manual_seed(5)
        model = EMOTNetFTModel(
            context_encoder=TinyEncoder(640),
            body_encoder=TinyEncoder(256),
            fusion_dim=8,
            dropout=0.0,
        )
        return EMOTNetFTBenchmarkMethod(
            tiny_protocol(),
            device="cpu",
            model=model,
            option_overrides={
                "fusion_dim": 8,
                "epochs": 1,
                "early_stopping_patience": 1,
                "learning_rate": 0.01,
                "lr_drop_epoch": 1,
                "dropout": 0.0,
                "amp": False,
                "tf32": False,
            },
        )

    def train(self, method, task_id):
        method.begin_task(task_context(method.protocol, task_id))
        batch = task_batch(method.protocol, task_id)
        method.train_task([batch], [batch])
        return batch

    def test_registration_and_frozen_conversion_contract(self):
        self.assertIn("emot_net_ft", method_names())
        self.assertIs(method_class("emot_net_ft"), EMOTNetFTBenchmarkMethod)
        config = self.make_method().resolved_method_config()
        self.assertEqual(config["upstream_commit"], "69c3a5106aed08121cd12f6a5b359c745136931e")
        self.assertEqual(config["input_mode"], "body_context")
        self.assertTrue(config["current_label_only"])
        self.assertFalse(config["distillation_enabled"])
        self.assertFalse(config["replay_enabled"])
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertFalse(config["clip_visual_encoder_used"])
        self.assertFalse(config["clip_text_encoder_used"])
        self.assertEqual(EMOTNetFTOptions().epochs, 21)
        self.assertEqual(config["epochs"], 1)
        self.assertAlmostEqual(config["discrete_loss_weight"], 1.0 / 6.0)
        self.assertEqual(config["native_body_variant"], "official_dropbox_alexnet")
        self.assertEqual(
            config["class_weight_scope"], "current mini-batch visible labels only"
        )

    def test_official_release_alexnet_body_shape_and_groups(self):
        body = EMOTNetBodyEncoder().eval()
        with torch.no_grad():
            output = body(torch.zeros(1, 3, 128, 128))
        self.assertEqual(tuple(output.shape), (1, 256))
        convolutions = [
            module for module in body.modules() if isinstance(module, nn.Conv2d)
        ]
        self.assertEqual([module.groups for module in convolutions], [1, 2, 1, 2, 2])
        pools = [module for module in body.modules() if isinstance(module, nn.MaxPool2d)]
        self.assertEqual(len(pools), 2)
        self.assertTrue(all(module.ceil_mode for module in pools))

    def test_source_weight_and_loss_formulas(self):
        targets = torch.tensor(
            [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        weights = class_weights_from_current_targets(targets, 1.2)
        expected_weight = float(1.0 / torch.log(torch.tensor(3.2)))
        expected = torch.tensor([expected_weight, 1.0e-4, expected_weight])
        self.assertTrue(torch.allclose(weights, expected))
        logits = torch.tensor([[0.2, -0.4, 0.7], [-0.8, 0.3, 0.1]])
        aligned_targets = targets[:2]
        direct = ((torch.sigmoid(logits) - aligned_targets).pow(2) * weights).mean()
        self.assertTrue(torch.allclose(weighted_sigmoid_mse(logits, aligned_targets, weights), direct))

    def test_lifecycle_label_firewall_prediction_and_checkpoint(self):
        method = self.make_method()
        batch0 = self.train(method, 0)
        self.assertEqual(method._completed_task_id, 0)
        self.assertEqual(method.model.head_sizes, (2,))
        malformed = task_batch(method.protocol, 1)
        malformed.visible_mask[:, 0] = True
        method.end_task()
        method.begin_task(task_context(method.protocol, 1))
        with self.assertRaisesRegex(ValueError, "outside current classes"):
            method.train_task([malformed], [malformed])
        malformed.visible_mask[:, 0] = False
        method.train_task([malformed], [malformed])
        evaluation = EvaluationBatch(
            images=malformed.images,
            sample_ids=list(malformed.sample_ids),
            targets_seen=torch.cat((batch0.targets_current, malformed.targets_current), dim=1),
            class_order_hash=method.protocol.class_order_hash,
            split_hash="emot-net-eval",
        )
        output = method.predict_scores([evaluation])
        self.assertEqual(tuple(output.scores.shape), (4, 4))
        self.assertEqual(output.sample_ids, malformed.sample_ids)
        stats = method.parameter_statistics()
        self.assertGreater(stats.trainable_parameters, 0)
        self.assertEqual(stats.incremental_parameters, 2 * 9)
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task1.pth"
            method.save_checkpoint(path)
            restored = self.make_method()
            restored.load_checkpoint(path)
            self.assertEqual(restored._completed_task_id, 1)
            self.assertEqual(restored.model.head_sizes, (2, 2))

    def test_rejects_single_view_images(self):
        method = self.make_method()
        method.begin_task(task_context(method.protocol, 0))
        batch = task_batch(method.protocol, 0)
        batch.images = batch.images[:, 0]
        with self.assertRaisesRegex(ValueError, "body_context"):
            method.train_task([batch], [batch])

    def test_loader_packages_224_context_and_native_128_body(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "sample.png"
            Image.new("RGB", (20, 10), color=(10, 20, 30)).save(image_path)
            dataset = EMOTIC.__new__(EMOTIC)
            dataset.file_paths = [str(image_path)]
            dataset.body_bboxes = [[2, 1, 12, 9]]
            dataset.classes = ["a"]
            dataset.targets = [[0]]
            dataset.input_mode = "body_context"
            dataset.transform = BodyContextTransforms(
                context=lambda image: torch.ones(3, 224, 224),
                body=lambda image: torch.full((3, 128, 128), 2.0),
            )
            images, target = dataset[0]
            self.assertEqual(tuple(images.shape), (2, 3, 224, 224))
            self.assertTrue(torch.equal(images[1, :, :128, :128], torch.full((3, 128, 128), 2.0)))
            self.assertEqual(float(images[1, :, 128:, :].abs().sum()), 0.0)
            self.assertEqual(target.tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
