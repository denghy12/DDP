import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from benchmarks.emotic_mlcil.methods.l3a import L3ABenchmarkMethod
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from tests.emotic_mlcil import protocol_config, task_context


class TinyFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.tensor(
                    [
                        [0.4, 0.1, 0.2, 0.3],
                        [0.2, 0.5, 0.1, 0.4],
                        [0.3, 0.2, 0.6, 0.1],
                    ]
                )
            )

    def forward(self, images):
        return self.projection(images.flatten(1))


def tiny_protocol():
    config = protocol_config(
        class_order=("a", "b", "c", "d"),
        tasks=(("a", "b"), ("c", "d")),
    )
    config["method_options"]["l3a"] = {
        "feature_dim": 3,
        "hidden_dim": 4,
        "base_epochs": 1,
        "base_learning_rate": 1.0e-2,
        "weight_decay": 0.0,
        "one_cycle_pct_start": 0.2,
        "analytic_repeats": 1,
        "ridge": 1.0,
        "pseudo_label": True,
        "pseudo_threshold": 0.7,
        "weighted_analytic": True,
        "amp": False,
        "tf32": False,
    }
    return BenchmarkProtocol.from_dict(config)


def task_batches(protocol, task_id):
    images = torch.tensor(
        [
            [[[1.0, 0.2], [0.4, 0.8]]],
            [[[0.3, 1.0], [0.7, 0.5]]],
            [[[0.8, 0.6], [1.0, 0.2]]],
            [[[0.5, 0.9], [0.3, 1.0]]],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]
    )
    visible = torch.zeros(4, protocol.num_classes, dtype=torch.bool)
    visible[:, list(protocol.current_class_indices(task_id))] = True
    batch = TrainBatch(
        images=images,
        sample_ids=[f"task{task_id}-{index}" for index in range(4)],
        targets_current=targets,
        visible_mask=visible,
    )
    # Four source-style steps keep the one-cycle schedule well defined while
    # remaining tiny enough for deterministic CPU tests.
    return [batch, batch, batch, batch], [batch]


class L3ABenchmarkMethodTest(unittest.TestCase):
    def make_method(self):
        return L3ABenchmarkMethod(
            tiny_protocol(),
            device="cpu",
            feature_extractor=TinyFeatureEncoder(),
        )

    def train_task(self, method, task_id):
        method.begin_task(task_context(method.protocol, task_id))
        train_loader, val_loader = task_batches(method.protocol, task_id)
        method.train_task(train_loader, val_loader)

    def test_registered_source_and_protocol_mapping(self):
        self.assertIn("l3a", method_names())
        self.assertIs(method_class("l3a"), L3ABenchmarkMethod)
        method = self.make_method()
        config = method.resolved_method_config()
        self.assertEqual(
            config["upstream_commit"],
            "1067bbd6124a7aa96136baa555080ba9183ab2ac",
        )
        self.assertEqual(
            config["upstream_archive_sha256"],
            "307d721d76069a7fe95a10346be1323a842cd9624f8d55200f4cf7cd45462409",
        )
        self.assertEqual(config["source_configuration_mapping"], "official l3a_vit_coco.yaml")
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertFalse(config["clip_text_encoder_used"])
        self.assertFalse(config["replay_enabled"])
        self.assertFalse(config["old_future_ground_truth_used_for_training"])

    def test_task0_trains_visual_then_replaces_head_and_freezes(self):
        method = self.make_method()
        visual_before = {
            name: value.detach().clone()
            for name, value in method.model.visual_encoder.state_dict().items()
        }
        self.train_task(method, 0)
        self.assertTrue(method.model.analytic_ready)
        self.assertIsNone(method.model.base_classifier)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in method.model.visual_encoder.parameters()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(value, visual_before[name])
                for name, value in method.model.visual_encoder.state_dict().items()
            )
        )
        self.assertEqual(tuple(method._analytic_a.shape), (4, 4))
        self.assertEqual(tuple(method._analytic_c.shape), (4, 2))
        expected = torch.linalg.inv(
            method._analytic_a + torch.eye(4, dtype=torch.float64)
        ) @ method._analytic_c
        self.assertTrue(
            torch.allclose(
                method.model.analytic_classifier.weight.t(),
                expected.float(),
                atol=1.0e-6,
            )
        )

    def test_training_rejects_hidden_labels(self):
        method = self.make_method()
        method.begin_task(task_context(method.protocol, 0))
        train_loader, val_loader = task_batches(method.protocol, 0)
        malformed = train_loader[0]
        malformed.visible_mask[:, 3] = True
        with self.assertRaisesRegex(ValueError, "outside current classes"):
            method.train_task([malformed], val_loader)

    def test_incremental_task_uses_positive_pseudo_labels_without_gradients(self):
        method = self.make_method()
        self.train_task(method, 0)
        method.end_task()
        with torch.no_grad():
            method.model.random_projection.weight.fill_(1.0)
            method.model.analytic_classifier.weight.fill_(10.0)
        old_visual = {
            name: value.detach().clone()
            for name, value in method.model.visual_encoder.state_dict().items()
        }
        self.train_task(method, 1)
        self.assertGreater(
            method.training_history[-1]["pseudo_positive_labels"], 0
        )
        self.assertEqual(tuple(method._analytic_c.shape), (4, 4))
        self.assertEqual(len(method._class_counts), 4)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in method.model.parameters()
            )
        )
        for name, value in method.model.visual_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, old_visual[name]), name)

    def test_checkpoint_round_trip_and_prediction_alignment(self):
        method = self.make_method()
        self.train_task(method, 0)
        _, val_loader = task_batches(method.protocol, 0)
        batch = val_loader[0]
        evaluation = EvaluationBatch(
            images=batch.images,
            sample_ids=list(batch.sample_ids),
            targets_seen=batch.targets_current,
            class_order_hash=method.protocol.class_order_hash,
            split_hash="l3a-fixed-eval",
        )
        prediction = method.predict_scores([evaluation])
        self.assertEqual(tuple(prediction.scores.shape), (4, 2))
        self.assertEqual(prediction.sample_ids, batch.sample_ids)
        self.assertTrue(torch.equal(prediction.targets, batch.targets_current))

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "task0.pth"
            method.save_checkpoint(checkpoint)
            restored = self.make_method()
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored._completed_task_id, 0)
            self.assertEqual(restored.model.task_sizes, (2,))
            self.assertTrue(torch.equal(restored._analytic_a, method._analytic_a))
            self.assertTrue(torch.equal(restored._analytic_c, method._analytic_c))
            for name, value in method.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value.cpu(), restored.model.state_dict()[name].cpu()),
                    name,
                )
            restored.begin_task(task_context(restored.protocol, 1))
            self.assertEqual(restored.model.task_sizes, (2, 2))

    def test_parameter_and_memory_statistics(self):
        method = self.make_method()
        self.train_task(method, 0)
        stats = method.parameter_statistics()
        self.assertGreater(stats.trainable_parameters, 0)
        self.assertEqual(stats.incremental_parameters, 0)
        self.assertEqual(
            dict(stats.per_task_incremental_parameters), {0: 0, 1: 8}
        )
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)
        self.assertEqual(method.memory_statistics().replay_memory_bytes, 0)
        method.end_task()
        self.train_task(method, 1)
        self.assertEqual(method.parameter_statistics().incremental_parameters, 8)


if __name__ == "__main__":
    unittest.main()
