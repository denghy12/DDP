import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from benchmarks.emotic_mlcil.methods.agcn import (
    AGCNBenchmarkMethod,
    CorrelationStatistics,
)
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from tests.emotic_mlcil import protocol_config, task_context


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)

    def forward(self, images):
        return self.projection(images.flatten(1))


def tiny_protocol():
    return BenchmarkProtocol.from_dict(
        protocol_config(class_order=("a", "b", "c", "d"), tasks=(("a", "b"), ("c", "d")))
    )


def task_batches(protocol, task_id):
    images = torch.tensor([
        [[[1.0, 0.2], [0.4, 0.8]]],
        [[[0.3, 1.0], [0.7, 0.5]]],
        [[[0.8, 0.6], [1.0, 0.2]]],
        [[[0.5, 0.9], [0.3, 1.0]]],
    ])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    visible = torch.zeros(4, protocol.num_classes, dtype=torch.bool)
    visible[:, list(protocol.current_class_indices(task_id))] = True
    batch = TrainBatch(
        images=images,
        sample_ids=[f"task{task_id}-{index}" for index in range(4)],
        targets_current=targets,
        visible_mask=visible,
    )
    return [batch], [batch]


class AGCNBenchmarkMethodTest(unittest.TestCase):
    def make_method(self):
        torch.manual_seed(4)
        return AGCNBenchmarkMethod(
            tiny_protocol(),
            device="cpu",
            visual_encoder=TinyVisual(),
            class_embeddings=torch.randn(4, 3),
            option_overrides={
                "visual_dim": 4,
                "embedding_dim": 3,
                "graph_hidden_dim": 5,
                "epochs": 1,
                "visual_learning_rate": 1e-2,
                "graph_learning_rate": 1e-2,
                "amp": False,
                "tf32": False,
            },
        )

    def train(self, method, task_id):
        method.begin_task(task_context(method.protocol, task_id))
        train_loader, val_loader = task_batches(method.protocol, task_id)
        method.train_task(train_loader, val_loader)

    def test_registered_provenance_and_no_extra_components(self):
        self.assertIn("agcn", method_names())
        self.assertIs(method_class("agcn"), AGCNBenchmarkMethod)
        config = self.make_method().resolved_method_config()
        self.assertEqual(config["upstream_commit"], "3afe2ecbbef0051c6e841a97c369885011a683f0")
        self.assertEqual(config["upstream_license"], "Apache-2.0")
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertFalse(config["clip_text_encoder_used"])
        self.assertFalse(config["replay_enabled"])
        self.assertEqual(config["classification_weight"], 0.07)
        self.assertEqual(config["distillation_weight"], 0.93)
        self.assertEqual(config["relationship_weight"], 1e5)

    def test_online_acm_uses_only_current_truth_and_old_soft_labels(self):
        state = CorrelationStatistics.create(2, 2, device=torch.device("cpu"))
        current = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        old_soft = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
        state.update(current, old_soft)
        self.assertTrue(torch.allclose(state.hard_cooccurrence, torch.tensor([[0.0, 1.0], [1.0, 0.0]])))
        self.assertTrue(torch.allclose(state.old_new_soft_hard, torch.tensor([[1.2, 0.4], [0.8, 0.6]])))
        adjacency = state.build_adjacency(
            torch.eye(2), task0_threshold=0.0, current_threshold=0.4,
            cross_threshold=0.3, task0_scale=0.28, later_scale=0.25,
            reverse_bayes_scale=0.5, task0_degree_exponent=-0.8,
            later_degree_exponent=-0.5,
        )
        self.assertEqual(tuple(adjacency.shape), (4, 4))
        self.assertTrue(torch.isfinite(adjacency).all())

    def test_task_lifecycle_teacher_losses_and_label_firewall(self):
        method = self.make_method()
        self.train(method, 0)
        self.assertEqual(method._completed_task_id, 0)
        self.assertEqual(tuple(method._adjacency.shape), (2, 2))
        method.end_task()
        method.begin_task(task_context(method.protocol, 1))
        self.assertIsNotNone(method._teacher)
        train_loader, val_loader = task_batches(method.protocol, 1)
        malformed = train_loader[0]
        malformed.visible_mask[:, 0] = True
        with self.assertRaisesRegex(ValueError, "outside current classes"):
            method.train_task([malformed], val_loader)
        malformed.visible_mask[:, 0] = False
        method.train_task(train_loader, val_loader)
        row = method.training_history[-1]
        self.assertGreaterEqual(row["distillation_loss"], 0.0)
        self.assertGreaterEqual(row["relationship_loss"], 0.0)
        self.assertEqual(tuple(method._adjacency.shape), (4, 4))

    def test_prediction_checkpoint_and_statistics(self):
        method = self.make_method()
        self.train(method, 0)
        _, val_loader = task_batches(method.protocol, 0)
        batch = val_loader[0]
        evaluation = EvaluationBatch(
            images=batch.images,
            sample_ids=list(batch.sample_ids),
            targets_seen=batch.targets_current,
            class_order_hash=method.protocol.class_order_hash,
            split_hash="agcn-eval",
        )
        output = method.predict_scores([evaluation])
        self.assertEqual(tuple(output.scores.shape), (4, 2))
        self.assertEqual(output.sample_ids, batch.sample_ids)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task0.pth"
            method.save_checkpoint(path)
            restored = self.make_method()
            restored.load_checkpoint(path)
            self.assertEqual(restored._completed_task_id, 0)
            self.assertTrue(torch.equal(restored._adjacency, method._adjacency))
        stats = method.parameter_statistics()
        self.assertGreater(stats.trainable_parameters, 0)
        self.assertEqual(stats.incremental_parameters, 0)
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)


if __name__ == "__main__":
    unittest.main()
