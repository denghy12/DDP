import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from benchmarks.emotic_mlcil.methods.csc import CSCBenchmarkMethod
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from tests.emotic_mlcil import protocol_config, task_context


class TinyTokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4))

    def forward(self, images):
        features = self.projection(images)
        return torch.stack(
            [features, 0.5 * features, 1.5 * features], dim=1
        )


def tiny_protocol():
    config = protocol_config(
        class_order=("a", "b", "c", "d"),
        tasks=(("a", "b"), ("c", "d")),
    )
    config["method_options"]["csc"] = {
        "token_dim": 4,
        "graph_dim": 6,
        "epochs": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "one_cycle_pct_start": 0.5,
        "alpha": 0.5,
        "entropy_strength": 0.04,
        "amp": False,
        "tf32": False,
    }
    return BenchmarkProtocol.from_dict(config)


def task_batches(protocol, task_id):
    if task_id == 0:
        images = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
            ]
        )
    else:
        images = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, -1.0],
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
    visible = torch.zeros(images.shape[0], protocol.num_classes, dtype=torch.bool)
    visible[:, list(protocol.current_class_indices(task_id))] = True
    batch = TrainBatch(
        images=images,
        sample_ids=[f"task{task_id}-{index}" for index in range(len(images))],
        targets_current=targets,
        visible_mask=visible,
    )
    return [batch, batch], [batch]


class CSCBenchmarkMethodTest(unittest.TestCase):
    def make_method(self):
        return CSCBenchmarkMethod(
            tiny_protocol(),
            device="cpu",
            token_encoder=TinyTokenEncoder(),
        )

    def train_task(self, method, task_id):
        method.begin_task(task_context(method.protocol, task_id))
        train_loader, val_loader = task_batches(method.protocol, task_id)
        method.train_task(train_loader, val_loader)

    def test_registered_source_and_no_added_components(self):
        self.assertIn("csc", method_names())
        self.assertIs(method_class("csc"), CSCBenchmarkMethod)
        method = self.make_method()
        config = method.resolved_method_config()
        self.assertEqual(
            config["upstream_commit"],
            "0bab38a00d6e0555f2df855ae2fe8db1fea68b12",
        )
        self.assertEqual(
            config["upstream_archive_sha256"],
            "588a098a1c7f3d813dee7df777c283fd27768a08a125d4e60b11b2d6ebdb7faa",
        )
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertFalse(config["clip_text_encoder_used"])
        self.assertFalse(config["replay_enabled"])
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)
        self.assertEqual(method.memory_statistics().replay_memory_bytes, 0)

    def test_dynamic_expansion_preserves_old_blocks_and_detaches_teacher(self):
        method = self.make_method()
        self.train_task(method, 0)
        snapshots = {
            "activation": method.model.class_activation.weight.detach().clone(),
            "general": method.model.general_relation.weight.detach().clone(),
            "specific_weight": (
                method.model.specific_relation.weight.detach().clone()
            ),
            "specific_bias": method.model.specific_relation.bias.detach().clone(),
            "graph_weight": method.model.graph_classifier.weight.detach().clone(),
            "graph_bias": method.model.graph_classifier.bias.detach().clone(),
            "identity": method.model.identity_mask.detach().clone(),
        }
        method.end_task()
        method.begin_task(task_context(method.protocol, 1))
        self.assertIsNotNone(method._teacher)
        self.assertEqual(method._teacher.num_classes, 2)
        self.assertFalse(
            any(parameter.requires_grad for parameter in method._teacher.parameters())
        )
        self.assertTrue(
            torch.equal(method.model.class_activation.weight[:2], snapshots["activation"])
        )
        self.assertTrue(
            torch.equal(
                method.model.general_relation.weight[:2, :2], snapshots["general"]
            )
        )
        self.assertTrue(
            torch.equal(
                method.model.specific_relation.weight[:2],
                snapshots["specific_weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                method.model.specific_relation.bias[:2], snapshots["specific_bias"]
            )
        )
        self.assertTrue(
            torch.equal(
                method.model.graph_classifier.weight[:2], snapshots["graph_weight"]
            )
        )
        self.assertTrue(
            torch.equal(method.model.graph_classifier.bias[:2], snapshots["graph_bias"])
        )
        self.assertTrue(
            torch.equal(method.model.identity_mask[:2, :2], snapshots["identity"])
        )

        train_loader, val_loader = task_batches(method.protocol, 1)
        output = method.model(train_loader[0].images)
        self.assertEqual(tuple(output["logits"].shape), (4, 4))
        self.assertEqual(tuple(output["sample_relation"].shape), (4, 4, 4))
        method.train_task(train_loader, val_loader)
        self.assertTrue(
            any(row["distillation_loss"] > 0 for row in method.training_history)
        )
        self.assertTrue(any(row["entropy_loss"] < 0 for row in method.training_history))
        self.assertTrue(
            all(row["optimizer_steps"] > 0 for row in method.training_history)
        )
        stats = method.parameter_statistics()
        self.assertEqual(stats.per_task_incremental_parameters, {0: 0, 1: 72})
        self.assertEqual(stats.incremental_parameters, 72)

    def test_entropy_regularizer_has_max_entropy_sign(self):
        logits = torch.zeros(3, 4)
        loss = CSCBenchmarkMethod._entropy_regularizer(logits, 0.04)
        self.assertLess(float(loss), 0.0)
        self.assertEqual(
            float(CSCBenchmarkMethod._entropy_regularizer(logits, 0.0)),
            0.0,
        )

    def test_hidden_labels_are_rejected(self):
        method = self.make_method()
        method.begin_task(task_context(method.protocol, 0))
        train_loader, _ = task_batches(method.protocol, 0)
        train_loader[0].visible_mask[:, 2] = True
        with self.assertRaisesRegex(ValueError, "outside current classes"):
            method.train_task(train_loader, train_loader)

    def test_predictions_are_seen_class_aligned(self):
        method = self.make_method()
        self.train_task(method, 0)
        evaluation = EvaluationBatch(
            images=torch.eye(4)[:2],
            sample_ids=["eval-0", "eval-1"],
            targets_seen=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            class_order_hash=method.protocol.class_order_hash,
            split_hash="test-split",
        )
        output = method.predict_scores([evaluation])
        self.assertEqual(tuple(output.scores.shape), (2, 2))
        self.assertEqual(output.targets.tolist(), evaluation.targets_seen.tolist())
        self.assertEqual(output.sample_ids, evaluation.sample_ids)
        self.assertTrue(torch.isfinite(output.scores).all())
        self.assertTrue(((output.scores >= 0) & (output.scores <= 1)).all())

    def test_checkpoint_restores_expansion_and_model_state(self):
        method = self.make_method()
        self.train_task(method, 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "task0.pth"
            method.save_checkpoint(checkpoint)
            restored = self.make_method()
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.model.task_sizes, (2,))
            self.assertEqual(restored._completed_task_id, 0)
            self.assertEqual(restored.memory_statistics(), method.memory_statistics())
            for key, value in method.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value.cpu(), restored.model.state_dict()[key])
                )


if __name__ == "__main__":
    unittest.main()
