import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from benchmarks.emotic_mlcil.methods.krt import KRTBenchmarkMethod
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
        return torch.stack([features, 0.5 * features, 1.5 * features], dim=1)


def tiny_protocol():
    config = protocol_config(
        class_order=("a", "b", "c", "d"),
        tasks=(("a", "b"), ("c", "d")),
    )
    config["method_options"]["krt"] = {
        "token_dim": 4,
        "embed_dim": 8,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "max_patch_tokens": 2,
        "use_positional_embedding": True,
        "epochs": 3,
        "early_stopping_patience": 3,
        "base_learning_rate": 0.01,
        "incremental_learning_rate": 0.01,
        "weight_decay": 0.0,
        "amp": False,
        "tf32": False,
        "pseudo_label": True,
        "pseudo_initial_threshold": 0.8,
        "pseudo_threshold_step": 0.05,
        "pseudo_threshold_tolerance": 0.1,
        "pseudo_threshold_iterations": 20,
        "token_distillation_weight": 1.0,
        "exemplars_per_class": 1,
        "replay_batch_size": 2,
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
    return [batch], [batch]


class KRTBenchmarkMethodTest(unittest.TestCase):
    def make_method(self):
        return KRTBenchmarkMethod(
            tiny_protocol(),
            device="cpu",
            token_encoder=TinyTokenEncoder(),
        )

    def train_task(self, method, task_id):
        method.begin_task(task_context(method.protocol, task_id))
        train_loader, val_loader = task_batches(method.protocol, task_id)
        method.train_task(train_loader, val_loader)

    def test_registered_and_declares_official_source(self):
        self.assertIn("krt", method_names())
        self.assertIs(method_class("krt"), KRTBenchmarkMethod)
        method = self.make_method()
        config = method.resolved_method_config()
        self.assertEqual(
            config["upstream_commit"],
            "3f79044001edfe9ef94b729cd905a535fe8dd478",
        )
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertFalse(config["clip_text_encoder_used"])

    def test_task_expansion_freezes_only_old_tokens_and_heads(self):
        method = self.make_method()
        self.train_task(method, 0)
        first_memory = method.memory_statistics()
        self.assertGreaterEqual(first_memory.replay_memory_samples, 1)
        self.assertLessEqual(first_memory.replay_memory_samples, 2)
        self.assertGreater(first_memory.replay_memory_bytes, 0)
        method.end_task()
        method.begin_task(task_context(method.protocol, 1))
        self.assertFalse(method.model.task_tokens[0].requires_grad)
        self.assertTrue(method.model.task_tokens[1].requires_grad)
        self.assertFalse(
            any(parameter.requires_grad for parameter in method.model.heads[0].parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in method.model.heads[1].parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in method.model.token_encoder.parameters()
            )
        )
        train_loader, val_loader = task_batches(method.protocol, 1)
        method.train_task(train_loader, val_loader)
        self.assertIsNotNone(method._pseudo_threshold)
        self.assertTrue(
            any(
                record["token_distillation_loss"] > 0
                for record in method.training_history
            )
        )
        memory = method.memory_statistics()
        self.assertGreater(memory.replay_memory_samples, first_memory.replay_memory_samples)
        self.assertLessEqual(memory.replay_memory_samples, 4)
        self.assertEqual(
            len({example.sample_id for example in method._replay_memory}),
            memory.replay_memory_samples,
        )
        self.assertTrue(
            all(
                example.visible_through_task_id >= example.captured_task_id
                for example in method._replay_memory
            )
        )
        stats = method.parameter_statistics()
        self.assertEqual(stats.per_task_incremental_parameters, {0: 0, 1: 42})
        self.assertEqual(stats.incremental_parameters, 42)

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

    def test_reappearing_person_updates_only_newly_visible_columns(self):
        method = self.make_method()
        self.train_task(method, 0)
        retained = method._replay_memory[0]
        original = retained.target_seen_at_capture.clone()
        method.end_task()
        method.begin_task(task_context(method.protocol, 1))
        train_loader, _ = task_batches(method.protocol, 1)
        train_loader[0].sample_ids[0] = retained.sample_id
        method._select_replay_examples(train_loader)
        updated = next(
            example
            for example in method._replay_memory
            if example.sample_id == retained.sample_id
        )
        self.assertTrue(torch.equal(updated.target_seen_at_capture[:2], original))
        self.assertEqual(updated.target_seen_at_capture[2:].tolist(), [1.0, 0.0])
        self.assertEqual(updated.visible_through_task_id, 1)
        self.assertEqual(
            len({example.sample_id for example in method._replay_memory}),
            len(method._replay_memory),
        )

    def test_checkpoint_restores_model_and_replay_memory(self):
        method = self.make_method()
        self.train_task(method, 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "task0.pth"
            method.save_checkpoint(checkpoint)
            restored = self.make_method()
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.model.head_sizes, (2,))
            self.assertEqual(restored._completed_task_id, 0)
            self.assertEqual(
                restored.memory_statistics(), method.memory_statistics()
            )
            for key, value in method.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value.cpu(), restored.model.state_dict()[key])
                )


if __name__ == "__main__":
    unittest.main()
