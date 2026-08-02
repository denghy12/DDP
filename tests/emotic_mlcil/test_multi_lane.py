import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from benchmarks.emotic_mlcil.methods.multi_lane import (
    MultiLaneBenchmarkMethod,
    MultiLaneModel,
)
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from tests.emotic_mlcil import protocol_config, task_context


class TinyResidualBlock(nn.Module):
    def __init__(self, width=4, heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(width, width * 2)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(width * 2, width)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, values):
        normalized = self.ln_1(values)
        attended = self.attn(
            normalized, normalized, normalized, need_weights=False
        )[0]
        values = values + attended
        return values + self.mlp(self.ln_2(values))


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [TinyResidualBlock(), TinyResidualBlock()]
        )


class TinyVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=1, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(4))
        self.positional_embedding = nn.Parameter(torch.randn(5, 4))
        self.ln_pre = nn.LayerNorm(4)
        self.transformer = TinyTransformer()
        self.ln_post = nn.LayerNorm(4)
        self.proj = nn.Parameter(torch.eye(4))
        self.output_dim = 4


def tiny_protocol():
    config = protocol_config(
        class_order=("a", "b", "c", "d"),
        tasks=(("a", "b"), ("c", "d")),
    )
    config["method_options"]["multi_lane"] = {
        "epochs": 2,
        "source_base_learning_rate": 0.04,
        "source_reference_batch_size": 4,
        "registered_train_batch_size": 4,
        "weight_decay": 0.0,
        "num_selectors": 2,
        "num_prompts": 2,
        "num_prompt_layers": 1,
        "normalize": "pre-head",
        "temperature": 1.0,
        "amp": False,
        "tf32": False,
    }
    return BenchmarkProtocol.from_dict(config)


def task_batches(protocol, task_id):
    generator = torch.Generator().manual_seed(21 + task_id)
    images = torch.randn(4, 1, 2, 2, generator=generator)
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
    return [batch, batch], [batch]


class MultiLaneBenchmarkMethodTest(unittest.TestCase):
    def make_method(self):
        return MultiLaneBenchmarkMethod(
            tiny_protocol(),
            device="cpu",
            visual_encoder=TinyVisualEncoder(),
        )

    def train_task(self, method, task_id):
        method.begin_task(task_context(method.protocol, task_id))
        train_loader, val_loader = task_batches(method.protocol, task_id)
        method.train_task(train_loader, val_loader)

    def test_registered_source_and_no_added_components(self):
        self.assertIn("multi_lane", method_names())
        self.assertIs(method_class("multi_lane"), MultiLaneBenchmarkMethod)
        method = self.make_method()
        config = method.resolved_method_config()
        self.assertEqual(
            config["upstream_commit"],
            "5ee982c9298d4cfd6af471d9bb2ef3c0aad05373",
        )
        self.assertEqual(
            config["upstream_archive_sha256"],
            "dfe84ea31f6d7e51877c2661791c716aed7d888b8d783d16ce067cb8ce022d49",
        )
        self.assertFalse(config["visual_encoder_trainable"])
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertFalse(config["clip_text_encoder_used"])
        self.assertFalse(config["replay_enabled"])
        self.assertFalse(config["task_oracle_at_inference"])
        self.assertEqual(config["effective_learning_rate"], 0.04)
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)
        self.assertEqual(method.memory_statistics().replay_memory_bytes, 0)

    def test_training_freezes_visual_and_only_changes_current_slices(self):
        method = self.make_method()
        visual_before = {
            name: value.detach().clone()
            for name, value in method.model.visual_encoder.state_dict().items()
        }
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in method.model.visual_encoder.parameters()
            )
        )
        self.train_task(method, 0)
        selector_zero = method.model.selectors[0].detach().clone()
        prompt_zero = method.model.prompts[0][:, 0].detach().clone()
        head_old = method.model.head.weight[:2].detach().clone()
        method.end_task()
        method.begin_task(task_context(method.protocol, 1))
        self.assertTrue(torch.equal(method.model.selectors[1], selector_zero))
        self.assertTrue(torch.equal(method.model.prompts[0][:, 1], prompt_zero))
        train_loader, val_loader = task_batches(method.protocol, 1)
        method.train_task(train_loader, val_loader)
        self.assertTrue(torch.equal(method.model.selectors[0], selector_zero))
        self.assertTrue(torch.equal(method.model.prompts[0][:, 0], prompt_zero))
        self.assertTrue(torch.equal(method.model.head.weight[:2], head_old))
        for name, value in method.model.visual_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, visual_before[name]), name)

    def test_concat_inference_uses_disjoint_seen_lane_masks(self):
        protocol = tiny_protocol()
        model = MultiLaneModel(
            TinyVisualEncoder(),
            task_sizes=(2, 2),
            num_selectors=2,
            num_prompts=2,
            num_prompt_layers=1,
        )
        model.activate_task(0)
        model.activate_task(1)
        images = torch.randn(3, 1, 2, 2)
        lane_logits = model.lane_logits(images, all_seen_lanes=True)
        expected = torch.sum(
            lane_logits * model.task_class_mask[:2].unsqueeze(0), dim=1
        )
        actual = model.seen_logits(images)
        self.assertEqual(tuple(lane_logits.shape), (3, 2, protocol.num_classes))
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-7))

    def test_training_rejects_hidden_labels_and_gradients_are_task_local(self):
        method = self.make_method()
        method.begin_task(task_context(method.protocol, 0))
        train_loader, val_loader = task_batches(method.protocol, 0)
        malformed = train_loader[0]
        malformed.visible_mask[:, 3] = True
        with self.assertRaisesRegex(ValueError, "outside current classes"):
            method.train_task([malformed], val_loader)

        model = MultiLaneModel(
            TinyVisualEncoder(),
            task_sizes=(2, 2),
            num_selectors=2,
            num_prompts=2,
            num_prompt_layers=1,
        )
        model.activate_task(0)
        model.activate_task(1)
        model.zero_grad(set_to_none=True)
        model.current_logits(torch.randn(2, 1, 2, 2)).sum().backward()
        self.assertEqual(torch.count_nonzero(model.selectors.grad[0]).item(), 0)
        self.assertGreater(torch.count_nonzero(model.selectors.grad[1]).item(), 0)
        self.assertEqual(
            torch.count_nonzero(model.prompts[0].grad[:, 0]).item(), 0
        )
        self.assertGreater(
            torch.count_nonzero(model.prompts[0].grad[:, 1]).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(model.head.weight.grad[:2]).item(), 0
        )
        self.assertGreater(
            torch.count_nonzero(model.head.weight.grad[2:]).item(), 0
        )

    def test_checkpoint_round_trip_and_prediction_alignment(self):
        method = self.make_method()
        self.train_task(method, 0)
        _, val_loader = task_batches(method.protocol, 0)
        train_batch = val_loader[0]
        evaluation = EvaluationBatch(
            images=train_batch.images,
            sample_ids=list(train_batch.sample_ids),
            targets_seen=train_batch.targets_current,
            class_order_hash=method.protocol.class_order_hash,
            split_hash="fixed-eval-split",
        )
        prediction = method.predict_scores([evaluation])
        self.assertEqual(tuple(prediction.scores.shape), (4, 2))
        self.assertEqual(prediction.sample_ids, train_batch.sample_ids)
        self.assertTrue(torch.equal(prediction.targets, train_batch.targets_current))
        self.assertEqual(prediction.split_hash, "fixed-eval-split")

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "task0.pth"
            method.save_checkpoint(checkpoint)
            restored = self.make_method()
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored._completed_task_id, 0)
            self.assertEqual(restored.model.current_task_id, 0)
            for name, value in method.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value.cpu(), restored.model.state_dict()[name].cpu()),
                    name,
                )
            restored.begin_task(task_context(restored.protocol, 1))
            self.assertEqual(restored.model.current_task_id, 1)

    def test_parameter_statistics_report_preallocated_zero_growth(self):
        method = self.make_method()
        stats = method.parameter_statistics()
        expected_trainable = sum(
            parameter.numel() for parameter in method.model.optimizer_parameters()
        )
        self.assertEqual(stats.trainable_parameters, expected_trainable)
        self.assertEqual(stats.incremental_parameters, 0)
        self.assertEqual(
            dict(stats.per_task_incremental_parameters), {0: 0, 1: 0}
        )
        self.assertGreater(stats.total_parameters, stats.trainable_parameters)


if __name__ == "__main__":
    unittest.main()
