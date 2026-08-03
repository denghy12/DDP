import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from benchmarks.emotic_mlcil.methods.original_ddp import (
    OriginalDDPBenchmarkMethod,
    OriginalDDPLoss,
)
from benchmarks.emotic_mlcil.methods.original_ddp.method import (
    SOURCE_DRIVER_SHA256,
    SOURCE_LOSS_SHA256,
    SOURCE_MODEL_SHA256,
    SOURCE_SNAPSHOT_TREE_SHA256,
    original_ddp_temperature,
)
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from tests.emotic_mlcil import protocol_config, task_context


class TinyPromptLearner(nn.Module):
    def __init__(self, classes=4, context=16, width=3):
        super().__init__()
        self.ctx_pos = nn.Parameter(torch.randn(classes, context, width) * 0.02)
        self.ctx_neg = nn.Parameter(torch.randn(classes, context, width) * 0.02)


class TinyOriginalDDP(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.n_cls = classes
        self.prompt_learner = TinyPromptLearner(classes=classes)
        self.visual_prompts = nn.Parameter(torch.randn(2 * classes, 16, 5) * 0.02)
        self.feature_adapter = None
        self.feature_adapter_bank = None
        self.text_feature_cache = {}

    def forward(self, images, cls_id=None, inference=False):
        low, high = cls_id
        signal = images.flatten(1).mean(dim=1, keepdim=True)
        negative = self.prompt_learner.ctx_neg[low:high].mean(dim=(1, 2))
        positive = self.prompt_learner.ctx_pos[low:high].mean(dim=(1, 2))
        negative = negative + self.visual_prompts[low:high].mean(dim=(1, 2))
        positive = positive + self.visual_prompts[
            self.n_cls + low : self.n_cls + high
        ].mean(dim=(1, 2))
        return torch.stack(
            [signal + negative.unsqueeze(0), signal + positive.unsqueeze(0)],
            dim=1,
        )


class BatchList(list):
    def __init__(self, values, batch_size):
        super().__init__(values)
        self.batch_size = batch_size


def tiny_protocol():
    config = protocol_config(
        class_order=("a", "b", "c", "d"),
        tasks=(("a", "b"), ("c", "d")),
    )
    config["method_options"]["original_ddp"] = {
        "epochs": 1,
        "registered_train_batch_size": 2,
        "amp": False,
        "tf32": False,
    }
    return BenchmarkProtocol.from_dict(config)


def train_batches(protocol, task_id):
    images = torch.tensor(
        [
            [[[1.0, 0.2], [0.4, 0.8]]],
            [[[0.3, 1.0], [0.7, 0.5]]],
        ]
    )
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    visible = torch.zeros(2, protocol.num_classes, dtype=torch.bool)
    visible[:, list(protocol.current_class_indices(task_id))] = True
    batch = TrainBatch(
        images=images,
        sample_ids=[f"task{task_id}-0", f"task{task_id}-1"],
        targets_current=targets,
        visible_mask=visible,
    )
    return BatchList([batch, batch], batch_size=2), BatchList([batch], batch_size=2)


class OriginalDDPBenchmarkMethodTest(unittest.TestCase):
    def make_method(self):
        return OriginalDDPBenchmarkMethod(
            tiny_protocol(), model=TinyOriginalDDP(), device="cpu"
        )

    def train_task(self, method, task_id):
        method.begin_task(task_context(method.protocol, task_id))
        train, val = train_batches(method.protocol, task_id)
        method.train_task(train, val)

    def test_registered_source_and_original_configuration(self):
        self.assertIn("original_ddp", method_names())
        self.assertIs(method_class("original_ddp"), OriginalDDPBenchmarkMethod)
        method = self.make_method()
        config = method.resolved_method_config()
        self.assertEqual(config["source_snapshot_tree_sha256"], SOURCE_SNAPSHOT_TREE_SHA256)
        self.assertEqual(config["source_model_sha256"], SOURCE_MODEL_SHA256)
        self.assertEqual(config["source_driver_sha256"], SOURCE_DRIVER_SHA256)
        self.assertEqual(config["source_loss_sha256"], SOURCE_LOSS_SHA256)
        self.assertEqual(config["epochs"], 1)
        self.assertEqual(config["n_ctx_positive"], 16)
        self.assertEqual(config["n_ctx_negative"], 16)
        self.assertEqual(config["visual_prompt_length"], 16)
        self.assertEqual(config["temperature_maximum"], 2.0)
        self.assertEqual(config["temperature_gamma"], 0.7)
        self.assertEqual(config["source_pcd_temperature"]["maximum"], 7.0)
        self.assertEqual(config["source_pcd_temperature"]["gamma"], 0.2)
        self.assertEqual(config["registered_pcd_temperature"]["maximum"], 2.0)
        self.assertEqual(config["registered_pcd_temperature"]["gamma"], 0.7)
        self.assertIn("requested", config["registered_pcd_mapping"])
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertFalse(config["replay_enabled"])
        self.assertTrue(config["clip_text_encoder_used"])

    def test_source_loss_is_summed_two_path_bce(self):
        logits = torch.tensor(
            [[[0.2, -0.5], [1.1, 0.4]], [[-0.3, 0.8], [0.7, -0.1]]]
        )
        targets = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        actual = OriginalDDPLoss()(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        expected = -(
            targets * torch.log(probabilities[:, 1].clamp_min(1.0e-6))
            + (1.0 - targets)
            * torch.log(probabilities[:, 0].clamp_min(1.0e-6))
        ).sum()
        self.assertTrue(torch.equal(actual, expected))

    def test_source_pcd_schedule(self):
        self.assertEqual(original_ddp_temperature(5, 26, 5), 1.0)
        self.assertEqual(original_ddp_temperature(26, 26, 5), 2.0)
        expected = 1.0 + ((8 - 5) / (26 - 5)) ** 0.7
        self.assertAlmostEqual(original_ddp_temperature(8, 26, 5), expected)

    def test_training_rejects_hidden_labels(self):
        method = self.make_method()
        method.begin_task(task_context(method.protocol, 0))
        train, val = train_batches(method.protocol, 0)
        train[0].visible_mask[:, 3] = True
        with self.assertRaisesRegex(ValueError, "old or future"):
            method.train_task(train, val)

    def test_optimizer_and_scheduler_continue_across_tasks(self):
        method = self.make_method()
        optimizer_id = id(method.optimizer)
        scheduler_id = id(method.scheduler)
        initial_last_epoch = method.scheduler.last_epoch
        self.train_task(method, 0)
        after_task0 = method.scheduler.last_epoch
        self.assertEqual(id(method.optimizer), optimizer_id)
        self.assertEqual(id(method.scheduler), scheduler_id)
        self.assertEqual(after_task0, initial_last_epoch + 1)
        method.end_task()
        self.train_task(method, 1)
        self.assertEqual(id(method.optimizer), optimizer_id)
        self.assertEqual(id(method.scheduler), scheduler_id)
        self.assertEqual(method.scheduler.last_epoch, after_task0 + 1)
        self.assertEqual(method._completed_task_id, 1)

    def test_checkpoint_round_trip_and_prediction_alignment(self):
        method = self.make_method()
        self.train_task(method, 0)
        _, val = train_batches(method.protocol, 0)
        batch = val[0]
        evaluation = EvaluationBatch(
            images=batch.images,
            sample_ids=list(batch.sample_ids),
            targets_seen=batch.targets_current,
            class_order_hash=method.protocol.class_order_hash,
            split_hash="original-ddp-fixed-eval",
        )
        prediction = method.predict_scores([evaluation])
        self.assertEqual(tuple(prediction.scores.shape), (2, 2))
        self.assertEqual(prediction.sample_ids, batch.sample_ids)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task0.pth"
            method.save_checkpoint(path)
            restored = self.make_method()
            restored.load_checkpoint(path)
            self.assertEqual(restored._completed_task_id, 0)
            self.assertEqual(
                restored.scheduler.last_epoch, method.scheduler.last_epoch
            )
            for name, value in method.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, restored.model.state_dict()[name]), name
                )
            restored.begin_task(task_context(restored.protocol, 1))

    def test_parameter_and_memory_statistics(self):
        method = self.make_method()
        stats = method.parameter_statistics()
        expected = 2 * 4 * 16 * 3 + 2 * 4 * 16 * 5
        self.assertEqual(stats.total_parameters, expected)
        self.assertEqual(stats.trainable_parameters, expected)
        self.assertEqual(stats.incremental_parameters, 0)
        self.assertEqual(dict(stats.per_task_incremental_parameters), {0: 0, 1: 0})
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)
        self.assertEqual(method.memory_statistics().replay_memory_bytes, 0)


if __name__ == "__main__":
    unittest.main()
