import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from benchmarks.emotic_mlcil.methods.clip_classifier import (
    ElasticWeightConsolidationMethod,
    LearningWithoutForgettingMethod,
    SequentialFineTuningMethod,
)
from benchmarks.emotic_mlcil.artifacts import ArtifactStore
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.runner import BenchmarkRunner
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from tests.emotic_mlcil import protocol_config, task_context


class IdentityFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4))

    def forward(self, images):
        return self.projection(images)


def tiny_protocol():
    config = protocol_config(
        class_order=("a", "b", "c", "d"),
        tasks=(("a", "b"), ("c", "d")),
    )
    config["method_options"]["clip_classifier"] = {
        "feature_dim": 4,
        "epochs": 3,
        "early_stopping_patience": 3,
        "backbone_learning_rate": 0.05,
        "head_learning_rate": 0.05,
        "weight_decay": 0.0,
        "gradient_clip_norm": 10.0,
        "amp": False,
        "tf32": False,
        "lwf_temperature": 2.0,
        "lwf_weight": 1.0,
        "ewc_lambda": 10.0,
        "ewc_decay": 1.0,
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
        targets = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
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
    mask = torch.zeros(images.shape[0], protocol.num_classes, dtype=torch.bool)
    mask[:, list(protocol.current_class_indices(task_id))] = True
    batch = TrainBatch(
        images=images,
        sample_ids=[f"task{task_id}-{index}" for index in range(images.shape[0])],
        targets_current=targets,
        visible_mask=mask,
    )
    return [batch], [batch]


class TinyDataModule:
    def __init__(self, protocol):
        self.protocol = protocol

    def method_loader(self, task_id, **kwargs):
        train, _ = task_batches(self.protocol, task_id)
        return train

    def evaluator_dataset(self, task_id, split, access):
        count = 2
        return SimpleNamespace(
            sample_ids=tuple(f"eval-{task_id}-{index}" for index in range(count)),
            split_hash=f"{split}-task{task_id}",
        )

    def evaluator_loader(self, task_id, split, access, **kwargs):
        if task_id == 0:
            images = torch.eye(4)[:2]
            targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        else:
            images = torch.eye(4)[2:]
            targets = torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        return [
            EvaluationBatch(
                images=images,
                sample_ids=[
                    f"eval-{task_id}-{index}"
                    for index in range(images.shape[0])
                ],
                targets_seen=targets,
                class_order_hash=self.protocol.class_order_hash,
                split_hash=f"{split}-task{task_id}",
            )
        ]


class CLIPContinualBaselineTest(unittest.TestCase):
    def make_method(self, method_class):
        return method_class(
            tiny_protocol(),
            device="cpu",
            feature_extractor=IdentityFeatureExtractor(),
        )

    def train_one_task(self, method, task_id):
        protocol = method.protocol
        method.begin_task(task_context(protocol, task_id))
        train_loader, val_loader = task_batches(protocol, task_id)
        method.train_task(train_loader, val_loader)

    def test_all_methods_are_registered(self):
        self.assertIn("finetune", method_names())
        self.assertIn("lwf", method_names())
        self.assertIn("ewc", method_names())
        self.assertIs(method_class("finetune"), SequentialFineTuningMethod)
        self.assertIs(method_class("lwf"), LearningWithoutForgettingMethod)
        self.assertIs(method_class("ewc"), ElasticWeightConsolidationMethod)

    def test_visual_encoder_is_trainable_and_no_adapter_is_present(self):
        method = self.make_method(SequentialFineTuningMethod)
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in method.model.visual_encoder.parameters()
            )
        )
        self.assertFalse(hasattr(method.model, "adapter"))
        initial = (
            method.model.visual_encoder.projection.weight.detach().clone()
        )
        self.train_one_task(method, 0)
        self.assertEqual(
            set(method._optimizer_parameter_names),
            set(dict(method.model.named_parameters())),
        )
        self.assertFalse(
            torch.equal(
                initial,
                method.model.visual_encoder.projection.weight.detach(),
            )
        )
        stats = method.parameter_statistics()
        self.assertEqual(stats.total_parameters, 26)
        self.assertEqual(stats.trainable_parameters, 26)
        self.assertEqual(stats.incremental_parameters, 0)
        self.assertEqual(stats.per_task_incremental_parameters, {0: 0, 1: 10})

    def test_sequential_finetuning_predicts_seen_classes_only(self):
        method = self.make_method(SequentialFineTuningMethod)
        self.train_one_task(method, 0)
        evaluation = EvaluationBatch(
            images=torch.eye(4)[:2],
            sample_ids=["sample-0", "sample-1"],
            targets_seen=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            class_order_hash=method.protocol.class_order_hash,
            split_hash="split",
        )
        output = method.predict_scores([evaluation])
        self.assertEqual(tuple(output.scores.shape), (2, 2))
        self.assertTrue(bool(((output.scores >= 0) & (output.scores <= 1)).all()))
        self.assertEqual(output.targets.tolist(), evaluation.targets_seen.tolist())
        self.assertEqual(output.sample_ids, evaluation.sample_ids)

    def test_tasks_must_be_trained_in_protocol_order(self):
        method = self.make_method(SequentialFineTuningMethod)
        with self.assertRaisesRegex(RuntimeError, "sequential tasks"):
            method.begin_task(task_context(method.protocol, 1))

    def test_training_rejects_visibility_outside_current_classes(self):
        method = self.make_method(SequentialFineTuningMethod)
        context = task_context(method.protocol, 0)
        method.begin_task(context)
        train_loader, _ = task_batches(method.protocol, 0)
        train_loader[0].visible_mask[:, 2] = True
        with self.assertRaisesRegex(ValueError, "outside current classes"):
            method.train_task(train_loader, train_loader)

    def test_lwf_uses_a_frozen_old_model_on_later_tasks(self):
        method = self.make_method(LearningWithoutForgettingMethod)
        self.train_one_task(method, 0)
        method.end_task()
        method.begin_task(task_context(method.protocol, 1))
        self.assertIsNotNone(method._teacher)
        self.assertFalse(
            any(parameter.requires_grad for parameter in method._teacher.parameters())
        )
        train_loader, val_loader = task_batches(method.protocol, 1)
        method.train_task(train_loader, val_loader)
        self.assertTrue(
            any(row["distillation_loss"] > 0 for row in method.training_history)
        )

    def test_ewc_estimates_fisher_and_penalizes_parameter_drift(self):
        method = self.make_method(ElasticWeightConsolidationMethod)
        self.train_one_task(method, 0)
        self.assertTrue(method._ewc_fisher)
        self.assertTrue(
            any(float(value.sum()) > 0 for value in method._ewc_fisher.values())
        )
        method.end_task()
        method.begin_task(task_context(method.protocol, 1))
        self.assertAlmostEqual(
            float(method._ewc_penalty().detach()),
            0.0,
            places=8,
        )
        with torch.no_grad():
            for name, parameter in method.model.named_parameters():
                if name in method._ewc_fisher:
                    parameter.add_(0.1)
        self.assertGreater(float(method._ewc_penalty().detach()), 0.0)

    def test_checkpoint_contains_visual_encoder_and_classifier(self):
        method = self.make_method(SequentialFineTuningMethod)
        self.train_one_task(method, 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "task0.pth"
            method.save_checkpoint(checkpoint)
            self.assertLess(checkpoint.stat().st_size, 1_000_000)
            payload = torch.load(checkpoint, map_location="cpu")
            self.assertEqual(payload["schema_version"], 2)
            self.assertIn(
                "visual_encoder.projection.weight",
                payload["model"],
            )
            self.assertFalse(
                any("adapter" in name for name in payload["model"])
            )
            restored = self.make_method(SequentialFineTuningMethod)
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.model.head_sizes, (2,))
            self.assertEqual(restored._completed_task_id, 0)
            for key, value in method.model.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), restored.model.state_dict()[key]))

    def test_standard_runner_writes_method_config_and_sequential_bundle(self):
        method = self.make_method(SequentialFineTuningMethod)
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(
                directory,
                method.protocol,
                track="A",
                method_name=method.method_name,
                seed=0,
            )
            runner = BenchmarkRunner(
                method.protocol,
                TinyDataModule(method.protocol),
                method,
                store,
                train_batch_size=2,
                eval_batch_size=2,
                num_workers=0,
            )
            summary = runner.run(reporting_split="val")
            self.assertEqual(len(summary.task_metrics), 2)
            manifest = json.loads(
                (store.root / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["method_configuration"]["strategy"],
                "finetune",
            )
            self.assertTrue(
                manifest["method_configuration"]["visual_encoder_trainable"]
            )
            self.assertFalse(
                manifest["method_configuration"]["clip_text_encoder_used"]
            )
            self.assertFalse(
                manifest["method_configuration"]["benchmark_added_adapter"]
            )
            self.assertFalse(manifest["eligible_for_main_table"])
            self.assertIn("selection_mAP", (store.root / "train.log").read_text())
            destination = store.export_sync_results("tiny_sequential")
            self.assertFalse(list(destination.rglob("*.pth")))


if __name__ == "__main__":
    unittest.main()
