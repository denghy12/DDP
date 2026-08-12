import json
import copy
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from benchmarks.emotic_mlcil.data_module import _stack_or_pad_images
from benchmarks.emotic_mlcil.methods.dsct_ft import (
    DSCTFTBenchmarkMethod,
    DSCTFTModel,
    DSCTFTOptions,
    dsct_current_task_loss,
    target_boxes_from_transport,
    target_query_indices,
)
from benchmarks.emotic_mlcil.methods.dsct_ft.model import _patch_ms_deform_attention_amp
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from src.helper_functions.emotic_loader import DSCTSceneTransform
from tests.emotic_mlcil import protocol_config, task_context


class Nested:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask


class TinyDSCTCore(nn.Module):
    def __init__(self, hidden=8, layers=2, queries=4):
        super().__init__()
        self.projection = nn.Linear(3, hidden)
        self.class_embed_dsct = nn.ModuleList([nn.Linear(hidden, 27) for _ in range(layers)])
        boxes = torch.tensor(
            [[0.25, 0.25, 0.2, 0.2], [0.5, 0.5, 0.4, 0.4],
             [0.75, 0.75, 0.2, 0.2], [0.5, 0.5, 0.9, 0.9]],
            dtype=torch.float32,
        )[:queries]
        self.boxes = nn.Parameter(boxes.clone())

    def forward(self, nested):
        features = self.projection(nested.tensors.mean(dim=(-2, -1)))
        features = features[:, None, :].expand(-1, self.boxes.shape[0], -1)
        logits = [head(features) for head in self.class_embed_dsct]
        boxes = self.boxes.sigmoid().unsqueeze(0).expand(features.shape[0], -1, -1)
        return {
            "pred_logits": logits[-1],
            "pred_boxes": boxes,
            "aux_outputs": [{"pred_logits": value, "pred_boxes": boxes} for value in logits[:-1]],
        }


def tiny_protocol():
    config = protocol_config(
        class_order=("a", "b", "c", "d"),
        tasks=(("a", "b"), ("c", "d")),
    )
    config["track"] = "B"
    return BenchmarkProtocol.from_dict(config)


def transport(batch=4, height=24, width=32):
    images = torch.zeros(batch, 5, height, width)
    images[:, :3] = torch.rand(batch, 3, height, width)
    images[:, 3, 6:18, 8:24] = 1
    images[:, 4] = 1
    return images


def train_batch(protocol, task_id=0):
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    visible = torch.zeros(4, protocol.num_classes, dtype=torch.bool)
    visible[:, list(protocol.current_class_indices(task_id))] = True
    return TrainBatch(
        images=transport(), sample_ids=[f"row-{i}" for i in range(4)],
        targets_current=targets, visible_mask=visible,
    )


class DSCTFTTest(unittest.TestCase):
    def make_method(self):
        protocol = tiny_protocol()
        model = DSCTFTModel(TinyDSCTCore(), Nested, hidden_dim=8)
        method = DSCTFTBenchmarkMethod(
            protocol, device="cpu", model=model,
            option_overrides={"hidden_dim": 8, "epochs": 1,
                              "early_stopping_patience": 1, "lr_drop_epoch": 1},
        )
        return protocol, method

    def test_registration_track_and_source_defaults(self):
        self.assertIn("dsct_ft", method_names())
        self.assertIs(method_class("dsct_ft"), DSCTFTBenchmarkMethod)
        options = DSCTFTOptions()
        self.assertEqual(options.epochs, 50)
        self.assertEqual(options.num_queries, 4)
        self.assertEqual(options.class_loss_weight, 5.0)
        self.assertEqual(options.effective_train_batch_size, 4)
        self.assertEqual(options.per_gpu_micro_batch_size, 2)
        self.assertEqual(options.maximum_data_parallel_replicas, 2)
        self.assertTrue(options.amp)
        self.assertTrue(options.tf32)
        self.assertTrue(options.channels_last)
        with self.assertRaisesRegex(ValueError, "effective train batch size 4"):
            DSCTFTOptions.from_mapping({"effective_train_batch_size": 2})
        with self.assertRaisesRegex(ValueError, "per-GPU micro-batch size 2"):
            DSCTFTOptions.from_mapping({"per_gpu_micro_batch_size": 1})
        with self.assertRaisesRegex(ValueError, "two DataParallel replicas"):
            DSCTFTOptions.from_mapping({"maximum_data_parallel_replicas": 4})
        with self.assertRaisesRegex(ValueError, "Track B"):
            DSCTFTBenchmarkMethod(
                BenchmarkProtocol.from_dict(protocol_config()), model=DSCTFTModel(TinyDSCTCore(), Nested)
            )

    def test_amp_bridge_is_class_level_and_replica_safe(self):
        class MSDeformAttn(nn.Module):
            def forward(self, value):
                return value + 1

        operator = MSDeformAttn()
        container = nn.Sequential(operator)
        self.assertEqual(_patch_ms_deform_attention_amp(container), 1)
        replica = copy.deepcopy(container)
        value = torch.ones(2, dtype=torch.float64)
        self.assertEqual(container(value).dtype, torch.float32)
        self.assertEqual(replica(value).dtype, torch.float32)
        self.assertTrue(torch.equal(container(value), replica(value)))

    def test_transport_box_and_official_iou_query_selection(self):
        images = transport(batch=1, height=20, width=40)
        box = target_boxes_from_transport(images)
        self.assertTrue(torch.allclose(box, torch.tensor([[0.4, 0.6, 0.4, 0.6]])))
        predictions = torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.4, 0.6, 0.4, 0.6]]])
        self.assertEqual(target_query_indices(predictions, box, torch.tensor([[20.0, 40.0]])).item(), 1)

    def test_loss_ignores_old_logits_and_is_finite(self):
        torch.manual_seed(3)
        output = {
            "pred_logits": torch.randn(2, 4, 5),
            "pred_boxes": torch.rand(2, 4, 4),
            "target_boxes": torch.tensor([[0.5, 0.5, 0.3, 0.4], [0.4, 0.4, 0.2, 0.3]]),
        }
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        first, _ = dsct_current_task_loss(output, targets, current_classes=2)
        changed = {key: value.clone() for key, value in output.items()}
        changed["pred_logits"][..., :2] += 1000
        second, _ = dsct_current_task_loss(changed, targets, current_classes=2)
        self.assertTrue(torch.isfinite(first))
        self.assertTrue(torch.allclose(first, second))

    def test_method_trains_predicts_and_checkpoints(self):
        protocol, method = self.make_method()
        context = task_context(protocol, 0)
        method.begin_task(context)
        batch = train_batch(protocol)
        progress = StringIO()
        with redirect_stdout(progress):
            method.train_task([batch], [batch])
        self.assertIn("DSCT_PROGRESS", progress.getvalue())
        progress_payload = json.loads(
            next(
                line.removeprefix("DSCT_PROGRESS ")
                for line in progress.getvalue().splitlines()
                if line.startswith("DSCT_PROGRESS ")
            )
        )
        self.assertEqual(progress_payload["task"], 0)
        self.assertEqual(progress_payload["epoch_number"], 1)
        self.assertEqual(progress_payload["epochs_planned"], 1)
        self.assertIn("task_eta_seconds", progress_payload)
        history = method.training_log_records()
        self.assertEqual(history[0]["optimizer_steps"], 1.0)
        self.assertEqual(history[0]["micro_batches"], 2.0)
        self.assertEqual(history[0]["effective_train_batch_size"], 4.0)
        evaluation = EvaluationBatch(
            images=batch.images, sample_ids=batch.sample_ids,
            targets_seen=batch.targets_current, class_order_hash=protocol.class_order_hash,
            split_hash="split",
        )
        prediction = method.predict_scores([evaluation])
        self.assertEqual(prediction.scores.shape, (4, 2))
        self.assertTrue((prediction.scores >= 0).all() and (prediction.scores <= 1).all())
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)
        config = method.resolved_method_config()
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertFalse(config["clip_visual_encoder_used"])
        self.assertEqual(config["execution_mode"], "data_parallel_micro_batch_then_accumulate")
        self.assertEqual(config["optimizer_steps_per_effective_batch"], 1)
        self.assertTrue(config["amp"])
        self.assertTrue(config["tf32"])
        self.assertTrue(config["channels_last"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "task0.pth"
            method.save_checkpoint(checkpoint)
            _, restored = self.make_method()
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.model.head_sizes, (2,))

    def test_micro_batch_partition_preserves_effective_batch(self):
        _, method = self.make_method()
        images = transport()
        targets = torch.zeros(4, 2)
        method._execution_gpu_count = 1
        single_gpu = list(method._training_micro_batches(images, targets))
        self.assertEqual([chunk[0].shape[0] for chunk in single_gpu], [2, 2])
        method._execution_gpu_count = 2
        two_gpu = list(method._training_micro_batches(images, targets))
        self.assertEqual([chunk[0].shape[0] for chunk in two_gpu], [4])

    def test_micro_batch_accumulation_matches_effective_batch_update(self):
        protocol, accumulated = self.make_method()
        _, effective_batch = self.make_method()
        context = task_context(protocol, 0)
        accumulated.begin_task(context)
        effective_batch.begin_task(context)
        effective_batch.model.load_state_dict(accumulated.model.state_dict(), strict=True)
        accumulated._execution_gpu_count = 1
        effective_batch._execution_gpu_count = 2
        accumulated._parallel_model = accumulated.model
        effective_batch._parallel_model = effective_batch.model
        batch = train_batch(protocol)
        with redirect_stdout(StringIO()):
            accumulated.train_task([batch], [batch])
            effective_batch.train_task([batch], [batch])
        self.assertEqual(accumulated.training_log_records()[0]["micro_batches"], 2.0)
        self.assertEqual(effective_batch.training_log_records()[0]["micro_batches"], 1.0)
        accumulated_state = accumulated.model.state_dict()
        effective_state = effective_batch.model.state_dict()
        self.assertEqual(tuple(accumulated_state), tuple(effective_state))
        for name in accumulated_state:
            self.assertTrue(
                torch.allclose(
                    accumulated_state[name], effective_state[name], rtol=1e-5, atol=1e-6
                ),
                msg=f"effective-batch update differs after micro accumulation: {name}",
            )

    def test_scene_transform_and_variable_padding(self):
        image = Image.new("RGB", (20, 10), "white")
        transformed = DSCTSceneTransform(train=False, eval_short_side=10, max_size=30)(
            image, [2, 1, 12, 9]
        )
        self.assertEqual(transformed.shape, (5, 10, 20))
        self.assertGreater(transformed[3].sum().item(), 0)
        padded = _stack_or_pad_images((transformed, transformed[:, :, :15]))
        self.assertEqual(padded.shape, (2, 5, 10, 20))
        self.assertEqual(padded[1, 4, :, 15:].sum().item(), 0)

    def test_formal_launcher_freezes_four_gpu_worst_case_contract(self):
        repository = Path(__file__).resolve().parents[2]
        launcher = (repository / "scripts/emotic-mlcil/launch_dsct_ft_formal_seed0_tmux.sh").read_text()
        worker = (repository / "scripts/emotic-mlcil/run_dsct_ft_formal_seed0.sh").read_text()
        self.assertIn('GPU="${GPU:-5,6}"', launcher)
        self.assertIn("--batch-size 4 --height 800 --width 1333", launcher)
        self.assertIn("DSCT_FT_TRACK_B_V0_4_FAST", launcher)
        self.assertIn("DSCT_FT_TRACK_B_V0_4_FAST", worker)
        self.assertIn("TRAIN_BATCH_SIZE=4", worker)
        self.assertIn("EVAL_BATCH_SIZE=32", worker)
        self.assertIn("WORKERS=2", worker)


if __name__ == "__main__":
    unittest.main()
