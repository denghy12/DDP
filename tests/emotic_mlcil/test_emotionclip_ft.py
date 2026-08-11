import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from benchmarks.emotic_mlcil.methods.emotionclip_ft import (
    EmotionCLIPFTBenchmarkMethod,
    EmotionCLIPFTModel,
    EmotionCLIPFTOptions,
    EmotionCLIPVisualTransformer,
)
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from src.helper_functions.emotic_loader import (
    EMOTIC,
    EmotionCLIPImageMaskTransform,
)
from tests.emotic_mlcil import protocol_config, task_context


class TinySubjectEncoder(nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.rgb = nn.Linear(3, 8)
        self.mask = nn.Linear(1, 8, bias=False)

    def forward(self, image, mask):
        return self.rgb(image.mean(dim=(-2, -1))) + self.mask(
            mask.mean(dim=(-2, -1), keepdim=False).unsqueeze(1)
        )


def tiny_protocol():
    config = protocol_config(
        class_order=("a", "b", "c", "d"),
        tasks=(("a", "b"), ("c", "d")),
    )
    config["track"] = "B"
    return BenchmarkProtocol.from_dict(config)


def task_batch(protocol, task_id):
    images = torch.zeros(4, 4, 224, 224)
    images[:, :3] = torch.tensor(
        [[1.0, 0.2, 0.4], [0.3, 1.0, 0.7], [0.8, 0.6, 1.0], [0.5, 0.9, 0.3]]
    ).view(4, 3, 1, 1)
    images[::2, 3, 40:160, 50:170] = 1
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


class EmotionCLIPFTTest(unittest.TestCase):
    def make_method(self):
        torch.manual_seed(7)
        return EmotionCLIPFTBenchmarkMethod(
            tiny_protocol(),
            device="cpu",
            model=EmotionCLIPFTModel(TinySubjectEncoder()),
            option_overrides={
                "feature_dim": 8,
                "epochs": 1,
                "early_stopping_patience": 1,
                "amp": False,
                "tf32": False,
            },
        )

    def train(self, method, task_id):
        method.begin_task(task_context(method.protocol, task_id))
        batch = task_batch(method.protocol, task_id)
        method.train_task([batch], [batch])
        return batch

    def test_registration_and_method_ft_contract(self):
        self.assertIn("emotionclip_ft", method_names())
        self.assertIs(method_class("emotionclip_ft"), EmotionCLIPFTBenchmarkMethod)
        config = self.make_method().resolved_method_config()
        self.assertEqual(config["upstream_commit"], "ca142dd4c9664acb2b59c8b14ef3169049f1180b")
        self.assertEqual(config["input_mode"], "image_bbox_mask")
        self.assertEqual(config["upstream_static_emotic_map"], 32.91)
        self.assertTrue(config["current_label_only"])
        self.assertTrue(config["visual_encoder_fine_tuned"])
        self.assertFalse(config["distillation_enabled"])
        self.assertFalse(config["replay_enabled"])
        self.assertFalse(config["ewc_enabled"])
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertFalse(config["clip_text_encoder_used"])
        self.assertEqual(EmotionCLIPFTOptions().epochs, 25)

    def test_subject_mask_changes_features(self):
        model = EmotionCLIPFTModel(TinySubjectEncoder()).eval()
        inputs = torch.zeros(2, 4, 224, 224)
        inputs[1, 3, 20:180, 30:190] = 1
        with torch.no_grad():
            features = model.encode(inputs)
        self.assertFalse(torch.equal(features[0], features[1]))

    def test_small_visual_tower_shapes_and_subject_token(self):
        model = EmotionCLIPVisualTransformer(
            image_size=32,
            patch_size=16,
            width=16,
            layers=1,
            heads=2,
            output_dim=8,
        ).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32), torch.ones(2, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertEqual(model.positional_embedding.shape[0], 5)

    def test_label_firewall_prediction_and_checkpoint(self):
        method = self.make_method()
        batch0 = self.train(method, 0)
        method.end_task()
        method.begin_task(task_context(method.protocol, 1))
        batch1 = task_batch(method.protocol, 1)
        batch1.visible_mask[:, 0] = True
        with self.assertRaisesRegex(ValueError, "outside current classes"):
            method.train_task([batch1], [batch1])
        batch1.visible_mask[:, 0] = False
        method.train_task([batch1], [batch1])
        evaluation = EvaluationBatch(
            images=batch1.images,
            sample_ids=list(batch1.sample_ids),
            targets_seen=torch.cat((batch0.targets_current, batch1.targets_current), dim=1),
            class_order_hash=method.protocol.class_order_hash,
            split_hash="emotionclip-eval",
        )
        output = method.predict_scores([evaluation])
        self.assertEqual(tuple(output.scores.shape), (4, 4))
        self.assertEqual(output.sample_ids, batch1.sample_ids)
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)
        self.assertGreater(method.parameter_statistics().trainable_parameters, 0)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "task1.pth"
            method.save_checkpoint(checkpoint)
            restored = self.make_method()
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.model.head_sizes, (2, 2))
            self.assertEqual(restored._completed_task_id, 1)

    def test_rejects_rgb_without_subject_mask(self):
        method = self.make_method()
        method.begin_task(task_context(method.protocol, 0))
        batch = task_batch(method.protocol, 0)
        batch.images = batch.images[:, :3]
        with self.assertRaisesRegex(ValueError, "image_bbox_mask"):
            method.train_task([batch], [batch])

    def test_loader_packages_rgb_and_bbox_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "sample.png"
            Image.new("RGB", (400, 200), color=(10, 20, 30)).save(image_path)
            dataset = EMOTIC.__new__(EMOTIC)
            dataset.file_paths = [str(image_path)]
            dataset.body_bboxes = [[100, 20, 300, 180]]
            dataset.classes = ["a"]
            dataset.targets = [[0]]
            dataset.input_mode = "image_bbox_mask"
            dataset.transform = EmotionCLIPImageMaskTransform()
            inputs, target = dataset[0]
            self.assertEqual(tuple(inputs.shape), (4, 224, 224))
            self.assertGreater(float(inputs[3].sum()), 0.0)
            self.assertLessEqual(float(inputs[3].max()), 1.0)
            self.assertEqual(target.tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
