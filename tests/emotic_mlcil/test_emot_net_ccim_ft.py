import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from benchmarks.emotic_mlcil.methods.emot_net_ccim_ft import (
    CCIMBackdoorIntervention,
    EMOTNetCCIMFTBenchmarkMethod,
    EMOTNetCCIMFTModel,
    EMOTNetCCIMFTOptions,
)
from benchmarks.emotic_mlcil.methods.emot_net_ccim_ft.method import (
    CCIM_DICTIONARY_SCOPE,
    CCIM_FEATURE_EXTRACTOR,
)
from benchmarks.emotic_mlcil.methods.emot_net_ccim_ft.model import (
    CCIM_SOURCE_SHA256,
    CCIM_UPSTREAM_COMMIT,
    CCIM_UPSTREAM_REPOSITORY,
)
from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.types import EvaluationBatch, TrainBatch
from tests.emotic_mlcil import protocol_config, task_context


class TinyEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(3, width)

    def forward(self, images):
        return self.projection(images.mean(dim=(-2, -1)))


def tiny_protocol(seed=0):
    config = protocol_config(
        class_order=("a", "b", "c", "d"),
        tasks=(("a", "b"), ("c", "d")),
    )
    config["track"] = "B"
    config["seed"] = seed
    return BenchmarkProtocol.from_dict(config)


def dictionary_payload(protocol):
    return {
        "schema_version": 1,
        "resource": "EMOT-Net+CCIM confounder dictionary",
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.with_seed(0).protocol_hash,
        "class_order_hash": protocol.class_order_hash,
        "task_id": 0,
        "scope": CCIM_DICTIONARY_SCOPE,
        "strategy": "dp_cause",
        "feature_extractor": CCIM_FEATURE_EXTRACTOR,
        "dictionary_size": 3,
        "confounder_dim": 6,
        "construction_seed": 0,
        "sample_ids_sha256": "1" * 64,
        "feature_checkpoint_sha256": "2" * 64,
        "feature_checkpoint_source": "resnet152_places365.pth.tar",
        "ccim_repository": CCIM_UPSTREAM_REPOSITORY,
        "ccim_commit": CCIM_UPSTREAM_COMMIT,
        "ccim_source_sha256": CCIM_SOURCE_SHA256,
        "dictionary": torch.arange(18, dtype=torch.float32).reshape(3, 6) / 18,
        "prior": torch.tensor([[0.2], [0.3], [0.5]]),
    }


def task_batch(protocol, task_id):
    context = torch.randn(4, 3, 224, 224)
    body = torch.randn(4, 3, 224, 224)
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


class EMOTNetCCIMFTTest(unittest.TestCase):
    def make_method(self, seed=0):
        protocol = tiny_protocol(seed)
        payload = dictionary_payload(protocol)
        torch.manual_seed(9)
        model = EMOTNetCCIMFTModel(
            payload["dictionary"],
            payload["prior"],
            context_encoder=TinyEncoder(640),
            body_encoder=TinyEncoder(256),
            fusion_dim=8,
            ccim_hidden_dim=4,
            ccim_attention_dim=5,
            dropout=0.0,
        )
        method = EMOTNetCCIMFTBenchmarkMethod(
            protocol,
            device="cpu",
            model=model,
            dictionary_payload=payload,
            option_overrides={
                "fusion_dim": 8,
                "epochs": 1,
                "early_stopping_patience": 1,
                "learning_rate": 0.01,
                "lr_drop_epoch": 1,
                "dropout": 0.0,
                "amp": False,
                "tf32": False,
                "ccim_hidden_dim": 4,
                "ccim_attention_dim": 5,
                "ccim_confounder_dim": 6,
                "ccim_dictionary_size": 3,
            },
        )
        return method

    def train(self, method, task_id):
        method.begin_task(task_context(method.protocol, task_id))
        batch = task_batch(method.protocol, task_id)
        method.train_task([batch], [batch])
        return batch

    def test_registration_and_frozen_contract(self):
        self.assertIn("emot_net_ccim_ft", method_names())
        self.assertIs(
            method_class("emot_net_ccim_ft"), EMOTNetCCIMFTBenchmarkMethod
        )
        config = self.make_method().resolved_method_config()
        self.assertEqual(config["ccim_commit"], CCIM_UPSTREAM_COMMIT)
        self.assertEqual(config["ccim_dictionary_size"], 3)
        self.assertEqual(config["ccim_confounder_dim"], 6)
        self.assertTrue(config["ccim_dictionary_frozen_after_task0"])
        self.assertFalse(config["ccim_future_task_images_used"])
        self.assertFalse(config["replay_enabled"])
        self.assertFalse(config["distillation_enabled"])
        self.assertFalse(config["benchmark_added_adapter"])
        self.assertEqual(EMOTNetCCIMFTOptions().ccim_dictionary_size, 256)

    def test_dot_intervention_matches_direct_formula(self):
        torch.manual_seed(4)
        module = CCIMBackdoorIntervention(7, 11, attention_dim=5).double().eval()
        joint = torch.randn(3, 7, dtype=torch.double)
        dictionary = torch.randn(6, 11, dtype=torch.double)
        prior = torch.rand(6, 1, dtype=torch.double)
        prior /= prior.sum()
        intervention, attention = module.intervene(joint, dictionary, prior)
        affinity = module.query(joint) @ module.key(dictionary).t() / (11 ** 0.5)
        expected_attention = affinity.softmax(dim=-1)
        expected = (
            expected_attention.unsqueeze(2)
            * dictionary
            * prior
        ).sum(dim=1)
        self.assertTrue(torch.allclose(attention, expected_attention))
        self.assertTrue(torch.allclose(intervention, expected))

    def test_lifecycle_seed_invariant_dictionary_and_checkpoint(self):
        method = self.make_method(seed=2)
        first = self.train(method, 0)
        method.end_task()
        second = self.train(method, 1)
        evaluation = EvaluationBatch(
            images=second.images,
            sample_ids=list(second.sample_ids),
            targets_seen=torch.cat((first.targets_current, second.targets_current), dim=1),
            class_order_hash=method.protocol.class_order_hash,
            split_hash="ccim-eval",
        )
        output = method.predict_scores([evaluation])
        self.assertEqual(tuple(output.scores.shape), (4, 4))
        self.assertEqual(method.model.head_sizes, (2, 2))
        self.assertEqual(method.parameter_statistics().incremental_parameters, 10)
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)
        self.assertGreater(
            method.resolved_method_config()["persistent_auxiliary_bytes"], 0
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "task1.pth"
            method.save_checkpoint(checkpoint)
            restored = self.make_method(seed=2)
            restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.model.head_sizes, (2, 2))
            self.assertTrue(
                torch.equal(
                    restored.model.confounder_dictionary,
                    method.model.confounder_dictionary,
                )
            )

    def test_rejects_future_or_mismatched_dictionary(self):
        protocol = tiny_protocol()
        payload = dictionary_payload(protocol)
        payload["scope"] = "all_training_images"
        with self.assertRaisesRegex(ValueError, "scope"):
            EMOTNetCCIMFTBenchmarkMethod(
                protocol,
                device="cpu",
                model=EMOTNetCCIMFTModel(
                    payload["dictionary"],
                    payload["prior"],
                    context_encoder=TinyEncoder(640),
                    body_encoder=TinyEncoder(256),
                    fusion_dim=8,
                    ccim_hidden_dim=4,
                    ccim_attention_dim=5,
                    dropout=0.0,
                ),
                dictionary_payload=payload,
                option_overrides={
                    "fusion_dim": 8,
                    "ccim_hidden_dim": 4,
                    "ccim_attention_dim": 5,
                    "ccim_confounder_dim": 6,
                    "ccim_dictionary_size": 3,
                },
            )


if __name__ == "__main__":
    unittest.main()
