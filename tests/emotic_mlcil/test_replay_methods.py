import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.emotic_mlcil.methods.replay import (
    ERBenchmarkMethod,
    PRSBenchmarkMethod,
)
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.replay_memory import ReplayMemoryContract
from tests.emotic_mlcil import task_context
from tests.emotic_mlcil.test_clip_continual_baselines import (
    IdentityFeatureExtractor,
    task_batches,
    tiny_protocol,
)


def tiny_replay_contract(protocol):
    return ReplayMemoryContract.from_mapping(
        {
            "contract_id": "tiny_replay_2c",
            "protocol_id": protocol.protocol_id,
            "sample_unit": "unique_emotic_person_sample",
            "capacity": {
                "kind": "per_seen_class",
                "samples_per_seen_class": 2,
                "task_capacities": [4, 8],
            },
            "payload": {
                "image": "post_transform_float32_tensor",
                "targets": "visible_binary_columns_only",
                "visible_mask": True,
                "stable_sample_id": True,
            },
            "training": {
                "update_timing": "task_end_single_pass",
                "replay_to_current_ratio": 1.0,
                "sample_without_replacement": True,
                "current_and_replay_equal_sample_weight": True,
            },
        },
        protocol,
    )


def make_method(method_class):
    protocol = tiny_protocol()
    return method_class(
        protocol,
        device="cpu",
        feature_extractor=IdentityFeatureExtractor(),
        memory_contract=tiny_replay_contract(protocol),
    )


def train_task(method, task_id):
    method.begin_task(task_context(method.protocol, task_id))
    train_loader, val_loader = task_batches(method.protocol, task_id)
    method.train_task(train_loader, val_loader)


class ReplayMethodTest(unittest.TestCase):
    def test_methods_are_registered_and_share_the_training_contract(self):
        self.assertIn("er", method_names())
        self.assertIn("prs", method_names())
        self.assertIs(method_class("er"), ERBenchmarkMethod)
        self.assertIs(method_class("prs"), PRSBenchmarkMethod)
        er = make_method(ERBenchmarkMethod)
        prs = make_method(PRSBenchmarkMethod)
        self.assertEqual(er.options, prs.options)
        self.assertEqual(er.memory_contract, prs.memory_contract)
        self.assertEqual(er.resolved_method_config()["replay_policy"], "reservoir")
        self.assertEqual(
            prs.resolved_method_config()["replay_policy"],
            "partitioned_reservoir",
        )
        self.assertFalse(hasattr(er.model, "adapter"))

    def test_er_replays_old_visible_labels_and_never_stores_future_truth(self):
        method = make_method(ERBenchmarkMethod)
        train_task(method, 0)
        self.assertEqual(method.memory_statistics().replay_memory_samples, 4)
        for row in method.replay_memory.records:
            self.assertTrue(bool(row.visible_mask[:2].all()))
            self.assertFalse(bool(row.visible_mask[2:].any()))
            self.assertFalse(bool((row.targets[2:] != 0).any()))
        method.end_task()
        train_task(method, 1)
        self.assertEqual(method.memory_statistics().replay_memory_samples, 8)
        self.assertTrue(
            any(row["replay_examples"] > 0 for row in method.training_history)
        )
        self.assertEqual(method.replay_memory.capacity, 8)
        self.assertGreater(method.memory_statistics().replay_memory_bytes, 0)

    def test_prs_uses_same_replay_exposure_and_records_source_identity(self):
        method = make_method(PRSBenchmarkMethod)
        train_task(method, 0)
        method.end_task()
        train_task(method, 1)
        config = method.resolved_method_config()
        self.assertEqual(config["prs_allocation_power"], -0.03)
        self.assertEqual(
            config["upstream_commit"],
            "136cee1863af03cc914dc05dfd41bda8b7bc0bf2",
        )
        self.assertTrue(any(row["replay_examples"] > 0 for row in method.training_history))

    def test_checkpoint_round_trip_restores_model_memory_and_rng(self):
        method = make_method(ERBenchmarkMethod)
        train_task(method, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task0.pth"
            method.save_checkpoint(path)
            restored = make_method(ERBenchmarkMethod)
            restored.load_checkpoint(path)
            self.assertEqual(restored._completed_task_id, 0)
            self.assertEqual(
                [row.sample_id for row in restored.replay_memory.records],
                [row.sample_id for row in method.replay_memory.records],
            )
            self.assertEqual(
                restored.memory_statistics(),
                method.memory_statistics(),
            )
            for key, value in method.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value.cpu(), restored.model.state_dict()[key].cpu())
                )
            self.assertEqual(
                restored.replay_memory.sample(2)[0].sample_id,
                method.replay_memory.sample(2)[0].sample_id,
            )


if __name__ == "__main__":
    unittest.main()
