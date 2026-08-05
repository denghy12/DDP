import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.emotic_mlcil.methods.replay import (
    DERPPBenchmarkMethod,
    ERBenchmarkMethod,
    PRSBenchmarkMethod,
    derpp_weighted_loss,
    masked_logit_mse,
)
from benchmarks.emotic_mlcil.methods.replay.method import ReplayContinualMethod
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


def tiny_derpp_contract(protocol):
    return ReplayMemoryContract.from_mapping(
        {
            "contract_id": "tiny_derpp_replay_2c",
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
                "logits": "capture_time_visible_logits_only",
                "logit_mask": True,
            },
            "training": {
                "update_timing": "online_after_optimizer_attempt",
                "replay_to_current_ratio": 1.0,
                "replay_draws_per_current_batch": 2,
                "sample_without_replacement": True,
                "current_and_replay_equal_sample_weight": False,
                "objective_weighting": "source_derpp_weighted_sum",
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
        memory_contract=(
            tiny_derpp_contract(protocol)
            if method_class is DERPPBenchmarkMethod
            else tiny_replay_contract(protocol)
        ),
    )


def train_task(method, task_id):
    method.begin_task(task_context(method.protocol, task_id))
    train_loader, val_loader = task_batches(method.protocol, task_id)
    method.train_task(train_loader, val_loader)


class ReplayMethodTest(unittest.TestCase):
    def test_v09_label_only_contract_checkpoint_is_normalized(self):
        normalized = ReplayContinualMethod._normalized_checkpoint_contract(
            {"contract_id": "legacy"}
        )
        self.assertFalse(normalized["stores_logits"])
        self.assertEqual(normalized["stored_logits"], "none")
        self.assertEqual(normalized["replay_draws_per_current_batch"], 1)
        self.assertEqual(normalized["objective_weighting"], "equal_sample_mean")

    def test_methods_are_registered_and_share_the_training_contract(self):
        self.assertIn("er", method_names())
        self.assertIn("prs", method_names())
        self.assertIn("derpp", method_names())
        self.assertIs(method_class("er"), ERBenchmarkMethod)
        self.assertIs(method_class("prs"), PRSBenchmarkMethod)
        self.assertIs(method_class("derpp"), DERPPBenchmarkMethod)
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

    def test_derpp_objective_and_masked_logit_mse(self):
        current = torch.tensor(1.25)
        dark = masked_logit_mse(
            torch.tensor([[1.0, 8.0, 3.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[True, False, True]]),
        )
        self.assertAlmostEqual(float(dark), 2.5, places=7)
        total = derpp_weighted_loss(
            current,
            dark,
            torch.tensor(0.75),
            alpha=0.5,
            beta=0.5,
        )
        self.assertAlmostEqual(float(total), 2.875, places=7)

    def test_derpp_stores_masked_logits_online_and_replays_within_task(self):
        method = make_method(DERPPBenchmarkMethod)
        train_task(method, 0)
        config = method.resolved_method_config()
        self.assertEqual(config["derpp_alpha"], 0.5)
        self.assertEqual(config["derpp_beta"], 0.5)
        self.assertEqual(config["independent_replay_draws_per_current_batch"], 2)
        self.assertTrue(config["same_task_replay_enabled"])
        self.assertEqual(method.memory_statistics().replay_memory_samples, 4)
        self.assertTrue(
            any(row["replay_examples"] > 0 for row in method.training_history)
        )
        for row in method.replay_memory.records:
            self.assertIsNotNone(row.logits)
            self.assertTrue(bool(row.logit_mask[:2].all()))
            self.assertFalse(bool(row.logit_mask[2:].any()))
            self.assertFalse(bool((row.logits[2:] != 0).any()))
            self.assertFalse(bool(row.visible_mask[2:].any()))
            self.assertFalse(bool((row.targets[2:] != 0).any()))
        method.end_task()
        train_task(method, 1)
        self.assertEqual(method.memory_statistics().replay_memory_samples, 8)
        old_rows = [
            row for row in method.replay_memory.records if row.sample_id.startswith("task0-")
        ]
        new_rows = [
            row for row in method.replay_memory.records if row.sample_id.startswith("task1-")
        ]
        self.assertTrue(old_rows and new_rows)
        for row in old_rows:
            self.assertFalse(bool(row.logit_mask[2:].any()))
            self.assertFalse(bool(row.visible_mask[2:].any()))
        for row in new_rows:
            self.assertTrue(bool(row.logit_mask.all()))
            self.assertFalse(bool(row.visible_mask[:2].any()))
            self.assertFalse(bool((row.targets[:2] != 0).any()))

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

    def test_derpp_checkpoint_round_trip_restores_dark_payload_and_rng(self):
        method = make_method(DERPPBenchmarkMethod)
        train_task(method, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task0.pth"
            method.save_checkpoint(path)
            restored = make_method(DERPPBenchmarkMethod)
            restored.load_checkpoint(path)
            self.assertEqual(restored._completed_task_id, 0)
            self.assertEqual(
                [row.sample_id for row in restored.replay_memory.records],
                [row.sample_id for row in method.replay_memory.records],
            )
            for left, right in zip(
                restored.replay_memory.records,
                method.replay_memory.records,
            ):
                self.assertTrue(torch.equal(left.logits, right.logits))
                self.assertTrue(torch.equal(left.logit_mask, right.logit_mask))
            self.assertEqual(
                restored.replay_memory.sample(2)[0].sample_id,
                method.replay_memory.sample(2)[0].sample_id,
            )


if __name__ == "__main__":
    unittest.main()
