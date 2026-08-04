import unittest
from pathlib import Path

import torch

from benchmarks.emotic_mlcil.protocol import load_protocol
from benchmarks.emotic_mlcil.replay_memory import (
    PartitionedReservoirReplayBuffer,
    ReplayMemoryContract,
    ReplayRecord,
    ReservoirReplayBuffer,
)


ROOT = Path(__file__).resolve().parents[2]


def record(sample_id, positives, visible=(0, 1, 2, 3), image_value=1.0):
    targets = torch.zeros(4)
    targets[list(positives)] = 1.0
    mask = torch.zeros(4, dtype=torch.bool)
    mask[list(visible)] = True
    return ReplayRecord(
        image=torch.full((3, 2, 2), image_value),
        sample_id=sample_id,
        targets=targets,
        visible_mask=mask,
    )


class ReplayMemoryTest(unittest.TestCase):
    def test_registered_emotic_budget_matches_b5c3_seen_classes(self):
        protocol = load_protocol(
            ROOT / "configs/emotic_mlcil/protocol_b5c3.yaml"
        )
        contract = ReplayMemoryContract.from_yaml(
            ROOT / "configs/emotic_mlcil/replay_20c_v0.1.yaml",
            protocol,
        )
        self.assertEqual(
            contract.task_capacities,
            (100, 160, 220, 280, 340, 400, 460, 520),
        )
        self.assertEqual(contract.samples_per_seen_class, 20)
        self.assertEqual(contract.replay_to_current_ratio, 1.0)

    def test_record_rejects_hidden_truth_and_merges_newly_visible_columns(self):
        first = record("person", (0,), visible=(0, 1), image_value=1.0)
        second = record("person", (2,), visible=(2, 3), image_value=2.0)
        merged = first.merged_with(second)
        self.assertEqual(merged.targets.tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(merged.visible_mask.tolist(), [True, True, True, True])
        self.assertTrue(torch.equal(merged.image, second.image))
        hidden = torch.tensor([1.0, 0.0, 1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "Hidden"):
            ReplayRecord(
                image=torch.zeros(1),
                sample_id="invalid",
                targets=hidden,
                visible_mask=torch.tensor([True, True, False, False]),
            )

    def test_reservoir_is_bounded_deterministic_and_checkpointable(self):
        left = ReservoirReplayBuffer(num_classes=4, seed=11)
        right = ReservoirReplayBuffer(num_classes=4, seed=11)
        left.set_capacity(3)
        right.set_capacity(3)
        stream = [record(f"sample-{index}", (index % 4,)) for index in range(20)]
        for item in stream:
            left.observe(item)
            right.observe(item)
        self.assertEqual(len(left), 3)
        self.assertEqual(
            [item.sample_id for item in left.records],
            [item.sample_id for item in right.records],
        )
        self.assertGreater(left.memory_statistics().replay_memory_bytes, 0)
        restored = ReservoirReplayBuffer(num_classes=4, seed=99)
        restored.load_state_dict(left.state_dict())
        self.assertEqual(
            [item.sample_id for item in restored.records],
            [item.sample_id for item in left.records],
        )
        self.assertEqual(restored.sample(2)[0].sample_id, left.sample(2)[0].sample_id)

    def test_prs_is_deterministic_and_tracks_positive_stream_frequency(self):
        left = PartitionedReservoirReplayBuffer(
            num_classes=4,
            seed=7,
            allocation_power=-0.03,
        )
        right = PartitionedReservoirReplayBuffer(
            num_classes=4,
            seed=7,
            allocation_power=-0.03,
        )
        for buffer in (left, right):
            buffer.set_capacity(4)
        stream = [
            record("a", (0,)),
            record("b", (0,)),
            record("c", (1,)),
            record("d", (0, 1)),
            record("e", (2,)),
            record("f", (3,)),
            record("g", (0,)),
            record("h", (2, 3)),
        ]
        for item in stream:
            left.observe(item)
            right.observe(item)
        self.assertEqual(len(left), 4)
        self.assertEqual(left.observed_positive_counts, [4.0, 2.0, 2.0, 2.0])
        self.assertEqual(
            [item.sample_id for item in left.records],
            [item.sample_id for item in right.records],
        )
        proportions = left._active_proportions()
        self.assertTrue(proportions)
        expected_weights = {
            index: left.observed_positive_counts[index] ** left.allocation_power
            for index in proportions
        }
        expected_total = sum(expected_weights.values())
        for index, proportion in proportions.items():
            self.assertAlmostEqual(
                proportion,
                expected_weights[index] / expected_total,
                places=12,
            )
        restored = PartitionedReservoirReplayBuffer(
            num_classes=4,
            seed=99,
            allocation_power=-0.03,
        )
        restored.load_state_dict(left.state_dict())
        self.assertEqual(
            restored.observed_positive_counts,
            left.observed_positive_counts,
        )
        self.assertEqual(restored.substream_order, left.substream_order)

    def test_repeated_id_is_merged_without_growing_sample_count(self):
        memory = ReservoirReplayBuffer(num_classes=4, seed=0)
        memory.set_capacity(4)
        memory.observe(record("same", (0,), visible=(0, 1)))
        memory.observe(record("same", (2,), visible=(2, 3)))
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory.records[0].positive_indices, (0, 2))


if __name__ == "__main__":
    unittest.main()
