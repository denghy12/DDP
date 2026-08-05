"""Tests for the alphabetic EMOTIC B4-C2 benchmark protocol."""

from __future__ import annotations

import unittest
from pathlib import Path

from benchmarks.emotic_mlcil.protocol import load_protocol
from benchmarks.emotic_mlcil.replay_memory import ReplayMemoryContract


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "emotic_mlcil"


class B4C2ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.b5c3 = load_protocol(CONFIG_ROOT / "protocol_b5c3.yaml")
        cls.b4c2 = load_protocol(CONFIG_ROOT / "protocol_b4c2.yaml")

    def test_alphabetic_class_order_is_unchanged(self) -> None:
        self.assertEqual(self.b4c2.class_order, self.b5c3.class_order)
        self.assertEqual(
            self.b4c2.class_order,
            tuple(sorted(self.b4c2.class_order, key=str.casefold)),
        )
        self.assertEqual(self.b4c2.class_order_hash, self.b5c3.class_order_hash)

    def test_b4c2_has_one_base_and_eleven_incremental_tasks(self) -> None:
        self.assertEqual(self.b4c2.protocol_id, "emotic_b4c2_v0.1")
        self.assertEqual(self.b4c2.num_tasks, 12)
        self.assertEqual(tuple(map(len, self.b4c2.tasks)), (4,) + (2,) * 11)
        self.assertEqual(
            tuple(len(self.b4c2.seen_class_indices(task)) for task in range(12)),
            tuple(range(4, 27, 2)),
        )
        self.assertEqual(
            self.b4c2.task_class_ranges,
            (
                (0, 4),
                (4, 6),
                (6, 8),
                (8, 10),
                (10, 12),
                (12, 14),
                (14, 16),
                (16, 18),
                (18, 20),
                (20, 22),
                (22, 24),
                (24, 26),
            ),
        )
        self.assertNotEqual(self.b4c2.protocol_hash, self.b5c3.protocol_hash)

    def test_original_ddp_temperature_mapping_is_explicit(self) -> None:
        options = self.b4c2.method_options("original_ddp")
        self.assertEqual(
            options,
            {
                "temperature_minimum": 1.0,
                "temperature_maximum": 2.0,
                "temperature_gamma": 0.7,
            },
        )

    def test_b4c2_replay_contracts_use_20_per_seen_class(self) -> None:
        standard = ReplayMemoryContract.from_yaml(
            CONFIG_ROOT / "replay_b4c2_20c_v0.1.yaml",
            self.b4c2,
        )
        derpp = ReplayMemoryContract.from_yaml(
            CONFIG_ROOT / "replay_derpp_b4c2_20c_v0.1.yaml",
            self.b4c2,
        )
        self.assertEqual(standard.task_capacities, tuple(range(80, 521, 40)))
        self.assertEqual(derpp.task_capacities, standard.task_capacities)
        self.assertFalse(standard.stores_logits)
        self.assertEqual(standard.replay_draws_per_current_batch, 1)
        self.assertTrue(derpp.stores_logits)
        self.assertEqual(derpp.replay_draws_per_current_batch, 2)
        self.assertEqual(derpp.objective_weighting, "source_derpp_weighted_sum")


if __name__ == "__main__":
    unittest.main()
