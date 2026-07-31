import copy
import unittest
from pathlib import Path

from benchmarks.emotic_mlcil.protocol import BenchmarkProtocol, load_protocol
from tests.emotic_mlcil import EMOTIC_CLASSES, make_protocol, protocol_config


class ProtocolTest(unittest.TestCase):
    def test_frozen_b5c3_order_and_boundaries(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "emotic_mlcil"
            / "protocol_b5c3.yaml"
        )
        protocol = load_protocol(path)
        self.assertEqual(list(protocol.class_order), EMOTIC_CLASSES)
        self.assertEqual(
            protocol.task_class_ranges,
            (
                (0, 5),
                (5, 8),
                (8, 11),
                (11, 14),
                (14, 17),
                (17, 20),
                (20, 23),
                (23, 26),
            ),
        )
        self.assertEqual(protocol.current_class_indices(0), tuple(range(5)))
        self.assertEqual(protocol.seen_class_indices(3), tuple(range(14)))
        self.assertEqual(protocol.future_class_indices(6), tuple(range(23, 26)))

    def test_task_count_changes_with_config_only(self):
        classes = ["a", "b", "c", "d", "e", "f"]
        config = protocol_config(
            class_order=classes,
            tasks=[classes[:2], classes[2:5], classes[5:]],
        )
        protocol = BenchmarkProtocol.from_dict(config)
        self.assertEqual(protocol.num_tasks, 3)
        self.assertEqual(protocol.task_class_ranges, ((0, 2), (2, 5), (5, 6)))
        self.assertEqual(protocol.seen_class_indices(1), (0, 1, 2, 3, 4))

    def test_hashes_are_stable_and_content_sensitive(self):
        first_config = protocol_config()
        second_config = dict(reversed(list(copy.deepcopy(first_config).items())))
        first = BenchmarkProtocol.from_dict(first_config)
        second = BenchmarkProtocol.from_dict(second_config)
        self.assertEqual(first.class_order_hash, second.class_order_hash)
        self.assertEqual(first.protocol_hash, second.protocol_hash)

        changed = copy.deepcopy(first_config)
        changed["seed"] = 1
        third = BenchmarkProtocol.from_dict(changed)
        self.assertEqual(first.class_order_hash, third.class_order_hash)
        self.assertNotEqual(first.protocol_hash, third.protocol_hash)

    def test_seed_override_changes_only_seed_and_protocol_hash(self):
        protocol = make_protocol()
        changed = protocol.with_seed(2)
        self.assertEqual(changed.seed, 2)
        self.assertEqual(changed.class_order, protocol.class_order)
        self.assertEqual(changed.class_order_hash, protocol.class_order_hash)
        self.assertNotEqual(changed.protocol_hash, protocol.protocol_hash)

    def test_flattened_tasks_must_equal_class_order(self):
        config = protocol_config()
        config["tasks"][1][0], config["tasks"][1][1] = (
            config["tasks"][1][1],
            config["tasks"][1][0],
        )
        with self.assertRaisesRegex(ValueError, "Flattened tasks"):
            BenchmarkProtocol.from_dict(config)

    def test_threshold_scans_are_rejected(self):
        config = protocol_config()
        config["threshold_policy"]["per_task_selection"] = True
        with self.assertRaisesRegex(ValueError, "Per-task"):
            BenchmarkProtocol.from_dict(config)

        config = protocol_config()
        config["threshold_policy"]["value"] = 0.6
        with self.assertRaisesRegex(ValueError, "0.5"):
            BenchmarkProtocol.from_dict(config)

    def test_invalid_task_and_class_ids_are_rejected(self):
        protocol = make_protocol()
        with self.assertRaises(ValueError):
            protocol.current_class_indices(protocol.num_tasks)
        with self.assertRaises(ValueError):
            protocol.introduction_task(protocol.num_classes)


if __name__ == "__main__":
    unittest.main()
