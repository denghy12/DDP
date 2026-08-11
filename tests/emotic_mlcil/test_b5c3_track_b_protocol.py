import unittest
from pathlib import Path

from benchmarks.emotic_mlcil.protocol import load_protocol


class B5C3TrackBProtocolTest(unittest.TestCase):
    def test_track_b_changes_model_track_not_data_or_metrics(self):
        root = Path(__file__).resolve().parents[2] / "configs" / "emotic_mlcil"
        track_a = load_protocol(root / "protocol_b5c3.yaml")
        track_b = load_protocol(root / "protocol_b5c3_track_b.yaml")
        self.assertEqual(track_a.track, "A")
        self.assertEqual(track_b.track, "B")
        self.assertEqual(track_a.class_order, track_b.class_order)
        self.assertEqual(track_a.tasks, track_b.tasks)
        self.assertEqual(track_a.train_split, track_b.train_split)
        self.assertEqual(track_a.validation_split, track_b.validation_split)
        self.assertEqual(track_a.test_split, track_b.test_split)
        self.assertEqual(track_a.threshold, track_b.threshold)
        self.assertEqual(track_a.class_order_hash, track_b.class_order_hash)
        self.assertNotEqual(track_a.protocol_hash, track_b.protocol_hash)


if __name__ == "__main__":
    unittest.main()
