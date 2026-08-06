import unittest

from benchmarks.emotic_mlcil.methods.task_adapter_bank import TaskAdapterBankOptions
from benchmarks.emotic_mlcil.protocol import load_protocol
from emotic_task_adapter_bank import task_ranges_from_sizes


class TaskAdapterBankFormalProtocolTest(unittest.TestCase):
    def test_frozen_protocols_map_to_expected_adapter_counts(self):
        cases = {
            "configs/emotic_mlcil/protocol_b10c4.yaml": [10, 4, 4, 4, 4],
            "configs/emotic_mlcil/protocol_b4c2.yaml": [4] + [2] * 11,
        }
        for path, sizes in cases.items():
            protocol = load_protocol(path)
            self.assertEqual([len(task) for task in protocol.tasks], sizes)
            ranges = task_ranges_from_sizes(sizes)
            self.assertEqual(ranges[-1][1], 26)
            self.assertEqual(len(ranges), protocol.num_tasks)
            self.assertEqual(protocol.threshold, 0.5)

    def test_locked_best_task_bank_hyperparameters(self):
        options = TaskAdapterBankOptions()
        self.assertEqual(options.adapter_epochs, 50)
        self.assertEqual(options.adapter_bottleneck_dim, 128)
        self.assertEqual(options.training_residual_scale, 0.1)
        self.assertEqual(options.inference_alpha, 0.03)
        self.assertEqual(options.identity_weight, 0.1)
        self.assertEqual(options.positive_weight_max, 20.0)

    def test_invalid_options_fail_closed(self):
        with self.assertRaises(ValueError):
            TaskAdapterBankOptions.from_mapping({"inference_alpha": -0.1})
        with self.assertRaises(ValueError):
            TaskAdapterBankOptions.from_mapping({"unknown": 1})


if __name__ == "__main__":
    unittest.main()
