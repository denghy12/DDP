import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from emotic_task_adapter_bank import (
    TaskRoutedAdapterBank,
    validate_task_adapter_checkpoint,
)
from train_emotic_ddp_task_adapter import loss_configuration


class TaskAdapterLossProtocolTest(unittest.TestCase):
    def _args(self, loss_name):
        return SimpleNamespace(
            classification_loss=loss_name,
            class_balanced_bce=(loss_name == "weighted_bce"),
            asl_gamma_neg=9.8,
            asl_gamma_pos=0.0,
            asl_clip=0.05,
            bal_weight_power=1.6,
            bal_label_smoothing=0.1,
            bal_smoothing_num_classes=26,
        )

    def test_loss_configuration_is_deterministic_and_loss_specific(self):
        first = loss_configuration(self._args("asl"))
        second = loss_configuration(self._args("asl"))
        bal = loss_configuration(self._args("bal_paper"))
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(first["sha256"], bal["sha256"])

    def test_bal_component_ablation_has_four_distinct_locked_configs(self):
        configurations = {
            name: loss_configuration(self._args(name))
            for name in (
                "asl",
                "asl_smoothing",
                "asl_positive_weight",
                "bal_paper",
            )
        }
        self.assertEqual(
            len({config["sha256"] for config in configurations.values()}),
            4,
        )
        self.assertIsNone(configurations["asl"]["bal_weight_power"])
        self.assertEqual(configurations["asl"]["bal_label_smoothing"], 0.0)
        self.assertIsNone(
            configurations["asl_smoothing"]["bal_weight_power"]
        )
        self.assertEqual(
            configurations["asl_smoothing"]["bal_label_smoothing"], 0.1
        )
        self.assertEqual(
            configurations["asl_positive_weight"]["bal_weight_power"], 1.6
        )
        self.assertEqual(
            configurations["asl_positive_weight"][
                "bal_label_smoothing"
            ],
            0.0,
        )
        self.assertEqual(configurations["bal_paper"]["bal_weight_power"], 1.6)
        self.assertEqual(
            configurations["bal_paper"]["bal_label_smoothing"], 0.1
        )

    def test_checkpoint_rejects_a_mixed_loss_bank(self):
        configuration = loss_configuration(self._args("asl"))
        checkpoint = {
            "model": {"up.weight": torch.zeros(1)},
            "classnames": [f"class{index}" for index in range(26)],
            "task_adapter": {
                "task_id": 0,
                "class_range": [0, 5],
                "training_mode": "full",
                "seed": 0,
                "classification_loss": "asl",
                "loss_config": configuration,
                "checkpoint_rule": "last_epoch",
            },
        }
        validate_task_adapter_checkpoint(
            checkpoint,
            task_id=0,
            training_mode="full",
            seed=0,
            classification_loss="asl",
            loss_config_sha256=configuration["sha256"],
            checkpoint_rule="last_epoch",
        )
        with self.assertRaises(ValueError):
            validate_task_adapter_checkpoint(
                checkpoint,
                classification_loss="bal_paper",
            )
        with self.assertRaises(ValueError):
            validate_task_adapter_checkpoint(
                checkpoint,
                checkpoint_rule="best_val",
            )

    def test_bank_rejects_dynamic_routing_or_non_feature_difference(self):
        adapter = torch.nn.Identity()
        with self.assertRaises(ValueError):
            TaskRoutedAdapterBank(
                {0: adapter},
                routing_mode="confidence_gate",
            )
        with self.assertRaises(ValueError):
            TaskRoutedAdapterBank(
                {0: adapter},
                correction_mode="cosine_difference",
            )

    def test_loss_checkpoint_filename_is_unambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_adapter.pth"
            torch.save({"checkpoint_rule": "last_epoch"}, path)
            self.assertTrue(path.is_file())
            self.assertFalse((Path(directory) / "best_adapter.pth").exists())


if __name__ == "__main__":
    unittest.main()
