import json
import tempfile
import unittest
from pathlib import Path

import torch

from ddp_internal_adapter import SharedResidualFeatureAdapter
from emotic_cls_to_final_token_transfer import (
    TRANSFER_RESIDUAL_SCALE,
    TaskRoutedCLSToFinalTokenTransferBank,
)
from emotic_final_token_adapter import (
    TaskRoutedFinalTokenAdapterBank,
    validate_final_token_checkpoint,
)
from emotic_task_adapter_bank import (
    class_to_task_map,
    file_sha256,
    task_class_range,
)


class CLSToFinalTokenTransferTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.classnames = [f"class{index}" for index in range(26)]
        self.source_states = {}
        entries = {}
        for task_id in range(2):
            adapter = SharedResidualFeatureAdapter(
                feature_dim=4,
                bottleneck_dim=1,
                residual_scale=0.1,
            )
            with torch.no_grad():
                adapter.down.weight.zero_()
                adapter.down.weight[0, 0] = 1.0
                adapter.up.weight.zero_()
                adapter.up.weight[1 + task_id, 0] = 1.0
            self.source_states[task_id] = {
                key: value.detach().clone()
                for key, value in adapter.state_dict().items()
            }
            task_dir = self.root / f"task{task_id}"
            task_dir.mkdir()
            checkpoint_path = task_dir / "best_adapter.pth"
            torch.save(
                {
                    "model": adapter.state_dict(),
                    "classnames": self.classnames,
                    "task_adapter": {
                        "schema_version": 1,
                        "task_id": task_id,
                        "class_range": list(task_class_range(task_id)),
                        "training_mode": "full",
                        "seed": 7,
                    },
                    "args": {
                        "feature_dim": 4,
                        "adapter_dim": 1,
                        "residual_scale": 0.1,
                        "inference_alpha": TRANSFER_RESIDUAL_SCALE,
                        "correction_mode": "linear_residual",
                        "formula": "feature_difference",
                    },
                },
                checkpoint_path,
            )
            entries[str(task_id)] = {
                "task_id": task_id,
                "class_range": list(task_class_range(task_id)),
                "checkpoint": str(checkpoint_path.relative_to(self.root)),
                "sha256": file_sha256(checkpoint_path),
            }

        self.manifest = {
            "schema_version": 1,
            "name": "test CLS Adapter Bank",
            "training_mode": "full",
            "seed": 7,
            "inference_formula": "feature_difference",
            "legacy_correction_mode": "linear_residual",
            "inference_alpha": TRANSFER_RESIDUAL_SCALE,
            "class_specific_gate": False,
            "task_specific_alpha": False,
            "test_used_for_selection": False,
            "classnames": self.classnames,
            "class_to_task": list(class_to_task_map()),
            "adapters": entries,
        }
        self.manifest_path = self.root / "adapter_bank_manifest.json"
        self.manifest_path.write_text(
            json.dumps(self.manifest),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def load_bank(self, application_mode, max_task=1):
        return TaskRoutedCLSToFinalTokenTransferBank.from_cls_manifest(
            self.manifest_path,
            application_mode=application_mode,
            device="cpu",
            max_task=max_task,
            classnames=self.classnames,
        )

    @staticmethod
    def source_tokens(seen_classes=8, token_count=5):
        tokens = torch.zeros(1, 2 * seen_classes, 4, token_count)
        tokens[:, :, 0, :] = 1.0
        return tokens

    def test_factory_copies_cls_weights_exactly_and_freezes_them(self):
        bank = self.load_bank("all_tokens")
        for task_id in range(2):
            transferred = bank.adapters[str(task_id)]
            for key, expected in self.source_states[task_id].items():
                self.assertTrue(torch.equal(transferred.state_dict()[key], expected))
            self.assertEqual(transferred.residual_scale, TRANSFER_RESIDUAL_SCALE)
            self.assertFalse(transferred.training)
            self.assertTrue(
                all(not parameter.requires_grad for parameter in transferred.parameters())
            )

    def test_all_tokens_mode_modifies_every_token_and_preserves_layout(self):
        bank = self.load_bank("all_tokens")
        source = self.source_tokens()
        adapted, aux = bank.adapt_token_features(
            source,
            seen_classes=8,
            return_aux=True,
        )
        self.assertEqual(tuple(adapted.shape), tuple(source.shape))
        delta = (adapted - source).norm(dim=2)
        self.assertTrue(torch.all(delta > 0))
        self.assertEqual(aux["application_mode"], "all_tokens")
        self.assertTrue(
            all(row["adapted_tokens_per_path"] == 5 for row in aux["tasks"])
        )

    def test_cls_only_mode_changes_only_token_zero_bitwise(self):
        bank = self.load_bank("cls_only")
        source = self.source_tokens()
        adapted = bank.adapt_token_features(source, seen_classes=8)
        self.assertGreater((adapted[..., 0] - source[..., 0]).norm().item(), 0.0)
        self.assertTrue(torch.equal(adapted[..., 1:], source[..., 1:]))

    def test_negative_and_positive_paths_share_introduction_adapter(self):
        bank = self.load_bank("cls_only")
        adapted = bank.adapt_token_features(
            self.source_tokens(),
            seen_classes=8,
        )
        task0_paths = list(range(0, 5)) + list(range(8, 13))
        task1_paths = list(range(5, 8)) + list(range(13, 16))
        self.assertTrue(torch.all(adapted[0, task0_paths, 1, 0] > 0))
        self.assertEqual(adapted[0, task0_paths, 2, 0].abs().max().item(), 0.0)
        self.assertTrue(torch.all(adapted[0, task1_paths, 2, 0] > 0))
        self.assertEqual(adapted[0, task1_paths, 1, 0].abs().max().item(), 0.0)

    def test_zero_cls_weights_recover_original_normalized_tokens(self):
        checkpoint_path = self.root / "task0" / "best_adapter.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["model"]["up.weight"].zero_()
        torch.save(checkpoint, checkpoint_path)
        self.manifest["adapters"]["0"]["sha256"] = file_sha256(checkpoint_path)
        self.manifest_path.write_text(
            json.dumps(self.manifest),
            encoding="utf-8",
        )
        bank = self.load_bank("all_tokens", max_task=0)
        source = self.source_tokens(seen_classes=5)
        adapted = bank.adapt_token_features(source, seen_classes=5)
        self.assertTrue(torch.equal(adapted, source))

    def test_provenance_exposes_manifest_and_checkpoint_hashes(self):
        bank = self.load_bank("all_tokens")
        self.assertEqual(bank.source_manifest_path, str(self.manifest_path.resolve()))
        self.assertEqual(
            bank.source_manifest_sha256,
            file_sha256(self.manifest_path),
        )
        self.assertEqual(
            bank.source_checkpoint_hashes[0],
            self.manifest["adapters"]["0"]["sha256"],
        )
        provenance = bank.transfer_provenance()
        provenance["source_checkpoints"]["0"]["sha256"] = "mutated"
        self.assertNotEqual(
            provenance["source_checkpoints"]["0"]["sha256"],
            bank.source_checkpoint_hashes[0],
        )

    def test_transfer_rejects_non_locked_alpha(self):
        self.manifest["inference_alpha"] = 0.1
        self.manifest_path.write_text(
            json.dumps(self.manifest),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "inference_alpha=0.03"):
            self.load_bank("all_tokens")

    def test_transfer_rejects_hash_mismatch(self):
        self.manifest["adapters"]["0"]["sha256"] = "0" * 64
        self.manifest_path.write_text(
            json.dumps(self.manifest),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load_bank("all_tokens")

    def test_transfer_rejects_invalid_application_mode(self):
        with self.assertRaisesRegex(ValueError, "application_mode"):
            self.load_bank("patch_only")

    def test_regular_final_token_loaders_still_reject_cls_artifacts(self):
        checkpoint = torch.load(
            self.root / "task0" / "best_adapter.pth",
            map_location="cpu",
        )
        with self.assertRaisesRegex(ValueError, "final_token_adapter"):
            validate_final_token_checkpoint(checkpoint)
        with self.assertRaisesRegex(ValueError, "manifest schema"):
            TaskRoutedFinalTokenAdapterBank.from_manifest(
                self.manifest_path,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
