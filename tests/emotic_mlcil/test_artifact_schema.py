import json
import tempfile
import unittest

from benchmarks.emotic_mlcil.artifacts import (
    ArtifactStore,
    MAIN_TABLE_FIELDS,
    RUN_MANIFEST_REQUIRED_FIELDS,
    validate_run_manifest,
)
from benchmarks.emotic_mlcil.types import BenchmarkSummary, TaskMetrics
from tests.emotic_mlcil import make_protocol


def manifest(protocol):
    return {
        "method": "DDP",
        "method_family": "Prompt",
        "protocol_id": protocol.protocol_id,
        "track": protocol.track,
        "seed": protocol.seed,
        "git_commit": "abc",
        "base_commit": "f9459d0",
        "backbone": "OpenAI CLIP ViT-B/16",
        "class_order_hash": protocol.class_order_hash,
        "protocol_hash": protocol.protocol_hash,
        "data_split_hash": {"task0": "hash"},
        "threshold_policy": {"kind": "fixed_global", "value": 0.5},
        "replay_memory_samples": 0,
        "replay_memory_bytes": 0,
        "total_parameters": 100,
        "trainable_parameters": 10,
        "incremental_parameters": 0,
        "upstream_repository": "repository-local",
        "upstream_commit": "f9459d0",
        "test_labels_used_for_selection": False,
    }


class ArtifactSchemaTest(unittest.TestCase):
    def test_manifest_requires_every_frozen_field(self):
        protocol = make_protocol()
        payload = manifest(protocol)
        validate_run_manifest(payload)
        for field in RUN_MANIFEST_REQUIRED_FIELDS:
            broken = dict(payload)
            broken.pop(field)
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_run_manifest(broken)

    def test_test_selection_flag_is_enforced(self):
        payload = manifest(make_protocol())
        payload["test_labels_used_for_selection"] = True
        with self.assertRaisesRegex(ValueError, "must be false"):
            validate_run_manifest(payload)

    def test_standard_files_and_main_table_schema_are_written(self):
        protocol = make_protocol()
        row = TaskMetrics(
            task_id=0,
            seen_classes=5,
            samples=2,
            threshold=0.5,
            per_class_ap=[100.0] * 5,
            mAP=100.0,
            cPrecision=100.0,
            cRecall=100.0,
            cF1=100.0,
            oPrecision=100.0,
            oRecall=100.0,
            oF1=100.0,
            class_order_hash=protocol.class_order_hash,
            split_hash="hash",
        )
        summary = BenchmarkSummary(
            task_metrics=[row],
            final_mAP=100.0,
            average_mAP=100.0,
            final_cF1=100.0,
            final_oF1=100.0,
            forgetting=0.0,
            per_class_forgetting={},
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(
                directory,
                protocol,
                track="A",
                method_name="DDP",
                seed=0,
            )
            store.initialize(protocol.as_dict(), manifest(protocol))
            store.write_task_metrics([row])
            store.write_summary(
                summary,
                method_family="Prompt",
                backbone="OpenAI CLIP ViT-B/16",
                replay_memory_samples=0,
                replay_memory_bytes=0,
                parameter_growth=0,
            )
            self.assertTrue((store.root / "config_resolved.json").is_file())
            self.assertTrue((store.root / "run_manifest.json").is_file())
            self.assertTrue((store.root / "checkpoints").is_dir())
            self.assertTrue((store.root / "scores").is_dir())
            self.assertTrue((store.root / "metrics" / "task_metrics.json").is_file())
            self.assertTrue((store.root / "metrics" / "summary.json").is_file())
            self.assertTrue((store.root / "report.html").is_file())
            self.assertTrue((store.root / "train.log").is_file())
            payload = json.loads(
                (store.root / "metrics" / "summary.json").read_text()
            )
            self.assertEqual(
                tuple(payload["main_table"]),
                MAIN_TABLE_FIELDS,
            )


if __name__ == "__main__":
    unittest.main()
