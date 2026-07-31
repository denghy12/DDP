import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.emotic_mlcil.artifacts import ArtifactStore
from benchmarks.emotic_mlcil.evaluator import BenchmarkEvaluator
from benchmarks.emotic_mlcil.runner import (
    BASE_COMMIT,
    CORE_RUNTIME_VERSION,
    _current_source_state,
    merge_task_shards,
)
from benchmarks.emotic_mlcil.types import PredictionOutput
from tests.emotic_mlcil.test_evaluator import small_protocol


class ParallelShardTest(unittest.TestCase):
    def _write_shard(self, store, protocol, run_id, task_id):
        seen = len(protocol.seen_class_indices(task_id))
        scores = torch.full((2, seen), 0.75)
        targets = torch.ones((2, seen))
        prediction = PredictionOutput(
            scores=scores,
            targets=targets,
            sample_ids=[f"task{task_id}-sample0", f"task{task_id}-sample1"],
            class_order_hash=protocol.class_order_hash,
            split_hash=f"split-{task_id}",
        )
        row = BenchmarkEvaluator(protocol).evaluate_task(
            task_id,
            prediction,
            expected_sample_ids=prediction.sample_ids,
            expected_split_hash=prediction.split_hash,
        )
        store.save_shard_scores(run_id, task_id, prediction)
        store.shard_checkpoint_path(run_id, task_id).write_bytes(b"checkpoint")
        store.append_shard_log(run_id, task_id, f"task={task_id}")
        state_dir = store.shard_dir / run_id / "_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / f"task{task_id}.log").write_text(
            f"console task {task_id}\n",
            encoding="utf-8",
        )
        (state_dir / f"task{task_id}.done").touch()
        store.write_task_shard_metadata(
            run_id,
            task_id,
            {
                "output_schema_version": protocol.output_schema_version,
                "protocol_id": protocol.protocol_id,
                "protocol_hash": protocol.protocol_hash,
                "class_order_hash": protocol.class_order_hash,
                "track": protocol.track,
                "seed": protocol.seed,
                "reporting_split": "val",
                "configuration_locked": False,
                "git_commit": "test-commit",
                "git_dirty": False,
                "source_tree_hash": "a" * 64,
                "base_commit": BASE_COMMIT,
                "core_runtime_version": CORE_RUNTIME_VERSION,
                "method": {
                    "name": "DDP",
                    "family": "Prompt",
                    "backbone": "OpenAI CLIP ViT-B/16",
                    "upstream_repository": "TBD",
                    "upstream_commit": "TBD",
                },
                "runner": {
                    "train_batch_size": 8,
                    "eval_batch_size": 4,
                    "num_workers": 0,
                },
                "task_metrics": row.as_dict(),
                "parameter_statistics": {
                    "total_parameters": 100,
                    "trainable_parameters": 10,
                    "incremental_parameters": 0,
                    "per_task_incremental_parameters": {"0": 0, "1": 0},
                },
                "memory_statistics": {
                    "replay_memory_samples": 0,
                    "replay_memory_bytes": 0,
                },
                "score_file": "scores.pt",
                "checkpoint_file": "checkpoint.pth",
            },
        )

    def test_two_task_shards_merge_into_standard_artifacts(self):
        protocol = small_protocol()
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(
                directory,
                protocol,
                track="A",
                method_name="DDP",
                seed=0,
            )
            for task_id in range(protocol.num_tasks):
                self._write_shard(store, protocol, "run_001", task_id)
            summary = merge_task_shards(
                protocol=protocol,
                artifact_store=store,
                shard_run_id="run_001",
                reporting_split="val",
                configuration_locked=False,
            )
            self.assertAlmostEqual(summary.final_mAP, 100.0, places=5)
            for task_id in range(protocol.num_tasks):
                self.assertTrue(
                    (store.score_dir / f"task{task_id}_scores.pt").is_file()
                )
                self.assertTrue(
                    (store.checkpoint_dir / f"task{task_id}.pth").is_file()
                )
            manifest = json.loads(
                (store.root / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["shard_run_id"], "run_001")
            self.assertEqual(manifest["git_commit"], "test-commit")
            self.assertTrue((store.root / "report.html").is_file())
            state_dir = store.shard_dir / "run_001" / "_state"
            (state_dir / "merge.log").write_text(
                "merge complete\n",
                encoding="utf-8",
            )
            sync_directory = store.export_sync_results("run_001")
            self.assertTrue(
                (sync_directory / "sync_manifest.json").is_file()
            )
            self.assertTrue(
                (sync_directory / "scores" / "task1_scores.pt").is_file()
            )
            self.assertFalse(list(sync_directory.rglob("*.pth")))

    def test_missing_task_shard_prevents_merge(self):
        protocol = small_protocol()
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(
                directory,
                protocol,
                track="A",
                method_name="DDP",
                seed=0,
            )
            self._write_shard(store, protocol, "incomplete", 0)
            with self.assertRaisesRegex(FileNotFoundError, "Missing completed"):
                merge_task_shards(
                    protocol=protocol,
                    artifact_store=store,
                    shard_run_id="incomplete",
                    reporting_split="val",
                    configuration_locked=False,
                )

    def test_unsafe_shard_run_id_is_rejected(self):
        protocol = small_protocol()
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(
                directory,
                protocol,
                track="A",
                method_name="DDP",
                seed=0,
            )
            with self.assertRaisesRegex(ValueError, "shard_run_id"):
                store.task_shard_dir("../escape", 0)

    def test_source_state_falls_back_when_server_mirror_has_no_git(self):
        unavailable = SimpleNamespace(returncode=128, stdout="", stderr="")
        with patch(
            "benchmarks.emotic_mlcil.runner.subprocess.run",
            return_value=unavailable,
        ):
            state = _current_source_state()
        self.assertEqual(state["git_commit"], BASE_COMMIT)
        self.assertTrue(state["git_dirty"])
        self.assertEqual(len(state["source_tree_hash"]), 64)


if __name__ == "__main__":
    unittest.main()
