import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "validate_original_ddp_formal_results.py"
)
LAUNCHER = SCRIPT.with_name("launch_original_ddp_formal_seed012_tmux.sh")
WORKER = SCRIPT.with_name("run_original_ddp_formal_seed012.sh")
SPEC = importlib.util.spec_from_file_location(
    "validate_original_ddp_formal_results", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


COMMIT = "a" * 40


def write_seed(
    root,
    seed,
    *,
    locked=True,
    temperature_gamma=0.7,
    workers=0,
    complete_log=True,
):
    seed_root = (
        root
        / "benchmarks"
        / "emotic_b5c3_v0.1"
        / "A"
        / "Original-DDP-Tau2"
        / f"seed{seed}"
    )
    (seed_root / "metrics").mkdir(parents=True)
    method_configuration = dict(MODULE.LOCKED_METHOD_CONFIGURATION)
    method_configuration["registered_pcd_temperature"] = {
        "minimum": 1.0,
        "maximum": 2.0,
        "gamma": temperature_gamma,
    }
    protocol_hash = f"protocol-{seed}"
    manifest = {
        "method": "Original-DDP-Tau2",
        "protocol_id": "emotic_b5c3_v0.1",
        "track": "A",
        "seed": seed,
        "git_commit": COMMIT,
        "git_dirty": False,
        "source_tree_hash": "tree",
        "protocol_hash": protocol_hash,
        "class_order_hash": "classes",
        "data_split_hash": {"task0": "test"},
        "core_base_commit": "core",
        "core_runtime_version": "0.7.0",
        "method_configuration": method_configuration,
        "test_labels_used_for_selection": False,
        "reporting_split": "test",
        "configuration_locked": locked,
        "prediction_reused_task_ids": [],
        "eligible_for_main_table": locked,
        "replay_memory_samples": 0,
        "replay_memory_bytes": 0,
        **MODULE.EXPECTED_PARAMETER_STATISTICS,
    }
    runner = dict(MODULE.LOCKED_RUNNER_CONFIGURATION)
    runner["configuration_locked"] = locked
    runner["num_workers"] = workers
    config = {
        "protocol": {
            "seed": seed,
            "protocol_hash": protocol_hash,
            "class_order_hash": "classes",
        },
        "runner": runner,
    }
    main_table = {
        "replay_memory": {"samples": 0, "bytes": 0},
        "final_mAP": 40.0 + seed,
        "final_cF1": 30.0 + seed,
        "final_oF1": 45.0 + seed,
        "average_mAP": 44.0 + seed,
        "forgetting": 2.0 - seed * 0.5,
        "parameter_growth": 0,
    }
    task_metrics = [
        {"task_id": task, "mAP": 50.0 - task + seed}
        for task in range(8)
    ]
    (seed_root / "run_manifest.json").write_text(json.dumps(manifest))
    (seed_root / "config_resolved.json").write_text(json.dumps(config))
    (seed_root / "metrics" / "summary.json").write_text(
        json.dumps({"main_table": main_table, "summary": {}})
    )
    (seed_root / "metrics" / "task_metrics.json").write_text(
        json.dumps({"tasks": task_metrics})
    )

    lines = []
    for task in range(8):
        for epoch in range(20):
            if not complete_log and task == 7 and epoch == 19:
                continue
            row = {
                "epoch": float(epoch),
                "learning_rate": 0.00059 if task == 0 else 0.000059,
                "next_learning_rate": 0.000059,
                "optimizer_steps": 10.0,
                "scaled_source_bce": 0.5 - epoch * 0.01,
                "scheduler_last_epoch": float(task * 20 + epoch + 1),
                "skipped_optimizer_steps": 1.0 if epoch == 0 else 0.0,
                "task_id": float(task),
            }
            if epoch == 19:
                row["validation_current_mAP"] = 50.0 + task
            lines.append(f"task={task} training=" + json.dumps(row))
    (seed_root / "train.log").write_text("\n".join(lines) + "\n")


class OriginalDDPFormalResultValidationTest(unittest.TestCase):
    def test_formal_launcher_packs_three_seeds_on_gpu_zero(self):
        launcher = LAUNCHER.read_text()
        worker = WORKER.read_text()
        self.assertIn('GPU="${GPU:-0}"', launcher)
        self.assertIn('GPUS="${GPUS:-${GPU} ${GPU} ${GPU}}"', launcher)
        self.assertIn('MIN_FREE_MIB="${ORIGINAL_DDP_MIN_FREE_MIB:-14000}"', launcher)
        self.assertIn('GPUS="${GPUS:-0 0 0}"', worker)

    def test_validates_and_aggregates_three_locked_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (0, 1, 2):
                write_seed(root, seed)
            payload = MODULE.validate_original_ddp_formal_results(
                root, "formal_run", (0, 1, 2), COMMIT
            )
            self.assertEqual(payload["status"], "eligible_for_main_table")
            self.assertEqual(payload["aggregate"]["final_mAP"]["mean"], 41.0)
            self.assertEqual(payload["aggregate"]["final_mAP"]["std"], 1.0)
            self.assertEqual(
                payload["aggregate"]["aggregation"],
                "mean_and_sample_standard_deviation",
            )
            stability = payload["training_stability"][0]
            self.assertEqual(stability["optimizer_updates"], 1600)
            self.assertEqual(stability["amp_overflow_skips"], 8)
            self.assertEqual(stability["scheduler_last_epoch"], 160)

    def test_rejects_unlocked_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0)
            write_seed(root, 1, locked=False)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "configuration_locked"):
                MODULE.validate_original_ddp_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )

    def test_rejects_pcd_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0, temperature_gamma=0.71)
            write_seed(root, 1)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "registered_pcd_temperature"):
                MODULE.validate_original_ddp_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )

    def test_rejects_worker_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0, workers=2)
            write_seed(root, 1)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "num_workers"):
                MODULE.validate_original_ddp_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )

    def test_rejects_incomplete_training_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0, complete_log=False)
            write_seed(root, 1)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "epoch record count"):
                MODULE.validate_original_ddp_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )


if __name__ == "__main__":
    unittest.main()
