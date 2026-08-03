import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "validate_multi_lane_formal_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_multi_lane_formal_results", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


COMMIT = "a" * 40


def write_seed(root, seed, *, locked=True, commit=COMMIT, lane_total=675840):
    seed_root = (
        root
        / "benchmarks"
        / "emotic_b5c3_v0.1"
        / "A"
        / "MULTI-LANE"
        / f"seed{seed}"
    )
    (seed_root / "metrics").mkdir(parents=True)
    method_configuration = dict(MODULE.LOCKED_METHOD_CONFIGURATION)
    method_configuration["task_lane_parameters_total"] = lane_total
    protocol_hash = f"protocol-{seed}"
    manifest = {
        "method": "MULTI-LANE",
        "protocol_id": "emotic_b5c3_v0.1",
        "track": "A",
        "seed": seed,
        "git_commit": commit,
        "git_dirty": False,
        "source_tree_hash": "tree",
        "protocol_hash": protocol_hash,
        "class_order_hash": "classes",
        "data_split_hash": {"task0": "test"},
        "core_base_commit": "core",
        "core_runtime_version": "0.5.0",
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
        "final_cF1": 35.0 + seed,
        "final_oF1": 55.0 + seed,
        "average_mAP": 48.0 + seed,
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
    log_lines = []
    for task in range(8):
        for epoch in range(30):
            record = {
                "current_loss": 0.7 - epoch / 1000,
                "epoch": float(epoch),
                "learning_rate": 0.01,
                "next_learning_rate": 0.0 if epoch == 29 else 0.01,
                "optimizer_steps": 10.0,
                "skipped_optimizer_steps": 0.0,
            }
            if epoch == 29:
                record["validation_current_mAP"] = 50.0
            log_lines.append(f"task={task} training={json.dumps(record)}")
    (seed_root / "train.log").write_text("\n".join(log_lines) + "\n")


class MultiLaneFormalResultValidationTest(unittest.TestCase):
    def test_validates_and_aggregates_three_locked_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (0, 1, 2):
                write_seed(root, seed)
            payload = MODULE.validate_multi_lane_formal_results(
                root, "formal_run", (0, 1, 2), COMMIT
            )
            self.assertEqual(payload["status"], "eligible_for_main_table")
            self.assertEqual(payload["aggregate"]["final_mAP"]["mean"], 41.0)
            self.assertEqual(payload["aggregate"]["final_mAP"]["std"], 1.0)
            self.assertEqual(
                payload["aggregate"]["aggregation"],
                "mean_and_sample_standard_deviation",
            )
            self.assertEqual(payload["training_stability"][0]["epochs_completed"], 240)
            self.assertEqual(
                payload["source"]["protocol_hash_by_seed"],
                {"0": "protocol-0", "1": "protocol-1", "2": "protocol-2"},
            )

    def test_rejects_unlocked_test_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0)
            write_seed(root, 1, locked=False)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "configuration_locked"):
                MODULE.validate_multi_lane_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )

    def test_rejects_changed_preallocated_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0)
            write_seed(root, 1, lane_total=675839)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "task_lane_parameters_total"):
                MODULE.validate_multi_lane_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )

    def test_rejects_incomplete_training_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (0, 1, 2):
                write_seed(root, seed)
            log = next(root.glob("benchmarks/*/A/MULTI-LANE/seed1/train.log"))
            lines = log.read_text().splitlines()
            log.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "epoch count"):
                MODULE.validate_multi_lane_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )


if __name__ == "__main__":
    unittest.main()
