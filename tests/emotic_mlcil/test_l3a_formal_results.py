import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "validate_l3a_formal_results.py"
)
SPEC = importlib.util.spec_from_file_location("validate_l3a_formal_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


COMMIT = "a" * 40


def write_seed(
    root,
    seed,
    *,
    locked=True,
    commit=COMMIT,
    pseudo_threshold=0.7,
    complete_log=True,
):
    seed_root = (
        root
        / "benchmarks"
        / "emotic_b5c3_v0.1"
        / "A"
        / "L3A"
        / f"seed{seed}"
    )
    (seed_root / "metrics").mkdir(parents=True)
    method_configuration = dict(MODULE.LOCKED_METHOD_CONFIGURATION)
    method_configuration["pseudo_threshold"] = pseudo_threshold
    protocol_hash = f"protocol-{seed}"
    manifest = {
        "method": "L3A",
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
        "core_runtime_version": "0.6.0",
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
        "final_cF1": 30.0 + seed,
        "final_oF1": 45.0 + seed,
        "average_mAP": 44.0 + seed,
        "forgetting": 2.0 - seed * 0.5,
        "parameter_growth": 86016,
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

    lines = [
        "task=0 training="
        + json.dumps(
            {
                "base_asl_loss": 30.0,
                "base_gradient_training": 1.0,
                "epoch": 0.0,
                "learning_rate": 1.0e-6,
                "optimizer_steps": 76.0,
                "skipped_optimizer_steps": 8.0,
                "task_id": 0.0,
            }
        )
    ]
    for task in range(8):
        lines.append(
            f"task={task} training="
            + json.dumps(
                {
                    "analytic_samples": 100.0 + task,
                    "base_gradient_training": 1.0 if task == 0 else 0.0,
                    "counted_samples": 100.0 + task,
                    "epoch": 1.0 if task == 0 else 0.0,
                    "pseudo_positive_labels": 0.0 if task == 0 else 5.0,
                    "task_id": float(task),
                    "validation_current_mAP": 50.0 + task,
                }
            )
        )
    if not complete_log:
        lines.pop()
    (seed_root / "train.log").write_text("\n".join(lines) + "\n")


class L3AFormalResultValidationTest(unittest.TestCase):
    def test_seed0_metric_blind_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0)
            payload = MODULE.validate_l3a_formal_results(
                root, "formal_run", (0,), COMMIT
            )
            self.assertEqual(payload["status"], "seed0_compliance_gate_passed")
            self.assertTrue(
                payload["seed0_gate_does_not_select_configuration_from_metrics"]
            )
            self.assertIsNone(payload["aggregate"]["final_mAP"]["std"])
            self.assertEqual(
                payload["training_stability"][0]["optimizer_updates"], 76
            )
            self.assertEqual(
                payload["training_stability"][0]["amp_overflow_skips"], 8
            )

    def test_validates_and_aggregates_three_locked_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (0, 1, 2):
                write_seed(root, seed)
            payload = MODULE.validate_l3a_formal_results(
                root, "formal_run", (0, 1, 2), COMMIT
            )
            self.assertEqual(payload["status"], "eligible_for_main_table")
            self.assertEqual(payload["aggregate"]["final_mAP"]["mean"], 41.0)
            self.assertEqual(payload["aggregate"]["final_mAP"]["std"], 1.0)
            self.assertEqual(
                payload["aggregate"]["aggregation"],
                "mean_and_sample_standard_deviation",
            )

    def test_rejects_unlocked_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0)
            write_seed(root, 1, locked=False)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "configuration_locked"):
                MODULE.validate_l3a_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )

    def test_rejects_configuration_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0, pseudo_threshold=0.71)
            with self.assertRaisesRegex(ValueError, "pseudo_threshold"):
                MODULE.validate_l3a_formal_results(
                    root, "formal_run", (0,), COMMIT
                )

    def test_rejects_incomplete_training_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0, complete_log=False)
            with self.assertRaisesRegex(ValueError, "training task IDs"):
                MODULE.validate_l3a_formal_results(
                    root, "formal_run", (0,), COMMIT
                )


if __name__ == "__main__":
    unittest.main()
