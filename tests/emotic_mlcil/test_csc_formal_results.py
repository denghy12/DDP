import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "validate_csc_formal_results.py"
)
SPEC = importlib.util.spec_from_file_location("validate_csc_formal_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


COMMIT = "a" * 40


def write_seed(root, seed, *, locked=True, commit=COMMIT):
    seed_root = (
        root
        / "benchmarks"
        / "emotic_b5c3_v0.1"
        / "A"
        / "CSC"
        / f"seed{seed}"
    )
    (seed_root / "metrics").mkdir(parents=True)
    method_configuration = dict(MODULE.LOCKED_METHOD_CONFIGURATION)
    protocol_hash = f"protocol-{seed}"
    manifest = {
        "method": "CSC",
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
        "core_runtime_version": "0.4.0",
        "method_configuration": method_configuration,
        "test_labels_used_for_selection": False,
        "reporting_split": "test",
        "configuration_locked": locked,
        "prediction_reused_task_ids": [],
        "eligible_for_main_table": locked,
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
        "final_mAP": 20.0 + seed,
        "final_cF1": 3.0 + seed,
        "final_oF1": 5.0 + seed,
        "average_mAP": 30.0 + seed,
        "forgetting": 8.0 - seed,
        "parameter_growth": 76608,
    }
    (seed_root / "run_manifest.json").write_text(json.dumps(manifest))
    (seed_root / "config_resolved.json").write_text(json.dumps(config))
    (seed_root / "metrics" / "summary.json").write_text(
        json.dumps({"main_table": main_table})
    )


class CSCFormalResultValidationTest(unittest.TestCase):
    def test_validates_and_aggregates_three_locked_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (0, 1, 2):
                write_seed(root, seed)
            payload = MODULE.validate_csc_formal_results(
                root, "formal_run", (0, 1, 2), COMMIT
            )
            self.assertEqual(payload["status"], "eligible_for_main_table")
            self.assertEqual(payload["aggregate"]["final_mAP"]["mean"], 21.0)
            self.assertAlmostEqual(
                payload["aggregate"]["final_mAP"]["std"],
                0.816496580927726,
            )
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
                MODULE.validate_csc_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )

    def test_rejects_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0)
            write_seed(root, 1, commit="b" * 40)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "git_commit"):
                MODULE.validate_csc_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )


if __name__ == "__main__":
    unittest.main()
