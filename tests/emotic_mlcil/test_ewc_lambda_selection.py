import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "select_ewc_lambda.py"
)
SPEC = importlib.util.spec_from_file_location("select_ewc_lambda", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
VALIDATE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "validate_ewc_formal_selection.py"
)
VALIDATE_SPEC = importlib.util.spec_from_file_location(
    "validate_ewc_formal_selection",
    VALIDATE_SCRIPT,
)
VALIDATE_MODULE = importlib.util.module_from_spec(VALIDATE_SPEC)
assert VALIDATE_SPEC.loader is not None
VALIDATE_SPEC.loader.exec_module(VALIDATE_MODULE)


class EWCLambdaSelectionTest(unittest.TestCase):
    def write_candidate(
        self,
        root,
        value,
        final_map,
        *,
        reporting_split="val",
    ):
        slug = MODULE.lambda_slug(value)
        run_root = (
            Path(root)
            / f"lambda_{slug}"
            / "benchmarks"
            / "protocol"
            / "A"
            / "EWC"
            / "seed0"
        )
        (run_root / "metrics").mkdir(parents=True)
        (run_root / "metrics" / "summary.json").write_text(
            json.dumps(
                {
                    "seed": 0,
                    "main_table": {
                        "final_mAP": final_map,
                        "average_mAP": final_map - 1.0,
                        "forgetting": 2.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "seed": 0,
                    "method": "EWC",
                    "reporting_split": reporting_split,
                    "configuration_locked": False,
                    "eligible_for_main_table": False,
                    "test_labels_used_for_selection": False,
                    "method_configuration": {"ewc_lambda": value},
                    "git_commit": "commit",
                    "source_tree_hash": "source",
                    "core_runtime_version": "runtime",
                    "protocol_hash": "protocol",
                    "class_order_hash": "classes",
                    "data_split_hash": {"task0": "val-split"},
                }
            ),
            encoding="utf-8",
        )
        (run_root / "train.log").write_text(
            "task=1 training="
            + json.dumps({"ewc_to_classification_ratio": value / 1.0e7})
            + "\n",
            encoding="utf-8",
        )

    def test_selects_highest_final_validation_map_and_breaks_ties_low(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_candidate(directory, 1.0e4, 30.0)
            self.write_candidate(directory, 1.0e5, 32.0)
            self.write_candidate(directory, 1.0e6, 32.0)
            payload = MODULE.select_lambda(
                Path(directory),
                [1.0e4, 1.0e5, 1.0e6],
                0,
            )
        self.assertEqual(payload["selected_ewc_lambda"], 1.0e5)
        self.assertEqual(payload["selection_split"], "val")
        self.assertFalse(payload["test_metrics_used"])

    def test_rejects_test_artifacts_for_hyperparameter_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_candidate(
                directory,
                1.0e4,
                30.0,
                reporting_split="test",
            )
            with self.assertRaisesRegex(ValueError, "reporting_split=val"):
                MODULE.select_lambda(Path(directory), [1.0e4], 0)

    def write_formal_manifest(self, root, seed, value):
        run_root = (
            Path(root)
            / "benchmarks"
            / "protocol"
            / "A"
            / "EWC"
            / f"seed{seed}"
        )
        run_root.mkdir(parents=True)
        (run_root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "method": "EWC",
                    "seed": seed,
                    "reporting_split": "test",
                    "configuration_locked": True,
                    "eligible_for_main_table": True,
                    "test_labels_used_for_selection": False,
                    "prediction_reused_task_ids": [],
                    "method_configuration": {"ewc_lambda": value},
                    "git_commit": "commit",
                    "source_tree_hash": "source",
                    "core_runtime_version": "runtime",
                    "protocol_hash": "protocol" if seed == 0 else f"protocol-{seed}",
                    "class_order_hash": "classes",
                    "data_split_hash": {"task0": "test-split"},
                }
            ),
            encoding="utf-8",
        )

    def test_formal_runs_must_match_locked_lambda_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            tuning = Path(directory) / "tuning"
            self.write_candidate(tuning, 1.0e5, 32.0)
            selection = MODULE.select_lambda(tuning, [1.0e5], 0)
            selection_path = Path(directory) / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            formal = Path(directory) / "formal"
            for seed in (0, 1, 2):
                self.write_formal_manifest(formal, seed, 1.0e5)
            payload = VALIDATE_MODULE.validate_formal_runs(
                selection_path,
                formal,
                [0, 1, 2],
            )
            self.assertTrue(payload["all_eligible_for_main_table"])

            manifest_path = (
                formal
                / "benchmarks"
                / "protocol"
                / "A"
                / "EWC"
                / "seed2"
                / "run_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["method_configuration"]["ewc_lambda"] = 1.0e6
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "locked"):
                VALIDATE_MODULE.validate_formal_runs(
                    selection_path,
                    formal,
                    [0, 1, 2],
                )


if __name__ == "__main__":
    unittest.main()
