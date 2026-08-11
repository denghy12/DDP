import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "emotic-mlcil" / "validate_emot_net_ft_formal_result.py"
SPEC = importlib.util.spec_from_file_location("validate_emot_net_ft_formal_result", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
COMMIT = "1" * 40


def write_formal_bundle(root: Path, *, configuration_locked: bool = True) -> None:
    seed_root = (
        root
        / "benchmarks"
        / "emotic_b5c3_track_b_v0.1"
        / "B"
        / "EMOT-Net-FT"
        / "seed0"
    )
    (seed_root / "metrics").mkdir(parents=True)
    (seed_root / "scores").mkdir()
    manifest = {
        "method": "EMOT-Net-FT",
        "method_family": "Static EMOTIC model / Sequential Fine-Tuning",
        "protocol_id": "emotic_b5c3_track_b_v0.1",
        "track": "B",
        "seed": 0,
        "git_commit": COMMIT,
        "git_dirty": False,
        "core_runtime_version": "0.10.0",
        "test_labels_used_for_selection": False,
        "checkpoint_selection_split": "val",
        "reporting_split": "test",
        "configuration_locked": configuration_locked,
        "prediction_reused_task_ids": [],
        "eligible_for_main_table": configuration_locked,
        "replay_memory_samples": 0,
        "replay_memory_bytes": 0,
        "protocol_hash": "protocol",
        "class_order_hash": "classes",
        **MODULE.EXPECTED_PARAMETER_STATISTICS,
        "method_configuration": dict(MODULE.LOCKED_METHOD_CONFIGURATION),
    }
    config = {"runner": dict(MODULE.LOCKED_RUNNER_CONFIGURATION)}
    config["runner"]["configuration_locked"] = configuration_locked
    tasks = []
    for task_id in range(8):
        tasks.append(
            {
                "task_id": task_id,
                "mAP": 30.0 - task_id,
                "cF1": 20.0 - task_id,
                "oF1": 50.0 - task_id,
            }
        )
        (seed_root / "scores" / f"task{task_id}_scores.pt").write_bytes(b"score")
    main_table = {
        "method": "EMOT-Net-FT",
        "final_mAP": 23.0,
        "final_cF1": 13.0,
        "final_oF1": 43.0,
        "average_mAP": 26.5,
        "forgetting": 6.0,
        "parameter_growth": 5397,
    }
    (seed_root / "run_manifest.json").write_text(json.dumps(manifest))
    (seed_root / "config_resolved.json").write_text(json.dumps(config))
    (seed_root / "metrics" / "summary.json").write_text(
        json.dumps({"main_table": main_table})
    )
    (seed_root / "metrics" / "task_metrics.json").write_text(
        json.dumps({"tasks": tasks})
    )
    records = []
    for task_id, steps in enumerate(MODULE.EXPECTED_STEPS_PER_EPOCH):
        for epoch in range(21):
            payload = {
                "epoch": float(epoch),
                "optimizer_steps": float(steps),
                "skipped_optimizer_steps": 0.0,
                "weighted_classification_loss": 0.1,
                "weighted_sigmoid_mse": 0.6,
                "validation_current_mAP": 30.0 + epoch / 100.0,
            }
            records.append(f"task={task_id} training={json.dumps(payload)}")
    (seed_root / "train.log").write_text("\n".join(records) + "\n")


class EMOTNetFTFormalResultTest(unittest.TestCase):
    def test_accepts_one_locked_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_formal_bundle(root)
            result = MODULE.validate_emot_net_ft_formal_result(
                root, "formal_seed0", COMMIT
            )
            self.assertEqual(result["formal_seeds"], [0])
            self.assertEqual(
                result["aggregate_statistics"], "not_applicable_single_seed"
            )
            self.assertTrue(result["eligibility"]["eligible_for_main_table"])
            self.assertEqual(result["training_stability"]["optimizer_updates"], 12012)

    def test_rejects_unlocked_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_formal_bundle(root, configuration_locked=False)
            with self.assertRaisesRegex(ValueError, "configuration_locked"):
                MODULE.validate_emot_net_ft_formal_result(
                    root, "formal_seed0", COMMIT
                )

    def test_formal_entrypoint_is_seed0_only(self):
        launcher = (
            ROOT
            / "scripts"
            / "emotic-mlcil"
            / "launch_emot_net_ft_formal_seed0_tmux.sh"
        ).read_text()
        worker = (
            ROOT
            / "scripts"
            / "emotic-mlcil"
            / "run_emot_net_ft_formal_seed0.sh"
        ).read_text()
        self.assertIn("REPORTING_SPLIT=test", worker)
        self.assertIn("SEED=0", worker)
        self.assertIn("--expected-bundles 1", worker)
        self.assertNotIn("for seed in", worker)
        self.assertIn("EXPECTED_GIT_COMMIT", launcher)
        self.assertIn("EMOT_NET_FT_TRACK_B_V0_1", launcher)

    def test_registered_formal_snapshot_is_single_seed_test_result(self):
        path = (
            ROOT
            / "docs"
            / "benchmarks"
            / "results"
            / "emot_net_ft_seed0_formal_v0.1.json"
        )
        payload = json.loads(path.read_text())
        self.assertEqual(payload["protocol_id"], "emotic_b5c3_track_b_v0.1")
        self.assertEqual(payload["track"], "B")
        self.assertEqual(payload["reporting_scope"]["formal_seeds"], [0])
        self.assertEqual(
            payload["reporting_scope"]["aggregate_statistics"],
            "not_applicable_single_seed",
        )
        self.assertTrue(payload["eligibility"]["eligible_for_main_table"])
        self.assertEqual(payload["eligibility"]["reporting_split"], "test")
        self.assertEqual(payload["training_stability"]["optimizer_updates"], 12012)
        self.assertEqual(payload["test_score_audit"]["test_prefixed_ids"], 5368)
        self.assertEqual(payload["test_score_audit"]["validation_prefixed_ids"], 0)
        self.assertFalse(payload["artifacts"]["contains_pth"])
        self.assertEqual(len(payload["per_task"]), 8)
        self.assertEqual(len(payload["per_class_forgetting"]["rows"]), 26)


if __name__ == "__main__":
    unittest.main()
