import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "validate_agcn_formal_results.py"
)
LAUNCHER = SCRIPT.with_name("launch_agcn_formal_seed012_tmux.sh")
WORKER = SCRIPT.with_name("run_agcn_formal_seed012.sh")
SPEC = importlib.util.spec_from_file_location("validate_agcn_formal_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


COMMIT = "a" * 40


def write_seed(
    root,
    seed,
    *,
    locked=True,
    embedding_sha=None,
    eval_batch_size=32,
    complete_log=True,
):
    seed_root = (
        root
        / "benchmarks"
        / "emotic_b5c3_v0.1"
        / "A"
        / "AGCN"
        / f"seed{seed}"
    )
    (seed_root / "metrics").mkdir(parents=True)
    (seed_root / "scores").mkdir()
    method_configuration = copy.deepcopy(MODULE.LOCKED_METHOD_CONFIGURATION)
    asset = copy.deepcopy(MODULE.LOCKED_EMBEDDING_ASSET)
    asset["path"] = "/fixed/emotic_glove_6b_300d.json"
    if embedding_sha is not None:
        asset["sha256"] = embedding_sha
    method_configuration["class_embedding_asset"] = asset
    protocol_hash = f"protocol-{seed}"
    manifest = {
        "method": "AGCN",
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
        "core_runtime_version": "0.8.0",
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
    runner["eval_batch_size"] = eval_batch_size
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
        "final_mAP": 25.0 + seed,
        "final_cF1": 16.0 + seed,
        "final_oF1": 55.0 + seed,
        "average_mAP": 29.0 + seed,
        "forgetting": 2.0 - seed * 0.5,
        "parameter_growth": 0,
    }
    task_metrics = [
        {"task_id": task, "mAP": 32.0 - task + seed}
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
    for task in range(8):
        (seed_root / "scores" / f"task{task}_scores.pt").write_text("scores")

    lines = []
    for task in range(8):
        if not complete_log and task == 7:
            continue
        old_value = 0.0 if task == 0 else 10.0 + task
        row = {
            "acm_cross_soft_hard_mass": old_value,
            "acm_current_positive_mass": 20.0 + task,
            "acm_old_soft_mass": old_value,
            "acm_samples": float(MODULE.EXPECTED_TRAINING_SAMPLES[task]),
            "adjacency_density": 1.0 / (task + 1),
            "adjacency_nonzero": float(25 + task),
            "current_loss": 0.2 + task * 0.01,
            "distillation_loss": 0.0 if task == 0 else 0.3,
            "epoch": 0.0,
            "optimizer_steps": float(MODULE.EXPECTED_OPTIMIZER_STEPS[task]),
            "relationship_loss": 0.0 if task == 0 else 1.0e-6,
            "skipped_optimizer_steps": 0.0,
            "total_loss": 0.4 + task * 0.01,
            "validation_current_mAP": 40.0 + task,
        }
        lines.append(f"task={task} training=" + json.dumps(row))
    (seed_root / "train.log").write_text("\n".join(lines) + "\n")


class AGCNFormalResultValidationTest(unittest.TestCase):
    def test_formal_launcher_packs_three_seeds_on_gpu_zero(self):
        launcher = LAUNCHER.read_text()
        worker = WORKER.read_text()
        self.assertIn('GPU="${GPU:-0}"', launcher)
        self.assertIn('GPUS="${GPUS:-${GPU} ${GPU} ${GPU}}"', launcher)
        self.assertIn('MIN_FREE_MIB="${AGCN_MIN_FREE_MIB:-12000}"', launcher)
        self.assertIn('GPUS="${GPUS:-0 0 0}"', worker)

    def test_validates_and_aggregates_three_locked_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (0, 1, 2):
                write_seed(root, seed)
            payload = MODULE.validate_agcn_formal_results(
                root, "formal_run", (0, 1, 2), COMMIT
            )
            self.assertEqual(payload["status"], "eligible_for_main_table")
            self.assertEqual(payload["aggregate"]["final_mAP"]["mean"], 26.0)
            self.assertEqual(payload["aggregate"]["final_mAP"]["std"], 1.0)
            self.assertEqual(
                payload["aggregate"]["aggregation"],
                "mean_and_sample_standard_deviation",
            )
            self.assertEqual(
                payload["training_stability"][0]["optimizer_updates"], 3701
            )
            self.assertEqual(
                payload["training_stability"][0]["skipped_optimizer_updates"], 0
            )

    def test_rejects_unlocked_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0)
            write_seed(root, 1, locked=False)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "configuration_locked"):
                MODULE.validate_agcn_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )

    def test_rejects_embedding_asset_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0, embedding_sha="b" * 64)
            write_seed(root, 1)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "embedding asset.sha256"):
                MODULE.validate_agcn_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )

    def test_rejects_runner_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0, eval_batch_size=16)
            write_seed(root, 1)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "eval_batch_size"):
                MODULE.validate_agcn_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )

    def test_rejects_incomplete_training_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_seed(root, 0, complete_log=False)
            write_seed(root, 1)
            write_seed(root, 2)
            with self.assertRaisesRegex(ValueError, "training task IDs"):
                MODULE.validate_agcn_formal_results(
                    root, "formal_run", (0, 1, 2), COMMIT
                )


if __name__ == "__main__":
    unittest.main()
