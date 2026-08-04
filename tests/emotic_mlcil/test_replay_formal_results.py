"""Tests for locked ER/PRS formal-result validation."""

from __future__ import annotations

import importlib.util
import json
import statistics
import tempfile
import unittest
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "validate_replay_formal_results.py"
)
_SPEC = importlib.util.spec_from_file_location("validate_replay_formal_results", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class ReplayFormalResultTest(unittest.TestCase):
    commit = "a" * 40

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _build_seed(self, root: Path, method: str, seed: int) -> Path:
        target = (
            root
            / "benchmarks"
            / "emotic_b5c3_v0.1"
            / "A"
            / method
            / f"seed{seed}"
        )
        method_specific = _MODULE.METHOD_CONFIGURATION[method]
        method_config = {
            **_MODULE.COMMON_METHOD_CONFIGURATION,
            **method_specific,
            "replay_contract_path": (
                "/server/repo/configs/emotic_mlcil/replay_20c_v0.1.yaml"
            ),
        }
        manifest = {
            "method": method,
            "protocol_id": "emotic_b5c3_v0.1",
            "track": "A",
            "seed": seed,
            "git_commit": self.commit,
            "git_dirty": False,
            "source_tree_hash": "tree-hash",
            "core_base_commit": "core-base",
            "core_runtime_version": "0.9.0",
            "protocol_hash": f"protocol-{seed}",
            "class_order_hash": "class-order",
            "data_split_hash": {"task0": "same-split"},
            "test_labels_used_for_selection": False,
            "reporting_split": "test",
            "configuration_locked": True,
            "prediction_reused_task_ids": [],
            "eligible_for_main_table": True,
            "replay_memory_samples": 520,
            "replay_memory_bytes": 313000000 + seed,
            **_MODULE.EXPECTED_PARAMETERS,
            "method_configuration": method_config,
        }
        self._write_json(target / "run_manifest.json", manifest)
        self._write_json(
            target / "config_resolved.json",
            {
                "runner": dict(_MODULE.LOCKED_RUNNER_CONFIGURATION),
                "protocol": {
                    **_MODULE.LOCKED_PROTOCOL_CONFIGURATION,
                    "seed": seed,
                    "protocol_hash": f"protocol-{seed}",
                },
            },
        )
        base = (1 if method == "ER" else 4) + seed
        main = {
            name: float(base + index)
            for index, name in enumerate(_MODULE.AGGREGATE_METRICS)
        }
        main["replay_memory"] = {
            "samples": 520,
            "bytes": 313000000 + seed,
        }
        self._write_json(target / "metrics" / "summary.json", {"main_table": main})
        self._write_json(
            target / "metrics" / "task_metrics.json",
            {
                "tasks": [
                    {"task_id": task, "mAP": float(base + task)}
                    for task in range(8)
                ]
            },
        )
        scores = target / "scores"
        scores.mkdir(parents=True, exist_ok=True)
        for task in range(8):
            (scores / f"task{task}_scores.pt").write_bytes(b"scores")
        log_rows = []
        for task in range(8):
            final = {
                "epoch": 0.0,
                "current_loss": 0.5,
                "replay_loss": 0.0 if task == 0 else 0.3,
                "optimizer_attempts": 10.0,
                "amp_overflow_skips": float(seed == 2 and task == 7),
                "memory_update_observations": float(
                    _MODULE.EXPECTED_TRAINING_SAMPLES[task]
                ),
                "replay_samples_after": float(_MODULE.EXPECTED_CAPACITIES[task]),
                "replay_bytes_after": float(60000000 + task),
            }
            log_rows.append(f"task={task} training={json.dumps(final)}")
        (target / "train.log").write_text("\n".join(log_rows) + "\n", encoding="utf-8")
        return target

    def _build_run(self, root: Path) -> None:
        for method in _MODULE.EXPECTED_METHODS:
            for seed in _MODULE.EXPECTED_SEEDS:
                self._build_seed(root, method, seed)

    def test_six_bundles_aggregate_with_sample_standard_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_run(root)
            payload = _MODULE.validate_replay_formal_results(
                root, "formal", (0, 1, 2), self.commit
            )
            self.assertEqual(payload["status"], "eligible_for_main_table")
            self.assertEqual(payload["execution"]["max_concurrent_training_processes"], 3)
            er_values = [1.0, 2.0, 3.0]
            self.assertAlmostEqual(
                payload["results"]["ER"]["aggregate"]["final_mAP"]["std"],
                statistics.stdev(er_values),
            )
            self.assertEqual(len(payload["results"]["PRS"]["per_seed"]), 3)

    def test_missing_method_seed_blocks_formal_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_run(root)
            missing = root / "benchmarks" / "emotic_b5c3_v0.1" / "A" / "PRS" / "seed2"
            for path in sorted(missing.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                else:
                    path.rmdir()
            missing.rmdir()
            with self.assertRaisesRegex(RuntimeError, "Expected one PRS seed 2"):
                _MODULE.validate_replay_formal_results(
                    root, "formal", (0, 1, 2), self.commit
                )

    def test_unlocked_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_run(root)
            path = (
                root
                / "benchmarks"
                / "emotic_b5c3_v0.1"
                / "A"
                / "ER"
                / "seed0"
                / "run_manifest.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["configuration_locked"] = False
            self._write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "configuration_locked"):
                _MODULE.validate_replay_formal_results(
                    root, "formal", (0, 1, 2), self.commit
                )


if __name__ == "__main__":
    unittest.main()
