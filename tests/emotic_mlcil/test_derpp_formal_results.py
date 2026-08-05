"""Tests for locked DER++ formal-result validation."""

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
    / "validate_derpp_formal_results.py"
)
_SPEC = importlib.util.spec_from_file_location("validate_derpp_formal_results", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class DERPPFormalResultTest(unittest.TestCase):
    commit = "a" * 40

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _build_seed(self, root: Path, seed: int, gpu: int) -> Path:
        target = (
            root
            / "benchmarks"
            / "emotic_b5c3_v0.1"
            / "A"
            / "DER++"
            / f"seed{seed}"
        )
        method_config = {
            **_MODULE.LOCKED_METHOD_CONFIGURATION,
            "replay_contract_path": (
                "/server/repo/configs/emotic_mlcil/replay_derpp_20c_v0.1.yaml"
            ),
        }
        replay_bytes = 313273870 + seed
        manifest = {
            "method": "DER++",
            "protocol_id": "emotic_b5c3_v0.1",
            "track": "A",
            "seed": seed,
            "git_commit": self.commit,
            "git_dirty": False,
            "source_tree_hash": "tree-hash",
            "core_base_commit": "core-base",
            "core_runtime_version": "0.10.0",
            "protocol_hash": f"protocol-{seed}",
            "class_order_hash": "class-order",
            "data_split_hash": {"task0": "same-split"},
            "test_labels_used_for_selection": False,
            "checkpoint_selection_split": "val",
            "reporting_split": "test",
            "configuration_locked": True,
            "prediction_reused_task_ids": [],
            "eligible_for_main_table": True,
            "replay_memory_samples": 520,
            "replay_memory_bytes": replay_bytes,
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
        base = 10 + seed
        main = {
            name: float(base + index)
            for index, name in enumerate(_MODULE.AGGREGATE_METRICS)
        }
        main["replay_memory"] = {"samples": 520, "bytes": replay_bytes}
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
            capacity = _MODULE.EXPECTED_CAPACITIES[task]
            selected_bytes = replay_bytes if task == 7 else 60000000 + task
            row = {
                "epoch": 0.0,
                "current_loss": 0.5,
                "dark_logit_loss": 0.1,
                "replay_label_loss": 0.2,
                "replay_examples": float(
                    2 * _MODULE.EXPECTED_TRAINING_SAMPLES[task]
                ),
                "memory_update_observations": float(
                    _MODULE.EXPECTED_TRAINING_SAMPLES[task]
                ),
                "selected_epoch": 0.0,
                "optimizer_attempts": 10.0,
                "amp_overflow_skips": float(seed == 2 and task == 7),
                "selected_replay_samples": float(capacity),
                "selected_replay_bytes": float(selected_bytes),
            }
            log_rows.append(f"task={task} training={json.dumps(row)}")
        (target / "train.log").write_text("\n".join(log_rows) + "\n", encoding="utf-8")

        marker = root / "runtime_logs" / f"seed{seed}.done"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            "\n".join(
                (
                    f"seed={seed}",
                    f"physical_gpu={gpu}",
                    "phase=three_seed_three_gpu_parallel",
                    "exit_code=0",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    def _build_run(self, root: Path) -> None:
        for seed, gpu in enumerate((0, 1, 2)):
            self._build_seed(root, seed, gpu)

    def test_three_gpu_bundles_use_sample_standard_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_run(root)
            payload = _MODULE.validate_derpp_formal_results(
                root, "formal", (0, 1, 2), (0, 1, 2), self.commit
            )
            self.assertEqual(payload["status"], "eligible_for_main_table")
            self.assertEqual(
                payload["execution"]["mode"], "three_seed_three_gpu_parallel"
            )
            self.assertAlmostEqual(
                payload["aggregate"]["final_mAP"]["std"],
                statistics.stdev([10.0, 11.0, 12.0]),
            )
            self.assertEqual(len(payload["per_seed"]), 3)

    def test_unlocked_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_run(root)
            path = (
                root
                / "benchmarks"
                / "emotic_b5c3_v0.1"
                / "A"
                / "DER++"
                / "seed0"
                / "run_manifest.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["configuration_locked"] = False
            self._write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "configuration_locked"):
                _MODULE.validate_derpp_formal_results(
                    root, "formal", (0, 1, 2), (0, 1, 2), self.commit
                )

    def test_reused_gpu_assignment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_run(root)
            with self.assertRaisesRegex(ValueError, "three distinct GPUs"):
                _MODULE.validate_derpp_formal_results(
                    root, "formal", (0, 1, 2), (0, 0, 1), self.commit
                )


if __name__ == "__main__":
    unittest.main()
