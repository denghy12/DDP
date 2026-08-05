"""Tests for the eight-slot B4-C2 sweep scheduler."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEDULER = ROOT / "scripts" / "emotic-mlcil" / "schedule_b4c2_12baseline_8gpu.py"
SPEC = importlib.util.spec_from_file_location("schedule_b4c2_12baseline_8gpu", SCHEDULER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class B4C2SchedulerTest(unittest.TestCase):
    def test_all_36_jobs_complete_on_eight_distinct_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            fake_job = Path(temporary) / "fake_job.sh"
            fake_job.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCHEDULER),
                    "--run-root",
                    str(run_root),
                    "--run-id",
                    "scheduler-test",
                    "--job-script",
                    str(fake_job),
                    "--gpus",
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "--poll-seconds",
                    "0.01",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            states = run_root / "runtime_state"
            self.assertEqual(len(list(states.glob("*.done.json"))), 36)
            self.assertEqual(list(states.glob("*.failed.json")), [])
            plan = json.loads((run_root / "sweep_plan.json").read_text())
            self.assertEqual(len(plan["jobs"]), 36)
            self.assertEqual(plan["slots_per_gpu"], 2)
            self.assertEqual(len(plan["slots"]), 16)
            self.assertEqual(
                sorted(slot["physical_gpu"] for slot in plan["slots"]),
                sorted(list(range(8)) * 2),
            )
            self.assertEqual(plan["maximum_concurrent_training_processes"], 16)
            self.assertTrue(plan["gpu_sharing"])
            self.assertEqual(
                plan["memory_safety_basis"]["maximum_processes_per_gpu"], 2
            )
            self.assertEqual(
                plan["memory_safety_basis"]["gpu_memory_budget_mib"], 20000
            )
            self.assertIn(
                ["derpp", "er"],
                plan["memory_safety_basis"]["incompatible_method_pairs"],
            )
            active = {gpu: [] for gpu in range(8)}
            events = [
                json.loads(line)
                for line in (run_root / "scheduler_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for event in events:
                if event["event"] == "job_started":
                    methods = active[event["physical_gpu"]]
                    methods.append(event["method"])
                    self.assertLessEqual(len(methods), 2)
                    self.assertTrue(MODULE.methods_can_share_gpu(methods))
                elif event["event"] in {"job_done", "job_failed"}:
                    active[event["physical_gpu"]].remove(event["method"])

    def test_known_oom_pairs_and_over_budget_pairs_are_rejected(self) -> None:
        self.assertFalse(MODULE.methods_can_share_gpu(["derpp", "er"]))
        self.assertFalse(MODULE.methods_can_share_gpu(["derpp", "prs"]))
        self.assertFalse(MODULE.methods_can_share_gpu(["derpp", "lwf"]))
        self.assertTrue(MODULE.methods_can_share_gpu(["derpp", "original_ddp"]))
        self.assertTrue(MODULE.methods_can_share_gpu(["er", "prs"]))
        self.assertFalse(MODULE.methods_can_share_gpu(["agcn", "l3a", "ewc"]))


if __name__ == "__main__":
    unittest.main()
