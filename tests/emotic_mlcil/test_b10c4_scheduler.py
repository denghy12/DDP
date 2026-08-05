"""Tests for the eight-slot B10-C4 sweep scheduler."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEDULER = ROOT / "scripts" / "emotic-mlcil" / "schedule_b10c4_12baseline_8gpu.py"


class B10C4SchedulerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
