#!/usr/bin/env python3
"""Run the B10-C4 12-baseline sweep with dynamic one-process-per-GPU slots."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, IO, List, Mapping, Sequence


# Longest-processing-time first reduces the final idle-GPU tail. Every method
# still uses exactly its frozen B5-C3 optimizer, epoch, batch, and memory setup.
METHOD_PRIORITY = (
    "original_ddp",
    "krt",
    "multi_lane",
    "csc",
    "lwf",
    "derpp",
    "ewc",
    "er",
    "prs",
    "finetune",
    "agcn",
    "l3a",
)
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Job:
    method: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.method}_seed{self.seed}"


@dataclass
class ActiveJob:
    job: Job
    gpu: int
    process: subprocess.Popen[bytes]
    log_stream: IO[bytes]
    started_monotonic: float
    log_path: Path


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _event(stream: IO[str], event: str, **payload: object) -> None:
    record = {"time_unix": time.time(), "event": event, **payload}
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-script", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8:
        raise ValueError("B10-C4 sweep requires exactly eight distinct GPUs")
    if min(args.gpus) < 0:
        raise ValueError("GPU indices must be non-negative")
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if not args.job_script.is_file():
        raise FileNotFoundError(f"Missing job script: {args.job_script}")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    run_root = args.run_root.resolve()
    state_dir = run_root / "runtime_state"
    log_dir = run_root / "job_logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [Job(method, seed) for method in METHOD_PRIORITY for seed in SEEDS]
    plan = {
        "schema_version": 1,
        "run_id": args.run_id,
        "protocol_id": "emotic_b10c4_v0.1",
        "methods": list(METHOD_PRIORITY),
        "seeds": list(SEEDS),
        "physical_gpus": args.gpus,
        "scheduling": "longest_processing_time_first_dynamic_backfill",
        "maximum_concurrent_training_processes": len(args.gpus),
        "gpu_sharing": False,
        "jobs": [job.key for job in jobs],
    }
    _write_json_atomic(run_root / "sweep_plan.json", plan)

    collisions = [
        path
        for job in jobs
        for path in (
            state_dir / f"{job.key}.started.json",
            state_dir / f"{job.key}.done.json",
            state_dir / f"{job.key}.failed.json",
        )
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "Run state already exists; use a new RUN_ID: "
            + ", ".join(str(path) for path in collisions[:3])
        )

    pending: List[Job] = list(jobs)
    free_gpus: List[int] = list(args.gpus)
    active: Dict[int, ActiveJob] = {}
    failures: List[str] = []
    interrupted = False

    def stop_children(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        print(f"Received signal {signum}; terminating active jobs", flush=True)
        for record in active.values():
            try:
                os.killpg(record.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    event_path = run_root / "scheduler_events.jsonl"
    with event_path.open("a", encoding="utf-8") as events:
        _event(events, "scheduler_started", jobs=len(jobs), gpus=args.gpus)
        while pending or active:
            while pending and free_gpus and not interrupted:
                job = pending.pop(0)
                gpu = free_gpus.pop(0)
                log_path = log_dir / f"{job.key}_gpu{gpu}.log"
                log_stream = log_path.open("wb")
                env = dict(os.environ)
                env.update(
                    {
                        "METHOD": job.method,
                        "SEED": str(job.seed),
                        "PHYSICAL_GPU": str(gpu),
                        "RUN_ID": args.run_id,
                        "RUN_OUTPUT_ROOT": str(run_root),
                    }
                )
                process = subprocess.Popen(
                    ["bash", str(args.job_script)],
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                started = {
                    "job": job.key,
                    "method": job.method,
                    "seed": job.seed,
                    "physical_gpu": gpu,
                    "pid": process.pid,
                    "log": str(log_path),
                    "started_unix": time.time(),
                }
                _write_json_atomic(state_dir / f"{job.key}.started.json", started)
                active[gpu] = ActiveJob(
                    job=job,
                    gpu=gpu,
                    process=process,
                    log_stream=log_stream,
                    started_monotonic=time.monotonic(),
                    log_path=log_path,
                )
                _event(events, "job_started", **started)
                print(
                    f"START {job.key:28s} GPU={gpu} "
                    f"active={len(active)}/8 pending={len(pending)}",
                    flush=True,
                )

            if interrupted and not active:
                break
            completed_gpus = [
                gpu
                for gpu, record in active.items()
                if record.process.poll() is not None
            ]
            if not completed_gpus:
                time.sleep(args.poll_seconds)
                continue
            for gpu in completed_gpus:
                record = active.pop(gpu)
                return_code = int(record.process.returncode)
                duration = time.monotonic() - record.started_monotonic
                record.log_stream.close()
                started_path = state_dir / f"{record.job.key}.started.json"
                if started_path.exists():
                    started_path.unlink()
                result = {
                    "job": record.job.key,
                    "method": record.job.method,
                    "seed": record.job.seed,
                    "physical_gpu": gpu,
                    "exit_code": return_code,
                    "duration_seconds": duration,
                    "log": str(record.log_path),
                    "completed_unix": time.time(),
                }
                suffix = "done" if return_code == 0 else "failed"
                _write_json_atomic(
                    state_dir / f"{record.job.key}.{suffix}.json",
                    result,
                )
                _event(events, f"job_{suffix}", **result)
                if return_code != 0:
                    failures.append(record.job.key)
                free_gpus.append(gpu)
                free_gpus.sort(key=args.gpus.index)
                print(
                    f"{suffix.upper():5s} {record.job.key:28s} GPU={gpu} "
                    f"seconds={duration:.1f} active={len(active)}/8 "
                    f"pending={len(pending)}",
                    flush=True,
                )

        if interrupted:
            _event(events, "scheduler_interrupted", failures=failures)
            raise SystemExit(130)
        _event(events, "scheduler_completed", failures=failures)

    if failures:
        raise RuntimeError(
            "B10-C4 jobs failed; aggregation is blocked: " + ", ".join(failures)
        )
    print("All 36 B10-C4 baseline jobs completed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"B10-C4 scheduler failed: {exc}", file=sys.stderr, flush=True)
        raise
