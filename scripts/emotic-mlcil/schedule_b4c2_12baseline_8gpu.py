#!/usr/bin/env python3
"""Run the B4-C2 12-baseline sweep with bounded multi-slot GPUs."""

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
from typing import Dict, FrozenSet, IO, List, Mapping, Optional, Sequence, Tuple


# Longest-processing-time first reduces the final idle-GPU tail. Every method
# still uses exactly its frozen B5-C3 optimizer, epoch, batch, and memory setup.
METHOD_PRIORITY = (
    "original_ddp",
    "krt",
    "derpp",
    "csc",
    "multi_lane",
    "prs",
    "ewc",
    "lwf",
    "er",
    "finetune",
    "agcn",
    "l3a",
)
SEEDS = (0, 1, 2)

# Reservations deliberately exceed isolated smoke peaks. The scheduler admits
# at most two processes per GPU and keeps the sum below 20 GiB, leaving more
# than 4 GiB of headroom on a 24 GiB RTX 4090 for allocator/dataloader overlap.
# B10-C4 proved that DER++ plus ER/PRS can OOM despite apparently sufficient
# instantaneous free memory, so those pairs are also explicitly forbidden.
DEFAULT_GPU_MEMORY_BUDGET_MIB = 20000
METHOD_MEMORY_RESERVATION_MIB = {
    "original_ddp": 6000,
    "krt": 12000,
    "multi_lane": 3500,
    "csc": 9000,
    "lwf": 6500,
    "derpp": 14000,
    "ewc": 8000,
    "er": 9500,
    "prs": 9500,
    "finetune": 6500,
    "agcn": 4500,
    "l3a": 8000,
}
INCOMPATIBLE_METHOD_PAIRS: FrozenSet[FrozenSet[str]] = frozenset(
    {
        frozenset(("derpp", "er")),
        frozenset(("derpp", "prs")),
    }
)


@dataclass(frozen=True)
class Job:
    method: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.method}_seed{self.seed}"

    @property
    def memory_reservation_mib(self) -> int:
        return METHOD_MEMORY_RESERVATION_MIB[self.method]


@dataclass
class ActiveJob:
    job: Job
    slot: int
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
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument(
        "--gpu-memory-budget-mib",
        type=int,
        default=DEFAULT_GPU_MEMORY_BUDGET_MIB,
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8:
        raise ValueError("B4-C2 sweep requires exactly eight distinct GPUs")
    if min(args.gpus) < 0:
        raise ValueError("GPU indices must be non-negative")
    if args.slots_per_gpu not in {1, 2}:
        raise ValueError("slots-per-gpu must be one or two")
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if args.gpu_memory_budget_mib <= 0:
        raise ValueError("gpu-memory-budget-mib must be positive")
    if max(METHOD_MEMORY_RESERVATION_MIB.values()) > args.gpu_memory_budget_mib:
        raise ValueError("GPU memory budget cannot admit the largest method")
    if not args.job_script.is_file():
        raise FileNotFoundError(f"Missing job script: {args.job_script}")


def methods_can_share_gpu(
    methods: Sequence[str],
    memory_budget_mib: int = DEFAULT_GPU_MEMORY_BUDGET_MIB,
) -> bool:
    """Return whether one/two registered methods fit the guarded GPU budget."""

    if len(methods) > 2 or any(
        method not in METHOD_MEMORY_RESERVATION_MIB for method in methods
    ):
        return False
    for index, first in enumerate(methods):
        for second in methods[index + 1 :]:
            if frozenset((first, second)) in INCOMPATIBLE_METHOD_PAIRS:
                return False
    return sum(METHOD_MEMORY_RESERVATION_MIB[method] for method in methods) <= int(
        memory_budget_mib
    )


def _job_fits_gpu(
    job: Job,
    gpu: int,
    active: Mapping[int, ActiveJob],
    slots_per_gpu: int,
    memory_budget_mib: int,
) -> bool:
    methods = [record.job.method for record in active.values() if record.gpu == gpu]
    if len(methods) >= slots_per_gpu:
        return False
    return methods_can_share_gpu([*methods, job.method], memory_budget_mib)


def _select_launch(
    pending: Sequence[Job],
    free_slots: Sequence[Tuple[int, int]],
    active: Mapping[int, ActiveJob],
    slots_per_gpu: int,
    memory_budget_mib: int,
) -> Optional[Tuple[int, int]]:
    """Select the earliest LPT job and earliest slot that satisfy memory guards."""

    for pending_index, job in enumerate(pending):
        for slot_index, (_slot, gpu) in enumerate(free_slots):
            if _job_fits_gpu(
                job,
                gpu,
                active,
                slots_per_gpu,
                memory_budget_mib,
            ):
                return pending_index, slot_index
    return None


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    run_root = args.run_root.resolve()
    state_dir = run_root / "runtime_state"
    log_dir = run_root / "job_logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [Job(method, seed) for method in METHOD_PRIORITY for seed in SEEDS]
    slot_assignments = [
        (slot, gpu)
        for slot, gpu in enumerate(
            gpu
            for _lane in range(args.slots_per_gpu)
            for gpu in args.gpus
        )
    ]
    slot_rank = {slot: rank for rank, (slot, _gpu) in enumerate(slot_assignments)}
    plan = {
        "schema_version": 1,
        "run_id": args.run_id,
        "protocol_id": "emotic_b4c2_v0.1",
        "methods": list(METHOD_PRIORITY),
        "seeds": list(SEEDS),
        "physical_gpus": args.gpus,
        "scheduling": "memory_guarded_longest_processing_time_first_dynamic_backfill",
        "slots_per_gpu": args.slots_per_gpu,
        "slots": [
            {"scheduler_slot": slot, "physical_gpu": gpu}
            for slot, gpu in slot_assignments
        ],
        "maximum_concurrent_training_processes": len(slot_assignments),
        "gpu_sharing": args.slots_per_gpu > 1,
        "memory_safety_basis": {
            "largest_registered_single_process_peak_mib": 8434.2,
            "maximum_processes_per_gpu": 2,
            "gpu_memory_budget_mib": args.gpu_memory_budget_mib,
            "method_memory_reservation_mib": METHOD_MEMORY_RESERVATION_MIB,
            "incompatible_method_pairs": [
                sorted(pair) for pair in sorted(
                    INCOMPATIBLE_METHOD_PAIRS,
                    key=lambda item: sorted(item),
                )
            ],
            "basis": "B5/B10 smoke peaks plus allocator headroom and B10 retry audit",
        },
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
    free_slots = list(slot_assignments)
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
        _event(
            events,
            "scheduler_started",
            jobs=len(jobs),
            gpus=args.gpus,
            gpu_memory_budget_mib=args.gpu_memory_budget_mib,
        )
        while pending or active:
            while pending and free_slots and not interrupted:
                selection = _select_launch(
                    pending,
                    free_slots,
                    active,
                    args.slots_per_gpu,
                    args.gpu_memory_budget_mib,
                )
                if selection is None:
                    break
                pending_index, slot_index = selection
                job = pending.pop(pending_index)
                slot, gpu = free_slots.pop(slot_index)
                log_path = log_dir / f"{job.key}_gpu{gpu}_slot{slot}.log"
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
                    "scheduler_slot": slot,
                    "memory_reservation_mib": job.memory_reservation_mib,
                    "gpu_memory_budget_mib": args.gpu_memory_budget_mib,
                    "pid": process.pid,
                    "log": str(log_path),
                    "started_unix": time.time(),
                }
                _write_json_atomic(state_dir / f"{job.key}.started.json", started)
                active[slot] = ActiveJob(
                    job=job,
                    slot=slot,
                    gpu=gpu,
                    process=process,
                    log_stream=log_stream,
                    started_monotonic=time.monotonic(),
                    log_path=log_path,
                )
                _event(events, "job_started", **started)
                print(
                    f"START {job.key:28s} GPU={gpu} slot={slot} "
                    f"reserve={job.memory_reservation_mib}MiB "
                    f"active={len(active)}/{len(slot_assignments)} "
                    f"pending={len(pending)}",
                    flush=True,
                )

            if interrupted and not active:
                break
            completed_slots = [
                slot
                for slot, record in active.items()
                if record.process.poll() is not None
            ]
            if not completed_slots:
                time.sleep(args.poll_seconds)
                continue
            for slot in completed_slots:
                record = active.pop(slot)
                gpu = record.gpu
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
                    "scheduler_slot": slot,
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
                free_slots.append((slot, gpu))
                free_slots.sort(key=lambda item: slot_rank[item[0]])
                print(
                    f"{suffix.upper():5s} {record.job.key:28s} GPU={gpu} "
                    f"slot={slot} seconds={duration:.1f} "
                    f"active={len(active)}/{len(slot_assignments)} "
                    f"pending={len(pending)}",
                    flush=True,
                )

        if interrupted:
            _event(events, "scheduler_interrupted", failures=failures)
            raise SystemExit(130)
        _event(events, "scheduler_completed", failures=failures)

    if failures:
        raise RuntimeError(
            "B4-C2 jobs failed; aggregation is blocked: " + ", ".join(failures)
        )
    print("All 36 B4-C2 baseline jobs completed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"B4-C2 scheduler failed: {exc}", file=sys.stderr, flush=True)
        raise
