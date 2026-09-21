"""Deterministic CPU/RAM admission for rolling benchmark task queues."""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from .config import BenchmarkConfigError

Payload = TypeVar("Payload")
Result = TypeVar("Result")


@dataclass(frozen=True, slots=True)
class ResourceCapacity:
    cpu_threads: int
    memory_bytes: int
    max_concurrent_tasks: int | None = None

    def __post_init__(self) -> None:
        if self.cpu_threads < 1 or self.memory_bytes < 1:
            raise BenchmarkConfigError("Scheduler CPU and RAM capacity must be positive")
        if self.max_concurrent_tasks is not None and self.max_concurrent_tasks < 1:
            raise BenchmarkConfigError("Scheduler concurrency ceiling must be positive")


@dataclass(frozen=True, slots=True)
class ResourceClaim:
    cpu_threads: int
    memory_bytes: int
    exclusive: bool = False

    def __post_init__(self) -> None:
        if self.cpu_threads < 1 or self.memory_bytes < 1:
            raise BenchmarkConfigError("Every task must claim positive CPU and RAM")


@dataclass(frozen=True, slots=True)
class ScheduledTask(Generic[Payload]):
    task_id: str
    payload: Payload
    claim: ResourceClaim

    def __post_init__(self) -> None:
        if not self.task_id:
            raise BenchmarkConfigError("Scheduled task identifiers cannot be empty")


@dataclass(frozen=True, slots=True)
class TaskExecution(Generic[Result]):
    task_id: str
    result: Result
    claim: ResourceClaim
    started_ns: int
    finished_ns: int


@dataclass(frozen=True, slots=True)
class TaskFailure:
    task_id: str
    claim: ResourceClaim
    started_ns: int
    finished_ns: int
    error_type: str
    error_message: str


def _task_manifest(
    task: TaskExecution[object] | TaskFailure, schedule_started_ns: int
) -> dict[str, object]:
    payload: dict[str, object] = {
        "task_id": task.task_id,
        "status": "failed" if isinstance(task, TaskFailure) else "complete",
        "cpu_threads": task.claim.cpu_threads,
        "memory_bytes": task.claim.memory_bytes,
        "exclusive": task.claim.exclusive,
        "started_ns": task.started_ns,
        "finished_ns": task.finished_ns,
        "start_offset_seconds": (task.started_ns - schedule_started_ns) / 1_000_000_000,
        "finish_offset_seconds": (task.finished_ns - schedule_started_ns) / 1_000_000_000,
        "wall_seconds": (task.finished_ns - task.started_ns) / 1_000_000_000,
    }
    if isinstance(task, TaskFailure):
        payload["error"] = {
            "type": task.error_type,
            "message": task.error_message,
        }
    return payload


def _resource_utilization(
    tasks: Sequence[TaskExecution[object] | TaskFailure],
    *,
    started_ns: int,
    finished_ns: int,
    capacity: ResourceCapacity,
) -> dict[str, object]:
    wall_seconds = (finished_ns - started_ns) / 1_000_000_000
    cpu_thread_seconds = sum(
        task.claim.cpu_threads * (task.finished_ns - task.started_ns) / 1_000_000_000
        for task in tasks
    )
    memory_byte_seconds = sum(
        task.claim.memory_bytes * (task.finished_ns - task.started_ns) / 1_000_000_000
        for task in tasks
    )
    events: list[tuple[int, int, ResourceClaim]] = []
    for task in tasks:
        events.append((task.started_ns, 1, task.claim))
        events.append((task.finished_ns, -1, task.claim))
    active_tasks = 0
    active_cpu = 0
    active_memory = 0
    peak_tasks = 0
    peak_cpu = 0
    peak_memory = 0
    for _when, delta, claim in sorted(events, key=lambda item: (item[0], item[1])):
        active_tasks += delta
        active_cpu += delta * claim.cpu_threads
        active_memory += delta * claim.memory_bytes
        peak_tasks = max(peak_tasks, active_tasks)
        peak_cpu = max(peak_cpu, active_cpu)
        peak_memory = max(peak_memory, active_memory)
    return {
        "wall_seconds": wall_seconds,
        "cpu_thread_seconds_reserved": cpu_thread_seconds,
        "memory_byte_seconds_reserved": memory_byte_seconds,
        "cpu_capacity_utilization": (
            cpu_thread_seconds / (wall_seconds * capacity.cpu_threads) if wall_seconds > 0 else 0.0
        ),
        "memory_capacity_utilization": (
            memory_byte_seconds / (wall_seconds * capacity.memory_bytes)
            if wall_seconds > 0
            else 0.0
        ),
        "peak_concurrent_tasks": peak_tasks,
        "peak_reserved_cpu_threads": peak_cpu,
        "peak_reserved_memory_bytes": peak_memory,
    }


@dataclass(frozen=True, slots=True)
class ScheduleResult(Generic[Result]):
    tasks: tuple[TaskExecution[Result], ...]
    started_ns: int
    finished_ns: int

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "wall_seconds": (self.finished_ns - self.started_ns) / 1_000_000_000,
            "tasks": [_task_manifest(task, self.started_ns) for task in self.tasks],
        }

    def utilization(self, capacity: ResourceCapacity) -> dict[str, object]:
        """Summarize reserved-resource utilization over the measured makespan."""

        return _resource_utilization(
            self.tasks,
            started_ns=self.started_ns,
            finished_ns=self.finished_ns,
            capacity=capacity,
        )


class ResourceScheduleError(RuntimeError):
    """One admitted task failed; no further queued work was launched."""

    def __init__(self, task_id: str, manifest: dict[str, object]) -> None:
        self.task_id = task_id
        self.manifest = manifest
        super().__init__(f"Scheduled task {task_id!r} failed")


def _canonical_lock_root(uid: int) -> Path:
    """Return a host-wide lease location that cannot vary with ``TMPDIR``."""

    return Path("/tmp") / f"spathi-benchmarks-{uid}"


@contextmanager
def host_execution_lease(
    *,
    workload: str,
    lock_root: Path | None = None,
) -> Iterator[Path]:
    """Prevent two benchmark workflows from contaminating resource measurements."""

    if not workload or any(character in workload for character in "\r\n"):
        raise BenchmarkConfigError("Workload lease label must be a non-empty single line")
    uid = os.getuid() if hasattr(os, "getuid") else 0
    directory = (lock_root if lock_root is not None else _canonical_lock_root(uid)).absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise BenchmarkConfigError(f"Host execution lock root is unsafe: {directory}")
    path = directory / f"spathi-benchmarks-execution-{uid}.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            holder = handle.read().strip() or "unknown holder"
            raise BenchmarkConfigError(
                f"Another SPATHI benchmark workload owns the host execution lease: {holder}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "workload": workload}, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class _Completed(Generic[Result]):
    result: Result | None
    error: BaseException | None
    started_ns: int
    finished_ns: int


def _invoke(worker: Callable[[Payload], Result], payload: Payload) -> _Completed[Result]:
    started = time.monotonic_ns()
    try:
        return _Completed(worker(payload), None, started, time.monotonic_ns())
    except BaseException as exc:
        return _Completed(None, exc, started, time.monotonic_ns())


def execute_resource_schedule(
    tasks: Sequence[ScheduledTask[Payload]],
    *,
    capacity: ResourceCapacity,
    worker: Callable[[Payload], Result],
    thread_name_prefix: str,
) -> ScheduleResult[Result]:
    """Execute a bounded rolling queue without CPU or RAM oversubscription.

    Admission scans pending tasks in declared order and chooses the first one
    that fits.  Results are returned in that same declared order.  Once any task
    fails, no additional pending task is launched; already-running tasks are
    allowed to terminate cleanly before the original exception is re-raised.
    """

    if not tasks:
        now = time.monotonic_ns()
        return ScheduleResult(tasks=(), started_ns=now, finished_ns=now)
    identifiers = [task.task_id for task in tasks]
    if len(set(identifiers)) != len(identifiers):
        raise BenchmarkConfigError("Scheduled task identifiers must be unique")
    for task in tasks:
        if task.claim.cpu_threads > capacity.cpu_threads:
            raise BenchmarkConfigError(
                f"Task {task.task_id!r} requests {task.claim.cpu_threads} CPU threads, "
                f"but only {capacity.cpu_threads} are schedulable"
            )
        if task.claim.memory_bytes > capacity.memory_bytes:
            raise BenchmarkConfigError(
                f"Task {task.task_id!r} requests {task.claim.memory_bytes} bytes of RAM, "
                f"but only {capacity.memory_bytes} are schedulable"
            )

    concurrent_ceiling = min(
        len(tasks),
        capacity.max_concurrent_tasks or capacity.cpu_threads,
    )
    pending = list(enumerate(tasks))
    active: dict[Future[_Completed[Result]], tuple[int, ScheduledTask[Payload]]] = {}
    completed: dict[int, TaskExecution[Result]] = {}
    failed: dict[int, TaskFailure] = {}
    used_cpu = 0
    used_memory = 0
    exclusive_active = False
    first_failure: tuple[str, BaseException] | None = None
    schedule_started = time.monotonic_ns()

    with ThreadPoolExecutor(
        max_workers=concurrent_ceiling,
        thread_name_prefix=thread_name_prefix,
    ) as pool:
        while pending or active:
            launched = False
            if first_failure is None and len(active) < concurrent_ceiling:
                for pending_position, (index, task) in enumerate(pending):
                    claim = task.claim
                    fits_exclusivity = not exclusive_active and (not claim.exclusive or not active)
                    if (
                        fits_exclusivity
                        and used_cpu + claim.cpu_threads <= capacity.cpu_threads
                        and used_memory + claim.memory_bytes <= capacity.memory_bytes
                    ):
                        future = pool.submit(copy_context().run, _invoke, worker, task.payload)
                        active[future] = (index, task)
                        used_cpu += claim.cpu_threads
                        used_memory += claim.memory_bytes
                        exclusive_active = claim.exclusive
                        pending.pop(pending_position)
                        launched = True
                        break
            if launched:
                continue
            if not active:
                if first_failure is not None:
                    break
                raise RuntimeError("Resource scheduler could not admit a prevalidated task")

            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: active[item][0]):
                index, task = active.pop(future)
                used_cpu -= task.claim.cpu_threads
                used_memory -= task.claim.memory_bytes
                if task.claim.exclusive:
                    exclusive_active = False
                outcome = future.result()
                if outcome.error is not None:
                    failed[index] = TaskFailure(
                        task_id=task.task_id,
                        claim=task.claim,
                        started_ns=outcome.started_ns,
                        finished_ns=outcome.finished_ns,
                        error_type=type(outcome.error).__name__,
                        error_message=str(outcome.error),
                    )
                    if first_failure is None:
                        first_failure = (task.task_id, outcome.error)
                else:
                    completed[index] = TaskExecution(
                        task_id=task.task_id,
                        result=outcome.result,
                        claim=task.claim,
                        started_ns=outcome.started_ns,
                        finished_ns=outcome.finished_ns,
                    )

    schedule_finished = time.monotonic_ns()
    if first_failure is not None:
        task_id, cause = first_failure
        executions: list[TaskExecution[object] | TaskFailure] = [
            completed[index] if index in completed else failed[index]
            for index in sorted(set(completed) | set(failed))
        ]
        partial: dict[str, object] = {
            "schema_version": "1.0",
            "wall_seconds": (schedule_finished - schedule_started) / 1_000_000_000,
            "tasks": [_task_manifest(task, schedule_started) for task in executions],
            "utilization": _resource_utilization(
                executions,
                started_ns=schedule_started,
                finished_ns=schedule_finished,
                capacity=capacity,
            ),
        }
        partial.update(
            {
                "status": "failed",
                "failed_task_id": task_id,
                "tasks_declared": len(tasks),
                "tasks_completed": len(completed),
                "tasks_failed": len(failed),
                "tasks_started": len(executions),
            }
        )
        raise ResourceScheduleError(task_id, partial) from cause
    if len(completed) != len(tasks):
        raise RuntimeError("Resource scheduler terminated without completing every task")
    return ScheduleResult(
        tasks=tuple(completed[index] for index in range(len(tasks))),
        started_ns=schedule_started,
        finished_ns=schedule_finished,
    )


__all__ = [
    "ResourceCapacity",
    "ResourceClaim",
    "ResourceScheduleError",
    "ScheduleResult",
    "ScheduledTask",
    "TaskFailure",
    "TaskExecution",
    "execute_resource_schedule",
    "host_execution_lease",
]
