from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from spathi_bench import resource_scheduler
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.resource_scheduler import (
    ResourceCapacity,
    ResourceClaim,
    ResourceScheduleError,
    ScheduledTask,
    execute_resource_schedule,
    host_execution_lease,
)


def _task(task_id: str, cpu: int, ram: int, *, exclusive: bool = False) -> ScheduledTask[str]:
    return ScheduledTask(task_id, task_id, ResourceClaim(cpu, ram, exclusive))


def test_scheduler_never_exceeds_cpu_or_ram_and_preserves_result_order() -> None:
    lock = threading.Lock()
    active_cpu = 0
    active_ram = 0
    peaks = [0, 0]
    claims = {"a": (2, 5), "b": (2, 5), "c": (1, 3)}

    def worker(task_id: str) -> str:
        nonlocal active_cpu, active_ram
        cpu, ram = claims[task_id]
        with lock:
            active_cpu += cpu
            active_ram += ram
            peaks[0] = max(peaks[0], active_cpu)
            peaks[1] = max(peaks[1], active_ram)
        time.sleep(0.02)
        with lock:
            active_cpu -= cpu
            active_ram -= ram
        return task_id.upper()

    capacity = ResourceCapacity(cpu_threads=4, memory_bytes=8)
    result = execute_resource_schedule(
        [_task("a", 2, 5), _task("b", 2, 5), _task("c", 1, 3)],
        capacity=capacity,
        worker=worker,
        thread_name_prefix="test-scheduler",
    )

    assert peaks[0] <= 4
    assert peaks[1] <= 8
    assert [task.task_id for task in result.tasks] == ["a", "b", "c"]
    assert [task.result for task in result.tasks] == ["A", "B", "C"]
    manifest = result.manifest()
    assert manifest["wall_seconds"] > 0
    assert {task["status"] for task in manifest["tasks"]} == {"complete"}
    utilization = result.utilization(capacity)
    assert 0 <= utilization["cpu_capacity_utilization"] <= 1
    assert 0 <= utilization["memory_capacity_utilization"] <= 1
    assert utilization["peak_reserved_cpu_threads"] <= capacity.cpu_threads
    assert utilization["peak_reserved_memory_bytes"] <= capacity.memory_bytes


def test_exclusive_task_never_overlaps_other_work() -> None:
    lock = threading.Lock()
    active: set[str] = set()

    def worker(task_id: str) -> str:
        with lock:
            if task_id == "exclusive":
                assert active == set()
            else:
                assert "exclusive" not in active
            active.add(task_id)
        time.sleep(0.015)
        with lock:
            active.remove(task_id)
        return task_id

    result = execute_resource_schedule(
        [
            _task("first", 1, 1),
            _task("exclusive", 1, 1, exclusive=True),
            _task("last", 1, 1),
        ],
        capacity=ResourceCapacity(cpu_threads=2, memory_bytes=2),
        worker=worker,
        thread_name_prefix="test-exclusive",
    )
    assert [task.result for task in result.tasks] == ["first", "exclusive", "last"]


def test_scheduler_fails_before_starting_an_oversized_task() -> None:
    called = False

    def worker(_: str) -> str:
        nonlocal called
        called = True
        return "unreachable"

    with pytest.raises(BenchmarkConfigError, match="requests 5 CPU"):
        execute_resource_schedule(
            [_task("large", 5, 1)],
            capacity=ResourceCapacity(cpu_threads=4, memory_bytes=8),
            worker=worker,
            thread_name_prefix="test-oversized",
        )
    assert not called


def test_failure_prevents_not_yet_admitted_tasks_from_starting() -> None:
    started: list[str] = []

    def worker(task_id: str) -> str:
        started.append(task_id)
        if task_id == "failure":
            raise ValueError("boom")
        return task_id

    with pytest.raises(ResourceScheduleError, match="failure") as captured:
        execute_resource_schedule(
            [_task("failure", 1, 1), _task("pending", 1, 1)],
            capacity=ResourceCapacity(cpu_threads=1, memory_bytes=1),
            worker=worker,
            thread_name_prefix="test-failure",
        )
    assert isinstance(captured.value.__cause__, ValueError)
    assert captured.value.manifest["status"] == "failed"
    assert captured.value.manifest["failed_task_id"] == "failure"
    assert captured.value.manifest["tasks_completed"] == 0
    assert captured.value.manifest["tasks_failed"] == 1
    assert captured.value.manifest["tasks_started"] == 1
    assert captured.value.manifest["utilization"]["peak_concurrent_tasks"] == 1
    [failure] = captured.value.manifest["tasks"]
    assert failure["task_id"] == "failure"
    assert failure["status"] == "failed"
    assert failure["cpu_threads"] == 1
    assert failure["memory_bytes"] == 1
    assert failure["started_ns"] <= failure["finished_ns"]
    assert failure["error"] == {"type": "ValueError", "message": "boom"}
    assert started == ["failure"]


def test_host_execution_lease_fails_fast_with_holder_identity(tmp_path) -> None:
    with host_execution_lease(workload="first", lock_root=tmp_path) as lock_path:
        assert lock_path.is_file()
        with pytest.raises(BenchmarkConfigError, match='"workload": "first"'):
            with host_execution_lease(workload="second", lock_root=tmp_path):
                raise AssertionError("unreachable")

    with host_execution_lease(workload="second", lock_root=tmp_path):
        pass


def test_default_host_lease_root_is_independent_of_tmpdir(monkeypatch) -> None:
    uid = 1234
    monkeypatch.setenv("TMPDIR", "/first/process-specific/path")
    first = resource_scheduler._canonical_lock_root(uid)
    monkeypatch.setenv("TMPDIR", "/second/process-specific/path")
    second = resource_scheduler._canonical_lock_root(uid)

    assert first == second == Path("/tmp/spathi-benchmarks-1234")
