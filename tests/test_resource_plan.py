from __future__ import annotations

import json
from pathlib import Path

import pytest

from spathi_bench.artifacts import canonical_json_sha256
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.resource_plan import (
    detect_host_capacity,
    load_or_create_runtime_plan,
    resolve_runtime_plan,
    validate_runtime_plan,
)

GIB = 1024**3


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _topology(root: Path, cpus: range) -> None:
    for cpu in cpus:
        # Two logical siblings per physical core.
        _write(root / f"cpu{cpu}/topology/physical_package_id", "0\n")
        _write(root / f"cpu{cpu}/topology/core_id", f"{cpu // 2}\n")


def test_detection_intersects_affinity_cpuset_quota_and_remaining_memory(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    sys_cpu = tmp_path / "sys-cpu"
    _write(proc / "self/cgroup", "0::/job/42\n")
    _write(cgroup / "cgroup.controllers", "cpu memory cpuset\n")
    _write(cgroup / "job/42/cpu.max", "350000 100000\n")
    _write(cgroup / "job/42/memory.max", str(12 * GIB))
    _write(cgroup / "job/42/memory.current", str(2 * GIB))
    _write(cgroup / "job/42/cpuset.cpus.effective", "2-7\n")
    _topology(sys_cpu, range(8))

    detected = detect_host_capacity(
        proc_root=proc,
        cgroup_root=cgroup,
        sys_cpu_root=sys_cpu,
        affinity_cpus=range(8),
        host_logical_cpus=8,
        host_total_memory_bytes=32 * GIB,
        host_available_memory_bytes=20 * GIB,
    )

    assert detected["allowed_cpus"] == [2, 3, 4, 5, 6, 7]
    assert detected["cpu_allocation_order"] == [2, 4, 6, 3, 5, 7]
    assert detected["cgroup_cpu_quota_cores"] == 3.5
    assert detected["effective_cpu_threads"] == 3
    assert detected["physical_cores_allowed"] == 3
    assert detected["cgroup_memory_remaining_bytes"] == 10 * GIB
    assert detected["effective_memory_bytes"] == 10 * GIB


def test_detection_uses_the_tightest_remaining_memory_across_cgroup_v2_ancestors(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _write(proc / "self/cgroup", "0::/parent/job\n")
    _write(cgroup / "cgroup.controllers", "memory\n")
    _write(cgroup / "parent/job/memory.max", str(20 * GIB))
    _write(cgroup / "parent/job/memory.current", str(2 * GIB))
    _write(cgroup / "parent/memory.max", str(12 * GIB))
    _write(cgroup / "parent/memory.current", str(5 * GIB))

    detected = detect_host_capacity(
        proc_root=proc,
        cgroup_root=cgroup,
        sys_cpu_root=tmp_path / "absent-topology",
        affinity_cpus=range(4),
        host_logical_cpus=4,
        host_total_memory_bytes=32 * GIB,
        host_available_memory_bytes=24 * GIB,
    )

    assert detected["cgroup_memory_limit_bytes"] == 12 * GIB
    assert detected["cgroup_memory_current_bytes"] == 5 * GIB
    assert detected["cgroup_memory_remaining_bytes"] == 7 * GIB
    assert detected["effective_memory_bytes"] == 7 * GIB


def test_detection_supports_cgroup_v1(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _write(
        proc / "self/cgroup",
        "4:cpu,cpuacct:/batch\n3:memory:/batch\n2:cpuset:/batch\n",
    )
    _write(cgroup / "cpu/batch/cpu.cfs_quota_us", "200000\n")
    _write(cgroup / "cpu/batch/cpu.cfs_period_us", "100000\n")
    _write(cgroup / "memory/batch/memory.limit_in_bytes", str(8 * GIB))
    _write(cgroup / "memory/batch/memory.usage_in_bytes", str(1 * GIB))
    _write(cgroup / "cpuset/batch/cpuset.cpus", "1-4\n")

    detected = detect_host_capacity(
        proc_root=proc,
        cgroup_root=cgroup,
        sys_cpu_root=tmp_path / "absent-topology",
        affinity_cpus=range(6),
        host_logical_cpus=6,
        host_total_memory_bytes=16 * GIB,
        host_available_memory_bytes=12 * GIB,
    )

    assert detected["allowed_cpus"] == [1, 2, 3, 4]
    assert detected["cpu_allocation_order"] == [1, 2, 3, 4]
    assert detected["effective_cpu_threads"] == 2
    assert detected["effective_memory_bytes"] == 7 * GIB


def test_detection_supports_combined_v1_mounts_and_inherited_limits(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _write(
        proc / "self/cgroup",
        "4:cpu,cpuacct:/parent/job\n3:memory:/parent/job\n2:cpuset:/parent/job\n",
    )
    _write(cgroup / "cpu,cpuacct/parent/job/cpu.cfs_quota_us", "-1\n")
    _write(cgroup / "cpu,cpuacct/parent/job/cpu.cfs_period_us", "100000\n")
    _write(cgroup / "cpu,cpuacct/parent/cpu.cfs_quota_us", "250000\n")
    _write(cgroup / "cpu,cpuacct/parent/cpu.cfs_period_us", "100000\n")
    _write(cgroup / "memory/parent/job/memory.limit_in_bytes", str(20 * GIB))
    _write(cgroup / "memory/parent/job/memory.usage_in_bytes", str(2 * GIB))
    _write(cgroup / "memory/parent/memory.limit_in_bytes", str(10 * GIB))
    _write(cgroup / "memory/parent/memory.usage_in_bytes", str(4 * GIB))
    _write(cgroup / "cpuset/parent/job/cpuset.cpus", "\n")
    _write(cgroup / "cpuset/parent/cpuset.effective_cpus", "2-5\n")

    detected = detect_host_capacity(
        proc_root=proc,
        cgroup_root=cgroup,
        sys_cpu_root=tmp_path / "absent-topology",
        affinity_cpus=range(8),
        host_logical_cpus=8,
        host_total_memory_bytes=32 * GIB,
        host_available_memory_bytes=24 * GIB,
    )

    assert detected["allowed_cpus"] == [2, 3, 4, 5]
    assert detected["cgroup_cpu_quota_cores"] == 2.5
    assert detected["effective_cpu_threads"] == 2
    assert detected["cgroup_memory_limit_bytes"] == 10 * GIB
    assert detected["cgroup_memory_current_bytes"] == 4 * GIB
    assert detected["effective_memory_bytes"] == 6 * GIB


def test_detection_normalizes_unlimited_cgroup_v1_memory(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _write(proc / "self/cgroup", "3:memory:/batch\n")
    _write(cgroup / "memory/batch/memory.limit_in_bytes", "9223372036854771712\n")
    _write(cgroup / "memory/batch/memory.usage_in_bytes", str(2 * GIB))

    detected = detect_host_capacity(
        proc_root=proc,
        cgroup_root=cgroup,
        sys_cpu_root=tmp_path / "absent-topology",
        affinity_cpus=range(4),
        host_logical_cpus=4,
        host_total_memory_bytes=16 * GIB,
        host_available_memory_bytes=10 * GIB,
    )

    assert detected["cgroup_memory_limit_bytes"] is None
    assert detected["cgroup_memory_remaining_bytes"] is None
    assert detected["effective_memory_bytes"] == 10 * GIB


def test_resolution_applies_operator_caps_and_reserves() -> None:
    detected = {
        "effective_cpu_threads": 60,
        "effective_memory_bytes": 1000 * GIB,
    }

    plan = resolve_runtime_plan(
        detected,
        max_cores=24,
        max_ram_gb=256,
        reserve_cores=2,
        reserve_ram_gb=16,
    )

    assert plan["capacity"] == {
        "cpu_threads": 22,
        "memory_bytes": 240 * GIB,
        "memory_gib": 240.0,
    }
    assert len(plan["fingerprint"]) == 64


def test_resolution_rejects_reserve_that_exhausts_capacity() -> None:
    with pytest.raises(BenchmarkConfigError, match="no schedulable thread"):
        resolve_runtime_plan(
            {"effective_cpu_threads": 4, "effective_memory_bytes": 8 * GIB},
            reserve_cores=4,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_cores": True}, "positive integer"),
        ({"max_ram_gb": True}, "finite and positive"),
        ({"reserve_cores": True}, "non-negative integer"),
        ({"reserve_ram_gb": True}, "finite and non-negative"),
    ],
)
def test_resolution_rejects_boolean_resource_options(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(BenchmarkConfigError, match=message):
        resolve_runtime_plan(
            {"effective_cpu_threads": 8, "effective_memory_bytes": 32 * GIB},
            **kwargs,
        )


def test_runtime_plan_rejects_rehashed_capacity_drift() -> None:
    plan = resolve_runtime_plan({"effective_cpu_threads": 8, "effective_memory_bytes": 32 * GIB})
    plan["capacity"]["cpu_threads"] = 7
    plan["fingerprint"] = canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "fingerprint"}
    )
    with pytest.raises(BenchmarkConfigError, match="capacity differs"):
        validate_runtime_plan(plan)


def test_runtime_plan_is_reused_when_current_capacity_still_covers_it(tmp_path: Path) -> None:
    first_detected = {
        "effective_cpu_threads": 8,
        "effective_memory_bytes": 32 * GIB,
    }
    path = tmp_path / "run/runtime-plan.json"
    first = load_or_create_runtime_plan(
        path,
        boundary=tmp_path,
        max_cores=8,
        detected=first_detected,
    )
    second = load_or_create_runtime_plan(
        path,
        boundary=tmp_path,
        max_cores=8,
        detected={"effective_cpu_threads": 12, "effective_memory_bytes": 64 * GIB},
    )

    assert second == first
    assert json.loads(path.read_text(encoding="utf-8")) == first
    with pytest.raises(BenchmarkConfigError, match="different resource options"):
        load_or_create_runtime_plan(
            path,
            boundary=tmp_path,
            max_cores=4,
            detected=first_detected,
        )


@pytest.mark.parametrize(
    ("current", "message"),
    [
        (
            {"effective_cpu_threads": 7, "effective_memory_bytes": 32 * GIB},
            "CPU capacity no longer covers",
        ),
        (
            {"effective_cpu_threads": 8, "effective_memory_bytes": 31 * GIB},
            "RAM capacity no longer covers",
        ),
    ],
)
def test_runtime_plan_reuse_rejects_reduced_current_capacity(
    tmp_path: Path, current: dict[str, int], message: str
) -> None:
    path = tmp_path / "run/runtime-plan.json"
    load_or_create_runtime_plan(
        path,
        boundary=tmp_path,
        detected={"effective_cpu_threads": 8, "effective_memory_bytes": 32 * GIB},
    )

    with pytest.raises(BenchmarkConfigError, match=message):
        load_or_create_runtime_plan(path, boundary=tmp_path, detected=current)


def test_runtime_plan_reuse_rejects_changed_cpu_affinity(tmp_path: Path) -> None:
    path = tmp_path / "run/runtime-plan.json"
    frozen = {
        "effective_cpu_threads": 4,
        "effective_memory_bytes": 16 * GIB,
        "allowed_cpus": [0, 1, 2, 3],
        "cpu_allocation_order": [0, 2, 1, 3],
    }
    load_or_create_runtime_plan(path, boundary=tmp_path, detected=frozen)

    current = {
        **frozen,
        "allowed_cpus": [1, 2, 3, 4],
        "cpu_allocation_order": [1, 3, 2, 4],
    }
    with pytest.raises(BenchmarkConfigError, match="CPU affinity/cgroup CPU set"):
        load_or_create_runtime_plan(path, boundary=tmp_path, detected=current)


def test_runtime_plan_reuse_checks_memory_ceiling_not_transient_usage(tmp_path: Path) -> None:
    path = tmp_path / "run/runtime-plan.json"
    frozen = {
        "effective_cpu_threads": 4,
        "effective_memory_bytes": 8 * GIB,
        "host_memory_total_bytes": 32 * GIB,
        "cgroup_memory_limit_bytes": 12 * GIB,
    }
    plan = load_or_create_runtime_plan(path, boundary=tmp_path, detected=frozen)

    current = {
        **frozen,
        "effective_memory_bytes": 7 * GIB,
    }
    assert load_or_create_runtime_plan(path, boundary=tmp_path, detected=current) == plan

    current["cgroup_memory_limit_bytes"] = 7 * GIB
    with pytest.raises(BenchmarkConfigError, match="RAM capacity no longer covers"):
        load_or_create_runtime_plan(path, boundary=tmp_path, detected=current)


def test_runtime_plan_rejects_tampered_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "runtime-plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "request": {
                    "max_cores": None,
                    "max_ram_gb": None,
                    "reserve_cores": 0,
                    "reserve_ram_gb": 0.0,
                },
                "capacity": {
                    "cpu_threads": 1,
                    "memory_bytes": GIB,
                    "memory_gib": 1.0,
                },
                "detected": {},
                "fingerprint": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkConfigError, match="fingerprint is invalid"):
        load_or_create_runtime_plan(path, boundary=tmp_path)
