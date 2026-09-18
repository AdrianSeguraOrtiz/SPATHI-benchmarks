from __future__ import annotations

from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.execution_policy import (
    GIB,
    load_execution_policy,
    materialize_inference_run_resources,
    resolve_execution_class,
    resolve_workload_execution_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _runtime_plan(cpu: int, memory_gib: int) -> dict[str, object]:
    return {
        "fingerprint": "a" * 64,
        "capacity": {
            "cpu_threads": cpu,
            "memory_bytes": memory_gib * GIB,
            "memory_gib": float(memory_gib),
        },
        "detected": {
            "allowed_cpus": list(range(cpu)),
            "cpu_allocation_order": list(range(cpu)),
        },
    }


def test_committed_execution_policy_validates() -> None:
    policy = load_execution_policy(REPO_ROOT)
    assert policy["id"] == "portable_v1"
    assert set(policy["classes"]) == {
        "bulk_canonical",
        "bulk_full",
        "performance_full",
        "scale_canonical",
        "scale_full",
        "smoke_full",
    }


@pytest.mark.parametrize(
    ("cpu", "memory_gib", "tasks", "expected_threads", "expected_slots"),
    [(120, 1024, 10, 12, 10), (120, 1024, 1, 120, 1), (120, 1024, 60, 4, 30),
     (24, 32, 20, 12, 2), (2, 10, 4, 2, 1)],
)
def test_calibration_elasticity_uses_declared_workload_and_host_limits(
    cpu: int, memory_gib: int, tasks: int, expected_threads: int, expected_slots: int
) -> None:
    execution = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="bulk_canonical",
        runtime_plan=_runtime_plan(cpu, memory_gib),
    )
    resolved = resolve_workload_execution_plan(
        execution, purpose="calibration", physical_tasks=tasks
    )
    allocation = resolved["operations"]["inference"]
    assert allocation["cpu_threads"] == expected_threads
    assert allocation["workload_allocation"]["parallel_slots"] == expected_slots
    assert allocation["workload_allocation"]["physical_tasks"] == tasks
    assert allocation["memory_bytes"] == execution["operations"]["inference"]["memory_bytes"]
    assert "workload_allocation" not in execution["operations"]["inference"]
    run = {"run_id": "candidate", "params": {"random_seed": 42}}
    materialized = materialize_inference_run_resources(
        run, toolspec=_toolspec(maximum=None), execution_plan=resolved
    )
    assert materialized["params"] == run["params"]
    assert materialized["resources"]["threads"] == expected_threads
    assert resolved == resolve_workload_execution_plan(
        execution, purpose="calibration", physical_tasks=tasks
    )


def test_elasticity_respects_concurrency_ceiling_and_leaves_comparisons_fixed() -> None:
    runtime = _runtime_plan(32, 128)
    runtime["capacity"]["max_concurrent_tasks"] = 2
    execution = resolve_execution_class(
        repo_root=REPO_ROOT, class_id="bulk_full", runtime_plan=runtime
    )
    resolved = resolve_workload_execution_plan(execution, purpose="selection", physical_tasks=20)
    assert resolved["operations"]["inference"]["cpu_threads"] == 16
    assert resolve_workload_execution_plan(
        execution, purpose="comparison", physical_tasks=1
    ) == execution


def test_portable_class_clamps_preferences_to_small_host() -> None:
    resolved = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="bulk_canonical",
        runtime_plan=_runtime_plan(2, 10),
    )

    assert resolved["operations"]["inference"] == {
        "cpu_threads": 2,
        "memory_bytes": 10 * GIB,
        "memory_gib": 10.0,
        "planner": "auto",
        "planner_time_limit_seconds": 300,
        "timeout_seconds": 14400,
    }
    assert resolved["operations"]["generate_data"]["max_parallel_tasks"] == 2
    assert resolved["output_profile"] == "canonical"
    assert len(resolved["fingerprint"]) == 64


def test_class_rejects_host_below_declared_minimum() -> None:
    with pytest.raises(BenchmarkConfigError, match="requires at least 8 GiB"):
        resolve_execution_class(
            repo_root=REPO_ROOT,
            class_id="performance_full",
            runtime_plan=_runtime_plan(8, 4),
        )


def test_unknown_execution_class_is_rejected() -> None:
    with pytest.raises(BenchmarkConfigError, match="Unknown execution class"):
        resolve_execution_class(
            repo_root=REPO_ROOT,
            class_id="missing",
            runtime_plan=_runtime_plan(8, 32),
        )


def _toolspec(*, supported: bool = True, maximum: int | None = 8) -> dict[str, object]:
    return {
        "id": "fixture",
        "runtime_resources": {
            "threading": {
                "supported": supported,
                "default_threads": 1,
                "max_threads": maximum,
            }
        },
    }


def test_bulk_run_receives_explicit_threads_without_changing_params() -> None:
    execution = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="bulk_canonical",
        runtime_plan=_runtime_plan(8, 32),
    )
    run = {"run_id": "fixture", "tool_id": "fixture", "params": {"seed": 7}}

    resolved = materialize_inference_run_resources(
        run,
        toolspec=_toolspec(maximum=2),
        execution_plan=execution,
    )

    assert resolved["resources"] == {
        "threads": 2,
        "ram_gb": 16.0,
        "cpuset_cpus": list(range(8)),
        "timeout_seconds": 14400.0,
    }
    assert resolved["params"] == {"seed": 7}
    assert "resources" not in run


def test_tool_without_intrinsic_thread_limit_uses_portable_class_budget() -> None:
    execution = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="bulk_canonical",
        runtime_plan=_runtime_plan(8, 32),
    )

    resolved = materialize_inference_run_resources(
        {"run_id": "spathi", "params": {}},
        toolspec=_toolspec(maximum=None),
        execution_plan=execution,
    )

    assert resolved["resources"] == {
        "threads": 4,
        "ram_gb": 16.0,
        "cpuset_cpus": list(range(8)),
        "timeout_seconds": 14400.0,
    }


def test_inference_timeout_is_bounded_by_the_frozen_policy() -> None:
    execution = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="scale_full",
        runtime_plan=_runtime_plan(8, 64),
    )
    shorter = materialize_inference_run_resources(
        {"run_id": "scale", "resources": {"timeout_seconds": 43200}},
        toolspec=_toolspec(maximum=None),
        execution_plan=execution,
    )
    assert shorter["resources"]["timeout_seconds"] == 43200.0
    with pytest.raises(BenchmarkConfigError, match="exceeds the frozen execution policy"):
        materialize_inference_run_resources(
            {"run_id": "scale", "resources": {"timeout_seconds": 86401}},
            toolspec=_toolspec(maximum=None),
            execution_plan=execution,
        )


def test_performance_run_keeps_exact_threads_and_pins_runtime_cpuset() -> None:
    execution = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="performance_full",
        runtime_plan=_runtime_plan(8, 64),
    )
    resolved = materialize_inference_run_resources(
        {"run_id": "fixture", "resources": {"threads": 4}, "params": {}},
        toolspec=_toolspec(),
        execution_plan=execution,
    )

    assert resolved["resources"] == {
        "threads": 4,
        "ram_gb": 64.0,
        "cpuset_cpus": [0, 1, 2, 3],
        "timeout_seconds": 14400.0,
    }


def test_topology_prefix_is_encoded_as_a_canonical_andrea_cpuset() -> None:
    plan = _runtime_plan(4, 64)
    plan["detected"] = {
        "allowed_cpus": [2, 4, 6, 8],
        "cpu_allocation_order": [2, 6, 4, 8],
    }
    execution = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="performance_full",
        runtime_plan=plan,
    )

    resolved = materialize_inference_run_resources(
        {"run_id": "fixture", "resources": {"threads": 4}, "params": {}},
        toolspec=_toolspec(),
        execution_plan=execution,
    )

    assert resolved["resources"]["cpuset_cpus"] == [2, 4, 6, 8]


@pytest.mark.parametrize("ram_gb", [0, -1, True, float("nan"), float("inf")])
def test_run_rejects_invalid_explicit_ram(ram_gb: object) -> None:
    execution = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="bulk_canonical",
        runtime_plan=_runtime_plan(8, 32),
    )

    with pytest.raises(BenchmarkConfigError, match="ram_gb must be finite and positive"):
        materialize_inference_run_resources(
            {"run_id": "fixture", "resources": {"ram_gb": ram_gb}, "params": {}},
            toolspec=_toolspec(),
            execution_plan=execution,
        )


def test_run_rejects_ram_above_runtime_allocation() -> None:
    execution = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="bulk_canonical",
        runtime_plan=_runtime_plan(8, 32),
    )

    with pytest.raises(BenchmarkConfigError, match="requests 17 GiB RAM.*allocates 16 GiB"):
        materialize_inference_run_resources(
            {"run_id": "fixture", "resources": {"ram_gb": 17}, "params": {}},
            toolspec=_toolspec(),
            execution_plan=execution,
        )


def test_exact_thread_request_above_runtime_allocation_is_rejected() -> None:
    execution = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="bulk_full",
        runtime_plan=_runtime_plan(2, 32),
    )
    with pytest.raises(BenchmarkConfigError, match="runtime plan allocates 2"):
        materialize_inference_run_resources(
            {"run_id": "fixture", "resources": {"threads": 4}, "params": {}},
            toolspec=_toolspec(),
            execution_plan=execution,
        )


def test_performance_cpuset_must_stay_inside_effective_affinity() -> None:
    execution = resolve_execution_class(
        repo_root=REPO_ROOT,
        class_id="performance_full",
        runtime_plan=_runtime_plan(8, 64),
    )
    with pytest.raises(BenchmarkConfigError, match="escapes effective affinity"):
        materialize_inference_run_resources(
            {
                "run_id": "fixture",
                "resources": {"threads": 2, "cpuset_cpus": [0, 99]},
                "params": {},
            },
            toolspec=_toolspec(),
            execution_plan=execution,
        )
