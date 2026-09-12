"""Load and resolve the versioned, machine-independent execution policy."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .config import BenchmarkConfigError, load_json_object, validate_json

GIB = 1024**3
POLICY_PATH = Path("experiments/v1/execution/policy.json")


def load_execution_policy(repo_root: Path) -> dict[str, Any]:
    path = repo_root / POLICY_PATH
    policy = load_json_object(path, "execution policy")
    validate_json(
        policy,
        repo_root / "schemas/execution-policy.schema.json",
        "execution policy",
    )
    return policy


def _resolve_allocation(
    allocation: Mapping[str, Any],
    *,
    capacity: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    available_cpu = int(capacity["cpu_threads"])
    available_memory = int(capacity["memory_bytes"])
    minimum_cpu = int(allocation["minimum_cpu_threads"])
    preferred_cpu = int(allocation["preferred_cpu_threads"])
    minimum_memory = int(float(allocation["minimum_memory_gib"]) * GIB)
    preferred_memory = int(float(allocation["preferred_memory_gib"]) * GIB)
    if preferred_cpu < minimum_cpu or preferred_memory < minimum_memory:
        raise BenchmarkConfigError(f"{label} preferred resources are below their minimum")
    if available_cpu < minimum_cpu:
        raise BenchmarkConfigError(
            f"{label} requires at least {minimum_cpu} CPU threads; {available_cpu} are available"
        )
    if available_memory < minimum_memory:
        raise BenchmarkConfigError(
            f"{label} requires at least {minimum_memory / GIB:g} GiB RAM; "
            f"{available_memory / GIB:g} GiB are available"
        )
    return {
        "cpu_threads": min(preferred_cpu, available_cpu),
        "memory_bytes": min(preferred_memory, available_memory),
        "memory_gib": min(preferred_memory, available_memory) / GIB,
    }


def resolve_execution_class(
    *,
    repo_root: Path,
    class_id: str,
    runtime_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one portable class against a frozen runtime plan."""

    policy = load_execution_policy(repo_root)
    classes = policy["classes"]
    if class_id not in classes:
        raise BenchmarkConfigError(f"Unknown execution class: {class_id!r}")
    selected = classes[class_id]
    capacity = runtime_plan.get("capacity")
    if not isinstance(capacity, Mapping):
        raise BenchmarkConfigError("Runtime plan has no valid capacity")
    resolved = {
        operation: _resolve_allocation(
            selected[operation],
            capacity=capacity,
            label=f"Execution class {class_id!r} {operation}",
        )
        for operation in ("generate_data", "inference", "evaluation")
    }
    resolved["generate_data"]["max_parallel_tasks"] = resolved["generate_data"]["cpu_threads"]
    resolved["inference"].update(
        {
            "planner": selected["inference"]["planner"],
            "planner_time_limit_seconds": selected["inference"]["planner_time_limit_seconds"],
        }
    )
    body: dict[str, Any] = {
        "schema_version": "1.0",
        "policy": {
            "id": policy["id"],
            "path": POLICY_PATH.as_posix(),
            "sha256": sha256_file(repo_root / POLICY_PATH),
        },
        "class_id": class_id,
        "runtime_plan_fingerprint": runtime_plan.get("fingerprint"),
        "scheduler": policy["scheduler"],
        "capacity": dict(capacity),
        "detected": dict(runtime_plan.get("detected", {})),
        "operations": resolved,
        "output_profile": selected["output_profile"],
        "exclusive_inference": selected["exclusive_inference"],
    }
    body["fingerprint"] = canonical_json_sha256(body)
    return body


def materialize_inference_run_resources(
    run: Mapping[str, Any],
    *,
    toolspec: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one run with explicit, validated ANDREA operational resources."""

    operations = execution_plan.get("operations")
    if not isinstance(operations, Mapping) or not isinstance(operations.get("inference"), Mapping):
        raise BenchmarkConfigError("Resolved execution plan has no inference allocation")
    allocation = operations["inference"]
    class_threads = allocation.get("cpu_threads")
    class_memory_gib = allocation.get("memory_gib")
    if not isinstance(class_threads, int) or isinstance(class_threads, bool) or class_threads < 1:
        raise BenchmarkConfigError("Resolved inference CPU allocation is invalid")
    if (
        not isinstance(class_memory_gib, (int, float))
        or isinstance(class_memory_gib, bool)
        or not math.isfinite(float(class_memory_gib))
        or float(class_memory_gib) <= 0
    ):
        raise BenchmarkConfigError("Resolved inference RAM allocation is invalid")

    runtime = toolspec.get("runtime_resources")
    threading = runtime.get("threading") if isinstance(runtime, Mapping) else None
    if not isinstance(threading, Mapping):
        raise BenchmarkConfigError(
            f"Tool {toolspec.get('id')!r} has no runtime_resources.threading contract"
        )
    supported = threading.get("supported")
    maximum = threading.get("max_threads")
    if not isinstance(supported, bool):
        raise BenchmarkConfigError("Tool threading supported flag is invalid")
    if maximum is not None and (
        not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1
    ):
        raise BenchmarkConfigError("Tool threading maximum is invalid")

    raw_resources = run.get("resources")
    if raw_resources is not None and not isinstance(raw_resources, Mapping):
        raise BenchmarkConfigError(f"Inference run {run.get('run_id')!r} resources are invalid")
    resources_input = dict(raw_resources or {})
    if set(resources_input) - {"threads", "ram_gb", "cpuset_cpus", "timeout_seconds"}:
        raise BenchmarkConfigError(
            f"Inference run {run.get('run_id')!r} has unsupported resource fields"
        )
    explicit_threads = resources_input.get("threads")
    if explicit_threads is None:
        threads = (
            min(class_threads, maximum)
            if supported and maximum is not None
            else (class_threads if supported else 1)
        )
    elif (
        not isinstance(explicit_threads, int)
        or isinstance(explicit_threads, bool)
        or explicit_threads < 1
    ):
        raise BenchmarkConfigError(
            f"Inference run {run.get('run_id')!r} threads must be a positive integer"
        )
    else:
        threads = explicit_threads
    if not supported and threads != 1:
        raise BenchmarkConfigError(
            f"Inference run {run.get('run_id')!r} requests parallelism from a serial tool"
        )
    if maximum is not None and threads > maximum:
        raise BenchmarkConfigError(
            f"Inference run {run.get('run_id')!r} requests {threads} threads above tool maximum "
            f"{maximum}"
        )
    if threads > class_threads:
        raise BenchmarkConfigError(
            f"Inference run {run.get('run_id')!r} requests {threads} threads but the runtime "
            f"plan allocates {class_threads}"
        )

    explicit_ram_gb = resources_input.get("ram_gb")
    if explicit_ram_gb is None:
        ram_gb = float(class_memory_gib)
    elif (
        not isinstance(explicit_ram_gb, (int, float))
        or isinstance(explicit_ram_gb, bool)
        or not math.isfinite(float(explicit_ram_gb))
        or float(explicit_ram_gb) <= 0
    ):
        raise BenchmarkConfigError(
            f"Inference run {run.get('run_id')!r} ram_gb must be finite and positive"
        )
    else:
        ram_gb = float(explicit_ram_gb)
    if ram_gb > float(class_memory_gib):
        raise BenchmarkConfigError(
            f"Inference run {run.get('run_id')!r} requests {ram_gb:g} GiB RAM but the runtime "
            f"plan allocates {float(class_memory_gib):g} GiB"
        )

    resources: dict[str, Any] = {"threads": threads, "ram_gb": ram_gb}
    detected = execution_plan.get("detected")
    capacity = execution_plan.get("capacity")
    allowed = detected.get("allowed_cpus") if isinstance(detected, Mapping) else None
    allocation_order = (
        detected.get("cpu_allocation_order") if isinstance(detected, Mapping) else None
    )
    schedulable_threads = capacity.get("cpu_threads") if isinstance(capacity, Mapping) else None
    if (
        not isinstance(allowed, list)
        or not isinstance(allocation_order, list)
        or not isinstance(schedulable_threads, int)
        or isinstance(schedulable_threads, bool)
        or schedulable_threads < 1
        or any(
            not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0
            for cpu in (*allowed, *allocation_order)
        )
        or allowed != sorted(set(allowed))
        or sorted(allocation_order) != allowed
        or len(set(allocation_order)) != len(allocation_order)
        or len(allocation_order) < schedulable_threads
    ):
        raise BenchmarkConfigError(
            "Inference requires the runtime plan's canonical CPU allocation order"
        )
    selected_cpus = (
        allocation_order[:threads]
        if execution_plan.get("exclusive_inference")
        else allocation_order[:schedulable_threads]
    )
    # ``cpu_allocation_order`` is ordered by topology (physical cores before
    # SMT siblings), whereas ANDREA's public resource contract requires the
    # JSON representation of a CPU *set* to be strictly increasing.  Select
    # from the topology-aware prefix first, then canonicalize its encoding.
    canonical_cpuset = sorted(selected_cpus)
    raw_cpuset = resources_input.get("cpuset_cpus", canonical_cpuset)
    if (
        not isinstance(raw_cpuset, list)
        or len(raw_cpuset) < threads
        or any(
            not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0 for cpu in raw_cpuset
        )
        or len(set(raw_cpuset)) != len(raw_cpuset)
    ):
        raise BenchmarkConfigError(
            f"Inference run {run.get('run_id')!r} cpuset_cpus is not canonical"
        )
    if not set(raw_cpuset).issubset(allowed):
        raise BenchmarkConfigError(
            f"Inference run {run.get('run_id')!r} cpuset_cpus escapes effective affinity"
        )
    if raw_cpuset != canonical_cpuset:
        scope = "isolated allocation" if execution_plan.get("exclusive_inference") else "shared domain"
        raise BenchmarkConfigError(
            f"Inference run {run.get('run_id')!r} cpuset_cpus differs from the canonical "
            f"topology-aware {scope}"
        )
    resources["cpuset_cpus"] = list(raw_cpuset)

    if "timeout_seconds" in resources_input:
        timeout_seconds = resources_input["timeout_seconds"]
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0
        ):
            raise BenchmarkConfigError(
                f"Inference run {run.get('run_id')!r} timeout_seconds must be finite and positive"
            )
        resources["timeout_seconds"] = float(timeout_seconds)

    resolved = copy.deepcopy(dict(run))
    resolved["resources"] = resources
    return resolved


def resolve_workload_execution_plan(
    execution_plan: Mapping[str, Any],
    *,
    purpose: str,
    physical_tasks: int,
) -> dict[str, Any]:
    """Allocate idle CPU capacity to a small, prospectively declared search stage.

    Allocation uses the full physical task ledger, never cache misses or completion
    times, so interruption and resumption cannot change a run's resources.  Formal
    timing and competitor comparisons retain their fixed per-run allocations.
    """

    resolved = copy.deepcopy(dict(execution_plan))
    if purpose not in {"calibration", "selection"}:
        return resolved
    if (
        type(physical_tasks) is not int
        or physical_tasks < 1
        or resolved.get("exclusive_inference") is not False
    ):
        raise BenchmarkConfigError("Elastic calibration requires a positive non-exclusive workload")
    if resolved.get("scheduler", {}).get("calibration_cpu_allocation") != "declared_workload_elastic":
        raise BenchmarkConfigError("Execution policy has no declared calibration CPU allocation")
    allocation = resolved["operations"]["inference"]
    capacity = resolved["capacity"]
    base_threads = int(allocation["cpu_threads"])
    cpu_threads = int(capacity["cpu_threads"])
    slots = min(
        physical_tasks,
        max(1, cpu_threads // base_threads),
        max(1, int(capacity["memory_bytes"]) // int(allocation["memory_bytes"])),
        int(capacity.get("max_concurrent_tasks") or cpu_threads),
    )
    allocation["cpu_threads"] = max(base_threads, cpu_threads // slots)
    allocation["workload_allocation"] = {
        "strategy": "declared_workload_elastic",
        "physical_tasks": physical_tasks,
        "parallel_slots": slots,
        "preferred_cpu_threads": base_threads,
    }
    resolved.pop("fingerprint", None)
    resolved["fingerprint"] = canonical_json_sha256(resolved)
    return resolved


def inference_task_execution_plan(
    execution_plan: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Isolate designated timing seeds while letting held-back accuracy run together."""

    resolved = copy.deepcopy(dict(execution_plan))
    if resolved.get("class_id") != "efficiency_development_canonical":
        return resolved
    role = metadata.get("seed_role")
    if metadata.get("family") != "spathi_efficiency_development" or role not in {"timing", "accuracy"}:
        raise BenchmarkConfigError("Efficiency-development task has no valid timing/accuracy role")
    resolved["exclusive_inference"] = role == "timing"
    resolved.pop("fingerprint", None)
    resolved["fingerprint"] = canonical_json_sha256(resolved)
    return resolved


__all__ = [
    "GIB",
    "POLICY_PATH",
    "load_execution_policy",
    "materialize_inference_run_resources",
    "inference_task_execution_plan",
    "resolve_execution_class",
    "resolve_workload_execution_plan",
]
