"""Machine-independent execution capacity detection and immutable runtime plans.

Scientific profiles describe *what* is measured.  This module describes the
resources available to execute that design on one particular host.  Keeping
those concerns separate lets the same reduced or full experiment run on a
laptop, a workstation, or a compute node without changing its scientific
identity.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256
from .config import BenchmarkConfigError
from .safe_io import path_below_no_symlinks, write_idempotent_json

_GIB = 1024**3


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None


def _positive_integer(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _nonnegative_integer(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value >= 0 else None


def _proc_memory(proc_root: Path) -> tuple[int, int]:
    values: dict[str, int] = {}
    text = _read_text(proc_root / "meminfo")
    if text is not None:
        for line in text.splitlines():
            name, separator, remainder = line.partition(":")
            if not separator:
                continue
            fields = remainder.split()
            if not fields:
                continue
            try:
                value = int(fields[0])
            except ValueError:
                continue
            multiplier = 1024 if len(fields) > 1 and fields[1].lower() == "kb" else 1
            values[name] = value * multiplier
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    if total <= 0 or available <= 0:
        raise BenchmarkConfigError("Cannot determine positive host memory capacity from /proc")
    return total, min(total, available)


def _parse_cpu_list(text: str | None) -> tuple[int, ...]:
    if not text:
        return ()
    cpus: set[int] = set()
    try:
        for field in text.split(","):
            bounds = field.strip().split("-", maxsplit=1)
            first = int(bounds[0])
            last = int(bounds[-1])
            if first < 0 or last < first:
                raise ValueError
            cpus.update(range(first, last + 1))
    except ValueError as exc:
        raise BenchmarkConfigError(f"Invalid cgroup CPU set: {text!r}") from exc
    return tuple(sorted(cpus))


def _v2_cgroup_directories(cgroup_root: Path, proc_root: Path) -> tuple[Path, ...]:
    """Return the current cgroup-v2 directory followed by its ancestors."""

    root = cgroup_root.absolute()
    membership = _read_text(proc_root / "self/cgroup")
    relative = Path(".")
    if membership:
        for line in membership.splitlines():
            fields = line.split(":", maxsplit=2)
            if len(fields) != 3:
                continue
            hierarchy, controllers, path = fields
            if hierarchy == "0" and not controllers:
                relative = Path(path.lstrip("/"))
                break
    leaf = (root / relative).absolute()
    if not leaf.is_relative_to(root):
        raise BenchmarkConfigError("Current cgroup escapes the cgroup filesystem")
    directories: list[Path] = []
    current = leaf
    while True:
        directories.append(current)
        if current == root:
            break
        current = current.parent
    return tuple(directories)


def _cgroup_v2_limits(
    cgroup_root: Path,
    proc_root: Path,
) -> tuple[float | None, int | None, int | None, tuple[int, ...]]:
    directories = _v2_cgroup_directories(cgroup_root, proc_root)
    quota_limits: list[float] = []
    memory_states: list[tuple[int, int, int]] = []
    cpuset: tuple[int, ...] = ()
    for directory in directories:
        cpu_max = _read_text(directory / "cpu.max")
        if cpu_max:
            fields = cpu_max.split()
            if len(fields) == 2 and fields[0] != "max":
                quota = _positive_integer(fields[0])
                period = _positive_integer(fields[1])
                if quota is not None and period is not None:
                    quota_limits.append(quota / period)
        memory_max = _read_text(directory / "memory.max")
        if memory_max and memory_max != "max":
            limit = _nonnegative_integer(memory_max)
            current = _nonnegative_integer(_read_text(directory / "memory.current"))
            if limit is None or current is None:
                raise BenchmarkConfigError(
                    f"Cannot determine the finite memory allowance for cgroup {directory}"
                )
            memory_states.append((limit, current, max(0, limit - current)))
        if not cpuset:
            cpuset = _parse_cpu_list(
                _read_text(directory / "cpuset.cpus.effective")
                or _read_text(directory / "cpuset.cpus")
            )
    if memory_states:
        limit, current, _remaining = min(memory_states, key=lambda state: state[2])
    else:
        limit = None
        current = _nonnegative_integer(_read_text(directories[0] / "memory.current"))
    return (
        min(quota_limits) if quota_limits else None,
        limit,
        current,
        cpuset,
    )


def _v1_controller_root(cgroup_root: Path, controller: str) -> Path:
    """Resolve a v1 controller mount, including comma-combined mounts."""

    direct = cgroup_root / controller
    if direct.is_dir():
        return direct.absolute()
    try:
        combined = sorted(
            (
                child.absolute()
                for child in cgroup_root.iterdir()
                if child.is_dir() and controller in child.name.split(",")
            ),
            key=lambda path: path.name,
        )
    except OSError as exc:
        raise BenchmarkConfigError(f"Cannot inspect cgroup-v1 mounts below {cgroup_root}") from exc
    if len(combined) > 1:
        names = ", ".join(path.name for path in combined)
        raise BenchmarkConfigError(
            f"Ambiguous cgroup-v1 mounts for controller {controller!r}: {names}"
        )
    return combined[0] if combined else cgroup_root.absolute()


def _cgroup_directories(controller_root: Path, relative: str) -> tuple[Path, ...]:
    """Return a cgroup leaf followed by every ancestor inside its mount."""

    root = controller_root.absolute()
    relative_path = Path(relative.lstrip("/"))
    if ".." in relative_path.parts:
        raise BenchmarkConfigError("Current cgroup escapes the cgroup filesystem")
    leaf = (root / relative_path).absolute()
    if not leaf.is_relative_to(root):
        raise BenchmarkConfigError("Current cgroup escapes the cgroup filesystem")
    directories: list[Path] = []
    current = leaf
    while True:
        directories.append(current)
        if current == root:
            break
        current = current.parent
    return tuple(directories)


def _cgroup_v1_limits(
    cgroup_root: Path,
    proc_root: Path,
) -> tuple[float | None, int | None, int | None, tuple[int, ...]]:
    membership = _read_text(proc_root / "self/cgroup") or ""
    paths: dict[str, str] = {}
    for line in membership.splitlines():
        fields = line.split(":", maxsplit=2)
        if len(fields) != 3:
            continue
        for controller in fields[1].split(","):
            if controller:
                paths[controller] = fields[2]

    cpu_directories = _cgroup_directories(
        _v1_controller_root(cgroup_root, "cpu"), paths.get("cpu", "/")
    )
    quota_limits: list[float] = []
    for directory in cpu_directories:
        quota = _positive_integer(_read_text(directory / "cpu.cfs_quota_us"))
        period = _positive_integer(_read_text(directory / "cpu.cfs_period_us"))
        if quota is not None and period is not None:
            quota_limits.append(quota / period)

    memory_directories = _cgroup_directories(
        _v1_controller_root(cgroup_root, "memory"), paths.get("memory", "/")
    )
    memory_states: list[tuple[int, int, int]] = []
    first_memory_current: int | None = None
    for directory in memory_directories:
        current = _nonnegative_integer(_read_text(directory / "memory.usage_in_bytes"))
        if first_memory_current is None and current is not None:
            first_memory_current = current
        limit = _positive_integer(_read_text(directory / "memory.limit_in_bytes"))
        if limit is not None:
            if current is None:
                raise BenchmarkConfigError(
                    f"Cannot determine the finite memory allowance for cgroup {directory}"
                )
            memory_states.append((limit, current, max(0, limit - current)))
    if memory_states:
        memory_limit, memory_current, _remaining = min(
            memory_states, key=lambda state: state[2]
        )
    else:
        memory_limit = None
        memory_current = first_memory_current

    cpuset_directories = _cgroup_directories(
        _v1_controller_root(cgroup_root, "cpuset"), paths.get("cpuset", "/")
    )
    cpuset: tuple[int, ...] = ()
    for directory in cpuset_directories:
        cpuset = _parse_cpu_list(
            _read_text(directory / "cpuset.effective_cpus")
            or _read_text(directory / "cpuset.cpus.effective")
            or _read_text(directory / "cpuset.cpus")
        )
        if cpuset:
            break
    return (
        min(quota_limits) if quota_limits else None,
        memory_limit,
        memory_current,
        cpuset,
    )


def _physical_topology(
    sys_cpu_root: Path,
    allowed_cpus: Sequence[int],
) -> tuple[int, int, tuple[int, ...]]:
    cores: dict[tuple[int, int], list[int]] = {}
    for cpu in allowed_cpus:
        topology = sys_cpu_root / f"cpu{cpu}" / "topology"
        package = _read_text(topology / "physical_package_id")
        core = _read_text(topology / "core_id")
        if package is None or core is None:
            return 0, 0, tuple(allowed_cpus)
        try:
            identity = (int(package), int(core))
        except ValueError:
            return 0, 0, tuple(allowed_cpus)
        cores.setdefault(identity, []).append(cpu)
    if not cores:
        return 0, 0, tuple(allowed_cpus)
    for siblings in cores.values():
        siblings.sort()
    allocation_order = tuple(
        siblings[sibling_index]
        for sibling_index in range(max(len(siblings) for siblings in cores.values()))
        for _identity, siblings in sorted(cores.items())
        if sibling_index < len(siblings)
    )
    packages = {package for package, _core in cores}
    return len(cores), len(packages), allocation_order


def detect_host_capacity(
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    sys_cpu_root: Path = Path("/sys/devices/system/cpu"),
    affinity_cpus: Sequence[int] | None = None,
    host_logical_cpus: int | None = None,
    host_total_memory_bytes: int | None = None,
    host_available_memory_bytes: int | None = None,
    workspace_path: Path | None = None,
) -> dict[str, Any]:
    """Observe the resources this process can actually use.

    CPU affinity and cgroup quotas are both honoured.  RAM capacity uses the
    smaller of host-available memory and the remaining cgroup allowance, so the
    scheduler never mistakes node-wide capacity for container capacity.
    """

    if affinity_cpus is None:
        if hasattr(os, "sched_getaffinity"):
            affinity_cpus = sorted(os.sched_getaffinity(0))
        else:
            count = os.cpu_count() or 1
            affinity_cpus = tuple(range(count))
    affinity = tuple(sorted(set(int(cpu) for cpu in affinity_cpus)))
    if not affinity or affinity[0] < 0:
        raise BenchmarkConfigError("CPU affinity must contain non-negative logical CPU identifiers")

    logical = int(host_logical_cpus or os.cpu_count() or len(affinity))
    if logical < 1:
        raise BenchmarkConfigError("Host logical CPU count must be positive")
    if host_total_memory_bytes is None or host_available_memory_bytes is None:
        observed_total, observed_available = _proc_memory(proc_root)
        host_total_memory_bytes = host_total_memory_bytes or observed_total
        host_available_memory_bytes = host_available_memory_bytes or observed_available
    host_total = int(host_total_memory_bytes)
    host_available = int(host_available_memory_bytes)
    if host_total < 1 or host_available < 1 or host_available > host_total:
        raise BenchmarkConfigError("Host memory observations are inconsistent")

    quota: float | None = None
    memory_limit: int | None = None
    memory_current: int | None = None
    cpuset: tuple[int, ...] = ()
    if (cgroup_root / "cgroup.controllers").is_file() or (cgroup_root / "cpu.max").is_file():
        quota, memory_limit, memory_current, cpuset = _cgroup_v2_limits(cgroup_root, proc_root)
    elif (proc_root / "self/cgroup").is_file():
        quota, memory_limit, memory_current, cpuset = _cgroup_v1_limits(cgroup_root, proc_root)

    # Cgroup v1 represents an unlimited memory controller with a page-aligned
    # value near LONG_MAX. A limit at or above physical host RAM cannot further
    # constrain this process, so expose it as unbounded instead of publishing a
    # misleading multi-exabyte quota.
    if memory_limit is not None and memory_limit >= host_total:
        memory_limit = None

    allowed = affinity
    if cpuset:
        intersection = tuple(sorted(set(affinity).intersection(cpuset)))
        if not intersection:
            raise BenchmarkConfigError("CPU affinity and cgroup CPU set do not overlap")
        allowed = intersection
    quota_tokens = max(1, math.floor(quota)) if quota is not None else len(allowed)
    effective_cpus = min(len(allowed), quota_tokens)

    cgroup_remaining: int | None = None
    if memory_limit is not None:
        used = memory_current or 0
        cgroup_remaining = max(0, memory_limit - used)
    effective_memory = min(
        host_available,
        cgroup_remaining if cgroup_remaining is not None else host_available,
    )
    if effective_memory < 1:
        raise BenchmarkConfigError("No positive memory capacity remains in the current cgroup")

    physical_cores, packages, allocation_order = _physical_topology(sys_cpu_root, allowed)
    storage = shutil.disk_usage(workspace_path or Path.cwd())
    payload: dict[str, Any] = {
        "logical_cpus_host": logical,
        "affinity_cpus": list(affinity),
        "allowed_cpus": list(allowed),
        "cpu_allocation_order": list(allocation_order),
        "physical_cores_allowed": physical_cores or None,
        "cpu_packages_allowed": packages or None,
        "cgroup_cpu_quota_cores": quota,
        "effective_cpu_threads": effective_cpus,
        "host_memory_total_bytes": host_total,
        "host_memory_available_bytes": host_available,
        "cgroup_memory_limit_bytes": memory_limit,
        "cgroup_memory_current_bytes": memory_current,
        "cgroup_memory_remaining_bytes": cgroup_remaining,
        "effective_memory_bytes": effective_memory,
        "workspace_storage_total_bytes": storage.total,
        "workspace_storage_free_bytes": storage.free,
    }
    return payload


def resolve_runtime_plan(
    detected: Mapping[str, Any],
    *,
    max_cores: int | None = None,
    max_ram_gb: float | None = None,
    reserve_cores: int = 0,
    reserve_ram_gb: float = 0.0,
) -> dict[str, Any]:
    """Resolve optional operator ceilings against observed machine capacity."""

    if max_cores is not None and (
        not isinstance(max_cores, int) or isinstance(max_cores, bool) or max_cores < 1
    ):
        raise BenchmarkConfigError("CPU ceiling (--cores) must be a positive integer")
    if max_ram_gb is not None and (
        not isinstance(max_ram_gb, (int, float))
        or isinstance(max_ram_gb, bool)
        or not math.isfinite(max_ram_gb)
        or max_ram_gb <= 0
    ):
        raise BenchmarkConfigError("RAM ceiling (--ram-gb) must be finite and positive")
    if not isinstance(reserve_cores, int) or isinstance(reserve_cores, bool) or reserve_cores < 0:
        raise BenchmarkConfigError("--reserve-cores must be a non-negative integer")
    if (
        not isinstance(reserve_ram_gb, (int, float))
        or isinstance(reserve_ram_gb, bool)
        or not math.isfinite(reserve_ram_gb)
        or reserve_ram_gb < 0
    ):
        raise BenchmarkConfigError("--reserve-ram-gb must be finite and non-negative")

    detected_cpus = detected.get("effective_cpu_threads")
    detected_memory = detected.get("effective_memory_bytes")
    if not isinstance(detected_cpus, int) or isinstance(detected_cpus, bool) or detected_cpus < 1:
        raise BenchmarkConfigError("Detected CPU capacity must be a positive integer")
    if (
        not isinstance(detected_memory, int)
        or isinstance(detected_memory, bool)
        or detected_memory < 1
    ):
        raise BenchmarkConfigError("Detected RAM capacity must be a positive integer")
    cpu_ceiling = min(detected_cpus, max_cores if max_cores is not None else detected_cpus)
    memory_ceiling = min(
        detected_memory,
        int(max_ram_gb * _GIB) if max_ram_gb is not None else detected_memory,
    )
    schedulable_cpus = cpu_ceiling - reserve_cores
    schedulable_memory = memory_ceiling - int(reserve_ram_gb * _GIB)
    if schedulable_cpus < 1:
        raise BenchmarkConfigError("CPU reserve leaves no schedulable thread")
    if schedulable_memory < _GIB:
        raise BenchmarkConfigError("RAM reserve leaves less than 1 GiB schedulable")

    request = {
        "max_cores": max_cores,
        "max_ram_gb": max_ram_gb,
        "reserve_cores": reserve_cores,
        "reserve_ram_gb": reserve_ram_gb,
    }
    body: dict[str, Any] = {
        "schema_version": "1.0",
        "detected": dict(detected),
        "request": request,
        "capacity": {
            "cpu_threads": schedulable_cpus,
            "memory_bytes": schedulable_memory,
            "memory_gib": schedulable_memory / _GIB,
        },
    }
    body["fingerprint"] = canonical_json_sha256(body)
    return body


def validate_runtime_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact persisted runtime-plan contract and its digest."""

    if set(plan) != {"schema_version", "detected", "request", "capacity", "fingerprint"}:
        raise BenchmarkConfigError("Runtime plan has unsupported or missing fields")
    if plan.get("schema_version") != "1.0":
        raise BenchmarkConfigError("Runtime plan schema_version must be '1.0'")
    detected = plan.get("detected")
    request = plan.get("request")
    capacity = plan.get("capacity")
    if not isinstance(detected, Mapping):
        raise BenchmarkConfigError("Runtime plan detected resources must be an object")
    if not isinstance(request, Mapping) or set(request) != {
        "max_cores",
        "max_ram_gb",
        "reserve_cores",
        "reserve_ram_gb",
    }:
        raise BenchmarkConfigError("Runtime plan resource request is invalid")
    if not isinstance(capacity, Mapping) or set(capacity) != {
        "cpu_threads",
        "memory_bytes",
        "memory_gib",
    }:
        raise BenchmarkConfigError("Runtime plan capacity is invalid")
    cpu_threads = capacity.get("cpu_threads")
    memory_bytes = capacity.get("memory_bytes")
    memory_gib = capacity.get("memory_gib")
    if not isinstance(cpu_threads, int) or isinstance(cpu_threads, bool) or cpu_threads < 1:
        raise BenchmarkConfigError("Runtime plan CPU capacity must be a positive integer")
    if not isinstance(memory_bytes, int) or isinstance(memory_bytes, bool) or memory_bytes < _GIB:
        raise BenchmarkConfigError("Runtime plan RAM capacity must be at least 1 GiB")
    if (
        not isinstance(memory_gib, (int, float))
        or isinstance(memory_gib, bool)
        or not math.isclose(float(memory_gib), memory_bytes / _GIB, rel_tol=0, abs_tol=1e-12)
    ):
        raise BenchmarkConfigError("Runtime plan RAM representations are inconsistent")
    fingerprint = plan.get("fingerprint")
    unsigned = {key: value for key, value in plan.items() if key != "fingerprint"}
    if fingerprint != canonical_json_sha256(unsigned):
        raise BenchmarkConfigError("Runtime plan fingerprint is invalid")
    try:
        expected = resolve_runtime_plan(
            detected,
            max_cores=request["max_cores"],
            max_ram_gb=request["max_ram_gb"],
            reserve_cores=request["reserve_cores"],
            reserve_ram_gb=request["reserve_ram_gb"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkConfigError("Runtime plan inputs cannot reproduce its capacity") from exc
    if capacity != expected["capacity"]:
        raise BenchmarkConfigError("Runtime plan capacity differs from its detected resources")
    return dict(plan)


def _cpu_identifiers(value: Any, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0 for cpu in value)
    ):
        raise BenchmarkConfigError(f"{label} must be a list of non-negative CPU identifiers")
    identifiers = tuple(value)
    if len(set(identifiers)) != len(identifiers):
        raise BenchmarkConfigError(f"{label} must not contain duplicate CPU identifiers")
    return identifiers


def _validate_runtime_plan_coverage(
    plan: Mapping[str, Any], current_detected: Mapping[str, Any]
) -> None:
    """Prove that current CPU/RAM constraints still cover a frozen plan."""

    request = plan["request"]
    try:
        current_plan = resolve_runtime_plan(
            current_detected,
            max_cores=request["max_cores"],
            max_ram_gb=request["max_ram_gb"],
            reserve_cores=request["reserve_cores"],
            reserve_ram_gb=request["reserve_ram_gb"],
        )
    except (KeyError, TypeError, ValueError, BenchmarkConfigError) as exc:
        raise BenchmarkConfigError(
            "Current host/cgroup capacity cannot satisfy the frozen runtime-plan request"
        ) from exc

    frozen_capacity = plan["capacity"]
    current_capacity = current_plan["capacity"]
    if current_capacity["cpu_threads"] < frozen_capacity["cpu_threads"]:
        raise BenchmarkConfigError(
            "Current CPU capacity no longer covers the frozen runtime plan: "
            f"{current_capacity['cpu_threads']} available after reserves, "
            f"{frozen_capacity['cpu_threads']} required"
        )
    # Available RAM and cgroup usage fluctuate as this very process resumes the
    # workflow.  They are appropriate when initially sizing the plan, but not
    # for deciding whether its allocation boundary was reduced.  On real host
    # observations compare the stable host/cgroup ceilings; minimal injected
    # observations used by callers/tests fall back to effective capacity.
    stable_memory_limits = [
        value
        for value in (
            current_detected.get("host_memory_total_bytes"),
            current_detected.get("cgroup_memory_limit_bytes"),
        )
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    if stable_memory_limits:
        current_memory = min(stable_memory_limits)
        if request["max_ram_gb"] is not None:
            current_memory = min(current_memory, int(request["max_ram_gb"] * _GIB))
        current_memory -= int(request["reserve_ram_gb"] * _GIB)
    else:
        current_memory = current_capacity["memory_bytes"]
    if current_memory < frozen_capacity["memory_bytes"]:
        raise BenchmarkConfigError(
            "Current RAM capacity no longer covers the frozen runtime plan: "
            f"{current_memory} bytes available after reserves, "
            f"{frozen_capacity['memory_bytes']} required"
        )

    frozen_detected = plan["detected"]
    frozen_order_raw = frozen_detected.get("cpu_allocation_order")
    current_allowed_raw = current_detected.get("allowed_cpus")
    if frozen_order_raw is None and current_allowed_raw is None:
        return
    if frozen_order_raw is None or current_allowed_raw is None:
        raise BenchmarkConfigError(
            "Current CPU affinity/cgroup CPU set cannot be compared with the frozen runtime plan"
        )
    frozen_order = _cpu_identifiers(
        frozen_order_raw, "Frozen runtime-plan CPU allocation order"
    )
    current_allowed = set(
        _cpu_identifiers(current_allowed_raw, "Current allowed CPU identifiers")
    )
    required_count = int(frozen_capacity["cpu_threads"])
    if len(frozen_order) < required_count:
        raise BenchmarkConfigError(
            "Frozen runtime plan has too few CPU identifiers for its declared capacity"
        )
    missing = sorted(set(frozen_order[:required_count]).difference(current_allowed))
    if missing:
        raise BenchmarkConfigError(
            "Current CPU affinity/cgroup CPU set no longer covers the frozen runtime plan; "
            f"unavailable CPU identifiers: {missing}"
        )


def load_or_create_runtime_plan(
    path: Path,
    *,
    boundary: Path,
    max_cores: int | None = None,
    max_ram_gb: float | None = None,
    reserve_cores: int = 0,
    reserve_ram_gb: float = 0.0,
    detected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one immutable plan, or reuse it only for the identical request."""

    safe_path = path_below_no_symlinks(path, boundary, "Runtime plan")
    request = {
        "max_cores": max_cores,
        "max_ram_gb": max_ram_gb,
        "reserve_cores": reserve_cores,
        "reserve_ram_gb": reserve_ram_gb,
    }
    if safe_path.exists():
        try:
            payload = json.loads(safe_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkConfigError(f"Cannot read runtime plan {safe_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise BenchmarkConfigError(f"Invalid runtime plan: {safe_path}")
        try:
            validated = validate_runtime_plan(payload)
        except BenchmarkConfigError as exc:
            raise BenchmarkConfigError(f"Invalid runtime plan {safe_path}: {exc}") from exc
        if validated["request"] != request:
            raise BenchmarkConfigError(
                "Existing runtime plan was created with different resource options"
            )
        current_detected = (
            detected
            if detected is not None
            else detect_host_capacity(workspace_path=boundary)
        )
        _validate_runtime_plan_coverage(validated, current_detected)
        return validated

    plan = resolve_runtime_plan(
        detected if detected is not None else detect_host_capacity(workspace_path=boundary),
        max_cores=max_cores,
        max_ram_gb=max_ram_gb,
        reserve_cores=reserve_cores,
        reserve_ram_gb=reserve_ram_gb,
    )
    write_idempotent_json(
        safe_path,
        plan,
        "Runtime plan",
        boundary=boundary,
    )
    return plan


__all__ = [
    "detect_host_capacity",
    "load_or_create_runtime_plan",
    "resolve_runtime_plan",
    "validate_runtime_plan",
]
