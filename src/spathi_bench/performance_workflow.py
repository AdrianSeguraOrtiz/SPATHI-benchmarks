"""Resumable, sequential execution of the formal computational-performance study."""

from __future__ import annotations

import copy
import csv
import math
import os
import shutil
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    load_profile,
    validate_json,
)
from .manifests import unique_direct_manifest
from .performance import (
    DESIGN_PATH,
    load_performance_design,
    materialize_performance_protocol,
    validate_performance_catalog_threading,
)
from .pipeline import build_spathi_image, generate_data, preflight
from .provenance import observe_benchmark_repository
from .report_publication import publish_workflow_report
from .reporting import render_performance_report
from .resource_plan import detect_host_capacity, load_or_create_runtime_plan
from .resource_scheduler import host_execution_lease
from .runtime import (
    CommandExecutionError,
    andrea_command,
    andrea_host_environment,
    benchmark_host_environment,
    catalog_image_lease,
    docker_image_details,
    docker_runtime_environment,
)
from .safe_io import atomic_write_text, path_below_no_symlinks, write_idempotent_json
from .selected_policy import completed_selected_config
from .state import load_state, run_lock
from .study_context import bind_phase_identity, phase_flow_id
from .tool_contract import oom_observed

_IDENTITY_INPUTS = (
    "experiments/v1/performance/design.json",
    "experiments/v1/execution/policy.json",
    "schemas/performance-design.schema.json",
    "schemas/performance-materialization.schema.json",
    "schemas/performance-results.schema.json",
    "src/spathi_bench/performance.py",
    "src/spathi_bench/performance_workflow.py",
    "src/spathi_bench/reporting/performance.py",
    "src/spathi_bench/reporting/templates/performance.html.j2",
    "src/spathi_bench/reporting/assets/performance.js",
)

_STABLE_HOST_FIELDS = (
    "logical_cpus_host",
    "affinity_cpus",
    "allowed_cpus",
    "cpu_allocation_order",
    "physical_cores_allowed",
    "cpu_packages_allowed",
    "cgroup_cpu_quota_cores",
    "host_memory_total_bytes",
    "cgroup_memory_limit_bytes",
)

_ANDREA_E2E_MEMORY_SOURCE = "child_container_cgroup_peaks"
_ANDREA_E2E_MEMORY_SEMANTICS = (
    "overlap_upper_bound_from_physical_container_peaks_excludes_andrea_host_process"
)


def _relative(repo_root: Path, path: Path, label: str) -> str:
    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    safe = path_below_no_symlinks(path, root, label)
    return safe.relative_to(root).as_posix()


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _validate_performance_host_identity(
    runtime_plan: Mapping[str, Any], detected: Mapping[str, Any]
) -> None:
    """Reject resumption on a different CPU or memory-limit topology.

    Free RAM, current cgroup use and free disk naturally fluctuate and therefore do
    not define host identity. The immutable plan still fixes the actual benchmark
    budget; stable topology and limits must remain identical across every task.
    """

    frozen = runtime_plan.get("detected")
    if not isinstance(frozen, Mapping):
        raise BenchmarkConfigError("Performance runtime plan has no detected host")
    changed = [field for field in _STABLE_HOST_FIELDS if frozen.get(field) != detected.get(field)]
    if changed:
        raise BenchmarkConfigError(
            "Performance cannot resume on a different host or cgroup topology; changed fields: "
            + ", ".join(changed)
        )


def _finite_nonnegative(value: Any, label: str, *, nullable: bool = False) -> float | None:
    if nullable and value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise BenchmarkConfigError(f"{label} must be a finite non-negative number")
    return float(value)


def _nonnegative_integer(value: Any, label: str, *, nullable: bool = False) -> int | None:
    if nullable and value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BenchmarkConfigError(f"{label} must be a non-negative integer")
    return value


def _resolve_report_member(report_path: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise BenchmarkConfigError(f"Performance run report has no {label}")
    raw_path = Path(raw)
    candidate = raw_path if raw_path.is_absolute() else report_path.parent / raw_path
    return path_below_no_symlinks(
        candidate,
        report_path.parent,
        f"Performance {label}",
        require_file=True,
    )


def _canonical_network_sha256(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise BenchmarkConfigError("Performance network CSV has no header")
        required = {"source", "target", "score", "sign", "evidence", "context"}
        if not required.issubset(reader.fieldnames):
            raise BenchmarkConfigError("Performance network CSV has an invalid schema")
        rows = [{field: str(row.get(field, "")) for field in reader.fieldnames} for row in reader]
    rows.sort(key=lambda row: tuple(row[field] for field in reader.fieldnames or ()))
    return canonical_json_sha256({"columns": reader.fieldnames, "rows": rows})


def _validate_affinity(measurement: Mapping[str, Any], task: Mapping[str, Any]) -> None:
    assigned = measurement.get("assigned_resources")
    if not isinstance(assigned, Mapping):
        raise BenchmarkConfigError("Performance telemetry has no assigned resources")
    expected = list(task["cpuset_cpus"])
    if (
        measurement.get("scope") != "andrea_execution"
        or assigned.get("cpuset_cpus_requested_sets") != [expected]
        or assigned.get("cpuset_cpus_effective_sets") != [expected]
        or assigned.get("threads_peak") != task["threads"]
        or assigned.get("ram_limit_peak_bytes") != round(float(task["ram_gb"]) * 1024**3)
    ):
        raise BenchmarkConfigError("End-to-end performance resources differ from their task")


def _measurement_record(
    *,
    repo_root: Path,
    task: Mapping[str, Any],
    dataset: Mapping[str, Any],
    request: Mapping[str, Any],
    report_path: Path,
    task_root: Path,
    command_return_code: int,
) -> dict[str, Any]:
    report = load_json_object(report_path, "performance run report")
    run = request["request"]
    run_id = str(run["run_id"])
    tools = report.get("tools")
    if (
        report.get("status") != "executed"
        or report.get("output_profile") != "full"
        or report.get("run_id") != run_id
        or not isinstance(tools, Mapping)
        or tools.get("selected") != [run_id]
        or (report.get("dataset") or {}).get("id") != dataset["dataset_id"]
    ):
        raise BenchmarkConfigError("Performance run report has the wrong execution identity")
    expected_origin = "catalog" if task["origin"] == "catalog" else "custom"
    if (tools.get("tool_origins") or {}).get(run_id) != expected_origin:
        raise BenchmarkConfigError("Performance run report has the wrong tool origin")
    results = tools.get("results")
    if not isinstance(results, Mapping) or not isinstance(results.get(run_id), Mapping):
        raise BenchmarkConfigError("Performance run report has no logical result")
    result = results[run_id]
    execution = report.get("execution")
    measurement = execution.get("measurement") if isinstance(execution, Mapping) else None
    if not isinstance(measurement, Mapping):
        raise BenchmarkConfigError("Performance report has no end-to-end execution measurement")
    if measurement.get("schema_version") != "1.0" or measurement.get("scope") != "andrea_execution":
        raise BenchmarkConfigError("Performance requires canonical end-to-end telemetry")
    wall_time = _finite_nonnegative(measurement.get("wall_time_seconds"), "wall time")
    cpu_time = _finite_nonnegative(measurement.get("cpu_time_seconds"), "CPU time")
    peak_memory = _nonnegative_integer(
        measurement.get("peak_memory_bytes"), "container-memory upper bound"
    )
    _nonnegative_integer(measurement.get("output_bytes"), "output bytes")
    if wall_time == 0 or cpu_time is None or peak_memory is None or peak_memory == 0:
        raise BenchmarkConfigError("Performance telemetry contains no measurable execution")
    _validate_affinity(measurement, task)
    telemetry = measurement.get("telemetry")
    if (
        not isinstance(telemetry, Mapping)
        or telemetry.get("status") != "complete"
        or telemetry.get("wall_source") != "andrea_monotonic_clock"
        or telemetry.get("wall_semantics")
        != "andrea_execution_end_to_end_including_input_verification_preparation_"
        "scheduling_merging_and_requested_output_materialization"
    ):
        raise BenchmarkConfigError("Performance requires complete end-to-end telemetry")
    if (
        telemetry.get("memory_source") != _ANDREA_E2E_MEMORY_SOURCE
        or telemetry.get("memory_semantics") != _ANDREA_E2E_MEMORY_SEMANTICS
    ):
        raise BenchmarkConfigError(
            "Performance requires ANDREA's container-memory overlap upper bound"
        )
    tool_status = (tools.get("status_by_tool") or {}).get(run_id)
    completed = tool_status in {"completed", "completed_with_warnings"}
    if completed:
        if tools.get("failed"):
            raise BenchmarkConfigError("Completed performance run reports a tool failure")
        network_path = _resolve_report_member(
            report_path,
            (report.get("outputs") or {}).get("merged_network_normalized"),
            "normalized network",
        )
        network_sha256 = _canonical_network_sha256(network_path)
        status = "completed"
    else:
        failures = tools.get("failed")
        if not isinstance(failures, Mapping) or run_id not in failures:
            raise BenchmarkConfigError("Failed performance run has no attributable tool failure")
        network_sha256 = None
        status = "failed"
    if (command_return_code == 0) != completed:
        raise BenchmarkConfigError(
            "ANDREA command status and attributable performance result disagree"
        )
    dataset_report = report.get("dataset") or {}
    record = {
        "schema_version": "1.0",
        "task_id": task["task_id"],
        "sequence_index": task["sequence_index"],
        "phase": task["phase"],
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "generation_job_id": task["generation_job_id"],
            "replicate_index": task["replicate_index"],
            "size_class": task["size_class"],
            "seed": dataset.get("seed"),
            "cells": dataset_report.get("columns"),
            "genes": dataset_report.get("genes"),
        },
        "configuration_id": task["configuration_id"],
        "tool_id": task["tool_id"],
        "origin": task["origin"],
        "threads": task["threads"],
        "ram_gb": float(task["ram_gb"]),
        "cpuset_cpus": list(task["cpuset_cpus"]),
        "system_repetition": task["system_repetition"],
        "algorithm_seed": request["algorithm_seed"],
        "status": status,
        "tool_status": tool_status,
        "command_return_code": command_return_code,
        "measurement": copy.deepcopy(dict(measurement)),
        "io": {
            "status": "unavailable",
            "read_bytes": None,
            "write_bytes": None,
            "reason": "ANDREA v1 telemetry records output bytes but not block-I/O counters.",
        },
        "oom_observed": oom_observed(result),
        "network_sha256": network_sha256,
        "run_report": {
            "path": _relative(repo_root, report_path, "performance run report"),
            "sha256": sha256_file(report_path),
        },
        "output": {
            "path": _relative(repo_root, task_root / "output", "performance output"),
            "tree_bytes": _tree_bytes(task_root / "output"),
        },
    }
    return record


def _validated_existing_record(
    *,
    repo_root: Path,
    task: Mapping[str, Any],
    dataset: Mapping[str, Any],
    request: Mapping[str, Any],
    task_root: Path,
) -> dict[str, Any]:
    record_path = path_below_no_symlinks(
        task_root / "measurement.json",
        task_root,
        "Performance task measurement",
        require_file=True,
    )
    recorded = load_json_object(record_path, "performance task measurement")
    report_path = unique_direct_manifest(task_root / "output", "run_report.json")
    observed = _measurement_record(
        repo_root=repo_root,
        task=task,
        dataset=dataset,
        request=request,
        report_path=report_path,
        task_root=task_root,
        command_return_code=int(recorded.get("command_return_code", -1)),
    )
    if observed != recorded:
        raise BenchmarkConfigError(f"Recorded performance task changed: {task['task_id']}")
    return recorded


def _execute_task(
    *,
    repo_root: Path,
    andrea_repo: Path,
    run_state: Mapping[str, Any],
    materialization: Mapping[str, Any],
    task: Mapping[str, Any],
    dataset: Mapping[str, Any],
    request: Mapping[str, Any],
    task_root: Path,
) -> dict[str, Any]:
    if task_root.exists() or task_root.is_symlink():
        return _validated_existing_record(
            repo_root=repo_root,
            task=task,
            dataset=dataset,
            request=request,
            task_root=task_root,
        )
    task_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{task['task_id']}-", dir=task_root.parent))
    try:
        request_dir = staging / "request"
        request_dir.mkdir()
        run = copy.deepcopy(dict(request["request"]))
        run["resources"] = {
            "threads": int(task["threads"]),
            "ram_gb": float(task["ram_gb"]),
            "cpuset_cpus": list(task["cpuset_cpus"]),
        }
        tools_path = write_idempotent_json(
            request_dir / "tools-params.json",
            {"runs": [run]},
            "performance task tools",
            boundary=staging,
        )
        custom_path: Path | None = None
        frozen_custom_image_id: str | None = None
        if request["custom_tool"] is not None:
            custom_tool = copy.deepcopy(request["custom_tool"])
            image = run_state.get("spathi_image")
            if not isinstance(image, Mapping) or not isinstance(image.get("id"), str):
                raise BenchmarkConfigError("Performance run has no SPATHI image identity")
            # Execute the image digest frozen by benchmark preflight.  The
            # materialized scientific request retains the public tag for
            # provenance, but a mutable tag must never select runtime code.
            frozen_custom_image_id = str(image["id"])
            custom_tool["docker_image"] = frozen_custom_image_id
            custom_path = write_idempotent_json(
                request_dir / "custom-tools.json",
                {"tools": [custom_tool]},
                "performance task custom tool",
                boundary=staging,
            )
        output_root = staging / "output"
        command = [
            "infer-network",
            "execute",
            "--dataset-manifest",
            str(dataset["dataset_manifest"]),
            "--tools-params",
            str(tools_path),
        ]
        if custom_path is not None:
            command.extend(["--custom-tools", str(custom_path)])
        command.extend(
            [
                "--output-dir",
                str(output_root),
                "--max-cores",
                str(task["threads"]),
                "--max-ram-gb",
                str(materialization["execution"]["max_ram_gb"]),
                "--planner",
                str(materialization["execution"]["planner"]),
                "--planner-time-limit-seconds",
                str(materialization["execution"]["planner_time_limit_seconds"]),
                "--output-profile",
                "full",
            ]
        )

        def execute() -> int:
            try:
                andrea_command(
                    command,
                    repo_root=repo_root,
                    andrea_repo=andrea_repo,
                    log_path=staging / "andrea.log",
                )
            except CommandExecutionError as exc:
                return exc.return_code
            return 0

        if task["origin"] == "catalog":
            image_key = f"tool:{task['tool_id']}"
            image_lock = load_image_lock(repo_root)["catalog_images"][image_key]
            observed = run_state.get("catalog_images", {}).get(image_key)
            if not isinstance(observed, Mapping) or not isinstance(observed.get("id"), str):
                raise BenchmarkConfigError(f"Performance preflight did not pin {image_key}")
            with catalog_image_lease(image_lock, expected_id=str(observed["id"])):
                return_code = execute()
        else:
            if frozen_custom_image_id is None:
                raise BenchmarkConfigError("Performance run has no SPATHI image identity")
            current = docker_image_details(frozen_custom_image_id)
            if current is None or current.get("id") != frozen_custom_image_id:
                raise BenchmarkConfigError("Performance SPATHI image changed after preflight")
            return_code = execute()
        report_path = unique_direct_manifest(output_root, "run_report.json")
        provisional_root = staging
        record = _measurement_record(
            repo_root=repo_root,
            task=task,
            dataset=dataset,
            request=request,
            report_path=report_path,
            task_root=provisional_root,
            command_return_code=return_code,
        )
        # References are computed for the final immutable location, never for staging.
        record["run_report"] = {
            "path": _relative(
                repo_root,
                task_root / report_path.relative_to(staging),
                "final performance run report",
            ),
            "sha256": sha256_file(report_path),
        }
        record["output"]["path"] = _relative(
            repo_root,
            task_root / "output",
            "final performance output",
        )
        write_idempotent_json(
            staging / "measurement.json",
            record,
            "performance task measurement",
            boundary=staging,
        )
        os.rename(staging, task_root)
        return record
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def summarize_performance_measurements(
    measurements: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate measured repetitions without treating them as algorithm replicates."""

    measured = [item for item in measurements if item["phase"] == "measurement"]
    equivalence_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in measured:
        equivalence_groups[(row["dataset"]["dataset_id"], row["configuration_id"])].append(row)
    equivalence = []
    equivalence_by_condition: dict[tuple[str, str], bool | None] = {}
    for (dataset_id, configuration_id), rows in sorted(equivalence_groups.items()):
        complete = all(row["status"] == "completed" for row in rows)
        hashes = {row["network_sha256"] for row in rows if row["status"] == "completed"}
        equivalent = len(hashes) == 1 if complete and rows else None
        equivalence_by_condition[(dataset_id, configuration_id)] = equivalent
        equivalence.append(
            {
                "dataset_id": dataset_id,
                "configuration_id": configuration_id,
                "runs_expected": len(rows),
                "runs_completed": sum(row["status"] == "completed" for row in rows),
                "equivalent_across_threads_and_repetitions": equivalent,
                "distinct_networks": len(hashes),
            }
        )

    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for item in measured:
        key = (
            str(item["dataset"]["dataset_id"]),
            str(item["configuration_id"]),
            int(item["threads"]),
        )
        groups[key].append(item)
    baseline: dict[tuple[str, str], float | None] = {}
    for (dataset_id, configuration_id, threads), rows in groups.items():
        if threads == 1:
            baseline[(dataset_id, configuration_id)] = _median(
                [
                    float(row["measurement"]["wall_time_seconds"])
                    for row in rows
                    if row["status"] == "completed"
                ]
            )
    aggregates: list[dict[str, Any]] = []
    for key in sorted(groups):
        dataset_id, configuration_id, threads = key
        rows = groups[key]
        completed = [row for row in rows if row["status"] == "completed"]
        wall = [float(row["measurement"]["wall_time_seconds"]) for row in completed]
        cpu = [
            float(value)
            for row in completed
            if (value := row["measurement"].get("cpu_time_seconds")) is not None
        ]
        memory = [
            int(value)
            for row in completed
            if (value := row["measurement"].get("peak_memory_bytes")) is not None
        ]
        outputs = [int(row["measurement"]["output_bytes"]) for row in completed]
        wall_median = _median(wall)
        one_thread = baseline.get((dataset_id, configuration_id))
        complete_condition = len(completed) == len(rows)
        hashes = {row["network_sha256"] for row in completed}
        equivalent = len(hashes) == 1 if complete_condition and completed else None
        cross_endpoint_equivalent = equivalence_by_condition[(dataset_id, configuration_id)]
        if cross_endpoint_equivalent is False:
            scaling_metrics_status = "withheld_semantic_drift"
        elif cross_endpoint_equivalent is None or not complete_condition:
            scaling_metrics_status = "withheld_incomplete_execution"
        elif one_thread is None:
            scaling_metrics_status = "withheld_missing_one_thread_baseline"
        else:
            scaling_metrics_status = "published"
        speedup = (
            one_thread / wall_median
            if scaling_metrics_status == "published"
            and wall_median is not None
            and wall_median > 0
            else None
        )
        aggregates.append(
            {
                "dataset_id": dataset_id,
                "size_class": rows[0]["dataset"]["size_class"],
                "configuration_id": configuration_id,
                "tool_id": rows[0]["tool_id"],
                "threads": threads,
                "repetitions_requested": len(rows),
                "repetitions_completed": len(completed),
                "wall_time_median_seconds": wall_median,
                "wall_time_min_seconds": min(wall) if wall else None,
                "wall_time_max_seconds": max(wall) if wall else None,
                "cpu_time_median_seconds": _median(cpu),
                "container_peak_overlap_upper_bound_max_bytes": (
                    max(memory) if memory else None
                ),
                "output_bytes_median": _median([float(value) for value in outputs]),
                "speedup_vs_one_thread": speedup,
                "parallel_efficiency": speedup / threads if speedup is not None else None,
                "scaling_metrics_status": scaling_metrics_status,
                "network_equivalent_within_condition": equivalent,
                "oom_count": sum(bool(row["oom_observed"]) for row in rows),
            }
        )
    summary = {
        "measured_runs_requested": len(measured),
        "measured_runs_completed": sum(item["status"] == "completed" for item in measured),
        "measured_runs_failed": sum(item["status"] == "failed" for item in measured),
        "oom_runs": sum(bool(item["oom_observed"]) for item in measured),
        "all_measured_runs_completed": all(item["status"] == "completed" for item in measured),
        "equivalence_conditions": len(equivalence),
        "equivalence_confirmed": sum(
            item["equivalent_across_threads_and_repetitions"] is True for item in equivalence
        ),
        "equivalence_failed": sum(
            item["equivalent_across_threads_and_repetitions"] is False for item in equivalence
        ),
        "equivalence_unavailable": sum(
            item["equivalent_across_threads_and_repetitions"] is None for item in equivalence
        ),
    }
    summary["equivalence"] = equivalence
    return aggregates, summary


def _write_csv_outputs(
    *,
    workflow_root: Path,
    measurements: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    raw_path = workflow_root / "performance-measurements.csv"
    aggregate_path = workflow_root / "performance-summary.csv"
    raw_fields = (
        "task_id",
        "sequence_index",
        "dataset_id",
        "size_class",
        "configuration_id",
        "tool_id",
        "threads",
        "ram_gb",
        "system_repetition",
        "algorithm_seed",
        "status",
        "wall_time_seconds",
        "cpu_time_seconds",
        "container_peak_overlap_upper_bound_bytes",
        "output_bytes",
        "io_status",
        "read_bytes",
        "write_bytes",
        "oom_observed",
        "network_sha256",
    )
    rows = []
    for item in measurements:
        rows.append(
            {
                "task_id": item["task_id"],
                "sequence_index": item["sequence_index"],
                "dataset_id": item["dataset"]["dataset_id"],
                "size_class": item["dataset"]["size_class"],
                "configuration_id": item["configuration_id"],
                "tool_id": item["tool_id"],
                "threads": item["threads"],
                "ram_gb": item["ram_gb"],
                "system_repetition": item["system_repetition"],
                "algorithm_seed": item["algorithm_seed"],
                "status": item["status"],
                "wall_time_seconds": item["measurement"]["wall_time_seconds"],
                "cpu_time_seconds": item["measurement"].get("cpu_time_seconds"),
                "container_peak_overlap_upper_bound_bytes": item["measurement"].get(
                    "peak_memory_bytes"
                ),
                "output_bytes": item["measurement"]["output_bytes"],
                "io_status": item["io"]["status"],
                "read_bytes": item["io"]["read_bytes"],
                "write_bytes": item["io"]["write_bytes"],
                "oom_observed": item["oom_observed"],
                "network_sha256": item["network_sha256"],
            }
        )

    def render(fields: Sequence[str], values: Sequence[Mapping[str, Any]]) -> str:
        import io

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
        return buffer.getvalue()

    atomic_write_text(raw_path, render(raw_fields, rows), "performance measurement CSV")
    aggregate_fields = tuple(aggregates[0]) if aggregates else ()
    atomic_write_text(
        aggregate_path,
        render(aggregate_fields, aggregates),
        "performance aggregate CSV",
    )
    return raw_path, aggregate_path


def _dataset_bindings(
    state: Mapping[str, Any], materialization: Mapping[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {
        (str(item["generation_job"]), int(item["replicate_index"])): dict(item)
        for item in state.get("datasets", [])
    }
    expected = {
        (str(item["generation_job_id"]), int(item["replicate_index"]))
        for item in materialization["datasets"]
    }
    if not expected.issubset(indexed):
        raise BenchmarkConfigError("Generated performance datasets differ from materialization")
    return {key: indexed[key] for key in expected}


def run_performance_workflow(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    profile: str,
    calibration_run_id: str | None = None,
    max_cores: int | None = None,
    max_ram_gb: float | None = None,
    reserve_cores: int = 0,
    reserve_ram_gb: float = 0.0,
) -> dict[str, Any]:
    """Run or resume the reduced or full performance protocol.

    One host-wide benchmark lease wraps the whole workflow. Internally, the frozen
    task queue is strictly sequential and ANDREA receives one logical run.
    """

    if profile not in {"reduced", "full"}:
        raise BenchmarkConfigError("Performance profile must be 'reduced' or 'full'")
    root = repo_root.resolve()
    work = path_below_no_symlinks(work_root, root / "work", "Performance work root")
    work.mkdir(parents=True, exist_ok=True)
    production_path, production = completed_selected_config(
        repo_root=root,
        work_root=work,
        budget=profile,
        calibration_run_id=calibration_run_id,
    )
    design = load_performance_design(root)
    validate_performance_catalog_threading(andrea_repo, design)
    template_id = f"performance-{profile}"
    template_path = root / f"experiments/v1/profiles/{template_id}.json"
    template = load_profile(root, template_id, require_runnable=True)
    benchmark = observe_benchmark_repository(root, require_clean=True)
    detected = detect_host_capacity(workspace_path=work)
    resource_request = {
        "max_cores": max_cores,
        "max_ram_gb": max_ram_gb,
        "reserve_cores": reserve_cores,
        "reserve_ram_gb": reserve_ram_gb,
    }
    identity = canonical_json_sha256(
        {
            "budget": profile,
            "design_sha256": sha256_file(root / DESIGN_PATH),
            "template_sha256": sha256_file(template_path),
            "selected_config_sha256": sha256_file(production_path),
            "resource_request": resource_request,
            "benchmark_repository": {
                "commit": benchmark["commit"],
                "origin": benchmark["origin"],
            },
            "andrea_environment": andrea_host_environment()["fingerprint"],
            "benchmark_environment": benchmark_host_environment()["fingerprint"],
            "docker_runtime": docker_runtime_environment()["fingerprint"],
            "implementation": {member: sha256_file(root / member) for member in _IDENTITY_INPUTS},
        }
    )
    flow_id = phase_flow_id(identity[:20])
    workflow_root = work / "v1/workflows/performance" / profile / flow_id
    identity = bind_phase_identity(workflow_root, {"fingerprint": identity})["fingerprint"]
    workflow_path = workflow_root / "workflow.json"
    runtime_plan_path = workflow_root / "runtime-plan.json"
    runtime_plan = load_or_create_runtime_plan(
        runtime_plan_path,
        boundary=workflow_root,
        max_cores=max_cores,
        max_ram_gb=max_ram_gb,
        reserve_cores=reserve_cores,
        reserve_ram_gb=reserve_ram_gb,
        detected=detected,
    )
    _validate_performance_host_identity(runtime_plan, detected)
    with host_execution_lease(workload=f"performance:{flow_id}"), run_lock(workflow_root):
        materialized = materialize_performance_protocol(
            repo_root=root,
            budget=profile,
            template_path=template_path,
            template_profile=template,
            production_path=production_path,
            production=production,
            runtime_plan_path=runtime_plan_path,
            runtime_plan=runtime_plan,
            output_dir=workflow_root / "materialized",
        )
        profile_id = str(materialized["profile_id"])
        generation_root = work / "v1/performance-inputs" / profile / flow_id
        common = {"repo_root": root, "profile_id": profile_id, "run_root": generation_root}
        with run_lock(generation_root):
            preflight(
                **common,
                andrea_repo=andrea_repo,
                spathi_repo=spathi_repo,
                profile_file=Path(materialized["profile_path"]),
                runtime_plan=runtime_plan,
            )
            build_spathi_image(**common, spathi_repo=spathi_repo)
            generate_data(**common, andrea_repo=andrea_repo)
        generation_state = load_state(generation_root, profile_id)
        datasets = _dataset_bindings(generation_state, materialized["manifest"])
        requests = materialized["scientific_requests"]
        records = []
        for task in materialized["manifest"]["tasks"]:
            dataset = datasets[(task["generation_job_id"], task["replicate_index"])]
            request = requests[task["configuration_id"]]
            records.append(
                _execute_task(
                    repo_root=root,
                    andrea_repo=andrea_repo,
                    run_state=generation_state,
                    materialization=materialized["manifest"],
                    task=task,
                    dataset=dataset,
                    request=request,
                    task_root=workflow_root / "runs" / task["task_id"],
                )
            )
        warmups = [item for item in records if item["phase"] == "warmup"]
        measurements = [item for item in records if item["phase"] == "measurement"]
        if (
            len(warmups) != design["budgets"][profile]["expected_warmup_runs"]
            or len(measurements) != design["budgets"][profile]["expected_measured_runs"]
        ):
            raise BenchmarkConfigError("Performance execution did not cover its frozen schedule")
        aggregates, summary = summarize_performance_measurements(records)
        results = {
            "schema_version": "1.0",
            "artifact_type": "spathi_performance_results",
            "budget": profile,
            "evidence_role": "publication" if profile == "full" else "operational",
            "claim_scope": "computational_performance_only",
            "status": "complete",
            "design": {
                "path": DESIGN_PATH.as_posix(),
                "sha256": sha256_file(root / DESIGN_PATH),
            },
            "materialization": {
                "path": _relative(
                    root, Path(materialized["manifest_path"]), "performance materialization"
                ),
                "sha256": sha256_file(Path(materialized["manifest_path"])),
                "schedule_sha256": materialized["manifest"]["schedule_sha256"],
            },
            "selected_config": materialized["manifest"]["selected_config"],
            "execution_environment": {
                "runtime_plan": materialized["manifest"]["runtime_plan"],
                "andrea": generation_state["host_environment"],
                "benchmark": generation_state["benchmark_environment"],
                "docker": generation_state["docker_runtime_environment"],
                "andrea_commit": generation_state["repositories"]["observed"]["andrea"]["commit"],
                "spathi_image_id": generation_state["spathi_image"]["id"],
            },
            "protocol": {
                **copy.deepcopy(design["protocol"]),
                "thread_counts": design["budgets"][profile]["thread_counts"],
                "system_repetitions": design["budgets"][profile]["system_repetitions"],
                "warmups_per_condition": design["budgets"][profile]["warmups_per_condition"],
            },
            "summary": summary,
            "aggregates": aggregates,
            "measurements": measurements,
            "warmups": warmups,
            "notes": [
                "Performance results are computational evidence and never select a method or parameter.",
                "Wall time uses ANDREA's execution-level clock from input verification through "
                "requested output materialization; it is not a kernel-only microbenchmark.",
                "All system repetitions use the same algorithm seed; they quantify system noise only.",
                "Warm-up executions are retained for audit but excluded from every aggregate.",
                "Inference caching is disabled and one logical ANDREA execution runs at a time.",
                "Block-I/O counters are unavailable in ANDREA v1; output bytes are measured exactly.",
                "Semantic network hashes test equivalence across thread counts and system repetitions.",
                "Speedup and parallel efficiency are published only when every thread endpoint "
                "and system repetition produced the same canonical normalized network.",
                "The memory value is an upper bound formed from overlapping physical-container "
                "cgroup peaks; it excludes the ANDREA host process and is not whole-process peak RAM.",
            ],
        }
        validate_json(
            results,
            root / "schemas/performance-results.schema.json",
            "performance results",
        )
        results_path = write_idempotent_json(
            workflow_root / "performance-results.json",
            results,
            "performance results",
            boundary=root,
        )
        measurements_csv, summary_csv = _write_csv_outputs(
            workflow_root=workflow_root,
            measurements=measurements,
            aggregates=aggregates,
        )
        report_path = work / "v1/reports" / profile / flow_id / "performance.html"
        publishing_state = {
            "schema_version": "1.0",
            "workflow_type": "performance",
            "profile": profile,
            "flow_id": flow_id,
            "identity": {"fingerprint": identity},
            "status": "publishing",
            "selected_config": _relative(root, production_path, "production config"),
            "selected_config_sha256": sha256_file(production_path),
            "template_profile": _relative(root, template_path, "performance template"),
            "template_profile_sha256": sha256_file(template_path),
            "runtime_profile": _relative(
                root, Path(materialized["profile_path"]), "performance runtime profile"
            ),
            "runtime_profile_sha256": sha256_file(Path(materialized["profile_path"])),
            "materialization": _relative(
                root, Path(materialized["manifest_path"]), "performance materialization"
            ),
            "materialization_sha256": sha256_file(Path(materialized["manifest_path"])),
            "runtime_plan": _relative(root, runtime_plan_path, "runtime plan"),
            "runtime_plan_sha256": sha256_file(runtime_plan_path),
            "results": _relative(root, results_path, "performance results"),
            "results_sha256": sha256_file(results_path),
            "measurements_csv": _relative(root, measurements_csv, "performance measurements CSV"),
            "measurements_csv_sha256": sha256_file(measurements_csv),
            "summary_csv": _relative(root, summary_csv, "performance summary CSV"),
            "summary_csv_sha256": sha256_file(summary_csv),
            "generation_root": _relative(root, generation_root, "performance generation run"),
            "interactive_report": _relative(root, report_path, "performance report"),
        }
        state, _report = publish_workflow_report(
            repo_root=root,
            workflow_path=workflow_path,
            publishing_state=publishing_state,
            html_path=report_path,
            manifest_path=report_path.parent / "performance-report.manifest.json",
            render=lambda: render_performance_report(
                repo_root=root,
                version_root=work / "v1",
                workflow_root=workflow_root,
            ),
            label="performance",
        )
    return state


__all__ = ["run_performance_workflow", "summarize_performance_measurements"]
