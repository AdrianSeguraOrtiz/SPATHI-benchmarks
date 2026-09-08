"""Interactive report for the formal computational-performance workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import BenchmarkConfigError, load_json_object, validate_json
from ..resource_plan import validate_runtime_plan
from .common import (
    central_report_path,
    path_below,
    reconcile_report_bundle,
    repo_file,
    require_array,
    require_mapping,
    require_reportable_workflow,
    safe_relative_link,
    verify_sha256,
    write_report_bundle,
)


def _validate_results(results: dict[str, Any], profile: str) -> None:
    if (
        results.get("status") != "complete"
        or results.get("budget") != profile
        or results.get("claim_scope") != "computational_performance_only"
    ):
        raise BenchmarkConfigError("Performance results disagree with their workflow")
    protocol = require_mapping(results.get("protocol"), "performance protocol")
    measurements = require_array(results.get("measurements"), "performance measurements")
    warmups = require_array(results.get("warmups"), "performance warmups")
    expected_tools = 4 if profile == "reduced" else 6
    expected_datasets = 1 if profile == "reduced" else 2
    expected_threads = [1, 8] if profile == "reduced" else [1, 2, 4, 8]
    expected_repetitions = 1 if profile == "reduced" else 3
    expected_measured = (
        expected_tools * expected_datasets * len(expected_threads) * expected_repetitions
    )
    expected_warmups = expected_tools * expected_datasets * len(expected_threads)
    if (
        protocol.get("thread_counts") != expected_threads
        or protocol.get("system_repetitions") != expected_repetitions
        or len(measurements) != expected_measured
        or len(warmups) != expected_warmups
    ):
        raise BenchmarkConfigError("Performance report inputs do not cover the frozen protocol")
    if any(item.get("phase") != "measurement" for item in measurements) or any(
        item.get("phase") != "warmup" for item in warmups
    ):
        raise BenchmarkConfigError("Performance report mixes measured and warm-up runs")
    summary = require_mapping(results.get("summary"), "performance summary")
    completed = sum(item.get("status") == "completed" for item in measurements)
    failed = sum(item.get("status") == "failed" for item in measurements)
    if (
        summary.get("measured_runs_requested") != len(measurements)
        or summary.get("measured_runs_completed") != completed
        or summary.get("measured_runs_failed") != failed
        or summary.get("all_measured_runs_completed") != (failed == 0)
    ):
        raise BenchmarkConfigError("Performance completion summary is inconsistent")
    equivalence = require_array(summary.get("equivalence"), "performance equivalence")
    equivalence_by_condition = {
        (str(item.get("dataset_id")), str(item.get("configuration_id"))): item.get(
            "equivalent_across_threads_and_repetitions"
        )
        for item in equivalence
    }
    aggregates = require_array(results.get("aggregates"), "performance aggregates")
    for aggregate in aggregates:
        key = (str(aggregate.get("dataset_id")), str(aggregate.get("configuration_id")))
        equivalent = equivalence_by_condition.get(key)
        expected_status = (
            "published"
            if equivalent is True
            else "withheld_semantic_drift"
            if equivalent is False
            else "withheld_incomplete_execution"
        )
        if aggregate.get("scaling_metrics_status") != expected_status:
            raise BenchmarkConfigError(
                "Performance scaling publication status disagrees with semantic equivalence"
            )
        if expected_status != "published" and (
            aggregate.get("speedup_vs_one_thread") is not None
            or aggregate.get("parallel_efficiency") is not None
        ):
            raise BenchmarkConfigError(
                "Performance publishes scaling metrics without semantic equivalence"
            )


def render_performance_report(
    *, repo_root: Path, version_root: Path, workflow_root: Path
) -> dict[str, Any]:
    """Render a self-contained English report from immutable performance evidence."""

    root = repo_root.resolve()
    version = path_below(version_root, root / "work", "benchmark version root", file=False)
    performance_root = path_below(
        workflow_root,
        version / "workflows/performance",
        "performance workflow",
        file=False,
    )
    workflow_path = performance_root / "workflow.json"
    workflow = require_reportable_workflow(workflow_path, "performance")
    profile = workflow.get("profile")
    if profile not in {"reduced", "full"} or performance_root.parent.name != profile:
        raise BenchmarkConfigError("Performance workflow path disagrees with its profile")
    if performance_root.name != workflow["flow_id"]:
        raise BenchmarkConfigError("Performance workflow path disagrees with its flow ID")

    runtime_plan_path = repo_file(
        root,
        workflow.get("runtime_plan"),
        "performance runtime plan",
        below=performance_root,
    )
    verify_sha256(
        runtime_plan_path,
        workflow.get("runtime_plan_sha256"),
        "performance runtime plan",
    )
    runtime_plan = validate_runtime_plan(
        load_json_object(runtime_plan_path, "performance runtime plan")
    )

    results_path = repo_file(
        root,
        workflow.get("results"),
        "performance results",
        below=performance_root,
    )
    verify_sha256(results_path, workflow.get("results_sha256"), "performance results")
    results = load_json_object(results_path, "performance results")
    validate_json(results, root / "schemas/performance-results.schema.json", "performance results")
    _validate_results(results, str(profile))

    materialization_path = repo_file(
        root,
        workflow.get("materialization"),
        "performance materialization",
        below=performance_root,
    )
    verify_sha256(
        materialization_path,
        workflow.get("materialization_sha256"),
        "performance materialization",
    )
    materialization = load_json_object(materialization_path, "performance materialization")
    validate_json(
        materialization,
        root / "schemas/performance-materialization.schema.json",
        "performance materialization",
    )
    if materialization.get("budget") != profile or materialization.get(
        "schedule_sha256"
    ) != require_mapping(results["materialization"], "results materialization").get(
        "schedule_sha256"
    ):
        raise BenchmarkConfigError("Performance results use a different frozen schedule")
    materialized_plan = require_mapping(
        materialization.get("runtime_plan"), "materialized performance runtime plan"
    )
    if materialized_plan.get("fingerprint") != runtime_plan["fingerprint"] or materialized_plan.get(
        "capacity"
    ) != {
        "cpu_threads": runtime_plan["capacity"]["cpu_threads"],
        "memory_bytes": runtime_plan["capacity"]["memory_bytes"],
    }:
        raise BenchmarkConfigError("Performance materialization uses another runtime plan")

    measurements_csv = repo_file(
        root,
        workflow.get("measurements_csv"),
        "performance measurement CSV",
        below=performance_root,
    )
    summary_csv = repo_file(
        root,
        workflow.get("summary_csv"),
        "performance summary CSV",
        below=performance_root,
    )
    verify_sha256(
        measurements_csv,
        workflow.get("measurements_csv_sha256"),
        "performance measurement CSV",
    )
    verify_sha256(summary_csv, workflow.get("summary_csv_sha256"), "performance summary CSV")
    report_path = central_report_path(
        version_root=version,
        report_type="performance",
        scope=str(profile),
        flow_id=str(workflow["flow_id"]),
    )
    if workflow.get("interactive_report") != report_path.relative_to(root).as_posix():
        raise BenchmarkConfigError("Performance workflow points to a non-canonical report")

    rows = require_array(results["measurements"], "performance measurements")
    datasets: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset = require_mapping(row.get("dataset"), "performance dataset")
        dataset_id = str(dataset["dataset_id"])
        previous = datasets.setdefault(dataset_id, dict(dataset))
        if previous != dataset:
            raise BenchmarkConfigError("Performance dataset metadata changes between runs")
    payload = {
        "schema_version": "1.0",
        "report_type": "performance",
        "workflow": {
            "workflow_type": "performance",
            "profile": profile,
            "flow_id": workflow["flow_id"],
            "status": "complete",
            "identity": workflow["identity"],
            "runtime_plan": runtime_plan,
        },
        "summary": results["summary"],
        "protocol": results["protocol"],
        "datasets": list(datasets.values()),
        "aggregates": results["aggregates"],
        "measurements": results["measurements"],
        "warmups": {
            "count": len(results["warmups"]),
            "completed": sum(item["status"] == "completed" for item in results["warmups"]),
            "excluded_from_aggregates": True,
        },
        "execution_environment": results["execution_environment"],
        "production_config": results["production_config"],
        "notes": results["notes"],
        "links": [
            {
                "label": "Performance results (JSON)",
                "href": safe_relative_link(report_path, results_path, root / "work"),
            },
            {
                "label": "Measured runs (CSV)",
                "href": safe_relative_link(report_path, measurements_csv, root / "work"),
            },
            {
                "label": "Aggregates (CSV)",
                "href": safe_relative_link(report_path, summary_csv, root / "work"),
            },
            {
                "label": "Frozen materialization (JSON)",
                "href": safe_relative_link(report_path, materialization_path, root / "work"),
            },
        ],
    }
    bundle = write_report_bundle(
        repo_root=root,
        kind="performance",
        output_path=report_path,
        payload=payload,
        input_paths=[
            workflow_path,
            runtime_plan_path,
            results_path,
            materialization_path,
            measurements_csv,
            summary_csv,
        ],
        manifest_name="performance-report.manifest.json",
        immutable=True,
    )
    return reconcile_report_bundle(workflow, bundle, "performance")


__all__ = ["render_performance_report"]
