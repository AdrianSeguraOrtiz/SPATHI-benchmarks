"""Interactive report for a frozen SPATHI comparison workflow."""

from __future__ import annotations

import csv
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..artifacts import sha256_file
from ..config import BenchmarkConfigError, load_json_object
from ..state import STATE_SCHEMA_VERSION
from .common import (
    central_report_path,
    path_below,
    repo_file,
    require_array,
    require_completed_workflow,
    require_mapping,
    safe_relative_link,
    validated_export_link,
    verify_sha256,
    write_report_bundle,
)
from .datasets import build_dataset_overview


def _optional_number(raw: Any, label: str) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError(f"{label} is not numeric: {raw!r}") from exc
    if not math.isfinite(value):
        raise BenchmarkConfigError(f"{label} is not finite: {raw!r}")
    return value


def _runtime_rows(metrics_path: Path) -> dict[str, Any]:
    """Extract one descriptive runtime record per immutable inference artifact."""

    by_artifact: dict[str, dict[str, Any]] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "inference_artifact",
            "configuration_id",
            "dataset_id",
            "simulator_id",
            "condition_id",
            "regime",
            "run_id",
            "catalog_tool_id",
            "family",
            "algorithm_seed",
            "wall_time_seconds",
            "peak_memory_mb",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise BenchmarkConfigError(
                f"Comparison metrics lack runtime provenance columns: {sorted(missing)}"
            )
        for row in reader:
            artifact = row["inference_artifact"].strip()
            if not artifact:
                raise BenchmarkConfigError("Comparison metric row has no inference artifact")
            family = row["family"].strip()
            tool_id = "SPATHI" if family == "spathi" else row["catalog_tool_id"].strip()
            record = {
                "inference_artifact": artifact,
                "configuration_id": row["configuration_id"].strip(),
                "dataset_id": row["dataset_id"].strip(),
                "simulator_id": row["simulator_id"].strip(),
                "condition_id": row["condition_id"].strip(),
                "dataset_role": (row.get("dataset_role") or "").strip() or None,
                "regime": row["regime"].strip(),
                "run_id": row["run_id"].strip(),
                "tool_id": tool_id,
                "family": family or None,
                "algorithm_seed": (
                    int(row["algorithm_seed"]) if row["algorithm_seed"].strip() else None
                ),
                "wall_time_seconds": _optional_number(
                    row["wall_time_seconds"], "wall_time_seconds"
                ),
                "peak_memory_mb": _optional_number(row["peak_memory_mb"], "peak_memory_mb"),
            }
            previous = by_artifact.setdefault(artifact, record)
            if previous != record:
                raise BenchmarkConfigError(
                    f"Inference artifact {artifact!r} has conflicting runtime provenance"
                )
    rows = sorted(
        by_artifact.values(),
        key=lambda item: (
            item["tool_id"],
            item["dataset_id"],
            item["regime"],
            item["run_id"],
        ),
    )
    return {
        "rows": rows,
        "wall_time_available": any(row["wall_time_seconds"] is not None for row in rows),
        "memory_available": any(row["peak_memory_mb"] is not None for row in rows),
        "deduplication_unit": "inference_artifact",
    }


def _export_datasets(
    *, exports: dict[str, Any], exports_root: Path, report_path: Path
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, str, str]] = set()
    for raw_bundle in require_array(exports.get("bundles"), "export bundles"):
        bundle = require_mapping(raw_bundle, "export bundle")
        dataset_id = bundle.get("dataset_id")
        producer = bundle.get("producer_command")
        scope = bundle.get("scope")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise BenchmarkConfigError("Export bundle has no dataset_id")
        if scope not in {"dataset", "joint"}:
            continue
        if scope == "dataset" and producer != "generate-data":
            continue
        if scope == "joint" and producer not in {"infer-network", "evaluate-inference"}:
            continue
        regime = bundle.get("regime") if scope == "joint" else ""
        if not isinstance(regime, str):
            raise BenchmarkConfigError("Joint export bundle has an invalid regime")
        kind = (
            "truth_zip"
            if producer == "generate-data"
            else "inference_zip"
            if producer == "infer-network"
            else "evaluation_zip"
        )
        duplicate_key = (dataset_id, regime, kind)
        if duplicate_key in seen:
            raise BenchmarkConfigError(f"Duplicate joint export bundle: {duplicate_key}")
        seen.add(duplicate_key)
        raw_analysis_zip = bundle.get("analysis_zip")
        href = validated_export_link(exports_root, raw_analysis_zip, report_path)
        if not isinstance(raw_analysis_zip, str):
            raise BenchmarkConfigError("Export bundle has no analysis_zip path")
        verify_sha256(
            path_below(exports_root / raw_analysis_zip, exports_root, "ANDREA analysis ZIP"),
            bundle.get("archive_sha256"),
            f"{kind} for {dataset_id}",
        )
        dataset = grouped.setdefault(
            dataset_id, {"dataset_id": dataset_id, "truth_zip": None, "regimes": {}}
        )
        if kind == "truth_zip":
            dataset["truth_zip"] = href
        else:
            regime_record = dataset["regimes"].setdefault(
                regime,
                {"regime": regime, "inference_zip": None, "evaluation_zip": None},
            )
            regime_record[kind] = href
    result = []
    for dataset_id, dataset in sorted(grouped.items()):
        regimes = [dataset["regimes"][key] for key in sorted(dataset["regimes"])]
        result.append(
            {
                "dataset_id": dataset_id,
                "truth_zip": dataset["truth_zip"],
                "regimes": regimes,
            }
        )
    return result


def _verify_statistics(
    *,
    state: dict[str, Any],
    run_root: Path,
    metrics_path: Path,
    summary_path: Path,
) -> tuple[dict[str, Any], list[Path]]:
    record = require_mapping(state.get("statistics"), "benchmark statistics state")
    if record.get("schema_version") != "1.0":
        raise BenchmarkConfigError("Benchmark statistics state has an unsupported schema")
    if record.get("input_metrics_sha256") != sha256_file(metrics_path):
        raise BenchmarkConfigError("Statistics were computed from different metrics")
    if record.get("input_summary_sha256") != sha256_file(summary_path):
        raise BenchmarkConfigError("Statistics were computed from a different summary")
    raw_directory = record.get("directory")
    if not isinstance(raw_directory, str):
        raise BenchmarkConfigError("Statistics directory is missing")
    directory_candidate = Path(raw_directory)
    if not directory_candidate.is_absolute():
        directory_candidate = run_root / directory_candidate
    directory = path_below(directory_candidate, run_root, "statistics", file=False)
    expected = (run_root / "reports/statistics").resolve()
    if directory != expected:
        raise BenchmarkConfigError("Recorded statistics directory differs from the run layout")
    files = require_mapping(record.get("files"), "statistics files")
    paths: list[Path] = []
    for filename in ("statistics.json", "rankings.csv", "pairwise.csv", "report.md"):
        file_record = require_mapping(files.get(filename), f"statistics {filename}")
        path = repo_file(run_root, file_record.get("path"), filename, below=directory)
        if path.parent != directory:
            raise BenchmarkConfigError(f"Statistics file escapes its directory: {path}")
        verify_sha256(path, file_record.get("sha256"), f"statistics {filename}")
        paths.append(path)
    report = load_json_object(paths[0], "statistical report")
    if report.get("schema_version") != "1.0":
        raise BenchmarkConfigError("Statistical report has an unsupported schema")
    inputs = require_mapping(report.get("inputs"), "statistical report inputs")
    if require_mapping(inputs.get("metrics_csv"), "statistics metrics input").get(
        "sha256"
    ) != sha256_file(metrics_path) or require_mapping(
        inputs.get("summary_json"), "statistics summary input"
    ).get("sha256") != sha256_file(summary_path):
        raise BenchmarkConfigError("Statistical report input hashes differ from published reports")
    return report, paths


def _shareable_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    """Remove machine-local paths while preserving the published statistical payload."""

    shareable = deepcopy(statistics)
    inputs = shareable.get("inputs")
    if not isinstance(inputs, dict):
        return shareable
    environment = inputs.get("benchmark_environment")
    if not isinstance(environment, dict):
        return shareable
    portable_fields = {
        "fingerprint",
        "machine",
        "python_executable_sha256",
        "python_implementation",
        "python_version",
        "system",
    }
    inputs["benchmark_environment"] = {
        key: value for key, value in environment.items() if key in portable_fields
    }
    return shareable


def _analysis_group_counts(analyses: list[Any]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for raw_analysis in analyses:
        analysis = require_mapping(raw_analysis, "statistical analysis")
        stratum = require_mapping(analysis.get("stratum"), "statistical analysis stratum")
        dataset_role = stratum.get("dataset_role")
        if not isinstance(dataset_role, str) or not dataset_role:
            raise BenchmarkConfigError("Statistical analysis has no dataset role")
        for raw_block in require_array(analysis.get("blocks"), "statistical analysis blocks"):
            block = require_mapping(raw_block, "statistical analysis block")
            dataset_id = block.get("dataset_id")
            groups = block.get("n_contexts")
            if not isinstance(dataset_id, str) or not dataset_id:
                raise BenchmarkConfigError("Statistical analysis block has no dataset ID")
            if isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0:
                raise BenchmarkConfigError(
                    f"Statistical analysis block for {dataset_id!r} has no valid group count"
                )
            key = (dataset_role, dataset_id)
            previous = counts.setdefault(key, groups)
            if previous != groups:
                raise BenchmarkConfigError(
                    f"Dataset {dataset_id!r} has inconsistent statistical group counts"
                )
    return counts


def render_comparison_report(
    *,
    repo_root: Path,
    version_root: Path,
    workflow_root: Path,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Render a completed comparison using only frozen published statistics."""

    root = repo_root.resolve()
    version = path_below(
        version_root,
        root / "work",
        "benchmark version root",
        file=False,
    )
    comparison_root = path_below(
        workflow_root,
        version / "workflows/comparison",
        "comparison workflow",
        file=False,
    )
    workflow_path = comparison_root / "workflow.json"
    workflow = require_completed_workflow(workflow_path, "comparison")
    profile = workflow.get("profile")
    if profile not in {"reduced", "full"} or comparison_root.parent.name != profile:
        raise BenchmarkConfigError("Comparison workflow path disagrees with its profile")
    if comparison_root.name != workflow["flow_id"]:
        raise BenchmarkConfigError("Comparison workflow path disagrees with its flow ID")
    calibration_flow_id = workflow.get("calibration_flow_id")
    if not isinstance(calibration_flow_id, str) or not calibration_flow_id:
        raise BenchmarkConfigError("Comparison workflow has no calibration flow ID")
    stage = require_mapping(
        require_mapping(workflow.get("stages"), "comparison workflow stages").get("comparison"),
        "comparison workflow stage",
    )
    if stage.get("status") != "complete":
        raise BenchmarkConfigError("Completed comparison workflow has no completed pipeline")
    pipeline = require_mapping(stage.get("pipeline"), "comparison pipeline")
    recorded_run_root = path_below(
        Path(str(pipeline.get("run_root", ""))),
        version,
        "comparison run",
        file=False,
    )
    if run_root is not None and recorded_run_root != run_root.resolve():
        raise BenchmarkConfigError("Requested run root differs from comparison workflow state")
    benchmark_run = recorded_run_root
    state_path = benchmark_run / "state.json"
    state = load_json_object(state_path, "comparison benchmark state")
    if (
        state.get("schema_version") != STATE_SCHEMA_VERSION
        or state.get("status") != "completed"
        or state.get("profile_id") != f"{profile}-v1"
    ):
        raise BenchmarkConfigError("Comparison benchmark state is not complete")

    profile_path = repo_file(
        root,
        workflow.get("profile_file"),
        "materialized comparison profile",
        below=comparison_root,
    )
    verify_sha256(profile_path, workflow.get("profile_sha256"), "comparison profile")
    reports = require_mapping(state.get("reports"), "comparison summary reports")
    reports_root = benchmark_run / "reports"
    metrics_path = repo_file(
        root, reports.get("metrics_csv"), "comparison metrics", below=reports_root
    )
    summary_path = repo_file(
        root, reports.get("summary_json"), "comparison summary", below=reports_root
    )
    summary = load_json_object(summary_path, "comparison summary")
    if (
        summary.get("schema_version") != "3.0"
        or summary.get("profile_id") != state.get("profile_id")
        or summary.get("purpose") != state.get("purpose")
    ):
        raise BenchmarkConfigError("Comparison summary differs from benchmark state")
    statistics, statistics_paths = _verify_statistics(
        state=state,
        run_root=benchmark_run,
        metrics_path=metrics_path,
        summary_path=summary_path,
    )
    if statistics.get("profile_id") != state.get("profile_id") or statistics.get(
        "purpose"
    ) != state.get("purpose"):
        raise BenchmarkConfigError("Statistical report differs from benchmark state")

    raw_exports_path = state.get("exports_manifest")
    exports_path = repo_file(
        root,
        raw_exports_path,
        "ANDREA export manifest",
        below=benchmark_run / "exports",
    )
    if exports_path != (benchmark_run / "exports/manifest.json").resolve():
        raise BenchmarkConfigError("ANDREA export manifest differs from the run layout")
    exports = load_json_object(exports_path, "ANDREA export manifest")
    if exports.get("status") != "complete" or exports.get("profile_id") != state.get("profile_id"):
        raise BenchmarkConfigError("ANDREA export manifest is not complete for this profile")

    report_path = central_report_path(
        version_root=version,
        report_type="comparison",
        scope=profile,
        flow_id=calibration_flow_id,
    )
    exported_datasets = _export_datasets(
        exports=exports,
        exports_root=benchmark_run / "exports",
        report_path=report_path,
    )
    runtimes = _runtime_rows(metrics_path)
    links = [
        {
            "label": "Workflow state",
            "href": safe_relative_link(report_path, workflow_path, version),
        },
        {
            "label": "Run state",
            "href": safe_relative_link(report_path, state_path, benchmark_run),
        },
        {
            "label": "Summary JSON",
            "href": safe_relative_link(report_path, summary_path, reports_root),
        },
        {
            "label": "Raw metrics CSV",
            "href": safe_relative_link(report_path, metrics_path, reports_root),
        },
        {
            "label": "Statistical JSON",
            "href": safe_relative_link(report_path, statistics_paths[0], reports_root),
        },
        {
            "label": "Rankings CSV",
            "href": safe_relative_link(report_path, statistics_paths[1], reports_root),
        },
        {
            "label": "Pairwise CSV",
            "href": safe_relative_link(report_path, statistics_paths[2], reports_root),
        },
        {
            "label": "Statistical Markdown",
            "href": safe_relative_link(report_path, statistics_paths[3], reports_root),
        },
        {
            "label": "ANDREA ZIP manifest",
            "href": safe_relative_link(report_path, exports_path, benchmark_run / "exports"),
        },
    ]
    analyses = require_array(statistics.get("analyses"), "statistical analyses")
    datasets, dataset_inputs = build_dataset_overview(
        repo_root=root,
        version_root=version,
        benchmark_state=state,
        group_counts=_analysis_group_counts(analyses),
    )
    analysis_roles = sorted(
        {
            analysis.get("analysis_role")
            for analysis in analyses
            if isinstance(analysis, dict) and isinstance(analysis.get("analysis_role"), str)
        }
    )
    comparison_roles = sorted(
        {
            analysis.get("comparison_role")
            for analysis in analyses
            if isinstance(analysis, dict) and isinstance(analysis.get("comparison_role"), str)
        }
    )
    payload = {
        "schema_version": "1.0",
        "report_type": "comparison",
        "workflow": {
            "flow_id": workflow["flow_id"],
            "status": workflow["status"],
            "profile": profile,
            "profile_id": state["profile_id"],
            "purpose": state.get("purpose"),
            "calibration_flow_id": calibration_flow_id,
            "analysis_roles": analysis_roles,
            "comparison_roles": comparison_roles,
            "identity": workflow["identity"],
        },
        "summary": summary,
        "statistics": _shareable_statistics(statistics),
        "datasets": datasets,
        "exports": {
            "status": exports.get("status"),
            "n_bundles": len(require_array(exports.get("bundles"), "export bundles")),
            "datasets": exported_datasets,
        },
        "runtimes": runtimes,
        "links": links,
    }
    return write_report_bundle(
        repo_root=root,
        kind="comparison",
        output_path=report_path,
        payload=payload,
        input_paths=[
            profile_path,
            state_path,
            metrics_path,
            summary_path,
            *statistics_paths,
            exports_path,
            *dataset_inputs,
        ],
        manifest_name="comparison-report.manifest.json",
    )
