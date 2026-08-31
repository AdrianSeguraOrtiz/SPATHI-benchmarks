"""Combine ANDREA evaluation metrics with configuration-level provenance."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from .artifacts import Artifact, ArtifactKey, ArtifactStore
from .config import BenchmarkConfigError, load_json_object
from .provenance import validate_run_inputs
from .state import load_state, require_status, status_at_least, write_state

PROVENANCE_FIELDS = (
    "profile_id",
    "purpose",
    "generation_job",
    "simulator_id",
    "generation_seed",
    "dataset_role",
    "split_id",
    "condition_id",
    "dataset_id",
    "batch_id",
    "batch_kind",
    "regime",
    "run_id",
    "catalog_tool_id",
    "tool_origin",
    "family",
    "configuration_id",
    "configuration_sha256",
    "selection_role",
    "algorithm_seed",
    "algorithm_seed_parameter",
    "execution_sha256",
    "tool_image",
    "wall_time_seconds",
    "peak_memory_mb",
    "inference_artifact",
    "evaluation_artifact",
)
SUMMARY_METRICS = ("auroc", "aupr", "f1_at_truth_count", "epr_at_truth_count")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _validate_artifact(store: ArtifactStore, reference: dict[str, Any]) -> Artifact:
    key = ArtifactKey(kind=str(reference["kind"]), digest=str(reference["digest"]))
    if reference.get("key") != key.value:
        raise BenchmarkConfigError(f"Report artifact key differs from its digest: {reference!r}")
    artifact = store.lookup(key)
    if artifact is None:
        raise BenchmarkConfigError(f"Report references a missing artifact: {key}")
    return artifact


def _artifact_member(artifact: Artifact, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise BenchmarkConfigError(f"{label} is missing")
    candidate = Path(raw_path).resolve()
    root = artifact.path.resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise BenchmarkConfigError(f"{label} escapes or is missing from its artifact")
    try:
        return artifact.file(candidate.relative_to(root).as_posix()).resolve()
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} is not declared by its artifact") from exc


def _existing_reports(state: dict[str, Any], run_root: Path) -> tuple[Path, Path]:
    reports = state.get("reports")
    if not isinstance(reports, dict):
        raise BenchmarkConfigError("Completed state has no reports")
    metrics = Path(str(reports.get("metrics_csv", "")))
    summary = Path(str(reports.get("summary_json", "")))
    expected_root = (run_root / "reports").resolve()
    if not metrics.resolve().is_relative_to(expected_root) or not summary.resolve().is_relative_to(
        expected_root
    ):
        raise BenchmarkConfigError("Recorded reports escape this run")
    if not metrics.is_file() or not summary.is_file():
        raise BenchmarkConfigError("Completed state references missing reports")
    return metrics, summary


def _recover_published_reports(
    *,
    state: dict[str, Any],
    profile_id: str,
    run_root: Path,
) -> tuple[Path, Path]:
    reports_dir = run_root / "reports"
    metrics = reports_dir / "metrics.csv"
    summary = reports_dir / "summary.json"
    try:
        payload = load_json_object(summary, "recoverable summary report")
    except BenchmarkConfigError as exc:
        raise BenchmarkConfigError(
            f"Incomplete reports directory already exists: {reports_dir}"
        ) from exc
    if (
        payload.get("schema_version") != "2.0"
        or payload.get("profile_id") != profile_id
        or payload.get("purpose") != state["purpose"]
        or not metrics.is_file()
        or metrics.stat().st_size == 0
    ):
        raise BenchmarkConfigError(f"Incomplete reports directory already exists: {reports_dir}")
    state["reports"] = {
        "metrics_csv": str(metrics.resolve()),
        "summary_json": str(summary.resolve()),
    }
    state["status"] = "completed"
    state["updated_at"] = _utc_now()
    write_state(run_root, state)
    return metrics, summary


def _calibration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["batch_kind"] == "spathi"
            and row["dataset_role"] in {"development", "validation"}
            and row.get("level") == "directed"
            and str(row.get("context", "")).startswith("group:")
        ):
            grouped[
                (
                    str(row["configuration_id"]),
                    str(row["dataset_role"]),
                    str(row["dataset_id"]),
                    int(row["algorithm_seed"]),
                )
            ].append(row)

    result: list[dict[str, Any]] = []
    for (configuration_id, dataset_role, dataset_id, algorithm_seed), members in sorted(
        grouped.items()
    ):
        evaluable = [
            row
            for row in members
            if row.get("status") in {"ok", "partial"}
            and isinstance(row.get("aupr"), (int, float))
            and not isinstance(row.get("aupr"), bool)
        ]
        group_contexts = sorted(str(row["context"]) for row in evaluable)
        if len(set(group_contexts)) != len(group_contexts):
            raise BenchmarkConfigError("Calibration metric rows contain duplicate group contexts")
        auprs = [float(row["aupr"]) for row in evaluable]
        wall_times = {row.get("wall_time_seconds") for row in members}
        peak_memories = {row.get("peak_memory_mb") for row in members}
        if len(wall_times) != 1 or len(peak_memories) != 1:
            raise BenchmarkConfigError("Calibration metric rows contain conflicting runtime data")
        tool_images = [row.get("tool_image") for row in members]
        if not all(isinstance(image, dict) for image in tool_images):
            raise BenchmarkConfigError("Calibration metric rows lack SPATHI image provenance")
        image_identities = {
            (
                image.get("source_commit"),
                image.get("build_fingerprint"),
                image.get("image_id"),
            )
            for image in tool_images
            if isinstance(image, dict)
        }
        if len(image_identities) != 1:
            raise BenchmarkConfigError("Calibration metric rows mix SPATHI image provenance")
        spathi_commit, spathi_build_fingerprint, spathi_image_id = next(iter(image_identities))
        if not all(
            isinstance(value, str) and value
            for value in (spathi_commit, spathi_build_fingerprint, spathi_image_id)
        ):
            raise BenchmarkConfigError("Calibration metric rows have incomplete image provenance")
        first = members[0]
        result.append(
            {
                "configuration_id": configuration_id,
                "configuration_sha256": first["configuration_sha256"],
                "dataset_role": dataset_role,
                "dataset_id": dataset_id,
                "algorithm_seed": algorithm_seed,
                "aupr_directed": fmean(auprs) if auprs else None,
                "wall_time_seconds": next(iter(wall_times)),
                "peak_memory_mb": next(iter(peak_memories)),
                "run_id": first["run_id"],
                "execution_sha256": first["execution_sha256"],
                "n_group_contexts": len(auprs),
                "group_contexts": group_contexts,
                "spathi_commit": spathi_commit,
                "spathi_build_fingerprint": spathi_build_fingerprint,
                "spathi_image_id": spathi_image_id,
            }
        )
    return result


def summarize(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
) -> tuple[Path, Path]:
    validate_run_inputs(repo_root, profile_id, run_root)
    state = load_state(run_root, profile_id)
    if status_at_least(state, "completed"):
        return _existing_reports(state, run_root)
    require_status(state, "evaluated", "Summarization")
    reports_dir = run_root / "reports"
    if reports_dir.exists():
        return _recover_published_reports(
            state=state,
            profile_id=profile_id,
            run_root=run_root,
        )

    store = ArtifactStore(Path(state["artifact_store"]))
    datasets = {item["dataset_id"]: item for item in state["datasets"]}
    rows: list[dict[str, Any]] = []
    for evaluation in state["evaluations"]:
        inference_artifact = _validate_artifact(store, evaluation["inference_artifact"])
        evaluation_artifact = _validate_artifact(store, evaluation["artifact"])
        dataset = datasets[evaluation["dataset_id"]]
        evaluation_report = _artifact_member(
            evaluation_artifact,
            evaluation.get("evaluation_report"),
            "evaluation report",
        )
        run_report_path = _artifact_member(
            inference_artifact,
            evaluation.get("run_report"),
            "inference run report",
        )
        report = load_json_object(evaluation_report, "evaluation report")
        run_report = load_json_object(run_report_path, "run report")
        tools = run_report.get("tools") or {}
        origins = tools.get("tool_origins", {})
        tool_result = (tools.get("results") or {}).get(evaluation["run_id"], {})
        wall_time_seconds = tool_result.get("duration_seconds")
        if not isinstance(wall_time_seconds, (int, float)) or isinstance(wall_time_seconds, bool):
            wall_time_seconds = None
        metrics = report.get("metrics")
        if not isinstance(metrics, list) or not all(isinstance(item, dict) for item in metrics):
            raise BenchmarkConfigError("evaluation_report.json metrics must be an array of objects")
        for metric in metrics:
            metric_run_id = metric.get("tool_id")
            if metric_run_id != evaluation["run_id"]:
                raise BenchmarkConfigError(
                    f"Evaluation metric tool_id {metric_run_id!r} differs from its single-run "
                    f"artifact {evaluation['run_id']!r}"
                )
            provenance = {
                "profile_id": profile_id,
                "purpose": state["purpose"],
                "generation_job": dataset["generation_job"],
                "simulator_id": dataset["simulator_id"],
                "generation_seed": dataset["seed"],
                "dataset_role": dataset["dataset_role"],
                "split_id": dataset["split_id"],
                "condition_id": dataset["condition_id"],
                "dataset_id": dataset["dataset_id"],
                "batch_id": evaluation["batch_id"],
                "batch_kind": evaluation["batch_kind"],
                "regime": evaluation["regime"],
                "run_id": evaluation["run_id"],
                "catalog_tool_id": evaluation["tool_id"],
                "tool_origin": origins.get(metric_run_id),
                "family": evaluation.get("family"),
                "configuration_id": evaluation["configuration_id"],
                "configuration_sha256": evaluation["configuration_sha256"],
                "selection_role": evaluation["selection_role"],
                "algorithm_seed": evaluation["algorithm_seed"],
                "algorithm_seed_parameter": evaluation["algorithm_seed_parameter"],
                "execution_sha256": evaluation["execution_sha256"],
                "tool_image": evaluation["tool_image"],
                "wall_time_seconds": wall_time_seconds,
                "peak_memory_mb": None,
                "inference_artifact": evaluation["inference_artifact"]["key"],
                "evaluation_artifact": evaluation["artifact"]["key"],
            }
            conflicts = {
                key: (metric[key], value)
                for key, value in provenance.items()
                if key in metric and metric[key] != value
            }
            if conflicts:
                raise BenchmarkConfigError(
                    f"Evaluation metric conflicts with frozen provenance: {conflicts}"
                )
            rows.append({**metric, **provenance})
    if not rows:
        raise BenchmarkConfigError("No evaluation metric rows were produced")

    metric_fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in PROVENANCE_FIELDS and key not in metric_fields:
                metric_fields.append(key)
    fields = [*PROVENANCE_FIELDS, *metric_fields]

    execution_key_fields = (
        "configuration_id",
        "level",
        "regime",
        "dataset_role",
        "split_id",
        "condition_id",
        "dataset_id",
        "run_id",
    )
    per_execution: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in SUMMARY_METRICS}
    )
    for row in rows:
        if row.get("status") not in {"ok", "partial"}:
            continue
        key = tuple(str(row.get(field)) for field in execution_key_fields)
        for metric_name in SUMMARY_METRICS:
            value = row.get(metric_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                per_execution[key][metric_name].append(float(value))

    per_dataset: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in SUMMARY_METRICS}
    )
    executions_per_dataset: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for execution_key, values in per_execution.items():
        *dataset_key_parts, run_id = execution_key
        dataset_key = tuple(dataset_key_parts)
        executions_per_dataset[dataset_key].add(run_id)
        for metric_name, context_values in values.items():
            if context_values:
                per_dataset[dataset_key][metric_name].append(fmean(context_values))

    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in SUMMARY_METRICS}
    )
    datasets_per_group: dict[tuple[str, ...], set[str]] = defaultdict(set)
    executions_per_group: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for dataset_key, values in per_dataset.items():
        *group_key_parts, dataset_id = dataset_key
        group_key = tuple(group_key_parts)
        datasets_per_group[group_key].add(dataset_id)
        executions_per_group[group_key].update(executions_per_dataset[dataset_key])
        for metric_name, execution_values in values.items():
            if execution_values:
                grouped[group_key][metric_name].append(fmean(execution_values))

    metadata_by_configuration: dict[str, dict[str, Any]] = {}
    for row in rows:
        configuration_id = str(row["configuration_id"])
        metadata = {
            "family": row.get("family"),
            "selection_role": row["selection_role"],
            "catalog_tool_id": row["catalog_tool_id"],
        }
        previous = metadata_by_configuration.setdefault(configuration_id, metadata)
        if previous != metadata:
            raise BenchmarkConfigError(
                f"Configuration {configuration_id!r} has conflicting summary metadata"
            )

    aggregates = []
    group_fields = execution_key_fields[:-2]
    for group_key, values in sorted(grouped.items()):
        group_identity = dict(zip(group_fields, group_key, strict=True))
        configuration_id = group_identity["configuration_id"]
        aggregates.append(
            {
                **group_identity,
                **metadata_by_configuration[configuration_id],
                "n_datasets": len(datasets_per_group[group_key]),
                "n_executions": len(executions_per_group[group_key]),
                "n_metric_rows": sum(
                    1
                    for row in rows
                    if all(str(row.get(field)) == value for field, value in group_identity.items())
                ),
                **{
                    f"mean_{name}": fmean(items) if items else None
                    for name, items in values.items()
                },
            }
        )

    notes = [
        "Each metric is averaged across contexts, then algorithmic seeds, then equally across datasets.",
        "Aggregates never mix dataset roles, splits or experimental conditions.",
        "Raw metric semantics and statuses come directly from ANDREA.",
    ]
    if state["purpose"] == "smoke":
        notes.insert(0, "The smoke profile validates contracts and has no statistical power.")

    temporary = Path(tempfile.mkdtemp(prefix=".reports-", dir=run_root))
    try:
        metrics_path = temporary / "metrics.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_value(row.get(field)) for field in fields})
        summary_path = temporary / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "profile_id": profile_id,
                    "purpose": state["purpose"],
                    "metric_rows": len(rows),
                    "aggregates": aggregates,
                    "calibration_rows": _calibration_rows(rows),
                    "cache": state["cache"],
                    "notes": notes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, reports_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    final_metrics = reports_dir / "metrics.csv"
    final_summary = reports_dir / "summary.json"
    state["reports"] = {
        "metrics_csv": str(final_metrics.resolve()),
        "summary_json": str(final_summary.resolve()),
    }
    state["status"] = "completed"
    state["updated_at"] = _utc_now()
    write_state(run_root, state)
    return final_metrics, final_summary
