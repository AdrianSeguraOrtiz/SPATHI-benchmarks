"""Combine ANDREA evaluation metrics without changing their semantics."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from .config import BenchmarkConfigError, load_json_object
from .state import load_state, write_state

PROVENANCE_FIELDS = (
    "profile_id",
    "generation_job",
    "simulator_id",
    "seed",
    "dataset_id",
    "batch_id",
    "regime",
    "tool_origin",
)
SUMMARY_METRICS = ("auroc", "aupr", "f1_at_truth_count", "epr_at_truth_count")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def summarize(*, profile_id: str, run_root: Path) -> tuple[Path, Path]:
    state = load_state(run_root, profile_id)
    if state.get("status") != "evaluated" or not state.get("evaluations"):
        raise BenchmarkConfigError("Summarization requires completed evaluations")
    reports_dir = run_root / "reports"
    if reports_dir.exists():
        raise BenchmarkConfigError(f"Reports stage already exists: {reports_dir}")
    reports_dir.mkdir(parents=True)

    datasets = {item["dataset_id"]: item for item in state["datasets"]}
    rows: list[dict[str, Any]] = []
    for evaluation in state["evaluations"]:
        dataset = datasets[evaluation["dataset_id"]]
        report = load_json_object(Path(evaluation["evaluation_report"]), "evaluation report")
        run_report = load_json_object(Path(evaluation["run_report"]), "run report")
        origins = (run_report.get("tools") or {}).get("tool_origins", {})
        metrics = report.get("metrics")
        if not isinstance(metrics, list) or not all(isinstance(item, dict) for item in metrics):
            raise BenchmarkConfigError("evaluation_report.json metrics must be an array of objects")
        for metric in metrics:
            tool_id = metric.get("tool_id")
            provenance = {
                "profile_id": profile_id,
                "generation_job": dataset["generation_job"],
                "simulator_id": dataset["simulator_id"],
                "seed": dataset["seed"],
                "dataset_id": dataset["dataset_id"],
                "batch_id": evaluation["batch_id"],
                "regime": evaluation["regime"],
                "tool_origin": origins.get(tool_id),
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
    metrics_path = reports_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})

    per_dataset: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in SUMMARY_METRICS}
    )
    for row in rows:
        if row.get("status") not in {"ok", "partial"}:
            continue
        key = (
            str(row.get("tool_id")),
            str(row.get("level")),
            str(row.get("regime")),
            str(row.get("dataset_id")),
        )
        for metric_name in SUMMARY_METRICS:
            value = row.get(metric_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                per_dataset[key][metric_name].append(float(value))

    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in SUMMARY_METRICS}
    )
    datasets_per_group: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for (tool_id, level, regime, dataset_id), values in per_dataset.items():
        group_key = (tool_id, level, regime)
        datasets_per_group[group_key].add(dataset_id)
        for metric_name, context_values in values.items():
            if context_values:
                grouped[group_key][metric_name].append(fmean(context_values))

    aggregates = []
    for (tool_id, level, regime), values in sorted(grouped.items()):
        aggregates.append(
            {
                "tool_id": tool_id,
                "level": level,
                "regime": regime,
                "n_datasets": len(datasets_per_group[(tool_id, level, regime)]),
                "n_metric_rows": sum(
                    1
                    for row in rows
                    if row.get("tool_id") == tool_id
                    and row.get("level") == level
                    and row.get("regime") == regime
                ),
                **{
                    f"mean_{name}": fmean(items) if items else None
                    for name, items in values.items()
                },
            }
        )
    summary_path = reports_dir / "summary.json"
    temporary = reports_dir / ".summary.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "profile_id": profile_id,
                "metric_rows": len(rows),
                "aggregates": aggregates,
                "notes": [
                    "Means are descriptive only; the smoke profile has no statistical power.",
                    "Each metric is averaged across contexts within a dataset, then equally across datasets.",
                    "Raw metric semantics and statuses come directly from ANDREA.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    state["reports"] = {"metrics_csv": str(metrics_path), "summary_json": str(summary_path)}
    state["status"] = "completed"
    write_state(run_root, state)
    return metrics_path, summary_path
