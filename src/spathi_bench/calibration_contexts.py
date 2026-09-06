"""Closed-world group-context contracts for SPATHI calibration evidence."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import BenchmarkConfigError, load_json_object
from .manifests import resolve_manifest_member


@dataclass(frozen=True, slots=True)
class DatasetGroupContract:
    """The complete group-level evaluation universe for one generated dataset."""

    dataset_id: str
    expected_group_contexts: tuple[str, ...]
    dataset_manifest_sha256: str
    group_assignments_sha256: str
    ground_truth_manifest_sha256: str
    truth_network_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_group_contexts"] = list(self.expected_group_contexts)
        payload["n_group_contexts"] = len(self.expected_group_contexts)
        return payload


def _group_contexts_from_assignments(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not {"column", "cluster"}.issubset(reader.fieldnames or ()):
                raise BenchmarkConfigError(
                    f"Group assignments must contain column and cluster fields: {path}"
                )
            assignments: dict[str, str] = {}
            for row_number, row in enumerate(reader, start=2):
                column = row.get("column")
                cluster = row.get("cluster")
                if (
                    not isinstance(column, str)
                    or not column
                    or column != column.strip()
                    or not isinstance(cluster, str)
                    or not cluster
                    or cluster != cluster.strip()
                ):
                    raise BenchmarkConfigError(f"Invalid group assignment at {path}:{row_number}")
                if column in assignments:
                    raise BenchmarkConfigError(
                        f"Duplicate cell {column!r} in group assignments: {path}"
                    )
                assignments[column] = cluster
    except (OSError, csv.Error) as exc:
        raise BenchmarkConfigError(f"Cannot read group assignments {path}: {exc}") from exc
    if not assignments:
        raise BenchmarkConfigError(f"Group assignments contain no cells: {path}")
    return tuple(sorted(f"group:{cluster}" for cluster in set(assignments.values())))


def _group_contexts_from_truth(path: Path) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "context" not in (reader.fieldnames or ()):
                raise BenchmarkConfigError(f"Truth network has no context field: {path}")
            contexts: set[str] = set()
            for row_number, row in enumerate(reader, start=2):
                context = row.get("context")
                if not isinstance(context, str) or not context or context != context.strip():
                    raise BenchmarkConfigError(f"Invalid truth context at {path}:{row_number}")
                if context.startswith("group:"):
                    if context == "group:":
                        raise BenchmarkConfigError(
                            f"Empty group label in truth network at {path}:{row_number}"
                        )
                    contexts.add(context)
    except (OSError, csv.Error) as exc:
        raise BenchmarkConfigError(f"Cannot read truth network {path}: {exc}") from exc
    if not contexts:
        raise BenchmarkConfigError(f"Truth network contains no group contexts: {path}")
    return tuple(sorted(contexts))


def load_dataset_group_contract(
    dataset_manifest_path: Path,
    ground_truth_manifest_path: Path,
    *,
    expected_dataset_id: str,
) -> DatasetGroupContract:
    """Resolve and reconcile group assignments with group-specific ground truth."""

    dataset_manifest = load_json_object(dataset_manifest_path, "dataset manifest")
    dataset = dataset_manifest.get("dataset")
    spec = dataset.get("spec") if isinstance(dataset, dict) else None
    if not isinstance(spec, dict) or spec.get("id") != expected_dataset_id:
        raise BenchmarkConfigError(
            f"Dataset manifest identity differs from {expected_dataset_id!r}"
        )
    extras = dataset_manifest.get("extras")
    groups_raw = extras.get("groups") if isinstance(extras, dict) else None
    groups_path = resolve_manifest_member(
        dataset_manifest_path.parent,
        groups_raw,
        "group assignments",
    )
    assignment_contexts = _group_contexts_from_assignments(groups_path)

    truth_manifest = load_json_object(ground_truth_manifest_path, "ground-truth manifest")
    if truth_manifest.get("dataset_id") != expected_dataset_id:
        raise BenchmarkConfigError(
            f"Ground-truth manifest identity differs from {expected_dataset_id!r}"
        )
    truth_requirements = truth_manifest.get("truth_requirements")
    required_contexts = (
        truth_requirements.get("contexts") if isinstance(truth_requirements, dict) else None
    )
    if not isinstance(required_contexts, list) or "group" not in required_contexts:
        raise BenchmarkConfigError(
            f"Ground truth for {expected_dataset_id!r} does not require group contexts"
        )
    outputs = truth_manifest.get("outputs")
    networks_raw = outputs.get("networks") if isinstance(outputs, dict) else None
    truth_network = resolve_manifest_member(
        ground_truth_manifest_path.parent,
        networks_raw,
        "truth network",
    )
    truth_contexts = _group_contexts_from_truth(truth_network)
    if truth_contexts != assignment_contexts:
        raise BenchmarkConfigError(
            f"Group assignments and truth contexts differ for {expected_dataset_id!r}: "
            f"assignments={list(assignment_contexts)}, truth={list(truth_contexts)}"
        )
    return DatasetGroupContract(
        dataset_id=expected_dataset_id,
        expected_group_contexts=assignment_contexts,
        dataset_manifest_sha256=sha256_file(dataset_manifest_path),
        group_assignments_sha256=sha256_file(groups_path),
        ground_truth_manifest_sha256=sha256_file(ground_truth_manifest_path),
        truth_network_sha256=sha256_file(truth_network),
    )


def completed_group_contexts(run_report: dict[str, Any], run_id: str) -> tuple[str, ...]:
    """Return the unique group contexts a completed run explicitly declares."""

    tools = run_report.get("tools")
    raw_inventory = tools.get("completed_contexts") if isinstance(tools, dict) else None
    raw_contexts = raw_inventory.get(run_id) if isinstance(raw_inventory, dict) else None
    if (
        not isinstance(raw_contexts, list)
        or not raw_contexts
        or not all(
            isinstance(context, str)
            and context.startswith("group:")
            and context != "group:"
            and context == context.strip()
            for context in raw_contexts
        )
        or len(set(raw_contexts)) != len(raw_contexts)
    ):
        raise BenchmarkConfigError(
            f"Run {run_id!r} has no complete, unique group-context inventory"
        )
    return tuple(sorted(raw_contexts))


def require_complete_group_contexts(
    run_report: dict[str, Any],
    run_id: str,
    expected_contexts: tuple[str, ...],
) -> tuple[str, ...]:
    """Fail unless one run declares the entire dataset group-context universe."""

    completed = completed_group_contexts(run_report, run_id)
    if completed != expected_contexts:
        raise BenchmarkConfigError(
            "Completed group contexts differ from the dataset contract: "
            f"expected={list(expected_contexts)}, completed={list(completed)}"
        )
    return completed
