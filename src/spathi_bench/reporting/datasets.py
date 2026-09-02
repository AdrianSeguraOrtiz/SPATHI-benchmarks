"""Validated dataset summaries shared by the interactive reports."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..artifacts import Artifact, ArtifactError, ArtifactKey, ArtifactStore
from ..config import BenchmarkConfigError, load_json_object
from .common import path_below, require_array, require_mapping


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkConfigError(f"{label} must be a positive integer")
    return value


def _range(rows: list[dict[str, Any]], field: str) -> tuple[int | None, int | None]:
    values = [row[field] for row in rows if isinstance(row.get(field), int)]
    return (min(values), max(values)) if values else (None, None)


def _overview_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    simulator_counts = Counter(row["simulator_id"] for row in rows)
    role_counts = Counter(row["dataset_role"] for row in rows)
    genes_min, genes_max = _range(rows, "genes")
    cells_min, cells_max = _range(rows, "cells")
    groups_min, groups_max = _range(rows, "groups")
    tf_min, tf_max = _range(rows, "transcription_factors")
    return {
        "summary": {
            "dataset_count": len(rows),
            "total_cells": sum(row["cells"] for row in rows),
            "simulators": [
                {"id": simulator_id, "count": count}
                for simulator_id, count in sorted(simulator_counts.items())
            ],
            "roles": [{"id": role, "count": count} for role, count in sorted(role_counts.items())],
            "genes_min": genes_min,
            "genes_max": genes_max,
            "cells_min": cells_min,
            "cells_max": cells_max,
            "groups_min": groups_min,
            "groups_max": groups_max,
            "transcription_factors_min": tf_min,
            "transcription_factors_max": tf_max,
        },
        "rows": rows,
    }


def _dataset_artifact(
    *,
    dataset: Mapping[str, Any],
    dataset_id: str,
    store: ArtifactStore,
    validated: dict[ArtifactKey, Artifact],
) -> Artifact:
    reference = require_mapping(
        dataset.get("dataset_artifact"), f"dataset artifact for {dataset_id}"
    )
    try:
        key = ArtifactKey(
            kind=str(reference["kind"]),
            digest=str(reference["digest"]),
        )
        if key.kind != "dataset":
            raise ArtifactError(f"Expected dataset artifact, found {key.kind!r}")
        if reference.get("key") != key.value:
            raise ArtifactError("Artifact reference key differs from its digest")
        artifact = validated.get(key)
        if artifact is None:
            artifact = store.validate(key)
            validated[key] = artifact
        if Path(str(reference.get("path", ""))).resolve() != artifact.path.resolve():
            raise ArtifactError("Artifact reference path differs from its content address")
        return artifact
    except (ArtifactError, KeyError, TypeError, ValueError) as exc:
        raise BenchmarkConfigError(
            f"Invalid content-addressed dataset artifact for {dataset_id}: {exc}"
        ) from exc


def _artifact_member(
    *,
    repo_root: Path,
    artifact: Artifact,
    raw_path: Any,
    label: str,
    relative_to: Path | None = None,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise BenchmarkConfigError(f"{label} path is missing")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (relative_to or repo_root) / candidate
    resolved = path_below(candidate, artifact.path, label)
    try:
        relative = resolved.relative_to(artifact.path.resolve()).as_posix()
        return artifact.file(relative).resolve()
    except (ArtifactError, ValueError) as exc:
        raise BenchmarkConfigError(f"{label} is not a declared member of {artifact.key}") from exc


def combine_dataset_overviews(overviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine disjoint role-specific panels into one deterministic report overview."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for overview in overviews:
        for raw_row in require_array(overview.get("rows"), "dataset overview rows"):
            row = require_mapping(raw_row, "dataset overview row")
            key = (str(row.get("dataset_role")), str(row.get("dataset_id")))
            if key in seen:
                raise BenchmarkConfigError(f"Duplicate dataset across report panels: {key}")
            seen.add(key)
            rows.append(row)
    rows.sort(
        key=lambda item: (
            item["dataset_role"],
            item["simulator_id"],
            str(item["condition_id"]),
            item["dataset_id"],
        )
    )
    return _overview_from_rows(rows)


def build_dataset_overview(
    *,
    repo_root: Path,
    version_root: Path,
    benchmark_state: Mapping[str, Any],
    group_counts: Mapping[tuple[str, str], int] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Describe the exact immutable datasets consumed by a completed benchmark run."""

    root = repo_root.resolve()
    version = version_root.resolve()
    expected_store_root = version / "artifacts"
    raw_store_root = benchmark_state.get("artifact_store")
    if not isinstance(raw_store_root, str) or not raw_store_root:
        raise BenchmarkConfigError("Benchmark state has no content-addressed artifact store")
    store_candidate = Path(raw_store_root)
    if not store_candidate.is_absolute():
        store_candidate = root / store_candidate
    store_root = path_below(
        store_candidate,
        version,
        "benchmark artifact store",
        file=False,
    )
    if store_root != expected_store_root.resolve():
        raise BenchmarkConfigError(
            "Benchmark artifact store differs from the shared version artifact store"
        )
    store = ArtifactStore(store_root)
    validated_artifacts: dict[ArtifactKey, Artifact] = {}
    known_group_counts = group_counts or {}
    rows: list[dict[str, Any]] = []
    inputs: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for raw_dataset in require_array(benchmark_state.get("datasets"), "benchmark datasets"):
        dataset = require_mapping(raw_dataset, "benchmark dataset")
        dataset_id = dataset.get("dataset_id")
        dataset_role = dataset.get("dataset_role")
        simulator_id = dataset.get("simulator_id")
        identity = (dataset_id, dataset_role, simulator_id)
        if not all(isinstance(item, str) and item for item in identity):
            raise BenchmarkConfigError("Benchmark dataset identity is incomplete")
        key = (dataset_role, dataset_id)
        if key in seen:
            raise BenchmarkConfigError(f"Duplicate benchmark dataset identity: {key}")
        seen.add(key)

        artifact = _dataset_artifact(
            dataset=dataset,
            dataset_id=dataset_id,
            store=store,
            validated=validated_artifacts,
        )
        manifest_path = _artifact_member(
            repo_root=root,
            artifact=artifact,
            raw_path=dataset.get("dataset_manifest"),
            label=f"dataset manifest for {dataset_id}",
        )
        manifest = load_json_object(manifest_path, f"dataset manifest for {dataset_id}")
        dataset_contract = require_mapping(
            manifest.get("dataset"), f"dataset contract for {dataset_id}"
        )
        specification = require_mapping(
            dataset_contract.get("spec"), f"dataset specification for {dataset_id}"
        )
        if specification.get("id") != dataset_id:
            raise BenchmarkConfigError(f"Dataset manifest identity differs for {dataset_id}")
        expression = require_mapping(
            specification.get("expression"), f"expression specification for {dataset_id}"
        )
        extras = require_mapping(manifest.get("extras"), f"dataset extras for {dataset_id}")
        groups_path = _artifact_member(
            repo_root=root,
            artifact=artifact,
            raw_path=extras.get("groups"),
            label=f"group assignments for {dataset_id}",
            relative_to=manifest_path.parent,
        )
        tf_path = _artifact_member(
            repo_root=root,
            artifact=artifact,
            raw_path=extras.get("tf_list"),
            label=f"TF list for {dataset_id}",
            relative_to=manifest_path.parent,
        )
        transcription_factors = sum(
            bool(line.strip()) for line in tf_path.read_text(encoding="utf-8").splitlines()
        )
        if transcription_factors <= 0:
            raise BenchmarkConfigError(f"TF list for {dataset_id} is empty")
        with groups_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "cluster" not in (reader.fieldnames or ()):
                raise BenchmarkConfigError(
                    f"Group assignments for {dataset_id} have no 'cluster' column"
                )
            group_labels = sorted({str(row.get("cluster", "")).strip() for row in reader} - {""})
        groups = _positive_integer(len(group_labels), f"group count for {dataset_id}")
        recorded_groups = known_group_counts.get(key)
        if recorded_groups is not None and groups != _positive_integer(
            recorded_groups, f"recorded group count for {dataset_id}"
        ):
            raise BenchmarkConfigError(
                f"Dataset {dataset_id!r} group assignments differ from the published results"
            )
        rows.append(
            {
                "dataset_id": dataset_id,
                "simulator_id": simulator_id,
                "condition_id": dataset.get("condition_id"),
                "dataset_role": dataset_role,
                "split_id": dataset.get("split_id"),
                "seed": dataset.get("seed"),
                "genes": _positive_integer(expression.get("genes"), f"gene count for {dataset_id}"),
                "cells": _positive_integer(
                    expression.get("columns"), f"expression columns for {dataset_id}"
                ),
                "column_kind": expression.get("column_kind"),
                "groups": groups,
                "group_labels": group_labels,
                "transcription_factors": transcription_factors,
            }
        )
        inputs.extend((manifest_path, groups_path, tf_path))

    rows.sort(
        key=lambda item: (
            item["dataset_role"],
            item["simulator_id"],
            str(item["condition_id"]),
            item["dataset_id"],
        )
    )
    return _overview_from_rows(rows), inputs
