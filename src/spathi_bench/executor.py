"""Execution boundary for the clean three-phase article study.

The planner owns scientific intent; this module is the only place that knows
how to materialize it with ANDREA. Expensive results are published atomically
in a content-addressed store, while selectors consume structured dependency
evidence. Git commits are provenance, not implicit cache invalidators.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from statistics import fmean, pstdev
from time import perf_counter_ns
from typing import Any

from .article_statistics import analyze_paired_statistics
from .artifacts import Artifact, ArtifactKey, ArtifactStore, canonical_json_sha256, sha256_file
from .catalog_identity import catalog_definition_identity, verify_catalog_definitions
from .config import BenchmarkConfigError
from .runtime import CommandExecutionError, andrea_command, stream_command
from .safe_io import atomic_write_json, atomic_write_text, path_below_no_symlinks
from .study_store import TaskRecord
from .synthetic import GENERATOR_CONTRACT, generate_scale_dataset

EXECUTOR_CONTRACT = "article-study-executor-v1"
DATASET_MATERIALIZATION_CONTRACT = "requested-observed-dataset-v1"
_BASE_IMAGE = (
    "python:3.13-slim@sha256:7ce4b6dfe35e55397b7cda544f8a13f191b7ae28dc5aad71fe664dbc9bc2623f"
)
_IMAGE_LOCK = threading.Lock()
_CATALOG_IMAGE_LOCK = threading.Lock()
_ID_CLEANER = re.compile(r"[^a-zA-Z0-9._-]+")
_METRIC_FIELDS = {"aupr": "aupr", "auroc": "auroc", "f1": "f1_at_truth_count"}
_PLANNING_RESOURCE_STATUSES = frozenset({"eligible", "estimated_infeasible_time"})


def _manifest_file(base: Path, raw: Any, label: str) -> Path:
    """Resolve one canonical manifest member without accepting path escapes."""

    if not isinstance(raw, str) or not raw:
        raise BenchmarkConfigError(f"{label} must be a non-empty relative path")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or raw != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise BenchmarkConfigError(f"{label} must be a canonical relative POSIX path")
    return path_below_no_symlinks(base.joinpath(*relative.parts), base, label, require_file=True)


def effective_route_threads(requested: int, max_threads: int | None) -> int:
    """Resolve a portable thread point against one frozen tool contract."""

    if requested < 1:
        raise BenchmarkConfigError("Requested threads must be positive")
    if isinstance(max_threads, bool) or (
        max_threads is not None and (not isinstance(max_threads, int) or max_threads < 1)
    ):
        raise BenchmarkConfigError("Route max_threads must be null or a positive integer")
    return requested if max_threads is None else min(requested, max_threads)


def _andrea_ram_gb(resources: Mapping[str, Any]) -> float:
    """Return the exact decimal RAM ceiling handed to ANDREA's CLI and plans."""

    memory_bytes = resources.get("memory_bytes")
    if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int) or memory_bytes < 1:
        raise BenchmarkConfigError("Task memory_bytes must be a positive integer")
    return float(f"{memory_bytes / 1024**3:.6g}")


def _unique_file(root: Path, name: str) -> Path:
    matches = sorted(path.resolve() for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise BenchmarkConfigError(f"Expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def _json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkConfigError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkConfigError(f"{label} must be a JSON object: {path}")
    return payload


def _planning_resource_decisions(
    plan: Mapping[str, Any], selected_run_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Validate ANDREA's advisory planning decisions without interpreting ETA fields.

    A conservative fallback is deliberately still eligible, even when its ETA is
    above the physical timeout. Only ANDREA's empirical
    ``estimated_infeasible_time`` status is an admissible reason to omit a run.
    Keeping the validation at this boundary prevents an absent cost profile from
    being mistaken for negative evidence while rejecting incomplete planner
    responses before any article result is journalled.
    """

    selected = list(selected_run_ids)
    if any(not isinstance(run_id, str) or not run_id for run_id in selected):
        raise BenchmarkConfigError("Inference preflight selected-run IDs are invalid")
    if len(selected) != len(set(selected)):
        raise BenchmarkConfigError("Inference preflight contains duplicate selected-run IDs")
    raw_runs = plan.get("runs")
    if not isinstance(raw_runs, list):
        raise BenchmarkConfigError("Inference plan runs must be an array")

    decisions: dict[str, dict[str, Any]] = {}
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, Mapping):
            raise BenchmarkConfigError(f"Inference plan run {index} must be an object")
        run_id = raw_run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise BenchmarkConfigError(f"Inference plan run {index} has no valid run_id")
        if run_id in decisions:
            raise BenchmarkConfigError(f"Inference plan contains duplicate run_id {run_id!r}")
        raw_decision = raw_run.get("resource_decision")
        if not isinstance(raw_decision, Mapping):
            raise BenchmarkConfigError(f"Inference plan run {run_id!r} omits its resource_decision")
        decision = dict(raw_decision)
        if decision.get("schema_version") != "1.0" or decision.get("advisory") is not True:
            raise BenchmarkConfigError(
                f"Inference plan run {run_id!r} has an unsupported resource decision"
            )
        status = decision.get("status")
        if status not in _PLANNING_RESOURCE_STATUSES:
            raise BenchmarkConfigError(
                f"Inference plan run {run_id!r} has invalid resource status {status!r}"
            )
        physical = decision.get("physical_tasks")
        if not isinstance(physical, list) or not physical:
            raise BenchmarkConfigError(
                f"Inference plan run {run_id!r} has no physical resource decisions"
            )
        physical_statuses: dict[str, str] = {}
        for task_index, raw_task in enumerate(physical):
            if not isinstance(raw_task, Mapping):
                raise BenchmarkConfigError(
                    f"Inference plan run {run_id!r} physical task {task_index} is invalid"
                )
            task_id = raw_task.get("task_id")
            task_status = raw_task.get("status")
            eta_source = raw_task.get("eta_source")
            evidence = raw_task.get("evidence")
            estimated = raw_task.get("estimated_seconds")
            timeout = raw_task.get("timeout_seconds")
            if not isinstance(task_id, str) or not task_id or task_id in physical_statuses:
                raise BenchmarkConfigError(
                    f"Inference plan run {run_id!r} has an invalid physical task_id"
                )
            if (
                task_status not in _PLANNING_RESOURCE_STATUSES
                or raw_task.get("schema_version") != "1.0"
                or raw_task.get("advisory") is not True
            ):
                raise BenchmarkConfigError(
                    f"Inference plan physical task {task_id!r} has an unsupported decision"
                )
            if isinstance(estimated, bool) or not isinstance(estimated, (int, float)):
                raise BenchmarkConfigError(
                    f"Inference plan physical task {task_id!r} has no numeric ETA"
                )
            if not math.isfinite(float(estimated)) or float(estimated) < 0:
                raise BenchmarkConfigError(
                    f"Inference plan physical task {task_id!r} has an invalid ETA"
                )
            if timeout is not None and (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(float(timeout))
                or float(timeout) <= 0
            ):
                raise BenchmarkConfigError(
                    f"Inference plan physical task {task_id!r} has an invalid timeout"
                )
            if eta_source == "cost_profile":
                if evidence != "empirical_cost_profile":
                    raise BenchmarkConfigError(
                        f"Inference plan physical task {task_id!r} mislabels empirical evidence"
                    )
            elif isinstance(eta_source, str) and eta_source:
                if evidence != "conservative_fallback" or task_status != "eligible":
                    raise BenchmarkConfigError(
                        f"Inference plan physical task {task_id!r} misuses fallback evidence"
                    )
            else:
                raise BenchmarkConfigError(
                    f"Inference plan physical task {task_id!r} has unknown eta_source "
                    f"{eta_source!r}"
                )
            physical_statuses[task_id] = str(task_status)

        infeasible = sorted(
            task_id
            for task_id, task_status in physical_statuses.items()
            if task_status == "estimated_infeasible_time"
        )
        declared_infeasible = decision.get("infeasible_physical_tasks")
        if (
            not isinstance(declared_infeasible, list)
            or any(not isinstance(task_id, str) or not task_id for task_id in declared_infeasible)
            or len(declared_infeasible) != len(set(declared_infeasible))
            or sorted(declared_infeasible) != infeasible
        ):
            raise BenchmarkConfigError(
                f"Inference plan run {run_id!r} has inconsistent infeasible-task evidence"
            )
        expected_status = "estimated_infeasible_time" if infeasible else "eligible"
        if status != expected_status:
            raise BenchmarkConfigError(
                f"Inference plan run {run_id!r} disagrees with its physical decisions"
            )
        decisions[run_id] = decision

    planned = set(decisions)
    expected = set(selected)
    if planned != expected:
        raise BenchmarkConfigError(
            "Inference plan run inventory differs from preflight selection: "
            f"missing={sorted(expected - planned)}, unexpected={sorted(planned - expected)}"
        )
    return decisions


def _relative(artifact: Artifact, path: Path) -> str:
    try:
        return path.resolve().relative_to(artifact.path.resolve()).as_posix()
    except ValueError as exc:
        raise BenchmarkConfigError(f"Artifact member escapes {artifact.path}: {path}") from exc


def artifact_reference(artifact: Artifact, *, members: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "kind": artifact.key.kind,
        "digest": artifact.key.digest,
        "key": artifact.key.value,
        "path": str(artifact.path),
        "members": {name: _relative(artifact, path) for name, path in members.items()},
    }


def resolve_artifact_reference(
    store: ArtifactStore, reference: Mapping[str, Any], member: str
) -> Path:
    key = ArtifactKey(kind=str(reference["kind"]), digest=str(reference["digest"]))
    artifact = store.validate(key)
    if artifact.key.value != reference.get("key"):
        raise BenchmarkConfigError("Saved artifact reference has the wrong content address")
    raw = reference.get("members", {}).get(member)
    if not isinstance(raw, str):
        raise BenchmarkConfigError(f"Saved artifact has no {member!r} member")
    return artifact.file(raw)


def _publish_existing(
    store: ArtifactStore, *, kind: str, spec: Mapping[str, Any], staging: Path
) -> Artifact:
    key = store.key_for(kind, spec)
    return store.lookup(key) or store.publish_from_temp(staging, kind=kind, spec=dict(spec))


def _slug(value: str, *, lower: bool = False) -> str:
    result = _ID_CLEANER.sub("_", value).strip("._-") or "run"
    return result.lower() if lower else result


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fingerprint_dataset(dataset_manifest: Path) -> dict[str, str]:
    manifest = _json_file(dataset_manifest, "dataset manifest")
    dataset = manifest.get("dataset")
    extras = manifest.get("extras")
    if not isinstance(dataset, Mapping) or not isinstance(extras, Mapping):
        raise BenchmarkConfigError("Dataset manifest has no dataset/extras identity")
    expression_raw = dataset.get("expression_matrix")
    if not isinstance(expression_raw, str):
        raise BenchmarkConfigError("Dataset manifest has no expression matrix")
    components = {"expression": sha256_file(dataset_manifest.parent / expression_raw)}
    for name, raw in sorted(extras.items()):
        if isinstance(raw, str):
            components[f"extra:{name}"] = sha256_file(dataset_manifest.parent / raw)
    encoded = json.dumps(components, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {"algorithm": "sha256", "value": hashlib.sha256(encoded.encode("ascii")).hexdigest()}


def _candidate_space_identity(truth_manifest: Path) -> dict[str, Any]:
    """Freeze the semantic TF-to-target universe declared by one truth bundle."""

    manifest = _json_file(truth_manifest, "ground-truth manifest")
    raw = manifest.get("candidate_space")
    if not isinstance(raw, Mapping):
        raise BenchmarkConfigError("Ground truth has no explicit candidate_space")
    if raw.get("allow_self_edges") is not False:
        raise BenchmarkConfigError("Article evaluation requires allow_self_edges=false")

    sets: dict[str, list[str]] = {}
    references: dict[str, str] = {}
    file_hashes: dict[str, str] = {}
    base = truth_manifest.parent
    for name in ("sources", "targets"):
        value = raw.get(name)
        if not isinstance(value, str) or not value.endswith(".txt"):
            raise BenchmarkConfigError(f"candidate_space.{name} must be a relative .txt path")
        relative = PurePosixPath(value)
        if (
            relative.is_absolute()
            or value != relative.as_posix()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise BenchmarkConfigError(
                f"candidate_space.{name} must be a canonical relative POSIX path"
            )
        path = path_below_no_symlinks(
            base.joinpath(*relative.parts),
            base,
            f"Candidate {name}",
            require_file=True,
        )
        genes = path.read_text(encoding="utf-8").splitlines()
        if not genes or any(not gene or gene != gene.strip() for gene in genes):
            raise BenchmarkConfigError(
                f"candidate_space.{name} contains an empty or non-canonical gene"
            )
        if len(genes) != len(set(genes)):
            raise BenchmarkConfigError(f"candidate_space.{name} contains duplicate genes")
        sets[name] = sorted(genes)
        references[name] = value
        file_hashes[name] = sha256_file(path)

    sources = set(sets["sources"])
    targets = set(sets["targets"])
    semantic = {
        "allow_self_edges": False,
        "sources_sha256": canonical_json_sha256(sets["sources"]),
        "targets_sha256": canonical_json_sha256(sets["targets"]),
    }
    return {
        "candidate_space_sha256": canonical_json_sha256(semantic),
        **semantic,
        "source_file_sha256": file_hashes["sources"],
        "target_file_sha256": file_hashes["targets"],
        "sources_reference": references["sources"],
        "targets_reference": references["targets"],
        "n_sources": len(sources),
        "n_targets": len(targets),
        "n_source_target_overlap": len(sources & targets),
    }


def _candidate_space_evidence(
    identity: Mapping[str, Any], evaluation_report: Mapping[str, Any]
) -> dict[str, Any]:
    ground_truth = evaluation_report.get("ground_truth")
    observed = ground_truth.get("candidate_space") if isinstance(ground_truth, Mapping) else None
    if not isinstance(observed, Mapping):
        raise BenchmarkConfigError("Evaluation report omits candidate-space evidence")
    expected = {
        "sources": identity["sources_reference"],
        "targets": identity["targets_reference"],
        "allow_self_edges": False,
        "n_sources": identity["n_sources"],
        "n_targets": identity["n_targets"],
        "n_source_target_overlap": identity["n_source_target_overlap"],
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise BenchmarkConfigError(
            "Evaluation changed the declared candidate space: "
            + json.dumps(mismatches, sort_keys=True)
        )
    fields = (
        "truth_rows_total",
        "n_truth_rows_outside_candidate_space_by_level",
        "n_truth_edges_outside_candidate_space_by_level",
        "truth_filtering_by_context",
        "n_candidates_by_level",
    )
    return {
        "schema_version": "1.0",
        "identity": dict(identity),
        "evaluation": {field: observed.get(field) for field in fields},
    }


def _positive_manifest_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BenchmarkConfigError(f"{label} must be a positive integer")
    return value


def _expression_axes(path: Path) -> tuple[set[str], set[str]]:
    """Inspect a TSV matrix in one bounded-memory streaming pass."""

    genes: set[str] = set()
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            header = handle.readline().rstrip("\r\n")
            fields = header.split("\t")
            columns = fields[1:]
            if (
                len(fields) < 2
                or not fields[0]
                or any(not value or value != value.strip() for value in columns)
            ):
                raise BenchmarkConfigError("Expression TSV has an invalid header")
            if len(columns) != len(set(columns)):
                raise BenchmarkConfigError("Expression TSV contains duplicate column identifiers")
            expected_tabs = len(columns)
            for line_number, raw in enumerate(handle, start=2):
                line = raw.rstrip("\r\n")
                if not line or line.count("\t") != expected_tabs:
                    raise BenchmarkConfigError(
                        f"Expression TSV row {line_number} is not rectangular"
                    )
                gene, _separator, _values = line.partition("\t")
                if not gene or gene != gene.strip() or gene in genes:
                    raise BenchmarkConfigError(
                        f"Expression TSV row {line_number} has an invalid gene identifier"
                    )
                genes.add(gene)
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkConfigError(f"Cannot inspect expression TSV {path}: {exc}") from exc
    if not genes:
        raise BenchmarkConfigError("Expression TSV contains no gene rows")
    return genes, set(columns)


def _text_identifiers(path: Path, label: str) -> set[str]:
    try:
        values = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkConfigError(f"Cannot inspect {label} {path}: {exc}") from exc
    if not values or any(not value or value != value.strip() for value in values):
        raise BenchmarkConfigError(f"{label} contains an empty or non-canonical identifier")
    if len(values) != len(set(values)):
        raise BenchmarkConfigError(f"{label} contains duplicate identifiers")
    return set(values)


def _group_contract(path: Path, expression_columns: set[str]) -> tuple[int, int]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = tuple(reader.fieldnames or ())
            if len(fields) < 2 or "cluster" not in fields:
                raise BenchmarkConfigError("Groups TSV must contain an identifier and cluster")
            identifier = fields[0]
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise BenchmarkConfigError(f"Cannot inspect groups TSV {path}: {exc}") from exc
    identifiers = [str(row.get(identifier, "")) for row in rows]
    clusters = [str(row.get("cluster", "")) for row in rows]
    if any(not value or value != value.strip() for value in identifiers + clusters):
        raise BenchmarkConfigError("Groups TSV contains an empty or non-canonical value")
    if len(identifiers) != len(set(identifiers)):
        raise BenchmarkConfigError("Groups TSV contains duplicate expression columns")
    if set(identifiers) != expression_columns:
        raise BenchmarkConfigError("Groups TSV does not cover expression columns exactly")
    return len(rows), len(set(clusters))


def _dataset_materialization_contract(
    benchmark_manifest: Path, requested: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare prospective dimensions with the dataset that was actually emitted.

    scMultiSim, SERGIO, BoolODE and the benchmark-owned scale generator have an
    exact contract for the panels declared by this study.  dyngen is different:
    its backbone may add regulators, so requested genes/TFs are prospective
    minima. Its requested group count is the number of backbone milestones, while
    exported groups are the milestones actually occupied as a cell's dominant
    membership. Cells remain exact; gene expansion and milestone occupancy are
    validated structurally instead of through arbitrary numeric tolerances.
    """

    benchmark = _json_file(benchmark_manifest, "benchmark manifest")
    raw_records = benchmark.get("datasets")
    if (
        not isinstance(raw_records, list)
        or len(raw_records) != 1
        or not isinstance(raw_records[0], Mapping)
    ):
        raise BenchmarkConfigError("A generated article artifact must contain one dataset")
    raw_record = raw_records[0]
    generator = requested.get("generator")
    expected_simulator = "spathi_scale_stress" if generator == "scale-stress" else generator
    if not isinstance(expected_simulator, str) or not expected_simulator:
        raise BenchmarkConfigError("Requested dataset generator is invalid")
    if raw_record.get("simulator_id") != expected_simulator:
        raise BenchmarkConfigError("Generated dataset simulator identity differs from the request")
    requested_seed = requested.get("seed")
    if (
        isinstance(requested_seed, bool)
        or not isinstance(requested_seed, int)
        or requested_seed < 1
        or raw_record.get("seed") != requested_seed
    ):
        raise BenchmarkConfigError("Generated dataset seed differs from the request")
    dataset_manifest = _manifest_file(
        benchmark_manifest.parent,
        raw_record.get("dataset_manifest"),
        "Benchmark dataset manifest",
    )
    truth_manifest = _manifest_file(
        benchmark_manifest.parent,
        raw_record.get("ground_truth_manifest"),
        "Benchmark ground-truth manifest",
    )
    manifest = _json_file(dataset_manifest, "dataset manifest")
    dataset = manifest.get("dataset")
    extras = manifest.get("extras")
    if not isinstance(dataset, Mapping) or not isinstance(extras, Mapping):
        raise BenchmarkConfigError("Dataset manifest omits dataset or extras")
    spec = dataset.get("spec")
    expression_spec = spec.get("expression") if isinstance(spec, Mapping) else None
    if not isinstance(expression_spec, Mapping):
        raise BenchmarkConfigError("Dataset manifest omits expression dimensions")
    expression = _manifest_file(
        dataset_manifest.parent,
        dataset.get("expression_matrix"),
        "Dataset expression matrix",
    )
    genes, columns = _expression_axes(expression)
    declared_genes = _positive_manifest_integer(
        expression_spec.get("genes"), "Manifest expression genes"
    )
    declared_columns = _positive_manifest_integer(
        expression_spec.get("columns"), "Manifest expression columns"
    )
    if declared_genes != len(genes) or declared_columns != len(columns):
        raise BenchmarkConfigError("Dataset manifest dimensions disagree with the expression TSV")
    if expression_spec.get("column_kind") != "cells":
        raise BenchmarkConfigError("Article datasets must materialize single-cell columns")

    tf_path = _manifest_file(dataset_manifest.parent, extras.get("tf_list"), "TF list")
    tf_ids = _text_identifiers(tf_path, "TF list")
    if not tf_ids <= genes:
        raise BenchmarkConfigError("TF list contains identifiers absent from expression")
    groups_path = _manifest_file(dataset_manifest.parent, extras.get("groups"), "Groups TSV")
    group_rows, group_count = _group_contract(groups_path, columns)
    if group_rows != len(columns):
        raise BenchmarkConfigError("Groups TSV row count differs from expression cells")

    truth = _json_file(truth_manifest, "ground-truth manifest")
    if truth.get("dataset_id") != raw_record.get("dataset_id"):
        raise BenchmarkConfigError("Ground truth and benchmark dataset identities disagree")
    if truth.get("simulator_id") != expected_simulator:
        raise BenchmarkConfigError("Ground truth simulator identity differs from the request")
    candidate = _candidate_space_identity(truth_manifest)
    if (
        candidate["n_sources"] != len(tf_ids)
        or candidate["n_targets"] != len(genes)
        or candidate["sources_sha256"] != canonical_json_sha256(sorted(tf_ids))
        or candidate["targets_sha256"] != canonical_json_sha256(sorted(genes))
    ):
        raise BenchmarkConfigError(
            "Ground-truth candidate space disagrees with expression or TF-list dimensions"
        )

    capability_files = {
        "group_labels": "groups",
        "tf_list": "tf_list",
        "ordered_groups": "column_phenotypes",
        "lineage_tree": "lineage_tree",
    }
    observed_capabilities = sorted(
        capability
        for capability, extra in capability_files.items()
        if extras.get(extra) is not None
    )
    requested_capabilities = requested.get("capabilities")
    if (
        isinstance(requested_capabilities, (str, bytes))
        or not isinstance(requested_capabilities, Sequence)
        or any(not isinstance(value, str) or not value for value in requested_capabilities)
    ):
        raise BenchmarkConfigError("Requested dataset capabilities are invalid")
    missing_capabilities = sorted(set(requested_capabilities) - set(observed_capabilities))
    if missing_capabilities:
        raise BenchmarkConfigError(
            "Generated dataset omits promised capabilities: " + ", ".join(missing_capabilities)
        )
    for capability in requested_capabilities:
        extra = capability_files.get(capability)
        if extra is not None:
            _manifest_file(
                dataset_manifest.parent,
                extras.get(extra),
                f"Dataset capability {capability}",
            )

    requested_dimensions = requested.get("dimensions")
    if not isinstance(requested_dimensions, Mapping):
        raise BenchmarkConfigError("Requested dataset dimensions are missing")
    normalized_requested = {
        field: _positive_manifest_integer(requested_dimensions.get(field), f"Requested {field}")
        for field in ("cells", "genes", "transcription_factors", "groups")
    }
    observed_dimensions = {
        "cells": len(columns),
        "genes": len(genes),
        "transcription_factors": len(tf_ids),
        "groups": group_count,
    }
    deviations = {
        field: {
            "requested": normalized_requested[field],
            "observed": observed_dimensions[field],
            "delta": observed_dimensions[field] - normalized_requested[field],
        }
        for field in normalized_requested
        if normalized_requested[field] != observed_dimensions[field]
    }
    exact_generators = {"scale-stress", "scmultisim", "sergio", "boolode"}
    if generator in exact_generators:
        policy = "exact"
        dimension_policy = {field: "exact" for field in normalized_requested}
    elif generator == "dyngen":
        policy = "dyngen_native_materialization"
        dimension_policy = {
            "cells": "exact",
            "genes": "dyngen_backbone_augmented",
            "transcription_factors": "dyngen_backbone_augmented",
            "groups": "dyngen_observed_milestone_occupancy",
        }
    else:
        raise BenchmarkConfigError(
            f"No requested-vs-observed dimensional policy exists for {generator!r}"
        )

    exact_deviations = {
        field: deviations[field]
        for field, field_policy in dimension_policy.items()
        if field_policy == "exact" and field in deviations
    }
    if exact_deviations:
        raise BenchmarkConfigError(
            f"{generator} dataset violated its exact dimensions: "
            + json.dumps(exact_deviations, sort_keys=True)
        )

    if generator == "dyngen":
        parameters = requested.get("simulator_parameters")
        if not isinstance(parameters, Mapping):
            raise BenchmarkConfigError("Requested dyngen parameters are missing")

        def non_negative_parameter(name: str) -> int:
            value = parameters.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BenchmarkConfigError(
                    f"Requested dyngen {name} must be a non-negative integer"
                )
            return value

        requested_tfs = non_negative_parameter("num_tfs")
        requested_targets = non_negative_parameter("num_targets")
        requested_housekeeping = non_negative_parameter("num_hks")
        if requested_tfs < 1 or requested_targets < 1:
            raise BenchmarkConfigError("Requested dyngen TF and target counts must be positive")
        if normalized_requested["transcription_factors"] != requested_tfs or (
            normalized_requested["genes"]
            != requested_tfs + requested_targets + requested_housekeeping
        ):
            raise BenchmarkConfigError(
                "Requested dyngen dimensions disagree with its simulator parameters"
            )
        if observed_dimensions["transcription_factors"] < requested_tfs or (
            observed_dimensions["genes"]
            != observed_dimensions["transcription_factors"]
            + requested_targets
            + requested_housekeeping
        ):
            raise BenchmarkConfigError(
                "Observed dyngen genes/TFs violate the declared backbone-augmentation contract"
            )
        observed_groups = observed_dimensions["groups"]
        requested_milestones = normalized_requested["groups"]
        if observed_groups < 2 or observed_groups > requested_milestones:
            raise BenchmarkConfigError(
                "Observed dyngen groups violate the declared milestone-occupancy contract: "
                f"observed={observed_groups}, requested_milestones={requested_milestones}"
            )

    return {
        "schema_version": "1.0",
        "contract": DATASET_MATERIALIZATION_CONTRACT,
        "policy": policy,
        "dimension_policy": dimension_policy,
        "status": "matched" if not deviations else "accepted_declared_deviation",
        "requested_dimensions": normalized_requested,
        "observed_dimensions": observed_dimensions,
        "dimension_deviations": deviations,
        "requested_capabilities": sorted(requested_capabilities),
        "observed_capabilities": observed_capabilities,
        "expression": {
            "genes": len(genes),
            "columns": len(columns),
            "column_kind": "cells",
        },
    }


def _materialized_dataset_view(
    requested: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    """Use observed axes downstream while retaining the prospective request."""

    observed = record.get("observed_dimensions")
    requested_dimensions = record.get("requested_dimensions")
    deviations = record.get("dimension_deviations")
    if not all(
        isinstance(value, Mapping) for value in (observed, requested_dimensions, deviations)
    ):
        raise BenchmarkConfigError("Generated dependency omits materialized dimensions")
    dimensions = dict(observed)
    materialization = record.get("materialization")
    if not isinstance(materialization, Mapping):
        raise BenchmarkConfigError("Generated dependency omits materialization evidence")
    return {
        **dict(requested),
        "dimensions": dimensions,
        "requested_dimensions": dict(requested_dimensions),
        "observed_dimensions": dimensions,
        "dimension_deviations": dict(deviations),
        "materialization_policy": materialization.get("policy"),
        "dimension_policy": dict(materialization.get("dimension_policy", {})),
        "materialization_status": materialization.get("status"),
        **dimensions,
        "tf_count": dimensions.get("transcription_factors"),
    }


def _neutralize_lineage_information(benchmark_manifest: Path) -> list[dict[str, Any]]:
    """Preserve lineage topology while removing simulator truth-derived rates."""

    benchmark = _json_file(benchmark_manifest, "benchmark manifest")
    transformations: list[dict[str, Any]] = []
    for record in benchmark.get("datasets", []):
        if not isinstance(record, Mapping):
            continue
        raw_dataset = record.get("dataset_manifest")
        raw_truth = record.get("ground_truth_manifest")
        if not isinstance(raw_dataset, str) or not isinstance(raw_truth, str):
            continue
        dataset_manifest = benchmark_manifest.parent / raw_dataset
        truth_manifest = benchmark_manifest.parent / raw_truth
        manifest = _json_file(dataset_manifest, "dataset manifest")
        extras = manifest.get("extras", {})
        lineage_raw = extras.get("lineage_tree") if isinstance(extras, Mapping) else None
        if not isinstance(lineage_raw, str):
            continue
        lineage = dataset_manifest.parent / lineage_raw
        with lineage.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
        if not {"child", "parent", "gain_rate", "loss_rate"} <= set(fields):
            raise BenchmarkConfigError(f"Lineage table has an unsupported schema: {lineage}")
        before = sha256_file(lineage)
        with lineage.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                root = not row.get("parent") or row.get("parent") in {"__root__", "NA"}
                row["gain_rate"] = "0" if root else "0.5"
                row["loss_rate"] = "0" if root else "0.5"
                writer.writerow(row)
        truth = _json_file(truth_manifest, "ground-truth manifest")
        truth["dataset_fingerprint"] = _fingerprint_dataset(dataset_manifest)
        atomic_write_json(
            truth_manifest,
            truth,
            "Ground-truth manifest with neutral lineage rates",
            boundary=benchmark_manifest.parent,
        )
        transformations.append(
            {
                "dataset_id": record.get("dataset_id"),
                "field": "lineage_tree.gain_rate/loss_rate",
                "before_sha256": before,
                "after_sha256": sha256_file(lineage),
                "policy": "root_zero_non_root_constant_0.5_topology_preserved",
            }
        )
    return transformations


def _write_network_recipe(path: Path, recipe: Mapping[str, Any]) -> None:
    genes = int(recipe["genes"])
    tf_count = int(recipe["transcription_factors"])
    edges = min(tf_count, int(recipe["edges_per_target"]))
    # scMultiSim requires ``num_genes`` to be strictly greater than the number
    # of genes named by an input GRN.  Naming exactly ``genes - 1`` here makes
    # its final expression matrix have the prospective ``genes`` columns while
    # retaining every declared TF in the candidate regulator universe.
    target_count = genes - tf_count - 1
    if not 1 <= tf_count < genes - 1 or edges < 1 or target_count < 1:
        raise BenchmarkConfigError("Invalid deterministic regulatory-network recipe")
    edge_slots = target_count * edges
    if edge_slots < tf_count:
        raise BenchmarkConfigError(
            "Regulatory-network recipe cannot give every transcription factor an edge"
        )
    rng = random.Random(int(recipe["seed"]))
    tf_width = max(2, len(str(tf_count)))
    target_width = max(3, len(str(target_count)))
    tfs = [f"TF{index:0{tf_width}d}" for index in range(1, tf_count + 1)]
    degrees = [1] * tf_count
    # Spread one mandatory occurrence of every TF across the edge table before
    # filling the remaining slots preferentially.  This avoids stochastic TF
    # loss (particularly in high-regulator panels) without changing the seeded
    # scale-free character of the rest of the topology.
    coverage_tfs = list(range(tf_count))
    rng.shuffle(coverage_tfs)
    coverage_positions = rng.sample(range(edge_slots), tf_count)
    required_by_position = dict(zip(coverage_positions, coverage_tfs, strict=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("target", "regulator", "effect"))
        for target_index in range(1, target_count + 1):
            chosen: list[int] = []
            first_slot = (target_index - 1) * edges
            for slot in range(first_slot, first_slot + edges):
                if slot in required_by_position:
                    chosen.append(required_by_position[slot])
            while len(chosen) < edges:
                regulator = rng.choices(range(tf_count), weights=degrees, k=1)[0]
                if regulator not in chosen:
                    chosen.append(regulator)
            for regulator in chosen:
                degrees[regulator] += 1
                magnitude = 0.4 + 1.2 * rng.random()
                sign = -1.0 if rng.random() < 0.35 else 1.0
                writer.writerow(
                    (f"G{target_index:0{target_width}d}", tfs[regulator], f"{sign * magnitude:.8g}")
                )


def _newick(groups: int) -> str:
    if groups < 1:
        raise BenchmarkConfigError("A population tree requires at least one group")

    labels = [f"pop_{index}" for index in range(1, groups + 1)]
    if groups == 1:
        return f"({labels[0]}:1);\n"
    tree = f"({labels[0]}:1,{labels[1]}:1)"
    for label in labels[2:]:
        tree = f"({tree}:1,{label}:1)"
    return tree + ";\n"


def _materialize_generated_inputs(root: Path, recipes: Mapping[str, Any]) -> None:
    for name, raw in sorted(recipes.items()):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise BenchmarkConfigError(f"Generated input {name!r} has no path recipe")
        path = (root / str(raw["path"])).resolve()
        if not path.is_relative_to(root.resolve()):
            raise BenchmarkConfigError(f"Generated input {name!r} escapes its request root")
        if raw.get("kind") == "directed_scale_free_v1":
            _write_network_recipe(path, raw)
        elif raw.get("kind") == "balanced_population_tree_v1":
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, _newick(int(raw["groups"])), "Population tree", boundary=root)
        else:
            raise BenchmarkConfigError(f"Unknown generated input recipe: {raw.get('kind')!r}")


def _spathi_content_tree(root: Path) -> str:
    """Hash exactly the SPATHI package sources copied into the benchmark image."""

    root = root.resolve()
    source = root / "src/spathi"
    project = root / "pyproject.toml"
    if not source.is_dir() or not project.is_file():
        raise BenchmarkConfigError(f"Invalid SPATHI source checkout: {root}")
    files = [project]
    files.extend(
        path
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    return canonical_json_sha256(
        {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    )


class AndreaExecutor:
    """Production implementation of ``TaskExecutor.execute_task``."""

    def __init__(
        self, *, repo_root: Path, andrea_repo: Path, spathi_repo: Path, study_root: Path
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.andrea_repo = andrea_repo.resolve()
        self.spathi_repo = spathi_repo.resolve()
        self.study_root = study_root.resolve()
        self.study_root.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore(self.study_root / "artifacts")
        self.logs = self.study_root / "logs"
        self._image: dict[str, Any] | None = None
        self._resolved_images: dict[str, dict[str, Any]] = {}

    def _route_threads(self, route: Mapping[str, Any], requested: int) -> int:
        return effective_route_threads(requested, route.get("max_threads"))

    @property
    def _image_lock_path(self) -> Path:
        return self.study_root / "resolved-images.json"

    def _freeze_image_identity(self, identity: Mapping[str, Any]) -> None:
        """Persist one tag-to-image binding for the lifetime of this study."""

        tag = str(identity["requested_image"])
        path = self._image_lock_path
        if path.is_file():
            payload = _json_file(path, "study image lock")
            if payload.get("schema_version") != "1.0" or not isinstance(
                payload.get("images"), Mapping
            ):
                raise BenchmarkConfigError("Study image lock has an invalid schema")
        else:
            payload = {"schema_version": "1.0", "images": {}}
        images = dict(payload["images"])
        frozen = images.get(tag)
        lock_identity = {
            "image_id": identity["image_id"],
            "repo_digest": identity.get("repo_digest"),
        }
        if frozen is not None and (
            not isinstance(frozen, Mapping) or frozen.get("image_id") != lock_identity["image_id"]
        ):
            raise BenchmarkConfigError(
                f"Docker image tag {tag!r} changed within this study: "
                f"{frozen!r} -> {lock_identity!r}. Start a new study ID or restore "
                "the originally resolved image digest."
            )
        if frozen is None:
            images[tag] = lock_identity
            atomic_write_json(
                path,
                {"schema_version": "1.0", "images": images},
                "Resolved Docker image lock",
                boundary=self.study_root,
            )

    def _docker_image_identity(
        self, image: str, *, freeze_mutable_tag: bool = True
    ) -> dict[str, Any]:
        """Resolve the exact local image ANDREA will execute, pulling only if absent."""

        if not image or image != image.strip():
            raise BenchmarkConfigError("Docker image reference must be non-empty canonical text")
        with _CATALOG_IMAGE_LOCK:
            inspected = subprocess.run(
                ["docker", "image", "inspect", image],
                check=False,
                capture_output=True,
                text=True,
            )
            if inspected.returncode != 0:
                stream_command(
                    ["docker", "pull", image],
                    cwd=self.repo_root,
                    log_path=self.logs / f"pull-{_slug(image, lower=True)}.log",
                )
                inspected = subprocess.run(
                    ["docker", "image", "inspect", image],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            if inspected.returncode != 0:
                raise BenchmarkConfigError(f"Docker image is unavailable after resolution: {image}")
            try:
                payload = json.loads(inspected.stdout)
            except json.JSONDecodeError as exc:
                raise BenchmarkConfigError(
                    f"Docker returned invalid inspection JSON for {image!r}"
                ) from exc
            if (
                not isinstance(payload, list)
                or len(payload) != 1
                or not isinstance(payload[0], Mapping)
            ):
                raise BenchmarkConfigError(
                    f"Docker returned ambiguous image identity for {image!r}"
                )
            image_id = payload[0].get("Id")
            repo_digests = payload[0].get("RepoDigests", [])
            if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
                raise BenchmarkConfigError(f"Docker image {image!r} has no immutable image ID")
            if not isinstance(repo_digests, list) or any(
                not isinstance(value, str) for value in repo_digests
            ):
                raise BenchmarkConfigError(f"Docker image {image!r} has invalid RepoDigests")
            identity = {
                "requested_image": image,
                "image_id": image_id,
                "repo_digest": sorted(repo_digests)[0] if repo_digests else None,
            }
            previous = self._resolved_images.get(image)
            if previous is not None and previous.get("image_id") != identity["image_id"]:
                raise BenchmarkConfigError(
                    f"Docker image tag {image!r} changed while this executor was active: "
                    f"{previous!r} -> {identity!r}"
                )
            if freeze_mutable_tag:
                self._freeze_image_identity(identity)
            self._resolved_images[image] = identity
            return dict(identity)

    def _inference_image_identities(
        self,
        tools_params: Mapping[str, Any],
        custom_tools: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        custom_by_id: dict[str, Mapping[str, Any]] = {}
        if custom_tools is not None:
            for raw in custom_tools.get("tools", []):
                if isinstance(raw, Mapping) and isinstance(raw.get("run_id"), str):
                    custom_by_id[f"custom_{raw['run_id']}"] = raw
        identities: dict[str, dict[str, Any]] = {}
        for raw in tools_params.get("runs", []):
            if not isinstance(raw, Mapping) or not isinstance(raw.get("tool_id"), str):
                continue
            tool_id = str(raw["tool_id"])
            if tool_id in identities:
                continue
            custom = custom_by_id.get(tool_id)
            if custom is not None:
                image = custom.get("docker_image")
                if not isinstance(image, str):
                    raise BenchmarkConfigError(f"Custom tool {tool_id!r} has no Docker image")
                identities[tool_id] = {
                    **self._docker_image_identity(image, freeze_mutable_tag=False),
                    "origin": "custom",
                    "tool_id": tool_id,
                    "definition_sha256": canonical_json_sha256(dict(custom)),
                }
                continue
            definition = catalog_definition_identity(
                self.andrea_repo, catalog_kind="tool", catalog_id=tool_id
            )
            image = definition["requested_image"]
            identities[tool_id] = {
                **self._docker_image_identity(image),
                "origin": "andrea_catalog",
                **definition,
            }
        return dict(sorted(identities.items()))

    def _generation_identity(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        simulator_id = str(dataset["generator"])
        if simulator_id == "scale-stress":
            return {
                "origin": "benchmark_host_generator",
                "generator_id": "spathi_scale_stress",
                "generator_contract": GENERATOR_CONTRACT,
            }
        definition = catalog_definition_identity(
            self.andrea_repo, catalog_kind="simulator", catalog_id=simulator_id
        )
        image = definition["requested_image"]
        return {
            **self._docker_image_identity(image),
            "origin": "andrea_simulator_catalog",
            **definition,
        }

    def build_spathi_image(self, source_identity: str) -> dict[str, Any]:
        adapter_files = (
            self.repo_root / "containers/spathi/Dockerfile",
            self.repo_root / "containers/spathi/constraints.txt",
            self.repo_root / "containers/spathi/run_tool.py",
            self.repo_root / "containers/spathi/run_convergence.py",
        )
        fingerprint = canonical_json_sha256(
            {
                "contract": EXECUTOR_CONTRACT,
                "source_identity": source_identity,
                "base_image": _BASE_IMAGE,
                "adapter_files": {
                    path.relative_to(self.repo_root).as_posix(): sha256_file(path)
                    for path in adapter_files
                },
            }
        )
        tag = f"spathi-benchmarks/spathi:{fingerprint[:16]}"
        with _IMAGE_LOCK:
            inspected = subprocess.run(
                ["docker", "image", "inspect", tag], check=False, capture_output=True, text=True
            )
            reused = False
            if inspected.returncode == 0:
                try:
                    inspected_payload = json.loads(inspected.stdout)
                    labels = inspected_payload[0].get("Config", {}).get("Labels", {})
                    reused = labels.get("dev.spathi.benchmark.build-fingerprint") == fingerprint
                except (AttributeError, IndexError, json.JSONDecodeError, TypeError):
                    reused = False
            if not reused:
                stream_command(
                    [
                        "docker",
                        "buildx",
                        "build",
                        "--load",
                        "--build-context",
                        f"spathi={self.spathi_repo}",
                        "--build-arg",
                        f"BASE_IMAGE={_BASE_IMAGE}",
                        "--build-arg",
                        f"SPATHI_COMMIT={source_identity}",
                        "--build-arg",
                        f"BENCH_BUILD_FINGERPRINT={fingerprint}",
                        "--file",
                        str(self.repo_root / "containers/spathi/Dockerfile"),
                        "--tag",
                        tag,
                        str(self.repo_root),
                    ],
                    cwd=self.repo_root,
                    log_path=self.logs / "build-spathi-image.log",
                )
            details = subprocess.run(
                ["docker", "image", "inspect", tag], check=False, capture_output=True, text=True
            )
        if details.returncode != 0:
            raise RuntimeError(f"SPATHI image is unavailable: {tag}")
        identity = self._docker_image_identity(tag, freeze_mutable_tag=False)
        return {
            "tag": tag,
            "id": identity["image_id"],
            "repo_digest": identity["repo_digest"],
            "fingerprint": fingerprint,
            "reused": reused,
        }

    def _spathi_image(self, _spec: Mapping[str, Any]) -> dict[str, Any]:
        if self._image is None:
            self._image = self.build_spathi_image(_spathi_content_tree(self.spathi_repo))
        return self._image

    def generate(self, dataset: Mapping[str, Any], resources: Mapping[str, Any]) -> dict[str, Any]:
        generator_identity = self._generation_identity(dataset)
        spec = {
            "contract": EXECUTOR_CONTRACT,
            "materialization_contract": DATASET_MATERIALIZATION_CONTRACT,
            "dataset": dict(dataset),
            "resources": dict(resources),
            "generator_identity": generator_identity,
        }
        key = self.store.key_for("dataset", spec)
        existing = self.store.lookup(key)
        if existing is not None:
            return self._dataset_reference(existing)
        with self.store.staging_directory(key) as staging:
            output = staging / "output"
            if dataset.get("generator") == "scale-stress":
                manifest = generate_scale_dataset(
                    output_root=output,
                    dataset_id=str(dataset["id"]),
                    seed=int(dataset["seed"]),
                    parameters=dict(dataset["simulator_parameters"]),
                )
                transformations: list[dict[str, Any]] = []
            else:
                request = staging / "request"
                request.mkdir()
                scenario = dict(dataset["scenario_request"])
                scenario["base_seed"] = int(dataset["seed"])
                _materialize_generated_inputs(request, dict(dataset.get("generated_inputs", {})))
                scenario_path = request / "scenario.json"
                atomic_write_json(scenario_path, scenario, "Scenario request", boundary=staging)
                runs = {
                    "schema_version": "1.0",
                    "runs": [
                        {
                            "run_id": _slug(str(dataset["id"]), lower=True),
                            "simulator_id": str(dataset["generator"]),
                            "replicates": 1,
                            "base_seed": int(dataset["seed"]),
                            "params": dict(dataset["simulator_parameters"]),
                            "native_outputs": [],
                        }
                    ],
                }
                runs_path = request / "simulator-runs.json"
                atomic_write_json(runs_path, runs, "Simulator runs", boundary=staging)
                cores = int(resources["cpu_threads"])
                memory_gb = int(resources["memory_bytes"]) / 1024**3
                andrea_command(
                    [
                        "generate-data",
                        "execute",
                        "--scenario",
                        str(scenario_path),
                        "--simulator-runs",
                        str(runs_path),
                        "--output-dir",
                        str(output),
                        "--max-parallel-tasks",
                        "1",
                        "--max-cores",
                        str(cores),
                        "--max-ram-gb",
                        f"{memory_gb:.6g}",
                    ],
                    repo_root=self.repo_root,
                    andrea_repo=self.andrea_repo,
                    log_path=self.logs / f"generate-{key.digest[:16]}.log",
                )
                manifest = _unique_file(output, "benchmark-manifest.json")
                transformations = _neutralize_lineage_information(manifest)
            materialization = _dataset_materialization_contract(manifest, dataset)
            atomic_write_json(
                staging / "generation-evidence.json",
                {
                    "generator_identity": generator_identity,
                    "lineage_transformations": transformations,
                    "materialization": materialization,
                },
                "Generation evidence",
                boundary=staging,
            )
            manifest_relative = manifest.relative_to(staging)
            artifact = _publish_existing(self.store, kind="dataset", spec=spec, staging=staging)
        return self._dataset_reference(artifact, manifest_relative=manifest_relative)

    def _dataset_reference(
        self, artifact: Artifact, *, manifest_relative: Path | None = None
    ) -> dict[str, Any]:
        manifest = (
            artifact.path / manifest_relative
            if manifest_relative
            else _unique_file(artifact.path, "benchmark-manifest.json")
        )
        return artifact_reference(
            artifact,
            members={
                "benchmark_manifest": manifest,
                "generation_evidence": _unique_file(artifact.path, "generation-evidence.json"),
            },
        )

    def datasets(self, reference: Mapping[str, Any]) -> list[dict[str, Any]]:
        manifest_path = resolve_artifact_reference(self.store, reference, "benchmark_manifest")
        manifest = _json_file(manifest_path, "benchmark manifest")
        evidence = _json_file(
            resolve_artifact_reference(self.store, reference, "generation_evidence"),
            "generation evidence",
        )
        materialization = evidence.get("materialization")
        if (
            not isinstance(materialization, Mapping)
            or materialization.get("contract") != DATASET_MATERIALIZATION_CONTRACT
            or not isinstance(materialization.get("requested_dimensions"), Mapping)
            or not isinstance(materialization.get("observed_dimensions"), Mapping)
            or not isinstance(materialization.get("dimension_deviations"), Mapping)
        ):
            raise BenchmarkConfigError("Generated artifact omits its materialization contract")
        raw_records = manifest.get("datasets")
        if not isinstance(raw_records, list) or not raw_records:
            raise BenchmarkConfigError("Benchmark manifest contains no datasets")
        records = []
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                raise BenchmarkConfigError("Benchmark dataset record must be an object")
            dataset_manifest = _manifest_file(
                manifest_path.parent,
                raw.get("dataset_manifest"),
                "Benchmark dataset manifest",
            )
            truth_manifest = _manifest_file(
                manifest_path.parent,
                raw.get("ground_truth_manifest"),
                "Benchmark ground-truth manifest",
            )
            records.append(
                {
                    "dataset_id": str(raw.get("dataset_id")),
                    "simulator_id": raw.get("simulator_id"),
                    "seed": raw.get("seed"),
                    "dataset_manifest": str(dataset_manifest.resolve()),
                    "ground_truth_manifest": str(truth_manifest.resolve()),
                    "benchmark_artifact": dict(reference),
                    "materialization": dict(materialization),
                    "requested_dimensions": dict(materialization["requested_dimensions"]),
                    "observed_dimensions": dict(materialization["observed_dimensions"]),
                    "dimension_deviations": dict(materialization["dimension_deviations"]),
                    "materialization_policy": materialization.get("policy"),
                    "dimension_policy": dict(materialization.get("dimension_policy", {})),
                    "materialization_status": materialization.get("status"),
                }
            )
        if len(records) != 1:
            raise BenchmarkConfigError("Each article task must materialize exactly one dataset")
        return records

    def infer(
        self,
        *,
        dataset_manifest: str,
        tools_params: Mapping[str, Any],
        custom_tools: Mapping[str, Any] | None,
        resources: Mapping[str, Any],
        purpose: str,
        respect_resource_decision: bool = False,
    ) -> dict[str, Any]:
        image_identities = self._inference_image_identities(tools_params, custom_tools)
        spec = {
            "contract": EXECUTOR_CONTRACT,
            "dataset_manifest": dataset_manifest,
            "tools_params": dict(tools_params),
            "custom_tools": dict(custom_tools) if custom_tools else None,
            "resources": dict(resources),
            "purpose": purpose,
            "respect_resource_decision": respect_resource_decision,
            "image_identities": image_identities,
        }
        key = self.store.key_for("inference", spec)
        existing = self.store.lookup(key)
        if existing is not None:
            return self._inference_reference(existing)
        with self.store.staging_directory(key) as staging:
            request = staging / "request"
            request.mkdir()
            tools_path = request / "tools-params.json"
            atomic_write_json(tools_path, dict(tools_params), "Tools parameters", boundary=staging)
            custom_path: Path | None = None
            if custom_tools:
                custom_path = request / "custom-tools.json"
                atomic_write_json(custom_path, dict(custom_tools), "Custom tools", boundary=staging)
            preflight = request / "preflight.json"
            command = [
                "infer-network",
                "preflight",
                "--dataset-manifest",
                dataset_manifest,
                "--tools-params",
                str(tools_path),
                "--output-json",
                str(preflight),
            ]
            if custom_path:
                command.extend(["--custom-tools", str(custom_path)])
            andrea_command(
                command,
                repo_root=self.repo_root,
                andrea_repo=self.andrea_repo,
                log_path=self.logs / f"preflight-{key.digest[:16]}.log",
            )
            preflight_payload = _json_file(preflight, "inference preflight")
            selected = preflight_payload.get("runs", {}).get("selected", [])
            if not isinstance(selected, list):
                raise BenchmarkConfigError("Inference preflight has invalid selected runs")
            if any(not isinstance(run_id, str) or not run_id for run_id in selected):
                raise BenchmarkConfigError("Inference preflight has invalid selected-run IDs")
            if len(selected) != len(set(selected)):
                raise BenchmarkConfigError("Inference preflight has duplicate selected-run IDs")
            if not selected:
                atomic_write_json(
                    staging / "decision.json",
                    {
                        "status": "no_contract_eligible_runs",
                        "preflight": preflight_payload,
                        "image_identities": image_identities,
                    },
                    "Inference decision",
                    boundary=staging,
                )
                artifact = _publish_existing(
                    self.store, kind="inference", spec=spec, staging=staging
                )
                return self._inference_reference(artifact)

            output = staging / "output"
            cores = int(resources["cpu_threads"])
            memory_gb = _andrea_ram_gb(resources)
            command = [
                "infer-network",
                "plan",
                "--dataset-manifest",
                dataset_manifest,
                "--tools-params",
                str(tools_path),
                "--output-dir",
                str(output),
                "--max-cores",
                str(cores),
                "--max-ram-gb",
                str(memory_gb),
                "--planner",
                "auto",
                "--planner-time-limit-seconds",
                "300",
            ]
            if custom_path:
                command.extend(["--custom-tools", str(custom_path)])
            andrea_command(
                command,
                repo_root=self.repo_root,
                andrea_repo=self.andrea_repo,
                log_path=self.logs / f"plan-{key.digest[:16]}.log",
            )
            plan_path = _unique_file(output, "plan.json")
            plan = _json_file(plan_path, "inference plan")
            resource_decisions = _planning_resource_decisions(plan, selected)
            infeasible = sorted(
                run_id
                for run_id, value in resource_decisions.items()
                if value.get("status") == "estimated_infeasible_time"
            )
            if respect_resource_decision and infeasible:
                requested_runs = list(tools_params.get("runs", []))
                eligible_runs = [
                    dict(run)
                    for run in requested_runs
                    if isinstance(run, Mapping)
                    and run.get("run_id") in selected
                    and run.get("run_id") not in infeasible
                ]
                if not eligible_runs:
                    atomic_write_json(
                        staging / "decision.json",
                        {
                            "status": "estimated_infeasible_time",
                            "infeasible_runs": infeasible,
                            "resource_decisions": resource_decisions,
                            "preflight": preflight_payload,
                            "image_identities": image_identities,
                        },
                        "Inference decision",
                        boundary=staging,
                    )
                    artifact = _publish_existing(
                        self.store, kind="inference", spec=spec, staging=staging
                    )
                    return self._inference_reference(artifact)

                # ANDREA resource decisions are advisory. Freeze their evidence,
                # then build a second executable plan containing only admitted
                # runs; an infeasible competitor is never silently executed.
                shutil.copyfile(plan_path, request / "admission-plan.json")
                atomic_write_json(
                    tools_path, {"runs": eligible_runs}, "Admitted tools", boundary=staging
                )
                if custom_path is not None and custom_tools is not None:
                    eligible_ids = {str(run["run_id"]) for run in eligible_runs}
                    admitted_custom = [
                        dict(item)
                        for item in custom_tools.get("tools", [])
                        if isinstance(item, Mapping) and item.get("run_id") in eligible_ids
                    ]
                    if admitted_custom:
                        atomic_write_json(
                            custom_path,
                            {"tools": admitted_custom},
                            "Admitted custom tools",
                            boundary=staging,
                        )
                    else:
                        custom_path = None
                shutil.rmtree(output)
                command = [
                    "infer-network",
                    "plan",
                    "--dataset-manifest",
                    dataset_manifest,
                    "--tools-params",
                    str(tools_path),
                    "--output-dir",
                    str(output),
                    "--max-cores",
                    str(cores),
                    "--max-ram-gb",
                    str(memory_gb),
                    "--planner",
                    "auto",
                    "--planner-time-limit-seconds",
                    "300",
                ]
                if custom_path:
                    command.extend(["--custom-tools", str(custom_path)])
                andrea_command(
                    command,
                    repo_root=self.repo_root,
                    andrea_repo=self.andrea_repo,
                    log_path=self.logs / f"plan-admitted-{key.digest[:16]}.log",
                )
                plan_path = _unique_file(output, "plan.json")
                admitted_plan = _json_file(plan_path, "admitted inference plan")
                admitted_ids = [str(run["run_id"]) for run in eligible_runs]
                admitted_decisions = _planning_resource_decisions(admitted_plan, admitted_ids)
                newly_infeasible = sorted(
                    run_id
                    for run_id, value in admitted_decisions.items()
                    if value["status"] == "estimated_infeasible_time"
                )
                if newly_infeasible:
                    raise BenchmarkConfigError(
                        "Admitted inference plan unexpectedly became infeasible: "
                        + ", ".join(newly_infeasible)
                    )

            failed: CommandExecutionError | None = None
            try:
                andrea_command(
                    ["infer-network", "run", "--run-dir", str(plan_path.parent)],
                    repo_root=self.repo_root,
                    andrea_repo=self.andrea_repo,
                    log_path=self.logs / f"infer-{key.digest[:16]}.log",
                )
            except CommandExecutionError as exc:
                failed = exc
            reports = sorted(output.rglob("run_report.json"))
            if len(reports) != 1:
                if failed:
                    raise failed
                raise BenchmarkConfigError(f"Inference produced {len(reports)} run reports")
            report = _json_file(reports[0], "inference run report")
            outcomes = report.get("tools", {}).get("resource_outcomes", {})
            atomic_write_json(
                staging / "decision.json",
                {
                    "status": (
                        "completed_with_failures"
                        if failed is not None
                        else ("completed_with_omissions" if infeasible else "completed")
                    ),
                    "infeasible_runs": infeasible,
                    "resource_decisions": resource_decisions,
                    "resource_outcomes": outcomes,
                    "preflight": preflight_payload,
                    "image_identities": image_identities,
                },
                "Inference decision",
                boundary=staging,
            )
            artifact = _publish_existing(self.store, kind="inference", spec=spec, staging=staging)
        return self._inference_reference(artifact)

    @staticmethod
    def _inference_reference(artifact: Artifact) -> dict[str, Any]:
        members = {"decision": _unique_file(artifact.path, "decision.json")}
        reports = sorted(artifact.path.rglob("run_report.json"))
        if reports:
            if len(reports) != 1:
                raise BenchmarkConfigError("Cached inference has ambiguous run reports")
            members["run_report"] = reports[0]
        return artifact_reference(artifact, members=members)

    def evaluate(self, *, run_report: str, truth_manifest: str) -> dict[str, Any]:
        original_truth = Path(truth_manifest).resolve()
        truth_payload = _json_file(original_truth, "ground-truth manifest")
        candidate_space = _candidate_space_identity(original_truth)
        spec = {
            "contract": EXECUTOR_CONTRACT,
            "run_report": run_report,
            "truth_manifest": str(original_truth),
            "truth": truth_payload,
            "candidate_space": candidate_space,
        }
        key = self.store.key_for("evaluation", spec)
        existing = self.store.lookup(key)
        if existing is not None:
            return self._evaluation_reference(existing)
        with self.store.staging_directory(key) as staging:
            output = staging / "output"
            andrea_command(
                [
                    "evaluate-inference",
                    "--run-report",
                    run_report,
                    "--ground-truth-manifest",
                    str(original_truth),
                    "--output-dir",
                    str(output),
                    "--no-view",
                ],
                repo_root=self.repo_root,
                andrea_repo=self.andrea_repo,
                log_path=self.logs / f"evaluate-{key.digest[:16]}.log",
            )
            report_path = _unique_file(output, "evaluation_report.json")
            _unique_file(output, "metrics.csv")
            evidence = _candidate_space_evidence(
                candidate_space,
                _json_file(report_path, "evaluation report"),
            )
            atomic_write_json(
                staging / "candidate-space-evidence.json",
                evidence,
                "Candidate-space evidence",
                boundary=staging,
            )
            artifact = _publish_existing(self.store, kind="evaluation", spec=spec, staging=staging)
        return self._evaluation_reference(artifact)

    @staticmethod
    def _evaluation_reference(artifact: Artifact) -> dict[str, Any]:
        return artifact_reference(
            artifact,
            members={
                "evaluation_report": _unique_file(artifact.path, "evaluation_report.json"),
                "metrics": _unique_file(artifact.path, "metrics.csv"),
                "candidate_space_evidence": _unique_file(
                    artifact.path, "candidate-space-evidence.json"
                ),
            },
        )

    @staticmethod
    def _complete_spathi_params(
        parameters: Mapping[str, Any], *, n_estimators: int, random_seed: int
    ) -> dict[str, Any]:
        return {
            **dict(parameters),
            "bandwidth": "auto",
            "pca_svd_solver": "auto",
            "bootstrap": None,
            "n_estimators": n_estimators,
            "random_seed": random_seed,
        }

    @staticmethod
    def _custom_definition(run_id: str, image: str, threads: int) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "name": f"SPATHI {run_id}",
            "docker_image": image,
            "execution_mode": "group_native",
            "extra_inputs": ["groups", "tf_list"],
            "outputs": {"directed": True, "sign": "none"},
            "runtime_resources": {
                "threading": {
                    "supported": True,
                    "default_threads": threads,
                    "max_threads": None,
                    "upstream_mapping": "cli:--threads",
                }
            },
        }

    @staticmethod
    def _report_measurements(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        tools = report.get("tools", {})
        results = tools.get("results", {}) if isinstance(tools, Mapping) else {}
        outcomes = tools.get("resource_outcomes", {}) if isinstance(tools, Mapping) else {}
        measured: dict[str, dict[str, Any]] = {}
        if not isinstance(results, Mapping):
            return measured
        for run_id, raw in results.items():
            value = raw if isinstance(raw, Mapping) else {}
            measurement = value.get("measurement", {})
            outcome = outcomes.get(run_id, {}) if isinstance(outcomes, Mapping) else {}
            measured[str(run_id)] = {
                "duration_seconds": (
                    measurement.get("wall_time_seconds")
                    if isinstance(measurement, Mapping)
                    else value.get("duration_seconds")
                ),
                "cpu_time_seconds": (
                    measurement.get("cpu_time_seconds")
                    if isinstance(measurement, Mapping)
                    else None
                ),
                "peak_memory_bytes": (
                    measurement.get("peak_memory_bytes")
                    if isinstance(measurement, Mapping)
                    else None
                ),
                "status": (
                    outcome.get("status", value.get("status"))
                    if isinstance(outcome, Mapping)
                    else value.get("status")
                ),
                "reason": (
                    outcome.get("reason") if isinstance(outcome, Mapping) else value.get("error")
                ),
            }
        return measured

    def _metric_rows(
        self,
        evaluation: Mapping[str, Any],
        *,
        dataset: Mapping[str, Any],
        runs: Mapping[str, Mapping[str, Any]],
        family: str | None = None,
        level: str = "directed",
        report: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        path = resolve_artifact_reference(self.store, evaluation, "metrics")
        candidate_evidence = _json_file(
            resolve_artifact_reference(self.store, evaluation, "candidate_space_evidence"),
            "candidate-space evidence",
        )
        candidate_identity = candidate_evidence.get("identity")
        candidate_evaluation = candidate_evidence.get("evaluation")
        if not isinstance(candidate_identity, Mapping) or not isinstance(
            candidate_evaluation, Mapping
        ):
            raise BenchmarkConfigError("Evaluation candidate-space evidence is incomplete")
        candidate_counts = candidate_evaluation.get("n_candidates_by_level", {})
        excluded_rows = candidate_evaluation.get(
            "n_truth_rows_outside_candidate_space_by_level", {}
        )
        excluded_edges = candidate_evaluation.get(
            "n_truth_edges_outside_candidate_space_by_level", {}
        )
        if not all(
            isinstance(value, Mapping)
            for value in (candidate_counts, excluded_rows, excluded_edges)
        ):
            raise BenchmarkConfigError("Evaluation candidate-space counts are incomplete")
        with path.open(encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        measured = self._report_measurements(report or {})
        values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        prediction_audit: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for raw in raw_rows:
            run_id = str(raw.get("tool_id") or raw.get("run_id") or "")
            if run_id not in runs or raw.get("level") != level:
                continue
            for field in (
                "n_prediction_rows",
                "n_prediction_rows_outside_candidate_space",
                "n_predicted_edges",
                "n_predicted_edges_outside_candidate_space",
            ):
                observed_count = _finite(raw.get(field))
                if observed_count is not None:
                    prediction_audit[run_id][field].append(int(observed_count))
            if raw.get("status") != "ok":
                continue
            for metric, column in _METRIC_FIELDS.items():
                value = _finite(raw.get(column))
                if value is not None:
                    values[run_id][metric].append(value)
        rows: list[dict[str, Any]] = []
        for run_id, metadata in runs.items():
            for metric in _METRIC_FIELDS:
                observed = values[run_id].get(metric, [])
                if not observed:
                    continue
                row = {
                    "dataset_id": dataset["id"],
                    "condition_id": dataset["condition_id"],
                    "dataset_role": dataset["role"],
                    "simulator": dataset["generator"],
                    "replicate": dataset["replicate"],
                    **dict(dataset.get("observed_dimensions", dataset.get("dimensions", {}))),
                    "tf_count": dataset.get(
                        "tf_count",
                        dataset.get("dimensions", {}).get("transcription_factors"),
                    ),
                    "requested_dimensions": dataset.get("requested_dimensions"),
                    "dimension_deviations": dataset.get("dimension_deviations"),
                    "method_id": metadata.get("method_id", "spathi"),
                    "configuration_id": metadata.get("configuration_id"),
                    "algorithm_seed": metadata.get("algorithm_seed"),
                    "metric": metric,
                    "value": fmean(observed),
                    "contexts": len(observed),
                    "status": "completed",
                    "duration_seconds": measured.get(run_id, {}).get("duration_seconds"),
                    "peak_memory_bytes": measured.get(run_id, {}).get("peak_memory_bytes"),
                    "parameters": metadata.get("parameters"),
                    "requested_threads": metadata.get("requested_threads"),
                    "effective_threads": metadata.get("effective_threads"),
                    "evaluation_level": level,
                    "candidate_space_sha256": candidate_identity.get("candidate_space_sha256"),
                    "candidate_sources_sha256": candidate_identity.get("sources_sha256"),
                    "candidate_targets_sha256": candidate_identity.get("targets_sha256"),
                    "n_candidate_sources": candidate_identity.get("n_sources"),
                    "n_candidate_targets": candidate_identity.get("n_targets"),
                    "n_candidates": candidate_counts.get(level),
                    "n_truth_rows_outside_candidate_space": excluded_rows.get(level),
                    "n_truth_edges_outside_candidate_space": excluded_edges.get(level),
                    **{
                        field: (
                            sum(prediction_audit[run_id][field])
                            if prediction_audit[run_id].get(field)
                            else None
                        )
                        for field in (
                            "n_prediction_rows",
                            "n_prediction_rows_outside_candidate_space",
                            "n_predicted_edges",
                            "n_predicted_edges_outside_candidate_space",
                        )
                    },
                }
                if family is not None:
                    row["family"] = family
                if "n_estimators" in metadata:
                    row["n_estimators"] = metadata["n_estimators"]
                rows.append(row)
        return rows

    @staticmethod
    def _require_comparable_candidate_spaces(rows: Sequence[Mapping[str, Any]]) -> None:
        """Reject a metric block that silently mixes evaluation universes."""

        by_dataset: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row.get("status") != "completed" or row.get("metric") not in _METRIC_FIELDS:
                continue
            dataset_id = row.get("dataset_id")
            identity = row.get("candidate_space_sha256")
            if not isinstance(dataset_id, str) or not isinstance(identity, str):
                raise BenchmarkConfigError(
                    "Completed metric rows must preserve dataset and candidate-space identity"
                )
            by_dataset[dataset_id].add(identity)
        mismatches = {
            dataset_id: sorted(identities)
            for dataset_id, identities in by_dataset.items()
            if len(identities) != 1
        }
        if mismatches:
            raise BenchmarkConfigError(
                "Comparable metric rows use different candidate spaces: "
                + json.dumps(mismatches, sort_keys=True)
            )

    @staticmethod
    def _generated_dependency(dependencies: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
        matches = [
            value for value in dependencies.values() if value.get("kind") == "generate_dataset"
        ]
        if len(matches) != 1:
            raise BenchmarkConfigError("Execution task requires exactly one generated dataset")
        return matches[0]

    @staticmethod
    def _dependency(dependencies: Mapping[str, Mapping[str, Any]], kind: str) -> Mapping[str, Any]:
        matches = [value for value in dependencies.values() if value.get("kind") == kind]
        if len(matches) != 1:
            raise BenchmarkConfigError(f"Task requires exactly one {kind} dependency")
        return matches[0]

    @staticmethod
    def _dataset_inputs(dataset_manifest: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
        payload = _json_file(dataset_manifest, "dataset manifest")
        dataset = payload.get("dataset")
        extras = payload.get("extras")
        if not isinstance(dataset, Mapping) or not isinstance(extras, Mapping):
            raise BenchmarkConfigError("Dataset manifest omits dataset or extras")
        expression_raw = dataset.get("expression_matrix")
        groups_raw = extras.get("groups")
        tf_raw = extras.get("tf_list")
        if not all(isinstance(value, str) for value in (expression_raw, groups_raw, tf_raw)):
            raise BenchmarkConfigError("SPATHI convergence requires expression, groups and tf_list")
        return (
            (dataset_manifest.parent / str(expression_raw)).resolve(),
            (dataset_manifest.parent / str(groups_raw)).resolve(),
            (dataset_manifest.parent / str(tf_raw)).resolve(),
            dict(dataset),
        )

    @staticmethod
    def _group_contexts(groups: Path) -> list[str]:
        with groups.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if (
                header is None
                or len(header) < 2
                or header[0].strip().lower()
                not in {"cell", "column", "sample", "observation", "id"}
                or header[1].strip().lower() not in {"cluster", "group", "cell_type", "label"}
            ):
                raise BenchmarkConfigError("groups.tsv has a non-canonical schema")
            labels = {row[1].strip() for row in reader if len(row) >= 2 and row[1].strip()}
        if not labels:
            raise BenchmarkConfigError("groups.tsv contains no group labels")
        return [f"group:{label}" for label in sorted(labels)]

    @staticmethod
    def _merge_prefix_network(source: Path, output: Path, run_id: str) -> tuple[int, list[str]]:
        with source.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = ("source", "target", "score", "sign", "evidence", "context")
            if tuple(reader.fieldnames or ()) != expected:
                raise BenchmarkConfigError("SPATHI prefix network has an invalid schema")
            rows = [dict(row) for row in reader]
        scores = [float(row["score"]) for row in rows]
        minimum = min(scores) if scores else 0.0
        maximum = max(scores) if scores else 0.0
        fields = (*expected, "tool_id")
        output.mkdir(parents=True)
        with (
            (output / "merged_network_raw.csv").open("x", encoding="utf-8", newline="") as raw,
            (output / "merged_network_normalized.csv").open(
                "x", encoding="utf-8", newline=""
            ) as normalized,
        ):
            raw_writer = csv.DictWriter(raw, fieldnames=fields, lineterminator="\n")
            norm_writer = csv.DictWriter(normalized, fieldnames=fields, lineterminator="\n")
            raw_writer.writeheader()
            norm_writer.writeheader()
            for row in rows:
                canonical = {
                    **row,
                    "sign": row.get("sign", "").strip() or "?",
                    "evidence": row.get("evidence", "").strip() or "model",
                }
                raw_writer.writerow({**canonical, "tool_id": run_id})
                value = float(row["score"])
                score = (value - minimum) / (maximum - minimum) if maximum > minimum else 1.0
                norm_writer.writerow({**canonical, "score": f"{score:.12g}", "tool_id": run_id})
        return len(rows), sorted({str(row["context"]) for row in rows})

    @staticmethod
    def _prefix_run_report(
        *,
        output: Path,
        run_id: str,
        dataset_id: str,
        dataset: Mapping[str, Any],
        fingerprint: Mapping[str, Any],
        contexts: Sequence[str],
        rows: int,
        duration_seconds: float | None,
        n_estimators: int,
        physical_digest: str,
    ) -> dict[str, Any]:
        spec = dataset.get("spec", {})
        expression = spec.get("expression", {}) if isinstance(spec, Mapping) else {}
        measurement = {
            "schema_version": "1.0",
            "scope": "logical_shared_forest_prefix",
            "wall_time_seconds": duration_seconds,
            "container_wall_time_seconds": None,
            "cpu_time_seconds": None,
            "peak_memory_bytes": None,
            "output_bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
            "io_read_bytes": None,
            "io_write_bytes": None,
            "assigned_resources": {},
            "telemetry": {
                "status": "partial",
                "wall_source": "unavailable_shared_physical_fit",
                "memory_source": None,
                "oom_killed": False,
            },
        }
        outcome = {
            "schema_version": "1.0",
            "status": "completed",
            "reason": "execution_completed",
            "exit_code": 0,
        }
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "executed",
            "output_profile": "canonical",
            "dataset": {
                "id": dataset_id,
                "fingerprint": dict(fingerprint),
                "column_kind": expression.get("column_kind", "cells"),
                "expression_profile": expression.get("expression_profile", "scrna"),
                "genes": expression.get("genes"),
                "columns": expression.get("columns"),
            },
            "tools": {
                "selected": [run_id],
                "catalog_tool_ids": {run_id: f"custom_{run_id}"},
                "tool_origins": {run_id: "custom"},
                "output_capabilities": {
                    run_id: {
                        "tool_origin": "custom",
                        "catalog_tool_id": f"custom_{run_id}",
                        "directed": True,
                        "sign": "none",
                    }
                },
                "skipped": {},
                "status_by_tool": {run_id: "completed"},
                "resource_decisions": {},
                "resource_outcomes": {run_id: outcome},
                "physical_resource_outcomes": {},
                "completed": [run_id],
                "completed_contexts": {run_id: list(contexts)},
                "failed": {},
                "results": {
                    run_id: {
                        "tool_id": run_id,
                        "status": "completed",
                        "exit_code": 0,
                        "execution": {"mode": "group_native"},
                        "measurement": measurement,
                        "resource_outcome": outcome,
                        "network_path": "merged_network_raw.csv",
                        "progress_path": None,
                        "logs_path": None,
                        "error": None,
                        "warnings": [],
                        "physical_tasks_total": 1,
                        "forest_prefix": {
                            "n_estimators": n_estimators,
                            "physical_artifact": f"sha256:{physical_digest}",
                        },
                    }
                },
            },
            "outputs": {
                "merged_network_raw": "merged_network_raw.csv",
                "merged_network_normalized": "merged_network_normalized.csv",
                "rows_per_tool": {run_id: rows},
            },
            "issues": [],
            "execution": {
                "elapsed_seconds": duration_seconds,
                "physical_fits": 1,
                "logical_results": 4,
                "measurement": measurement,
            },
        }

    def _run_convergence_forest(
        self,
        *,
        record: Mapping[str, Any],
        configuration_id: str,
        parameters: Mapping[str, Any],
        seed: int,
        prefixes: Sequence[int],
        image: Mapping[str, Any],
        threads: int,
        memory_bytes: int,
        purpose: str,
    ) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
        if len(prefixes) != 4:
            raise BenchmarkConfigError("Exact convergence validation requires four prefixes")
        model = self._complete_spathi_params(
            parameters, n_estimators=int(prefixes[-1]), random_seed=seed
        )
        physical_id = _slug(f"{configuration_id}__seed_{seed}__forest")
        logical = []
        metadata: dict[str, Mapping[str, Any]] = {}
        for trees in prefixes:
            run_id = _slug(f"{configuration_id}__trees_{trees}__seed_{seed}")
            complete = {**model, "n_estimators": int(trees)}
            logical.append(
                {
                    "run_id": run_id,
                    "configuration_id": configuration_id,
                    "configuration_sha256": canonical_json_sha256(complete),
                    "n_estimators": int(trees),
                }
            )
            metadata[run_id] = {
                "method_id": "spathi",
                "configuration_id": configuration_id,
                "algorithm_seed": seed,
                "parameters": complete,
                "n_estimators": int(trees),
            }
        plan = {
            "schema_version": "1.0",
            "physical_run_id": physical_id,
            "parent_configuration_id": configuration_id,
            "algorithm_seed": seed,
            "model": model,
            "logical_prefixes": logical,
        }
        spec = {
            "contract": f"{EXECUTOR_CONTRACT}:exact-forest-prefix-v1",
            "dataset_manifest": record["dataset_manifest"],
            "ground_truth_manifest": record["ground_truth_manifest"],
            "plan": plan,
            "image": {
                "requested_image": image["tag"],
                "image_id": image["id"],
                "repo_digest": image.get("repo_digest"),
                "source_fingerprint": image["fingerprint"],
            },
            "threads": threads,
            "memory_bytes": memory_bytes,
            "purpose": purpose,
        }
        key = self.store.key_for("convergence", spec)
        existing = self.store.lookup(key)
        if existing is not None:
            members = {
                item["run_id"]: existing.path / item["run_id"] / "run_report.json"
                for item in logical
            }
            members["physical_evidence"] = existing.path / "physical-evidence.json"
            return artifact_reference(existing, members=members), metadata

        dataset_manifest = Path(str(record["dataset_manifest"]))
        expression, groups, tf_list, dataset = self._dataset_inputs(dataset_manifest)
        expected_contexts = self._group_contexts(groups)
        truth = _json_file(Path(str(record["ground_truth_manifest"])), "ground-truth manifest")
        fingerprint = truth.get("dataset_fingerprint")
        dataset_id = truth.get("dataset_id")
        if not isinstance(fingerprint, Mapping) or not isinstance(dataset_id, str):
            raise BenchmarkConfigError("Ground truth omits dataset identity")
        with self.store.staging_directory(key) as staging:
            runtime = Path(tempfile.mkdtemp(prefix="runtime-", dir=staging))
            request = runtime / "request"
            extra = runtime / "extra"
            output = runtime / "output"
            request.mkdir()
            extra.mkdir()
            output.mkdir()
            atomic_write_json(
                request / "convergence-plan.json", plan, "Convergence plan", boundary=staging
            )
            atomic_write_json(
                request / "execution.json",
                {"mode": "group_native"},
                "Convergence execution",
                boundary=staging,
            )
            shutil.copyfile(groups, extra / "groups.tsv")
            shutil.copyfile(tf_list, extra / "tf_list.txt")
            uid = os.getuid() if hasattr(os, "getuid") else 0
            gid = os.getgid() if hasattr(os, "getgid") else 0
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cpus",
                str(threads),
                "--memory",
                str(memory_bytes),
                "--memory-swap",
                str(memory_bytes),
                "--pids-limit",
                "4096",
                "--user",
                f"{uid}:{gid}",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=1073741824",
                "--volume",
                f"{runtime.resolve()}:/io:rw",
                "--volume",
                f"{expression.resolve()}:/io/input.tsv:ro",
                "--entrypoint",
                "python",
                str(image["tag"]),
                "/usr/local/bin/run_spathi_convergence.py",
                "--input",
                "/io/input.tsv",
                "--params",
                "/io/request/convergence-plan.json",
                "--extra",
                "/io/extra",
                "--output-dir",
                "/io/output",
                "--threads",
                str(threads),
            ]
            started = perf_counter_ns()
            stream_command(
                command,
                cwd=self.repo_root,
                log_path=self.logs / f"convergence-{key.digest[:16]}.log",
            )
            physical_seconds = (perf_counter_ns() - started) / 1_000_000_000
            manifest = _json_file(output / "fanout-manifest.json", "prefix fan-out manifest")
            outputs = manifest.get("logical_outputs")
            if manifest.get("status") != "complete" or not isinstance(outputs, list):
                raise BenchmarkConfigError("Convergence runner returned an incomplete fan-out")
            by_run = {str(item["run_id"]): item for item in outputs if isinstance(item, Mapping)}
            for item in logical:
                run_id = str(item["run_id"])
                prefix = by_run.get(run_id)
                if (
                    not isinstance(prefix, Mapping)
                    or prefix.get("n_estimators") != item["n_estimators"]
                ):
                    raise BenchmarkConfigError("Convergence fan-out changed a logical prefix")
                prefix_root = output / str(prefix["relative_path"])
                metadata_payload = _json_file(prefix_root / "run_metadata.json", "prefix metadata")
                if metadata_payload.get("logical_runtime_seconds") is not None:
                    raise BenchmarkConfigError(
                        "Exact-prefix runner must not attribute shared forest time to a prefix"
                    )
                logical_root = staging / run_id
                row_count, observed_contexts = self._merge_prefix_network(
                    prefix_root / "network.csv", logical_root, run_id
                )
                unexpected = sorted(set(observed_contexts) - set(expected_contexts))
                if unexpected:
                    raise BenchmarkConfigError(
                        f"Convergence prefix emitted undeclared contexts: {unexpected}"
                    )
                report = self._prefix_run_report(
                    output=logical_root,
                    run_id=run_id,
                    dataset_id=dataset_id,
                    dataset=dataset,
                    fingerprint=fingerprint,
                    contexts=expected_contexts,
                    rows=row_count,
                    duration_seconds=None,
                    n_estimators=int(item["n_estimators"]),
                    physical_digest=key.digest,
                )
                atomic_write_json(
                    logical_root / "run_report.json", report, "Prefix run report", boundary=staging
                )
            atomic_write_json(
                staging / "physical-evidence.json",
                {
                    "schema_version": "1.0",
                    "physical_fits": 1,
                    "logical_results": len(logical),
                    "wall_time_seconds": physical_seconds,
                    "image_identity": {
                        "requested_image": image["tag"],
                        "image_id": image["id"],
                        "repo_digest": image.get("repo_digest"),
                        "source_fingerprint": image["fingerprint"],
                    },
                    "plan": plan,
                    "runner_manifest": manifest,
                },
                "Physical convergence evidence",
                boundary=staging,
            )
            shutil.rmtree(runtime)
            artifact = _publish_existing(self.store, kind="convergence", spec=spec, staging=staging)
        members = {
            item["run_id"]: artifact.path / item["run_id"] / "run_report.json" for item in logical
        }
        members["physical_evidence"] = artifact.path / "physical-evidence.json"
        return artifact_reference(artifact, members=members), metadata

    def _run_convergence_panel(
        self,
        *,
        spec: Mapping[str, Any],
        dependencies: Mapping[str, Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        record = self._generated_dependency(dependencies)["record"]
        materialized_dataset = _materialized_dataset_view(spec["dataset"], record)
        resources = dict(spec["resources"])
        cores = int(resources["cpu_threads"])
        memory = int(resources["memory_bytes"])
        jobs = [
            (candidate, int(seed)) for candidate in candidates for seed in spec["algorithm_seeds"]
        ]
        minimum_worker_memory = 512 * 1024**2
        workers = max(
            1,
            min(
                len(jobs),
                cores // 8 or 1,
                max(1, memory // minimum_worker_memory),
            ),
        )
        threads = max(1, cores // workers)
        memory_each = max(1, memory // workers)
        image = self._spathi_image(spec)

        def run(
            job: tuple[Mapping[str, Any], int],
        ) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
            candidate, seed = job
            return self._run_convergence_forest(
                record=record,
                configuration_id=str(candidate["id"]),
                parameters=dict(candidate["parameters"]),
                seed=seed,
                prefixes=tuple(int(value) for value in spec["exact_prefixes"]),
                image=image,
                threads=threads,
                memory_bytes=memory_each,
                purpose=f"calibration-validation-{spec['dataset']['id']}",
            )

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spathi-prefix") as pool:
            forests = list(pool.map(run, jobs))
        rows: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        for reference, metadata in forests:
            physical = _json_file(
                resolve_artifact_reference(self.store, reference, "physical_evidence"),
                "physical convergence evidence",
            )
            plan = physical.get("plan", {})
            reference = {
                **reference,
                "physical_measurement": {
                    "physical_run_id": plan.get("physical_run_id"),
                    "configuration_id": plan.get("parent_configuration_id"),
                    "algorithm_seed": plan.get("algorithm_seed"),
                    "wall_time_seconds": physical.get("wall_time_seconds"),
                    "logical_prefixes": len(plan.get("logical_prefixes", [])),
                    "runtime_semantics": "shared_physical_fit_not_attributable_to_prefix",
                    "image_identity": physical.get("image_identity"),
                },
            }
            references.append(reference)
            for run_id, run_metadata in metadata.items():
                report_path = resolve_artifact_reference(self.store, reference, run_id)
                report = _json_file(report_path, "prefix run report")
                evaluation = self.evaluate(
                    run_report=str(report_path),
                    truth_manifest=str(record["ground_truth_manifest"]),
                )
                rows.extend(
                    self._metric_rows(
                        evaluation,
                        dataset=materialized_dataset,
                        runs={run_id: run_metadata},
                        report=report,
                    )
                )
        expected = len(jobs) * len(spec["exact_prefixes"]) * len(_METRIC_FIELDS)
        if len(rows) != expected:
            raise BenchmarkConfigError("Exact-prefix validation omitted required metrics")
        return rows, references

    def _run_spathi_panel(
        self,
        *,
        spec: Mapping[str, Any],
        dependencies: Mapping[str, Mapping[str, Any]],
        configurations: Sequence[tuple[str, Mapping[str, Any], int, int]],
        purpose: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        generated = self._generated_dependency(dependencies)
        record = generated["record"]
        materialized_dataset = _materialized_dataset_view(spec["dataset"], record)
        resources = dict(spec["resources"])
        threads = max(1, min(8, int(resources["cpu_threads"])))
        image = self._spathi_image(spec)
        runs = []
        custom = []
        metadata: dict[str, Mapping[str, Any]] = {}
        for configuration_id, parameters, trees, seed in configurations:
            run_id = _slug(f"{configuration_id}__trees_{trees}__seed_{seed}")
            complete = self._complete_spathi_params(
                parameters, n_estimators=trees, random_seed=seed
            )
            runs.append(
                {
                    "run_id": run_id,
                    "tool_id": f"custom_{run_id}",
                    "params": complete,
                    "execution": {"mode": "group_native"},
                    "resources": {
                        "threads": threads,
                        **(
                            {"timeout_seconds": int(spec["timeout_seconds"])}
                            if spec.get("timeout_seconds") is not None
                            else {}
                        ),
                    },
                }
            )
            custom.append(self._custom_definition(run_id, str(image["tag"]), threads))
            metadata[run_id] = {
                "method_id": "spathi",
                "configuration_id": configuration_id,
                "algorithm_seed": seed,
                "parameters": complete,
                "n_estimators": trees,
            }
        inference = self.infer(
            dataset_manifest=str(record["dataset_manifest"]),
            tools_params={"runs": runs},
            custom_tools={"tools": custom},
            resources=resources,
            purpose=purpose,
        )
        if "run_report" not in inference.get("members", {}):
            raise BenchmarkConfigError(f"{purpose} produced no evaluable inference report")
        report_path = resolve_artifact_reference(self.store, inference, "run_report")
        report = _json_file(report_path, "inference run report")
        evaluation = self.evaluate(
            run_report=str(report_path), truth_manifest=str(record["ground_truth_manifest"])
        )
        rows = self._metric_rows(
            evaluation, dataset=materialized_dataset, runs=metadata, report=report
        )
        observed = {
            (row["configuration_id"], row["algorithm_seed"], row["n_estimators"], row["metric"])
            for row in rows
        }
        expected = {
            (configuration_id, seed, trees, metric)
            for configuration_id, _parameters, trees, seed in configurations
            for metric in _METRIC_FIELDS
        }
        if expected - observed:
            raise BenchmarkConfigError(f"{purpose} omitted required directed metrics")
        return rows, inference

    def execute_task(
        self, task: TaskRecord, dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        expected_definitions = task.spec.get("catalog_definitions")
        if not isinstance(expected_definitions, Mapping):
            raise BenchmarkConfigError(
                f"Task {task.task_id!r} omits its effective ANDREA catalog definitions"
            )
        verify_catalog_definitions(self.andrea_repo, expected_definitions)
        handlers = {
            "generate_dataset": self._execute_generate,
            "calibration_screen": self._execute_calibration_screen,
            "select_calibration_finalists": self._execute_select_finalists,
            "calibration_validate": self._execute_calibration_validate,
            "select_calibration_winner": self._execute_select_winner,
            "comparison_dataset": self._execute_comparison,
            "comparison_statistics": self._execute_statistics,
            "scalability_measurement": self._execute_scalability,
            "scalability_summary": self._execute_scalability_summary,
        }
        try:
            handler = handlers[task.kind]
        except KeyError as exc:
            raise BenchmarkConfigError(f"Unknown article task kind: {task.kind}") from exc
        result = dict(handler(task.spec, dependencies))
        result["kind"] = task.kind
        return result

    def _execute_generate(
        self, spec: Mapping[str, Any], _dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        dataset = dict(spec["dataset"])
        reference = self.generate(dataset, dict(spec["resources"]))
        record = self.datasets(reference)[0]
        evidence = _json_file(
            resolve_artifact_reference(self.store, reference, "generation_evidence"),
            "generation evidence",
        )
        generator_identity = evidence.get("generator_identity", {})
        materialized = _materialized_dataset_view(dataset, record)
        return {
            "artifact": reference,
            "record": record,
            "summary": {
                "dataset_id": dataset["id"],
                "condition_id": dataset["condition_id"],
                "simulator": dataset["generator"],
                "dataset_role": dataset["role"],
                "generator_identity": generator_identity,
                "materialization": record["materialization"],
                "requested_dimensions": materialized["requested_dimensions"],
                "observed_dimensions": materialized["observed_dimensions"],
                "dimension_deviations": materialized["dimension_deviations"],
                "image_identities": (
                    {f"simulator:{dataset['generator']}": generator_identity}
                    if isinstance(generator_identity, Mapping)
                    and generator_identity.get("requested_image")
                    else {}
                ),
                **dict(materialized["observed_dimensions"]),
                "tf_count": materialized["tf_count"],
            },
            "rows": [],
        }

    def _execute_calibration_screen(
        self, spec: Mapping[str, Any], dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        configurations = [
            (
                str(candidate["id"]),
                dict(candidate["parameters"]),
                int(spec["n_estimators"]),
                int(seed),
            )
            for candidate in spec["candidates"]
            for seed in spec["algorithm_seeds"]
        ]
        rows, inference = self._run_spathi_panel(
            spec=spec,
            dependencies=dependencies,
            configurations=configurations,
            purpose=f"calibration-screen-{spec['dataset']['id']}",
        )
        decision = _json_file(
            resolve_artifact_reference(self.store, inference, "decision"),
            "calibration inference decision",
        )
        return {
            "rows": rows,
            "inference": inference,
            "summary": {
                "fits": len(configurations),
                "image_identities": decision.get("image_identities", {}),
            },
        }

    def _execute_select_finalists(
        self, spec: Mapping[str, Any], dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        rows = [row for value in dependencies.values() for row in value.get("rows", [])]
        self._require_comparable_candidate_spaces(rows)
        by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("metric") == "aupr":
                by_candidate[str(row["configuration_id"])].append(row)
        expected = {str(value) for value in spec["candidate_ids"]}
        if set(by_candidate) != expected:
            raise BenchmarkConfigError("Calibration screening has an incomplete candidate matrix")
        ranking = []
        for candidate_id in sorted(expected):
            values = by_candidate[candidate_id]
            scores = [float(row["value"]) for row in values]
            durations = [
                value
                for row in values
                if (value := _finite(row.get("duration_seconds"))) is not None
            ]
            ranking.append(
                {
                    "configuration_id": candidate_id,
                    "mean_aupr": fmean(scores),
                    "stability_sd": pstdev(scores),
                    "mean_duration_seconds": fmean(durations) if durations else None,
                    "parameters": values[0]["parameters"],
                }
            )
        ranking.sort(
            key=lambda row: (
                -row["mean_aupr"],
                row["mean_duration_seconds"] is None,
                row["mean_duration_seconds"] or 0.0,
                row["stability_sd"],
                row["configuration_id"],
            )
        )
        count = int(spec["protocol"]["finalists"])
        selected = ranking[:count]
        return {
            "rows": ranking,
            "selected_candidates": [
                {"id": row["configuration_id"], "parameters": row["parameters"]} for row in selected
            ],
            "decision": {
                "decision": "advance_equal_candidates",
                "selected": [row["configuration_id"] for row in selected],
            },
            "summary": {"candidate_count": len(ranking), "finalist_count": count},
        }

    def _execute_calibration_validate(
        self, spec: Mapping[str, Any], dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        finalists = self._dependency(dependencies, "select_calibration_finalists")
        rows, forests = self._run_convergence_panel(
            spec=spec,
            dependencies=dependencies,
            candidates=finalists["selected_candidates"],
        )
        physical_fits = len(finalists["selected_candidates"]) * len(spec["algorithm_seeds"])
        spathi_identity: dict[str, Any] | None = None
        for reference in forests:
            measurement = reference.get("physical_measurement")
            identity = (
                measurement.get("image_identity") if isinstance(measurement, Mapping) else None
            )
            if not isinstance(identity, Mapping):
                continue
            observed = dict(identity)
            if spathi_identity is not None and spathi_identity != observed:
                raise BenchmarkConfigError(
                    "Calibration forests were produced by different SPATHI images"
                )
            spathi_identity = observed
        image_identities = {"custom_spathi": spathi_identity} if spathi_identity is not None else {}
        return {
            "rows": rows,
            "forest_artifacts": forests,
            "summary": {
                "physical_fits": physical_fits,
                "logical_results": physical_fits * len(spec["exact_prefixes"]),
                "prefixes": list(spec["exact_prefixes"]),
                "prefix_semantics": "one shared deterministic forest per candidate and seed",
                "image_identities": image_identities,
                "shared_fit_measurements": [
                    dict(reference["physical_measurement"])
                    for reference in forests
                    if isinstance(reference.get("physical_measurement"), Mapping)
                ],
            },
        }

    def _execute_select_winner(
        self, spec: Mapping[str, Any], dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        rows = [row for value in dependencies.values() for row in value.get("rows", [])]
        self._require_comparable_candidate_spaces(rows)
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("metric") == "aupr":
                grouped[(str(row["configuration_id"]), int(row["n_estimators"]))].append(row)
        if not grouped:
            raise BenchmarkConfigError("Calibration validation produced no AUPR observations")
        ranking = []
        for (candidate, trees), values in grouped.items():
            scores = [float(row["value"]) for row in values]
            by_dataset: dict[str, list[float]] = defaultdict(list)
            for row in values:
                by_dataset[str(row["dataset_id"])].append(float(row["value"]))
            dataset_means = [fmean(by_dataset[key]) for key in sorted(by_dataset)]
            ranking.append(
                {
                    "configuration_id": candidate,
                    "n_estimators": trees,
                    "mean_aupr": fmean(scores),
                    "stability_sd": pstdev(dataset_means),
                    "stability_unit": "validation_dataset_after_seed_averaging",
                    "parameters": values[0]["parameters"],
                }
            )
        best = max(row["mean_aupr"] for row in ranking)
        threshold = best * (1.0 - float(spec["protocol"]["practical_equivalence_fraction"]))
        admissible = [row for row in ranking if row["mean_aupr"] >= threshold]
        admissible.sort(
            key=lambda row: (
                row["stability_sd"],
                row["n_estimators"],
                -row["mean_aupr"],
                row["configuration_id"],
            )
        )
        winner = dict(admissible[0])
        parameters = dict(winner["parameters"])
        parameters.pop("random_seed", None)
        parameters["n_estimators"] = winner["n_estimators"]
        ranking.sort(
            key=lambda row: (
                -row["mean_aupr"],
                row["stability_sd"],
                row["n_estimators"],
                row["configuration_id"],
            )
        )
        return {
            "rows": ranking,
            "winner": {
                "configuration_id": winner["configuration_id"],
                "n_estimators": winner["n_estimators"],
                "parameters": parameters,
                "mean_aupr": winner["mean_aupr"],
            },
            "decision": {
                "decision": "selected_configuration",
                "selected": winner["configuration_id"],
                "n_estimators": winner["n_estimators"],
                "practical_equivalence_threshold": threshold,
            },
            "summary": {
                "primary_metric": "aupr",
                "evaluated_configurations": len(ranking),
            },
        }

    def _comparison_requests(
        self, spec: Mapping[str, Any], winner: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Mapping[str, Any]]]:
        default_threads = max(1, min(8, int(spec["resources"]["cpu_threads"])))
        requested_threads = int(spec.get("effective_threads", default_threads))
        image: dict[str, Any] | None = None
        runs = []
        custom = []
        metadata: dict[str, Mapping[str, Any]] = {}
        for route in spec["routes"]:
            threads = self._route_threads(route, requested_threads)
            run_id = _slug(str(route["id"]))
            seed = (
                int.from_bytes(
                    hashlib.sha256(f"{spec['dataset']['seed']}\0{run_id}".encode()).digest()[:4],
                    "big",
                )
                % 2_147_483_646
                + 1
            )
            if route["tool_id"] == "spathi":
                image = image or self._spathi_image(spec)
                parameters = dict(winner["parameters"])
                trees = int(parameters.pop("n_estimators"))
                params = self._complete_spathi_params(
                    parameters, n_estimators=trees, random_seed=seed
                )
                tool_id = f"custom_{run_id}"
                custom.append(self._custom_definition(run_id, str(image["tag"]), threads))
            else:
                params = dict(route["parameters"])
                seed_parameter = route.get("seed_parameter")
                if isinstance(seed_parameter, str):
                    params[seed_parameter] = seed
                tool_id = str(route["tool_id"])
            runs.append(
                {
                    "run_id": run_id,
                    "tool_id": tool_id,
                    "params": params,
                    "execution": {"mode": route["delivery"]},
                    "resources": {
                        "threads": threads,
                        **(
                            {"timeout_seconds": int(spec["timeout_seconds"])}
                            if spec.get("timeout_seconds") is not None
                            else {}
                        ),
                    },
                }
            )
            metadata[run_id] = {
                "method_id": route["id"],
                "configuration_id": route["configuration_id"],
                "algorithm_seed": seed if route.get("seed_parameter") else None,
                "parameters": params,
                "output_semantics": route["output_semantics"],
                "requested_threads": requested_threads,
                "effective_threads": threads,
            }
        return {"runs": runs}, ({"tools": custom} if custom else None), metadata

    @staticmethod
    def _comparison_statuses(
        *,
        metadata: Mapping[str, Mapping[str, Any]],
        admission: Mapping[str, Any],
        report: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        """Partition every requested logical run into one canonical outcome.

        Preflight exclusions, planner infeasibility decisions and physical-run
        outcomes are mutually exclusive.  Requiring a complete partition keeps
        failed methods in the denominator without inventing an outcome when
        ANDREA evidence is malformed or incomplete.
        """

        statuses: dict[str, dict[str, Any]] = {}

        def register(run_id: Any, outcome: Mapping[str, Any]) -> None:
            key = str(run_id)
            if key not in metadata:
                raise BenchmarkConfigError(f"Comparison evidence references undeclared run {key!r}")
            if key in statuses:
                raise BenchmarkConfigError(
                    f"Comparison run {key!r} has more than one resource outcome"
                )
            status = outcome.get("status")
            if not isinstance(status, str) or not status:
                raise BenchmarkConfigError(
                    f"Comparison run {key!r} has no canonical resource status"
                )
            statuses[key] = dict(outcome)

        preflight = admission.get("preflight", {})
        runs = preflight.get("runs", {}) if isinstance(preflight, Mapping) else {}
        skipped = runs.get("skipped", {}) if isinstance(runs, Mapping) else {}
        if not isinstance(skipped, Mapping):
            raise BenchmarkConfigError("Comparison preflight has invalid skipped-run evidence")
        for run_id, reason in skipped.items():
            register(
                run_id,
                {
                    "status": "contract_ineligible",
                    "reason": str(reason),
                },
            )

        infeasible = admission.get("infeasible_runs", [])
        if not isinstance(infeasible, list):
            raise BenchmarkConfigError("Comparison admission has invalid infeasible-run evidence")
        for run_id in infeasible:
            register(
                run_id,
                {
                    "status": "estimated_infeasible_time",
                    "reason": "empirical_runtime_estimate_exceeds_timeout",
                },
            )

        if report is not None:
            tools = report.get("tools")
            outcomes = tools.get("resource_outcomes") if isinstance(tools, Mapping) else None
            if not isinstance(outcomes, Mapping):
                raise BenchmarkConfigError("Comparison run report has invalid resource outcomes")
            for run_id, outcome in outcomes.items():
                if not isinstance(outcome, Mapping):
                    raise BenchmarkConfigError(
                        f"Comparison run {run_id!r} has an invalid resource outcome"
                    )
                register(run_id, outcome)

        missing = sorted(set(metadata).difference(statuses))
        if missing:
            raise BenchmarkConfigError(
                "Comparison evidence omits requested run outcomes: " + ", ".join(missing)
            )
        return statuses

    @staticmethod
    def _comparison_coverage(
        *,
        spec: Mapping[str, Any],
        metadata: Mapping[str, Mapping[str, Any]],
        statuses: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Project logical run outcomes into every applicable competitor family."""

        coverage: list[dict[str, Any]] = []
        dataset = spec["dataset"]
        for family in spec["families"]:
            family_id = str(family["id"])
            declared = {
                run_id: value
                for run_id, value in metadata.items()
                if value.get("method_id") in family["route_ids"]
            }
            # Conditional-information families are absent when the dataset did
            # not expose enough inputs to admit both sides of the comparison.
            if len({str(value.get("method_id")) for value in declared.values()}) < 2:
                continue
            for run_id, value in declared.items():
                outcome = statuses[run_id]
                coverage.append(
                    {
                        "dataset_id": dataset["id"],
                        "condition_id": dataset.get("condition_id"),
                        "replicate": dataset.get("replicate"),
                        "family": family_id,
                        "method_id": value.get("method_id", run_id),
                        "configuration_id": value.get("configuration_id"),
                        "status": outcome["status"],
                        "reason": outcome.get("reason"),
                        "requested_threads": value.get("requested_threads"),
                        "effective_threads": value.get("effective_threads"),
                    }
                )
        return sorted(
            coverage,
            key=lambda row: (
                str(row["dataset_id"]),
                str(row["family"]),
                str(row["method_id"]),
            ),
        )

    def _execute_comparison(
        self, spec: Mapping[str, Any], dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        generated = self._generated_dependency(dependencies)
        record = generated["record"]
        materialized_dataset = _materialized_dataset_view(spec["dataset"], record)
        winner = self._dependency(dependencies, "select_calibration_winner")["winner"]
        tools, custom, metadata = self._comparison_requests(spec, winner)
        inference = self.infer(
            dataset_manifest=str(record["dataset_manifest"]),
            tools_params=tools,
            custom_tools=custom,
            resources=dict(spec["resources"]),
            purpose=f"comparison-{spec['dataset']['id']}",
            respect_resource_decision=spec.get("timeout_seconds") is not None,
        )
        admission = _json_file(
            resolve_artifact_reference(self.store, inference, "decision"),
            "comparison admission decision",
        )
        if "run_report" not in inference.get("members", {}):
            statuses = self._comparison_statuses(
                metadata=metadata,
                admission=admission,
                report=None,
            )
            failures = [
                {
                    "dataset_id": spec["dataset"]["id"],
                    "method_id": metadata[run_id].get("method_id", run_id),
                    **outcome,
                }
                for run_id, outcome in statuses.items()
            ]
            coverage = self._comparison_coverage(
                spec=spec,
                metadata=metadata,
                statuses=statuses,
            )
            return {
                "rows": [],
                "failures": failures,
                "decisions": [
                    {"decision": "omit_before_execution", **failure} for failure in failures
                ],
                "execution_coverage": coverage,
                "inference": inference,
                "evaluations": {},
                "summary": {
                    "methods_requested": len(metadata),
                    "methods_completed": 0,
                    "metric_rows": 0,
                    "image_identities": admission.get("image_identities", {}),
                },
            }
        report_path = resolve_artifact_reference(self.store, inference, "run_report")
        report = _json_file(report_path, "comparison run report")
        report_tools = report.get("tools")
        if not isinstance(report_tools, Mapping) or not isinstance(
            report_tools.get("completed"), list
        ):
            raise BenchmarkConfigError("Comparison run report omits completed-run inventory")
        completed_runs = {
            str(run_id) for run_id in report_tools["completed"] if isinstance(run_id, str)
        }
        statuses = self._comparison_statuses(
            metadata=metadata,
            admission=admission,
            report=report,
        )
        completed_from_outcomes = {
            run_id
            for run_id, outcome in statuses.items()
            if outcome.get("status") in {"completed", "completed_with_warnings"}
        }
        if completed_runs != completed_from_outcomes:
            raise BenchmarkConfigError(
                "Comparison completed-run inventory disagrees with resource outcomes"
            )
        coverage = self._comparison_coverage(
            spec=spec,
            metadata=metadata,
            statuses=statuses,
        )
        failures = [
            {
                "dataset_id": spec["dataset"]["id"],
                "method_id": metadata[run_id].get("method_id", run_id),
                **outcome,
            }
            for run_id, outcome in statuses.items()
            if outcome.get("status") not in {"completed", "completed_with_warnings"}
        ]
        if not completed_runs:
            decisions = [
                {
                    "decision": (
                        "omit_before_execution"
                        if failure.get("status")
                        in {"contract_ineligible", "estimated_infeasible_time"}
                        else "execution_did_not_complete"
                    ),
                    **failure,
                }
                for failure in failures
            ]
            return {
                "rows": [],
                "failures": failures,
                "decisions": decisions,
                "execution_coverage": coverage,
                "inference": inference,
                "evaluations": {},
                "summary": {
                    "methods_requested": len(metadata),
                    "methods_completed": 0,
                    "metric_rows": 0,
                    "resource_outcomes": dict(statuses),
                    "image_identities": admission.get("image_identities", {}),
                },
            }
        rows: list[dict[str, Any]] = []
        evaluations: dict[str, Any] = {}
        candidate_spaces: dict[str, Any] = {}
        for family in spec["families"]:
            family_id = str(family["id"])
            declared_members = {
                run_id: value
                for run_id, value in metadata.items()
                if value["method_id"] in family["route_ids"]
            }
            # A conditional-information family exists on this dataset only
            # when at least two of its declared methods are contract-eligible.
            # Publishing a SPATHI-only row for (for example) an order-aware
            # family on a dataset without order would distort descriptive
            # coverage even though paired statistics later reject that block.
            if len({str(value["method_id"]) for value in declared_members.values()}) < 2:
                continue
            members = {
                run_id: value
                for run_id, value in declared_members.items()
                if run_id in completed_runs
            }
            if not members:
                continue
            semantics = {str(value["output_semantics"]) for value in declared_members.values()}
            level = "topology" if "undirected_topology" in semantics else "directed"
            evaluation = self.evaluate(
                run_report=str(report_path),
                truth_manifest=str(record["ground_truth_manifest"]),
            )
            evaluations[family_id] = evaluation
            candidate_spaces[family_id] = _json_file(
                resolve_artifact_reference(self.store, evaluation, "candidate_space_evidence"),
                "candidate-space evidence",
            )
            rows.extend(
                self._metric_rows(
                    evaluation,
                    dataset=materialized_dataset,
                    runs=members,
                    family=family_id,
                    level=level,
                    report=report,
                )
            )
        return {
            "rows": rows,
            "failures": failures,
            "decisions": [
                {"decision": "omit_before_execution", **failure}
                for failure in failures
                if failure.get("status")
                in {
                    "contract_ineligible",
                    "estimated_infeasible_time",
                }
            ],
            "execution_coverage": coverage,
            "inference": inference,
            "evaluations": evaluations,
            "summary": {
                "methods_requested": len(metadata),
                "methods_completed": len(completed_runs),
                "metric_rows": len(rows),
                "candidate_spaces": candidate_spaces,
                "image_identities": admission.get("image_identities", {}),
            },
        }

    def _execute_statistics(
        self, spec: Mapping[str, Any], dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        rows = [row for value in dependencies.values() for row in value.get("rows", [])]
        self._require_comparable_candidate_spaces(rows)
        families = {str(item["id"]): list(item["route_ids"]) for item in spec["families"]}
        protocol = spec["statistics"]
        family_statistics = [
            analyze_paired_statistics(
                [row for row in rows if row.get("family") == family],
                metrics=list(protocol["metrics"]),
                families={family: methods},
                block_fields=("dataset_id",),
                alpha=float(protocol["alpha"]),
                min_complete_blocks=3,
                random_seed=13_371,
            )
            for family, methods in families.items()
        ]
        if not family_statistics:
            raise BenchmarkConfigError("Comparison statistics declare no method families")
        statistics = {
            **family_statistics[0],
            "analyses": [
                analysis for result in family_statistics for analysis in result["analyses"]
            ],
        }
        ranking_rows = [
            {"family": analysis["family"], "metric": analysis["metric"], **rank}
            for analysis in statistics["analyses"]
            for rank in analysis["rankings"]
        ]
        return {
            "rows": [],
            "statistics": statistics,
            "decisions": [{"decision": "primary_family", "selected": protocol["primary_family"]}],
            "summary": {
                "primary_family": protocol["primary_family"],
                "estimand": dict(protocol["estimand"]),
                "families": [dict(family) for family in spec["families"]],
                "rankings": ranking_rows,
                "statistics": statistics,
            },
        }

    @staticmethod
    def _blocked_by_predecessor(
        dependencies: Mapping[str, Mapping[str, Any]], advance_dependency: Any
    ) -> Mapping[str, Any] | None:
        if not isinstance(advance_dependency, str):
            return None
        previous = dependencies.get(advance_dependency)
        if not isinstance(previous, Mapping):
            raise BenchmarkConfigError("Scalability predecessor evidence is missing")
        row = next(iter(previous.get("rows", [])), {})
        return row if row.get("status") not in {"completed", "completed_with_warnings"} else None

    def _execute_scalability(
        self, spec: Mapping[str, Any], dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        route = spec["route"]
        requested_threads = int(spec["requested_threads"])
        effective_threads = int(spec["effective_threads"])
        ram_gb = _andrea_ram_gb(spec["resources"])
        if effective_threads != self._route_threads(route, requested_threads):
            raise BenchmarkConfigError(
                "Scalability task effective threads differ from its frozen route contract"
            )
        generated = self._generated_dependency(dependencies)
        record = generated["record"]
        materialized_dataset = _materialized_dataset_view(spec["dataset"], record)
        blocked = self._blocked_by_predecessor(dependencies, spec.get("advance_dependency"))
        base = {
            "dataset_id": spec["dataset"]["id"],
            "scale_id": spec["dataset"]["condition_id"],
            "method_id": route["id"],
            "measurement": spec["measurement"],
            "requested_threads": requested_threads,
            "effective_threads": effective_threads,
            "threads": effective_threads,
            "ram_gb": ram_gb,
            **dict(materialized_dataset["observed_dimensions"]),
            "tf_count": materialized_dataset["tf_count"],
            "requested_dimensions": materialized_dataset["requested_dimensions"],
            "dimension_deviations": materialized_dataset["dimension_deviations"],
        }
        if blocked is not None:
            predecessor_status = blocked.get("predecessor_status", blocked.get("status"))
            predecessor_reason = blocked.get("predecessor_reason", blocked.get("reason"))
            row = {
                **base,
                "status": "not_attempted",
                "reason": "desisted_after_andrea_resource_outcome",
                "predecessor_status": predecessor_status,
                "predecessor_reason": predecessor_reason,
            }
            return {
                "rows": [row],
                "decisions": [{"decision": "not_attempted_after_resource_outcome", **row}],
                "failures": [row],
            }
        winner = self._dependency(dependencies, "select_calibration_winner")["winner"]
        isolated_spec = {**dict(spec), "routes": [route]}
        tools, custom, metadata = self._comparison_requests(isolated_spec, winner)
        tools["runs"][0]["resources"] = {
            "threads": effective_threads,
            # Scalability is an isolated per-method measurement. Give every
            # method the same frozen host RAM ceiling; otherwise ANDREA's
            # conservative no-profile fallback would cap custom SPATHI at 4 GiB
            # and turn the large-scale frontier into an artificial OOM test.
            "ram_gb": ram_gb,
            "timeout_seconds": int(spec["timeout_seconds"]),
        }
        inference = self.infer(
            dataset_manifest=str(record["dataset_manifest"]),
            tools_params=tools,
            custom_tools=custom,
            resources=dict(spec["resources"]),
            purpose=(f"scalability-{spec['dataset']['id']}-{route['id']}-{effective_threads}"),
            respect_resource_decision=True,
        )
        decision = _json_file(
            resolve_artifact_reference(self.store, inference, "decision"),
            "inference decision",
        )
        base = {**base, "image_identities": decision.get("image_identities", {})}
        if "run_report" not in inference.get("members", {}):
            status = str(decision["status"])
            row = {**base, "status": status, "reason": status}
            return {
                "rows": [row],
                "decisions": [{"decision": "desist", **row}],
                "failures": [row],
                "inference": inference,
                "summary": {"image_identities": decision.get("image_identities", {})},
            }
        report = _json_file(
            resolve_artifact_reference(self.store, inference, "run_report"),
            "scalability run report",
        )
        run_id = next(iter(metadata))
        measurement = self._report_measurements(report).get(run_id, {})
        row = {
            **base,
            "status": measurement.get("status", "runtime_failure"),
            "reason": measurement.get("reason"),
            "duration_seconds": measurement.get("duration_seconds"),
            "cpu_time_seconds": measurement.get("cpu_time_seconds"),
            "peak_memory_bytes": measurement.get("peak_memory_bytes"),
        }
        completed = row["status"] in {"completed", "completed_with_warnings"}
        return {
            "rows": [row],
            "decisions": [{"decision": "advance" if completed else "desist", **row}],
            "failures": [] if completed else [row],
            "inference": inference,
            "summary": {"image_identities": decision.get("image_identities", {})},
        }

    def _execute_scalability_summary(
        self, spec: Mapping[str, Any], dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        rows = [row for value in dependencies.values() for row in value.get("rows", [])]
        return {
            "rows": [],
            "decisions": [dict(value) for value in spec.get("thread_resolution", [])],
            "failures": [],
            "summary": {
                "measurements": len(rows),
                "completed": sum(
                    row.get("status") in {"completed", "completed_with_warnings"} for row in rows
                ),
                "resource_capacity": dict(spec["resource_capacity"]),
            },
        }


__all__ = [
    "EXECUTOR_CONTRACT",
    "AndreaExecutor",
    "artifact_reference",
    "effective_route_threads",
    "resolve_artifact_reference",
]
