"""Combine ANDREA evaluation metrics with configuration-level provenance."""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from .artifacts import Artifact, ArtifactKey, ArtifactStore, sha256_file
from .calibration_contexts import DatasetGroupContract, load_dataset_group_contract
from .config import BenchmarkConfigError, load_json_object
from .evaluation_identity import EvaluationSpaceIdentity, load_evaluation_space_identity
from .execution_records import unique_execution_records
from .pipeline import _batch_selects_dataset, validate_inference_attribution_contract
from .provenance import frozen_path, frozen_profile, validate_run_inputs
from .safe_io import path_below_no_symlinks
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
    "dataset_fingerprint_sha256",
    "candidate_source_universe_sha256",
    "candidate_target_universe_sha256",
    "candidate_edge_universe_sha256",
    "truth_edge_universe_sha256",
    "inference_artifact",
    "evaluation_artifact",
)
SUMMARY_METRICS = (
    "auroc",
    "aupr",
    "f1_at_truth_count",
    "epr_at_truth_count",
    "invariant_edge_stability",
)
SUMMARY_SCHEMA_VERSION = "1.0"


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


def _reported_artifact_member(
    artifact: Artifact,
    report_path: Path,
    raw_path: Any,
    label: str,
) -> Path | None:
    if raw_path is None:
        return None
    if not isinstance(raw_path, str) or not raw_path:
        raise BenchmarkConfigError(f"{label} must be a non-empty relative path")
    candidate = (report_path.parent / raw_path).resolve()
    if not candidate.is_relative_to(artifact.path.resolve()) or not candidate.is_file():
        raise BenchmarkConfigError(f"{label} escapes or is missing from its inference artifact")
    try:
        return artifact.file(candidate.relative_to(artifact.path.resolve()).as_posix()).resolve()
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} is not declared by its inference artifact") from exc


def _invariant_edge_stability(
    network_path: Path | None,
    metrics: list[dict[str, Any]],
    *,
    candidate_sources: frozenset[str],
    candidate_targets: frozenset[str],
    allow_self_edges: bool,
) -> float | None:
    """Conservative pairwise top-k overlap without arbitrary tie breaking.

    ``k`` is the common number of directed truth edges in each invariant group.
    Only edges strictly above the kth-score boundary are identifiable members of
    top-k.  Boundary ties are not resolved lexicographically, so a flat network has
    zero stability instead of perfect but arbitrary overlap.  Missing predictions
    have the same zero score used by ANDREA's sparse evaluator.  The denominator
    remains ``k`` so unresolved slots and missing predictions are penalized.
    """

    truth_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    for metric in metrics:
        context = str(metric.get("context", ""))
        count = metric.get("n_truth_edges")
        if (
            metric.get("level") == "directed"
            and context.startswith("group:")
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
        ):
            truth_counts[context] = count
            candidate_count = metric.get("n_candidates")
            if (
                isinstance(candidate_count, int)
                and not isinstance(candidate_count, bool)
                and candidate_count >= count
            ):
                candidate_counts[context] = candidate_count
    if (
        network_path is None
        or len(truth_counts) < 2
        or set(candidate_counts) != set(truth_counts)
        or len(set(truth_counts.values())) != 1
        or len(set(candidate_counts.values())) != 1
    ):
        return None
    truth_count = next(iter(truth_counts.values()))
    candidate_count = next(iter(candidate_counts.values()))
    by_context: dict[str, dict[tuple[str, str], float]] = {context: {} for context in truth_counts}
    try:
        with network_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"source", "target", "score", "context"}
            if not required.issubset(reader.fieldnames or ()):  # pragma: no cover - defensive
                raise BenchmarkConfigError(
                    f"Inference network lacks columns required for invariant stability: {network_path}"
                )
            for row in reader:
                context = str(row.get("context", ""))
                if context not in by_context:
                    continue
                source = str(row.get("source", ""))
                target = str(row.get("target", ""))
                if (
                    not source
                    or not target
                    or (not allow_self_edges and source == target)
                    or source not in candidate_sources
                    or target not in candidate_targets
                ):
                    continue
                try:
                    score = float(row["score"])
                except (TypeError, ValueError) as exc:
                    raise BenchmarkConfigError(
                        f"Inference network contains an invalid score: {network_path}"
                    ) from exc
                if not math.isfinite(score):
                    raise BenchmarkConfigError(
                        f"Inference network contains a non-finite score: {network_path}"
                    )
                edge = (source, target)
                previous = by_context[context].get(edge)
                if previous is None or score > previous:
                    by_context[context][edge] = score
    except OSError as exc:
        raise BenchmarkConfigError(f"Cannot read inference network {network_path}: {exc}") from exc

    selected: dict[str, set[tuple[str, str]]] = {}
    for context, scores in by_context.items():
        implicit_count = candidate_count - len(scores)
        if implicit_count < 0:  # pragma: no cover - protected by candidate filtering
            raise BenchmarkConfigError(
                f"Inference network has more unique edges than its candidate universe: {context}"
            )
        # At most k implicit zeros are needed to locate the kth order statistic.
        ranked_scores = sorted(
            [*scores.values(), *([0.0] * min(implicit_count, truth_count))],
            reverse=True,
        )
        boundary = ranked_scores[truth_count - 1]
        certain = {edge for edge, score in scores.items() if score > boundary}
        boundary_edges = {edge for edge, score in scores.items() if score == boundary}
        implicit_boundary = implicit_count if boundary == 0.0 else 0
        available_at_boundary = len(boundary_edges) + implicit_boundary
        remaining_slots = truth_count - len(certain)
        if available_at_boundary == remaining_slots:
            # The boundary is not ambiguous: every tied candidate must occupy a
            # remaining slot.  Otherwise none is treated as an identifiable top-k
            # member, which is the conservative lower bound over tie resolutions.
            certain.update(boundary_edges)
        selected[context] = certain
    overlaps = [
        len(selected[left] & selected[right]) / truth_count
        for left, right in itertools.combinations(sorted(selected), 2)
    ]
    return fmean(overlaps) if overlaps else None


def validate_summary_reports(state: dict[str, Any], run_root: Path) -> tuple[Path, Path]:
    """Return summary reports only when state-bound paths and hashes still match."""

    reports = state.get("reports")
    if not isinstance(reports, dict):
        raise BenchmarkConfigError("Completed state has no reports")
    safe_run_root = path_below_no_symlinks(
        run_root,
        run_root,
        "summary run root",
        require_directory=True,
    )
    expected_root = path_below_no_symlinks(
        safe_run_root / "reports",
        safe_run_root,
        "summary reports directory",
        require_directory=True,
    )
    metrics = expected_root / "metrics.csv"
    summary = expected_root / "summary.json"
    recorded_metrics = reports.get("metrics_csv")
    recorded_summary = reports.get("summary_json")
    if recorded_metrics != str(metrics) or recorded_summary != str(summary):
        raise BenchmarkConfigError(
            "Completed state must reference canonical reports/metrics.csv and reports/summary.json"
        )
    metrics = path_below_no_symlinks(
        metrics,
        expected_root,
        "summary metrics.csv",
        require_file=True,
    )
    summary = path_below_no_symlinks(
        summary,
        expected_root,
        "summary summary.json",
        require_file=True,
    )
    for path, field in (
        (metrics, "metrics_csv_sha256"),
        (summary, "summary_json_sha256"),
    ):
        expected = reports.get(field)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise BenchmarkConfigError(f"Completed summary report is unbound or changed: {path}")
    return metrics, summary


def _scheduler_evidence(run_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Bind rolling-queue utilization manifests into the canonical summary."""

    result: dict[str, Any] = {}
    capacity = state.get("runtime_plan", {}).get("capacity")
    if not isinstance(capacity, dict):
        raise BenchmarkConfigError("Run state has no scheduler capacity")
    for phase in ("inference", "evaluation"):
        path = path_below_no_symlinks(
            run_root / "schedules" / f"{phase}.json",
            run_root,
            f"{phase} resource schedule",
            require_file=True,
        )
        manifest = load_json_object(path, f"{phase} resource schedule")
        utilization = manifest.get("utilization")
        if (
            manifest.get("schema_version") != "1.0"
            or manifest.get("status") != "complete"
            or manifest.get("capacity")
            != {
                "cpu_threads": capacity.get("cpu_threads"),
                "memory_bytes": capacity.get("memory_bytes"),
                "max_concurrent_tasks": None,
            }
            or not isinstance(utilization, dict)
        ):
            raise BenchmarkConfigError(f"{phase.title()} resource schedule is invalid")
        for name in ("cpu_capacity_utilization", "memory_capacity_utilization"):
            value = utilization.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise BenchmarkConfigError(f"{phase.title()} resource schedule has invalid {name}")
        result[phase] = {
            "path": path.relative_to(run_root).as_posix(),
            "sha256": sha256_file(path),
            "logical_tasks": manifest.get("logical_tasks"),
            "physical_tasks": manifest.get("physical_tasks"),
            "cache_hits": manifest.get("cache_hits"),
            "wall_seconds": manifest.get("wall_seconds"),
            "utilization": utilization,
        }
    return result


def _calibration_panel_contract(
    *,
    state: dict[str, Any],
    store: ArtifactStore,
    run_root: Path,
) -> tuple[dict[str, Any] | None, dict[tuple[str, str], DatasetGroupContract]]:
    if state["purpose"] not in {"calibration", "selection"}:
        return None, {}

    expected_role = "validation" if state["purpose"] == "selection" else "development"
    datasets = state.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise BenchmarkConfigError("Calibration has no generated datasets")
    dataset_contracts: dict[str, DatasetGroupContract] = {}
    for dataset in datasets:
        if not isinstance(dataset, dict) or dataset.get("dataset_role") != expected_role:
            raise BenchmarkConfigError(f"Calibration datasets must all use role {expected_role!r}")
        dataset_id = dataset.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id or dataset_id in dataset_contracts:
            raise BenchmarkConfigError(f"Invalid or duplicate calibration dataset: {dataset_id!r}")
        artifact = _validate_artifact(store, dataset.get("dataset_artifact"))
        dataset_manifest = _artifact_member(
            artifact,
            dataset.get("dataset_manifest"),
            "calibration dataset manifest",
        )
        ground_truth_manifest = _artifact_member(
            artifact,
            dataset.get("ground_truth_manifest"),
            "calibration ground-truth manifest",
        )
        dataset_contracts[dataset_id] = load_dataset_group_contract(
            dataset_manifest,
            ground_truth_manifest,
            expected_dataset_id=dataset_id,
        )

    # Generation may publish replicas reserved for another calibration fidelity.
    # Derive this panel from the frozen requests, never from completed runs alone.
    profile = frozen_profile(run_root, state["profile_id"])
    planned: dict[tuple[str, str], str] = {}
    run_ids_by_dataset: dict[str, set[str]] = defaultdict(set)
    for batch in profile["inference_batches"]:
        if batch["kind"] != "spathi" or batch["regime"] != "group_only":
            raise BenchmarkConfigError("Calibration plan must contain only SPATHI group_only runs")
        request = load_json_object(
            frozen_path(run_root, batch["tools_params"], "calibration tools params"),
            "calibration tools params",
        )
        for dataset in datasets:
            if not _batch_selects_dataset(batch, dataset):
                continue
            dataset_id = dataset["dataset_id"]
            for run in request["runs"]:
                run_id = run["run_id"]
                key = (dataset_id, run_id)
                if key in planned:
                    raise BenchmarkConfigError(f"Duplicate planned calibration execution: {key!r}")
                planned[key] = batch["id"]
                run_ids_by_dataset[dataset_id].add(run_id)
    run_id_sets = {frozenset(run_ids) for run_ids in run_ids_by_dataset.values()}
    if not planned or len(run_id_sets) != 1:
        raise BenchmarkConfigError(
            "Every selected calibration dataset must request exactly the same frozen runs"
        )
    run_ids = sorted(next(iter(run_id_sets)))
    dataset_contracts = {
        dataset_id: contract
        for dataset_id, contract in dataset_contracts.items()
        if dataset_id in run_ids_by_dataset
    }

    inference_runs = state.get("inference_runs")
    if not isinstance(inference_runs, list) or not inference_runs:
        raise BenchmarkConfigError("Calibration has no inference runs")
    executions: dict[tuple[str, str], dict[str, Any]] = {}
    for inference in inference_runs:
        if not isinstance(inference, dict):
            raise BenchmarkConfigError("Calibration inference metadata must be objects")
        dataset_id = inference.get("dataset_id")
        run_id = inference.get("run_id")
        if dataset_id not in dataset_contracts or not isinstance(run_id, str) or not run_id:
            raise BenchmarkConfigError("Calibration inference references an unknown execution")
        if inference.get("batch_kind") != "spathi" or inference.get("regime") != "group_only":
            raise BenchmarkConfigError(
                "Calibration inference must contain only SPATHI group_only executions"
            )
        key = (dataset_id, run_id)
        if key in executions:
            raise BenchmarkConfigError(
                f"Duplicate calibration inference for {dataset_id!r}/{run_id!r}"
            )
        if key not in planned or inference.get("batch_id") != planned[key]:
            raise BenchmarkConfigError(f"Unplanned calibration inference: {key!r}")
        executions[key] = inference
    if set(executions) != set(planned):
        raise BenchmarkConfigError(
            f"Calibration inference omits planned executions: {sorted(set(planned) - set(executions))}"
        )

    handoff_path = Path(str(state.get("handoff_validation", ""))).resolve()
    if handoff_path != (run_root / "handoff-validation.json").resolve():
        raise BenchmarkConfigError("Calibration state has no canonical handoff validation")
    handoff = load_json_object(handoff_path, "calibration handoff validation")
    if handoff.get("profile_id") != state.get("profile_id") or handoff.get("status") != "passed":
        raise BenchmarkConfigError("Calibration handoff validation did not pass")
    raw_results = handoff.get("results")
    if not isinstance(raw_results, list):
        raise BenchmarkConfigError("Calibration handoff validation has no execution results")
    handoff_results: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            raise BenchmarkConfigError("Calibration handoff results must be objects")
        key = (raw_result.get("dataset_id"), raw_result.get("run_id"))
        if key not in executions or key in handoff_results:
            raise BenchmarkConfigError(
                f"Calibration handoff contains an unknown or duplicate execution: {key!r}"
            )
        handoff_results[key] = raw_result
    if set(handoff_results) != set(executions):
        raise BenchmarkConfigError("Calibration handoff omits planned executions")

    execution_contracts: dict[tuple[str, str], DatasetGroupContract] = {}
    for key, inference in executions.items():
        dataset_id, _run_id = key
        contract = dataset_contracts[dataset_id]
        result = handoff_results[key]
        artifact_reference = inference.get("artifact")
        artifact_key = (
            artifact_reference.get("key") if isinstance(artifact_reference, dict) else None
        )
        expected_handoff = {
            **contract.to_dict(),
            "completed_group_contexts": list(contract.expected_group_contexts),
        }
        if (
            result.get("status") != "passed"
            or not isinstance(artifact_key, str)
            or result.get("artifact_key") != artifact_key
            or result.get("calibration_context_contract") != expected_handoff
        ):
            raise BenchmarkConfigError(
                f"Calibration handoff context evidence is incomplete for {key!r}"
            )
        execution_contracts[key] = contract

    dataset_ids = sorted(dataset_contracts)
    panel = {
        "schema_version": "1.0",
        "dataset_role": expected_role,
        "dataset_ids": dataset_ids,
        "run_ids": run_ids,
        "n_expected_executions": len(dataset_ids) * len(run_ids),
        "handoff_validation_sha256": sha256_file(handoff_path),
        "datasets": [dataset_contracts[dataset_id].to_dict() for dataset_id in dataset_ids],
    }
    return panel, execution_contracts


def _validate_group_metric_inventory(
    metrics: list[dict[str, Any]],
    contract: DatasetGroupContract,
    *,
    run_id: str,
) -> None:
    """Require every reported evaluation level to cover every declared group exactly once."""

    by_level: dict[str, list[str]] = defaultdict(list)
    for metric in metrics:
        context = metric.get("context")
        level = metric.get("level")
        if isinstance(context, str) and context.startswith("group:") and isinstance(level, str):
            by_level[level].append(context)
    if not by_level:
        raise BenchmarkConfigError(f"Evaluation {run_id!r} contains no group-level metrics")
    expected = list(contract.expected_group_contexts)
    for level, contexts in sorted(by_level.items()):
        if sorted(contexts) != expected:
            raise BenchmarkConfigError(
                f"Evaluation {run_id!r}/{level!r} does not cover the complete context contract "
                f"for dataset groups: expected={expected}, observed={sorted(contexts)}"
            )


def _finite_unit_metric(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise BenchmarkConfigError(f"{label} must be a finite number in [0, 1]")
    return float(value)


def _calibration_context_aupr(row: dict[str, Any]) -> float:
    context = row.get("context")
    status = row.get("status")
    aupr = _finite_unit_metric(row.get("aupr"), f"Calibration AUPR for {context!r}")
    if status == "ok":
        _finite_unit_metric(row.get("auroc"), f"Calibration AUROC for {context!r}")
    elif status == "partial":
        if (
            row.get("auroc") is not None
            or row.get("reason")
            != "AUROC/AUPR require at least one positive and one negative candidate"
        ):
            raise BenchmarkConfigError(
                f"Calibration context {context!r} has an unsupported partial status"
            )
    else:
        raise BenchmarkConfigError(
            f"Calibration context {context!r} is not evaluable: status={status!r}"
        )
    if row.get("truth_context") != context:
        raise BenchmarkConfigError(
            f"Calibration context {context!r} is paired with different ground truth"
        )
    return aupr


def _calibration_rows(
    rows: list[dict[str, Any]],
    execution_contracts: dict[tuple[str, str], DatasetGroupContract],
) -> list[dict[str, Any]]:
    if not execution_contracts:
        return []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["batch_kind"] != "spathi" or row["dataset_role"] not in {
            "development",
            "validation",
        }:
            continue
        key = (str(row["dataset_id"]), str(row["run_id"]))
        if key not in execution_contracts:
            raise BenchmarkConfigError(f"Unexpected calibration metric execution: {key!r}")
        grouped[key].append(row)
    if set(grouped) != set(execution_contracts):
        missing = sorted(set(execution_contracts) - set(grouped))
        raise BenchmarkConfigError(f"Calibration metrics omit planned executions: {missing}")

    result: list[dict[str, Any]] = []
    for key in sorted(execution_contracts):
        dataset_id, _run_id = key
        contract = execution_contracts[key]
        all_members = grouped[key]
        members = [
            row
            for row in all_members
            if row.get("level") == "directed" and str(row.get("context", "")).startswith("group:")
        ]
        by_context: dict[str, dict[str, Any]] = {}
        for row in members:
            context = str(row["context"])
            if context in by_context:
                raise BenchmarkConfigError(
                    f"Duplicate calibration metric for {dataset_id!r}/{_run_id!r}/{context!r}"
                )
            by_context[context] = row
        if tuple(sorted(by_context)) != contract.expected_group_contexts:
            raise BenchmarkConfigError(
                f"Calibration metrics do not cover the complete context contract for "
                f"{dataset_id!r}/{_run_id!r}: expected="
                f"{list(contract.expected_group_contexts)}, observed={sorted(by_context)}"
            )
        auprs = [
            _calibration_context_aupr(by_context[context])
            for context in contract.expected_group_contexts
        ]
        wall_times = {row.get("wall_time_seconds") for row in all_members}
        peak_memories = {row.get("peak_memory_mb") for row in all_members}
        if len(wall_times) != 1 or len(peak_memories) != 1:
            raise BenchmarkConfigError("Calibration metric rows contain conflicting runtime data")
        tool_images = [row.get("tool_image") for row in all_members]
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
                "configuration_id": first["configuration_id"],
                "configuration_sha256": first["configuration_sha256"],
                "dataset_role": first["dataset_role"],
                "dataset_id": dataset_id,
                "algorithm_seed": first["algorithm_seed"],
                "aupr_directed": fmean(auprs),
                "wall_time_seconds": next(iter(wall_times)),
                "peak_memory_mb": next(iter(peak_memories)),
                "run_id": first["run_id"],
                "execution_sha256": first["execution_sha256"],
                "n_group_contexts": len(contract.expected_group_contexts),
                "group_contexts": list(contract.expected_group_contexts),
                "spathi_commit": spathi_commit,
                "spathi_build_fingerprint": spathi_build_fingerprint,
                "spathi_image_id": spathi_image_id,
            }
        )
    return sorted(
        result,
        key=lambda row: (
            str(row["configuration_id"]),
            str(row["dataset_id"]),
            int(row["algorithm_seed"]),
        ),
    )


def summarize(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
) -> tuple[Path, Path]:
    validate_run_inputs(repo_root, profile_id, run_root)
    validate_inference_attribution_contract(
        repo_root=repo_root,
        profile_id=profile_id,
        run_root=run_root,
    )
    state = load_state(run_root, profile_id)
    if status_at_least(state, "completed"):
        return validate_summary_reports(state, run_root)
    require_status(state, "exported", "Summarization")
    safe_run_root = path_below_no_symlinks(
        run_root,
        run_root,
        "summary run root",
        require_directory=True,
    )
    reports_dir = safe_run_root / "reports"

    store = ArtifactStore(Path(state["artifact_store"]))
    datasets = {item["dataset_id"]: item for item in state["datasets"]}
    identities: dict[str, EvaluationSpaceIdentity] = {}
    group_contracts: dict[str, DatasetGroupContract] = {}
    rows: list[dict[str, Any]] = []
    for evaluation in unique_execution_records(state["evaluations"]):
        inference_artifact = _validate_artifact(store, evaluation["inference_artifact"])
        evaluation_artifact = _validate_artifact(store, evaluation["artifact"])
        dataset = datasets[evaluation["dataset_id"]]
        identity = identities.get(dataset["dataset_id"])
        if identity is None:
            dataset_artifact = _validate_artifact(store, dataset["dataset_artifact"])
            dataset_manifest = _artifact_member(
                dataset_artifact,
                dataset.get("dataset_manifest"),
                "dataset manifest",
            )
            ground_truth_manifest = _artifact_member(
                dataset_artifact,
                dataset.get("ground_truth_manifest"),
                "ground-truth manifest",
            )
            identity = load_evaluation_space_identity(
                dataset_artifact=dataset_artifact,
                ground_truth_manifest_path=ground_truth_manifest,
                expected_dataset_id=dataset["dataset_id"],
            )
            identities[dataset["dataset_id"]] = identity
            group_contracts[dataset["dataset_id"]] = load_dataset_group_contract(
                dataset_manifest,
                ground_truth_manifest,
                expected_dataset_id=dataset["dataset_id"],
            )
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
        identity.validate_evaluation_report(report, dataset_id=dataset["dataset_id"])
        run_report = load_json_object(run_report_path, "run report")
        tools = run_report.get("tools") or {}
        origins = tools.get("tool_origins", {})
        tool_result = (tools.get("results") or {}).get(evaluation["run_id"], {})
        if not isinstance(tool_result, dict):
            raise BenchmarkConfigError("Inference run report contains invalid tool result metadata")
        measurement = tool_result.get("measurement")
        if (
            not isinstance(measurement, dict)
            or measurement.get("schema_version") != "1.0"
            or measurement.get("scope") not in {"physical_task", "logical_run"}
        ):
            raise BenchmarkConfigError("Inference run report omits canonical resource telemetry")
        wall_time_seconds = measurement.get("wall_time_seconds")
        peak_memory_bytes = measurement.get("peak_memory_bytes")
        if (
            not isinstance(wall_time_seconds, (int, float))
            or isinstance(wall_time_seconds, bool)
            or not math.isfinite(float(wall_time_seconds))
            or float(wall_time_seconds) < 0
        ):
            raise BenchmarkConfigError("Inference telemetry has invalid wall time")
        if peak_memory_bytes is not None and (
            not isinstance(peak_memory_bytes, int)
            or isinstance(peak_memory_bytes, bool)
            or peak_memory_bytes < 0
        ):
            raise BenchmarkConfigError("Inference telemetry has invalid peak memory")
        metrics = report.get("metrics")
        if not isinstance(metrics, list) or not all(isinstance(item, dict) for item in metrics):
            raise BenchmarkConfigError("evaluation_report.json metrics must be an array of objects")
        _validate_group_metric_inventory(
            metrics,
            group_contracts[dataset["dataset_id"]],
            run_id=evaluation["run_id"],
        )
        network_path = _reported_artifact_member(
            inference_artifact,
            run_report_path,
            tool_result.get("network_path"),
            "inference network path",
        )
        invariant_stability = _invariant_edge_stability(
            network_path,
            metrics,
            candidate_sources=identity.sources,
            candidate_targets=identity.targets,
            allow_self_edges=identity.allow_self_edges,
        )
        for metric in metrics:
            metric_run_id = metric.get("tool_id")
            if metric_run_id != evaluation["run_id"]:
                raise BenchmarkConfigError(
                    f"Evaluation metric tool_id {metric_run_id!r} differs from its single-run "
                    f"artifact {evaluation['run_id']!r}"
                )
            identity.validate_metric(metric)
            semantic_provenance = identity.metric_provenance(
                context=str(metric["context"]),
                level=str(metric["level"]),
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
                "wall_time_seconds": float(wall_time_seconds),
                "peak_memory_mb": (
                    None if peak_memory_bytes is None else peak_memory_bytes / 1024**2
                ),
                **semantic_provenance,
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
            rows.append(
                {
                    **metric,
                    "invariant_edge_stability": invariant_stability,
                    **provenance,
                }
            )
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
    execution_count_per_group: dict[tuple[str, ...], int] = defaultdict(int)
    for dataset_key, values in per_dataset.items():
        *group_key_parts, dataset_id = dataset_key
        group_key = tuple(group_key_parts)
        datasets_per_group[group_key].add(dataset_id)
        execution_count_per_group[group_key] += len(executions_per_dataset[dataset_key])
        for metric_name, execution_values in values.items():
            if execution_values:
                grouped[group_key][metric_name].append(fmean(execution_values))

    metadata_by_configuration: dict[str, dict[str, Any]] = {}
    for row in rows:
        configuration_id = str(row["configuration_id"])
        metadata = {
            "family": row.get("family"),
            "selection_role": row["selection_role"],
            "configuration_sha256": row["configuration_sha256"],
        }
        previous = metadata_by_configuration.setdefault(configuration_id, metadata)
        if previous != metadata:
            raise BenchmarkConfigError(
                f"Configuration {configuration_id!r} has conflicting invariant summary metadata"
            )

    aggregates = []
    group_fields = execution_key_fields[:-2]
    for group_key, values in sorted(grouped.items()):
        group_identity = dict(zip(group_fields, group_key, strict=True))
        configuration_id = group_identity["configuration_id"]
        members = [
            row
            for row in rows
            if all(str(row.get(field)) == value for field, value in group_identity.items())
        ]
        raw_catalog_tool_ids = {row["catalog_tool_id"] for row in members}
        if not all(
            isinstance(catalog_tool_id, str) and catalog_tool_id
            for catalog_tool_id in raw_catalog_tool_ids
        ):
            raise BenchmarkConfigError("Aggregate metric rows contain an invalid catalog_tool_id")
        catalog_tool_ids = sorted(raw_catalog_tool_ids)
        family = metadata_by_configuration[configuration_id]["family"]
        if family == "spathi":
            tool_id = "SPATHI"
        elif len(catalog_tool_ids) == 1:
            tool_id = catalog_tool_ids[0]
        else:
            tool_id = configuration_id
        aggregates.append(
            {
                **group_identity,
                **metadata_by_configuration[configuration_id],
                "tool_id": tool_id,
                "catalog_tool_ids": catalog_tool_ids,
                "n_datasets": len(datasets_per_group[group_key]),
                "n_executions": execution_count_per_group[group_key],
                "n_metric_rows": len(members),
                **{
                    f"mean_{name}": fmean(items) if items else None
                    for name, items in values.items()
                },
            }
        )

    notes = [
        "Each metric is averaged across contexts, then algorithmic seeds, then equally across datasets.",
        "Aggregates never mix dataset roles, splits or experimental conditions.",
        "Aggregate tool_id is a stable display identity; catalog_tool_ids lists the underlying ANDREA execution IDs.",
        "Raw metric semantics and statuses come directly from ANDREA.",
    ]
    if state["purpose"] == "smoke":
        notes.insert(0, "The smoke profile validates contracts and has no statistical power.")
    calibration_panel, calibration_execution_contracts = _calibration_panel_contract(
        state=state,
        store=store,
        run_root=run_root,
    )
    calibration_rows = _calibration_rows(rows, calibration_execution_contracts)

    temporary = Path(tempfile.mkdtemp(prefix=".reports-", dir=safe_run_root))
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
                    "schema_version": SUMMARY_SCHEMA_VERSION,
                    "profile_id": profile_id,
                    "purpose": state["purpose"],
                    "benchmark_commit": state["benchmark_repository"]["commit"],
                    "metric_rows": len(rows),
                    "aggregates": aggregates,
                    "calibration_panel": calibration_panel,
                    "calibration_rows": calibration_rows,
                    "cache": state["cache"],
                    "scheduling": _scheduler_evidence(safe_run_root, state),
                    "notes": notes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if reports_dir.exists() or reports_dir.is_symlink():
            reports_dir = path_below_no_symlinks(
                reports_dir,
                safe_run_root,
                "summary reports directory",
                require_directory=True,
            )
            published_metrics = reports_dir / "metrics.csv"
            published_summary = reports_dir / "summary.json"
            observed_members = {member.name for member in reports_dir.iterdir()}
            if observed_members != {"metrics.csv", "summary.json"}:
                raise BenchmarkConfigError(
                    f"Incomplete reports directory already exists or contents are stale: "
                    f"{reports_dir}"
                )
            published_metrics = path_below_no_symlinks(
                published_metrics,
                reports_dir,
                "published summary metrics.csv",
                require_file=True,
            )
            published_summary = path_below_no_symlinks(
                published_summary,
                reports_dir,
                "published summary summary.json",
                require_file=True,
            )
            if sha256_file(published_metrics) != sha256_file(metrics_path) or sha256_file(
                published_summary
            ) != sha256_file(summary_path):
                raise BenchmarkConfigError(
                    f"Incomplete reports directory already exists or contents are stale: "
                    f"{reports_dir}"
                )
        else:
            os.rename(temporary, reports_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    reports_dir = path_below_no_symlinks(
        reports_dir,
        safe_run_root,
        "summary reports directory",
        require_directory=True,
    )
    final_metrics = path_below_no_symlinks(
        reports_dir / "metrics.csv",
        reports_dir,
        "summary metrics.csv",
        require_file=True,
    )
    final_summary = path_below_no_symlinks(
        reports_dir / "summary.json",
        reports_dir,
        "summary summary.json",
        require_file=True,
    )
    state["reports"] = {
        "metrics_csv": str(final_metrics),
        "metrics_csv_sha256": sha256_file(final_metrics),
        "summary_json": str(final_summary),
        "summary_json_sha256": sha256_file(final_summary),
    }
    state["status"] = "completed"
    state["updated_at"] = _utc_now()
    write_state(run_root, state)
    return final_metrics, final_summary
