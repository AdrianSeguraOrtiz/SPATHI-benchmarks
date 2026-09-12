"""Build efficiency non-inferiority evidence from immutable execution artifacts.

This module is deliberately a verifier, not an experiment runner.  It accepts a
predeclared panel and a small index of content-addressed inference artifacts,
validates every file through :class:`ArtifactStore`, derives target spaces from
SPATHI diagnostics, and recomputes directed AUPR from the network and truth
tables.  Consequently, the statistical analyzer never needs to trust metric or
coverage values typed into an evidence JSON document.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .artifacts import (
    Artifact,
    ArtifactError,
    ArtifactKey,
    ArtifactStore,
    canonical_json_sha256,
    sha256_file,
)
from .calibration import resolve_group_policy
from .config import BenchmarkConfigError, load_json_object, validate_json
from .efficiency_order import counterbalanced_execution_orders, runner_implementation_sha256
from .safe_io import path_below_no_symlinks, write_idempotent_json
from .spathi_diagnostics import read_model_accounting
from .study_context import permit_revision_change
from .tool_contract import oom_observed, spathi_runtime_resources

_NETWORK_COLUMNS = ("source", "target", "score", "sign", "evidence", "context")
_ELIGIBILITY_COLUMNS = (
    "target",
    "mode",
    "eligible",
    "detected_cells",
    "detected_fraction",
    "expression_min",
    "expression_max",
    "required_detected_cells",
    "reason",
)
_INPUT_NAMES = ("expression", "groups", "tf_list")


@dataclass(frozen=True, slots=True)
class _Dataset:
    artifact: Artifact
    dataset_id: str
    simulator_id: str
    condition_id: str
    generation_seed: int
    contexts: tuple[str, ...]
    manifest_path: Path
    expression_path: Path
    groups_path: Path
    tf_path: Path
    genes_path: Path
    truth_path: Path
    fingerprint: str
    sources: tuple[str, ...]
    targets: tuple[str, ...]
    truth: dict[str, frozenset[tuple[str, str]]]
    truth_outside_candidate_space: dict[str, frozenset[tuple[str, str]]]


@dataclass(frozen=True, slots=True)
class _Inference:
    artifact: Artifact
    network_path: Path
    eligibility_path: Path
    diagnostics_path: Path
    metadata_path: Path
    report_path: Path
    measurements: dict[str, Any]
    network: dict[str, dict[tuple[str, str], float]]
    eligible_targets: dict[str, tuple[str, ...]]


def _validated_json(path: Path, schema: Path, label: str) -> dict[str, Any]:
    payload = load_json_object(path, label)
    validate_json(payload, schema, label)
    return payload


def _inside(repo_root: Path, path: Path, label: str) -> str:
    root = repo_root.resolve()
    safe = path_below_no_symlinks(path, root, label)
    return safe.relative_to(root).as_posix()


def _single_member(artifact: Artifact, suffix: str, label: str) -> Path:
    matches = [
        entry["path"]
        for entry in artifact.manifest["files"]
        if entry["path"] == suffix or entry["path"].endswith("/" + suffix)
    ]
    if len(matches) != 1:
        raise BenchmarkConfigError(
            f"{artifact.key}: expected exactly one {label} ({suffix}), observed {matches}"
        )
    return artifact.file(matches[0])


def _read_lines(path: Path, label: str) -> tuple[str, ...]:
    try:
        values = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkConfigError(f"Cannot read {label}: {path}: {exc}") from exc
    if not values or any(not value for value in values) or len(values) != len(set(values)):
        raise BenchmarkConfigError(f"{label} must be a non-empty unique list")
    return values


def _read_dict_rows(path: Path, *, gzipped: bool = False) -> tuple[list[str], list[dict[str, str]]]:
    try:
        handle: TextIO
        if gzipped:
            handle = gzip.open(path, "rt", encoding="utf-8", newline="")
        else:
            handle = path.open("r", encoding="utf-8", newline="")
        with handle:
            reader = csv.DictReader(handle, delimiter="\t" if gzipped else ",")
            header = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise BenchmarkConfigError(f"Cannot read tabular artifact {path}: {exc}") from exc
    return header, rows


def _parse_bool(raw: str, label: str) -> bool:
    normalized = raw.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise BenchmarkConfigError(f"{label} must be true or false, got {raw!r}")


def _finite_float(raw: str, label: str, *, minimum: float | None = None) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} must be numeric, got {raw!r}") from exc
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise BenchmarkConfigError(f"{label} is outside its valid range: {raw!r}")
    return value


def _integer(raw: str, label: str, *, minimum: int = 0) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise BenchmarkConfigError(f"{label} is below {minimum}: {value}")
    return value


def _artifact(store: ArtifactStore, raw: Mapping[str, Any], expected_kind: str) -> Artifact:
    if raw.get("kind") != expected_kind:
        raise BenchmarkConfigError(f"Expected a {expected_kind!r} artifact reference")
    try:
        return store.validate(ArtifactKey(kind=expected_kind, digest=str(raw.get("digest"))))
    except ArtifactError as exc:
        raise BenchmarkConfigError(f"Invalid {expected_kind} artifact: {exc}") from exc


def _dataset_from_artifact(
    artifact: Artifact,
    declaration: Mapping[str, Any],
    *,
    dataset_role: str,
    split_id: str,
    generation_contract: Mapping[str, Any] | None = None,
) -> _Dataset:
    spec = artifact.manifest["spec"]
    dataset_id = declaration["dataset_id"]
    expected = {
        "condition_id": declaration["condition_id"],
        "dataset_role": dataset_role,
        "split_id": split_id,
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise BenchmarkConfigError(
                f"Dataset artifact {artifact.key} has {key}={spec.get(key)!r}, expected {value!r}"
            )
    if generation_contract is not None:
        expected_generation = {
            "generation_job": generation_contract["generation_job"],
            "scenario_sha256": generation_contract["scenario"]["sha256"],
            "simulator_runs_sha256": generation_contract["simulator_runs"]["sha256"],
        }
        if any(spec.get(key) != value for key, value in expected_generation.items()):
            raise BenchmarkConfigError(
                "Safety dataset artifact differs from its frozen generation request"
            )
    manifest_path = _single_member(
        artifact, f"datasets/{dataset_id}/dataset-manifest.json", "dataset manifest"
    )
    ground_truth_path = _single_member(
        artifact, f"datasets/{dataset_id}/ground-truth-manifest.json", "ground-truth manifest"
    )
    expression_path = _single_member(
        artifact, f"datasets/{dataset_id}/expression.tsv", "expression matrix"
    )
    groups_path = _single_member(
        artifact, f"datasets/{dataset_id}/extras/groups.tsv", "group assignments"
    )
    tf_path = _single_member(artifact, f"datasets/{dataset_id}/extras/tf_list.txt", "TF list")
    genes_path = _single_member(
        artifact, f"datasets/{dataset_id}/truth/gene_universe.txt", "gene universe"
    )
    truth_path = _single_member(
        artifact, f"datasets/{dataset_id}/truth/networks.csv", "truth network"
    )
    simulator_path = _single_member(
        artifact, f"datasets/{dataset_id}/provenance/simulator-run.json", "simulator run"
    )
    dataset_manifest = load_json_object(manifest_path, "dataset manifest")
    ground_truth = load_json_object(ground_truth_path, "ground-truth manifest")
    simulator = load_json_object(simulator_path, "simulator run")
    if dataset_manifest.get("dataset", {}).get("spec", {}).get("id") != dataset_id:
        raise BenchmarkConfigError(f"Dataset manifest ID does not match {dataset_id!r}")
    if ground_truth.get("dataset_id") != dataset_id or simulator.get("dataset_id") != dataset_id:
        raise BenchmarkConfigError(f"Dataset provenance does not match {dataset_id!r}")
    if ground_truth.get("simulator_id") != declaration["simulator_id"]:
        raise BenchmarkConfigError(f"Simulator identity differs for {dataset_id!r}")
    if simulator.get("simulator_id") != declaration["simulator_id"]:
        raise BenchmarkConfigError(f"Simulator run identity differs for {dataset_id!r}")
    if simulator.get("seed") != declaration["generation_seed"]:
        raise BenchmarkConfigError(f"Generation seed differs for {dataset_id!r}")

    groups_header, group_rows = _read_dict_rows_tsv(groups_path)
    if groups_header != ["column", "cluster"] or not group_rows:
        raise BenchmarkConfigError(f"Groups file has an unsupported contract: {groups_path}")
    observed_contexts = tuple(sorted({f"group:{row['cluster']}" for row in group_rows}))
    declared_contexts = tuple(sorted(declaration["contexts"]))
    if observed_contexts != declared_contexts:
        raise BenchmarkConfigError(
            f"Predeclared contexts differ from groups.tsv for {dataset_id!r}: "
            f"declared={declared_contexts}, observed={observed_contexts}"
        )
    if any(not row["column"] or not row["cluster"] for row in group_rows):
        raise BenchmarkConfigError(f"Groups file contains blank identifiers: {groups_path}")
    if len({row["column"] for row in group_rows}) != len(group_rows):
        raise BenchmarkConfigError(f"Groups file contains duplicate cells: {groups_path}")

    sources = _read_lines(tf_path, "TF list")
    targets = _read_lines(genes_path, "gene universe")
    if generation_contract is not None:
        group_sizes = sorted(Counter(row["cluster"] for row in group_rows).values())
        if (
            len(sources) != generation_contract["transcription_factors"]
            or len(targets) != generation_contract["genes"]
            or group_sizes != sorted(generation_contract["group_sizes"])
        ):
            raise BenchmarkConfigError(
                "Safety dataset dimensions differ from its frozen condition scale"
            )
    if not set(sources) <= set(targets):
        raise BenchmarkConfigError(f"TF list is not a subset of the gene universe: {dataset_id}")
    header, rows = _read_dict_rows(truth_path)
    if tuple(header) != _NETWORK_COLUMNS:
        raise BenchmarkConfigError(f"Truth network has an unexpected header: {truth_path}")
    truth: dict[str, set[tuple[str, str]]] = defaultdict(set)
    truth_outside_candidate_space: dict[str, set[tuple[str, str]]] = defaultdict(set)
    seen_truth: dict[str, set[tuple[str, str]]] = defaultdict(set)
    source_set = set(sources)
    gene_universe = set(targets)
    for row in rows:
        context = row["context"]
        edge = (row["source"], row["target"])
        if context not in declared_contexts and context != "global":
            raise BenchmarkConfigError(f"Truth contains undeclared context {context!r}")
        if edge[0] not in gene_universe or edge[1] not in gene_universe:
            raise BenchmarkConfigError(
                f"Truth edge references a gene outside the declared gene universe: {edge}"
            )
        if edge in seen_truth[context]:
            raise BenchmarkConfigError(f"Truth contains duplicate edge {edge} in {context!r}")
        seen_truth[context].add(edge)
        if edge[0] in source_set and edge[0] != edge[1]:
            truth[context].add(edge)
        else:
            truth_outside_candidate_space[context].add(edge)
    for context in declared_contexts:
        if not truth.get(context):
            raise BenchmarkConfigError(
                f"Directed AUPR is unverifiable because {dataset_id}/{context} has no truth edges"
            )
    fingerprint = ground_truth.get("dataset_fingerprint", {})
    if fingerprint.get("algorithm") != "sha256" or not isinstance(fingerprint.get("value"), str):
        raise BenchmarkConfigError(f"Ground-truth fingerprint is invalid for {dataset_id!r}")
    return _Dataset(
        artifact=artifact,
        dataset_id=dataset_id,
        simulator_id=declaration["simulator_id"],
        condition_id=declaration["condition_id"],
        generation_seed=declaration["generation_seed"],
        contexts=declared_contexts,
        manifest_path=manifest_path,
        expression_path=expression_path,
        groups_path=groups_path,
        tf_path=tf_path,
        genes_path=genes_path,
        truth_path=truth_path,
        fingerprint=fingerprint["value"],
        sources=sources,
        targets=targets,
        truth={context: frozenset(edges) for context, edges in truth.items()},
        truth_outside_candidate_space={
            context: frozenset(edges) for context, edges in truth_outside_candidate_space.items()
        },
    )


def _read_dict_rows_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return list(reader.fieldnames or []), list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise BenchmarkConfigError(f"Cannot read TSV artifact {path}: {exc}") from exc


def _expression_statistics(dataset: _Dataset) -> tuple[int, dict[str, tuple[int, float, float]]]:
    """Independently recalculate global detection and range for every target."""

    try:
        with dataset.expression_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
            n_cells = len(header) - 1
            if n_cells < 1:
                raise BenchmarkConfigError("Expression matrix has no sample columns")
            statistics: dict[str, tuple[int, float, float]] = {}
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise BenchmarkConfigError(
                        f"Expression row {row_number} has {len(row)} columns, expected {len(header)}"
                    )
                gene = row[0]
                if not gene or gene in statistics:
                    raise BenchmarkConfigError(
                        f"Expression matrix has a blank/duplicate gene: {gene!r}"
                    )
                values = [_finite_float(raw, f"expression[{gene}]", minimum=0.0) for raw in row[1:]]
                statistics[gene] = (
                    sum(value > 0.0 for value in values),
                    min(values),
                    max(values),
                )
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise BenchmarkConfigError(
            f"Cannot read expression matrix: {dataset.expression_path}: {exc}"
        ) from exc
    if set(statistics) != set(dataset.targets):
        raise BenchmarkConfigError("Expression genes and truth gene universe are not identical")
    return n_cells, statistics


def _required_detected_count(n_cells: int, fraction: float) -> int:
    required = int(n_cells * fraction)
    while required / n_cells < fraction:
        required += 1
    return required


def _validate_eligibility(
    path: Path,
    *,
    dataset: _Dataset,
    parameters: Mapping[str, Any],
    expression_statistics: tuple[int, dict[str, tuple[int, float, float]]],
) -> tuple[str, ...]:
    header, rows = _read_dict_rows(path, gzipped=True)
    if tuple(header) != _ELIGIBILITY_COLUMNS:
        raise BenchmarkConfigError(f"Target eligibility has an unexpected header: {path}")
    if len(rows) != len(dataset.targets) or {row["target"] for row in rows} != set(dataset.targets):
        raise BenchmarkConfigError("Target eligibility does not cover the complete target universe")
    if len({row["target"] for row in rows}) != len(rows):
        raise BenchmarkConfigError("Target eligibility contains duplicate targets")
    mode = parameters["target_eligibility"]
    n_cells, statistics = expression_statistics
    required = max(
        int(parameters["min_target_detected_cells"]),
        _required_detected_count(n_cells, float(parameters["min_target_detected_fraction"])),
    )
    eligible: list[str] = []
    for row in rows:
        target = row["target"]
        if row["mode"] != mode:
            raise BenchmarkConfigError(f"Target eligibility mode differs for {target!r}")
        observed = _parse_bool(row["eligible"], f"eligibility[{target}]")
        if mode == "all":
            expected = True
            if row["reason"] != "unfiltered_reference":
                raise BenchmarkConfigError(f"Unfiltered eligibility reason differs for {target!r}")
        elif mode == "automatic":
            count, minimum, maximum = statistics[target]
            expected = count >= required and minimum != maximum
            expected_reason = (
                "insufficient_detected_cells"
                if count < required
                else "zero_variance"
                if minimum == maximum
                else "eligible"
            )
            observed_count = _integer(row["detected_cells"], f"detected_cells[{target}]")
            observed_fraction = _finite_float(
                row["detected_fraction"], f"detected_fraction[{target}]", minimum=0.0
            )
            if (
                observed_count != count
                or not math.isclose(
                    observed_fraction, count / n_cells, rel_tol=1e-12, abs_tol=1e-12
                )
                or _finite_float(row["expression_min"], f"expression_min[{target}]") != minimum
                or _finite_float(row["expression_max"], f"expression_max[{target}]") != maximum
                or _integer(row["required_detected_cells"], f"required[{target}]") != required
                or row["reason"] != expected_reason
            ):
                raise BenchmarkConfigError(
                    f"Target eligibility diagnostics do not reproduce expression data for {target!r}"
                )
        else:
            raise BenchmarkConfigError(f"Unsupported target eligibility mode: {mode!r}")
        if observed != expected:
            raise BenchmarkConfigError(f"Target eligibility decision is incorrect for {target!r}")
        if observed:
            eligible.append(target)
    return tuple(sorted(eligible))


def _input_members(artifact: Artifact) -> dict[str, Path]:
    return {
        "expression": _single_member(artifact, "input/expression.tsv", "inference expression"),
        "groups": _single_member(artifact, "input/extra/groups.tsv", "inference groups"),
        "tf_list": _single_member(artifact, "input/extra/tf_list.txt", "inference TF list"),
    }


def _validate_run_metadata(
    metadata_path: Path,
    *,
    input_members: Mapping[str, Path],
    parameters: Mapping[str, Any],
    algorithm_seed: int,
    dataset: _Dataset,
) -> dict[str, int | bool]:
    metadata = load_json_object(metadata_path, "SPATHI run metadata")
    if metadata.get("status") != "complete" or metadata.get("random_seed") != algorithm_seed:
        raise BenchmarkConfigError(
            "SPATHI run metadata does not describe the expected complete run"
        )
    requested = metadata.get("requested_parameters")
    if not isinstance(requested, dict):
        raise BenchmarkConfigError("SPATHI run metadata omits requested_parameters")
    expected_parameters = {**parameters, "random_seed": algorithm_seed}
    mismatches = {
        key: (requested.get(key), value)
        for key, value in expected_parameters.items()
        if requested.get(key) != value
    }
    if mismatches:
        raise BenchmarkConfigError(
            f"SPATHI requested parameters differ from the plan: {mismatches}"
        )
    if requested.get("target_list") not in {None, ""}:
        raise BenchmarkConfigError("Target eligibility study must not use an explicit target list")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, dict):
        raise BenchmarkConfigError("SPATHI run metadata omits input fingerprints")
    if set(inputs) != set(_INPUT_NAMES):
        raise BenchmarkConfigError(
            "Efficiency runs must use exactly expression, groups and tf_list inputs"
        )
    for name in _INPUT_NAMES:
        entry = inputs.get(name)
        if not isinstance(entry, dict) or entry.get("sha256") != sha256_file(input_members[name]):
            raise BenchmarkConfigError(f"SPATHI input fingerprint differs for {name}")
    dimensions = metadata.get("input_dimensions")
    if not isinstance(dimensions, dict) or dimensions.get("targets") != len(dataset.targets):
        raise BenchmarkConfigError(
            "SPATHI did not consider the complete expression target universe"
        )
    models = metadata.get("models")
    effective = metadata.get("effective_parameters")
    target_eligibility = (
        effective.get("target_eligibility") if isinstance(effective, dict) else None
    )
    if not isinstance(models, dict) or not isinstance(target_eligibility, dict):
        raise BenchmarkConfigError("SPATHI metadata omits efficiency accounting fields")
    counts = {
        "run_completed": metadata.get("status") == "complete",
        "targets_requested": dimensions["targets"],
        "targets_processed": target_eligibility.get("globally_eligible_targets"),
        "models_requested": models.get("requested"),
        "models_completed": models.get("completed"),
        "models_trained": models.get("trained"),
        "trees_fitted": models.get("fitted_estimators_total"),
    }
    for name in (
        "targets_requested",
        "targets_processed",
        "models_requested",
        "models_completed",
        "models_trained",
        "trees_fitted",
    ):
        value = counts[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BenchmarkConfigError(f"SPATHI metadata has invalid {name}: {value!r}")
    if (
        counts["targets_requested"] < 1
        or counts["targets_processed"] > counts["targets_requested"]
        or counts["models_requested"] < 1
        or counts["models_completed"] > counts["models_requested"]
        or counts["models_trained"] > counts["models_completed"]
    ):
        raise BenchmarkConfigError("SPATHI metadata contains inconsistent efficiency counts")
    return counts


def _resource_measurements(
    report: Mapping[str, Any],
    *,
    metadata_counts: Mapping[str, int | bool],
    execution_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify canonical ANDREA telemetry and combine it with SPATHI counters."""

    protocol = execution_contract.get("measurement_protocol")
    execution = report.get("execution")
    measurement = execution.get("measurement") if isinstance(execution, Mapping) else None
    resources = execution_contract.get("resources")
    expected_threads = resources.get("cpu_threads") if isinstance(resources, Mapping) else None
    expected_memory_bytes = (
        resources.get("memory_bytes") if isinstance(resources, Mapping) else None
    )
    if not isinstance(protocol, Mapping) or not isinstance(measurement, Mapping):
        raise BenchmarkConfigError("Efficiency inference omits canonical ANDREA telemetry")
    wall_time = measurement.get("wall_time_seconds")
    peak_bytes = measurement.get("peak_memory_bytes")
    output_bytes = measurement.get("output_bytes")
    assigned = measurement.get("assigned_resources")
    telemetry = measurement.get("telemetry")
    if (
        protocol.get("source") != "andrea_execution_measurement"
        or protocol.get("scope") != "andrea_execution"
        or protocol.get("wall_time")
        != "andrea_execution_end_to_end_including_input_verification_preparation_"
        "scheduling_merging_and_requested_output_materialization"
        or protocol.get("peak_memory")
        != "physical_container_peak_overlap_upper_bound_excluding_andrea_host"
        or protocol.get("output_bytes") != "andrea_execution_output_tree_at_measurement"
        or protocol.get("telemetry_status") != "complete_required"
        or measurement.get("schema_version") != "1.0"
        or measurement.get("scope") != "andrea_execution"
        or isinstance(wall_time, bool)
        or not isinstance(wall_time, (int, float))
        or not math.isfinite(float(wall_time))
        or float(wall_time) <= 0
        or isinstance(peak_bytes, bool)
        or not isinstance(peak_bytes, int)
        or peak_bytes < 1
        or isinstance(output_bytes, bool)
        or not isinstance(output_bytes, int)
        or output_bytes < 1
        or not isinstance(assigned, Mapping)
        or assigned.get("threads_peak") != expected_threads
        or assigned.get("ram_limit_peak_bytes") != expected_memory_bytes
        or not isinstance(telemetry, Mapping)
        or telemetry.get("status") != "complete"
        or telemetry.get("wall_source") != "andrea_monotonic_clock"
        or telemetry.get("wall_semantics")
        != "andrea_execution_end_to_end_including_input_verification_preparation_"
        "scheduling_merging_and_requested_output_materialization"
        or telemetry.get("memory_source") != "child_container_cgroup_peaks"
        or telemetry.get("memory_semantics")
        != "overlap_upper_bound_from_physical_container_peaks_excludes_andrea_host_process"
    ):
        raise BenchmarkConfigError("Efficiency canonical ANDREA telemetry is invalid")
    return {
        **dict(metadata_counts),
        "wall_time_seconds": float(wall_time),
        "peak_memory_bytes": peak_bytes,
        "output_bytes": output_bytes,
    }


def _network(path: Path, dataset: _Dataset) -> dict[str, dict[tuple[str, str], float]]:
    header, rows = _read_dict_rows(path)
    if tuple(header) != _NETWORK_COLUMNS:
        raise BenchmarkConfigError(f"Inferred network has an unexpected header: {path}")
    source_set = set(dataset.sources)
    target_set = set(dataset.targets)
    contexts = set(dataset.contexts)
    result: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        context = row["context"]
        edge = (row["source"], row["target"])
        if context not in contexts:
            raise BenchmarkConfigError(f"Inferred network contains undeclared context {context!r}")
        if edge[0] not in source_set or edge[1] not in target_set or edge[0] == edge[1]:
            raise BenchmarkConfigError(
                f"Inferred edge is outside the complete candidate space: {edge}"
            )
        if edge in result[context]:
            raise BenchmarkConfigError(
                f"Inferred network contains duplicate edge {edge} in {context}"
            )
        result[context][edge] = _finite_float(row["score"], f"score[{context}/{edge}]", minimum=0.0)
    return {context: result[context] for context in dataset.contexts}


def _validate_inference_binding(
    artifact: Artifact,
    *,
    dataset: _Dataset,
    parameters: Mapping[str, Any],
    configuration_id: str,
    configuration_sha256: str,
    policy_sha256: str,
    group_count: int,
    spathi_commit: str,
    algorithm_seed: int,
    seed_derivation: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    execution_order_block: Mapping[str, Any],
    execution_position: int,
    batch_kind: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    spec = artifact.manifest["spec"]
    if (
        spec.get("dataset_artifact") != dataset.artifact.key.value
        or spec.get("dataset_id") != dataset.dataset_id
    ):
        raise BenchmarkConfigError(f"Inference artifact {artifact.key} is bound to another dataset")
    if (
        spec.get("batch_kind") != batch_kind
        or spec.get("andrea_commit") != execution_contract["andrea_commit"]
        or spec.get("andrea_environment") != execution_contract["andrea_environment"]
        or spec.get("benchmark_environment") != execution_contract["benchmark_environment"]
        or spec.get("docker_runtime") != execution_contract["docker_runtime"]
        or spec.get("resources") != execution_contract["resources"]
        or spec.get("measurement_protocol") != execution_contract["measurement_protocol"]
        or spec.get("runner_sha256") != execution_contract["runner_sha256"]
        or spec.get("execution_order_block") != execution_order_block
        or spec.get("execution_position") != execution_position
    ):
        raise BenchmarkConfigError(
            "Inference artifact execution provenance differs from its artifact index"
        )
    run = spec.get("run")
    if not isinstance(run, dict) or run.get("execution") != {"mode": "group_native"}:
        raise BenchmarkConfigError("Non-inferiority inference must use group_native execution")
    resources = execution_contract.get("resources")
    expected_threads = resources.get("cpu_threads") if isinstance(resources, Mapping) else None
    expected_memory_gib = resources.get("memory_gib") if isinstance(resources, Mapping) else None
    if run.get("resources") != {
        "threads": expected_threads,
        "ram_gb": expected_memory_gib,
    }:
        raise BenchmarkConfigError(
            "Non-inferiority inference resources differ from the execution plan"
        )
    expected_params = {**parameters, "random_seed": algorithm_seed}
    if run.get("params") != expected_params:
        raise BenchmarkConfigError("Inference artifact parameters differ from the frozen plan")
    metadata_spec = spec.get("run_metadata")
    parameter_digest = canonical_json_sha256(parameters)
    if (
        not isinstance(metadata_spec, dict)
        or metadata_spec.get("algorithm_seed") != algorithm_seed
        or metadata_spec.get("algorithm_seed_parameter") != "random_seed"
        or metadata_spec.get("seed_derivation") != seed_derivation
        or metadata_spec.get("configuration_id") != configuration_id
        or metadata_spec.get("configuration_sha256") != configuration_sha256
        or metadata_spec.get("policy_sha256") != policy_sha256
        or metadata_spec.get("group_count") != group_count
        or metadata_spec.get("family") != "spathi"
        or parameter_digest != configuration_sha256
    ):
        raise BenchmarkConfigError(
            "Inference artifact run metadata is not bound to the configuration"
        )
    image = spec.get("tool_image")
    if (
        not isinstance(image, dict)
        or image.get("kind") != "spathi"
        or image.get("source_commit") != spathi_commit
        or image.get("image_id") != execution_contract["spathi_image_id"]
        or image.get("build_fingerprint") != execution_contract["spathi_build_fingerprint"]
    ):
        raise BenchmarkConfigError(
            "Inference artifact does not use the frozen SPATHI source commit"
        )
    custom = spec.get("custom_definition")
    run_id = run.get("run_id")
    # The role itself is an audit label; configuration binding above is the
    # authoritative distinction.  It must nevertheless use a known value.
    if metadata_spec.get("selection_role") not in {"reference", "candidate"}:
        raise BenchmarkConfigError("Inference artifact has an invalid efficiency role")
    if (
        not isinstance(run_id, str)
        or run.get("tool_id") != f"custom_{run_id}"
        or not isinstance(custom, dict)
        or custom.get("run_id") != run_id
        or custom.get("docker_image") != execution_contract["spathi_image_id"]
        or custom.get("execution_mode") != "group_native"
        or custom.get("extra_inputs") != ["groups", "tf_list"]
        or custom.get("outputs") != {"directed": True, "sign": "none"}
        or custom.get("runtime_resources") != spathi_runtime_resources()
    ):
        raise BenchmarkConfigError("Inference artifact lacks the group-native SPATHI definition")
    forbidden = [
        entry["path"]
        for entry in artifact.manifest["files"]
        if any("truth" in part.lower() for part in Path(entry["path"]).parts)
    ]
    if forbidden:
        raise BenchmarkConfigError(
            f"Inference artifact contains truth-bearing inputs and cannot prove blindness: {forbidden}"
        )
    return spec, metadata_spec, run_id


def _execution_failure_from_artifact(
    artifact: Artifact,
    *,
    dataset: _Dataset,
    parameters: Mapping[str, Any],
    configuration_id: str,
    configuration_sha256: str,
    policy_sha256: str,
    group_count: int,
    spathi_commit: str,
    algorithm_seed: int,
    seed_derivation: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    execution_order_block: Mapping[str, Any],
    execution_position: int,
    batch_kind: str,
) -> dict[str, Any] | None:
    marker_members = [
        entry["path"]
        for entry in artifact.manifest["files"]
        if Path(entry["path"]).name == "efficiency-execution-failure.json"
    ]
    if not marker_members:
        return None
    if len(marker_members) != 1:
        raise BenchmarkConfigError("Inference artifact has duplicate execution-failure markers")
    _spec, metadata_spec, run_id = _validate_inference_binding(
        artifact,
        dataset=dataset,
        parameters=parameters,
        configuration_id=configuration_id,
        configuration_sha256=configuration_sha256,
        policy_sha256=policy_sha256,
        group_count=group_count,
        spathi_commit=spathi_commit,
        algorithm_seed=algorithm_seed,
        seed_derivation=seed_derivation,
        execution_contract=execution_contract,
        execution_order_block=execution_order_block,
        execution_position=execution_position,
        batch_kind=batch_kind,
    )
    marker_path = artifact.file(marker_members[0])
    if batch_kind not in {"spathi_efficiency", "spathi_efficiency_safety"}:
        raise BenchmarkConfigError("Execution-failure evidence has an invalid batch kind")
    if metadata_spec.get("selection_role") != "candidate":
        raise BenchmarkConfigError(
            "An exact-reference failure cannot be converted into candidate evidence"
        )
    marker = load_json_object(marker_path, "efficiency execution-failure marker")
    expected_keys = {
        "schema_version",
        "status",
        "dataset_id",
        "run_id",
        "tool_status",
        "failure_message",
        "run_report_sha256",
        "measurement_sha256",
        "wall_time_seconds",
        "peak_memory_bytes",
        "oom_killed",
    }
    if set(marker) != expected_keys or marker.get("schema_version") != "1.0":
        raise BenchmarkConfigError("Efficiency execution-failure marker has an invalid contract")
    report_path = _single_member(artifact, "run_report.json", "ANDREA failure report")
    report = load_json_object(report_path, "ANDREA failure report")
    tools = report.get("tools")
    failed = tools.get("failed") if isinstance(tools, Mapping) else None
    failure_message = failed.get(run_id) if isinstance(failed, Mapping) else None
    tool_status = (
        tools.get("status_by_tool", {}).get(run_id) if isinstance(tools, Mapping) else None
    )
    execution = report.get("execution")
    measurement = execution.get("measurement") if isinstance(execution, Mapping) else None
    wall_time = measurement.get("wall_time_seconds") if isinstance(measurement, Mapping) else None
    peak_memory = measurement.get("peak_memory_bytes") if isinstance(measurement, Mapping) else None
    result = tools.get("results", {}).get(run_id) if isinstance(tools, Mapping) else None

    if (
        marker.get("status") != "tool_execution_failed"
        or marker.get("dataset_id") != dataset.dataset_id
        or marker.get("run_id") != run_id
        or marker.get("tool_status") != tool_status
        or marker.get("failure_message") != failure_message
        or marker.get("run_report_sha256") != sha256_file(report_path)
        or not isinstance(measurement, Mapping)
        or marker.get("measurement_sha256") != canonical_json_sha256(measurement)
        or marker.get("wall_time_seconds") != wall_time
        or marker.get("peak_memory_bytes") != peak_memory
        or marker.get("oom_killed") != (isinstance(result, Mapping) and oom_observed(result))
        or report.get("status") != "executed"
        or report.get("output_profile") != execution_contract["resources"]["output_profile"]
        or report.get("run_id") != run_id
        or report.get("dataset", {}).get("id") != dataset.dataset_id
        or report.get("dataset", {}).get("fingerprint", {}).get("value") != dataset.fingerprint
        or not isinstance(tools, Mapping)
        or tools.get("selected") != [run_id]
        or run_id in (tools.get("completed") or [])
        or tools.get("tool_origins", {}).get(run_id) != "custom"
        or not isinstance(tool_status, str)
        or tool_status in {"completed", "completed_with_warnings"}
        or not isinstance(failure_message, str)
        or not failure_message.strip()
        or not isinstance(wall_time, (int, float))
        or isinstance(wall_time, bool)
        or not math.isfinite(float(wall_time))
        or wall_time < 0
    ):
        raise BenchmarkConfigError(
            "Execution-failure artifact does not prove one bound SPATHI tool failure"
        )
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_manifest_sha256": sha256_file(dataset.manifest_path),
        "generation_seed": dataset.generation_seed,
        "algorithm_seed": algorithm_seed,
        "configuration_id": configuration_id,
        "selection_role": metadata_spec["selection_role"],
        "inference_artifact": artifact.key.digest,
        "failure_marker_sha256": sha256_file(marker_path),
        "run_report_sha256": sha256_file(report_path),
        "tool_status": tool_status,
        "failure_message": failure_message,
    }


def _inference_from_artifact(
    artifact: Artifact,
    *,
    dataset: _Dataset,
    parameters: Mapping[str, Any],
    configuration_id: str,
    configuration_sha256: str,
    policy_sha256: str,
    group_count: int,
    spathi_commit: str,
    algorithm_seed: int,
    seed_derivation: Mapping[str, Any],
    expression_statistics: tuple[int, dict[str, tuple[int, float, float]]],
    execution_contract: Mapping[str, Any],
    execution_order_block: Mapping[str, Any],
    execution_position: int,
    batch_kind: str,
) -> _Inference:
    _spec, _metadata_spec, run_id = _validate_inference_binding(
        artifact,
        dataset=dataset,
        parameters=parameters,
        configuration_id=configuration_id,
        configuration_sha256=configuration_sha256,
        policy_sha256=policy_sha256,
        group_count=group_count,
        spathi_commit=spathi_commit,
        algorithm_seed=algorithm_seed,
        seed_derivation=seed_derivation,
        execution_contract=execution_contract,
        execution_order_block=execution_order_block,
        execution_position=execution_position,
        batch_kind=batch_kind,
    )

    network_path = _single_member(artifact, "io/out/network.csv", "raw SPATHI network")
    eligibility_path = _single_member(
        artifact, "io/out/target_eligibility.tsv.gz", "target eligibility sidecar"
    )
    diagnostics_path = _single_member(
        artifact, "io/out/model_diagnostics.tsv.gz", "model diagnostics sidecar"
    )
    metadata_path = _single_member(artifact, "io/out/run_metadata.json", "SPATHI metadata sidecar")
    report_path = _single_member(artifact, "run_report.json", "ANDREA inference report")
    report = load_json_object(report_path, "ANDREA inference report")
    if (
        report.get("status") != "executed"
        or report.get("output_profile") != execution_contract["resources"]["output_profile"]
        or report.get("dataset", {}).get("id") != dataset.dataset_id
    ):
        raise BenchmarkConfigError("ANDREA report does not describe the expected completed dataset")
    if report.get("dataset", {}).get("fingerprint", {}).get("value") != dataset.fingerprint:
        raise BenchmarkConfigError(
            "ANDREA report dataset fingerprint differs from the truth package"
        )
    selected = report.get("tools", {}).get("selected")
    if not isinstance(selected, list) or len(selected) != 1:
        raise BenchmarkConfigError(
            "Evidence requires exactly one SPATHI tool per inference artifact"
        )
    if selected[0] != run_id or report.get("run_id") != run_id:
        raise BenchmarkConfigError("ANDREA report run identity differs from the immutable request")
    tools = report["tools"]
    if (
        tools.get("completed") != [run_id]
        or tools.get("status_by_tool", {}).get(run_id)
        not in {"completed", "completed_with_warnings"}
        or tuple(sorted(tools.get("completed_contexts", {}).get(run_id, []))) != dataset.contexts
        or tools.get("failed")
    ):
        raise BenchmarkConfigError(
            "ANDREA report does not prove complete execution of every context"
        )
    inputs = _input_members(artifact)
    expected_input_hashes = {
        "expression": sha256_file(dataset.expression_path),
        "groups": sha256_file(dataset.groups_path),
        "tf_list": sha256_file(dataset.tf_path),
    }
    if any(sha256_file(inputs[name]) != digest for name, digest in expected_input_hashes.items()):
        raise BenchmarkConfigError(
            "Inference input files differ from the validated dataset artifact"
        )
    metadata_counts = _validate_run_metadata(
        metadata_path,
        input_members=inputs,
        parameters=parameters,
        algorithm_seed=algorithm_seed,
        dataset=dataset,
    )
    globally_eligible = _validate_eligibility(
        eligibility_path,
        dataset=dataset,
        parameters=parameters,
        expression_statistics=expression_statistics,
    )
    accounting = read_model_accounting(
        diagnostics_path,
        contexts=dataset.contexts,
        targets=dataset.targets,
        parameters=parameters,
        algorithm_seed=algorithm_seed,
        globally_eligible=globally_eligible,
    )
    accounting.validate_metadata(load_json_object(metadata_path)["models"])
    if (
        metadata_counts["targets_processed"] != len(globally_eligible)
        or metadata_counts["models_requested"] != accounting.models_requested
        or metadata_counts["trees_fitted"] != accounting.trees_fitted
    ):
        raise BenchmarkConfigError(
            "SPATHI metadata accounting differs from validated eligibility/diagnostics"
        )
    measurements = _resource_measurements(
        report,
        metadata_counts=metadata_counts,
        execution_contract=execution_contract,
    )
    return _Inference(
        artifact=artifact,
        network_path=network_path,
        eligibility_path=eligibility_path,
        diagnostics_path=diagnostics_path,
        metadata_path=metadata_path,
        report_path=report_path,
        measurements=measurements,
        network=_network(network_path, dataset),
        eligible_targets=accounting.eligible_targets,
    )


def _space(sources: Sequence[str], targets: Sequence[str]) -> tuple[str, int, int]:
    overlap = len(set(sources) & set(targets))
    count = len(sources) * len(targets) - overlap
    digest = canonical_json_sha256(
        {"allow_self_edges": False, "sources": sorted(sources), "targets": sorted(targets)}
    )
    return digest, overlap, count


def _aupr(
    *,
    sources: Sequence[str],
    targets: Sequence[str],
    truth: frozenset[tuple[str, str]],
    predictions: Mapping[tuple[str, str], float],
) -> float:
    source_set = set(sources)
    target_set = set(targets)
    positives = {edge for edge in truth if edge[0] in source_set and edge[1] in target_set}
    if not positives:
        raise BenchmarkConfigError(
            "Directed AUPR is undefined because the evaluation space has no positives"
        )
    _digest, _overlap, candidate_count = _space(sources, targets)
    if candidate_count == len(positives):
        raise BenchmarkConfigError(
            "Directed AUPR is undefined because the evaluation space has no negatives"
        )
    buckets: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    explicit = 0
    explicit_positive = 0
    for edge, score in predictions.items():
        if edge[0] not in source_set or edge[1] not in target_set:
            continue
        buckets[score][0] += 1
        buckets[score][1] += int(edge in positives)
        explicit += 1
        explicit_positive += int(edge in positives)
    missing = candidate_count - explicit
    missing_positive = len(positives) - explicit_positive
    if missing < 0 or missing_positive < 0:
        raise BenchmarkConfigError("Prediction table cannot be reconciled with the candidate space")
    buckets[0.0][0] += missing
    buckets[0.0][1] += missing_positive
    seen = 0
    true_positive = 0
    area = 0.0
    for score in sorted(buckets, reverse=True):
        count, positive = buckets[score]
        seen += count
        true_positive += positive
        if positive:
            area += (positive / len(positives)) * (true_positive / seen)
    return area


def _semantic_list_hash(values: Sequence[str]) -> str:
    return canonical_json_sha256(sorted(values))


def _evidence_builder_sha256(repo_root: Path, *, safety: bool) -> str:
    digest = hashlib.sha256()
    files = (
        ("src/spathi_bench/noninferiority_evidence.py", Path(__file__).resolve()),
        (
            "src/spathi_bench/spathi_diagnostics.py",
            Path(__file__).with_name("spathi_diagnostics.py"),
        ),
        (
            "schemas/noninferiority-panel.schema.json",
            repo_root / "schemas/noninferiority-panel.schema.json",
        ),
        (
            "schemas/noninferiority-artifact-index.schema.json",
            repo_root / "schemas/noninferiority-artifact-index.schema.json",
        ),
        (
            "schemas/noninferiority-evidence.schema.json",
            repo_root / "schemas/noninferiority-evidence.schema.json",
        ),
        (
            "src/spathi_bench/efficiency_order.py",
            Path(__file__).with_name("efficiency_order.py"),
        ),
    ) + (
        (
            (
                "schemas/efficiency-safety-panel.schema.json",
                repo_root / "schemas/efficiency-safety-panel.schema.json",
            ),
            (
                "schemas/efficiency-safety-plan.schema.json",
                repo_root / "schemas/efficiency-safety-plan.schema.json",
            ),
        )
        if safety
        else ()
    )
    for relative, path in files:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_panel(repo_root: Path, plan: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    reference = plan["panel"]
    path = (repo_root / reference["path"]).resolve()
    _inside(repo_root, path, "Predeclared panel")
    if sha256_file(path) != reference["sha256"]:
        raise BenchmarkConfigError("Predeclared panel changed after plan materialization")
    panel = _validated_json(
        path,
        repo_root / "schemas/noninferiority-panel.schema.json",
        "predeclared non-inferiority panel",
    )
    if (
        panel["budget"] != plan["budget"]["id"]
        or panel["dataset_role"] != plan["analysis"]["validation_dataset_role"]
        or panel["selection_validation_manifest_sha256"]
        != plan["reference"]["selection_validation_manifest_sha256"]
    ):
        raise BenchmarkConfigError("Predeclared panel is not bound to the plan/reference")
    identities = [
        (item["dataset_id"], item["simulator_id"], item["generation_seed"])
        for item in panel["datasets"]
    ]
    if len(identities) != len(set(identities)):
        raise BenchmarkConfigError("Predeclared panel contains duplicate datasets")
    independent = [(simulator, seed) for _dataset, simulator, seed in identities]
    if len(independent) != len(set(independent)):
        raise BenchmarkConfigError("Predeclared panel reuses an independent simulation unit")
    expected = int(plan["budget"]["expected_independent_datasets"])
    if len(identities) != expected:
        raise BenchmarkConfigError(
            f"Predeclared panel must contain exactly {expected} independent datasets"
        )
    return path, panel


def construct_noninferiority_evidence(
    *,
    repo_root: Path,
    plan_path: Path,
    artifact_index_path: Path,
) -> dict[str, Any]:
    """Construct and fully verify a canonical evidence payload from real artifacts."""

    root = repo_root.resolve()
    _inside(root, plan_path, "Non-inferiority plan")
    _inside(root, artifact_index_path, "Non-inferiority artifact index")
    plan = load_json_object(plan_path, "efficiency policy-pair plan")
    safety = plan.get("artifact_type") == "spathi_efficiency_safety_plan"
    # Local imports avoid cycles while ensuring evidence cannot be built from a
    # plan whose source selection or formal efficiency decision has drifted.
    if safety:
        from .efficiency_safety import load_safety_panel, validate_safety_plan

        validate_json(
            plan,
            root / "schemas/efficiency-safety-plan.schema.json",
            "efficiency safety plan",
        )
        validate_safety_plan(root, plan)
        panel_path, panel = load_safety_panel(root, plan)
    else:
        from .noninferiority import _validate_plan_binding

        validate_json(
            plan,
            root / "schemas/noninferiority-plan.schema.json",
            "non-inferiority plan",
        )
        _validate_plan_binding(root, plan)
        panel_path, panel = _load_panel(root, plan)
    namespace = (
        "full.efficiency_safety" if safety else f"{plan['budget']['id']}.efficiency_validation"
    )
    split_id = plan["analysis"]["split_id"] if safety else "efficiency_validation_v1"
    batch_kind = "spathi_efficiency_safety" if safety else "spathi_efficiency"
    index = _validated_json(
        artifact_index_path,
        root / "schemas/noninferiority-artifact-index.schema.json",
        "non-inferiority artifact index",
    )
    expected_index_type = (
        "spathi_efficiency_safety_artifact_index" if safety else "spathi_efficiency_artifact_index"
    )
    if index["artifact_type"] != expected_index_type:
        raise BenchmarkConfigError("Artifact index type does not match its efficiency plan")
    plan_digest = sha256_file(plan_path)
    panel_digest = sha256_file(panel_path)
    if index["plan_sha256"] != plan_digest or index["panel_sha256"] != panel_digest:
        raise BenchmarkConfigError("Artifact index is not bound to this exact plan and panel")
    execution = index["execution"]
    current_runner = runner_implementation_sha256()
    if execution["runner_sha256"] != current_runner and not permit_revision_change(
        "Efficiency evidence runner", execution["runner_sha256"], current_runner
    ):
        raise BenchmarkConfigError(
            "Artifact index was produced by a different efficiency runner implementation"
        )
    if (
        execution["spathi_commit"] != plan["reference"]["spathi_commit"]
        or execution["spathi_build_fingerprint"] != plan["reference"]["spathi_build_fingerprint"]
        or execution["spathi_image_id"] != plan["reference"]["spathi_image_id"]
    ):
        raise BenchmarkConfigError("Artifact index uses a different frozen SPATHI build")
    expected_order_derivation = {
        "namespace": namespace,
        "stream": "execution_order",
        "coordinates": {"purpose": "counterbalanced_execution_order"},
    }
    if (
        execution["execution_order_seed"] != plan["budget"]["execution_order_seed"]
        or execution["execution_order_seed_derivation"] != expected_order_derivation
    ):
        raise BenchmarkConfigError(
            "Artifact index execution order is not bound to the predeclared seed"
        )
    store_path = (root / index["artifact_store"]).resolve()
    _inside(root, store_path, "Artifact store")
    store = ArtifactStore(store_path)
    declarations = {item["dataset_id"]: item for item in panel["datasets"]}
    candidates = {item["id"]: item for item in plan["candidates"]}
    expected_runs = {
        (dataset_id, seed)
        for dataset_id in declarations
        for seed in plan["budget"]["algorithm_seeds"]
    }
    indexed_runs: dict[tuple[str, int], dict[str, Any]] = {}
    expected_orders = counterbalanced_execution_orders(
        declarations,
        plan["budget"]["algorithm_seeds"],
        ["reference", *sorted(candidates)],
        execution_order_seed=plan["budget"]["execution_order_seed"],
    )
    for run in index["runs"]:
        key = (run["dataset_id"], run["algorithm_seed"])
        if key in indexed_runs:
            raise BenchmarkConfigError(f"Artifact index contains duplicate run {key!r}")
        indexed_runs[key] = run
        candidate_ids = [item["candidate_id"] for item in run["candidates"]]
        if len(candidate_ids) != len(set(candidate_ids)) or set(candidate_ids) != set(candidates):
            raise BenchmarkConfigError(f"Artifact index has incomplete candidates for {key!r}")
        if run["execution_order"] != expected_orders.get(key):
            raise BenchmarkConfigError(
                f"Artifact index has a non-predeclared execution order for {key!r}"
            )
    if set(indexed_runs) != expected_runs:
        raise BenchmarkConfigError(
            "Artifact index does not contain the exact predeclared dataset-by-seed product"
        )

    datasets: dict[str, _Dataset] = {}
    expression_statistics: dict[str, tuple[int, dict[str, tuple[int, float, float]]]] = {}
    for dataset_id, declaration in declarations.items():
        dataset_artifact = _artifact(store, declaration["dataset_artifact"], "dataset")
        datasets[dataset_id] = _dataset_from_artifact(
            dataset_artifact,
            declaration,
            dataset_role=panel["dataset_role"],
            split_id=split_id,
            generation_contract=(
                panel["condition_scales"][declaration["condition_id"]] if safety else None
            ),
        )
        expression_statistics[dataset_id] = _expression_statistics(datasets[dataset_id])
        if (
            safety
            and expression_statistics[dataset_id][0]
            != panel["condition_scales"][declaration["condition_id"]]["cells"]
        ):
            raise BenchmarkConfigError(
                "Safety expression cell count differs from its frozen condition scale"
            )

    used_inference_artifacts: dict[str, tuple[str, int, str]] = {}
    rows: list[dict[str, Any]] = []
    execution_failures: list[dict[str, Any]] = []
    for (dataset_id, seed), indexed in sorted(indexed_runs.items()):
        dataset = datasets[dataset_id]
        execution_order = indexed["execution_order"]
        execution_order_block = {
            "dataset_id": dataset_id,
            "algorithm_seed": seed,
            "execution_order_seed": execution["execution_order_seed"],
            "execution_order": list(execution_order),
        }
        replicate_index = plan["budget"]["algorithm_seeds"].index(seed) + 1
        seed_derivation = (
            dict(plan["seed_contract"]["algorithm_seed"]["seed_derivation"])
            if safety
            else {
                "namespace": namespace,
                "stream": "inference",
                "coordinates": {
                    "purpose": "algorithm_replicate",
                    "replicate_index": replicate_index,
                },
            }
        )
        if indexed["seed_derivation"] != seed_derivation:
            raise BenchmarkConfigError(
                f"Artifact index seed derivation differs for {dataset_id!r}/{seed}"
            )
        group_count = len(dataset.contexts)
        reference_parameters = resolve_group_policy(plan["reference"]["group_policy"], group_count)
        reference_configuration_sha256 = canonical_json_sha256(reference_parameters)
        reference_artifact = _artifact(store, indexed["reference"], "inference")
        reference_identity = (dataset_id, seed, "reference")
        previous = used_inference_artifacts.setdefault(
            reference_artifact.key.digest, reference_identity
        )
        if previous != reference_identity:
            raise BenchmarkConfigError(
                "One inference artifact is reused for different planned runs"
            )
        reference_failure = (
            _execution_failure_from_artifact(
                reference_artifact,
                dataset=dataset,
                parameters=reference_parameters,
                configuration_id=plan["reference"]["policy_id"],
                configuration_sha256=reference_configuration_sha256,
                policy_sha256=plan["reference"]["policy_sha256"],
                group_count=group_count,
                spathi_commit=plan["reference"]["spathi_commit"],
                algorithm_seed=seed,
                seed_derivation=seed_derivation,
                execution_contract=execution,
                execution_order_block=execution_order_block,
                execution_position=execution_order.index("reference") + 1,
                batch_kind=batch_kind,
            )
            if safety
            else None
        )
        if reference_failure is None:
            reference = _inference_from_artifact(
                reference_artifact,
                dataset=dataset,
                parameters=reference_parameters,
                configuration_id=plan["reference"]["policy_id"],
                configuration_sha256=reference_configuration_sha256,
                policy_sha256=plan["reference"]["policy_sha256"],
                group_count=group_count,
                spathi_commit=plan["reference"]["spathi_commit"],
                algorithm_seed=seed,
                seed_derivation=seed_derivation,
                expression_statistics=expression_statistics[dataset_id],
                execution_contract=execution,
                execution_order_block=execution_order_block,
                execution_position=execution_order.index("reference") + 1,
                batch_kind=batch_kind,
            )
        else:
            execution_failures.append(reference_failure)
            reference = None
        full_targets = tuple(sorted(dataset.targets))
        sources = tuple(sorted(dataset.sources))
        source_hash = _semantic_list_hash(sources)
        full_target_hash = _semantic_list_hash(full_targets)
        by_candidate = {item["candidate_id"]: item for item in indexed["candidates"]}
        for candidate_id, candidate in sorted(candidates.items()):
            candidate_parameters = resolve_group_policy(candidate["group_policy"], group_count)
            candidate_configuration_sha256 = canonical_json_sha256(candidate_parameters)
            candidate_artifact = _artifact(
                store, by_candidate[candidate_id]["inference_artifact"], "inference"
            )
            identity = (dataset_id, seed, candidate_id)
            previous = used_inference_artifacts.setdefault(candidate_artifact.key.digest, identity)
            if previous != identity:
                raise BenchmarkConfigError(
                    "One inference artifact is reused for different configurations"
                )
            candidate_failure = _execution_failure_from_artifact(
                candidate_artifact,
                dataset=dataset,
                parameters=candidate_parameters,
                configuration_id=candidate_id,
                configuration_sha256=candidate_configuration_sha256,
                policy_sha256=candidate["policy_sha256"],
                group_count=group_count,
                spathi_commit=plan["reference"]["spathi_commit"],
                algorithm_seed=seed,
                seed_derivation=seed_derivation,
                execution_contract=execution,
                execution_order_block=execution_order_block,
                execution_position=execution_order.index(candidate_id) + 1,
                batch_kind=batch_kind,
            )
            if candidate_failure is None:
                inferred = _inference_from_artifact(
                    candidate_artifact,
                    dataset=dataset,
                    parameters=candidate_parameters,
                    configuration_id=candidate_id,
                    configuration_sha256=candidate_configuration_sha256,
                    policy_sha256=candidate["policy_sha256"],
                    group_count=group_count,
                    spathi_commit=plan["reference"]["spathi_commit"],
                    algorithm_seed=seed,
                    seed_derivation=seed_derivation,
                    expression_statistics=expression_statistics[dataset_id],
                    execution_contract=execution,
                    execution_order_block=execution_order_block,
                    execution_position=execution_order.index(candidate_id) + 1,
                    batch_kind=batch_kind,
                )
            else:
                execution_failures.append(candidate_failure)
                continue
            if reference is None:
                continue
            for context in dataset.contexts:
                candidate_targets = inferred.eligible_targets[context]
                candidate_target_hash = _semantic_list_hash(candidate_targets)
                full_space_hash, full_overlap, full_count = _space(sources, full_targets)
                truth = dataset.truth[context]
                truth_outside_candidate_space = dataset.truth_outside_candidate_space.get(
                    context, frozenset()
                )
                retained_truth = {edge for edge in truth if edge[1] in set(candidate_targets)}
                target_coverage = None
                intersection_diagnostic = None
                if candidate["intersection_diagnostic"]:
                    target_coverage = {
                        "eligibility_manifest_sha256": canonical_json_sha256(
                            {
                                "context": context,
                                "eligible_targets": list(candidate_targets),
                                "policy_parameters": {
                                    key: candidate_parameters[key]
                                    for key in (
                                        "target_eligibility",
                                        "min_target_detected_cells",
                                        "min_target_detected_fraction",
                                        "min_target_weighted_detected_fraction",
                                        "min_target_weighted_detected_ess",
                                    )
                                },
                            }
                        ),
                        "truth_accessed_by_eligibility": False,
                        "n_reference_truth_edges": len(truth),
                        "n_retained_truth_edges": len(retained_truth),
                    }
                    if candidate_targets and retained_truth:
                        intersection_hash, _intersection_overlap, intersection_count = _space(
                            sources, candidate_targets
                        )
                        intersection_diagnostic = {
                            "target_universe_sha256": candidate_target_hash,
                            "candidate_space_sha256": intersection_hash,
                            "n_targets": len(candidate_targets),
                            "n_candidates": intersection_count,
                            "n_truth_edges": len(retained_truth),
                            "reference_value": _aupr(
                                sources=sources,
                                targets=candidate_targets,
                                truth=truth,
                                predictions=reference.network[context],
                            ),
                            "candidate_value": _aupr(
                                sources=sources,
                                targets=candidate_targets,
                                truth=truth,
                                predictions=inferred.network[context],
                            ),
                        }
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "dataset_id": dataset_id,
                        "dataset_manifest_sha256": sha256_file(dataset.manifest_path),
                        "context_id": context,
                        "simulator_id": dataset.simulator_id,
                        "condition_id": dataset.condition_id,
                        "generation_seed": dataset.generation_seed,
                        "algorithm_seed": seed,
                        "reference_policy_sha256": plan["reference"]["policy_sha256"],
                        "candidate_policy_sha256": candidate["policy_sha256"],
                        "reference_configuration_sha256": reference_configuration_sha256,
                        "candidate_configuration_sha256": candidate_configuration_sha256,
                        "metric": "aupr",
                        "level": "directed",
                        "reference_value": _aupr(
                            sources=sources,
                            targets=full_targets,
                            truth=truth,
                            predictions=reference.network[context],
                        ),
                        "candidate_value": _aupr(
                            sources=sources,
                            targets=full_targets,
                            truth=truth,
                            predictions=inferred.network[context],
                        ),
                        "reference_measurements": reference.measurements,
                        "candidate_measurements": inferred.measurements,
                        "candidate_space": {
                            "policy": candidate["primary_evaluation_policy"],
                            "construction": "directed_source_target_cartesian_excluding_self_edges",
                            "omitted_target_prediction_policy": "minimum_score_zero",
                            "reference_source_universe_sha256": source_hash,
                            "candidate_source_universe_sha256": source_hash,
                            "reference_target_universe_sha256": full_target_hash,
                            "candidate_target_universe_sha256": candidate_target_hash,
                            "reference_evaluation_target_universe_sha256": full_target_hash,
                            "candidate_evaluation_target_universe_sha256": full_target_hash,
                            "reference_evaluation_candidate_space_sha256": full_space_hash,
                            "candidate_evaluation_candidate_space_sha256": full_space_hash,
                            "n_reference_sources": len(sources),
                            "n_candidate_sources": len(sources),
                            "n_reference_targets": len(full_targets),
                            "n_candidate_targets": len(candidate_targets),
                            "n_reference_evaluation_targets": len(full_targets),
                            "n_candidate_evaluation_targets": len(full_targets),
                            "n_reference_evaluation_source_target_overlaps": full_overlap,
                            "n_candidate_evaluation_source_target_overlaps": full_overlap,
                            "n_reference_evaluation_candidates": full_count,
                            "n_candidate_evaluation_candidates": full_count,
                            "n_truth_edges_in_candidate_space": len(truth),
                            "n_truth_edges_outside_candidate_space": len(
                                truth_outside_candidate_space
                            ),
                            "truth_outside_candidate_space_examples": [
                                f"{source}->{target}"
                                for source, target in sorted(truth_outside_candidate_space)[:8]
                            ],
                        },
                        "target_coverage": target_coverage,
                        "intersection_diagnostic": intersection_diagnostic,
                        "source_artifacts": {
                            "dataset": dataset.artifact.key.digest,
                            "reference_inference": reference.artifact.key.digest,
                            "candidate_inference": inferred.artifact.key.digest,
                            "reference_network_sha256": sha256_file(reference.network_path),
                            "candidate_network_sha256": sha256_file(inferred.network_path),
                            "reference_target_eligibility_sha256": sha256_file(
                                reference.eligibility_path
                            ),
                            "reference_model_diagnostics_sha256": sha256_file(
                                reference.diagnostics_path
                            ),
                            "reference_run_metadata_sha256": sha256_file(reference.metadata_path),
                            "reference_run_report_sha256": sha256_file(reference.report_path),
                            "candidate_target_eligibility_sha256": sha256_file(
                                inferred.eligibility_path
                            ),
                            "candidate_model_diagnostics_sha256": sha256_file(
                                inferred.diagnostics_path
                            ),
                            "candidate_run_metadata_sha256": sha256_file(inferred.metadata_path),
                            "candidate_run_report_sha256": sha256_file(inferred.report_path),
                        },
                    }
                )
    evidence = {
        "schema_version": "1.0",
        "artifact_type": (
            "spathi_efficiency_safety_evidence" if safety else "spathi_efficiency_paired_evidence"
        ),
        "study_id": load_json_object(root / plan["study"]["path"], "study")["id"],
        "plan_sha256": plan_digest,
        "panel_sha256": panel_digest,
        "artifact_index": {
            "path": _inside(root, artifact_index_path, "Non-inferiority artifact index"),
            "sha256": sha256_file(artifact_index_path),
        },
        "evidence_builder_sha256": _evidence_builder_sha256(root, safety=safety),
        "selection_validation_manifest_sha256": panel["selection_validation_manifest_sha256"],
        "budget": panel["budget"],
        "dataset_role": panel["dataset_role"],
        "execution_failures": execution_failures,
        "rows": rows,
    }
    validate_json(
        evidence,
        root / "schemas/noninferiority-evidence.schema.json",
        "constructed non-inferiority evidence",
    )
    return evidence


def write_noninferiority_evidence(
    *,
    repo_root: Path,
    plan_path: Path,
    artifact_index_path: Path,
    output_path: Path,
) -> Path:
    """Build evidence and publish it atomically, refusing incompatible overwrite."""

    root = repo_root.resolve()
    _inside(root, output_path, "Non-inferiority evidence output")
    payload = construct_noninferiority_evidence(
        repo_root=root,
        plan_path=plan_path,
        artifact_index_path=artifact_index_path,
    )
    if output_path.exists():
        recorded = _validated_json(
            output_path, root / "schemas/noninferiority-evidence.schema.json", "existing evidence"
        )
        payload = _retain_evidence_implementation(recorded, payload)
    return write_idempotent_json(
        output_path,
        payload,
        "non-inferiority evidence",
        boundary=root,
    )


def _retain_evidence_implementation(
    recorded: Mapping[str, Any], rebuilt: dict[str, Any]
) -> dict[str, Any]:
    """Keep original producer attribution while checking every rebuilt scientific value."""

    previous = recorded["evidence_builder_sha256"]
    current = rebuilt["evidence_builder_sha256"]
    if previous != current and permit_revision_change("Efficiency evidence builder", previous, current):
        return {**rebuilt, "evidence_builder_sha256": previous}
    return rebuilt


def verify_noninferiority_evidence(
    *,
    repo_root: Path,
    plan_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    """Rebuild evidence from its index and reject any self-declared divergence."""

    root = repo_root.resolve()
    evidence = _validated_json(
        evidence_path,
        root / "schemas/noninferiority-evidence.schema.json",
        "paired non-inferiority evidence",
    )
    index_path = (root / evidence["artifact_index"]["path"]).resolve()
    _inside(root, index_path, "Non-inferiority artifact index")
    if sha256_file(index_path) != evidence["artifact_index"]["sha256"]:
        raise BenchmarkConfigError("Artifact index changed after evidence construction")
    rebuilt = construct_noninferiority_evidence(
        repo_root=root,
        plan_path=plan_path,
        artifact_index_path=index_path,
    )
    rebuilt = _retain_evidence_implementation(evidence, rebuilt)
    if evidence != rebuilt:
        raise BenchmarkConfigError(
            "Evidence does not exactly match values recalculated from immutable source artifacts"
        )
    return rebuilt


__all__ = [
    "construct_noninferiority_evidence",
    "verify_noninferiority_evidence",
    "write_noninferiority_evidence",
]
