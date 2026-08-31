"""Executable, bounded operations for the SPATHI calibration contract.

Selection rows are deliberately more granular than the final benchmark summary. Each
row represents one configuration, dataset and algorithmic seed. ``aupr_directed`` is
the equal-weight mean across evaluable groups in that dataset. This module then gives
algorithmic seeds equal weight inside each dataset and datasets equal weight overall.
"""

from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    resolve_repo_path,
    validate_json,
)

SEARCH_SPACE_PATH = Path("experiments/v1/spathi/search-space.json")
SEARCH_SPACE_SCHEMA_PATH = Path("schemas/spathi-search-space.schema.json")
RANKING_SCHEMA_PATH = Path("schemas/calibration-ranking.schema.json")
MANIFEST_SCHEMA_PATH = Path("schemas/calibration-manifest.schema.json")
SELECTED_CONFIG_SCHEMA_PATH = Path("schemas/spathi-selected-config.schema.json")

_CALIBRATION_BUDGETS = frozenset({"reduced", "full"})
_CALIBRATION_STAGES = frozenset({"convergence", "structure", "estimator", "validation"})
_PARENT_STAGES = {
    "convergence": None,
    "structure": "convergence",
    "estimator": "structure",
    "validation": "estimator",
}
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class CalibrationError(BenchmarkConfigError):
    """Raised when calibration inputs violate the predeclared design."""


@dataclass(frozen=True, slots=True)
class CandidateConfiguration:
    """One deterministic SPATHI configuration produced by a search stage."""

    configuration_id: str
    configuration_sha256: str
    stage_id: str
    dataset_role: str
    parent_configuration_id: str | None
    variant_id: str
    parameters: Mapping[str, Any]
    nonreference_parameter_changes: int


@dataclass(frozen=True, slots=True)
class CalibrationRun:
    """One candidate/algorithmic-seed execution requested by a search stage."""

    run_id: str
    configuration_id: str
    configuration_sha256: str
    stage_id: str
    dataset_role: str
    algorithm_seed: int
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Hierarchical score and ordered tie breakers for one candidate."""

    candidate: CandidateConfiguration
    hierarchical_aupr: float
    stability: float
    median_wall_time_seconds: float
    median_peak_memory_mb: float | None
    simplicity: int
    dataset_count: int
    algorithm_seed_count: int


def candidate_to_dict(candidate: CandidateConfiguration) -> dict[str, Any]:
    return {
        "configuration_id": candidate.configuration_id,
        "configuration_sha256": candidate.configuration_sha256,
        "stage_id": candidate.stage_id,
        "dataset_role": candidate.dataset_role,
        "parent_configuration_id": candidate.parent_configuration_id,
        "variant_id": candidate.variant_id,
        "parameters": copy.deepcopy(dict(candidate.parameters)),
        "nonreference_parameter_changes": candidate.nonreference_parameter_changes,
    }


def candidate_from_dict(payload: Mapping[str, Any]) -> CandidateConfiguration:
    """Restore and minimally type-check a candidate emitted by this module."""

    try:
        parent = payload["parent_configuration_id"]
        if parent is not None and not isinstance(parent, str):
            raise TypeError("parent_configuration_id")
        return CandidateConfiguration(
            configuration_id=str(payload["configuration_id"]),
            configuration_sha256=str(payload["configuration_sha256"]),
            stage_id=str(payload["stage_id"]),
            dataset_role=str(payload["dataset_role"]),
            parent_configuration_id=parent,
            variant_id=str(payload["variant_id"]),
            parameters=copy.deepcopy(dict(_mapping(payload["parameters"], "candidate parameters"))),
            nonreference_parameter_changes=_integer(
                payload["nonreference_parameter_changes"],
                "candidate simplicity",
                minimum=0,
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"Invalid serialized calibration candidate: {payload!r}") from exc


def score_to_dict(score: CandidateScore) -> dict[str, Any]:
    return {
        "candidate": candidate_to_dict(score.candidate),
        "hierarchical_aupr": score.hierarchical_aupr,
        "stability": score.stability,
        "median_wall_time_seconds": score.median_wall_time_seconds,
        "median_peak_memory_mb": score.median_peak_memory_mb,
        "simplicity": score.simplicity,
        "dataset_count": score.dataset_count,
        "algorithm_seed_count": score.algorithm_seed_count,
    }


def _score_from_dict(payload: object) -> CandidateScore:
    raw = _mapping(payload, "calibration score")
    try:
        raw_memory = raw["median_peak_memory_mb"]
        memory = (
            None
            if raw_memory is None
            else _finite_number(raw_memory, "score peak memory", minimum=0)
        )
        candidate = candidate_from_dict(_mapping(raw["candidate"], "score candidate"))
        simplicity = _integer(raw["simplicity"], "score simplicity", minimum=0)
        if simplicity != candidate.nonreference_parameter_changes:
            raise CalibrationError("Score simplicity differs from its candidate")
        return CandidateScore(
            candidate=candidate,
            hierarchical_aupr=_finite_number(
                raw["hierarchical_aupr"],
                "score hierarchical AUPR",
                minimum=0,
                maximum=1,
            ),
            stability=_finite_number(raw["stability"], "score stability", minimum=0),
            median_wall_time_seconds=_finite_number(
                raw["median_wall_time_seconds"],
                "score wall time",
                minimum=0,
            ),
            median_peak_memory_mb=memory,
            simplicity=simplicity,
            dataset_count=_integer(raw["dataset_count"], "score dataset count", minimum=1),
            algorithm_seed_count=_integer(
                raw["algorithm_seed_count"],
                "score algorithm seed count",
                minimum=2,
            ),
        )
    except KeyError as exc:
        raise CalibrationError(f"Calibration score is missing {exc.args[0]!r}") from exc


def _score_sort_key(score: CandidateScore) -> tuple[float, float, float, float, int, str]:
    return (
        -score.hierarchical_aupr,
        score.stability,
        score.median_wall_time_seconds,
        score.median_peak_memory_mb if score.median_peak_memory_mb is not None else math.inf,
        score.simplicity,
        score.candidate.configuration_id,
    )


def locked_spathi_identity(repo_root: Path) -> dict[str, str]:
    """Return the source and deterministic build identity required for calibration."""

    image = load_image_lock(repo_root.resolve())["images"]["spathi"]
    fingerprint = canonical_json_sha256(
        {
            "source_commit": image["source_commit"],
            "base_image": image["base_image"],
            "build_inputs": image["build_inputs"],
            "python_dependencies": image["python_dependencies"],
        }
    )
    return {
        "spathi_commit": str(image["source_commit"]),
        "spathi_build_fingerprint": fingerprint,
        "spathi_image_tag": str(image["tag"]),
    }


def _normalized_evidence_rows(
    summary_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    required = (
        "configuration_id",
        "configuration_sha256",
        "dataset_role",
        "dataset_id",
        "algorithm_seed",
        "aupr_directed",
        "wall_time_seconds",
        "peak_memory_mb",
        "n_group_contexts",
        "group_contexts",
        "spathi_commit",
        "spathi_build_fingerprint",
        "spathi_image_id",
        "run_id",
        "execution_sha256",
    )
    optional: tuple[str, ...] = ()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(summary_rows):
        row = _mapping(raw, f"summary row {index}")
        missing = [field for field in required if field not in row]
        if missing:
            raise CalibrationError(
                f"Summary row {index} is missing ranking evidence fields: {missing}"
            )
        normalized.append(
            {
                **{field: copy.deepcopy(row[field]) for field in required},
                **{field: copy.deepcopy(row[field]) for field in optional if field in row},
            }
        )
    return sorted(
        normalized,
        key=lambda row: (
            str(row["configuration_id"]),
            str(row["dataset_id"]),
            int(row["algorithm_seed"]),
        ),
    )


def load_search_space(repo_root: Path) -> dict[str, Any]:
    """Load and validate the repository's v1 SPATHI search-space contract."""

    root = repo_root.resolve()
    search_space = load_json_object(root / SEARCH_SPACE_PATH, "SPATHI search space")
    validate_json(
        search_space,
        root / SEARCH_SPACE_SCHEMA_PATH,
        "SPATHI search space",
    )
    _validate_search_semantics(search_space)
    return search_space


def stage_profile_id(stage_id: str, calibration_budget: str) -> str:
    """Return the canonical runnable profile ID for one stage and evidence budget."""

    _validate_calibration_stage(stage_id)
    budget = _validate_calibration_budget_id(calibration_budget)
    purpose = "selection" if stage_id == "validation" else "calibration"
    return f"{purpose}-{budget}-{stage_id}"


def canonical_run_records(runs: Sequence[CalibrationRun]) -> list[dict[str, Any]]:
    """Serialize the complete run identity used to reconcile summary evidence."""

    records = [
        {
            "run_id": run.run_id,
            "configuration_id": run.configuration_id,
            "configuration_sha256": run.configuration_sha256,
            "algorithm_seed": run.algorithm_seed,
            "execution_sha256": canonical_json_sha256(dict(run.parameters)),
        }
        for run in runs
    ]
    if len({record["run_id"] for record in records}) != len(records):
        raise CalibrationError("Calibration runs contain duplicate run IDs")
    return records


def build_stage_profile(
    template: Mapping[str, Any],
    stage_id: str,
    calibration_budget: str,
    *,
    bundle_path: str,
) -> dict[str, Any]:
    """Materialize one stage-specific runnable profile from its planned template."""

    budget = _validate_calibration_budget_id(calibration_budget)
    profile_id = stage_profile_id(stage_id, budget)
    expected_purpose = "selection" if stage_id == "validation" else "calibration"
    expected_role = "validation" if stage_id == "validation" else "development"
    if template.get("status") != "planned" or template.get("purpose") != expected_purpose:
        raise CalibrationError(
            f"Stage {stage_id!r} requires a planned {expected_purpose!r} profile template"
        )
    if template.get("calibration_budget") != budget:
        raise CalibrationError(
            f"Stage {stage_id!r} requires a {budget!r} calibration-budget template"
        )
    raw_jobs = template.get("generation_jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise CalibrationError(f"The {expected_purpose} template has no frozen generation jobs")
    observed_roles = {job.get("dataset_role") for job in raw_jobs if isinstance(job, Mapping)}
    if observed_roles != {expected_role}:
        raise CalibrationError(
            f"Stage {stage_id!r} requires only {expected_role!r} generation jobs; "
            f"found {sorted(str(role) for role in observed_roles)}"
        )
    if not isinstance(bundle_path, str) or not bundle_path:
        raise CalibrationError("Calibration profile requires a repository-relative bundle path")

    profile = copy.deepcopy(dict(template))
    profile.update(
        {
            "id": profile_id,
            "status": "runnable",
            "calibration_stage": stage_id,
            "calibration_budget": budget,
            "description": (
                f"{str(template.get('description', '')).strip()} "
                f"Materialized immutable {stage_id} stage."
            ).strip(),
            "inference_batches": [
                {
                    "id": f"spathi-{stage_id}",
                    "kind": "spathi",
                    "dataset_role": expected_role,
                    "regime": "group_only",
                    "tools_params": f"{bundle_path}/tools-params.json",
                    "custom_tools": f"{bundle_path}/custom-tools.json",
                    "calibration_manifest": f"{bundle_path}/calibration-manifest.json",
                    "run_metadata": f"{bundle_path}/run-metadata.json",
                }
            ],
        }
    )
    statistics = _mapping(profile.get("statistics"), "profile statistics")
    profile["statistics"] = {
        **copy.deepcopy(dict(statistics)),
        "status": "not_applicable",
        "comparisons": [],
    }
    return profile


def load_parent_ranking(
    repo_root: Path,
    search_space: Mapping[str, Any],
    stage_id: str,
    calibration_budget: str,
    ranking_path: Path | None,
) -> tuple[tuple[CandidateConfiguration, ...], dict[str, Any] | None]:
    """Validate a preceding ranking and return its frozen shortlist and link."""

    budget = _validate_calibration_budget(search_space, calibration_budget)
    if stage_id not in _PARENT_STAGES:
        raise CalibrationError(f"Unknown calibration stage: {stage_id!r}")
    expected_stage = _PARENT_STAGES[stage_id]
    if expected_stage is None:
        if ranking_path is not None:
            raise CalibrationError("The convergence stage cannot receive --parents")
        return (), None
    if ranking_path is None:
        raise CalibrationError(f"Stage {stage_id!r} requires --parents from {expected_stage!r}")

    root = repo_root.resolve()
    resolved = ranking_path.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise CalibrationError(
            "Parent ranking must be stored inside the benchmark repository"
        ) from exc
    resolved = resolve_repo_path(root, relative, "parent calibration ranking")
    ranking = load_json_object(resolved, "parent calibration ranking")
    validate_json(ranking, root / RANKING_SCHEMA_PATH, "parent calibration ranking")
    search_digest = sha256_file(root / SEARCH_SPACE_PATH)
    locked_identity = locked_spathi_identity(root)
    if any(ranking.get(field) != value for field, value in locked_identity.items()):
        raise CalibrationError("Parent ranking was produced by another SPATHI build")
    if ranking.get("stage_id") != expected_stage:
        raise CalibrationError(
            f"Stage {stage_id!r} requires a {expected_stage!r} ranking, "
            f"not {ranking.get('stage_id')!r}"
        )
    if ranking.get("calibration_budget") != budget:
        raise CalibrationError(
            f"Stage {stage_id!r} requires a {budget!r} calibration ranking, "
            f"not {ranking.get('calibration_budget')!r}"
        )
    validate_ranking_chain(
        root,
        search_space,
        ranking,
        expected_search_space_sha256=search_digest,
        expected_calibration_budget=budget,
    )
    raw_candidates = ranking.get("selected_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise CalibrationError("Parent ranking contains no selected candidates")
    candidates = tuple(candidate_from_dict(item) for item in raw_candidates)
    return candidates, {
        "path": relative,
        "sha256": sha256_file(resolved),
        "stage_id": expected_stage,
        "calibration_budget": budget,
    }


def expand_stage_candidates(
    search_space: Mapping[str, Any],
    stage_id: str,
    *,
    parents: Sequence[CandidateConfiguration] = (),
) -> tuple[CandidateConfiguration, ...]:
    """Expand one stage into a deterministic, bounded set of configurations.

    ``convergence`` starts from ``reference_configuration`` and takes no parents.
    Later stages require the selected/shortlisted candidates from the preceding
    stage. Parent input order is ignored so repeated expansion is stable.
    """

    stage = _stage(search_space, stage_id)
    reference = _mapping(search_space.get("reference_configuration"), "reference_configuration")
    parent_items = _validated_parents(stage, parents)
    _validate_candidate_integrity(
        tuple(parent for parent in parent_items if parent is not None),
        reference,
    )
    variants = [
        (str(item["id"]), _mapping(item["parameter_overrides"], "parameter_overrides"))
        for item in _sequence(stage.get("candidates"), f"{stage_id}.candidates")
    ]
    expected_per_parent = int(bool(stage["include_inherited_configuration"])) + len(variants)
    budget = _integer(stage.get("candidate_budget_per_parent"), "candidate budget", minimum=1)
    if expected_per_parent > budget:
        raise CalibrationError(
            f"Stage {stage_id!r} expands {expected_per_parent} configurations per parent, "
            f"above its budget of {budget}"
        )

    candidates: list[CandidateConfiguration] = []
    for parent in parent_items:
        base = dict(reference if parent is None else parent.parameters)
        prefix = stage_id if parent is None else f"{stage_id}__{parent.configuration_id}"
        inherited_variant = "reference" if parent is None else "inherited"
        if stage["include_inherited_configuration"]:
            candidates.append(
                _candidate(
                    stage_id=stage_id,
                    role=str(stage["dataset_role"]),
                    configuration_id=f"{prefix}__{inherited_variant}",
                    parent=parent,
                    variant_id=inherited_variant,
                    parameters=base,
                    reference=reference,
                )
            )
        for variant_id, overrides in variants:
            parameters = {**base, **overrides}
            candidates.append(
                _candidate(
                    stage_id=stage_id,
                    role=str(stage["dataset_role"]),
                    configuration_id=f"{prefix}__{variant_id}",
                    parent=parent,
                    variant_id=variant_id,
                    parameters=parameters,
                    reference=reference,
                )
            )

    identifiers = [candidate.configuration_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise CalibrationError(f"Stage {stage_id!r} generated duplicate configuration IDs")
    digests = [candidate.configuration_sha256 for candidate in candidates]
    if len(set(digests)) != len(digests):
        raise CalibrationError(f"Stage {stage_id!r} generated duplicate parameter configurations")
    return tuple(candidates)


def expand_stage_runs(
    search_space: Mapping[str, Any],
    stage_id: str,
    calibration_budget: str,
    *,
    parents: Sequence[CandidateConfiguration] = (),
) -> tuple[CalibrationRun, ...]:
    """Expand every stage candidate across its predeclared algorithmic seeds."""

    _stage(search_space, stage_id)
    seeds = _algorithmic_seeds(search_space, calibration_budget, stage_id)
    candidates = expand_stage_candidates(search_space, stage_id, parents=parents)
    return tuple(
        CalibrationRun(
            run_id=f"{candidate.configuration_id}__seed_{seed}",
            configuration_id=candidate.configuration_id,
            configuration_sha256=candidate.configuration_sha256,
            stage_id=stage_id,
            dataset_role=candidate.dataset_role,
            algorithm_seed=seed,
            parameters={**copy.deepcopy(dict(candidate.parameters)), "random_seed": seed},
        )
        for candidate in candidates
        for seed in seeds
    )


def expand_validation_runs(
    search_space: Mapping[str, Any],
    finalists: Sequence[CandidateConfiguration],
    calibration_budget: str,
) -> tuple[CalibrationRun, ...]:
    """Expand the bounded estimator finalists on validation-only algorithm seeds."""

    protocol = _mapping(search_space.get("selection_protocol"), "selection_protocol")
    maximum = _integer(
        protocol.get("maximum_validation_finalists"),
        "maximum validation finalists",
        minimum=1,
    )
    if not 2 <= len(finalists) <= maximum:
        raise CalibrationError(
            f"Validation requires between 2 and {maximum} estimator finalists; "
            f"received {len(finalists)}"
        )
    if any(candidate.stage_id != "estimator" for candidate in finalists):
        raise CalibrationError("Validation finalists must come from the estimator stage")
    reference = _mapping(search_space.get("reference_configuration"), "reference_configuration")
    _validate_candidate_integrity(finalists, reference)
    seeds = _algorithmic_seeds(search_space, calibration_budget, "validation")
    return tuple(
        CalibrationRun(
            run_id=f"validation__{candidate.configuration_id}__seed_{seed}",
            configuration_id=candidate.configuration_id,
            configuration_sha256=candidate.configuration_sha256,
            stage_id="validation",
            dataset_role="validation",
            algorithm_seed=seed,
            parameters={**copy.deepcopy(dict(candidate.parameters)), "random_seed": seed},
        )
        for candidate in sorted(finalists, key=lambda item: item.configuration_id)
        for seed in seeds
    )


def validate_calibration_manifest(
    repo_root: Path,
    search_space: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[tuple[CandidateConfiguration, ...], tuple[CalibrationRun, ...]]:
    """Reconstruct and validate every candidate and run frozen by a manifest."""

    root = repo_root.resolve()
    validate_json(dict(manifest), root / MANIFEST_SCHEMA_PATH, "calibration manifest")
    stage_id = str(manifest.get("stage_id"))
    _validate_calibration_stage(stage_id, label="calibration manifest stage")
    calibration_budget = _validate_calibration_budget(
        search_space,
        str(manifest.get("calibration_budget")),
    )
    search_digest = sha256_file(root / SEARCH_SPACE_PATH)
    if (
        manifest.get("search_space_id") != search_space.get("id")
        or manifest.get("search_space_sha256") != search_digest
    ):
        raise CalibrationError("Calibration manifest search-space identity is stale")
    if manifest.get("profile_id") != stage_profile_id(stage_id, calibration_budget):
        raise CalibrationError("Calibration manifest has the wrong stage profile ID")
    locked_identity = locked_spathi_identity(root)
    if any(manifest.get(field) != value for field, value in locked_identity.items()):
        raise CalibrationError("Calibration manifest was generated for another SPATHI build")

    raw_parent = manifest.get("parent_ranking")
    parent_path: Path | None = None
    if raw_parent is not None:
        parent = _mapping(raw_parent, "manifest parent ranking")
        parent_path = root / str(parent.get("path"))
    parents, expected_parent = load_parent_ranking(
        root,
        search_space,
        stage_id,
        calibration_budget,
        parent_path,
    )
    if raw_parent != expected_parent:
        raise CalibrationError("Calibration manifest parent-ranking link is stale")

    if stage_id == "validation":
        candidates = parents
        runs = expand_validation_runs(search_space, candidates, calibration_budget)
        expected_role = "validation"
    else:
        candidates = expand_stage_candidates(search_space, stage_id, parents=parents)
        runs = expand_stage_runs(
            search_space,
            stage_id,
            calibration_budget,
            parents=parents,
        )
        expected_role = "development"
    if manifest.get("dataset_role") != expected_role:
        raise CalibrationError("Calibration manifest has the wrong dataset role")
    if manifest.get("candidates") != [candidate_to_dict(item) for item in candidates]:
        raise CalibrationError("Calibration manifest candidates differ from the search contract")
    if manifest.get("runs") != canonical_run_records(runs):
        raise CalibrationError("Calibration manifest runs differ from the canonical stage plan")
    return candidates, runs


def reconcile_summary_runs(
    manifest: Mapping[str, Any],
    summary_rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Require every summary row to originate from exactly one frozen manifest run."""

    rows = list(summary_rows)
    expected_items = manifest.get("runs")
    if not isinstance(expected_items, list) or not expected_items:
        raise CalibrationError("Calibration manifest contains no canonical runs")
    expected = {
        str(item["run_id"]): {
            "configuration_id": item["configuration_id"],
            "configuration_sha256": item["configuration_sha256"],
            "algorithm_seed": item["algorithm_seed"],
            "execution_sha256": item["execution_sha256"],
        }
        for item in expected_items
        if isinstance(item, Mapping)
    }
    if len(expected) != len(expected_items):
        raise CalibrationError("Calibration manifest contains invalid or duplicate run IDs")

    observed: set[str] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"summary row {index}")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or run_id not in expected:
            raise CalibrationError(
                f"Summary row {index} references unknown calibration run {run_id!r}"
            )
        signature = expected[run_id]
        mismatches = {
            field: (row.get(field), value)
            for field, value in signature.items()
            if row.get(field) != value
        }
        if mismatches:
            raise CalibrationError(
                f"Summary row {index} differs from manifest run {run_id!r}: {mismatches}"
            )
        observed.add(run_id)
    missing = sorted(set(expected) - observed)
    if missing:
        raise CalibrationError(f"Summary is missing manifest runs: {missing}")
    return rows


def build_andrea_requests(
    runs: Sequence[CalibrationRun],
    *,
    docker_image: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Render calibration runs as ANDREA tools, custom-tool and metadata requests."""

    if not runs:
        raise CalibrationError("Cannot build ANDREA requests for an empty calibration stage")
    if not docker_image:
        raise CalibrationError("Calibration requests require a SPATHI Docker image")
    run_ids = [run.run_id for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise CalibrationError("Calibration runs contain duplicate run IDs")
    tools_params = {
        "runs": [
            {
                "run_id": run.run_id,
                "tool_id": f"custom_{run.run_id}",
                "params": copy.deepcopy(dict(run.parameters)),
                "execution": {"mode": "group_native"},
            }
            for run in runs
        ]
    }
    custom_tools = {
        "tools": [
            {
                "run_id": run.run_id,
                "name": f"SPATHI calibration {run.configuration_id} seed {run.algorithm_seed}",
                "docker_image": docker_image,
                "execution_mode": "group_native",
                "extra_inputs": ["groups", "tf_list"],
                "outputs": {"directed": True, "sign": "none"},
            }
            for run in runs
        ]
    }
    metadata = {
        "schema_version": "1.0",
        "runs": [
            {
                "run_id": run.run_id,
                "configuration_id": run.configuration_id,
                "selection_role": "reference"
                if run.configuration_id.endswith("__reference")
                else "candidate",
                "algorithm_seed": run.algorithm_seed,
                "algorithm_seed_parameter": "random_seed",
                "family": "spathi",
            }
            for run in runs
        ],
    }
    return tools_params, custom_tools, metadata


def select_stage_candidates(
    search_space: Mapping[str, Any],
    stage_id: str,
    candidates: Sequence[CandidateConfiguration],
    summary_rows: Iterable[Mapping[str, Any]],
    *,
    calibration_budget: str,
) -> tuple[CandidateScore, ...]:
    """Rank a complete development-stage batch and return its bounded shortlist."""

    scores = rank_stage_candidates(
        search_space,
        stage_id,
        candidates,
        summary_rows,
        calibration_budget=calibration_budget,
    )
    stage = _stage(search_space, stage_id)
    limit = _integer(stage.get("advance_limit"), f"{stage_id} advance limit", minimum=1)
    if limit > len(scores):
        raise CalibrationError(
            f"Stage {stage_id!r} advance limit {limit} exceeds its {len(scores)} candidates"
        )
    return scores[:limit]


def rank_stage_candidates(
    search_space: Mapping[str, Any],
    stage_id: str,
    candidates: Sequence[CandidateConfiguration],
    summary_rows: Iterable[Mapping[str, Any]],
    *,
    calibration_budget: str,
) -> tuple[CandidateScore, ...]:
    """Return the complete ordered ranking for one development stage."""

    stage = _stage(search_space, stage_id)
    reference = _mapping(search_space.get("reference_configuration"), "reference_configuration")
    _validate_candidate_integrity(candidates, reference)
    _validate_stage_candidate_set(stage, candidates)
    expected_seeds = _algorithmic_seeds(search_space, calibration_budget, stage_id)
    scores = _rank_balanced_rows(
        candidates,
        summary_rows,
        expected_role="development",
        expected_seeds=expected_seeds,
    )
    return tuple(scores)


def select_validation_candidate(
    search_space: Mapping[str, Any],
    finalists: Sequence[CandidateConfiguration],
    summary_rows: Iterable[Mapping[str, Any]],
    *,
    calibration_budget: str,
) -> CandidateScore:
    """Select one global candidate using validation rows, never test rows.

    This function returns evidence in memory; it does not freeze or modify
    ``selected-config.json``.
    """

    return rank_validation_candidates(
        search_space,
        finalists,
        summary_rows,
        calibration_budget=calibration_budget,
    )[0]


def rank_validation_candidates(
    search_space: Mapping[str, Any],
    finalists: Sequence[CandidateConfiguration],
    summary_rows: Iterable[Mapping[str, Any]],
    *,
    calibration_budget: str,
) -> tuple[CandidateScore, ...]:
    """Return the complete validation ranking without freezing a selection."""

    protocol = _mapping(search_space.get("selection_protocol"), "selection_protocol")
    maximum = _integer(
        protocol.get("maximum_validation_finalists"),
        "maximum validation finalists",
        minimum=1,
    )
    if not 2 <= len(finalists) <= maximum:
        raise CalibrationError(
            f"Validation requires between 2 and {maximum} predeclared finalists; "
            f"received {len(finalists)}"
        )
    if any(candidate.stage_id != "estimator" for candidate in finalists):
        raise CalibrationError("Validation finalists must come from the estimator stage")
    reference = _mapping(search_space.get("reference_configuration"), "reference_configuration")
    _validate_candidate_integrity(finalists, reference)
    expected_seeds = _algorithmic_seeds(search_space, calibration_budget, "validation")
    return tuple(
        _rank_balanced_rows(
            finalists,
            summary_rows,
            expected_role="validation",
            expected_seeds=expected_seeds,
        )
    )


def build_calibration_ranking(
    search_space: Mapping[str, Any],
    stage_id: str,
    candidates: Sequence[CandidateConfiguration],
    summary_rows: Iterable[Mapping[str, Any]],
    *,
    calibration_budget: str,
    profile_id: str,
    parent_ranking: Mapping[str, Any] | None,
    search_space_sha256: str,
    calibration_manifest_sha256: str,
    benchmark_summary_sha256: str,
    benchmark_commit: str,
    spathi_commit: str,
    spathi_build_fingerprint: str,
    spathi_image_tag: str,
) -> dict[str, Any]:
    """Build a self-consistent ranking artifact without changing selected-config."""

    budget = _validate_calibration_budget(search_space, calibration_budget)
    if profile_id != stage_profile_id(stage_id, budget):
        raise CalibrationError("Calibration ranking has the wrong stage profile ID")
    if _GIT_COMMIT.fullmatch(benchmark_commit) is None:
        raise CalibrationError("Calibration ranking requires a valid benchmark commit")
    expected_parent_stage = _PARENT_STAGES[stage_id]
    if expected_parent_stage is None:
        if parent_ranking is not None:
            raise CalibrationError("Convergence ranking cannot link a parent ranking")
    else:
        parent = _mapping(parent_ranking, "ranking parent link")
        if parent.get("stage_id") != expected_parent_stage:
            raise CalibrationError("Calibration ranking links the wrong parent stage")
        if parent.get("calibration_budget") != budget:
            raise CalibrationError("Calibration ranking links a different calibration budget")
    evidence_rows = _normalized_evidence_rows(summary_rows)
    if stage_id == "validation":
        scores = rank_validation_candidates(
            search_space,
            candidates,
            evidence_rows,
            calibration_budget=budget,
        )
        selected_scores = scores[:1]
        dataset_role = "validation"
    else:
        scores = rank_stage_candidates(
            search_space,
            stage_id,
            candidates,
            evidence_rows,
            calibration_budget=budget,
        )
        stage = _stage(search_space, stage_id)
        limit = _integer(stage.get("advance_limit"), f"{stage_id} advance limit", minimum=1)
        selected_scores = scores[:limit]
        dataset_role = "development"
    winner = selected_scores[0] if stage_id == "validation" else None
    payload = {
        "schema_version": "1.0",
        "artifact_type": "spathi_calibration_ranking",
        "calibration_budget": budget,
        "profile_id": profile_id,
        "search_space_id": search_space["id"],
        "search_space_sha256": search_space_sha256,
        "stage_id": stage_id,
        "dataset_role": dataset_role,
        "parent_ranking": None if parent_ranking is None else copy.deepcopy(dict(parent_ranking)),
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "benchmark_summary_sha256": benchmark_summary_sha256,
        "benchmark_commit": benchmark_commit,
        "spathi_commit": spathi_commit,
        "spathi_build_fingerprint": spathi_build_fingerprint,
        "spathi_image_tag": spathi_image_tag,
        "spathi_image_ids": sorted({str(row["spathi_image_id"]) for row in evidence_rows}),
        "primary_metric": "aupr_directed",
        "tie_breakers": [item["id"] for item in search_space["objective"]["tie_breakers"]],
        "calibration_rows": evidence_rows,
        "ranking": [score_to_dict(score) for score in scores],
        "selected_candidates": [candidate_to_dict(score.candidate) for score in selected_scores],
        "winner": None
        if winner is None
        else {
            "candidate_id": winner.candidate.configuration_id,
            "configuration_sha256": winner.candidate.configuration_sha256,
            "hierarchical_score": winner.hierarchical_aupr,
        },
        "test_data_accessed": False,
    }
    validate_ranking_artifact(
        search_space,
        payload,
        expected_search_space_sha256=search_space_sha256,
        expected_calibration_budget=budget,
    )
    return payload


def validate_ranking_artifact(
    search_space: Mapping[str, Any],
    ranking: Mapping[str, Any],
    *,
    expected_search_space_sha256: str,
    expected_calibration_budget: str | None = None,
) -> tuple[CandidateScore, ...]:
    """Validate ranking identity, ordering, candidates and winner reconciliation."""

    if ranking.get("search_space_id") != search_space.get("id"):
        raise CalibrationError("Calibration ranking references a different search space")
    if ranking.get("search_space_sha256") != expected_search_space_sha256:
        raise CalibrationError("Calibration ranking search-space hash is stale")
    if ranking.get("test_data_accessed") is not False:
        raise CalibrationError("Calibration ranking must not access test data")
    expected_ties = [item["id"] for item in search_space["objective"]["tie_breakers"]]
    if (
        ranking.get("primary_metric") != "aupr_directed"
        or ranking.get("tie_breakers") != expected_ties
    ):
        raise CalibrationError("Calibration ranking objective differs from the search contract")

    calibration_budget = _validate_calibration_budget(
        search_space,
        str(ranking.get("calibration_budget")),
    )
    if (
        expected_calibration_budget is not None
        and calibration_budget
        != _validate_calibration_budget(
            search_space,
            expected_calibration_budget,
        )
    ):
        raise CalibrationError("Calibration ranking uses a different calibration budget")
    stage_id = ranking.get("stage_id")
    role = ranking.get("dataset_role")
    if ranking.get("profile_id") != stage_profile_id(str(stage_id), calibration_budget):
        raise CalibrationError("Calibration ranking has the wrong stage profile ID")
    if _GIT_COMMIT.fullmatch(str(ranking.get("benchmark_commit", ""))) is None:
        raise CalibrationError("Calibration ranking has an invalid benchmark commit")
    expected_parent_stage = _PARENT_STAGES.get(str(stage_id))
    parent_ranking = ranking.get("parent_ranking")
    if expected_parent_stage is None:
        if parent_ranking is not None:
            raise CalibrationError("Convergence ranking cannot link a parent ranking")
    else:
        parent = _mapping(parent_ranking, "ranking parent link")
        if parent.get("stage_id") != expected_parent_stage:
            raise CalibrationError("Calibration ranking links the wrong parent stage")
        if parent.get("calibration_budget") != calibration_budget:
            raise CalibrationError("Calibration ranking links a different calibration budget")
    if stage_id == "validation":
        if role != "validation":
            raise CalibrationError("Validation ranking has the wrong dataset role")
    elif stage_id in {"convergence", "structure", "estimator"}:
        if role != "development":
            raise CalibrationError("Development ranking has the wrong dataset role")
    else:
        raise CalibrationError(f"Unknown calibration ranking stage: {stage_id!r}")

    expected_provenance = {
        "spathi_commit": ranking.get("spathi_commit"),
        "spathi_build_fingerprint": ranking.get("spathi_build_fingerprint"),
    }
    raw_image_ids = ranking.get("spathi_image_ids")
    if not isinstance(raw_image_ids, list) or len(raw_image_ids) != 1:
        raise CalibrationError("Calibration ranking must use exactly one SPATHI image ID")

    raw_scores = ranking.get("ranking")
    if not isinstance(raw_scores, list) or not raw_scores:
        raise CalibrationError("Calibration ranking contains no scores")
    serialized_scores = tuple(_score_from_dict(item) for item in raw_scores)
    reference = _mapping(search_space.get("reference_configuration"), "reference_configuration")
    candidates = tuple(score.candidate for score in serialized_scores)
    _validate_candidate_integrity(candidates, reference)
    identifiers = [candidate.configuration_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise CalibrationError("Calibration ranking contains duplicate candidates")
    if (
        len({score.dataset_count for score in serialized_scores}) != 1
        or len({score.algorithm_seed_count for score in serialized_scores}) != 1
    ):
        raise CalibrationError("Calibration ranking scores use an unbalanced evaluation panel")

    raw_rows = ranking.get("calibration_rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise CalibrationError("Calibration ranking contains no auditable calibration rows")
    for index, row_value in enumerate(raw_rows):
        row = _mapping(row_value, f"ranking calibration row {index}")
        if any(row.get(field) != expected for field, expected in expected_provenance.items()):
            raise CalibrationError(
                f"Calibration row {index} SPATHI source differs from ranking provenance"
            )
        if row.get("spathi_image_id") != raw_image_ids[0]:
            raise CalibrationError(
                f"Calibration row {index} SPATHI image differs from ranking provenance"
            )

    if stage_id == "validation":
        protocol = _mapping(search_space.get("selection_protocol"), "selection_protocol")
        maximum = int(protocol["maximum_validation_finalists"])
        if not 2 <= len(serialized_scores) <= maximum:
            raise CalibrationError(f"Validation ranking requires between 2 and {maximum} finalists")
        if any(candidate.stage_id != "estimator" for candidate in candidates):
            raise CalibrationError("Validation ranking contains a non-estimator candidate")
        recomputed_scores = rank_validation_candidates(
            search_space,
            candidates,
            raw_rows,
            calibration_budget=calibration_budget,
        )
        selected_scores = recomputed_scores[:1]
    else:
        stage = _stage(search_space, str(stage_id))
        _validate_stage_candidate_set(stage, candidates)
        recomputed_scores = rank_stage_candidates(
            search_space,
            str(stage_id),
            candidates,
            raw_rows,
            calibration_budget=calibration_budget,
        )
        selected_scores = recomputed_scores[: int(stage["advance_limit"])]

    if [score_to_dict(score) for score in recomputed_scores] != raw_scores:
        raise CalibrationError(
            "Calibration ranking scores are not reconciled with its calibration rows"
        )

    serialized_selected = ranking.get("selected_candidates")
    expected_selected = [candidate_to_dict(score.candidate) for score in selected_scores]
    if serialized_selected != expected_selected:
        raise CalibrationError("Calibration ranking shortlist is not reconciled with its scores")
    expected_winner = (
        {
            "candidate_id": recomputed_scores[0].candidate.configuration_id,
            "configuration_sha256": recomputed_scores[0].candidate.configuration_sha256,
            "hierarchical_score": recomputed_scores[0].hierarchical_aupr,
        }
        if stage_id == "validation"
        else None
    )
    if ranking.get("winner") != expected_winner:
        raise CalibrationError("Calibration ranking winner is not reconciled with its scores")
    return recomputed_scores


def validate_ranking_chain(
    repo_root: Path,
    search_space: Mapping[str, Any],
    ranking: Mapping[str, Any],
    *,
    expected_search_space_sha256: str,
    expected_calibration_budget: str | None = None,
    _visited: frozenset[str] = frozenset(),
) -> tuple[CandidateScore, ...]:
    """Validate a ranking and every content-addressed predecessor back to convergence."""

    root = repo_root.resolve()
    scores = validate_ranking_artifact(
        search_space,
        ranking,
        expected_search_space_sha256=expected_search_space_sha256,
        expected_calibration_budget=expected_calibration_budget,
    )
    stage_id = str(ranking["stage_id"])
    calibration_budget = str(ranking["calibration_budget"])
    expected_parent_stage = _PARENT_STAGES[stage_id]
    parent_value = ranking.get("parent_ranking")
    if expected_parent_stage is None:
        if parent_value is not None:
            raise CalibrationError("Convergence ranking has an unexpected parent link")
        return scores
    parent = _mapping(parent_value, "ranking parent link")
    raw_path = parent.get("path")
    if not isinstance(raw_path, str):
        raise CalibrationError("Ranking parent link has no repository-relative path")
    parent_path = resolve_repo_path(root, raw_path, "parent ranking evidence")
    digest = sha256_file(parent_path)
    if digest != parent.get("sha256"):
        raise CalibrationError("Parent ranking hash differs from its child link")
    if digest in _visited:
        raise CalibrationError("Calibration ranking chain contains a cycle")
    parent_ranking = load_json_object(parent_path, "parent ranking evidence")
    validate_json(parent_ranking, root / RANKING_SCHEMA_PATH, "parent ranking evidence")
    if parent_ranking.get("stage_id") != expected_parent_stage:
        raise CalibrationError("Calibration ranking chain has an invalid stage transition")
    if (
        parent.get("calibration_budget") != calibration_budget
        or parent_ranking.get("calibration_budget") != calibration_budget
    ):
        raise CalibrationError("Calibration ranking chain mixes calibration budgets")
    if parent_ranking.get("benchmark_commit") != ranking.get("benchmark_commit"):
        raise CalibrationError("Calibration ranking chain mixes benchmark commits")
    locked_identity = locked_spathi_identity(root)
    if any(parent_ranking.get(field) != value for field, value in locked_identity.items()):
        raise CalibrationError("Calibration ranking chain mixes SPATHI builds")
    validate_ranking_chain(
        root,
        search_space,
        parent_ranking,
        expected_search_space_sha256=expected_search_space_sha256,
        expected_calibration_budget=calibration_budget,
        _visited=_visited | {digest},
    )
    raw_selected = parent_ranking.get("selected_candidates")
    if not isinstance(raw_selected, list) or not raw_selected:
        raise CalibrationError("Parent ranking chain contains no selected candidates")
    selected_parents = tuple(candidate_from_dict(item) for item in raw_selected)
    current_candidates = tuple(score.candidate for score in scores)
    if stage_id == "validation":
        expected = {
            (candidate.configuration_id, candidate.configuration_sha256)
            for candidate in selected_parents
        }
        observed = {
            (candidate.configuration_id, candidate.configuration_sha256)
            for candidate in current_candidates
        }
        if observed != expected:
            raise CalibrationError(
                "Validation ranking finalists differ from the estimator shortlist"
            )
    else:
        expected_parent_ids = {candidate.configuration_id for candidate in selected_parents}
        observed_parent_ids = {
            candidate.parent_configuration_id for candidate in current_candidates
        }
        if observed_parent_ids != expected_parent_ids:
            raise CalibrationError(
                f"{stage_id.capitalize()} ranking parents differ from the preceding shortlist"
            )
    return scores


def build_frozen_selected_config(
    repo_root: Path,
    ranking_path: Path,
    *,
    frozen_at: str | None = None,
) -> dict[str, Any]:
    """Build, but do not persist, a selected-config linked to validation evidence."""

    root = repo_root.resolve()
    search = load_search_space(root)
    resolved_ranking = resolve_repo_path(
        root,
        ranking_path.resolve().relative_to(root).as_posix(),
        "validation ranking",
    )
    ranking = load_json_object(resolved_ranking, "validation ranking")
    validate_json(ranking, root / RANKING_SCHEMA_PATH, "validation ranking")
    locked_identity = locked_spathi_identity(root)
    if any(ranking.get(field) != value for field, value in locked_identity.items()):
        raise CalibrationError("Validation ranking was produced by a different SPATHI build")
    search_digest = sha256_file(root / SEARCH_SPACE_PATH)
    scores = validate_ranking_chain(
        root,
        search,
        ranking,
        expected_search_space_sha256=search_digest,
        expected_calibration_budget=str(ranking.get("calibration_budget")),
    )
    if ranking["stage_id"] != "validation":
        raise CalibrationError("Only a validation ranking can freeze selected-config")
    winner = scores[0]
    timestamp = frozen_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    relative_ranking = resolved_ranking.relative_to(root).as_posix()
    evidence = {
        "calibration_budget": ranking["calibration_budget"],
        "validation_ranking": relative_ranking,
        "validation_ranking_sha256": sha256_file(resolved_ranking),
        "search_space_sha256": search_digest,
        "validation_manifest_sha256": ranking["calibration_manifest_sha256"],
        "benchmark_summary_sha256": ranking["benchmark_summary_sha256"],
        "spathi_build_fingerprint": ranking["spathi_build_fingerprint"],
        "spathi_image_id": ranking["spathi_image_ids"][0],
        "primary_metric": "aupr_directed",
        "hierarchical_score": winner.hierarchical_aupr,
        "winner_candidate_id": winner.candidate.configuration_id,
        "winner_configuration_sha256": winner.candidate.configuration_sha256,
        "tie_breakers_applied": ranking["tie_breakers"],
        "frozen_at": timestamp,
    }
    selected = {
        "schema_version": "1.0",
        "id": "spathi_selected_v1",
        "search_space": SEARCH_SPACE_PATH.as_posix(),
        "calibration_budget": ranking["calibration_budget"],
        "status": "selected",
        "phase": "frozen",
        "selected_candidate": {
            "candidate_id": winner.candidate.configuration_id,
            "spathi_commit": ranking["spathi_commit"],
            "configuration_sha256": winner.candidate.configuration_sha256,
            "parameters": copy.deepcopy(dict(winner.candidate.parameters)),
        },
        "selection_evidence": evidence,
        "test_accessed_before_freeze": False,
        "notes": [
            "Generated from the cryptographically linked validation ranking.",
            "Test results cannot change this frozen global configuration.",
        ],
    }
    validate_json(selected, root / SELECTED_CONFIG_SCHEMA_PATH, "frozen selected configuration")
    reconcile_frozen_selection(root, search, selected)
    return selected


def reconcile_frozen_selection(
    repo_root: Path,
    search_space: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile selected-config with its immutable validation-ranking artifact."""

    if selected.get("status") != "selected" or selected.get("phase") != "frozen":
        raise CalibrationError("SPATHI selection is not frozen")
    root = repo_root.resolve()
    evidence = _mapping(selected.get("selection_evidence"), "selection evidence")
    raw_ranking_path = evidence.get("validation_ranking")
    if not isinstance(raw_ranking_path, str):
        raise CalibrationError("Frozen selection does not identify validation ranking evidence")
    ranking_path = resolve_repo_path(root, raw_ranking_path, "validation ranking evidence")
    if sha256_file(ranking_path) != evidence.get("validation_ranking_sha256"):
        raise CalibrationError("Validation ranking hash does not match selected-config")
    ranking = load_json_object(ranking_path, "validation ranking evidence")
    validate_json(ranking, root / RANKING_SCHEMA_PATH, "validation ranking evidence")
    locked_identity = locked_spathi_identity(root)
    if any(ranking.get(field) != value for field, value in locked_identity.items()):
        raise CalibrationError("Validation ranking was produced by a different SPATHI build")
    search_digest = sha256_file(root / SEARCH_SPACE_PATH)
    scores = validate_ranking_chain(
        root,
        search_space,
        ranking,
        expected_search_space_sha256=search_digest,
        expected_calibration_budget=str(selected.get("calibration_budget")),
    )
    if ranking["stage_id"] != "validation":
        raise CalibrationError("Frozen selection evidence is not a validation ranking")
    winner = scores[0]
    selected_candidate = _mapping(selected.get("selected_candidate"), "selected candidate")
    expected_candidate = {
        "candidate_id": winner.candidate.configuration_id,
        "spathi_commit": ranking["spathi_commit"],
        "configuration_sha256": winner.candidate.configuration_sha256,
        "parameters": dict(winner.candidate.parameters),
    }
    if dict(selected_candidate) != expected_candidate:
        raise CalibrationError(
            "Frozen selected candidate is not reconciled with validation ranking"
        )
    expected_evidence = {
        "calibration_budget": ranking["calibration_budget"],
        "validation_ranking": ranking_path.relative_to(root).as_posix(),
        "validation_ranking_sha256": sha256_file(ranking_path),
        "search_space_sha256": search_digest,
        "validation_manifest_sha256": ranking["calibration_manifest_sha256"],
        "benchmark_summary_sha256": ranking["benchmark_summary_sha256"],
        "spathi_build_fingerprint": ranking["spathi_build_fingerprint"],
        "spathi_image_id": ranking["spathi_image_ids"][0],
        "primary_metric": "aupr_directed",
        "hierarchical_score": winner.hierarchical_aupr,
        "winner_candidate_id": winner.candidate.configuration_id,
        "winner_configuration_sha256": winner.candidate.configuration_sha256,
        "tie_breakers_applied": ranking["tie_breakers"],
    }
    for field, expected in expected_evidence.items():
        if evidence.get(field) != expected:
            raise CalibrationError(
                f"Frozen selection evidence field {field!r} is not reconciled with ranking"
            )
    return ranking


def _validate_search_semantics(search_space: Mapping[str, Any]) -> None:
    budgets = _mapping(search_space.get("calibration_budgets"), "calibration_budgets")
    if set(budgets) != _CALIBRATION_BUDGETS:
        raise CalibrationError(
            "Calibration budgets must be exactly "
            f"{sorted(_CALIBRATION_BUDGETS)}; found {sorted(str(item) for item in budgets)}"
        )
    for calibration_budget in sorted(_CALIBRATION_BUDGETS):
        _validate_calibration_budget(search_space, calibration_budget)

    stages = _sequence(search_space.get("stages"), "stages")
    seen: set[str] = set()
    for stage_value in stages:
        stage = _mapping(stage_value, "stage")
        stage_id = str(stage.get("id"))
        if "algorithmic_seeds" in stage:
            raise CalibrationError(
                f"Stage {stage_id!r} contains legacy algorithmic_seeds; "
                "seeds must be declared per calibration budget"
            )
        candidate_values = _sequence(stage.get("candidates"), f"{stage_id}.candidates")
        candidate_ids = [str(_mapping(item, "candidate").get("id")) for item in candidate_values]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise CalibrationError(f"Stage {stage_id!r} contains duplicate candidate IDs")
        per_parent = int(bool(stage.get("include_inherited_configuration"))) + len(candidate_ids)
        budget = _integer(
            stage.get("candidate_budget_per_parent"),
            f"{stage_id} candidate budget",
            minimum=1,
        )
        if per_parent > budget:
            raise CalibrationError(
                f"Stage {stage_id!r} declares {per_parent} configurations per parent, "
                f"above budget {budget}"
            )
        if stage_id in seen:
            raise CalibrationError(f"Duplicate calibration stage: {stage_id!r}")
        seen.add(stage_id)
    if seen != _CALIBRATION_STAGES - {"validation"}:
        raise CalibrationError(
            "Calibration search stages must be exactly convergence, structure and estimator"
        )

    for stage_id in sorted(_CALIBRATION_STAGES):
        reduced = set(_algorithmic_seeds(search_space, "reduced", stage_id))
        full = set(_algorithmic_seeds(search_space, "full", stage_id))
        if not reduced < full:
            raise CalibrationError(
                f"Reduced {stage_id!r} seeds must be a strict subset of the full budget"
            )


def _validated_parents(
    stage: Mapping[str, Any],
    parents: Sequence[CandidateConfiguration],
) -> tuple[CandidateConfiguration | None, ...]:
    stage_id = str(stage["id"])
    if stage_id == "convergence":
        if parents:
            raise CalibrationError("The convergence stage cannot receive parent configurations")
        return (None,)
    if not parents:
        raise CalibrationError(f"Stage {stage_id!r} requires parent configurations")
    maximum = _integer(
        stage.get("maximum_parent_configurations"),
        f"{stage_id} maximum parents",
        minimum=1,
    )
    if len(parents) > maximum:
        raise CalibrationError(
            f"Stage {stage_id!r} accepts at most {maximum} parents; received {len(parents)}"
        )
    expected_parent_stage = "convergence" if stage_id == "structure" else "structure"
    identifiers = [parent.configuration_id for parent in parents]
    if len(set(identifiers)) != len(identifiers):
        raise CalibrationError(f"Stage {stage_id!r} received duplicate parents")
    for parent in parents:
        if parent.stage_id != expected_parent_stage:
            raise CalibrationError(
                f"Stage {stage_id!r} requires parents from {expected_parent_stage!r}, "
                f"not {parent.stage_id!r}"
            )
        if parent.dataset_role != "development":
            raise CalibrationError("Search-stage parents must come from development data")
    return tuple(sorted(parents, key=lambda candidate: candidate.configuration_id))


def _candidate(
    *,
    stage_id: str,
    role: str,
    configuration_id: str,
    parent: CandidateConfiguration | None,
    variant_id: str,
    parameters: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> CandidateConfiguration:
    owned_parameters = copy.deepcopy(dict(parameters))
    digest = _configuration_digest(owned_parameters)
    simplicity = sum(owned_parameters.get(key) != value for key, value in reference.items())
    return CandidateConfiguration(
        configuration_id=configuration_id,
        configuration_sha256=digest,
        stage_id=stage_id,
        dataset_role=role,
        parent_configuration_id=None if parent is None else parent.configuration_id,
        variant_id=variant_id,
        parameters=owned_parameters,
        nonreference_parameter_changes=simplicity,
    )


def _validate_stage_candidate_set(
    stage: Mapping[str, Any],
    candidates: Sequence[CandidateConfiguration],
) -> None:
    stage_id = str(stage["id"])
    if not candidates:
        raise CalibrationError(f"Stage {stage_id!r} has no candidates to select")
    if any(candidate.stage_id != stage_id for candidate in candidates):
        raise CalibrationError(f"Stage {stage_id!r} received candidates from another stage")
    if any(candidate.dataset_role != "development" for candidate in candidates):
        raise CalibrationError("Search-stage candidates must use development data")
    identifiers = [candidate.configuration_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise CalibrationError(f"Stage {stage_id!r} received duplicate candidates")

    expected_variants = {str(item["id"]) for item in stage["candidates"]}
    expected_variants.add("reference" if stage_id == "convergence" else "inherited")
    by_parent: dict[str | None, set[str]] = defaultdict(set)
    for candidate in candidates:
        by_parent[candidate.parent_configuration_id].add(candidate.variant_id)
    if stage_id == "convergence" and set(by_parent) != {None}:
        raise CalibrationError("Convergence candidates cannot have parent configurations")
    if stage_id != "convergence" and None in by_parent:
        raise CalibrationError(f"Stage {stage_id!r} candidates must identify their parent")
    maximum_parents = int(stage["maximum_parent_configurations"])
    if len(by_parent) > maximum_parents:
        raise CalibrationError(
            f"Stage {stage_id!r} candidate set exceeds its {maximum_parents}-parent budget"
        )
    for variants in by_parent.values():
        if variants != expected_variants:
            raise CalibrationError(
                f"Stage {stage_id!r} selection requires every predeclared variant; "
                f"expected {sorted(expected_variants)}, found {sorted(variants)}"
            )


def _validate_candidate_integrity(
    candidates: Sequence[CandidateConfiguration],
    reference: Mapping[str, Any],
) -> None:
    reference_keys = set(reference)
    for candidate in candidates:
        parameters = dict(candidate.parameters)
        if set(parameters) != reference_keys:
            raise CalibrationError(
                f"Candidate {candidate.configuration_id!r} does not contain the complete "
                "reference parameter set"
            )
        expected_digest = _configuration_digest(parameters)
        if candidate.configuration_sha256 != expected_digest:
            raise CalibrationError(
                f"Candidate {candidate.configuration_id!r} parameter digest is inconsistent"
            )
        expected_simplicity = sum(parameters[key] != value for key, value in reference.items())
        if candidate.nonreference_parameter_changes != expected_simplicity:
            raise CalibrationError(
                f"Candidate {candidate.configuration_id!r} simplicity is inconsistent"
            )


def _configuration_digest(parameters: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(parameters))


def _rank_balanced_rows(
    candidates: Sequence[CandidateConfiguration],
    summary_rows: Iterable[Mapping[str, Any]],
    *,
    expected_role: str,
    expected_seeds: tuple[int, ...] | None,
) -> list[CandidateScore]:
    by_id = {candidate.configuration_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise CalibrationError("Candidate IDs must be unique")
    rows = list(summary_rows)
    if not rows:
        raise CalibrationError("Calibration selection requires summary rows")

    values: dict[
        str,
        dict[str, dict[int, tuple[float, float, float | None, tuple[str, ...]]]],
    ] = defaultdict(lambda: defaultdict(dict))
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"summary row {index}")
        role = row.get("dataset_role")
        if role != expected_role:
            raise CalibrationError(
                f"Summary row {index} uses dataset_role={role!r}; expected {expected_role!r}"
            )
        configuration_id = row.get("configuration_id")
        if not isinstance(configuration_id, str) or configuration_id not in by_id:
            raise CalibrationError(
                f"Summary row {index} references unknown configuration {configuration_id!r}"
            )
        if row.get("configuration_sha256") != by_id[configuration_id].configuration_sha256:
            raise CalibrationError(
                f"Summary row {index} configuration digest differs from its frozen candidate"
            )
        dataset_id = row.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise CalibrationError(f"Summary row {index} has an invalid dataset_id")
        seed = _integer(row.get("algorithm_seed"), f"summary row {index} seed", minimum=0)
        expected_run_id = (
            f"validation__{configuration_id}__seed_{seed}"
            if expected_role == "validation"
            else f"{configuration_id}__seed_{seed}"
        )
        if row.get("run_id") != expected_run_id:
            raise CalibrationError(
                f"Summary row {index} run_id differs from its configuration and seed"
            )
        if seed in values[configuration_id][dataset_id]:
            raise CalibrationError(
                "Duplicate calibration summary row for "
                f"{configuration_id!r}, {dataset_id!r}, seed {seed}"
            )
        aupr = _finite_number(
            row.get("aupr_directed"),
            f"summary row {index} aupr_directed",
            minimum=0,
            maximum=1,
        )
        wall_time = _finite_number(
            row.get("wall_time_seconds"),
            f"summary row {index} wall_time_seconds",
            minimum=0,
        )
        raw_peak_memory = row.get("peak_memory_mb")
        peak_memory = (
            None
            if raw_peak_memory is None
            else _finite_number(
                raw_peak_memory,
                f"summary row {index} peak_memory_mb",
                minimum=0,
            )
        )
        raw_contexts = row.get("group_contexts")
        if (
            not isinstance(raw_contexts, list)
            or not raw_contexts
            or not all(
                isinstance(context, str) and context.startswith("group:")
                for context in raw_contexts
            )
            or len(set(raw_contexts)) != len(raw_contexts)
            or raw_contexts != sorted(raw_contexts)
        ):
            raise CalibrationError(
                f"Summary row {index} must contain sorted, unique group_contexts"
            )
        context_count = _integer(
            row.get("n_group_contexts"),
            f"summary row {index} n_group_contexts",
            minimum=1,
        )
        if context_count != len(raw_contexts):
            raise CalibrationError(
                f"Summary row {index} group-context count differs from its context list"
            )
        values[configuration_id][dataset_id][seed] = (
            aupr,
            wall_time,
            peak_memory,
            tuple(raw_contexts),
        )

    if set(values) != set(by_id):
        missing = sorted(set(by_id) - set(values))
        raise CalibrationError(f"Missing summary rows for candidates: {missing}")
    dataset_sets = {frozenset(per_dataset) for per_dataset in values.values()}
    if len(dataset_sets) != 1:
        raise CalibrationError("Every candidate must be evaluated on the same datasets")
    dataset_ids = sorted(next(iter(dataset_sets)))
    if not dataset_ids:
        raise CalibrationError("Calibration selection requires at least one dataset")

    observed_seed_sets: set[frozenset[int]] = set()
    for per_dataset in values.values():
        observed_seed_sets.update(frozenset(per_seed) for per_seed in per_dataset.values())
    if len(observed_seed_sets) != 1:
        raise CalibrationError("Every candidate/dataset pair must use the same algorithmic seeds")
    observed_seeds = tuple(sorted(next(iter(observed_seed_sets))))
    if expected_seeds is not None and set(observed_seeds) != set(expected_seeds):
        raise CalibrationError(
            f"Summary seed set {list(observed_seeds)} differs from the predeclared "
            f"set {sorted(expected_seeds)}"
        )
    if len(observed_seeds) < 2:
        raise CalibrationError("At least two algorithmic seeds are required for stability")
    for dataset_id in dataset_ids:
        context_sets = {
            executions[seed][3]
            for per_dataset in values.values()
            for executions in (per_dataset[dataset_id],)
            for seed in observed_seeds
        }
        if len(context_sets) != 1:
            raise CalibrationError(
                f"Every candidate and seed must evaluate the same group contexts for "
                f"dataset {dataset_id!r}"
            )
    memory_availability = {
        execution[2] is not None
        for per_dataset in values.values()
        for per_seed in per_dataset.values()
        for execution in per_seed.values()
    }
    if len(memory_availability) > 1:
        raise CalibrationError("Peak memory must be available for every run or for none")

    scores: list[CandidateScore] = []
    for configuration_id, candidate in by_id.items():
        dataset_means: list[float] = []
        dataset_stabilities: list[float] = []
        wall_times: list[float] = []
        peak_memories: list[float] = []
        for dataset_id in dataset_ids:
            executions = values[configuration_id][dataset_id]
            auprs = [executions[seed][0] for seed in observed_seeds]
            dataset_means.append(fmean(auprs))
            dataset_stabilities.append(pstdev(auprs))
            wall_times.extend(executions[seed][1] for seed in observed_seeds)
            peak_memories.extend(
                memory for seed in observed_seeds if (memory := executions[seed][2]) is not None
            )
        scores.append(
            CandidateScore(
                candidate=candidate,
                hierarchical_aupr=fmean(dataset_means),
                stability=fmean(dataset_stabilities),
                median_wall_time_seconds=median(wall_times),
                median_peak_memory_mb=median(peak_memories) if peak_memories else None,
                simplicity=candidate.nonreference_parameter_changes,
                dataset_count=len(dataset_ids),
                algorithm_seed_count=len(observed_seeds),
            )
        )
    return sorted(scores, key=_score_sort_key)


def _validate_calibration_stage(stage_id: object, *, label: str = "calibration stage") -> str:
    if not isinstance(stage_id, str) or stage_id not in _CALIBRATION_STAGES:
        raise CalibrationError(f"Unknown {label}: {stage_id!r}")
    return stage_id


def _validate_calibration_budget_id(calibration_budget: object) -> str:
    if not isinstance(calibration_budget, str) or calibration_budget not in _CALIBRATION_BUDGETS:
        raise CalibrationError(
            f"Unknown calibration budget: {calibration_budget!r}; "
            f"expected one of {sorted(_CALIBRATION_BUDGETS)}"
        )
    return calibration_budget


def _validate_calibration_budget(
    search_space: Mapping[str, Any],
    calibration_budget: object,
) -> str:
    budget_id = _validate_calibration_budget_id(calibration_budget)
    budgets = _mapping(search_space.get("calibration_budgets"), "calibration_budgets")
    budget = _mapping(budgets.get(budget_id), f"calibration budget {budget_id!r}")
    seeds = _mapping(
        budget.get("algorithmic_seeds"),
        f"{budget_id}.algorithmic_seeds",
    )
    if set(seeds) != _CALIBRATION_STAGES:
        raise CalibrationError(
            f"Calibration budget {budget_id!r} must declare seeds for exactly "
            f"{sorted(_CALIBRATION_STAGES)}"
        )
    for stage_id in sorted(_CALIBRATION_STAGES):
        raw = _sequence(seeds.get(stage_id), f"{budget_id}.{stage_id} algorithmic seeds")
        normalized = tuple(
            _integer(seed, f"{budget_id}.{stage_id} algorithmic seed", minimum=0) for seed in raw
        )
        if len(normalized) < 2:
            raise CalibrationError(
                f"Calibration budget {budget_id!r} stage {stage_id!r} requires at least "
                "two algorithmic seeds"
            )
        if len(set(normalized)) != len(normalized):
            raise CalibrationError(
                f"Calibration budget {budget_id!r} stage {stage_id!r} contains duplicate seeds"
            )
    return budget_id


def _algorithmic_seeds(
    search_space: Mapping[str, Any],
    calibration_budget: str,
    stage_id: str,
) -> tuple[int, ...]:
    budget_id = _validate_calibration_budget(search_space, calibration_budget)
    stage = _validate_calibration_stage(stage_id)
    budgets = _mapping(search_space.get("calibration_budgets"), "calibration_budgets")
    budget = _mapping(budgets[budget_id], f"calibration budget {budget_id!r}")
    seeds = _mapping(budget["algorithmic_seeds"], f"{budget_id}.algorithmic_seeds")
    return tuple(int(seed) for seed in _sequence(seeds[stage], f"{budget_id}.{stage} seeds"))


def _stage(search_space: Mapping[str, Any], stage_id: str) -> Mapping[str, Any]:
    stages = _sequence(search_space.get("stages"), "stages")
    matches = [
        _mapping(stage, "stage")
        for stage in stages
        if isinstance(stage, Mapping) and stage.get("id") == stage_id
    ]
    if len(matches) != 1:
        raise CalibrationError(f"Unknown or duplicate calibration stage: {stage_id!r}")
    return matches[0]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CalibrationError(f"{label} must be an array")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise CalibrationError(f"{label} must be an integer of at least {minimum}")
    return value


def _finite_number(
    value: object,
    label: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise CalibrationError(f"{label} must be finite and at least {minimum}")
    if maximum is not None and result > maximum:
        raise CalibrationError(f"{label} must be at most {maximum}")
    return result
