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
from .calibration_design import (
    FIDELITIES,
    MULTIFIDELITY_STAGES,
    calibration_phase,
    validate_multifidelity_design,
)
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    resolve_repo_path,
    validate_json,
)
from .seeds import SeedPlanError, derive_plan_seed, load_seed_plan
from .tool_contract import spathi_runtime_resources

SEARCH_SPACE_PATH = Path("experiments/v1/spathi/search-space.json")
SEARCH_SPACE_SCHEMA_PATH = Path("schemas/spathi-search-space.schema.json")
RANKING_SCHEMA_PATH = Path("schemas/calibration-ranking.schema.json")
MANIFEST_SCHEMA_PATH = Path("schemas/calibration-manifest.schema.json")
SELECTED_CONFIG_SCHEMA_PATH = Path("schemas/spathi-selected-config.schema.json")
SEED_PLAN_PATH = Path("experiments/v1/seeds/seed-plan.json")

_CALIBRATION_BUDGETS = frozenset({"reduced", "full"})
_SEARCH_STAGES = (
    "geometry",
    "estimator",
    "interaction",
    "estimator_scale_transfer",
    "convergence",
)
_CALIBRATION_STAGES = frozenset((*_SEARCH_STAGES, "validation"))
_PARENT_PHASES = {
    ("geometry", "scout"): None,
    ("geometry", "confirmation"): ("geometry", "scout"),
    ("estimator", "scout"): ("geometry", "confirmation"),
    ("estimator", "confirmation"): ("estimator", "scout"),
    ("interaction", "scout"): ("estimator", "confirmation"),
    ("interaction", "confirmation"): ("interaction", "scout"),
    ("estimator_scale_transfer", "scout"): ("interaction", "confirmation"),
    ("estimator_scale_transfer", "confirmation"): (
        "estimator_scale_transfer",
        "scout",
    ),
    ("convergence", "final"): ("estimator_scale_transfer", "confirmation"),
    ("validation", "final"): ("convergence", "final"),
}
_LOGICAL_PARENT_STAGES = {
    "geometry": None,
    "estimator": "geometry",
    "interaction": "estimator",
    "estimator_scale_transfer": "interaction",
    "convergence": "estimator_scale_transfer",
}
_LOGICAL_CHILD_STAGES = {
    parent: child for child, parent in _LOGICAL_PARENT_STAGES.items() if parent is not None
}
_FIXED_CALIBRATION_PARAMETERS: Mapping[str, Any] = {
    "bandwidth": "auto",
}
_GEOMETRY_DEVELOPMENT_SELECTOR: Mapping[str, Any] = {
    "split_id": "development_v1",
    "condition_ids": [
        "linear_baseline",
        "linear_noisy",
        "bifurcating_baseline",
        "bifurcating_noisy",
        "scmultisim_dynamic",
        "scmultisim_imbalanced",
        "scmultisim_single_group",
    ],
}
_MULTISCALE_DEVELOPMENT_SELECTOR: Mapping[str, Any] = {
    "split_id": "development_v1",
    "condition_ids": [
        "linear_baseline",
        "linear_noisy",
        "bifurcating_baseline",
        "bifurcating_noisy",
        "scmultisim_dynamic",
        "scmultisim_imbalanced",
        "scmultisim_single_group",
        "scmultisim_medium",
    ],
}
_SCALE_TRANSFER_SELECTOR: Mapping[str, Any] = {
    "split_id": "development_estimator_scale_transfer_v1",
    "condition_ids": ["scmultisim_estimator_scale_transfer"],
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
    base_parameter_changes: int


@dataclass(frozen=True, slots=True)
class CalibrationRun:
    """One candidate/algorithmic-seed execution requested by a search stage."""

    run_id: str
    configuration_id: str
    configuration_sha256: str
    stage_id: str
    fidelity: str
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
        "base_parameter_changes": candidate.base_parameter_changes,
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
            base_parameter_changes=_integer(
                payload["base_parameter_changes"],
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
        if simplicity != candidate.base_parameter_changes:
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
                minimum=1,
            ),
        )
    except KeyError as exc:
        raise CalibrationError(f"Calibration score is missing {exc.args[0]!r}") from exc


def _score_sort_key(score: CandidateScore) -> tuple[float, float, int, str]:
    return (
        -score.hierarchical_aupr,
        score.stability,
        score.simplicity,
        score.candidate.configuration_id,
    )


def _rank_convergence_scores(
    stage: Mapping[str, Any],
    scores: Sequence[CandidateScore],
) -> list[CandidateScore]:
    """Put one smallest non-inferior stable ceiling per parent first.

    The largest requested forest within each inherited estimator policy is the
    preregistered numerical reference.  This prevents small Monte Carlo AUPR
    fluctuations from turning the convergence stage into a maximize-tree-count
    search while still preserving a complete deterministic ranking artifact.
    """

    rule = _mapping(stage.get("selection_rule"), "convergence selection rule")
    if rule.get("method") != "smallest_noninferior_stable_per_parent":
        raise CalibrationError("Convergence stage has an unsupported selection rule")
    aupr_margin = _finite_number(
        rule.get("absolute_aupr_margin"),
        "convergence absolute AUPR margin",
        minimum=0,
        maximum=1,
    )
    stability_increase = _finite_number(
        rule.get("maximum_stability_increase"),
        "convergence maximum stability increase",
        minimum=0,
        maximum=1,
    )
    by_parent: dict[str, list[CandidateScore]] = defaultdict(list)
    for score in scores:
        parent = score.candidate.parent_configuration_id
        if parent is None:
            raise CalibrationError("Convergence candidates must identify an inherited parent")
        by_parent[parent].append(score)

    references: dict[str, CandidateScore] = {}
    eligible: set[str] = set()
    representatives: list[CandidateScore] = []
    for parent, parent_scores in sorted(by_parent.items()):
        reference = min(
            parent_scores,
            key=lambda item: (
                -int(item.candidate.parameters["n_estimators"]),
                *_score_sort_key(item),
            ),
        )
        references[parent] = reference
        parent_eligible = [
            item
            for item in parent_scores
            if item.hierarchical_aupr >= reference.hierarchical_aupr - aupr_margin
            and item.stability <= reference.stability + stability_increase
        ]
        if not parent_eligible:
            raise CalibrationError(
                f"Convergence parent {parent!r} has no candidate satisfying its frozen rule"
            )
        eligible.update(item.candidate.configuration_id for item in parent_eligible)
        representatives.append(
            min(
                parent_eligible,
                key=lambda item: (
                    int(item.candidate.parameters["n_estimators"]),
                    *_score_sort_key(item),
                ),
            )
        )

    representatives.sort(key=_score_sort_key)
    representative_ids = {item.candidate.configuration_id for item in representatives}
    remaining = sorted(
        (item for item in scores if item.candidate.configuration_id not in representative_ids),
        key=lambda item: (
            item.candidate.configuration_id not in eligible,
            -references[str(item.candidate.parent_configuration_id)].hierarchical_aupr,
            int(item.candidate.parameters["n_estimators"]),
            *_score_sort_key(item),
        ),
    )
    return [*representatives, *remaining]


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
    _validate_seed_contract(root, search_space)
    validate_multifidelity_design(search_space)
    return search_space


def materialize_group_policy(
    search_space: Mapping[str, Any],
    multi_group_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete deterministic one/many-group policy for a candidate."""

    base = _mapping(search_space.get("base_configuration"), "base_configuration")
    parameters = copy.deepcopy(dict(multi_group_parameters))
    if set(parameters) != set(base):
        raise CalibrationError("Policy parameters must contain the complete SPATHI parameter set")
    template = _mapping(
        search_space.get("conditional_group_policy"),
        "conditional_group_policy",
    )
    branches = _mapping(template.get("branches"), "conditional group policy branches")
    single_template = _mapping(branches.get("single_group"), "single-group policy branch")
    multi_template = _mapping(branches.get("multi_group"), "multi-group policy branch")
    overrides = _mapping(
        single_template.get("parameter_overrides"),
        "single-group parameter overrides",
    )
    single_parameters = {**parameters, **copy.deepcopy(dict(overrides))}
    shared = tuple(
        str(item)
        for item in _sequence(template.get("shared_parameters"), "shared policy parameters")
    )
    if set(shared) != set(parameters) - set(overrides):
        raise CalibrationError(
            "Conditional policy shared_parameters must be every parameter not overridden "
            "by the single-group branch"
        )
    if any(single_parameters[key] != parameters[key] for key in shared):
        raise CalibrationError("Conditional group policy changed a shared parameter")
    return {
        "routing_parameter": "group_count",
        "shared_parameters": list(shared),
        "branches": {
            "single_group": {
                "condition": copy.deepcopy(single_template["condition"]),
                "parameters": single_parameters,
            },
            "multi_group": {
                "condition": copy.deepcopy(multi_template["condition"]),
                "parameters": parameters,
            },
        },
    }


def resolve_group_policy(policy: Mapping[str, Any], group_count: int) -> dict[str, Any]:
    """Resolve a frozen policy to plain SPATHI parameters for one dataset."""

    if type(group_count) is not int or group_count < 1:
        raise CalibrationError("group_count must be a positive integer")
    branches = _mapping(policy.get("branches"), "group policy branches")
    branch_id = "single_group" if group_count == 1 else "multi_group"
    branch = _mapping(branches.get(branch_id), f"{branch_id} policy branch")
    condition = _mapping(branch.get("condition"), f"{branch_id} policy condition")
    minimum = _integer(
        condition.get("minimum_group_count"),
        f"{branch_id} minimum_group_count",
        minimum=1,
    )
    maximum_value = condition.get("maximum_group_count")
    maximum = (
        None
        if maximum_value is None
        else _integer(maximum_value, f"{branch_id} maximum_group_count", minimum=1)
    )
    if group_count < minimum or (maximum is not None and group_count > maximum):
        raise CalibrationError(f"Policy branch {branch_id!r} does not cover {group_count} groups")
    return copy.deepcopy(dict(_mapping(branch.get("parameters"), f"{branch_id} parameters")))


def stage_profile_id(stage_id: str, calibration_budget: str, fidelity: str) -> str:
    """Return the canonical runnable profile ID for one stage and evidence budget."""

    _validate_calibration_stage(stage_id)
    budget = _validate_calibration_budget_id(calibration_budget)
    if stage_id in MULTIFIDELITY_STAGES:
        if fidelity not in FIDELITIES:
            raise CalibrationError(f"Stage {stage_id!r} requires scout or confirmation fidelity")
    elif fidelity != "final":
        raise CalibrationError(f"Stage {stage_id!r} requires final fidelity")
    purpose = "selection" if stage_id == "validation" else "calibration"
    return f"{purpose}-{budget}-{stage_id}-{fidelity}"


def canonical_run_records(runs: Sequence[CalibrationRun]) -> list[dict[str, Any]]:
    """Serialize the complete run identity used to reconcile summary evidence."""

    records = [
        {
            "run_id": run.run_id,
            "configuration_id": run.configuration_id,
            "configuration_sha256": run.configuration_sha256,
            "fidelity": run.fidelity,
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
    fidelity: str,
    *,
    bundle_path: str,
    search_space: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize one stage-specific runnable profile from its planned template."""

    budget = _validate_calibration_budget_id(calibration_budget)
    if search_space is None:
        raise CalibrationError(f"Stage {stage_id!r} requires its frozen phase design")
    phase = calibration_phase(search_space, budget, stage_id, fidelity)
    profile_id = stage_profile_id(stage_id, budget, fidelity)
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
    if stage_id != "validation":
        stage = _stage(search_space, stage_id)
        selector = _mapping(stage.get("dataset_selector"), f"{stage_id} dataset selector")
        split_id = selector.get("split_id")
        condition_ids = list(phase.condition_ids)
        if (
            not isinstance(split_id, str)
            or not split_id
            or not isinstance(condition_ids, list)
            or not condition_ids
            or not all(isinstance(item, str) and item for item in condition_ids)
        ):
            raise CalibrationError(f"Stage {stage_id!r} has an invalid dataset selector")
        allowed_conditions = set(condition_ids)
        raw_jobs = [
            job
            for job in raw_jobs
            if isinstance(job, Mapping)
            and job.get("split_id") == split_id
            and job.get("condition_id") in allowed_conditions
        ]
        observed_conditions = {
            str(job.get("condition_id")) for job in raw_jobs if isinstance(job, Mapping)
        }
        if observed_conditions != allowed_conditions:
            raise CalibrationError(
                f"Stage {stage_id!r} dataset selector expected conditions "
                f"{sorted(allowed_conditions)}; found {sorted(observed_conditions)}"
            )
    observed_roles = {job.get("dataset_role") for job in raw_jobs if isinstance(job, Mapping)}
    if observed_roles != {expected_role}:
        raise CalibrationError(
            f"Stage {stage_id!r} requires only {expected_role!r} generation jobs; "
            f"found {sorted(str(role) for role in observed_roles)}"
        )
    if not isinstance(bundle_path, str) or not bundle_path:
        raise CalibrationError("Calibration profile requires a repository-relative bundle path")

    profile = copy.deepcopy(dict(template))
    profile["generation_jobs"] = copy.deepcopy(raw_jobs)
    profile.update(
        {
            "id": profile_id,
            "status": "runnable",
            "calibration_stage": stage_id,
            "calibration_fidelity": fidelity,
            "calibration_budget": budget,
            "description": (
                f"{str(template.get('description', '')).strip()} "
                f"Materialized immutable {stage_id} {fidelity} phase."
            ).strip(),
            "inference_batches": [
                {
                    "id": f"spathi-{stage_id}",
                    "kind": "spathi",
                    "dataset_role": expected_role,
                    "generation_jobs": [str(job["id"]) for job in raw_jobs],
                    "regime": "group_only",
                    "tools_params": f"{bundle_path}/tools-params.json",
                    "custom_tools": f"{bundle_path}/custom-tools.json",
                    "calibration_manifest": f"{bundle_path}/calibration-manifest.json",
                    "run_metadata": f"{bundle_path}/run-metadata.json",
                }
            ],
        }
    )
    if stage_id != "validation":
        profile["inference_batches"][0]["replicate_indices"] = {
            str(job["id"]): list(phase.replicate_indices) for job in raw_jobs
        }
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
    fidelity: str,
    ranking_path: Path | None,
) -> tuple[tuple[CandidateConfiguration, ...], dict[str, Any] | None]:
    """Validate a preceding ranking and return its frozen shortlist and link."""

    budget = _validate_calibration_budget(search_space, calibration_budget)
    expected_phase = _PARENT_PHASES.get((stage_id, fidelity))
    if (stage_id, fidelity) not in _PARENT_PHASES:
        raise CalibrationError(f"Unknown calibration phase: {stage_id}.{fidelity}")
    if expected_phase is None:
        if ranking_path is not None:
            raise CalibrationError("The geometry scout cannot receive --parents")
        return (), None
    expected_stage, expected_fidelity = expected_phase
    if ranking_path is None:
        raise CalibrationError(
            f"Phase {stage_id}.{fidelity} requires --parents from "
            f"{expected_stage}.{expected_fidelity}"
        )

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
    if ranking.get("stage_id") != expected_stage or ranking.get("fidelity") != expected_fidelity:
        raise CalibrationError(
            f"Phase {stage_id}.{fidelity} requires a {expected_stage}.{expected_fidelity} ranking"
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
        "fidelity": expected_fidelity,
        "calibration_budget": budget,
    }


def expand_stage_candidates(
    search_space: Mapping[str, Any],
    stage_id: str,
    *,
    parents: Sequence[CandidateConfiguration] = (),
) -> tuple[CandidateConfiguration, ...]:
    """Expand one stage into a deterministic, bounded set of configurations.

    ``geometry`` starts from ``base_configuration`` and takes no parents.
    Later stages require the selected/shortlisted candidates from the preceding
    stage. Parent input order is ignored so repeated expansion is stable.
    """

    stage = _stage(search_space, stage_id)
    base_configuration = _mapping(
        search_space.get("base_configuration"),
        "base_configuration",
    )
    parent_items = _validated_parents(stage, parents)
    _validate_candidate_integrity(
        tuple(parent for parent in parent_items if parent is not None),
        base_configuration,
    )
    _validate_parent_expansion_identities(
        stage,
        tuple(parent for parent in parent_items if parent is not None),
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
        base = dict(base_configuration if parent is None else parent.parameters)
        prefix = (
            stage_id if parent is None else f"{stage_id}__from_{parent.configuration_sha256[:12]}"
        )
        inherited_variant = "base" if parent is None else "inherited"
        if stage["include_inherited_configuration"]:
            candidates.append(
                _candidate(
                    stage_id=stage_id,
                    role=str(stage["dataset_role"]),
                    configuration_id=f"{prefix}__{inherited_variant}",
                    parent=parent,
                    variant_id=inherited_variant,
                    parameters=base,
                    base_configuration=base_configuration,
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
                    base_configuration=base_configuration,
                )
            )

    identifiers = [candidate.configuration_id for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise CalibrationError(f"Stage {stage_id!r} generated duplicate configuration IDs")
    by_configuration: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        by_configuration[candidate.configuration_sha256].append(candidate.configuration_id)
    collisions = [identifiers for identifiers in by_configuration.values() if len(identifiers) > 1]
    if collisions:
        rendered = "; ".join(", ".join(items) for items in collisions)
        raise CalibrationError(
            f"Stage {stage_id!r} generated duplicate effective configurations: {rendered}. "
            "The preregistered candidate cardinality cannot be changed after ranking."
        )
    return tuple(candidates)


def expand_stage_runs(
    search_space: Mapping[str, Any],
    stage_id: str,
    calibration_budget: str,
    fidelity: str,
    *,
    parents: Sequence[CandidateConfiguration] = (),
    candidates: Sequence[CandidateConfiguration] | None = None,
) -> tuple[CalibrationRun, ...]:
    """Expand one scout or confirmation phase over its frozen seed subset."""

    _stage(search_space, stage_id)
    phase = calibration_phase(search_space, calibration_budget, stage_id, fidelity)
    all_seeds = _algorithmic_seeds(search_space, calibration_budget, stage_id)
    seeds = tuple(all_seeds[index - 1] for index in phase.algorithm_seed_indices)
    if fidelity == "scout" or stage_id == "convergence":
        if candidates is not None:
            raise CalibrationError("Scout candidates must use the canonical stage expansion")
        selected_candidates = expand_stage_candidates(search_space, stage_id, parents=parents)
    else:
        if candidates is None:
            raise CalibrationError("Confirmation requires its frozen scout shortlist")
        selected_candidates = tuple(candidates)
        if len(selected_candidates) != phase.candidate_count:
            raise CalibrationError(
                f"{stage_id} confirmation requires {phase.candidate_count} candidates"
            )
        base = _mapping(search_space.get("base_configuration"), "base_configuration")
        _validate_candidate_integrity(selected_candidates, base)
        if any(candidate.stage_id != stage_id for candidate in selected_candidates):
            raise CalibrationError("Confirmation candidates must come from the same stage scout")
    expected_candidate_count = phase.candidate_count
    if stage_id == "convergence":
        expected_candidate_count *= 4
    if len(selected_candidates) != expected_candidate_count:
        raise CalibrationError(
            f"{stage_id}.{fidelity} requires exactly {expected_candidate_count} distinct "
            f"logical candidates; received {len(selected_candidates)}"
        )
    return tuple(
        CalibrationRun(
            run_id=f"{candidate.configuration_id}__{fidelity}__seed_{seed}",
            configuration_id=candidate.configuration_id,
            configuration_sha256=candidate.configuration_sha256,
            stage_id=stage_id,
            fidelity=fidelity,
            dataset_role=candidate.dataset_role,
            algorithm_seed=seed,
            parameters={
                "group_policy": materialize_group_policy(search_space, candidate.parameters),
                "random_seed": seed,
                "target_scope": copy.deepcopy(dict(phase.target_scope)),
            },
        )
        for candidate in selected_candidates
        for seed in seeds
    )


def expand_validation_runs(
    search_space: Mapping[str, Any],
    finalists: Sequence[CandidateConfiguration],
    calibration_budget: str,
) -> tuple[CalibrationRun, ...]:
    """Expand the bounded convergence finalists on validation-only algorithm seeds."""

    protocol = _mapping(search_space.get("selection_protocol"), "selection_protocol")
    maximum = _integer(
        protocol.get("maximum_validation_finalists"),
        "maximum validation finalists",
        minimum=1,
    )
    if len(finalists) != maximum:
        raise CalibrationError(
            f"Validation requires exactly {maximum} convergence finalists; "
            f"received {len(finalists)}"
        )
    if any(candidate.stage_id != "convergence" for candidate in finalists):
        raise CalibrationError("Validation finalists must come from the convergence stage")
    base_configuration = _mapping(
        search_space.get("base_configuration"),
        "base_configuration",
    )
    _validate_candidate_integrity(finalists, base_configuration)
    seeds = _algorithmic_seeds(search_space, calibration_budget, "validation")
    return tuple(
        CalibrationRun(
            run_id=f"validation__{candidate.configuration_id}__final__seed_{seed}",
            configuration_id=candidate.configuration_id,
            configuration_sha256=candidate.configuration_sha256,
            stage_id="validation",
            fidelity="final",
            dataset_role="validation",
            algorithm_seed=seed,
            parameters={
                "group_policy": materialize_group_policy(search_space, candidate.parameters),
                "random_seed": seed,
                "target_scope": {"method": "all"},
            },
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
    fidelity = str(manifest.get("fidelity"))
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
    if manifest.get("profile_id") != stage_profile_id(stage_id, calibration_budget, fidelity):
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
        fidelity,
        parent_path,
    )
    if raw_parent != expected_parent:
        raise CalibrationError("Calibration manifest parent-ranking link is stale")

    if stage_id == "validation":
        candidates = parents
        runs = expand_validation_runs(search_space, candidates, calibration_budget)
        expected_role = "validation"
    elif fidelity == "scout" or stage_id == "convergence":
        candidates = expand_stage_candidates(search_space, stage_id, parents=parents)
        runs = expand_stage_runs(
            search_space,
            stage_id,
            calibration_budget,
            fidelity,
            parents=parents,
        )
        expected_role = "development"
    else:
        candidates = parents
        runs = expand_stage_runs(
            search_space,
            stage_id,
            calibration_budget,
            fidelity,
            candidates=candidates,
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
    calibration_panel: object,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Reconcile every dataset/run/context cell with the frozen stage manifest."""

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
    panel = validate_calibration_panel(
        calibration_panel,
        rows,
        expected_role=str(manifest.get("dataset_role")),
        expected_run_ids=tuple(sorted(expected)),
    )
    return rows, panel


def validate_calibration_panel(
    calibration_panel: object,
    summary_rows: Iterable[Mapping[str, Any]],
    *,
    expected_role: str,
    expected_run_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate a closed Cartesian dataset/run panel and its context inventories."""

    panel = _mapping(calibration_panel, "calibration panel")
    if panel.get("schema_version") != "1.0" or panel.get("dataset_role") != expected_role:
        raise CalibrationError("Calibration panel identity differs from its stage")
    dataset_ids = panel.get("dataset_ids")
    run_ids = panel.get("run_ids")
    if (
        not isinstance(dataset_ids, list)
        or not dataset_ids
        or not all(isinstance(item, str) and item for item in dataset_ids)
        or dataset_ids != sorted(set(dataset_ids))
    ):
        raise CalibrationError("Calibration panel dataset_ids must be sorted and unique")
    if (
        not isinstance(run_ids, list)
        or not run_ids
        or not all(isinstance(item, str) and item for item in run_ids)
        or run_ids != sorted(set(run_ids))
    ):
        raise CalibrationError("Calibration panel run_ids must be sorted and unique")
    if expected_run_ids is not None and tuple(run_ids) != expected_run_ids:
        raise CalibrationError("Calibration panel run IDs differ from the stage manifest")
    if panel.get("n_expected_executions") != len(dataset_ids) * len(run_ids):
        raise CalibrationError("Calibration panel has an invalid execution count")
    handoff_digest = panel.get("handoff_validation_sha256")
    if not isinstance(handoff_digest, str) or re.fullmatch(r"[0-9a-f]{64}", handoff_digest) is None:
        raise CalibrationError("Calibration panel has an invalid handoff digest")

    raw_datasets = panel.get("datasets")
    if not isinstance(raw_datasets, list) or len(raw_datasets) != len(dataset_ids):
        raise CalibrationError("Calibration panel dataset contracts are incomplete")
    contexts_by_dataset: dict[str, tuple[str, ...]] = {}
    required_dataset_fields = {
        "dataset_id",
        "expected_group_contexts",
        "n_group_contexts",
        "dataset_manifest_sha256",
        "group_assignments_sha256",
        "ground_truth_manifest_sha256",
        "truth_network_sha256",
    }
    for index, raw_dataset in enumerate(raw_datasets):
        dataset = _mapping(raw_dataset, f"calibration panel dataset {index}")
        if set(dataset) != required_dataset_fields:
            raise CalibrationError(
                f"Calibration panel dataset {index} has an invalid evidence shape"
            )
        dataset_id = dataset.get("dataset_id")
        raw_contexts = dataset.get("expected_group_contexts")
        if (
            not isinstance(dataset_id, str)
            or dataset_id not in dataset_ids
            or dataset_id in contexts_by_dataset
            or not isinstance(raw_contexts, list)
            or not raw_contexts
            or not all(
                isinstance(context, str) and context.startswith("group:") and context != "group:"
                for context in raw_contexts
            )
            or raw_contexts != sorted(set(raw_contexts))
            or dataset.get("n_group_contexts") != len(raw_contexts)
        ):
            raise CalibrationError(
                f"Calibration panel dataset {index} has an invalid context inventory"
            )
        for field in required_dataset_fields - {
            "dataset_id",
            "expected_group_contexts",
            "n_group_contexts",
        }:
            digest = dataset.get(field)
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise CalibrationError(f"Calibration panel dataset {index} has an invalid {field}")
        contexts_by_dataset[dataset_id] = tuple(raw_contexts)
    if set(contexts_by_dataset) != set(dataset_ids):
        raise CalibrationError("Calibration panel omits dataset context contracts")

    rows = list(summary_rows)
    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"summary row {index}")
        key = (str(row.get("dataset_id")), str(row.get("run_id")))
        if key in observed:
            raise CalibrationError(f"Calibration summary duplicates dataset/run execution {key!r}")
        if key[0] not in contexts_by_dataset or key[1] not in run_ids:
            raise CalibrationError(f"Calibration summary contains an unplanned execution {key!r}")
        if row.get("dataset_role") != expected_role:
            raise CalibrationError(f"Calibration summary row {index} has the wrong dataset role")
        if tuple(row.get("group_contexts", ())) != contexts_by_dataset[key[0]]:
            raise CalibrationError(
                f"Calibration summary row {index} does not cover its declared group contexts"
            )
        if row.get("n_group_contexts") != len(contexts_by_dataset[key[0]]):
            raise CalibrationError(
                f"Calibration summary row {index} has the wrong group-context count"
            )
        observed[key] = row
    expected = {(dataset_id, run_id) for dataset_id in dataset_ids for run_id in run_ids}
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        raise CalibrationError(f"Calibration summary omits planned executions: {missing}")
    return copy.deepcopy(dict(panel))


def build_andrea_requests(
    runs: Sequence[CalibrationRun],
    *,
    docker_image: str,
    calibration_budget: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Render calibration runs as ANDREA tools, custom-tool and metadata requests."""

    if not runs:
        raise CalibrationError("Cannot build ANDREA requests for an empty calibration stage")
    if not docker_image:
        raise CalibrationError("Calibration requests require a SPATHI Docker image")
    run_ids = [run.run_id for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise CalibrationError("Calibration runs contain duplicate run IDs")
    stage_ids = {run.stage_id for run in runs}
    fidelities = {run.fidelity for run in runs}
    dataset_roles = {run.dataset_role for run in runs}
    if len(stage_ids) != 1 or len(fidelities) != 1 or len(dataset_roles) != 1:
        raise CalibrationError("One calibration request bundle must use one phase and role")
    stage_id = next(iter(stage_ids))
    fidelity = next(iter(fidelities))
    dataset_role = next(iter(dataset_roles))
    seed_order = tuple(dict.fromkeys(run.algorithm_seed for run in runs))
    seed_indexes = {seed: index for index, seed in enumerate(seed_order, start=1)}
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
                "runtime_resources": spathi_runtime_resources(),
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
                "configuration_sha256": run.configuration_sha256,
                "selection_role": "base"
                if run.configuration_id.endswith("__base")
                else "candidate",
                "algorithm_seed": run.algorithm_seed,
                "algorithm_seed_parameter": "random_seed",
                "family": "spathi",
                "calibration_fidelity": fidelity,
                "target_scope": copy.deepcopy(dict(run.parameters["target_scope"])),
                "seed_derivation": {
                    "namespace": f"{calibration_budget}.{dataset_role}",
                    "stream": "inference",
                    "coordinates": {
                        "phase": "calibration",
                        "stage": stage_id,
                        "fidelity": fidelity,
                        "replicate": seed_indexes[run.algorithm_seed],
                    },
                },
            }
            for run in runs
        ],
    }
    return tools_params, custom_tools, metadata


def rank_stage_candidates(
    search_space: Mapping[str, Any],
    stage_id: str,
    candidates: Sequence[CandidateConfiguration],
    summary_rows: Iterable[Mapping[str, Any]],
    *,
    calibration_budget: str,
    fidelity: str,
) -> tuple[CandidateScore, ...]:
    """Return the complete ordered ranking for one development stage."""

    stage = _stage(search_space, stage_id)
    base_configuration = _mapping(
        search_space.get("base_configuration"),
        "base_configuration",
    )
    _validate_candidate_integrity(candidates, base_configuration)
    if fidelity == "scout" or stage_id == "convergence":
        _validate_stage_candidate_set(stage, candidates)
    else:
        if any(candidate.stage_id != stage_id for candidate in candidates):
            raise CalibrationError("Confirmation ranking contains candidates from another stage")
    phase = calibration_phase(search_space, calibration_budget, stage_id, fidelity)
    all_seeds = _algorithmic_seeds(search_space, calibration_budget, stage_id)
    expected_seeds = tuple(all_seeds[index - 1] for index in phase.algorithm_seed_indices)
    scores = _rank_balanced_rows(
        candidates,
        summary_rows,
        expected_role="development",
        expected_seeds=expected_seeds,
        expected_fidelity=fidelity,
        expected_dataset_count=phase.dataset_count,
    )
    if stage_id == "convergence":
        scores = _rank_convergence_scores(stage, scores)
    return tuple(scores)


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
    if len(finalists) != maximum:
        raise CalibrationError(
            f"Validation requires exactly {maximum} predeclared finalists; "
            f"received {len(finalists)}"
        )
    if any(candidate.stage_id != "convergence" for candidate in finalists):
        raise CalibrationError("Validation finalists must come from the convergence stage")
    base_configuration = _mapping(
        search_space.get("base_configuration"),
        "base_configuration",
    )
    _validate_candidate_integrity(finalists, base_configuration)
    expected_seeds = _algorithmic_seeds(search_space, calibration_budget, "validation")
    return tuple(
        _rank_balanced_rows(
            finalists,
            summary_rows,
            expected_role="validation",
            expected_seeds=expected_seeds,
            expected_fidelity="final",
            expected_dataset_count=calibration_phase(
                search_space, calibration_budget, "validation", "final"
            ).dataset_count,
        )
    )


def build_calibration_ranking(
    search_space: Mapping[str, Any],
    stage_id: str,
    fidelity: str,
    candidates: Sequence[CandidateConfiguration],
    summary_rows: Iterable[Mapping[str, Any]],
    *,
    calibration_panel: Mapping[str, Any],
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
    if profile_id != stage_profile_id(stage_id, budget, fidelity):
        raise CalibrationError("Calibration ranking has the wrong stage profile ID")
    if _GIT_COMMIT.fullmatch(benchmark_commit) is None:
        raise CalibrationError("Calibration ranking requires a valid benchmark commit")
    expected_parent_phase = _PARENT_PHASES[(stage_id, fidelity)]
    if expected_parent_phase is None:
        if parent_ranking is not None:
            raise CalibrationError("Geometry ranking cannot link a parent ranking")
    else:
        expected_parent_stage, expected_parent_fidelity = expected_parent_phase
        parent = _mapping(parent_ranking, "ranking parent link")
        if (
            parent.get("stage_id") != expected_parent_stage
            or parent.get("fidelity") != expected_parent_fidelity
        ):
            raise CalibrationError("Calibration ranking links the wrong parent stage")
        if parent.get("calibration_budget") != budget:
            raise CalibrationError("Calibration ranking links a different calibration budget")
    raw_evidence_rows = list(summary_rows)
    normalized_panel = validate_calibration_panel(
        calibration_panel,
        raw_evidence_rows,
        expected_role="validation" if stage_id == "validation" else "development",
    )
    evidence_rows = _normalized_evidence_rows(raw_evidence_rows)
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
            fidelity=fidelity,
        )
        limit = calibration_phase(search_space, budget, stage_id, fidelity).advance_limit
        selected_scores = _select_stage_shortlist(
            search_space,
            str(stage_id),
            str(fidelity),
            scores,
            limit,
        )
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
        "fidelity": fidelity,
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
        "calibration_panel": normalized_panel,
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
    fidelity = ranking.get("fidelity")
    role = ranking.get("dataset_role")
    if ranking.get("profile_id") != stage_profile_id(
        str(stage_id), calibration_budget, str(fidelity)
    ):
        raise CalibrationError("Calibration ranking has the wrong stage profile ID")
    if _GIT_COMMIT.fullmatch(str(ranking.get("benchmark_commit", ""))) is None:
        raise CalibrationError("Calibration ranking has an invalid benchmark commit")
    expected_parent_phase = _PARENT_PHASES.get((str(stage_id), str(fidelity)))
    if (str(stage_id), str(fidelity)) not in _PARENT_PHASES:
        raise CalibrationError("Calibration ranking identifies an unknown phase")
    parent_ranking = ranking.get("parent_ranking")
    if expected_parent_phase is None:
        if parent_ranking is not None:
            raise CalibrationError("Geometry ranking cannot link a parent ranking")
    else:
        expected_parent_stage, expected_parent_fidelity = expected_parent_phase
        parent = _mapping(parent_ranking, "ranking parent link")
        if (
            parent.get("stage_id") != expected_parent_stage
            or parent.get("fidelity") != expected_parent_fidelity
        ):
            raise CalibrationError("Calibration ranking links the wrong parent stage")
        if parent.get("calibration_budget") != calibration_budget:
            raise CalibrationError("Calibration ranking links a different calibration budget")
    if stage_id == "validation":
        if role != "validation":
            raise CalibrationError("Validation ranking has the wrong dataset role")
    elif stage_id in set(_SEARCH_STAGES):
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
    base_configuration = _mapping(
        search_space.get("base_configuration"),
        "base_configuration",
    )
    candidates = tuple(score.candidate for score in serialized_scores)
    _validate_candidate_integrity(candidates, base_configuration)
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
    validate_calibration_panel(
        ranking.get("calibration_panel"),
        raw_rows,
        expected_role=str(role),
    )
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
        if len(serialized_scores) != maximum:
            raise CalibrationError(f"Validation ranking requires exactly {maximum} finalists")
        if any(candidate.stage_id != "convergence" for candidate in candidates):
            raise CalibrationError("Validation ranking contains a non-convergence candidate")
        recomputed_scores = rank_validation_candidates(
            search_space,
            candidates,
            raw_rows,
            calibration_budget=calibration_budget,
        )
        selected_scores = recomputed_scores[:1]
    else:
        stage = _stage(search_space, str(stage_id))
        if fidelity == "scout" or stage_id == "convergence":
            _validate_stage_candidate_set(stage, candidates)
        recomputed_scores = rank_stage_candidates(
            search_space,
            str(stage_id),
            candidates,
            raw_rows,
            calibration_budget=calibration_budget,
            fidelity=str(fidelity),
        )
        selected_scores = _select_stage_shortlist(
            search_space,
            str(stage_id),
            str(fidelity),
            recomputed_scores,
            calibration_phase(
                search_space, calibration_budget, str(stage_id), str(fidelity)
            ).advance_limit,
        )

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


def _validate_canonical_parent_expansion(
    search_space: Mapping[str, Any],
    stage_id: str,
    selected_parents: Sequence[CandidateConfiguration],
    current_candidates: Sequence[CandidateConfiguration],
) -> None:
    expected_candidates = expand_stage_candidates(
        search_space,
        stage_id,
        parents=selected_parents,
    )
    expected_by_id = {
        candidate.configuration_id: candidate_to_dict(candidate)
        for candidate in expected_candidates
    }
    observed_by_id = {
        candidate.configuration_id: candidate_to_dict(candidate) for candidate in current_candidates
    }
    if observed_by_id != expected_by_id:
        raise CalibrationError(
            f"{stage_id.capitalize()} ranking candidates differ from the canonical parent expansion"
        )


def validate_ranking_chain(
    repo_root: Path,
    search_space: Mapping[str, Any],
    ranking: Mapping[str, Any],
    *,
    expected_search_space_sha256: str,
    expected_calibration_budget: str | None = None,
    _visited: frozenset[str] = frozenset(),
) -> tuple[CandidateScore, ...]:
    """Validate a ranking and every content-addressed predecessor back to geometry."""

    root = repo_root.resolve()
    scores = validate_ranking_artifact(
        search_space,
        ranking,
        expected_search_space_sha256=expected_search_space_sha256,
        expected_calibration_budget=expected_calibration_budget,
    )
    stage_id = str(ranking["stage_id"])
    fidelity = str(ranking["fidelity"])
    calibration_budget = str(ranking["calibration_budget"])
    expected_parent_phase = _PARENT_PHASES[(stage_id, fidelity)]
    parent_value = ranking.get("parent_ranking")
    if expected_parent_phase is None:
        if parent_value is not None:
            raise CalibrationError("Geometry ranking has an unexpected parent link")
        return scores
    expected_parent_stage, expected_parent_fidelity = expected_parent_phase
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
    if (
        parent_ranking.get("stage_id") != expected_parent_stage
        or parent_ranking.get("fidelity") != expected_parent_fidelity
    ):
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
    if stage_id == "validation" or fidelity == "confirmation":
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
                f"{stage_id}.{fidelity} candidates differ from its parent shortlist"
            )
    else:
        _validate_canonical_parent_expansion(
            search_space,
            stage_id,
            selected_parents,
            current_candidates,
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
    group_policy = materialize_group_policy(search, winner.candidate.parameters)
    policy_sha256 = canonical_json_sha256(group_policy)
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
        "winner_policy_sha256": policy_sha256,
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
        "selected_policy": {
            "policy_id": "spathi_conditional_v1",
            "source_candidate_id": winner.candidate.configuration_id,
            "spathi_commit": ranking["spathi_commit"],
            "source_configuration_sha256": winner.candidate.configuration_sha256,
            "policy_sha256": policy_sha256,
            "group_policy": group_policy,
        },
        "selection_evidence": evidence,
        "test_accessed_before_freeze": False,
        "notes": [
            "Generated from the cryptographically linked validation ranking.",
            "Test results cannot change this frozen conditional group policy.",
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
    group_policy = materialize_group_policy(search_space, winner.candidate.parameters)
    policy_sha256 = canonical_json_sha256(group_policy)
    selected_policy = _mapping(selected.get("selected_policy"), "selected policy")
    expected_policy = {
        "policy_id": "spathi_conditional_v1",
        "source_candidate_id": winner.candidate.configuration_id,
        "spathi_commit": ranking["spathi_commit"],
        "source_configuration_sha256": winner.candidate.configuration_sha256,
        "policy_sha256": policy_sha256,
        "group_policy": group_policy,
    }
    if dict(selected_policy) != expected_policy:
        raise CalibrationError("Frozen selected policy is not reconciled with validation ranking")
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
        "winner_policy_sha256": policy_sha256,
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
    if seen != set(_SEARCH_STAGES):
        raise CalibrationError(f"Calibration search stages must be exactly {list(_SEARCH_STAGES)}")

    ordered = tuple(
        str(_mapping(item, "stage")["id"])
        for item in sorted(
            stages,
            key=lambda item: int(_mapping(item, "stage")["order"]),
        )
    )
    if ordered != _SEARCH_STAGES:
        raise CalibrationError(f"Calibration stage order must be {list(_SEARCH_STAGES)}")

    base_configuration = _mapping(
        search_space.get("base_configuration"),
        "base_configuration",
    )
    fixed_parameters = _mapping(
        _mapping(
            search_space.get("calibration_constraints"),
            "calibration_constraints",
        ).get("fixed_parameters"),
        "fixed calibration parameters",
    )
    if dict(fixed_parameters) != dict(_FIXED_CALIBRATION_PARAMETERS):
        raise CalibrationError("Fixed calibration parameters differ from the v1 contract")
    if any(base_configuration.get(key) != value for key, value in fixed_parameters.items()):
        raise CalibrationError("Base configuration violates fixed calibration parameters")
    for stage_value in stages:
        stage = _mapping(stage_value, "stage")
        stage_id = str(stage.get("id"))
        expected_selector = {
            "geometry": _GEOMETRY_DEVELOPMENT_SELECTOR,
            "estimator": _MULTISCALE_DEVELOPMENT_SELECTOR,
            "interaction": _MULTISCALE_DEVELOPMENT_SELECTOR,
            "estimator_scale_transfer": _SCALE_TRANSFER_SELECTOR,
            "convergence": _MULTISCALE_DEVELOPMENT_SELECTOR,
        }[stage_id]
        if stage.get("dataset_selector") != expected_selector:
            raise CalibrationError(
                f"Stage {stage_id!r} changes its frozen development dataset selector"
            )
        identity_contract = _parent_expansion_identity_contract(stage)
        if identity_contract is not None:
            excluded_parameters, required_distinct_parents = identity_contract
            unknown_parameters = excluded_parameters - set(base_configuration)
            if unknown_parameters:
                raise CalibrationError(
                    f"Stage {stage_id!r} excludes unknown parent-identity parameters: "
                    f"{sorted(unknown_parameters)}"
                )
            maximum_parents = _integer(
                stage.get("maximum_parent_configurations"),
                f"{stage_id} maximum parents",
                minimum=1,
            )
            if required_distinct_parents != maximum_parents:
                raise CalibrationError(
                    f"Stage {stage_id!r} must require one distinct identity per parent"
                )
            if stage.get("include_inherited_configuration") is not False:
                raise CalibrationError(
                    f"Stage {stage_id!r} cannot combine an inherited candidate with a "
                    "parent-normalizing crossed design"
                )
            override_signatures: set[str] = set()
            for candidate_value in _sequence(stage.get("candidates"), "stage candidates"):
                candidate = _mapping(candidate_value, "candidate")
                overrides = _mapping(
                    candidate.get("parameter_overrides"),
                    "parameter overrides",
                )
                if set(overrides) != set(excluded_parameters):
                    raise CalibrationError(
                        f"Calibration candidate {candidate.get('id')!r} must override exactly "
                        f"the parent-identity parameters {sorted(excluded_parameters)}"
                    )
                signature = canonical_json_sha256(dict(overrides))
                if signature in override_signatures:
                    raise CalibrationError(
                        f"Stage {stage_id!r} contains duplicate crossed override tuples"
                    )
                override_signatures.add(signature)
        for candidate_value in _sequence(stage.get("candidates"), "stage candidates"):
            candidate = _mapping(candidate_value, "candidate")
            overrides = _mapping(candidate.get("parameter_overrides"), "parameter overrides")
            forbidden = sorted(set(overrides) & set(fixed_parameters))
            if forbidden:
                raise CalibrationError(
                    f"Calibration candidate {candidate.get('id')!r} overrides fixed "
                    f"parameters: {forbidden}"
                )
    materialize_group_policy(search_space, base_configuration)

    for stage_id in sorted(_CALIBRATION_STAGES):
        reduced = set(_algorithmic_seeds(search_space, "reduced", stage_id))
        full = set(_algorithmic_seeds(search_space, "full", stage_id))
        if reduced & full:
            raise CalibrationError(f"Reduced and full {stage_id!r} seeds must be disjoint")


def _validate_seed_contract(repo_root: Path, search_space: Mapping[str, Any]) -> None:
    """Recalculate every declared inference seed from the canonical seed ledger."""

    try:
        plan = load_seed_plan(repo_root / SEED_PLAN_PATH)
        for budget in sorted(_CALIBRATION_BUDGETS):
            for stage_id in (*_SEARCH_STAGES, "validation"):
                role = "validation" if stage_id == "validation" else "development"
                declared = _algorithmic_seeds(search_space, budget, stage_id)
                expected = tuple(
                    derive_plan_seed(
                        plan,
                        namespace=f"{budget}.{role}",
                        stream="inference",
                        coordinates={
                            "phase": "calibration",
                            "stage": stage_id,
                            "replicate": replicate,
                        },
                    )
                    for replicate in range(1, len(declared) + 1)
                )
                if declared != expected:
                    raise CalibrationError(
                        f"Calibration seeds for {budget}.{stage_id} differ from the "
                        "canonical seed ledger derivation"
                    )
    except SeedPlanError as exc:
        raise CalibrationError(f"Invalid benchmark seed contract: {exc}") from exc


def _validated_parents(
    stage: Mapping[str, Any],
    parents: Sequence[CandidateConfiguration],
) -> tuple[CandidateConfiguration | None, ...]:
    stage_id = str(stage["id"])
    if stage_id == "geometry":
        if parents:
            raise CalibrationError("The geometry stage cannot receive parent configurations")
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
    expected_parent_stage = _LOGICAL_PARENT_STAGES[stage_id]
    if expected_parent_stage is None:
        raise CalibrationError(f"Stage {stage_id!r} has no parent stage")
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
    base_configuration: Mapping[str, Any],
) -> CandidateConfiguration:
    owned_parameters = copy.deepcopy(dict(parameters))
    digest = _configuration_digest(owned_parameters)
    simplicity = sum(
        owned_parameters.get(key) != value for key, value in base_configuration.items()
    )
    return CandidateConfiguration(
        configuration_id=configuration_id,
        configuration_sha256=digest,
        stage_id=stage_id,
        dataset_role=role,
        parent_configuration_id=None if parent is None else parent.configuration_id,
        variant_id=variant_id,
        parameters=owned_parameters,
        base_parameter_changes=simplicity,
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
    configuration_hashes = [candidate.configuration_sha256 for candidate in candidates]
    if len(set(configuration_hashes)) != len(configuration_hashes):
        raise CalibrationError(f"Stage {stage_id!r} received duplicate effective configurations")

    inherited_variant = "base" if stage_id == "geometry" else "inherited"
    include_inherited = bool(stage["include_inherited_configuration"])
    expected_variants = {str(item["id"]) for item in stage["candidates"]}
    if include_inherited:
        expected_variants.add(inherited_variant)
    by_parent: dict[str | None, set[str]] = defaultdict(set)
    for candidate in candidates:
        by_parent[candidate.parent_configuration_id].add(candidate.variant_id)
    if stage_id == "geometry" and set(by_parent) != {None}:
        raise CalibrationError("Geometry candidates cannot have parent configurations")
    if stage_id != "geometry" and None in by_parent:
        raise CalibrationError(f"Stage {stage_id!r} candidates must identify their parent")
    maximum_parents = int(stage["maximum_parent_configurations"])
    if len(by_parent) > maximum_parents:
        raise CalibrationError(
            f"Stage {stage_id!r} candidate set exceeds its {maximum_parents}-parent budget"
        )
    for variants in by_parent.values():
        has_inherited = inherited_variant in variants
        if not variants <= expected_variants or has_inherited != include_inherited:
            raise CalibrationError(
                f"Stage {stage_id!r} contains variants outside its constrained design: "
                f"allowed {sorted(expected_variants)}, found {sorted(variants)}"
            )


def _validate_candidate_integrity(
    candidates: Sequence[CandidateConfiguration],
    base_configuration: Mapping[str, Any],
) -> None:
    base_keys = set(base_configuration)
    for candidate in candidates:
        parameters = dict(candidate.parameters)
        if set(parameters) != base_keys:
            raise CalibrationError(
                f"Candidate {candidate.configuration_id!r} does not contain the complete "
                "base parameter set"
            )
        expected_digest = _configuration_digest(parameters)
        if candidate.configuration_sha256 != expected_digest:
            raise CalibrationError(
                f"Candidate {candidate.configuration_id!r} parameter digest is inconsistent"
            )
        expected_simplicity = sum(
            parameters[key] != value for key, value in base_configuration.items()
        )
        if candidate.base_parameter_changes != expected_simplicity:
            raise CalibrationError(
                f"Candidate {candidate.configuration_id!r} simplicity is inconsistent"
            )
        if any(
            parameters.get(key) != value for key, value in _FIXED_CALIBRATION_PARAMETERS.items()
        ):
            raise CalibrationError(
                f"Candidate {candidate.configuration_id!r} violates fixed calibration parameters"
            )


def _configuration_digest(parameters: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(parameters))


def _parent_expansion_identity_contract(
    stage: Mapping[str, Any],
) -> tuple[frozenset[str], int] | None:
    raw_contract = stage.get("parent_expansion_identity")
    if raw_contract is None:
        return None
    contract = _mapping(raw_contract, "parent expansion identity")
    raw_excluded = _sequence(
        contract.get("excluded_parameters"),
        "parent expansion excluded parameters",
    )
    excluded = tuple(str(item) for item in raw_excluded)
    if not excluded or len(set(excluded)) != len(excluded):
        raise CalibrationError("Parent expansion excluded parameters must be non-empty and unique")
    required = _integer(
        contract.get("required_distinct_parents"),
        "required distinct parent identities",
        minimum=2,
    )
    return frozenset(excluded), required


def _retained_parameter_identity(
    candidate: CandidateConfiguration,
    excluded_parameters: frozenset[str],
) -> str:
    unknown = excluded_parameters - set(candidate.parameters)
    if unknown:
        raise CalibrationError(
            f"Candidate {candidate.configuration_id!r} lacks parent-identity parameters: "
            f"{sorted(unknown)}"
        )
    return canonical_json_sha256(
        {
            key: copy.deepcopy(value)
            for key, value in candidate.parameters.items()
            if key not in excluded_parameters
        }
    )


def _validate_parent_expansion_identities(
    stage: Mapping[str, Any],
    parents: Sequence[CandidateConfiguration],
) -> None:
    contract = _parent_expansion_identity_contract(stage)
    if contract is None or not parents:
        return
    excluded, _required = contract
    identities = [_retained_parameter_identity(parent, excluded) for parent in parents]
    if len(set(identities)) != len(identities):
        raise CalibrationError(
            f"Stage {stage['id']!r} received parents that become duplicate effective "
            "configurations after its declared overrides"
        )


def _select_stage_shortlist(
    search_space: Mapping[str, Any],
    stage_id: str,
    fidelity: str,
    scores: Sequence[CandidateScore],
    limit: int,
) -> tuple[CandidateScore, ...]:
    """Choose the best outcome-ranked shortlist satisfying downstream identifiability.

    A crossed child can deliberately overwrite parameters explored by its parent.
    In that case, the parent shortlist must retain enough distinct identities in the
    parameters that survive the child overrides. Representatives and remaining
    candidates are both chosen in the frozen score order; no candidate is silently
    deduplicated and the declared shortlist cardinality is preserved.
    """

    if limit < 1 or limit > len(scores):
        raise CalibrationError(
            f"Stage {stage_id!r} cannot select {limit} candidates from {len(scores)} scores"
        )
    child_id = _LOGICAL_CHILD_STAGES.get(stage_id)
    if child_id is None or fidelity not in FIDELITIES:
        return tuple(scores[:limit])
    child = _stage(search_space, child_id)
    contract = _parent_expansion_identity_contract(child)
    if contract is None:
        return tuple(scores[:limit])
    excluded, required = contract
    if limit < required:
        raise CalibrationError(
            f"Stage {stage_id!r} shortlist limit {limit} cannot provide the "
            f"{required} parent identities required by {child_id!r}"
        )

    representative_indexes: list[int] = []
    observed_identities: set[str] = set()
    for index, score in enumerate(scores):
        identity = _retained_parameter_identity(score.candidate, excluded)
        if identity in observed_identities:
            continue
        observed_identities.add(identity)
        representative_indexes.append(index)
        if len(representative_indexes) == required:
            break
    if len(representative_indexes) != required:
        raise CalibrationError(
            f"Stage {stage_id!r} provides only {len(representative_indexes)} distinct "
            f"parent identities; {child_id!r} requires {required}"
        )

    selected_indexes = set(representative_indexes)
    for index in range(len(scores)):
        if len(selected_indexes) == limit:
            break
        selected_indexes.add(index)
    return tuple(scores[index] for index in sorted(selected_indexes))


def _rank_balanced_rows(
    candidates: Sequence[CandidateConfiguration],
    summary_rows: Iterable[Mapping[str, Any]],
    *,
    expected_role: str,
    expected_seeds: tuple[int, ...] | None,
    expected_fidelity: str,
    expected_dataset_count: int,
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
            f"validation__{configuration_id}__final__seed_{seed}"
            if expected_role == "validation"
            else f"{configuration_id}__{expected_fidelity}__seed_{seed}"
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
    if len(dataset_ids) != expected_dataset_count:
        raise CalibrationError(
            f"Summary dataset count {len(dataset_ids)} differs from the predeclared "
            f"count {expected_dataset_count}"
        )

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
                simplicity=candidate.base_parameter_changes,
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
