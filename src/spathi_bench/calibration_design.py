"""Prospective multi-fidelity design and exact SPATHI invocation ledger."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import BenchmarkConfigError

MULTIFIDELITY_STAGES = (
    "geometry",
    "estimator",
    "interaction",
    "estimator_scale_transfer",
)
FINAL_STAGES = ("convergence", "validation")
FIDELITIES = ("scout", "confirmation")


class CalibrationDesignError(BenchmarkConfigError):
    """Raised when the prospective multi-fidelity ledger is inconsistent."""


@dataclass(frozen=True, slots=True)
class CalibrationPhase:
    """One immutable unit in a calibration budget ledger."""

    stage_id: str
    fidelity: str
    candidate_count: int
    dataset_count: int
    algorithm_seed_count: int
    expected_physical_fits: int
    expected_logical_results: int
    candidate_source: str
    advance_limit: int
    condition_ids: tuple[str, ...]
    replicate_indices: tuple[int, ...]
    algorithm_seed_indices: tuple[int, ...]
    target_scope: Mapping[str, Any]
    execution_primitive: str | None = None


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationDesignError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise CalibrationDesignError(f"{label} must be an array")
    return value


def _integer(value: object, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise CalibrationDesignError(f"{label} must be an integer >= {minimum}")
    return value


def _stage(search_space: Mapping[str, Any], stage_id: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in _sequence(search_space.get("stages"), "calibration stages")
        if isinstance(item, Mapping) and item.get("id") == stage_id
    ]
    if len(matches) != 1:
        raise CalibrationDesignError(f"Unknown or duplicate calibration stage {stage_id!r}")
    return matches[0]


def _expanded_candidate_count(search_space: Mapping[str, Any], stage_id: str) -> int:
    stage = _stage(search_space, stage_id)
    variants = len(_sequence(stage.get("candidates"), f"{stage_id} candidates"))
    per_parent = variants + int(bool(stage.get("include_inherited_configuration")))
    parents = _integer(
        stage.get("maximum_parent_configurations"),
        f"{stage_id} maximum parents",
    )
    return per_parent * parents


def _budget(search_space: Mapping[str, Any], budget_id: str) -> Mapping[str, Any]:
    budgets = _mapping(search_space.get("calibration_budgets"), "calibration budgets")
    if budget_id not in {"reduced", "full"}:
        raise CalibrationDesignError(f"Unknown calibration budget {budget_id!r}")
    return _mapping(budgets.get(budget_id), f"{budget_id} calibration budget")


def _tier(
    search_space: Mapping[str, Any], budget_id: str, stage_id: str, fidelity: str
) -> Mapping[str, Any]:
    budget = _budget(search_space, budget_id)
    multifidelity = _mapping(budget.get("multifidelity"), f"{budget_id} multifidelity")
    stages = _mapping(multifidelity.get("stages"), f"{budget_id} multifidelity stages")
    stage = _mapping(stages.get(stage_id), f"{budget_id}.{stage_id} design")
    if stage_id in MULTIFIDELITY_STAGES:
        if fidelity not in FIDELITIES:
            raise CalibrationDesignError(
                f"Stage {stage_id!r} requires fidelity scout or confirmation"
            )
        return _mapping(stage.get(fidelity), f"{budget_id}.{stage_id}.{fidelity}")
    if stage_id not in FINAL_STAGES or fidelity != "final":
        raise CalibrationDesignError(f"Unknown calibration phase {stage_id}.{fidelity}")
    return stage


def calibration_phase(
    search_space: Mapping[str, Any], budget_id: str, stage_id: str, fidelity: str
) -> CalibrationPhase:
    """Resolve one phase and recompute its Cartesian invocation count."""

    tier = _tier(search_space, budget_id, stage_id, fidelity)
    if stage_id in MULTIFIDELITY_STAGES:
        if fidelity == "scout":
            candidate_count = _expanded_candidate_count(search_space, stage_id)
        else:
            scout = _tier(search_space, budget_id, stage_id, "scout")
            candidate_count = _integer(
                scout.get("advance_limit"),
                f"{budget_id}.{stage_id}.scout advance limit",
            )
        selector = _mapping(tier.get("dataset_selector"), "fidelity dataset selector")
        condition_ids = tuple(str(item) for item in selector["condition_ids"])
        replicate_indices = tuple(int(item) for item in selector["replicate_indices"])
        dataset_count = len(condition_ids) * len(replicate_indices)
    elif stage_id == "convergence":
        stage = _stage(search_space, stage_id)
        candidate_count = _integer(
            stage.get("maximum_parent_configurations"),
            "convergence parent count",
        )
        selector = _mapping(tier.get("dataset_selector"), "convergence dataset selector")
        condition_ids = tuple(str(item) for item in selector["condition_ids"])
        replicate_indices = tuple(int(item) for item in selector["replicate_indices"])
        dataset_count = len(condition_ids) * len(replicate_indices)
    else:
        protocol = _mapping(search_space.get("selection_protocol"), "selection protocol")
        candidate_count = _integer(
            protocol.get("maximum_validation_finalists"),
            "validation finalist count",
        )
        if candidate_count != 2:
            raise CalibrationDesignError("Validation must compare exactly two finalists")
        condition_ids = ()
        replicate_indices = ()
        dataset_count = _integer(tier.get("expected_dataset_count"), "validation datasets")

    seed_indices = tuple(int(item) for item in tier["algorithm_seed_indices"])
    seed_count = len(seed_indices)
    expected = candidate_count * dataset_count * seed_count
    declared = _integer(tier.get("expected_physical_fits"), "expected physical fits")
    if expected != declared:
        raise CalibrationDesignError(
            f"{budget_id}.{stage_id}.{fidelity} declares {declared} physical fits; "
            f"its frozen Cartesian design yields {expected}"
        )
    logical_results = (
        _integer(tier.get("expected_logical_results"), "expected logical results")
        if stage_id == "convergence"
        else expected
    )
    if stage_id == "convergence" and logical_results != expected * 4:
        raise CalibrationDesignError(
            f"{budget_id}.convergence must emit four prefix results per physical fit"
        )
    return CalibrationPhase(
        stage_id=stage_id,
        fidelity=fidelity,
        candidate_count=candidate_count,
        dataset_count=dataset_count,
        algorithm_seed_count=seed_count,
        expected_physical_fits=expected,
        expected_logical_results=logical_results,
        candidate_source=str(tier["candidate_source"]),
        advance_limit=_integer(tier.get("advance_limit"), "advance limit"),
        condition_ids=condition_ids,
        replicate_indices=replicate_indices,
        algorithm_seed_indices=seed_indices,
        target_scope=dict(_mapping(tier.get("target_scope"), "target scope")),
        execution_primitive=(
            str(tier["execution_primitive"]) if "execution_primitive" in tier else None
        ),
    )


def calibration_ledger(
    search_space: Mapping[str, Any], budget_id: str
) -> tuple[CalibrationPhase, ...]:
    """Return and validate the complete ordered invocation ledger for one profile."""

    phases = tuple(
        calibration_phase(search_space, budget_id, stage, fidelity)
        for stage in MULTIFIDELITY_STAGES
        for fidelity in FIDELITIES
    ) + tuple(calibration_phase(search_space, budget_id, stage, "final") for stage in FINAL_STAGES)
    budget = _budget(search_space, budget_id)
    declared_total = _integer(
        budget.get("expected_physical_fits"),
        f"{budget_id} expected SPATHI invocations",
    )
    observed_total = sum(phase.expected_physical_fits for phase in phases)
    if observed_total != declared_total:
        raise CalibrationDesignError(
            f"{budget_id} calibration ledger totals {observed_total}, not {declared_total}"
        )
    declared_logical = _integer(
        budget.get("expected_logical_results"),
        f"{budget_id} expected logical results",
    )
    observed_logical = sum(phase.expected_logical_results for phase in phases)
    if observed_logical != declared_logical:
        raise CalibrationDesignError(
            f"{budget_id} calibration emits {observed_logical} logical results, "
            f"not {declared_logical}"
        )
    return phases


def validate_multifidelity_design(search_space: Mapping[str, Any]) -> None:
    """Enforce prospective independence, coverage and high-TF confirmation rules."""

    for budget_id in ("reduced", "full"):
        budget = _budget(search_space, budget_id)
        multifidelity = _mapping(budget.get("multifidelity"), f"{budget_id} multifidelity")
        if (
            multifidelity.get("independent_profile") is not True
            or multifidelity.get("may_consume_other_profile_outcomes") is not False
            or multifidelity.get("panel_assignment") != "predeclared_before_outcomes"
        ):
            raise CalibrationDesignError(
                f"{budget_id} calibration is not prospectively independent"
            )
        forbidden = set(
            str(item)
            for item in _sequence(
                multifidelity.get("forbidden_scout_inputs"),
                f"{budget_id} forbidden scout inputs",
            )
        )
        other = "full_outcomes" if budget_id == "reduced" else "reduced_outcomes"
        required_forbidden = {
            "ground_truth_for_target_selection",
            "expression_values_for_target_selection",
            "test_outcomes",
            other,
        }
        if forbidden != required_forbidden:
            raise CalibrationDesignError(
                f"{budget_id} scout forbidden-input contract is incomplete"
            )

        seeds_by_stage = _mapping(budget.get("algorithmic_seeds"), "algorithmic seeds")
        for stage_id in MULTIFIDELITY_STAGES:
            stage = _stage(search_space, stage_id)
            allowed_conditions = set(
                str(item)
                for item in _mapping(stage.get("dataset_selector"), f"{stage_id} dataset selector")[
                    "condition_ids"
                ]
            )
            scout = calibration_phase(search_space, budget_id, stage_id, "scout")
            confirmation = calibration_phase(search_space, budget_id, stage_id, "confirmation")
            if (
                not set(scout.condition_ids) <= allowed_conditions
                or not set(confirmation.condition_ids) <= allowed_conditions
            ):
                raise CalibrationDesignError(
                    f"{budget_id}.{stage_id} selects conditions outside its development panel"
                )
            if scout.candidate_source != "all_expanded":
                raise CalibrationDesignError(f"{budget_id}.{stage_id} scout omits candidates")
            if confirmation.candidate_source != "scout_shortlist":
                raise CalibrationDesignError(
                    f"{budget_id}.{stage_id} confirmation is not bound to its scout"
                )
            if confirmation.candidate_count != scout.advance_limit:
                raise CalibrationDesignError(
                    f"{budget_id}.{stage_id} confirmation quota differs from scout advance"
                )
            stage_seeds = tuple(seeds_by_stage[stage_id])
            used_indices = set(scout.algorithm_seed_indices) | set(
                confirmation.algorithm_seed_indices
            )
            if (
                used_indices != set(range(1, len(stage_seeds) + 1))
                or set(scout.algorithm_seed_indices)
                & set(confirmation.algorithm_seed_indices)
            ):
                raise CalibrationDesignError(
                    f"{budget_id}.{stage_id} fidelity seed allocation is invalid"
                )

        for child_id in MULTIFIDELITY_STAGES:
            child = _stage(search_space, child_id)
            raw_identity = child.get("parent_expansion_identity")
            if raw_identity is None:
                continue
            identity = _mapping(
                raw_identity,
                f"{child_id} parent expansion identity",
            )
            inheritance = str(child.get("inherits_from"))
            if not inheritance.startswith("shortlist:"):
                raise CalibrationDesignError(
                    f"{child_id} declares parent identities without a shortlist parent"
                )
            parent_id = inheritance.removeprefix("shortlist:")
            required = _integer(
                identity.get("required_distinct_parents"),
                f"{child_id} required distinct parents",
            )
            for fidelity in FIDELITIES:
                parent_phase = calibration_phase(
                    search_space,
                    budget_id,
                    parent_id,
                    fidelity,
                )
                if parent_phase.advance_limit < required:
                    raise CalibrationDesignError(
                        f"{budget_id}.{parent_id}.{fidelity} cannot retain the {required} "
                        f"identities required by {child_id}"
                    )

        high_tf_scout = calibration_phase(
            search_space, budget_id, "estimator_scale_transfer", "scout"
        )
        high_tf_confirmation = calibration_phase(
            search_space, budget_id, "estimator_scale_transfer", "confirmation"
        )
        if high_tf_scout.target_scope != {
            "method": "sha256_gene_id",
            "size": 256,
            "salt": "spathi-calibration-high-tf-v1",
        } or high_tf_confirmation.target_scope != {"method": "all"}:
            raise CalibrationDesignError(
                f"{budget_id} high-TF stage must scout 256 hash-only targets and confirm all"
            )
        for phase in calibration_ledger(search_space, budget_id):
            stage_seeds = tuple(seeds_by_stage[phase.stage_id])
            if max(phase.algorithm_seed_indices) > len(stage_seeds) or (
                phase.stage_id in FINAL_STAGES
                and set(phase.algorithm_seed_indices)
                != set(range(1, len(stage_seeds) + 1))
            ):
                raise CalibrationDesignError(
                    f"{budget_id}.{phase.stage_id} algorithm-seed allocation is invalid"
                )


__all__ = [
    "CalibrationDesignError",
    "CalibrationPhase",
    "FIDELITIES",
    "FINAL_STAGES",
    "MULTIFIDELITY_STAGES",
    "calibration_ledger",
    "calibration_phase",
    "validate_multifidelity_design",
]
