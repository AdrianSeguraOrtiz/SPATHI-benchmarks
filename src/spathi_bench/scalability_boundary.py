"""Prospective, outcome-blind resource boundaries for scalability runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .config import BenchmarkConfigError, load_json_object, validate_json

CONTRACT_PATH = Path("experiments/v1/spathi/scalability-boundary.json")
SCHEMA_PATH = Path("schemas/scalability-boundary.schema.json")
_RESULT_FIELDS = frozenset(
    {
        "dataset_id",
        "stage",
        "status",
        "elapsed_seconds",
        "peak_memory_bytes",
        "target_count",
        "scout_target_count",
    }
)
_STATUSES = frozenset({"completed", "oom", "timeout", "tool_error"})
_SKIPPED = "not_attempted_after_preregistered_resource_boundary"


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalabilityDecision:
    """One auditable attempt/stop decision made before a scalability run."""

    dataset_id: str
    lane_id: str
    stage: Literal["scout", "all_targets"]
    decision: Literal["attempt", "not_attempted_after_preregistered_resource_boundary"]
    reason: str
    evidence_dataset_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "dataset_id": self.dataset_id,
            "lane_id": self.lane_id,
            "stage": self.stage,
            "decision": self.decision,
            "reason": self.reason,
            "evidence_dataset_id": self.evidence_dataset_id,
        }


def load_scalability_boundary(repo_root: Path) -> dict[str, Any]:
    """Load and cross-check the committed boundary and public scale panels."""

    contract = load_json_object(repo_root / CONTRACT_PATH, "scalability boundary")
    validate_json(contract, repo_root / SCHEMA_PATH, "scalability boundary")
    for budget in ("reduced", "full"):
        panel = load_json_object(
            repo_root / f"experiments/v1/simulations/scale_stress/scale-{budget}.json",
            f"{budget} scalability panel",
        )
        declared = contract["panels"][budget]
        observed_ids = [str(item["id"]) for item in panel["datasets"]]
        if declared["dataset_ids"] != observed_ids:
            raise BenchmarkConfigError(
                f"{budget} scalability boundary dataset order differs from its panel"
            )
        lanes = declared["lanes"]
        lane_ids = [str(lane["id"]) for lane in lanes]
        if len(lane_ids) != len(set(lane_ids)):
            raise BenchmarkConfigError(f"{budget} scalability lanes have duplicate IDs")
        covered = {item for lane in lanes for item in lane["ordered_dataset_ids"]}
        if covered != set(observed_ids):
            raise BenchmarkConfigError(
                f"{budget} scalability lanes must cover every declared dataset"
            )
        if any(item not in observed_ids for lane in lanes for item in lane["ordered_dataset_ids"]):
            raise BenchmarkConfigError(f"{budget} scalability lane references an unknown dataset")
        positions = {dataset_id: index for index, dataset_id in enumerate(observed_ids)}
        anchor = observed_ids[0]
        appearances = {dataset_id: 0 for dataset_id in observed_ids[1:]}
        for lane in lanes:
            ordered = lane["ordered_dataset_ids"]
            if ordered[0] != anchor:
                raise BenchmarkConfigError(
                    f"{budget} scalability lane {lane['id']!r} does not start at the panel anchor"
                )
            if [positions[item] for item in ordered] != sorted(positions[item] for item in ordered):
                raise BenchmarkConfigError(
                    f"{budget} scalability lane {lane['id']!r} differs from panel order"
                )
            for dataset_id in ordered[1:]:
                appearances[dataset_id] += 1
        if any(count != 1 for count in appearances.values()):
            raise BenchmarkConfigError(
                f"{budget} non-anchor datasets must belong to exactly one progression lane"
            )
    return contract


def target_scout_scope(
    contract: Mapping[str, Any], *, budget: str, dataset_id: str
) -> dict[str, Any]:
    """Materialize the frozen, dataset-specific hash target scope."""

    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError("Scalability budget must be 'reduced' or 'full'")
    panel = contract["panels"][budget]
    if dataset_id not in panel["dataset_ids"]:
        raise BenchmarkConfigError(f"Unknown {budget} scalability dataset {dataset_id!r}")
    scout = contract["target_scout"]
    return {
        "method": scout["method"],
        "size": scout["sizes"][budget],
        "salt": scout["salt"],
        "dataset_id": dataset_id,
    }


def _validated_results(results: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    validated: list[Mapping[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(results):
        unexpected = set(raw) - _RESULT_FIELDS
        missing = {"dataset_id", "stage", "status"} - set(raw)
        if unexpected or missing:
            raise BenchmarkConfigError(
                f"Scalability resource result {index} has unexpected={sorted(unexpected)} "
                f"and missing={sorted(missing)} fields"
            )
        dataset_id = raw["dataset_id"]
        stage = raw["stage"]
        status = raw["status"]
        if not isinstance(dataset_id, str) or not dataset_id:
            raise BenchmarkConfigError("Scalability result dataset_id must be non-empty text")
        if stage not in {"scout", "all_targets"} or status not in {*_STATUSES, _SKIPPED}:
            raise BenchmarkConfigError("Scalability result has an invalid stage or status")
        identity = (dataset_id, stage)
        if identity in identities:
            raise BenchmarkConfigError(f"Duplicate scalability resource result {identity!r}")
        identities.add(identity)
        for field in (
            "elapsed_seconds",
            "peak_memory_bytes",
            "target_count",
            "scout_target_count",
        ):
            value = raw.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            ):
                raise BenchmarkConfigError(f"Scalability result {field} must be non-negative")
        validated.append(raw)
    return tuple(validated)


def decide_scalability_attempt(
    contract: Mapping[str, Any],
    *,
    budget: str,
    lane_id: str,
    dataset_id: str,
    stage: Literal["scout", "all_targets"],
    prior_results: Sequence[Mapping[str, Any]],
    allocation_memory_bytes: int,
) -> ScalabilityDecision:
    """Apply the preregistered boundary using resource evidence only.

    ``prior_results`` must be a prefix of the lane's prescribed scout/full order.
    Accuracy, truth, expression or network content cannot enter this API.
    """

    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError("Scalability budget must be 'reduced' or 'full'")
    if stage not in {"scout", "all_targets"}:
        raise BenchmarkConfigError("Scalability stage must be scout or all_targets")
    if (
        isinstance(allocation_memory_bytes, bool)
        or not isinstance(allocation_memory_bytes, int)
        or allocation_memory_bytes <= 0
    ):
        raise BenchmarkConfigError("Scalability resource limits must be positive")
    lanes = {lane["id"]: lane for lane in contract["panels"][budget]["lanes"]}
    if lane_id not in lanes:
        raise BenchmarkConfigError(f"Unknown {budget} scalability lane {lane_id!r}")
    ordered = tuple(lanes[lane_id]["ordered_dataset_ids"])
    if dataset_id not in ordered:
        raise BenchmarkConfigError(f"Dataset {dataset_id!r} is outside lane {lane_id!r}")
    results = _validated_results(prior_results)
    expected_order = tuple(
        (candidate, phase) for candidate in ordered for phase in ("scout", "all_targets")
    )
    requested = (dataset_id, stage)
    requested_index = expected_order.index(requested)
    observed = tuple((str(row["dataset_id"]), str(row["stage"])) for row in results)
    if observed != expected_order[: len(observed)] or len(observed) != requested_index:
        raise BenchmarkConfigError(
            "prior_results must be the exact preregistered prefix preceding the request"
        )
    if not results:
        return ScalabilityDecision(
            dataset_id=dataset_id,
            lane_id=lane_id,
            stage=stage,
            decision="attempt",
            reason="first_preregistered_scout",
        )

    previous = results[-1]
    previous_id = str(previous["dataset_id"])
    boundary = contract["resource_boundary"]
    if previous["status"] == _SKIPPED:
        return ScalabilityDecision(
            dataset_id=dataset_id,
            lane_id=lane_id,
            stage=stage,
            decision=_SKIPPED,
            reason="preceding_preregistered_resource_boundary",
            evidence_dataset_id=previous_id,
        )
    if previous["status"] in set(boundary["terminal_statuses"]):
        return ScalabilityDecision(
            dataset_id=dataset_id,
            lane_id=lane_id,
            stage=stage,
            decision="not_attempted_after_preregistered_resource_boundary",
            reason=f"preceding_{previous['status']}",
            evidence_dataset_id=previous_id,
        )
    elapsed = previous.get("elapsed_seconds")
    wall_time_limit_seconds = float(boundary["wall_time_limit_seconds"][budget])
    wall_fraction = float(boundary["maximum_projected_wall_time_fraction"])
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
        raise BenchmarkConfigError(
            "A completed scalability result needs canonical ANDREA execution elapsed_seconds"
        )
    if float(elapsed) > wall_fraction * wall_time_limit_seconds:
        return ScalabilityDecision(
            dataset_id=dataset_id,
            lane_id=lane_id,
            stage=stage,
            decision="not_attempted_after_preregistered_resource_boundary",
            reason="preceding_run_exceeds_wall_time_boundary",
            evidence_dataset_id=previous_id,
        )
    if stage == "all_targets":
        target_count = previous.get("target_count")
        scout_count = previous.get("scout_target_count")
        if not all(
            isinstance(value, (int, float)) and value > 0
            for value in (
                elapsed,
                target_count,
                scout_count,
            )
        ):
            raise BenchmarkConfigError(
                "A completed scout needs positive elapsed, target_count and scout_target_count"
            )
        projected = float(elapsed) * float(target_count) / float(scout_count)
        if projected > wall_fraction * wall_time_limit_seconds:
            return ScalabilityDecision(
                dataset_id=dataset_id,
                lane_id=lane_id,
                stage=stage,
                decision="not_attempted_after_preregistered_resource_boundary",
                reason="scout_projection_exceeds_wall_time_boundary",
                evidence_dataset_id=previous_id,
            )
    peak = previous.get("peak_memory_bytes")
    if not isinstance(peak, (int, float)):
        raise BenchmarkConfigError(
            "A completed scalability result needs canonical container peak_memory_bytes"
        )
    if float(peak) > float(boundary["maximum_observed_memory_fraction"]) * allocation_memory_bytes:
        return ScalabilityDecision(
            dataset_id=dataset_id,
            lane_id=lane_id,
            stage=stage,
            decision="not_attempted_after_preregistered_resource_boundary",
            reason="preceding_run_exceeds_memory_boundary",
            evidence_dataset_id=previous_id,
        )
    return ScalabilityDecision(
        dataset_id=dataset_id,
        lane_id=lane_id,
        stage=stage,
        decision="attempt",
        reason="within_preregistered_resource_boundary",
        evidence_dataset_id=previous_id,
    )


__all__ = [
    "ScalabilityDecision",
    "decide_scalability_attempt",
    "load_scalability_boundary",
    "target_scout_scope",
]
