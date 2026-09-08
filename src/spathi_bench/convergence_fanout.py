"""Materialize strict one-fit/four-result SPATHI convergence plans."""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256
from .calibration import CalibrationError, CalibrationRun, CandidateConfiguration
from .config import validate_json

SCHEMA_PATH = Path("schemas/calibration-convergence-plans.schema.json")
ESTIMATOR_COUNTS = (250, 500, 750, 1000)


def _physical_run_id(parent_id: str, seed: int) -> str:
    token = re.sub(r"[^a-z0-9_-]+", "_", parent_id.casefold()).strip("_-")
    token = token[:48] or "parent"
    digest = canonical_json_sha256({"parent_configuration_id": parent_id, "seed": seed})[:12]
    return f"convergence_{token}_{digest}_seed_{seed}"


def build_convergence_plans(
    repo_root: Path,
    candidates: Sequence[CandidateConfiguration],
    runs: Sequence[CalibrationRun],
) -> dict[str, Any]:
    """Collapse logical tree-count runs into one maximum fit per parent and seed."""

    by_configuration = {candidate.configuration_id: candidate for candidate in candidates}
    if len(by_configuration) != len(candidates) or any(
        candidate.stage_id != "convergence" for candidate in candidates
    ):
        raise CalibrationError("Convergence fan-out requires unique convergence candidates")
    if any(run.stage_id != "convergence" or run.fidelity != "final" for run in runs):
        raise CalibrationError("Convergence fan-out received a non-final convergence run")
    grouped: dict[tuple[str, int], list[tuple[CandidateConfiguration, CalibrationRun]]] = (
        defaultdict(list)
    )
    for run in runs:
        candidate = by_configuration.get(run.configuration_id)
        if candidate is None:
            raise CalibrationError(f"Convergence run {run.run_id!r} has no candidate")
        parent = candidate.parent_configuration_id
        if not isinstance(parent, str) or not parent:
            raise CalibrationError("Convergence candidates require a parent configuration")
        grouped[(parent, run.algorithm_seed)].append((candidate, run))

    plans: list[dict[str, Any]] = []
    for (parent, seed), entries in sorted(grouped.items()):
        entries.sort(key=lambda item: int(item[0].parameters["n_estimators"]))
        counts = tuple(int(candidate.parameters["n_estimators"]) for candidate, _run in entries)
        if counts != ESTIMATOR_COUNTS:
            raise CalibrationError(
                f"Convergence parent {parent!r} seed {seed} must expose {ESTIMATOR_COUNTS}, "
                f"not {counts}"
            )
        maximum_run = entries[-1][1]
        maximum_parameters = copy.deepcopy(dict(maximum_run.parameters))
        if maximum_parameters.pop("target_scope", None) != {"method": "all"}:
            raise CalibrationError("Convergence fan-out must use all targets")
        if maximum_parameters.get("random_seed") != seed:
            raise CalibrationError("Convergence physical model seed is inconsistent")
        maximum_policy = maximum_parameters.get("group_policy")
        if not isinstance(maximum_policy, Mapping):
            raise CalibrationError("Convergence physical model requires a group policy")
        for candidate, run in entries:
            logical_parameters = copy.deepcopy(dict(run.parameters))
            if logical_parameters.pop("target_scope", None) != {"method": "all"}:
                raise CalibrationError("Convergence logical prefixes must use all targets")
            if logical_parameters.get("random_seed") != seed:
                raise CalibrationError("Convergence logical prefix seed is inconsistent")
            policy = logical_parameters.get("group_policy")
            if not isinstance(policy, Mapping):
                raise CalibrationError("Convergence logical prefix requires a group policy")
            for branch in ("single_group", "multi_group"):
                candidate_model = dict(policy["branches"][branch]["parameters"])
                maximum_model = dict(maximum_policy["branches"][branch]["parameters"])
                candidate_count = candidate_model.pop("n_estimators")
                maximum_model.pop("n_estimators")
                if candidate_count != candidate.parameters["n_estimators"]:
                    raise CalibrationError("Convergence run tree count differs from its candidate")
                if candidate_model != maximum_model:
                    raise CalibrationError(
                        "Convergence prefixes differ in parameters other than n_estimators"
                    )
        plans.append(
            {
                "schema_version": "1.0",
                "physical_run_id": _physical_run_id(parent, seed),
                "parent_configuration_id": parent,
                "algorithm_seed": seed,
                "model": maximum_parameters,
                "logical_prefixes": [
                    {
                        "run_id": run.run_id,
                        "configuration_id": candidate.configuration_id,
                        "configuration_sha256": candidate.configuration_sha256,
                        "n_estimators": int(candidate.parameters["n_estimators"]),
                    }
                    for candidate, run in entries
                ],
            }
        )
    if not plans:
        raise CalibrationError("Convergence fan-out cannot be empty")
    payload = {
        "schema_version": "1.0",
        "artifact_type": "spathi_calibration_convergence_plans",
        "physical_plans_per_dataset": len(plans),
        "logical_results_per_dataset": 4 * len(plans),
        "plans": plans,
    }
    validate_json(payload, repo_root / SCHEMA_PATH, "calibration convergence plans")
    return payload


def validate_convergence_plans(
    repo_root: Path,
    payload: Mapping[str, Any],
    candidates: Sequence[CandidateConfiguration],
    runs: Sequence[CalibrationRun],
) -> None:
    """Require a convergence plan to equal its canonical logical-run collapse."""

    validate_json(
        dict(payload),
        repo_root / SCHEMA_PATH,
        "calibration convergence plans",
    )
    if dict(payload) != build_convergence_plans(repo_root, candidates, runs):
        raise CalibrationError("Convergence plans differ from the canonical calibration runs")


__all__ = [
    "ESTIMATOR_COUNTS",
    "build_convergence_plans",
    "validate_convergence_plans",
]
