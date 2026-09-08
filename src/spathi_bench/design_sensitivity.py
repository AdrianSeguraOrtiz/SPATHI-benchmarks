"""Prospective, outcome-free operating-characteristic checks for efficiency design."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .artifacts import canonical_json_sha256
from .config import (
    BenchmarkConfigError,
    load_json_object,
    load_profile,
    resolve_repo_path,
    validate_json,
)

SENSITIVITY_RELATIVE_PATH = "experiments/v1/spathi/efficiency-design-sensitivity.json"


def efficiency_profile_strata(repo_root: Path, budget: str) -> list[dict[str, Any]]:
    """Return the exact simulator-condition replication contract for one budget."""

    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError("Efficiency stratum budget must be 'reduced' or 'full'")
    profile = load_profile(repo_root, f"efficiency-{budget}", require_runnable=True)
    strata: list[dict[str, Any]] = []
    for job in profile["generation_jobs"]:
        if job["dataset_role"] != "efficiency_validation":
            continue
        runs_path = resolve_repo_path(
            repo_root,
            str(job["simulator_runs"]),
            f"efficiency design-sensitivity runs for {job['id']}",
        )
        payload = load_json_object(runs_path, f"efficiency runs for {job['id']}")
        runs = payload.get("runs")
        if not isinstance(runs, list) or len(runs) != 1:
            raise BenchmarkConfigError(
                f"Efficiency job {job['id']!r} must resolve to exactly one simulator run"
            )
        run = runs[0]
        strata.append(
            {
                "generation_job": job["id"],
                "simulator_id": run["simulator_id"],
                "condition_id": job["condition_id"],
                "replicates": run["replicates"],
            }
        )
    return sorted(strata, key=lambda item: item["generation_job"])


def _effective_sample_size(strata: list[dict[str, Any]]) -> float:
    conditions_by_simulator = Counter(stratum["simulator_id"] for stratum in strata)
    simulator_count = len(conditions_by_simulator)
    if simulator_count < 1:
        raise BenchmarkConfigError("Efficiency sensitivity design contains no simulators")
    weights: list[float] = []
    for stratum in strata:
        replicates = int(stratum["replicates"])
        weight = (
            1.0 / simulator_count / conditions_by_simulator[stratum["simulator_id"]] / replicates
        )
        weights.extend([weight] * replicates)
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise BenchmarkConfigError("Efficiency sensitivity weights do not sum to one")
    return 1.0 / sum(weight * weight for weight in weights)


def _strictly_increasing(values: list[float], label: str) -> None:
    if any(right <= left for left, right in zip(values, values[1:], strict=False)):
        raise BenchmarkConfigError(f"{label} must be strictly increasing")


def evaluate_design_sensitivity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the declared normal-location scenarios without random sampling.

    These curves are deliberately assumption-conditional. They make the information
    content of the frozen design visible without pretending to estimate unknown power.
    """

    method = contract["method"]
    target = contract["target_design"]
    grid = contract["scenario_grid"]
    alpha = float(method["per_candidate_one_sided_alpha"])
    critical_value = NormalDist().inv_cdf(1.0 - alpha)
    effective_n = float(target["expected_effective_sample_size"])
    margin = float(grid["accuracy"]["noninferiority_margin"])
    accuracy: list[dict[str, float]] = []
    for standard_deviation in grid["accuracy"]["within_stratum_standard_deviations"]:
        standard_error = float(standard_deviation) / math.sqrt(effective_n)
        for true_difference in grid["accuracy"]["true_mean_differences"]:
            probability = NormalDist().cdf(
                (float(true_difference) + margin) / standard_error - critical_value
            )
            accuracy.append(
                {
                    "true_mean_difference": float(true_difference),
                    "within_stratum_standard_deviation": float(standard_deviation),
                    "approximate_gate_probability": probability,
                }
            )

    maximum_ratio = float(grid["runtime"]["maximum_ratio"])
    runtime: list[dict[str, float]] = []
    for log_standard_deviation in grid["runtime"]["within_stratum_log_standard_deviations"]:
        standard_error = float(log_standard_deviation) / math.sqrt(effective_n)
        for true_ratio in grid["runtime"]["true_geometric_mean_ratios"]:
            probability = NormalDist().cdf(
                (math.log(maximum_ratio) - math.log(float(true_ratio))) / standard_error
                - critical_value
            )
            runtime.append(
                {
                    "true_geometric_mean_ratio": float(true_ratio),
                    "within_stratum_log_standard_deviation": float(log_standard_deviation),
                    "approximate_gate_probability": probability,
                }
            )
    result = {
        "schema_version": "1.0",
        "contract_id": contract["id"],
        "interpretation": contract["interpretation"],
        "critical_value": critical_value,
        "effective_sample_size": effective_n,
        "accuracy": accuracy,
        "runtime": runtime,
    }
    result["operating_characteristics_sha256"] = canonical_json_sha256(result)
    return result


def load_design_sensitivity_contract(
    repo_root: Path,
    *,
    study: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the prospective contract against the current frozen full design."""

    root = repo_root.resolve()
    path = root / SENSITIVITY_RELATIVE_PATH
    contract = load_json_object(path, "efficiency design-sensitivity contract")
    validate_json(
        contract,
        root / "schemas/efficiency-design-sensitivity.schema.json",
        "efficiency design-sensitivity contract",
    )
    if study is None:
        study = load_json_object(
            root / contract["study_path"],
            "efficiency study for design sensitivity",
        )
    reference = study["analysis"]["design_sensitivity_contract"]
    if reference["path"] != SENSITIVITY_RELATIVE_PATH:
        raise BenchmarkConfigError("Efficiency study points to another sensitivity contract")
    if reference["interpretation"] != contract["interpretation"]:
        raise BenchmarkConfigError("Efficiency study changes the sensitivity interpretation")

    method = contract["method"]
    analysis = study["analysis"]
    expected_alpha = float(analysis["familywise_alpha"])
    if (
        analysis.get("multiplicity") != "single_hypothesis_no_adjustment"
        or float(method["familywise_alpha"]) != expected_alpha
        or method["confirmatory_candidate_count"] != 1
        or not math.isclose(
            float(method["per_candidate_one_sided_alpha"]),
            expected_alpha,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or method["weighting"] != study["analysis"]["aggregation"]
    ):
        raise BenchmarkConfigError(
            "Efficiency sensitivity single-nominee inference or weighting differs from the study"
        )

    target = contract["target_design"]
    expected_strata = efficiency_profile_strata(root, "full")
    observed_strata = sorted(target["strata"], key=lambda item: item["generation_job"])
    if observed_strata != expected_strata:
        raise BenchmarkConfigError(
            "Efficiency sensitivity strata differ from the runnable full profile"
        )
    expected_datasets = sum(int(item["replicates"]) for item in expected_strata)
    effective_n = _effective_sample_size(expected_strata)
    full_budget = study["budgets"]["full"]
    if (
        target["expected_independent_datasets"] != expected_datasets
        or full_budget["expected_independent_datasets"] != expected_datasets
        or any(
            item["replicates"] != full_budget["replicates_per_stratum"] for item in expected_strata
        )
        or not math.isclose(
            float(target["expected_effective_sample_size"]),
            effective_n,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise BenchmarkConfigError(
            "Efficiency sensitivity sample-size contract differs from its weighted full panel"
        )

    accuracy_grid = contract["scenario_grid"]["accuracy"]
    runtime_grid = contract["scenario_grid"]["runtime"]
    if float(accuracy_grid["noninferiority_margin"]) != float(
        study["analysis"]["absolute_noninferiority_margin"]
    ) or float(runtime_grid["maximum_ratio"]) != float(
        study["analysis"]["runtime_gate"]["maximum_one_sided_upper_confidence_bound"]
    ):
        raise BenchmarkConfigError("Efficiency sensitivity endpoints differ from study gates")
    for values, label in (
        (accuracy_grid["true_mean_differences"], "Accuracy mean-difference grid"),
        (
            accuracy_grid["within_stratum_standard_deviations"],
            "Accuracy standard-deviation grid",
        ),
        (runtime_grid["true_geometric_mean_ratios"], "Runtime ratio grid"),
        (
            runtime_grid["within_stratum_log_standard_deviations"],
            "Runtime log-standard-deviation grid",
        ),
    ):
        _strictly_increasing([float(value) for value in values], label)
    return contract, evaluate_design_sensitivity(contract)


__all__ = [
    "SENSITIVITY_RELATIVE_PATH",
    "efficiency_profile_strata",
    "evaluate_design_sensitivity",
    "load_design_sensitivity_contract",
]
