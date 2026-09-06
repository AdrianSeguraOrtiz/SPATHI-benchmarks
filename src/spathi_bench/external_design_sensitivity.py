"""Outcome-free power and precision scenarios for the primary external comparison."""

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

CONTRACT_PATH = Path("experiments/v1/statistics/external-comparison-design-sensitivity.json")


def evaluate_external_design_sensitivity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate deterministic normal-location scenarios without outcome data."""

    method = contract["method"]
    target = contract["target_design"]
    n_blocks = int(target["expected_independent_blocks"])
    alpha = float(method["planning_alpha"])
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    power: list[dict[str, float]] = []
    for effect in contract["scenario_grid"]["standardized_paired_effects"]:
        noncentrality = float(effect) * math.sqrt(n_blocks)
        probability = (
            1.0
            - NormalDist().cdf(critical - noncentrality)
            + NormalDist().cdf(-critical - noncentrality)
        )
        power.append(
            {
                "standardized_paired_effect": float(effect),
                "approximate_power": probability,
            }
        )
    precision = [
        {
            "paired_aupr_standard_deviation": float(sd),
            "familywise_95_percent_half_width": (critical * float(sd) / math.sqrt(n_blocks)),
        }
        for sd in contract["scenario_grid"]["paired_aupr_standard_deviations"]
    ]
    payload = {
        "schema_version": "1.0",
        "contract_id": contract["id"],
        "interpretation": contract["interpretation"],
        "independent_blocks": n_blocks,
        "first_holm_two_sided_alpha": alpha,
        "precision_two_sided_alpha": alpha,
        "critical_value": critical,
        "power": power,
        "precision": precision,
    }
    payload["operating_characteristics_sha256"] = canonical_json_sha256(payload)
    return payload


def load_external_design_sensitivity(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the prospective contract against the runnable full plan and profile."""

    root = repo_root.resolve()
    contract = load_json_object(root / CONTRACT_PATH, "external comparison design sensitivity")
    validate_json(
        contract,
        root / "schemas/external-comparison-design-sensitivity.schema.json",
        "external comparison design sensitivity",
    )
    plan = load_json_object(
        resolve_repo_path(root, contract["plan_path"], "external comparison plan"),
        "external comparison plan",
    )
    comparisons = {item["id"]: item for item in plan["statistics"]["comparisons"]}
    comparison = comparisons.get(contract["comparison_id"])
    if not isinstance(comparison, Mapping):
        raise BenchmarkConfigError("External sensitivity comparison is absent from the full plan")
    configurations = list(comparison["configuration_ids"])
    target = contract["target_design"]
    excluded = set(target["implementation_sensitivities_excluded"])
    if excluded & set(configurations):
        raise BenchmarkConfigError(
            "Implementation-sensitivity configurations entered the primary external ranking"
        )
    if (
        comparison["comparison_role"] != "confirmatory"
        or comparison["primary_metric"] != "aupr"
        or comparison["primary_level"] != "directed"
        or comparison["aggregation"]["weighting"] != contract["method"]["weighting"]
        or comparison["expected_blocks"] != target["expected_independent_blocks"]
        or len(configurations) != target["primary_configuration_count"]
    ):
        raise BenchmarkConfigError("External sensitivity differs from the primary comparison")
    family_size = len(configurations) - 1
    method = contract["method"]
    if method["holm_family_size"] != family_size or not math.isclose(
        float(method["planning_alpha"]),
        float(method["familywise_alpha"]) / family_size,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise BenchmarkConfigError("External sensitivity does not use the first Holm threshold")

    # The committed profile is the immutable planned template; materialization adds
    # request bundles without changing its generation panel.
    profile = load_profile(root, contract["profile"])
    jobs = {job["id"]: job for job in profile["generation_jobs"]}
    simulator_blocks: Counter[str] = Counter()
    total = 0
    for job_id in comparison["generation_jobs"]:
        job = jobs[job_id]
        runs = load_json_object(
            resolve_repo_path(root, job["simulator_runs"], f"runs for {job_id}"),
            f"runs for {job_id}",
        )["runs"]
        if len(runs) != 1:
            raise BenchmarkConfigError(f"External comparison job {job_id!r} is not atomic")
        replicas = int(runs[0]["replicates"])
        simulator_blocks[str(runs[0]["simulator_id"])] += replicas
        total += replicas
    if total != target["expected_independent_blocks"] or dict(
        sorted(simulator_blocks.items())
    ) != dict(sorted(target["simulator_blocks"].items())):
        raise BenchmarkConfigError("External sensitivity block counts differ from the full panel")
    effects = [float(value) for value in contract["scenario_grid"]["standardized_paired_effects"]]
    deviations = [
        float(value) for value in contract["scenario_grid"]["paired_aupr_standard_deviations"]
    ]
    if effects != sorted(effects) or len(set(effects)) != len(effects):
        raise BenchmarkConfigError("External sensitivity effect grid must increase strictly")
    if deviations != sorted(deviations) or len(set(deviations)) != len(deviations):
        raise BenchmarkConfigError("External sensitivity precision grid must increase strictly")
    return contract, evaluate_external_design_sensitivity(contract)


__all__ = [
    "CONTRACT_PATH",
    "evaluate_external_design_sensitivity",
    "load_external_design_sensitivity",
]
