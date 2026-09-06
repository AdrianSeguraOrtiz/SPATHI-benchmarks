"""Paired, non-parametric comparisons over independent benchmark datasets."""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import random
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from functools import reduce
from math import factorial, gcd
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .algorithm_seed_contract import algorithm_seeds_by_configuration
from .artifacts import canonical_json_sha256, sha256_file
from .config import BenchmarkConfigError
from .inference_scope_contract import inference_regulator_scopes
from .pipeline import validate_inference_attribution_contract
from .provenance import frozen_profile, validate_run_inputs
from .runtime import benchmark_host_environment
from .safe_io import path_below_no_symlinks
from .seeds import (
    SeedPlanError,
    load_seed_plan,
    plan_seed_record,
    validate_plan_seed_record,
)
from .state import load_state, require_status, write_state
from .summarize import validate_summary_reports

STATISTICS_SCHEMA_VERSION = "1.0"
STATISTICS_FILENAMES = ("statistics.json", "rankings.csv", "pairwise.csv", "report.md")
SUPPORTED_METRICS = (
    "auroc",
    "aupr",
    "f1_at_truth_count",
    "epr_at_truth_count",
    "invariant_edge_stability",
)
STRATUM_FIELDS = (
    "dataset_role",
    "split_id",
    "condition_id",
    "simulator_id",
    "generation_job",
    "regime",
    "level",
)
DATASET_STRATUM_FIELDS = {
    "dataset_role",
    "split_id",
    "condition_id",
    "simulator_id",
    "generation_job",
}
SEMANTIC_UNIVERSE_FIELDS = (
    "dataset_fingerprint_sha256",
    "candidate_source_universe_sha256",
    "candidate_target_universe_sha256",
    "candidate_edge_universe_sha256",
    "truth_edge_universe_sha256",
)
REQUIRED_METRIC_FIELDS = {
    "configuration_id",
    "dataset_id",
    "dataset_role",
    "split_id",
    "condition_id",
    "simulator_id",
    "generation_job",
    "regime",
    "level",
    "run_id",
    "configuration_sha256",
    "algorithm_seed",
    "context",
    "status",
    "catalog_tool_id",
    "family",
    "n_candidates",
    "n_candidate_genes",
    "n_candidate_sources",
    "n_candidate_targets",
    *SEMANTIC_UNIVERSE_FIELDS,
    *SUPPORTED_METRICS,
}
RANKING_FIELDS = (
    "analysis_id",
    "comparison_id",
    "comparison_role",
    "candidate_space",
    "inference_regulator_scope",
    *STRATUM_FIELDS,
    "metric",
    "analysis_role",
    "status",
    "n_expected_blocks",
    "n_complete_blocks",
    "configuration_id",
    "configuration_sha256",
    "tool_id",
    "mean_score",
    "median_score",
    "mean_rank",
)
PAIRWISE_FIELDS = (
    "analysis_id",
    "comparison_id",
    "comparison_role",
    "candidate_space",
    "control_inference_regulator_scope",
    "competitor_inference_regulator_scope",
    *STRATUM_FIELDS,
    "metric",
    "analysis_role",
    "status",
    "multiplicity_family",
    "control_configuration_id",
    "control_configuration_sha256",
    "control_tool_id",
    "competitor_configuration_id",
    "competitor_configuration_sha256",
    "competitor_tool_id",
    "n_blocks",
    "n_nonzero",
    "wins",
    "ties",
    "losses",
    "probability_of_superiority",
    "mean_difference",
    "median_difference",
    "hodges_lehmann",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_resamples",
    "bootstrap_random_seed",
    "rank_biserial",
    "wilcoxon_w",
    "p_raw",
    "p_holm",
    "reject",
    "exact_minimum_attainable_p",
    "monte_carlo_resolution",
    "method",
    "bootstrap_confidence_level",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_value(metric: str, value: Any, *, label: str) -> float | None:
    number = _finite_number(value)
    if number is None:
        return None
    tolerance = 1e-12
    if metric in {"auroc", "aupr", "f1_at_truth_count", "invariant_edge_stability"}:
        if number < -tolerance or number > 1.0 + tolerance:
            raise BenchmarkConfigError(
                f"Metric {metric} is outside its [0, 1] domain for {label}: {number}"
            )
        return min(1.0, max(0.0, number))
    if metric == "epr_at_truth_count":
        if number < -tolerance:
            raise BenchmarkConfigError(
                f"Metric {metric} is outside its non-negative domain for {label}: {number}"
            )
        return max(0.0, number)
    raise BenchmarkConfigError(f"Unsupported statistical metric: {metric}")


def _nonnegative_integer(value: Any, *, label: str) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError(f"{label} must be a non-negative integer") from exc
    if number < 0 or str(number) != str(value).strip():
        raise BenchmarkConfigError(f"{label} must be a non-negative integer")
    return number


def _algorithm_seed(value: Any, *, label: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return _nonnegative_integer(value, label=label)


def _sha256_digest(value: Any, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise BenchmarkConfigError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _analysis_seed_record(
    seed_plan: dict[str, Any],
    *,
    namespace: str,
    purpose: str,
    analysis_id: str,
    control_configuration_id: str | None = None,
    competitor_configuration_id: str | None = None,
) -> dict[str, Any]:
    coordinates: dict[str, str] = {
        "purpose": purpose,
        "analysis_id": analysis_id,
    }
    if control_configuration_id is not None:
        coordinates["control_configuration_id"] = control_configuration_id
    if competitor_configuration_id is not None:
        coordinates["competitor_configuration_id"] = competitor_configuration_id
    return plan_seed_record(
        seed_plan,
        namespace=namespace,
        stream="statistical",
        coordinates=coordinates,
    )


def _midranks(values: Sequence[float], *, descending: bool) -> tuple[list[float], int]:
    indexed = sorted(enumerate(values), key=lambda item: item[1], reverse=descending)
    ranks = [0.0] * len(values)
    tie_sum = 0
    start = 0
    while start < len(indexed):
        stop = start + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[indexed[position][0]] = rank
        size = stop - start
        if size > 1:
            tie_sum += size**3 - size
        start = stop
    return ranks, tie_sum


def _friedman_test(
    matrix: Sequence[Sequence[float]],
    *,
    permutations: int,
    random_seed: int,
    exact_limit: int = 1_000_000,
) -> dict[str, Any]:
    """Friedman test with tie correction and an exact/permutation null distribution."""

    if not matrix:
        raise ValueError("Friedman requires at least one block")
    n_blocks = len(matrix)
    n_tools = len(matrix[0])
    if n_tools < 3 or any(len(row) != n_tools for row in matrix):
        raise ValueError("Friedman requires a rectangular matrix with at least three tools")
    rank_rows: list[tuple[int, ...]] = []
    tie_sum = 0
    for row in matrix:
        if any(not math.isfinite(value) for value in row):
            raise ValueError("Friedman scores must be finite")
        ranks, row_ties = _midranks(row, descending=True)
        rank_rows.append(tuple(int(round(rank * 2)) for rank in ranks))
        tie_sum += row_ties

    rank_sums2 = [sum(row[column] for row in rank_rows) for column in range(n_tools)]
    expected_sum2 = n_blocks * (n_tools + 1)
    observed_d2 = sum((value - expected_sum2) ** 2 for value in rank_sums2)
    denominator = n_blocks * (n_tools**3 - n_tools) - tie_sum
    if denominator == 0:
        statistic = 0.0
        kendalls_w = 0.0
        return {
            "status": "all_tied",
            "statistic": statistic,
            "p_raw": 1.0,
            "p_holm": None,
            "reject": False,
            "kendalls_w": kendalls_w,
            "tie_correction": 0.0,
            "method": "degenerate_all_tied",
            "permutations": 1,
            "random_seed": None,
            "monte_carlo_standard_error": None,
        }

    statistic = 3.0 * (n_tools - 1) * observed_d2 / denominator
    kendalls_w = max(0.0, min(1.0, statistic / (n_blocks * (n_tools - 1))))
    tie_correction = denominator / (n_blocks * (n_tools**3 - n_tools))
    total_assignments = factorial(n_tools) ** n_blocks
    if total_assignments <= exact_limit:
        extreme = 0
        row_permutations = [tuple(itertools.permutations(row)) for row in rank_rows]
        for permuted_rows in itertools.product(*row_permutations):
            sums = [sum(row[column] for row in permuted_rows) for column in range(n_tools)]
            permuted_d2 = sum((value - expected_sum2) ** 2 for value in sums)
            if permuted_d2 >= observed_d2:
                extreme += 1
        p_value = extreme / total_assignments
        method = "exact_within_block_label_permutation"
        used_seed: int | None = None
        used_permutations = total_assignments
        standard_error: float | None = None
    else:
        rng = random.Random(random_seed)
        extreme = 0
        for _ in range(permutations):
            sums = [0] * n_tools
            for row in rank_rows:
                shuffled = list(row)
                rng.shuffle(shuffled)
                for column, value in enumerate(shuffled):
                    sums[column] += value
            permuted_d2 = sum((value - expected_sum2) ** 2 for value in sums)
            if permuted_d2 >= observed_d2:
                extreme += 1
        p_value = (extreme + 1) / (permutations + 1)
        method = "monte_carlo_within_block_label_permutation"
        used_seed = random_seed
        used_permutations = permutations
        standard_error = math.sqrt(p_value * (1.0 - p_value) / (permutations + 1))

    return {
        "status": "tested",
        "statistic": statistic,
        "p_raw": p_value,
        "p_holm": None,
        "reject": False,
        "kendalls_w": kendalls_w,
        "tie_correction": tie_correction,
        "method": method,
        "permutations": used_permutations,
        "random_seed": used_seed,
        "monte_carlo_standard_error": standard_error,
    }


def _hodges_lehmann(differences: Sequence[float]) -> float | None:
    if not differences:
        return None
    walsh = [
        (differences[left] + differences[right]) / 2.0
        for left in range(len(differences))
        for right in range(left, len(differences))
    ]
    return float(median(walsh))


def _paired_descriptives(differences: Sequence[float]) -> dict[str, Any]:
    """Return paired point summaries without invoking an inferential procedure."""

    if any(not math.isfinite(value) for value in differences):
        raise ValueError("Paired differences must be finite")
    wins = sum(value > 0 for value in differences)
    ties = sum(value == 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    n_total = len(differences)
    return {
        "n_blocks": n_total,
        "n_nonzero": n_total - ties,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "probability_of_superiority": ((wins + 0.5 * ties) / n_total if n_total else None),
        "mean_difference": fmean(differences) if differences else None,
        "median_difference": float(median(differences)) if differences else None,
        "hodges_lehmann": _hodges_lehmann(differences),
    }


def _wilcoxon_test(
    differences: Sequence[float],
    *,
    permutations: int,
    random_seed: int,
    exact_state_limit: int = 5_000_000,
) -> dict[str, Any]:
    """Two-sided paired signed-rank test using an exact sign-flip distribution."""

    descriptive = _paired_descriptives(differences)
    nonzero = [value for value in differences if value != 0]
    if not differences:
        return {
            **descriptive,
            "status": "no_pairs",
            "wilcoxon_w": None,
            "rank_biserial": None,
            "p_raw": None,
            "exact_minimum_attainable_p": None,
            "monte_carlo_resolution": None,
            "method": None,
            "permutations": 0,
            "random_seed": None,
            "monte_carlo_standard_error": None,
        }
    if not nonzero:
        return {
            **descriptive,
            "status": "all_zero_differences",
            "wilcoxon_w": 0.0,
            "rank_biserial": 0.0,
            "p_raw": 1.0,
            "exact_minimum_attainable_p": 1.0,
            "monte_carlo_resolution": None,
            "method": "degenerate_all_zero_differences",
            "permutations": 1,
            "random_seed": None,
            "monte_carlo_standard_error": None,
        }

    absolute = [abs(value) for value in nonzero]
    ranks, _ties = _midranks(absolute, descending=False)
    ranks2 = [int(round(rank * 2)) for rank in ranks]
    total_rank2 = sum(ranks2)
    positive_rank2 = sum(rank for rank, value in zip(ranks2, nonzero, strict=True) if value > 0)
    negative_rank2 = total_rank2 - positive_rank2
    observed_deviation = abs(2 * positive_rank2 - total_rank2)
    divisor = reduce(gcd, ranks2)
    scaled = [rank // divisor for rank in ranks2]
    scaled_total = sum(scaled)
    scaled_observed = observed_deviation // divisor
    exact_work = len(scaled) * scaled_total
    if exact_work <= exact_state_limit:
        counts = [0] * (scaled_total + 1)
        counts[0] = 1
        reachable = 0
        for rank in scaled:
            for subtotal in range(reachable, -1, -1):
                count = counts[subtotal]
                if count:
                    counts[subtotal + rank] += count
            reachable += rank
        extreme = sum(
            count
            for subtotal, count in enumerate(counts)
            if abs(2 * subtotal - scaled_total) >= scaled_observed
        )
        total_assignments = 2 ** len(scaled)
        p_value = extreme / total_assignments
        method = "exact_conditional_signed_rank_permutation"
        used_permutations = total_assignments
        used_seed: int | None = None
        standard_error: float | None = None
    else:
        rng = random.Random(random_seed)
        extreme = 0
        for _ in range(permutations):
            permuted_positive = sum(rank for rank in scaled if rng.getrandbits(1))
            if abs(2 * permuted_positive - scaled_total) >= scaled_observed:
                extreme += 1
        p_value = (extreme + 1) / (permutations + 1)
        method = "monte_carlo_conditional_signed_rank_permutation"
        used_permutations = permutations
        used_seed = random_seed
        standard_error = math.sqrt(p_value * (1.0 - p_value) / (permutations + 1))

    return {
        **descriptive,
        "status": "tested",
        "wilcoxon_w": min(positive_rank2, negative_rank2) / 2.0,
        "rank_biserial": (positive_rank2 - negative_rank2) / total_rank2,
        "p_raw": p_value,
        "exact_minimum_attainable_p": min(1.0, 2.0 / (2 ** len(nonzero))),
        "monte_carlo_resolution": (1.0 / (permutations + 1) if used_seed is not None else None),
        "method": method,
        "permutations": used_permutations,
        "random_seed": used_seed,
        "monte_carlo_standard_error": standard_error,
    }


def _holm_adjust(records: Sequence[dict[str, Any]], *, alpha: float) -> None:
    """Apply Holm step-down adjustment, retaining null p-values in family size."""

    ordered = sorted(
        records,
        key=lambda record: (
            1.0 if record.get("p_raw") is None else float(record["p_raw"]),
            str(record.get("contrast_id", record.get("analysis_id", ""))),
        ),
    )
    running = 0.0
    total = len(ordered)
    for position, record in enumerate(ordered):
        raw = record.get("p_raw")
        effective = 1.0 if raw is None else float(raw)
        running = min(1.0, max(running, (total - position) * effective))
        record["p_holm"] = None if raw is None else running
        record["reject"] = bool(raw is not None and running <= alpha)


def _bootstrap_median_interval(
    differences: Sequence[float],
    design_strata: Sequence[tuple[str, str]],
    *,
    resamples: int,
    random_seed: int,
    alpha: float,
) -> tuple[float, float] | None:
    if len(differences) < 2:
        return None
    by_stratum: dict[tuple[str, str], list[float]] = defaultdict(list)
    for difference, design_stratum in zip(differences, design_strata, strict=True):
        by_stratum[design_stratum].append(difference)
    rng = random.Random(random_seed)
    samples: list[float] = []
    for _ in range(resamples):
        draw: list[float] = []
        for design_stratum in sorted(by_stratum):
            values = by_stratum[design_stratum]
            draw.extend(values[rng.randrange(len(values))] for _ in values)
        samples.append(float(median(draw)))
    return _percentile(samples, alpha / 2.0), _percentile(samples, 1.0 - alpha / 2.0)


def _tool_metadata(rows: Sequence[dict[str, str]], configuration_id: str) -> dict[str, Any]:
    members = [row for row in rows if row["configuration_id"] == configuration_id]
    families = {row.get("family", "") for row in members}
    selection_roles = {row.get("selection_role", "") for row in members}
    catalog_ids = {row.get("catalog_tool_id", "") for row in members}
    configuration_hashes = {row.get("configuration_sha256", "") for row in members}
    if len(families) > 1 or len(selection_roles) > 1 or len(configuration_hashes) != 1:
        raise BenchmarkConfigError(
            f"Configuration {configuration_id!r} has inconsistent family, selection or "
            "parameter identity metadata"
        )
    configuration_sha256 = next(iter(configuration_hashes), "")
    if len(configuration_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in configuration_sha256
    ):
        raise BenchmarkConfigError(
            f"Configuration {configuration_id!r} has an invalid configuration_sha256"
        )
    family = next(iter(families), "")
    if family == "spathi":
        tool_id = "SPATHI"
    elif len(catalog_ids) == 1 and next(iter(catalog_ids)):
        tool_id = next(iter(catalog_ids))
    else:
        tool_id = configuration_id
    return {
        "configuration_id": configuration_id,
        "tool_id": tool_id,
        "family": family or None,
        "selection_role": next(iter(selection_roles), None),
        "configuration_sha256": configuration_sha256,
    }


def _read_metrics(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = sorted(REQUIRED_METRIC_FIELDS - fields)
            if missing:
                raise BenchmarkConfigError(
                    f"Statistical analysis metrics.csv lacks required fields: {missing}"
                )
            rows = list(reader)
    except OSError as exc:
        raise BenchmarkConfigError(f"Cannot read statistical metrics: {exc}") from exc
    if not rows:
        raise BenchmarkConfigError("Statistical analysis metrics.csv is empty")
    return rows


def _validate_statistics_contract(
    profile: dict[str, Any], seed_plan: dict[str, Any]
) -> dict[str, Any]:
    contract = profile.get("statistics")
    if not isinstance(contract, dict) or contract.get("status") not in {
        "inferential",
        "descriptive",
    }:
        raise BenchmarkConfigError(
            f"Runnable profile {profile.get('id')!r} has no analysis contract"
        )
    inferential = contract["status"] == "inferential"
    inferential_fields = {
        "alpha",
        "random_seed",
        "seed_derivation",
        "permutations",
        "bootstrap_resamples",
    }
    if inferential:
        missing = sorted(inferential_fields - set(contract))
        if missing:
            raise BenchmarkConfigError(
                f"Inferential analysis contract omits required fields: {missing}"
            )
        try:
            validate_plan_seed_record(
                seed_plan,
                seed=contract["random_seed"],
                seed_derivation=contract["seed_derivation"],
                required_namespace="full.test",
                required_stream="statistical",
                required_coordinates={"purpose": "phase_root"},
            )
        except SeedPlanError as exc:
            raise BenchmarkConfigError(f"Invalid statistical seed contract: {exc}") from exc
    else:
        unexpected = sorted(inferential_fields & set(contract))
        if unexpected:
            raise BenchmarkConfigError(
                f"Descriptive analysis contract contains inferential fields: {unexpected}"
            )
    for comparison in contract["comparisons"]:
        inferential_comparison_fields = {"posthoc_scope", "correction"}
        observed_inferential_fields = inferential_comparison_fields & set(comparison)
        if inferential and observed_inferential_fields != inferential_comparison_fields:
            raise BenchmarkConfigError(
                f"Inferential comparison {comparison.get('id')!r} has no complete "
                "multiplicity contract"
            )
        if not inferential and observed_inferential_fields:
            raise BenchmarkConfigError(
                f"Descriptive comparison {comparison.get('id')!r} contains inferential "
                f"fields: {sorted(observed_inferential_fields)}"
            )
        if not inferential and comparison.get("comparison_role") != "exploratory":
            raise BenchmarkConfigError(
                f"Descriptive comparison {comparison.get('id')!r} cannot claim a confirmatory role"
            )
        if comparison.get("aggregation") != {
            "block_unit": "dataset_id",
            "hierarchy": ["simulator_id", "condition_id", "dataset_id"],
            "weighting": "equal_dataset",
        }:
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']!r} has an unsupported aggregation"
            )
        analysis_kind = comparison.get("analysis_kind")
        truth_scope = comparison.get("truth_scope")
        if analysis_kind == "invariant_network_stability":
            if (
                truth_scope != "invariant"
                or comparison["comparison_role"] != "exploratory"
                or comparison["primary_metric"] != "invariant_edge_stability"
            ):
                raise BenchmarkConfigError(
                    f"Invariant comparison {comparison['id']!r} has inconsistent semantics"
                )
        elif analysis_kind != "network_accuracy":
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']!r} has an unknown analysis kind"
            )
        configurations = comparison["configuration_ids"]
        if comparison["control_configuration_id"] not in configurations:
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']!r} control is not a configuration"
            )
        if comparison["primary_metric"] not in comparison["metrics"]:
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']!r} primary metric is not selected"
            )
        if comparison["primary_level"] not in comparison["levels"]:
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']!r} primary level is not selected"
            )
        if comparison["comparison_role"] == "confirmatory" and (
            comparison["primary_metric"] != "aupr" or comparison["primary_level"] != "directed"
        ):
            raise BenchmarkConfigError(
                f"Confirmatory comparison {comparison['id']!r} must use directed AUPR "
                "as its primary endpoint"
            )
        stratify_by = comparison["stratify_by"]
        required_strata = {"dataset_role", "regime", "level"}
        if not required_strata.issubset(stratify_by):
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']!r} must stratify by "
                "dataset_role, regime and level"
            )
    return contract


def _expected_strata(state: dict[str, Any], comparison: dict[str, Any]) -> list[dict[str, str]]:
    strata: set[tuple[tuple[str, str], ...]] = set()
    for dataset in state.get("datasets", []):
        if not _comparison_selects_dataset(comparison, dataset):
            continue
        for level in comparison["levels"]:
            values: dict[str, str] = {}
            for field in comparison["stratify_by"]:
                if field == "regime":
                    raw: Any = comparison["regime"]
                elif field == "level":
                    raw = level
                else:
                    raw = dataset.get(field)
                if raw is None or str(raw) == "":
                    raise BenchmarkConfigError(
                        f"Dataset {dataset.get('dataset_id')!r} lacks statistics stratum {field!r}"
                    )
                values[field] = str(raw)
            strata.add(tuple(sorted(values.items())))
    return [dict(items) for items in sorted(strata)]


def _matches_stratum(row: dict[str, str], stratum: dict[str, str]) -> bool:
    return all(row.get(field) == value for field, value in stratum.items())


def _comparison_selects_dataset(comparison: dict[str, Any], dataset: dict[str, Any]) -> bool:
    if dataset.get("dataset_role") not in comparison["dataset_roles"]:
        return False
    selected_jobs = comparison.get("generation_jobs")
    return selected_jobs is None or dataset.get("generation_job") in selected_jobs


def _expected_dataset_ids(
    state: dict[str, Any], comparison: dict[str, Any], stratum: dict[str, str]
) -> list[str]:
    selected: list[dict[str, Any]] = []
    for dataset in state.get("datasets", []):
        if not _comparison_selects_dataset(comparison, dataset):
            continue
        if all(
            field not in DATASET_STRATUM_FIELDS or str(dataset.get(field)) == value
            for field, value in stratum.items()
        ):
            selected.append(dataset)
    result = [str(dataset["dataset_id"]) for dataset in selected]
    if len(set(result)) != len(result):
        raise BenchmarkConfigError("Benchmark state contains duplicate dataset IDs")
    independent_units: dict[tuple[str, str], str] = {}
    for dataset in selected:
        try:
            unit = (
                str(dataset["simulator_id"]),
                str(dataset["seed"]),
            )
        except KeyError as exc:
            raise BenchmarkConfigError(
                f"Dataset {dataset.get('dataset_id')!r} lacks independent-unit provenance"
            ) from exc
        previous = independent_units.get(unit)
        if previous is not None:
            raise BenchmarkConfigError(
                "Duplicate independent simulation unit in one statistical stratum "
                f"{unit}: {previous!r} and {dataset['dataset_id']!r}"
            )
        independent_units[unit] = str(dataset["dataset_id"])
    return sorted(result)


def _extract_panel(
    rows: Sequence[dict[str, str]],
    *,
    state: dict[str, Any],
    comparison: dict[str, Any],
    stratum: dict[str, str],
    metric: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configurations = list(comparison["configuration_ids"])
    expected_seeds_by_configuration = algorithm_seeds_by_configuration(comparison)
    selected = [
        row
        for row in rows
        if row["configuration_id"] in configurations
        and _comparison_selects_dataset(comparison, row)
        and row["regime"] == comparison["regime"]
        and row["level"] in comparison["levels"]
        and row["context"].startswith("group:")
        and _matches_stratum(row, stratum)
    ]
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_dataset[row["dataset_id"]].append(row)

    blocks: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for dataset_id in _expected_dataset_ids(state, comparison, stratum):
        members = by_dataset.get(dataset_id, [])
        reasons: list[str] = []
        context_sets: list[tuple[str, str, set[str]]] = []
        scores: dict[str, float] = {}
        execution_counts: dict[str, int] = {}
        seeds_by_configuration: dict[str, set[int | None]] = {}
        candidate_universes: dict[str, set[tuple[int, int, int, int]]] = defaultdict(set)
        semantic_universes: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        simulator_ids = {row["simulator_id"] for row in members}
        if len(simulator_ids) > 1:
            raise BenchmarkConfigError(
                f"Dataset {dataset_id!r} has inconsistent simulator provenance"
            )
        simulator_id = next(iter(simulator_ids), "")
        if not simulator_id:
            state_dataset = next(
                item for item in state["datasets"] if str(item["dataset_id"]) == dataset_id
            )
            simulator_id = str(state_dataset["simulator_id"])

        for configuration_id in configurations:
            configuration_rows = [
                row for row in members if row["configuration_id"] == configuration_id
            ]
            if not configuration_rows:
                reasons.append(f"missing_configuration:{configuration_id}")
                continue
            by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in configuration_rows:
                by_run[row["run_id"]].append(row)
            execution_values: list[float] = []
            observed_seeds: dict[int | None, str] = {}
            for run_id, run_rows in sorted(by_run.items()):
                seed_values = {
                    _algorithm_seed(
                        row["algorithm_seed"],
                        label=f"Algorithm seed for {configuration_id}/{run_id}",
                    )
                    for row in run_rows
                }
                if len(seed_values) != 1:
                    raise BenchmarkConfigError(
                        f"Run {configuration_id}/{run_id} mixes algorithm seeds"
                    )
                algorithm_seed = next(iter(seed_values))
                previous_run = observed_seeds.get(algorithm_seed)
                if previous_run is not None:
                    reasons.append(
                        f"duplicate_algorithm_seed:{configuration_id}:"
                        f"{algorithm_seed}:{previous_run},{run_id}"
                    )
                observed_seeds[algorithm_seed] = run_id
                by_context: dict[str, list[dict[str, str]]] = defaultdict(list)
                for row in run_rows:
                    by_context[row["context"]].append(row)
                duplicate_contexts = sorted(
                    context
                    for context, context_rows in by_context.items()
                    if len(context_rows) != 1
                )
                if duplicate_contexts:
                    reasons.append(
                        f"duplicate_contexts:{configuration_id}:{run_id}:"
                        + ",".join(duplicate_contexts)
                    )
                    continue
                contexts = set(by_context)
                context_sets.append((configuration_id, run_id, contexts))
                values: list[float] = []
                for context, context_rows in sorted(by_context.items()):
                    row = context_rows[0]
                    value = _metric_value(
                        metric,
                        row.get(metric),
                        label=f"{configuration_id}/{run_id}/{context}",
                    )
                    if row.get("status") not in {"ok", "partial"} or value is None:
                        reasons.append(
                            f"non_evaluable:{configuration_id}:{run_id}:{context}:"
                            f"{row.get('status') or 'missing_status'}"
                        )
                    else:
                        values.append(value)
                        candidate_universes[context].add(
                            tuple(
                                _nonnegative_integer(
                                    row[field],
                                    label=f"{field} for {configuration_id}/{run_id}/{context}",
                                )
                                for field in (
                                    "n_candidates",
                                    "n_candidate_genes",
                                    "n_candidate_sources",
                                    "n_candidate_targets",
                                )
                            )
                        )
                        semantic_universes[context].add(
                            tuple(
                                _sha256_digest(
                                    row[field],
                                    label=(f"{field} for {configuration_id}/{run_id}/{context}"),
                                )
                                for field in SEMANTIC_UNIVERSE_FIELDS
                            )
                        )
                if len(values) == len(contexts) and values:
                    execution_values.append(fmean(values))
            seeds_by_configuration[configuration_id] = set(observed_seeds)
            execution_counts[configuration_id] = len(by_run)
            expected_seeds = set(expected_seeds_by_configuration[configuration_id])
            if set(observed_seeds) != expected_seeds:
                reasons.append(
                    f"algorithm_seed_contract:{configuration_id}:"
                    f"expected={sorted(expected_seeds)},"
                    "observed="
                    f"{sorted(observed_seeds, key=lambda seed: -1 if seed is None else seed)}"
                )
            if len(execution_values) == len(by_run) and execution_values:
                scores[configuration_id] = fmean(execution_values)

        unique_context_sets = {frozenset(contexts) for _, _, contexts in context_sets}
        if len(unique_context_sets) > 1:
            rendered = ";".join(
                f"{configuration_id}/{run_id}={','.join(sorted(contexts))}"
                for configuration_id, run_id, contexts in context_sets
            )
            reasons.append(f"unbalanced_contexts:{rendered}")
        if context_sets and not next(iter(unique_context_sets), frozenset()):
            reasons.append("no_group_contexts")
        if set(scores) != set(configurations):
            reasons.append("incomplete_configuration_scores")
        inconsistent_universes = {
            context: universes
            for context, universes in candidate_universes.items()
            if len(universes) > 1
        }
        if inconsistent_universes:
            rendered = ";".join(
                f"{context}={sorted(universes)}"
                for context, universes in sorted(inconsistent_universes.items())
            )
            reasons.append(f"unbalanced_candidate_universe:{rendered}")
        inconsistent_semantic_universes = {
            context: universes
            for context, universes in semantic_universes.items()
            if len(universes) > 1
        }
        if inconsistent_semantic_universes:
            rendered = ";".join(
                f"{context}={sorted(universes)}"
                for context, universes in sorted(inconsistent_semantic_universes.items())
            )
            reasons.append(f"unbalanced_semantic_universe:{rendered}")
        if reasons:
            excluded.append(
                {
                    "dataset_id": dataset_id,
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        contexts = next(iter(unique_context_sets))
        blocks.append(
            {
                "dataset_id": dataset_id,
                "simulator_id": simulator_id,
                "condition_id": str(
                    next(
                        item["condition_id"]
                        for item in state["datasets"]
                        if str(item["dataset_id"]) == dataset_id
                    )
                ),
                "scores": scores,
                "n_contexts": len(contexts),
                "contexts": sorted(contexts),
                "executions_per_configuration": execution_counts,
                "algorithm_seeds": {
                    configuration_id: sorted(seeds, key=lambda seed: -1 if seed is None else seed)
                    for configuration_id, seeds in sorted(seeds_by_configuration.items())
                },
                "expected_algorithm_seeds": {
                    configuration_id: list(seeds)
                    for configuration_id, seeds in expected_seeds_by_configuration.items()
                },
                "candidate_universe_by_context": {
                    context: dict(
                        zip(
                            (
                                "n_candidates",
                                "n_candidate_genes",
                                "n_candidate_sources",
                                "n_candidate_targets",
                            ),
                            next(iter(universes)),
                            strict=True,
                        )
                    )
                    for context, universes in sorted(candidate_universes.items())
                },
                "semantic_universe_by_context": {
                    context: dict(
                        zip(
                            SEMANTIC_UNIVERSE_FIELDS,
                            next(iter(universes)),
                            strict=True,
                        )
                    )
                    for context, universes in sorted(semantic_universes.items())
                },
            }
        )
    return blocks, excluded


def _analysis_id(comparison_id: str, stratum: dict[str, str], metric: str) -> str:
    digest = canonical_json_sha256(
        {"comparison_id": comparison_id, "stratum": stratum, "metric": metric}
    )[:12]
    return f"{comparison_id}__{metric}__{digest}"


def _analyze_one(
    rows: Sequence[dict[str, str]],
    *,
    state: dict[str, Any],
    contract: dict[str, Any],
    comparison: dict[str, Any],
    stratum: dict[str, str],
    metric: str,
    seed_plan: dict[str, Any],
    statistical_namespace: str | None,
) -> dict[str, Any]:
    configurations = list(comparison["configuration_ids"])
    regulator_scopes = inference_regulator_scopes(comparison)
    metadata = {
        configuration_id: _tool_metadata(rows, configuration_id)
        for configuration_id in configurations
    }
    blocks, excluded = _extract_panel(
        rows,
        state=state,
        comparison=comparison,
        stratum=stratum,
        metric=metric,
    )
    analysis_id = _analysis_id(comparison["id"], stratum, metric)
    is_primary = (
        state.get("purpose") == "benchmark"
        and stratum.get("dataset_role") == "test"
        and comparison["comparison_role"] == "confirmatory"
        and metric == comparison["primary_metric"]
        and stratum.get("level") == comparison["primary_level"]
    )
    matrix = [
        [block["scores"][configuration_id] for configuration_id in configurations]
        for block in blocks
    ]
    ranks_by_block = [_midranks(row, descending=True)[0] for row in matrix]
    rankings = []
    for index, configuration_id in enumerate(configurations):
        values = [row[index] for row in matrix]
        ranks = [row[index] for row in ranks_by_block]
        rankings.append(
            {
                **metadata[configuration_id],
                "inference_regulator_scope": regulator_scopes[configuration_id],
                "mean_score": fmean(values) if values else None,
                "median_score": float(median(values)) if values else None,
                "mean_rank": fmean(ranks) if ranks else None,
            }
        )
    rankings.sort(
        key=lambda item: (
            math.inf if item["mean_rank"] is None else item["mean_rank"],
            item["configuration_id"],
        )
    )

    n_expected = len(blocks) + len(excluded)
    n_complete = len(blocks)
    declared_blocks = int(comparison["expected_blocks"])
    if n_expected != declared_blocks:
        raise BenchmarkConfigError(
            f"Comparison {comparison['id']!r} declares {declared_blocks} expected blocks "
            f"but its frozen dataset panel contains {n_expected}."
        )
    if excluded:
        status = "incomplete_panel"
        reason = "At least one predeclared dataset lacks a complete paired panel."
    elif contract["status"] == "descriptive" and state.get("purpose") == "smoke":
        status = "contract_only"
        reason = "The smoke profile verifies the software contract; it makes no accuracy claim."
    elif contract["status"] == "descriptive":
        status = "descriptive_only"
        reason = (
            "This descriptive profile does not "
            "authorize confidence intervals, hypothesis tests or winner claims."
        )
    else:
        status = "tested"
        reason = None

    friedman_seed = None
    if status == "tested" and len(configurations) >= 3:
        assert statistical_namespace is not None
        friedman_seed = _analysis_seed_record(
            seed_plan,
            namespace=statistical_namespace,
            purpose="friedman_permutation",
            analysis_id=analysis_id,
        )
        omnibus = _friedman_test(
            matrix,
            permutations=int(contract["permutations"]),
            random_seed=friedman_seed["seed"],
        )
    elif len(configurations) < 3:
        omnibus = {
            "status": "not_applicable_two_tools",
            "statistic": None,
            "p_raw": None,
            "p_holm": None,
            "reject": False,
            "kendalls_w": None,
            "tie_correction": None,
            "method": None,
            "permutations": 0,
            "random_seed": None,
            "monte_carlo_standard_error": None,
        }
    else:
        omnibus = {
            "status": status,
            "statistic": None,
            "p_raw": None,
            "p_holm": None,
            "reject": False,
            "kendalls_w": None,
            "tie_correction": None,
            "method": None,
            "permutations": 0,
            "random_seed": None,
            "monte_carlo_standard_error": None,
        }
    omnibus["seed_record"] = friedman_seed

    control = comparison["control_configuration_id"]
    control_index = configurations.index(control)
    pairwise: list[dict[str, Any]] = []
    for competitor in configurations:
        if competitor == control:
            continue
        competitor_index = configurations.index(competitor)
        differences = [row[control_index] - row[competitor_index] for row in matrix]
        if status == "tested":
            assert statistical_namespace is not None
            wilcoxon_seed = _analysis_seed_record(
                seed_plan,
                namespace=statistical_namespace,
                purpose="wilcoxon_permutation",
                analysis_id=analysis_id,
                control_configuration_id=control,
                competitor_configuration_id=competitor,
            )
            result = _wilcoxon_test(
                differences,
                permutations=int(contract["permutations"]),
                random_seed=wilcoxon_seed["seed"],
            )
            bootstrap_seed = _analysis_seed_record(
                seed_plan,
                namespace=statistical_namespace,
                purpose="paired_bootstrap",
                analysis_id=analysis_id,
                control_configuration_id=control,
                competitor_configuration_id=competitor,
            )
            interval = _bootstrap_median_interval(
                differences,
                [(str(block["simulator_id"]), str(block["condition_id"])) for block in blocks],
                resamples=int(contract["bootstrap_resamples"]),
                random_seed=bootstrap_seed["seed"],
                alpha=float(contract["alpha"]),
            )
        else:
            wilcoxon_seed = None
            bootstrap_seed = None
            result = {
                **_paired_descriptives(differences),
                "status": status,
                "wilcoxon_w": None,
                "rank_biserial": None,
                "p_raw": None,
                "exact_minimum_attainable_p": None,
                "monte_carlo_resolution": None,
                "method": None,
                "permutations": 0,
                "random_seed": None,
                "monte_carlo_standard_error": None,
            }
            interval = None
        pairwise.append(
            {
                "contrast_id": f"{analysis_id}::{control}::{competitor}",
                "status": result.pop("status"),
                "control_configuration_id": control,
                "control_configuration_sha256": metadata[control]["configuration_sha256"],
                "control_tool_id": metadata[control]["tool_id"],
                "control_inference_regulator_scope": regulator_scopes[control],
                "competitor_configuration_id": competitor,
                "competitor_configuration_sha256": metadata[competitor]["configuration_sha256"],
                "competitor_tool_id": metadata[competitor]["tool_id"],
                "competitor_inference_regulator_scope": regulator_scopes[competitor],
                **result,
                "bootstrap_ci_low": interval[0] if interval else None,
                "bootstrap_ci_high": interval[1] if interval else None,
                "bootstrap_method": (
                    "paired_percentile_bootstrap_fixed_simulator_condition_strata"
                    if interval
                    else None
                ),
                "bootstrap_resamples": (int(contract["bootstrap_resamples"]) if interval else 0),
                "permutation_seed_record": wilcoxon_seed,
                "bootstrap_random_seed": bootstrap_seed["seed"] if interval else None,
                "bootstrap_seed_record": bootstrap_seed if interval else None,
                "bootstrap_confidence_level": (
                    1.0 - float(contract["alpha"]) if interval else None
                ),
                "p_holm": None,
                "reject": False,
                "multiplicity_family": None,
            }
        )

    return {
        "analysis_id": analysis_id,
        "comparison_id": comparison["id"],
        "comparison_role": comparison["comparison_role"],
        "analysis_kind": comparison["analysis_kind"],
        "truth_scope": comparison["truth_scope"],
        "aggregation": comparison["aggregation"],
        "candidate_space": comparison["candidate_space"],
        "inference_regulator_scope_by_configuration": regulator_scopes,
        "stratum": stratum,
        "metric": metric,
        "analysis_role": (
            "confirmatory"
            if is_primary
            else "contract_check"
            if state.get("purpose") == "smoke"
            else "descriptive"
            if contract["status"] == "descriptive"
            else "exploratory"
        ),
        "higher_is_better": True,
        "status": status,
        "reason": reason,
        "n_expected_blocks": n_expected,
        "n_complete_blocks": n_complete,
        "n_tools": len(configurations),
        "configurations": [metadata[item] for item in configurations],
        "blocks": blocks,
        "excluded_blocks": excluded,
        "rankings": rankings,
        "omnibus": omnibus,
        "pairwise": pairwise,
        "conclusion": None,
    }


def _apply_multiplicity(analyses: list[dict[str, Any]], *, alpha: float) -> None:
    primary_by_comparison: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for analysis in analyses:
        if analysis["analysis_role"] == "confirmatory":
            primary_by_comparison[analysis["comparison_id"]].append(analysis)

    for comparison_id, primary in primary_by_comparison.items():
        omnibus_records = [
            {
                "analysis_id": analysis["analysis_id"],
                "p_raw": analysis["omnibus"]["p_raw"],
                "analysis": analysis,
            }
            for analysis in primary
            if analysis["n_tools"] >= 3
        ]
        _holm_adjust(omnibus_records, alpha=alpha)
        for record in omnibus_records:
            omnibus = record["analysis"]["omnibus"]
            omnibus["p_holm"] = record["p_holm"]
            omnibus["reject"] = record["reject"]
        pairwise = [record for analysis in primary for record in analysis["pairwise"]]
        family_id = f"{comparison_id}:confirmatory"
        for record in pairwise:
            record["multiplicity_family"] = family_id
        _holm_adjust(pairwise, alpha=alpha)

    for analysis in analyses:
        if analysis["analysis_role"] != "exploratory":
            continue
        family_id = f"{analysis['analysis_id']}:exploratory"
        for record in analysis["pairwise"]:
            record["multiplicity_family"] = family_id
        _holm_adjust(analysis["pairwise"], alpha=alpha)

    for analysis in analyses:
        omnibus = analysis["omnibus"]
        if analysis["analysis_role"] == "exploratory" and omnibus["p_raw"] is not None:
            omnibus["p_holm"] = None
            omnibus["reject"] = omnibus["p_raw"] <= alpha
        gate = True if analysis["n_tools"] == 2 else bool(omnibus["reject"])
        for record in analysis["pairwise"]:
            record["reject"] = bool(record["reject"] and gate)


def _conclude(analysis: dict[str, Any], control: str) -> str:
    if analysis["status"] != "tested":
        return analysis["status"]
    ranked = [item for item in analysis["rankings"] if item["mean_rank"] is not None]
    if not ranked:
        return "not_testable"
    best_rank = ranked[0]["mean_rank"]
    best = [item for item in ranked if item["mean_rank"] == best_rank]
    contrasts = analysis["pairwise"]
    significant_adverse = [
        record
        for record in contrasts
        if record["reject"] and record["rank_biserial"] is not None and record["rank_biserial"] < 0
    ]
    if significant_adverse:
        return "control_significantly_worse_in_at_least_one_contrast"
    if len(best) != 1 or best[0]["configuration_id"] != control:
        return "control_not_uniquely_top_ranked"
    if contrasts and all(
        record["reject"] and record["rank_biserial"] is not None and record["rank_biserial"] > 0
        for record in contrasts
    ):
        return "control_wins"
    if any(record["reject"] for record in contrasts):
        return "control_mixed_or_inconsistent_evidence"
    return "control_top_ranked_not_distinguishable"


def _build_report(
    *,
    profile_id: str,
    purpose: str,
    input_manifest: dict[str, Any],
    metrics_path: Path,
    summary_path: Path,
    contract: dict[str, Any],
    rows: list[dict[str, str]],
    state: dict[str, Any],
    benchmark_environment: dict[str, Any],
    seed_plan: dict[str, Any],
) -> dict[str, Any]:
    inferential = contract["status"] == "inferential"
    root_seed = (
        validate_plan_seed_record(
            seed_plan,
            seed=contract["random_seed"],
            seed_derivation=contract["seed_derivation"],
            required_namespace="full.test",
            required_stream="statistical",
            required_coordinates={"purpose": "phase_root"},
        )
        if inferential
        else None
    )
    statistical_namespace = (
        root_seed["seed_derivation"]["namespace"] if root_seed is not None else None
    )
    analyses: list[dict[str, Any]] = []
    for comparison in contract["comparisons"]:
        for stratum in _expected_strata(state, comparison):
            for metric in comparison["metrics"]:
                analyses.append(
                    _analyze_one(
                        rows,
                        state=state,
                        contract=contract,
                        comparison=comparison,
                        stratum=stratum,
                        metric=metric,
                        seed_plan=seed_plan,
                        statistical_namespace=statistical_namespace,
                    )
                )
    if inferential:
        _apply_multiplicity(analyses, alpha=float(contract["alpha"]))
    controls = {
        comparison["id"]: comparison["control_configuration_id"]
        for comparison in contract["comparisons"]
    }
    for analysis in analyses:
        analysis["conclusion"] = _conclude(
            analysis,
            controls[analysis["comparison_id"]],
        )

    tested = sum(analysis["status"] == "tested" for analysis in analyses)
    incomplete = sum(analysis["status"] == "incomplete_panel" for analysis in analyses)
    confirmatory = [
        analysis for analysis in analyses if analysis["analysis_role"] == "confirmatory"
    ]
    confirmatory_tested = sum(analysis["status"] == "tested" for analysis in confirmatory)
    if contract["status"] == "descriptive":
        overall_status = (
            ("contract_only" if purpose == "smoke" else "descriptive_only")
            if analyses
            else "incomplete_panel"
        )
    else:
        overall_status = "tested" if tested else "incomplete_panel"
    if inferential and confirmatory and confirmatory_tested != len(confirmatory):
        overall_status = "confirmatory_incomplete"
    elif incomplete:
        overall_status = "incomplete_panel" if not tested else "partially_tested"
    benchmark = input_manifest.get("benchmark_repository") or {}
    report = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "profile_id": profile_id,
        "purpose": purpose,
        "status": overall_status,
        "inputs": {
            "benchmark_commit": benchmark.get("commit"),
            "benchmark_environment": benchmark_environment,
            "metrics_csv": {
                "path": "../metrics.csv",
                "sha256": sha256_file(metrics_path),
            },
            "summary_json": {
                "path": "../summary.json",
                "sha256": sha256_file(summary_path),
            },
        },
        "contract": contract,
        "methods": {
            "block_unit": (
                "one dataset_id per unique simulator_id/generation seed within each "
                "predeclared stratum"
            ),
            "aggregation_order": [
                "equal-weight group contexts within one execution",
                "equal-weight algorithmic executions within configuration and dataset",
                "one equally weighted paired score per configuration and independent dataset",
            ],
            "analysis_mode": contract["status"],
            "omnibus": (
                "Friedman with midranks, tie correction and exact/Monte Carlo "
                "within-block label permutations"
                if inferential
                else None
            ),
            "posthoc": (
                "two-sided paired Wilcoxon signed-rank with zero differences discarded "
                "and exact/Monte Carlo sign permutations"
                if inferential
                else None
            ),
            "multiplicity": "Holm step-down family-wise error control" if inferential else None,
            "effect_interval": (
                "paired percentile bootstrap of the median difference with fixed "
                "simulator-by-condition design strata"
                if inferential
                else None
            ),
            "descriptive_effects": (
                None
                if inferential
                else "paired point estimates, ranks, win/tie/loss counts and effect sizes"
            ),
            "missing_data": "complete predeclared panel required; no imputation or pairwise deletion",
            "candidate_space_validation": (
                "identical dataset, source, target, candidate-edge and truth-edge identities, "
                "plus identical ANDREA candidate-universe counts, required for every paired "
                "context"
            ),
        },
        "summary": {
            "n_analyses": len(analyses),
            "n_inferential": tested,
            "n_descriptive": sum(analysis["status"] == "descriptive_only" for analysis in analyses),
            "n_contract_only": sum(analysis["status"] == "contract_only" for analysis in analyses),
            "n_incomplete": incomplete,
            "n_confirmatory": len(confirmatory),
            "n_confirmatory_inferential": confirmatory_tested,
            "n_control_wins": sum(
                analysis["conclusion"] == "control_wins" for analysis in analyses
            ),
        },
        "analyses": analyses,
    }
    if root_seed is not None:
        report["seed_contract"] = {
            "seed_plan": "experiments/v1/seeds/seed-plan.json",
            "seed_plan_sha256": canonical_json_sha256(seed_plan),
            "root": root_seed,
        }
    return report


def _ranking_rows(report: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for analysis in report["analyses"]:
        common = {
            "analysis_id": analysis["analysis_id"],
            "comparison_id": analysis["comparison_id"],
            "comparison_role": analysis["comparison_role"],
            "candidate_space": analysis["candidate_space"],
            **{field: analysis["stratum"].get(field) for field in STRATUM_FIELDS},
            "metric": analysis["metric"],
            "analysis_role": analysis["analysis_role"],
            "status": analysis["status"],
            "n_expected_blocks": analysis["n_expected_blocks"],
            "n_complete_blocks": analysis["n_complete_blocks"],
        }
        for ranking in analysis["rankings"]:
            yield {
                **common,
                "configuration_id": ranking["configuration_id"],
                "inference_regulator_scope": ranking["inference_regulator_scope"],
                "configuration_sha256": ranking["configuration_sha256"],
                "tool_id": ranking["tool_id"],
                "mean_score": ranking["mean_score"],
                "median_score": ranking["median_score"],
                "mean_rank": ranking["mean_rank"],
            }


def _pairwise_rows(report: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for analysis in report["analyses"]:
        common = {
            "analysis_id": analysis["analysis_id"],
            "comparison_id": analysis["comparison_id"],
            "comparison_role": analysis["comparison_role"],
            "candidate_space": analysis["candidate_space"],
            **{field: analysis["stratum"].get(field) for field in STRATUM_FIELDS},
            "metric": analysis["metric"],
            "analysis_role": analysis["analysis_role"],
        }
        for pair in analysis["pairwise"]:
            yield {
                **common,
                **{field: pair.get(field) for field in PAIRWISE_FIELDS if field not in common},
            }


def _markdown_number(value: Any) -> str:
    number = _finite_number(value)
    return "" if number is None else f"{number:.6g}"


def _markdown(report: dict[str, Any]) -> str:
    inferential = report["contract"]["status"] == "inferential"
    lines = [
        f"# Benchmark analysis report: {report['profile_id']}",
        "",
        f"Overall status: `{report['status']}`.",
        "",
        "Independent datasets are the paired blocks. Group contexts and algorithmic seeds "
        + (
            "are averaged before testing; missing panels are never imputed."
            if inferential
            else "are averaged before descriptive comparison; no hypothesis test, confidence "
            "interval or winner claim is produced."
        ),
        "",
    ]
    for analysis in report["analyses"]:
        stratum = ", ".join(f"{key}={value}" for key, value in sorted(analysis["stratum"].items()))
        lines.extend(
            [
                f"## {analysis['comparison_id']} / {analysis['metric']}",
                "",
                f"- Stratum: `{stratum}`",
                f"- Evaluation candidate space: `{analysis['candidate_space']}`",
                "- Inference regulator scopes: `"
                + ", ".join(
                    f"{configuration}={scope}"
                    for configuration, scope in analysis[
                        "inference_regulator_scope_by_configuration"
                    ].items()
                )
                + "`",
                f"- Role: `{analysis['analysis_role']}`",
                f"- Status: `{analysis['status']}`",
                f"- Blocks: {analysis['n_complete_blocks']}/{analysis['n_expected_blocks']}",
                f"- Conclusion: `{analysis['conclusion']}`",
                "",
                "| Rank | Tool | Configuration | Mean rank | Mean score |",
                "| ---: | --- | --- | ---: | ---: |",
            ]
        )
        for position, ranking in enumerate(analysis["rankings"], start=1):
            mean_rank = _markdown_number(ranking["mean_rank"])
            mean_score = _markdown_number(ranking["mean_score"])
            lines.append(
                f"| {position} | {ranking['tool_id']} | {ranking['configuration_id']} | "
                f"{mean_rank} | {mean_score} |"
            )
        lines.append("")
        if not inferential:
            lines.extend(
                [
                    "### Paired descriptive effects",
                    "",
                    "Positive differences favor the SPATHI control.",
                    "",
                    "| Competitor | Wins | Ties | Losses | Probability of superiority | "
                    "Median difference | Hodges-Lehmann |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for pair in analysis["pairwise"]:
                lines.append(
                    f"| {pair['competitor_tool_id']} "
                    f"(`{pair['competitor_configuration_id']}`) | {pair['wins']} | "
                    f"{pair['ties']} | {pair['losses']} | "
                    f"{_markdown_number(pair['probability_of_superiority'])} | "
                    f"{_markdown_number(pair['median_difference'])} | "
                    f"{_markdown_number(pair['hodges_lehmann'])} |"
                )
            lines.append("")
            if analysis["reason"]:
                lines.extend([analysis["reason"], ""])
            continue
        omnibus = analysis["omnibus"]
        lines.extend(
            [
                "### Omnibus Friedman test",
                "",
                "| Status | Q | Raw p | Holm-adjusted p | Reject | Kendall's W | Method |",
                "| --- | ---: | ---: | ---: | --- | ---: | --- |",
                f"| {omnibus['status']} | {_markdown_number(omnibus['statistic'])} | "
                f"{_markdown_number(omnibus['p_raw'])} | "
                f"{_markdown_number(omnibus['p_holm'])} | {str(omnibus['reject']).lower()} | "
                f"{_markdown_number(omnibus['kendalls_w'])} | {omnibus['method'] or ''} |",
                "",
                "### Post-hoc paired Wilcoxon tests",
                "",
                "Positive differences and rank-biserial effects favor the SPATHI control. "
                "Intervals are paired percentile bootstrap intervals with fixed "
                "simulator-by-condition design strata.",
                "",
                "| Competitor | W | Raw p | p-Holm | Reject | Rank-biserial | Median difference | "
                "Hodges-Lehmann | Bootstrap interval |",
                "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for pair in analysis["pairwise"]:
            confidence = pair.get("bootstrap_confidence_level")
            confidence_label = (
                f"{100.0 * confidence:.3g}% " if isinstance(confidence, (int, float)) else ""
            )
            interval = ""
            if pair.get("bootstrap_ci_low") is not None:
                interval = (
                    f"{confidence_label}[{_markdown_number(pair['bootstrap_ci_low'])}, "
                    f"{_markdown_number(pair['bootstrap_ci_high'])}]"
                )
            lines.append(
                f"| {pair['competitor_tool_id']} (`{pair['competitor_configuration_id']}`) | "
                f"{_markdown_number(pair['wilcoxon_w'])} | {_markdown_number(pair['p_raw'])} | "
                f"{_markdown_number(pair['p_holm'])} | {str(pair['reject']).lower()} | "
                f"{_markdown_number(pair['rank_biserial'])} | "
                f"{_markdown_number(pair['median_difference'])} | "
                f"{_markdown_number(pair['hodges_lehmann'])} | {interval} |"
            )
        lines.append("")
        if analysis["reason"]:
            lines.extend([analysis["reason"], ""])
    return "\n".join(lines).rstrip() + "\n"


def _write_outputs(directory: Path, report: dict[str, Any]) -> None:
    with (directory / "rankings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        for row in _ranking_rows(report):
            writer.writerow({field: _csv_value(row.get(field)) for field in RANKING_FIELDS})
    with (directory / "pairwise.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRWISE_FIELDS)
        writer.writeheader()
        for row in _pairwise_rows(report):
            writer.writerow({field: _csv_value(row.get(field)) for field in PAIRWISE_FIELDS})
    (directory / "report.md").write_text(_markdown(report), encoding="utf-8")
    report["outputs"] = {
        filename: {"sha256": sha256_file(directory / filename)}
        for filename in ("rankings.csv", "pairwise.csv", "report.md")
    }
    (directory / "statistics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _output_paths(directory: Path) -> tuple[Path, Path, Path, Path]:
    return (
        directory / STATISTICS_FILENAMES[0],
        directory / STATISTICS_FILENAMES[1],
        directory / STATISTICS_FILENAMES[2],
        directory / STATISTICS_FILENAMES[3],
    )


def _statistics_state_record(
    directory: Path,
    report: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    safe_run_root = path_below_no_symlinks(
        run_root,
        run_root,
        "statistics run root",
        require_directory=True,
    )
    reports_root = path_below_no_symlinks(
        safe_run_root / "reports",
        safe_run_root,
        "statistics reports root",
        require_directory=True,
    )
    expected_directory = reports_root / "statistics"
    if directory != expected_directory:
        raise BenchmarkConfigError("Statistics directory is not the canonical run directory")
    directory = path_below_no_symlinks(
        directory,
        reports_root,
        "statistics directory",
        require_directory=True,
    )
    if {member.name for member in directory.iterdir()} != set(STATISTICS_FILENAMES):
        raise BenchmarkConfigError("Statistics directory has a non-canonical member set")
    paths = tuple(
        path_below_no_symlinks(
            path,
            directory,
            f"statistics output {path.name}",
            require_file=True,
        )
        for path in _output_paths(directory)
    )
    files = {
        path.name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "directory": str(directory),
        "input_metrics_sha256": report["inputs"]["metrics_csv"]["sha256"],
        "input_summary_sha256": report["inputs"]["summary_json"]["sha256"],
        "benchmark_environment_fingerprint": report["inputs"]["benchmark_environment"][
            "fingerprint"
        ],
        "files": files,
    }


def _validate_existing_outputs(
    *,
    state: dict[str, Any],
    run_root: Path,
    metrics_path: Path,
    summary_path: Path,
) -> tuple[Path, Path, Path, Path] | None:
    record = state.get("statistics")
    if record is None:
        return None
    if not isinstance(record, dict) or record.get("schema_version") != STATISTICS_SCHEMA_VERSION:
        raise BenchmarkConfigError("Completed statistics state has an invalid schema")
    safe_run_root = path_below_no_symlinks(
        run_root,
        run_root,
        "statistics run root",
        require_directory=True,
    )
    reports_root = path_below_no_symlinks(
        safe_run_root / "reports",
        safe_run_root,
        "statistics reports root",
        require_directory=True,
    )
    expected_directory = reports_root / "statistics"
    if record.get("directory") != str(expected_directory):
        raise BenchmarkConfigError("Recorded statistics directory is not canonical for this run")
    directory = path_below_no_symlinks(
        expected_directory,
        reports_root,
        "statistics directory",
        require_directory=True,
    )
    if record.get("benchmark_environment_fingerprint") != (
        state.get("benchmark_environment") or {}
    ).get("fingerprint"):
        raise BenchmarkConfigError(
            "Recorded statistics were computed in a different benchmark environment"
        )
    files = record.get("files")
    if not isinstance(files, dict) or set(files) != set(STATISTICS_FILENAMES):
        raise BenchmarkConfigError("Completed statistics state has a non-canonical file manifest")
    observed_members = {member.name for member in directory.iterdir()}
    if observed_members != set(STATISTICS_FILENAMES):
        raise BenchmarkConfigError("Completed statistics directory has non-canonical members")
    paths = _output_paths(directory)
    validated: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        entry = files.get(path.name)
        if not isinstance(entry, dict) or entry.get("path") != str(path):
            raise BenchmarkConfigError(f"Recorded statistical report is missing or changed: {path}")
        safe_path = path_below_no_symlinks(
            path,
            directory,
            f"statistics output {path.name}",
            require_file=True,
        )
        validated.append((safe_path, entry))
    if record.get("input_metrics_sha256") != sha256_file(metrics_path) or record.get(
        "input_summary_sha256"
    ) != sha256_file(summary_path):
        raise BenchmarkConfigError("Recorded statistics were computed from different reports")
    for path, entry in validated:
        if entry.get("sha256") != sha256_file(path):
            raise BenchmarkConfigError(f"Recorded statistical report is missing or changed: {path}")
    return paths


def analyze_statistics(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
) -> tuple[Path, Path, Path, Path]:
    """Produce deterministic rankings and paired non-parametric comparisons."""

    input_manifest = validate_run_inputs(repo_root, profile_id, run_root)
    validate_inference_attribution_contract(
        repo_root=repo_root,
        profile_id=profile_id,
        run_root=run_root,
    )
    state = load_state(run_root, profile_id)
    require_status(state, "completed", "Statistical analysis")
    benchmark_environment = benchmark_host_environment()
    recorded_environment = state.get("benchmark_environment")
    if not isinstance(recorded_environment, dict) or recorded_environment.get(
        "fingerprint"
    ) != benchmark_environment.get("fingerprint"):
        raise BenchmarkConfigError(
            "The SPATHI-benchmarks Python environment changed after preflight; use a new run ID"
        )
    metrics_path, summary_path = validate_summary_reports(state, run_root)
    safe_run_root = path_below_no_symlinks(
        run_root,
        run_root,
        "statistics run root",
        require_directory=True,
    )
    reports_root = path_below_no_symlinks(
        safe_run_root / "reports",
        safe_run_root,
        "statistics reports root",
        require_directory=True,
    )
    existing = _validate_existing_outputs(
        state=state,
        run_root=run_root,
        metrics_path=metrics_path,
        summary_path=summary_path,
    )
    if existing is not None:
        return existing

    profile = frozen_profile(run_root, profile_id)
    try:
        seed_plan = load_seed_plan(repo_root / "experiments/v1/seeds/seed-plan.json")
    except SeedPlanError as exc:
        raise BenchmarkConfigError(f"Invalid benchmark seed plan: {exc}") from exc
    contract = _validate_statistics_contract(profile, seed_plan)
    rows = _read_metrics(metrics_path)
    report = _build_report(
        profile_id=profile_id,
        purpose=state["purpose"],
        input_manifest=input_manifest,
        metrics_path=metrics_path,
        summary_path=summary_path,
        contract=contract,
        rows=rows,
        state=state,
        benchmark_environment=benchmark_environment,
        seed_plan=seed_plan,
    )
    statistics_dir = reports_root / "statistics"
    if statistics_dir.exists() or statistics_dir.is_symlink():
        statistics_dir = path_below_no_symlinks(
            statistics_dir,
            reports_root,
            "statistics directory",
            require_directory=True,
        )
        expected_directory = Path(
            tempfile.mkdtemp(prefix=".statistics-recovery-", dir=reports_root)
        )
        try:
            _write_outputs(expected_directory, report)
            expected_paths = _output_paths(expected_directory)
            observed_paths = _output_paths(statistics_dir)
            expected_members = {path.name for path in expected_paths}
            observed_members = (
                {path.name for path in statistics_dir.iterdir()}
                if statistics_dir.is_dir() and not statistics_dir.is_symlink()
                else set()
            )
            matches = observed_members == expected_members and all(
                observed.is_file()
                and not observed.is_symlink()
                and sha256_file(observed) == sha256_file(expected)
                for observed, expected in zip(observed_paths, expected_paths, strict=True)
            )
        finally:
            shutil.rmtree(expected_directory)
        if not matches:
            raise BenchmarkConfigError(
                f"Incomplete or stale statistics directory already exists: {statistics_dir}"
            )
        state["statistics"] = _statistics_state_record(statistics_dir, report, safe_run_root)
        state["updated_at"] = _utc_now()
        write_state(run_root, state)
        return _output_paths(statistics_dir)

    temporary = Path(tempfile.mkdtemp(prefix=".statistics-", dir=reports_root))
    try:
        _write_outputs(temporary, report)
        os.rename(temporary, statistics_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    state["statistics"] = _statistics_state_record(statistics_dir, report, safe_run_root)
    state["updated_at"] = _utc_now()
    write_state(run_root, state)
    return _output_paths(statistics_dir)


__all__ = ["analyze_statistics"]
