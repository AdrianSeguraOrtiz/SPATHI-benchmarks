"""Symmetric paired statistics for the article benchmark.

The statistical unit is a *complete block*: one benchmark dataset (or another
caller-defined tuple of fields) for which every method in a comparison family
has exactly one finite score.  Omnibus and post-hoc tests always use the same
complete blocks, so missing or duplicated results cannot silently change the
population of an individual pairwise comparison.

There is deliberately no concept of a control or reference method.  Every
method receives an average rank and every unordered pair is tested.  Holm's
step-down correction is applied over all pairs within one family and metric.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean, median
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"


class StatisticsInputError(ValueError):
    """Raised when rows cannot define an unambiguous paired comparison."""


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rank(values: Sequence[float], *, higher_is_better: bool) -> list[float]:
    """Return one-based midranks, with rank one assigned to the best value."""

    ordered = sorted(
        range(len(values)),
        key=lambda index: (-values[index] if higher_is_better else values[index], index),
    )
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        value = values[ordered[start]]
        while end < len(ordered) and values[ordered[end]] == value:
            end += 1
        midrank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = midrank
        start = end
    return ranks


def _regularized_gamma_q(shape: float, value: float) -> float:
    """Regularized upper incomplete gamma, implemented without SciPy."""

    if shape <= 0.0 or value < 0.0:
        raise ValueError("Gamma arguments are outside their domain")
    if value == 0.0:
        return 1.0

    epsilon = 3.0e-14
    tiny = 1.0e-300
    iterations = 10_000
    log_scale = -value + shape * math.log(value) - math.lgamma(shape)

    if value < shape + 1.0:
        term = 1.0 / shape
        series = term
        denominator = shape
        for _ in range(iterations):
            denominator += 1.0
            term *= value / denominator
            series += term
            if abs(term) <= abs(series) * epsilon:
                lower = series * math.exp(log_scale)
                return min(1.0, max(0.0, 1.0 - lower))
        raise ArithmeticError("Incomplete-gamma series did not converge")

    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / max(abs(b), tiny)
    if b < 0.0:
        d = -d
    fraction = d
    for index in range(1, iterations + 1):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            upper = math.exp(log_scale) * fraction
            return min(1.0, max(0.0, upper))
    raise ArithmeticError("Incomplete-gamma continued fraction did not converge")


def friedman_test(
    matrix: Sequence[Sequence[float]],
    *,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Run the tie-corrected Friedman omnibus test on complete paired blocks."""

    scores = np.asarray(matrix, dtype=float)
    if scores.ndim != 2:
        raise StatisticsInputError("Friedman input must be a two-dimensional matrix")
    blocks, methods = scores.shape
    if blocks < 2 or methods < 2:
        raise StatisticsInputError("Friedman requires at least two blocks and two methods")
    if not np.isfinite(scores).all():
        raise StatisticsInputError("Friedman input contains a non-finite value")

    rank_matrix = np.asarray(
        [_rank(row.tolist(), higher_is_better=higher_is_better) for row in scores]
    )
    rank_sums = rank_matrix.sum(axis=0)
    statistic = 12.0 * float(np.square(rank_sums).sum()) / (
        blocks * methods * (methods + 1.0)
    ) - 3.0 * blocks * (methods + 1.0)

    tie_sum = 0
    for row in scores:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += int(sum(count**3 - count for count in counts if count > 1))
    correction = 1.0 - tie_sum / (blocks * (methods**3 - methods))
    if correction <= 0.0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = max(0.0, statistic / correction)
        p_value = _regularized_gamma_q((methods - 1) / 2.0, statistic / 2.0)

    return {
        "test": "friedman",
        "method": "chi_square_approximation_with_tie_correction",
        "statistic": statistic,
        "degrees_of_freedom": methods - 1,
        "p_value": p_value,
        "tie_correction": correction,
        "n_blocks": blocks,
        "n_methods": methods,
        "average_ranks": rank_matrix.mean(axis=0).tolist(),
    }


def _midranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        value = values[ordered[start]]
        while end < len(ordered) and values[ordered[end]] == value:
            end += 1
        midrank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = midrank
        start = end
    return ranks


def _exact_signed_rank_p_value(scaled_ranks: Sequence[int], observed: int) -> float:
    counts: dict[int, int] = {0: 1}
    for rank in scaled_ranks:
        updated = counts.copy()
        for subtotal, count in counts.items():
            updated[subtotal + rank] = updated.get(subtotal + rank, 0) + count
        counts = updated
    total_rank = sum(scaled_ranks)
    observed_extremeness = min(observed, total_rank - observed)
    extreme = sum(
        count
        for subtotal, count in counts.items()
        if min(subtotal, total_rank - subtotal) <= observed_extremeness
    )
    return min(1.0, extreme / (2 ** len(scaled_ranks)))


def _stable_seed(*parts: str, base_seed: int) -> int:
    document = "\0".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(document).digest()[:8], "big", signed=False)


def wilcoxon_signed_rank(
    differences: Sequence[float],
    *,
    exact_max_nonzero: int = 200,
    permutations: int = 99_999,
    random_seed: int = 0,
    zero_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Two-sided paired Wilcoxon test using exact signs or a seeded fallback."""

    finite = [float(value) for value in differences if math.isfinite(float(value))]
    nonzero = [value for value in finite if abs(value) > zero_tolerance]
    zeros = len(finite) - len(nonzero)
    if not nonzero:
        return {
            "test": "wilcoxon_signed_rank",
            "method": "all_zero_differences",
            "statistic": 0.0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
            "n_pairs": len(finite),
            "n_nonzero": 0,
            "n_zero": zeros,
        }

    ranks = _midranks([abs(value) for value in nonzero])
    w_plus = sum(rank for rank, difference in zip(ranks, nonzero, strict=True) if difference > 0)
    w_minus = sum(rank for rank, difference in zip(ranks, nonzero, strict=True) if difference < 0)
    total_rank = w_plus + w_minus
    statistic = min(w_plus, w_minus)

    # Midranks are integer or half-integer, so multiplying by two permits exact
    # integer dynamic programming even when absolute differences contain ties.
    scaled_ranks = [int(round(rank * 2.0)) for rank in ranks]
    observed = int(round(w_plus * 2.0))
    if len(nonzero) <= exact_max_nonzero:
        p_value = _exact_signed_rank_p_value(scaled_ranks, observed)
        method = "exact_sign_permutation"
        effective_permutations: int | None = 2 ** len(nonzero)
    else:
        if permutations < 1:
            raise StatisticsInputError("permutations must be positive")
        rng = np.random.default_rng(random_seed)
        rank_array = np.asarray(scaled_ranks, dtype=np.int64)
        observed_extremeness = min(observed, int(rank_array.sum()) - observed)
        extreme = 0
        remaining = permutations
        while remaining:
            batch = min(remaining, 10_000)
            signs = rng.integers(0, 2, size=(batch, len(nonzero)), dtype=np.int8)
            totals = signs @ rank_array
            extremeness = np.minimum(totals, int(rank_array.sum()) - totals)
            extreme += int(np.count_nonzero(extremeness <= observed_extremeness))
            remaining -= batch
        p_value = (extreme + 1.0) / (permutations + 1.0)
        method = "monte_carlo_sign_permutation"
        effective_permutations = permutations

    return {
        "test": "wilcoxon_signed_rank",
        "method": method,
        "statistic": statistic,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_value": p_value,
        "rank_biserial": (w_plus - w_minus) / total_rank,
        "n_pairs": len(finite),
        "n_nonzero": len(nonzero),
        "n_zero": zeros,
        "permutations": effective_permutations,
    }


def holm_adjust(records: Sequence[dict[str, Any]], *, alpha: float) -> None:
    """Add monotone Holm-adjusted p-values and decisions in place."""

    eligible = [record for record in records if _finite_float(record.get("p_value")) is not None]
    ordered = sorted(eligible, key=lambda record: (float(record["p_value"]), record["pair_id"]))
    family_size = len(ordered)
    running = 0.0
    for index, record in enumerate(ordered):
        adjusted = min(1.0, (family_size - index) * float(record["p_value"]))
        running = max(running, adjusted)
        record["p_holm"] = running
        record["reject_holm"] = running <= alpha
    for record in records:
        if record not in eligible:
            record["p_holm"] = None
            record["reject_holm"] = None


def _block_key(row: Mapping[str, Any], block_fields: Sequence[str]) -> tuple[Any, ...]:
    missing = [field for field in block_fields if field not in row]
    if missing:
        raise StatisticsInputError(f"Row omits block field(s): {', '.join(missing)}")
    return tuple(row[field] for field in block_fields)


def _serializable_block(block_fields: Sequence[str], key: Sequence[Any]) -> dict[str, Any]:
    return {field: value for field, value in zip(block_fields, key, strict=True)}


def _sortable_block(key: Sequence[Any]) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)


def _score(row: Mapping[str, Any], metric: str) -> float | None:
    if metric in row:
        return _finite_float(row[metric])
    if row.get("metric") == metric and "value" in row:
        return _finite_float(row["value"])
    return None


def _contains_metric(row: Mapping[str, Any], metric: str) -> bool:
    return metric in row or (row.get("metric") == metric and "value" in row)


def _descriptive_rankings(
    *,
    methods: Sequence[str],
    observations: Mapping[str, Sequence[float]],
    complete_matrix: Sequence[Sequence[float]],
    higher_is_better: bool,
) -> list[dict[str, Any]]:
    average_ranks: list[float | None] = [None] * len(methods)
    complete_columns: list[list[float]] = [[] for _ in methods]
    if complete_matrix:
        rank_matrix = [_rank(row, higher_is_better=higher_is_better) for row in complete_matrix]
        average_ranks = [fmean(row[index] for row in rank_matrix) for index in range(len(methods))]
        complete_columns = [
            [row[index] for row in complete_matrix] for index in range(len(methods))
        ]

    rankings = []
    for index, method in enumerate(methods):
        observed = list(observations.get(method, ()))
        complete = complete_columns[index]
        rankings.append(
            {
                "method_id": method,
                "n_observed": len(observed),
                "n_complete_blocks": len(complete),
                "mean_score_observed": fmean(observed) if observed else None,
                "median_score_observed": median(observed) if observed else None,
                "mean_score_complete": fmean(complete) if complete else None,
                "median_score_complete": median(complete) if complete else None,
                "average_rank": average_ranks[index],
            }
        )
    return rankings


def analyze_family_metric(
    rows: Iterable[Mapping[str, Any]],
    *,
    family: str,
    metric: str,
    methods: Sequence[str],
    block_fields: Sequence[str] = ("dataset_id",),
    method_field: str = "method_id",
    higher_is_better: bool = True,
    alpha: float = 0.05,
    min_complete_blocks: int = 3,
    permutations: int = 99_999,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Analyze one metric for one predeclared method family."""

    declared = tuple(str(method) for method in methods)
    if len(declared) != len(set(declared)):
        raise StatisticsInputError(f"Family {family!r} declares duplicate methods")
    if not declared:
        raise StatisticsInputError(f"Family {family!r} does not declare any methods")
    if not block_fields:
        raise StatisticsInputError("At least one block field is required")
    if not 0.0 < alpha < 1.0:
        raise StatisticsInputError("alpha must be between zero and one")
    if min_complete_blocks < 2:
        raise StatisticsInputError("min_complete_blocks must be at least two")

    cells: dict[tuple[Any, ...], dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    observations: dict[str, list[float]] = defaultdict(list)
    observed_methods: set[str] = set()
    declared_set = set(declared)
    for row in rows:
        if method_field not in row:
            raise StatisticsInputError(f"Row omits method field {method_field!r}")
        method = str(row[method_field])
        if method not in declared_set:
            continue
        # A normalized result table may mix wide score rows with long-form
        # resource rows.  Rows about another metric are not duplicate results.
        if not _contains_metric(row, metric):
            continue
        block = _block_key(row, block_fields)
        value = _score(row, metric)
        cells[block][method].append(value)
        observed_methods.add(method)
        if value is not None:
            observations[method].append(value)

    complete_blocks: list[tuple[Any, ...]] = []
    complete_matrix: list[list[float]] = []
    excluded_blocks: list[dict[str, Any]] = []
    for block in sorted(cells, key=_sortable_block):
        values: list[float] = []
        missing: list[str] = []
        duplicate: list[str] = []
        non_finite: list[str] = []
        for method in declared:
            method_values = cells[block].get(method, [])
            if not method_values:
                missing.append(method)
            elif len(method_values) > 1:
                duplicate.append(method)
            elif method_values[0] is None:
                non_finite.append(method)
            else:
                values.append(float(method_values[0]))
        if not missing and not duplicate and not non_finite:
            complete_blocks.append(block)
            complete_matrix.append(values)
        else:
            excluded_blocks.append(
                {
                    "block": _serializable_block(block_fields, block),
                    "missing_methods": missing,
                    "duplicate_methods": duplicate,
                    "non_finite_methods": non_finite,
                }
            )

    absent = [method for method in declared if method not in observed_methods]
    if not cells:
        status = "no_data"
        reason = "No rows contain a declared method for this family."
    elif absent:
        status = "missing_methods"
        reason = "Declared methods are absent from every candidate block: " + ", ".join(absent)
    elif len(declared) < 2:
        status = "insufficient_methods"
        reason = "At least two methods are required for a comparison."
    elif len(complete_blocks) < min_complete_blocks:
        status = "insufficient_complete_blocks"
        reason = (
            f"Found {len(complete_blocks)} complete paired blocks; "
            f"at least {min_complete_blocks} are required."
        )
    else:
        status = "ok"
        reason = None

    rankings = _descriptive_rankings(
        methods=declared,
        observations=observations,
        complete_matrix=complete_matrix,
        higher_is_better=higher_is_better,
    )

    if status == "ok":
        omnibus = {
            "status": "ok",
            **friedman_test(complete_matrix, higher_is_better=higher_is_better),
        }
    else:
        omnibus = {
            "status": status,
            "reason": reason,
            "test": "friedman",
            "statistic": None,
            "degrees_of_freedom": len(declared) - 1 if len(declared) >= 2 else None,
            "p_value": None,
            "n_blocks": len(complete_blocks),
            "n_methods": len(declared),
        }

    pairs: list[dict[str, Any]] = []
    for first_index, second_index in itertools.combinations(range(len(declared)), 2):
        first = declared[first_index]
        second = declared[second_index]
        pair_id = f"{first}__vs__{second}"
        differences = [row[first_index] - row[second_index] for row in complete_matrix]
        wins = sum(
            difference > 0.0 if higher_is_better else difference < 0.0 for difference in differences
        )
        losses = sum(
            difference < 0.0 if higher_is_better else difference > 0.0 for difference in differences
        )
        ties = len(differences) - wins - losses
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "method_a": first,
            "method_b": second,
            "status": status,
            "reason": reason,
            "n_blocks": len(differences),
            "wins_a": wins,
            "ties": ties,
            "wins_b": losses,
            "mean_difference_a_minus_b": fmean(differences) if differences else None,
            "median_difference_a_minus_b": median(differences) if differences else None,
            "p_value": None,
            "p_holm": None,
            "reject_holm": None,
        }
        if status == "ok":
            test = wilcoxon_signed_rank(
                differences,
                permutations=permutations,
                random_seed=_stable_seed(family, metric, pair_id, base_seed=random_seed),
            )
            record.update(test)
            record["status"] = "ok"
            record["reason"] = None
        pairs.append(record)

    if status == "ok":
        holm_adjust(pairs, alpha=alpha)

    return {
        "analysis_id": f"{family}::{metric}",
        "family": family,
        "metric": metric,
        "status": status,
        "reason": reason,
        "higher_is_better": higher_is_better,
        "alpha": alpha,
        "block_fields": list(block_fields),
        "methods": list(declared),
        "observed_methods": sorted(observed_methods),
        "missing_methods": absent,
        "n_candidate_blocks": len(cells),
        "n_complete_blocks": len(complete_blocks),
        "complete_blocks": [_serializable_block(block_fields, block) for block in complete_blocks],
        "excluded_blocks": excluded_blocks,
        "rankings": rankings,
        "omnibus": omnibus,
        "pairwise": pairs,
    }


def analyze_paired_statistics(
    rows: Iterable[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
    families: Mapping[str, Sequence[str]],
    block_fields: Sequence[str] = ("dataset_id",),
    method_field: str = "method_id",
    higher_is_better: Mapping[str, bool] | None = None,
    alpha: float = 0.05,
    min_complete_blocks: int = 3,
    permutations: int = 99_999,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Analyze every declared family/metric pair using a symmetric design."""

    materialized_rows = list(rows)
    directions = dict(higher_is_better or {})
    analyses = [
        analyze_family_metric(
            materialized_rows,
            family=family,
            metric=metric,
            methods=methods,
            block_fields=block_fields,
            method_field=method_field,
            higher_is_better=directions.get(metric, True),
            alpha=alpha,
            min_complete_blocks=min_complete_blocks,
            permutations=permutations,
            random_seed=random_seed,
        )
        for family, methods in families.items()
        for metric in metrics
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "design": "complete_paired_blocks",
        "post_hoc": "all_pairs_wilcoxon_holm",
        "alpha": alpha,
        "block_fields": list(block_fields),
        "analyses": analyses,
    }


__all__ = [
    "SCHEMA_VERSION",
    "StatisticsInputError",
    "analyze_family_metric",
    "analyze_paired_statistics",
    "friedman_test",
    "holm_adjust",
    "wilcoxon_signed_rank",
]
