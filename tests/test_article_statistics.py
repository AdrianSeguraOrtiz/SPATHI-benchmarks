from __future__ import annotations

import math

import pytest

from spathi_bench.article_statistics import (
    StatisticsInputError,
    analyze_family_metric,
    analyze_paired_statistics,
    friedman_test,
    holm_adjust,
    wilcoxon_signed_rank,
)


def _rows() -> list[dict[str, object]]:
    scores = {
        "d1": {"a": 0.90, "b": 0.70, "c": 0.50},
        "d2": {"a": 0.80, "b": 0.65, "c": 0.40},
        "d3": {"a": 0.85, "b": 0.60, "c": 0.45},
        "d4": {"a": 0.75, "b": 0.55, "c": 0.35},
        "d5": {"a": 0.95, "b": 0.75, "c": 0.60},
    }
    return [
        {"dataset_id": dataset, "method_id": method, "aupr": value}
        for dataset, methods in scores.items()
        for method, value in methods.items()
    ]


def test_complete_family_is_ranked_and_tests_every_unordered_pair() -> None:
    result = analyze_family_metric(
        _rows(), family="same_information", metric="aupr", methods=["a", "b", "c"]
    )

    assert result["status"] == "ok"
    assert result["n_complete_blocks"] == 5
    assert [row["method_id"] for row in result["rankings"]] == ["a", "b", "c"]
    assert [row["average_rank"] for row in result["rankings"]] == [1.0, 2.0, 3.0]
    assert result["omnibus"]["test"] == "friedman"
    assert result["omnibus"]["p_value"] < 0.05
    assert {row["pair_id"] for row in result["pairwise"]} == {
        "a__vs__b",
        "a__vs__c",
        "b__vs__c",
    }
    assert all(row["p_holm"] is not None for row in result["pairwise"])
    assert all("control" not in key for row in result["pairwise"] for key in row)


def test_every_test_uses_the_same_complete_blocks() -> None:
    rows = _rows()
    rows = [row for row in rows if not (row["dataset_id"] == "d2" and row["method_id"] == "c")]
    result = analyze_family_metric(rows, family="main", metric="aupr", methods=["a", "b", "c"])

    assert result["n_candidate_blocks"] == 5
    assert result["n_complete_blocks"] == 4
    assert {tuple(block.values()) for block in result["complete_blocks"]} == {
        ("d1",),
        ("d3",),
        ("d4",),
        ("d5",),
    }
    assert {row["n_blocks"] for row in result["pairwise"]} == {4}
    assert result["excluded_blocks"] == [
        {
            "block": {"dataset_id": "d2"},
            "missing_methods": ["c"],
            "duplicate_methods": [],
            "non_finite_methods": [],
        }
    ]


def test_missing_declared_method_is_not_silently_dropped() -> None:
    result = analyze_family_metric(
        [row for row in _rows() if row["method_id"] != "c"],
        family="main",
        metric="aupr",
        methods=["a", "b", "c"],
    )

    assert result["status"] == "missing_methods"
    assert result["missing_methods"] == ["c"]
    assert result["omnibus"]["p_value"] is None
    assert all(row["p_value"] is None for row in result["pairwise"])
    assert "c" in result["reason"]


def test_insufficient_blocks_and_duplicates_have_explicit_diagnostics() -> None:
    rows = [row for row in _rows() if row["dataset_id"] in {"d1", "d2"}]
    rows.append({"dataset_id": "d2", "method_id": "b", "aupr": 0.7})
    result = analyze_family_metric(rows, family="main", metric="aupr", methods=["a", "b", "c"])

    assert result["status"] == "insufficient_complete_blocks"
    assert result["n_complete_blocks"] == 1
    assert result["omnibus"]["status"] == "insufficient_complete_blocks"
    assert result["excluded_blocks"][0]["duplicate_methods"] == ["b"]


def test_wilcoxon_exact_sign_distribution_handles_ties_and_zeroes() -> None:
    result = wilcoxon_signed_rank([0.0, 1.0, 1.0, 2.0, 3.0])

    assert result["method"] == "exact_sign_permutation"
    assert result["n_pairs"] == 5
    assert result["n_nonzero"] == 4
    assert result["n_zero"] == 1
    assert result["statistic"] == 0.0
    assert result["p_value"] == pytest.approx(0.125)
    assert result["rank_biserial"] == 1.0

    tied = wilcoxon_signed_rank([0.0, 0.0, 0.0])
    assert tied["method"] == "all_zero_differences"
    assert tied["p_value"] == 1.0


def test_wilcoxon_monte_carlo_fallback_is_seeded_and_never_zero() -> None:
    differences = [float(index + 1) for index in range(12)]
    first = wilcoxon_signed_rank(differences, exact_max_nonzero=5, permutations=999, random_seed=17)
    second = wilcoxon_signed_rank(
        differences, exact_max_nonzero=5, permutations=999, random_seed=17
    )

    assert first == second
    assert first["method"] == "monte_carlo_sign_permutation"
    assert first["p_value"] >= 1 / 1000


def test_friedman_tie_correction_and_all_tied_case() -> None:
    ordered = friedman_test([[3, 2, 1], [3, 2, 1], [3, 2, 1]])
    assert ordered["statistic"] == pytest.approx(6.0)
    assert ordered["average_ranks"] == [1.0, 2.0, 3.0]

    all_tied = friedman_test([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert all_tied["statistic"] == 0.0
    assert all_tied["p_value"] == 1.0
    assert all_tied["tie_correction"] == 0.0


def test_holm_is_monotone_over_all_pairs() -> None:
    rows = [
        {"pair_id": "a__vs__b", "p_value": 0.01},
        {"pair_id": "a__vs__c", "p_value": 0.03},
        {"pair_id": "b__vs__c", "p_value": 0.04},
    ]
    holm_adjust(rows, alpha=0.05)

    assert [row["p_holm"] for row in rows] == pytest.approx([0.03, 0.06, 0.06])
    assert [row["reject_holm"] for row in rows] == [True, False, False]


def test_top_level_supports_wide_and_long_metrics_and_direction() -> None:
    rows = _rows()
    rows.extend(
        {
            "dataset_id": dataset,
            "method_id": method,
            "metric": "runtime_seconds",
            "value": value,
        }
        for dataset, methods in {
            "d1": {"a": 3, "b": 2, "c": 1},
            "d2": {"a": 3, "b": 2, "c": 1},
            "d3": {"a": 3, "b": 2, "c": 1},
            "d4": {"a": 3, "b": 2, "c": 1},
            "d5": {"a": 3, "b": 2, "c": 1},
        }.items()
        for method, value in methods.items()
    )
    result = analyze_paired_statistics(
        rows,
        metrics=["aupr", "runtime_seconds"],
        families={"main": ["a", "b", "c"]},
        higher_is_better={"runtime_seconds": False},
    )

    assert result["design"] == "complete_paired_blocks"
    assert len(result["analyses"]) == 2
    runtime = next(
        analysis for analysis in result["analyses"] if analysis["metric"] == "runtime_seconds"
    )
    assert [row["average_rank"] for row in runtime["rankings"]] == [3.0, 2.0, 1.0]


def test_invalid_design_is_rejected() -> None:
    with pytest.raises(StatisticsInputError, match="duplicate methods"):
        analyze_family_metric([], family="main", metric="aupr", methods=["a", "a"])
    with pytest.raises(StatisticsInputError, match="block field"):
        analyze_family_metric(
            [{"method_id": "a", "aupr": 0.5}],
            family="main",
            metric="aupr",
            methods=["a", "b"],
        )
    with pytest.raises(StatisticsInputError, match="two-dimensional"):
        friedman_test([1.0, 2.0])  # type: ignore[list-item]


def test_nonfinite_scores_exclude_the_entire_block() -> None:
    rows = _rows()
    for row in rows:
        if row["dataset_id"] == "d1" and row["method_id"] == "a":
            row["aupr"] = math.nan
    result = analyze_family_metric(rows, family="main", metric="aupr", methods=["a", "b", "c"])

    assert result["n_complete_blocks"] == 4
    assert result["excluded_blocks"][0]["non_finite_methods"] == ["a"]
