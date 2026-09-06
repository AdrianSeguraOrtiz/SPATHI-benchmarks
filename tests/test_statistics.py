from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.artifacts import sha256_file
from spathi_bench.cli import main
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.seeds import load_seed_plan, validate_plan_seed_record
from spathi_bench.state import load_state, write_state
from spathi_bench.statistics import (
    _friedman_test,
    _holm_adjust,
    _wilcoxon_test,
    analyze_statistics,
)

METRICS = ("aupr", "auroc", "f1_at_truth_count", "epr_at_truth_count")
FIELDS = (
    "configuration_id",
    "configuration_sha256",
    "dataset_id",
    "dataset_role",
    "split_id",
    "condition_id",
    "simulator_id",
    "generation_job",
    "regime",
    "level",
    "run_id",
    "algorithm_seed",
    "context",
    "status",
    "catalog_tool_id",
    "family",
    "selection_role",
    "n_candidates",
    "n_candidate_genes",
    "n_candidate_sources",
    "n_candidate_targets",
    "dataset_fingerprint_sha256",
    "candidate_source_universe_sha256",
    "candidate_target_universe_sha256",
    "candidate_edge_universe_sha256",
    "truth_edge_universe_sha256",
    *METRICS,
    "invariant_edge_stability",
)


def _comparison(
    configurations: list[str],
    *,
    control: str,
    levels: list[str] | None = None,
    metrics: list[str] | None = None,
    expected_blocks: int = 3,
    dataset_role: str = "smoke",
    comparison_id: str = "primary",
    comparison_role: str = "confirmatory",
    algorithm_seeds_by_configuration: dict[str, list[int | None]] | None = None,
    inference_regulator_scope_by_configuration: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "id": comparison_id,
        "comparison_role": comparison_role,
        "analysis_kind": "network_accuracy",
        "truth_scope": "group_specific",
        "aggregation": {
            "block_unit": "dataset_id",
            "hierarchy": ["simulator_id", "condition_id", "dataset_id"],
            "weighting": "equal_dataset",
        },
        "candidate_space": "tf_restricted",
        "inference_regulator_scope_by_configuration": (
            inference_regulator_scope_by_configuration
            or {
                configuration: (
                    "tf_restricted"
                    if comparison_role == "confirmatory" or configuration == control
                    else "all_gene"
                )
                for configuration in configurations
            }
        ),
        "dataset_roles": [dataset_role],
        "regime": "group_only",
        "levels": levels or ["directed"],
        "metrics": metrics or ["aupr"],
        "primary_level": "directed",
        "primary_metric": "aupr",
        "configuration_ids": configurations,
        "algorithm_seeds_by_configuration": algorithm_seeds_by_configuration
        or {configuration: [1] for configuration in configurations},
        "control_configuration_id": control,
        "stratify_by": [
            "dataset_role",
            "split_id",
            "condition_id",
            "simulator_id",
            "generation_job",
            "regime",
            "level",
        ],
        "expected_blocks": expected_blocks,
        "posthoc_scope": "control_vs_rest",
        "correction": "holm",
    }


def _profile(
    comparison: dict[str, Any] | list[dict[str, Any]],
    *,
    mode: str = "inferential",
) -> dict[str, Any]:
    comparisons = comparison if isinstance(comparison, list) else [comparison]
    if mode == "descriptive":
        comparisons = [
            {
                key: value
                for key, value in item.items()
                if key not in {"posthoc_scope", "correction"}
            }
            for item in comparisons
        ]
        for item in comparisons:
            item["comparison_role"] = "exploratory"
        statistics = {"status": "descriptive", "comparisons": comparisons}
    else:
        statistics = {
            "status": "inferential",
            "alpha": 0.05,
            "random_seed": 1698096914,
            "seed_derivation": {
                "namespace": "full.test",
                "stream": "statistical",
                "coordinates": {"purpose": "phase_root"},
            },
            "permutations": 9999,
            "bootstrap_resamples": 1000,
            "comparisons": comparisons,
        }
    return {
        "id": "smoke",
        "statistics": statistics,
    }


def _dataset(index: int, *, dataset_role: str = "smoke") -> dict[str, Any]:
    return {
        "dataset_id": f"dataset_{index}",
        "dataset_role": dataset_role,
        "split_id": "smoke",
        "condition_id": "contract",
        "simulator_id": "dyngen",
        "generation_job": "dyngen_smoke",
        "seed": 1700 + index,
    }


def _metric_row(
    *,
    configuration_id: str,
    dataset_id: str,
    run_id: str,
    context: str,
    score: float | None,
    level: str = "directed",
    status: str = "ok",
    dataset_role: str = "smoke",
    algorithm_seed: int | None = 1,
    configuration_sha256: str | None = None,
    n_candidates: int = 36,
    n_candidate_genes: int = 10,
    n_candidate_sources: int = 4,
    n_candidate_targets: int = 10,
    candidate_edge_universe_sha256: str | None = None,
) -> dict[str, Any]:
    tool_id = "custom_spathi" if configuration_id.startswith("spathi") else configuration_id
    family = "spathi" if configuration_id.startswith("spathi") else "catalog_baseline"
    return {
        "configuration_id": configuration_id,
        "configuration_sha256": configuration_sha256
        or hashlib.sha256(configuration_id.encode("utf-8")).hexdigest(),
        "dataset_id": dataset_id,
        "dataset_role": dataset_role,
        "split_id": "smoke",
        "condition_id": "contract",
        "simulator_id": "dyngen",
        "generation_job": "dyngen_smoke",
        "regime": "group_only",
        "level": level,
        "run_id": run_id,
        "algorithm_seed": algorithm_seed,
        "context": context,
        "status": status,
        "catalog_tool_id": tool_id,
        "family": family,
        "selection_role": "smoke" if family == "spathi" else "baseline",
        "n_candidates": n_candidates,
        "n_candidate_genes": n_candidate_genes,
        "n_candidate_sources": n_candidate_sources,
        "n_candidate_targets": n_candidate_targets,
        "dataset_fingerprint_sha256": hashlib.sha256(dataset_id.encode("utf-8")).hexdigest(),
        "candidate_source_universe_sha256": hashlib.sha256(b"candidate-sources").hexdigest(),
        "candidate_target_universe_sha256": hashlib.sha256(b"candidate-targets").hexdigest(),
        "candidate_edge_universe_sha256": candidate_edge_universe_sha256
        or hashlib.sha256(f"candidate-edges:{level}".encode()).hexdigest(),
        "truth_edge_universe_sha256": hashlib.sha256(
            f"truth:{dataset_id}:{context}:{level}".encode()
        ).hexdigest(),
        **{metric: score for metric in METRICS},
        "invariant_edge_stability": score,
    }


def _run(
    tmp_path: Path,
    *,
    profile: dict[str, Any],
    datasets: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    purpose: str = "smoke",
) -> tuple[Path, Path]:
    benchmark_environment = {
        "fingerprint": "e" * 64,
        "python_executable": "/fixture/python",
        "python_version": "3.11.0",
        "packages": [],
    }
    monkeypatch.setattr(
        "spathi_bench.statistics.benchmark_host_environment",
        lambda: benchmark_environment,
    )
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    seed_plan_path = repo_root / "experiments/v1/seeds/seed-plan.json"
    seed_plan_path.parent.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "experiments/v1/seeds/seed-plan.json",
        seed_plan_path,
    )
    run_root = tmp_path / "run"
    reports = run_root / "reports"
    reports.mkdir(parents=True)
    metrics_path = reports / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = reports / "summary.json"
    summary_path.write_text('{"schema_version":"1.0"}\n', encoding="utf-8")
    write_state(
        run_root,
        {
            "schema_version": "1.0",
            "profile_id": "smoke",
            "purpose": purpose,
            "status": "completed",
            "benchmark_environment": benchmark_environment,
            "datasets": datasets,
            "reports": {
                "metrics_csv": str(metrics_path.resolve()),
                "metrics_csv_sha256": sha256_file(metrics_path),
                "summary_json": str(summary_path.resolve()),
                "summary_json_sha256": sha256_file(summary_path),
            },
        },
    )
    monkeypatch.setattr(
        "spathi_bench.statistics.validate_run_inputs",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "benchmark_repository": {"commit": "1" * 40},
        },
    )
    monkeypatch.setattr(
        "spathi_bench.statistics.validate_inference_attribution_contract",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "spathi_bench.statistics.frozen_profile",
        lambda *_args, **_kwargs: profile,
    )
    return repo_root, run_root


def test_friedman_exact_oracles_and_ties() -> None:
    balanced = _friedman_test([[3, 2, 1], [1, 2, 3]], permutations=9999, random_seed=1)
    assert balanced["statistic"] == pytest.approx(0.0)
    assert balanced["p_raw"] == pytest.approx(1.0)
    assert balanced["kendalls_w"] == pytest.approx(0.0)

    ordered = _friedman_test([[3, 2, 1], [3, 2, 1], [3, 2, 1]], permutations=9999, random_seed=1)
    assert ordered["statistic"] == pytest.approx(6.0)
    assert ordered["p_raw"] == pytest.approx(1 / 36)
    assert ordered["kendalls_w"] == pytest.approx(1.0)
    assert ordered["method"] == "exact_within_block_label_permutation"

    tied = _friedman_test([[2, 2, 1], [2, 2, 1]], permutations=9999, random_seed=1)
    assert tied["statistic"] == pytest.approx(4.0)
    assert tied["p_raw"] == pytest.approx(1 / 3)
    assert tied["tie_correction"] == pytest.approx(0.75)

    all_tied = _friedman_test([[1, 1, 1]] * 3, permutations=9999, random_seed=1)
    assert all_tied["status"] == "all_tied"
    assert all_tied["p_raw"] == 1.0


def test_statistics_rejects_a_changed_benchmark_python_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(["genie3", "spathi"], control="spathi")
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id="dataset_1",
            run_id=configuration,
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
        )
        for configuration in ("genie3", "spathi")
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison, mode="descriptive"),
        datasets=[_dataset(1)],
        rows=rows,
        monkeypatch=monkeypatch,
    )
    state = load_state(run_root, "smoke")
    state["benchmark_environment"]["fingerprint"] = "c" * 64
    write_state(run_root, state)

    with pytest.raises(BenchmarkConfigError, match="Python environment changed"):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_friedman_monte_carlo_is_seeded_and_never_returns_zero() -> None:
    matrix = [[float(10 - column + row / 100) for column in range(4)] for row in range(8)]
    first = _friedman_test(matrix, permutations=9999, random_seed=42, exact_limit=1)
    second = _friedman_test(matrix, permutations=9999, random_seed=42, exact_limit=1)
    assert first == second
    assert first["method"] == "monte_carlo_within_block_label_permutation"
    assert first["p_raw"] >= 1 / 10000


def test_wilcoxon_exact_oracles_include_effects_and_zero_differences() -> None:
    result = _wilcoxon_test([1, 2, 3, 4, 5], permutations=9999, random_seed=1)
    assert result["wilcoxon_w"] == 0
    assert result["p_raw"] == pytest.approx(0.0625)
    assert result["rank_biserial"] == pytest.approx(1.0)
    assert result["exact_minimum_attainable_p"] == pytest.approx(0.0625)

    mixed = _wilcoxon_test([1, -2, 3, -4], permutations=9999, random_seed=1)
    assert mixed["p_raw"] == pytest.approx(0.875)
    assert mixed["rank_biserial"] == pytest.approx(-0.2)

    with_zero = _wilcoxon_test([0, 1, -2], permutations=9999, random_seed=1)
    assert with_zero["n_blocks"] == 3
    assert with_zero["n_nonzero"] == 2
    assert with_zero["ties"] == 1


def test_holm_is_step_down_monotone_and_counts_missing_contrasts() -> None:
    records = [
        {"contrast_id": "a", "p_raw": 0.01},
        {"contrast_id": "b", "p_raw": 0.04},
        {"contrast_id": "c", "p_raw": 0.03},
        {"contrast_id": "missing", "p_raw": None},
    ]
    _holm_adjust(records, alpha=0.05)
    assert [record["p_holm"] for record in records[:3]] == pytest.approx([0.04, 0.09, 0.09])
    assert records[0]["reject"] is True
    assert records[1]["reject"] is False
    assert records[3]["p_holm"] is None


def test_smoke_writes_contract_only_reports_without_treating_groups_as_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(
        ["genie3", "spathi"],
        control="spathi",
        levels=["directed", "topology"],
        metrics=list(METRICS),
        expected_blocks=1,
    )
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id="dataset_1",
            run_id=f"{configuration}_seed_1",
            context=f"group:{group}",
            score=0.8 if configuration == "spathi" else 0.5,
            level=level,
        )
        for configuration in ("genie3", "spathi")
        for level in ("directed", "topology")
        for group in range(7)
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison, mode="descriptive"),
        datasets=[_dataset(1)],
        rows=rows,
        monkeypatch=monkeypatch,
    )

    paths = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    assert report["status"] == "contract_only"
    assert "seed_contract" not in report
    assert report["summary"] == {
        "n_analyses": 8,
        "n_confirmatory": 0,
        "n_confirmatory_inferential": 0,
        "n_contract_only": 8,
        "n_control_wins": 0,
        "n_descriptive": 0,
        "n_incomplete": 0,
        "n_inferential": 0,
    }
    assert {analysis["n_complete_blocks"] for analysis in report["analyses"]} == {1}
    assert {analysis["blocks"][0]["n_contexts"] for analysis in report["analyses"]} == {7}
    assert all(analysis["omnibus"]["p_raw"] is None for analysis in report["analyses"])
    assert load_state(run_root, "smoke")["statistics"]["files"]["statistics.json"][
        "sha256"
    ] == sha256_file(paths[0])
    assert analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root) == paths


def test_reduced_regression_report_is_strictly_descriptive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurations = ["weak", "middle", "spathi"]
    comparison = _comparison(configurations, control="spathi", expected_blocks=3)
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score={"weak": 0.2, "middle": 0.5, "spathi": 0.8}[configuration],
        )
        for dataset_index in range(1, 4)
        for configuration in configurations
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison, mode="descriptive"),
        datasets=[_dataset(index) for index in range(1, 4)],
        rows=rows,
        monkeypatch=monkeypatch,
        purpose="regression",
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Reduced statistics must not invoke inferential procedures or seeds")

    monkeypatch.setattr("spathi_bench.statistics._friedman_test", forbidden)
    monkeypatch.setattr("spathi_bench.statistics._wilcoxon_test", forbidden)
    monkeypatch.setattr("spathi_bench.statistics._bootstrap_median_interval", forbidden)
    monkeypatch.setattr("spathi_bench.statistics._analysis_seed_record", forbidden)

    statistics, *_ = analyze_statistics(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )
    report = json.loads(statistics.read_text(encoding="utf-8"))
    assert report["status"] == "descriptive_only"
    assert "seed_contract" not in report
    analysis = report["analyses"][0]
    assert analysis["analysis_role"] == "descriptive"
    assert analysis["status"] == "descriptive_only"
    assert analysis["conclusion"] == "descriptive_only"
    assert analysis["omnibus"]["p_raw"] is None
    assert analysis["omnibus"]["p_holm"] is None
    assert analysis["omnibus"]["reject"] is False
    assert analysis["omnibus"]["seed_record"] is None
    for contrast in analysis["pairwise"]:
        assert contrast["p_raw"] is None
        assert contrast["p_holm"] is None
        assert contrast["reject"] is False
        assert contrast["bootstrap_ci_low"] is None
        assert contrast["bootstrap_ci_high"] is None
        assert contrast["bootstrap_resamples"] == 0
        assert contrast["permutation_seed_record"] is None
        assert contrast["bootstrap_seed_record"] is None


def test_comparison_generation_jobs_exclude_other_jobs_from_strata_and_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(["genie3", "spathi"], control="spathi", expected_blocks=1)
    comparison["generation_jobs"] = ["selected_job"]
    comparison["stratify_by"] = ["dataset_role", "regime", "level"]
    datasets = [_dataset(1), _dataset(2)]
    datasets[0]["generation_job"] = "selected_job"
    datasets[1]["generation_job"] = "excluded_job"
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=dataset["dataset_id"],
            run_id=configuration,
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
        )
        for dataset in datasets
        for configuration in ("genie3", "spathi")
    ]
    for row in rows:
        row["generation_job"] = next(
            dataset["generation_job"]
            for dataset in datasets
            if dataset["dataset_id"] == row["dataset_id"]
        )
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=datasets,
        rows=rows,
        monkeypatch=monkeypatch,
    )

    statistics, *_ = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    report = json.loads(statistics.read_text(encoding="utf-8"))
    assert len(report["analyses"]) == 1
    analysis = report["analyses"][0]
    assert analysis["n_expected_blocks"] == 1
    assert [block["dataset_id"] for block in analysis["blocks"]] == ["dataset_1"]


def test_complete_panel_averages_contexts_then_seeds_and_runs_friedman(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurations = ["weak", "middle", "spathi"]
    comparison = _comparison(
        configurations,
        control="spathi",
        expected_blocks=3,
        dataset_role="test",
        algorithm_seeds_by_configuration={
            configuration: [1, 2] for configuration in configurations
        },
    )
    # A numeric zero is a valid evaluated empty-network score, not missing data.
    base_scores = {"weak": 0.05, "middle": 0.5, "spathi": 0.8}
    rows = []
    for dataset_index in range(1, 4):
        for configuration in configurations:
            for algorithm_seed in (1, 2):
                for group_offset, group in ((-0.05, "a"), (0.05, "b")):
                    rows.append(
                        _metric_row(
                            configuration_id=configuration,
                            dataset_id=f"dataset_{dataset_index}",
                            run_id=f"{configuration}_seed_{algorithm_seed}",
                            context=f"group:{group}",
                            score=base_scores[configuration]
                            + group_offset
                            + (algorithm_seed - 1) * 0.02,
                            dataset_role="test",
                            algorithm_seed=algorithm_seed,
                        )
                    )
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(index, dataset_role="test") for index in range(1, 4)],
        rows=rows,
        monkeypatch=monkeypatch,
        purpose="benchmark",
    )

    statistics, rankings, pairwise, _markdown = analyze_statistics(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )
    report = json.loads(statistics.read_text(encoding="utf-8"))
    analysis = report["analyses"][0]
    assert analysis["status"] == "tested"
    assert analysis["omnibus"]["statistic"] == pytest.approx(6.0)
    assert analysis["omnibus"]["p_raw"] == pytest.approx(1 / 36)
    assert analysis["omnibus"]["p_holm"] == pytest.approx(1 / 36)
    assert analysis["omnibus"]["reject"] is True
    assert analysis["blocks"][0]["scores"] == pytest.approx(
        {"weak": 0.06, "middle": 0.51, "spathi": 0.81}
    )
    assert analysis["blocks"][0]["executions_per_configuration"] == {
        "middle": 2,
        "spathi": 2,
        "weak": 2,
    }
    assert analysis["rankings"][0]["configuration_id"] == "spathi"
    assert analysis["conclusion"] == "control_top_ranked_not_distinguishable"
    seed_plan = load_seed_plan(repo_root / "experiments/v1/seeds/seed-plan.json")
    assert (
        validate_plan_seed_record(
            seed_plan,
            seed=report["seed_contract"]["root"]["seed"],
            seed_derivation=report["seed_contract"]["root"]["seed_derivation"],
        )
        == report["seed_contract"]["root"]
    )
    assert (
        validate_plan_seed_record(
            seed_plan,
            seed=analysis["omnibus"]["seed_record"]["seed"],
            seed_derivation=analysis["omnibus"]["seed_record"]["seed_derivation"],
        )
        == analysis["omnibus"]["seed_record"]
    )
    for contrast in analysis["pairwise"]:
        for key in ("permutation_seed_record", "bootstrap_seed_record"):
            record = contrast[key]
            assert (
                validate_plan_seed_record(
                    seed_plan,
                    seed=record["seed"],
                    seed_derivation=record["seed_derivation"],
                )
                == record
            )
    assert rankings.read_text(encoding="utf-8").count("\n") == 4
    assert pairwise.read_text(encoding="utf-8").count("\n") == 3


def test_complete_panel_combines_one_deterministic_run_with_seeded_replicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurations = ["deterministic", "stochastic", "spathi"]
    comparison = _comparison(
        configurations,
        control="spathi",
        expected_blocks=3,
        dataset_role="test",
        algorithm_seeds_by_configuration={
            "deterministic": [None],
            "stochastic": [1, 2],
            "spathi": [1, 2],
        },
    )
    scores = {"deterministic": 0.4, "stochastic": 0.6, "spathi": 0.8}
    rows = []
    for dataset_index in range(1, 4):
        for configuration in configurations:
            seeds = (None,) if configuration == "deterministic" else (1, 2)
            for algorithm_seed in seeds:
                rows.append(
                    _metric_row(
                        configuration_id=configuration,
                        dataset_id=f"dataset_{dataset_index}",
                        run_id=(
                            configuration
                            if algorithm_seed is None
                            else f"{configuration}_seed_{algorithm_seed}"
                        ),
                        context="group:a",
                        score=scores[configuration],
                        dataset_role="test",
                        algorithm_seed=algorithm_seed,
                    )
                )
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(index, dataset_role="test") for index in range(1, 4)],
        rows=rows,
        monkeypatch=monkeypatch,
        purpose="benchmark",
    )

    statistics, *_ = analyze_statistics(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )
    analysis = json.loads(statistics.read_text(encoding="utf-8"))["analyses"][0]
    assert analysis["status"] == "tested"
    block = analysis["blocks"][0]
    assert block["executions_per_configuration"] == {
        "deterministic": 1,
        "spathi": 2,
        "stochastic": 2,
    }
    assert block["algorithm_seeds"] == {
        "deterministic": [None],
        "spathi": [1, 2],
        "stochastic": [1, 2],
    }
    assert block["expected_algorithm_seeds"] == {
        "deterministic": [None],
        "stochastic": [1, 2],
        "spathi": [1, 2],
    }


def test_complete_declared_panel_with_unequal_design_strata_is_tested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurations = ["weak", "middle", "spathi"]
    comparison = _comparison(
        configurations,
        control="spathi",
        expected_blocks=5,
        dataset_role="test",
    )
    comparison["stratify_by"] = ["dataset_role", "regime", "level"]
    datasets = [_dataset(index, dataset_role="test") for index in range(1, 6)]
    design = [
        ("dyngen", "linear", "dyngen_linear"),
        ("dyngen", "linear", "dyngen_linear"),
        ("dyngen", "branching", "dyngen_branching"),
        ("scmultisim", "rewiring", "scmultisim_rewiring"),
        ("scmultisim", "rewiring", "scmultisim_rewiring"),
    ]
    for dataset, (simulator_id, condition_id, generation_job) in zip(datasets, design, strict=True):
        dataset.update(
            simulator_id=simulator_id,
            condition_id=condition_id,
            generation_job=generation_job,
        )
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=dataset["dataset_id"],
            run_id=configuration,
            context="group:a",
            score={"weak": 0.2, "middle": 0.5, "spathi": 0.8}[configuration],
            dataset_role="test",
        )
        for dataset in datasets
        for configuration in configurations
    ]
    by_dataset = {dataset["dataset_id"]: dataset for dataset in datasets}
    for row in rows:
        dataset = by_dataset[row["dataset_id"]]
        for field in ("simulator_id", "condition_id", "generation_job"):
            row[field] = dataset[field]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=datasets,
        rows=rows,
        monkeypatch=monkeypatch,
        purpose="benchmark",
    )

    statistics, *_ = analyze_statistics(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )
    analysis = json.loads(statistics.read_text(encoding="utf-8"))["analyses"][0]
    assert analysis["status"] == "tested"
    assert analysis["n_complete_blocks"] == 5
    assert {pair["bootstrap_method"] for pair in analysis["pairwise"]} == {
        "paired_percentile_bootstrap_fixed_simulator_condition_strata"
    }


def test_incomplete_context_panel_is_reported_and_never_pairwise_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurations = ["weak", "middle", "spathi"]
    comparison = _comparison(configurations, control="spathi", expected_blocks=2)
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=f"{configuration}_seed_1",
            context=f"group:{group}",
            score={"weak": 0.2, "middle": 0.5, "spathi": 0.8}[configuration],
        )
        for dataset_index in (1, 2)
        for configuration in configurations
        for group in ("a", "b")
        if not (dataset_index == 2 and configuration == "middle" and group == "b")
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1), _dataset(2)],
        rows=rows,
        monkeypatch=monkeypatch,
    )

    statistics, *_ = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    analysis = json.loads(statistics.read_text(encoding="utf-8"))["analyses"][0]
    assert analysis["status"] == "incomplete_panel"
    assert analysis["n_expected_blocks"] == 2
    assert analysis["n_complete_blocks"] == 1
    assert analysis["excluded_blocks"][0]["dataset_id"] == "dataset_2"
    assert any(
        reason.startswith("unbalanced_contexts:")
        for reason in analysis["excluded_blocks"][0]["reasons"]
    )
    assert analysis["omnibus"]["p_raw"] is None
    assert all(pair["p_raw"] is None for pair in analysis["pairwise"])


def test_configuration_hashes_cannot_change_under_one_configuration_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(["genie3", "spathi"], control="spathi", expected_blocks=2)
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=f"{configuration}_seed_1",
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
        )
        for dataset_index in (1, 2)
        for configuration in ("genie3", "spathi")
    ]
    rows[-1]["configuration_sha256"] = "f" * 64
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1), _dataset(2)],
        rows=rows,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(
        BenchmarkConfigError,
        match=(
            r"[Cc]onfiguration.*(?:hash|sha256|parameter identity)|"
            r"sha256.*configuration"
        ),
    ):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_different_algorithm_seed_panels_invalidate_the_predeclared_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(
        ["deterministic", "spathi"],
        control="spathi",
        expected_blocks=2,
        dataset_role="test",
    )
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=(
                configuration
                if algorithm_seed is None
                else f"{configuration}_seed_{algorithm_seed}"
            ),
            context="group:a",
            score=(
                0.5 if configuration == "deterministic" else 0.7 if algorithm_seed == 1 else 0.9
            ),
            dataset_role="test",
            algorithm_seed=algorithm_seed,
        )
        for dataset_index in (1, 2)
        for configuration, seeds in (("deterministic", (None,)), ("spathi", (1, 2)))
        for algorithm_seed in seeds
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1, dataset_role="test"), _dataset(2, dataset_role="test")],
        rows=rows,
        monkeypatch=monkeypatch,
        purpose="benchmark",
    )

    statistics, *_ = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    analysis = json.loads(statistics.read_text(encoding="utf-8"))["analyses"][0]
    assert analysis["status"] == "incomplete_panel"
    assert analysis["n_complete_blocks"] == 0
    assert all(
        any("algorithm_seed_contract" in reason for reason in block["reasons"])
        for block in analysis["excluded_blocks"]
    )


def test_declared_cardinality_mismatch_is_rejected_before_any_winner_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confirmatory = _comparison(
        ["weak", "middle", "spathi"],
        control="spathi",
        expected_blocks=4,
        dataset_role="test",
        comparison_id="confirmatory",
    )
    exploratory = _comparison(
        ["weak", "middle", "spathi"],
        control="spathi",
        expected_blocks=3,
        dataset_role="test",
        comparison_id="exploratory",
        comparison_role="exploratory",
    )
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score={"weak": 0.2, "middle": 0.5, "spathi": 0.8}[configuration],
            dataset_role="test",
        )
        for dataset_index in range(1, 4)
        for configuration in ("weak", "middle", "spathi")
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile([confirmatory, exploratory]),
        datasets=[_dataset(index, dataset_role="test") for index in range(1, 4)],
        rows=rows,
        monkeypatch=monkeypatch,
        purpose="benchmark",
    )

    with pytest.raises(BenchmarkConfigError, match="declares 4 expected blocks"):
        analyze_statistics(
            repo_root=repo_root,
            profile_id="smoke",
            run_root=run_root,
        )


def test_one_algorithm_seed_cannot_be_counted_twice_under_distinct_run_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(["genie3", "spathi"], control="spathi", expected_blocks=2)
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=(
                f"{configuration}_duplicate_{duplicate}"
                if configuration == "genie3"
                else configuration
            ),
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
            algorithm_seed=1,
        )
        for dataset_index in (1, 2)
        for configuration, duplicates in (("genie3", (1, 2)), ("spathi", (1,)))
        for duplicate in duplicates
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1), _dataset(2)],
        rows=rows,
        monkeypatch=monkeypatch,
    )

    statistics, *_ = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    analysis = json.loads(statistics.read_text(encoding="utf-8"))["analyses"][0]
    assert analysis["status"] == "incomplete_panel"
    assert all(
        any("duplicate_algorithm_seed" in reason for reason in block["reasons"])
        for block in analysis["excluded_blocks"]
    )


def test_candidate_universes_must_match_within_each_paired_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(["genie3", "spathi"], control="spathi", expected_blocks=2)
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
            n_candidates=45 if configuration == "spathi" else 36,
            n_candidate_sources=5 if configuration == "spathi" else 4,
        )
        for dataset_index in (1, 2)
        for configuration in ("genie3", "spathi")
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1), _dataset(2)],
        rows=rows,
        monkeypatch=monkeypatch,
    )

    statistics, *_ = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    analysis = json.loads(statistics.read_text(encoding="utf-8"))["analyses"][0]
    assert analysis["status"] == "incomplete_panel"
    assert analysis["n_complete_blocks"] == 0
    assert all(
        any("candidate" in reason for reason in block["reasons"])
        for block in analysis["excluded_blocks"]
    )


def test_candidate_universe_identities_must_match_when_counts_are_equal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(["genie3", "spathi"], control="spathi", expected_blocks=2)
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
            candidate_edge_universe_sha256=(
                hashlib.sha256(b"different-candidate-edges").hexdigest()
                if configuration == "spathi"
                else None
            ),
        )
        for dataset_index in (1, 2)
        for configuration in ("genie3", "spathi")
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1), _dataset(2)],
        rows=rows,
        monkeypatch=monkeypatch,
    )

    statistics, *_ = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    analysis = json.loads(statistics.read_text(encoding="utf-8"))["analyses"][0]
    assert analysis["status"] == "incomplete_panel"
    assert analysis["n_complete_blocks"] == 0
    assert all(
        any("semantic_universe" in reason for reason in block["reasons"])
        for block in analysis["excluded_blocks"]
    )


def test_every_configuration_must_include_the_predeclared_algorithm_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(
        ["genie3", "spathi"],
        control="spathi",
        expected_blocks=2,
        algorithm_seeds_by_configuration={"genie3": [1, 2], "spathi": [1, 2]},
    )
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=f"{configuration}_seed_1",
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
            algorithm_seed=1,
        )
        for dataset_index in (1, 2)
        for configuration in ("genie3", "spathi")
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1), _dataset(2)],
        rows=rows,
        monkeypatch=monkeypatch,
    )

    statistics, *_ = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    analysis = json.loads(statistics.read_text(encoding="utf-8"))["analyses"][0]
    assert analysis["status"] == "incomplete_panel"
    assert analysis["n_complete_blocks"] == 0
    assert all(
        any("algorithm_seed_contract" in reason for reason in block["reasons"])
        for block in analysis["excluded_blocks"]
    )


def test_mixed_missing_and_numeric_algorithm_seeds_have_a_stable_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(
        ["genie3", "spathi"],
        control="spathi",
        expected_blocks=2,
        algorithm_seeds_by_configuration={"genie3": [1, 2], "spathi": [1, 2]},
    )
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=(
                f"{configuration}_without_seed"
                if algorithm_seed is None
                else f"{configuration}_seed_{algorithm_seed}"
            ),
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
            algorithm_seed=algorithm_seed,
        )
        for dataset_index in (1, 2)
        for configuration, seeds in (("genie3", (None, 1)), ("spathi", (1, 2)))
        for algorithm_seed in seeds
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1), _dataset(2)],
        rows=rows,
        monkeypatch=monkeypatch,
    )

    statistics, *_ = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    analysis = json.loads(statistics.read_text(encoding="utf-8"))["analyses"][0]
    assert analysis["status"] == "incomplete_panel"
    assert all(
        any("observed=[None, 1]" in reason for reason in block["reasons"])
        for block in analysis["excluded_blocks"]
    )


@pytest.mark.parametrize(
    ("metric", "invalid_value"),
    [
        ("aupr", -0.01),
        ("auroc", 1.01),
        ("f1_at_truth_count", 1.01),
        ("epr_at_truth_count", -0.01),
    ],
)
def test_metrics_outside_their_mathematical_domain_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    invalid_value: float,
) -> None:
    comparison = _comparison(
        ["genie3", "spathi"],
        control="spathi",
        expected_blocks=2,
        metrics=[metric],
        comparison_role="exploratory",
    )
    comparison["primary_metric"] = metric
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score=0.5,
        )
        for dataset_index in (1, 2)
        for configuration in ("genie3", "spathi")
    ]
    rows[-1][metric] = invalid_value
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1), _dataset(2)],
        rows=rows,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(
        BenchmarkConfigError,
        match=rf"{metric}.*(?:domain|range)|(?:domain|range).*{metric}",
    ):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_distinct_dataset_ids_cannot_duplicate_one_independent_simulation_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(["genie3", "spathi"], control="spathi", expected_blocks=2)
    datasets = [_dataset(1), _dataset(2)]
    datasets[1]["seed"] = datasets[0]["seed"]
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
        )
        for dataset_index in (1, 2)
        for configuration in ("genie3", "spathi")
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=datasets,
        rows=rows,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(
        BenchmarkConfigError,
        match=r"[Dd]uplicate independent (?:simulation unit|block)",
    ):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_significant_adverse_result_is_reported_explicitly_for_the_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurations = ["winner", "runner", "spathi"]
    comparison = _comparison(
        configurations,
        control="spathi",
        expected_blocks=8,
        dataset_role="test",
    )
    scores = {"winner": 0.9, "runner": 0.6, "spathi": 0.1}
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score=scores[configuration] + dataset_index / 1000,
            dataset_role="test",
        )
        for dataset_index in range(1, 9)
        for configuration in configurations
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(index, dataset_role="test") for index in range(1, 9)],
        rows=rows,
        monkeypatch=monkeypatch,
        purpose="benchmark",
    )

    statistics, *_ = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    analysis = json.loads(statistics.read_text(encoding="utf-8"))["analyses"][0]
    assert analysis["omnibus"]["reject"] is True
    assert all(pair["reject"] for pair in analysis["pairwise"])
    assert all(pair["rank_biserial"] < 0 for pair in analysis["pairwise"])
    assert all(
        pair["bootstrap_confidence_level"] == pytest.approx(0.95) for pair in analysis["pairwise"]
    )
    assert analysis["conclusion"].startswith("control_significantly_worse")


def test_markdown_report_contains_omnibus_posthoc_effect_and_interval_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurations = ["weak", "middle", "spathi"]
    comparison = _comparison(
        configurations,
        control="spathi",
        expected_blocks=8,
        dataset_role="test",
    )
    scores = {"weak": 0.1, "middle": 0.5, "spathi": 0.9}
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score=scores[configuration] + dataset_index / 1000,
            dataset_role="test",
        )
        for dataset_index in range(1, 9)
        for configuration in configurations
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(index, dataset_role="test") for index in range(1, 9)],
        rows=rows,
        monkeypatch=monkeypatch,
        purpose="benchmark",
    )

    *_, markdown = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    rendered = markdown.read_text(encoding="utf-8").lower()
    assert "friedman" in rendered
    assert "kendall" in rendered
    assert "wilcoxon" in rendered
    assert "p-holm" in rendered or "holm-adjusted" in rendered
    assert "rank-biserial" in rendered or "rank biserial" in rendered
    assert "bootstrap" in rendered and "interval" in rendered


def test_exploratory_comparison_cannot_be_promoted_by_primary_metric_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configurations = ["weak", "middle", "spathi"]
    confirmatory = _comparison(
        configurations,
        control="spathi",
        expected_blocks=2,
        dataset_role="test",
    )
    exploratory = _comparison(
        configurations,
        control="spathi",
        expected_blocks=2,
        dataset_role="test",
        comparison_id="all_gene",
        comparison_role="exploratory",
    )
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score={"weak": 0.2, "middle": 0.5, "spathi": 0.8}[configuration],
            dataset_role="test",
        )
        for dataset_index in (1, 2)
        for configuration in configurations
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile([confirmatory, exploratory]),
        datasets=[_dataset(1, dataset_role="test"), _dataset(2, dataset_role="test")],
        rows=rows,
        monkeypatch=monkeypatch,
        purpose="benchmark",
    )

    statistics, *_ = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    analyses = {
        item["comparison_id"]: item
        for item in json.loads(statistics.read_text(encoding="utf-8"))["analyses"]
    }
    assert analyses["primary"]["analysis_role"] == "confirmatory"
    assert analyses["primary"]["candidate_space"] == "tf_restricted"
    assert analyses["primary"]["inference_regulator_scope_by_configuration"] == {
        "weak": "tf_restricted",
        "middle": "tf_restricted",
        "spathi": "tf_restricted",
    }
    assert analyses["all_gene"]["analysis_role"] == "exploratory"
    assert analyses["all_gene"]["candidate_space"] == "tf_restricted"
    assert analyses["all_gene"]["inference_regulator_scope_by_configuration"] == {
        "weak": "all_gene",
        "middle": "all_gene",
        "spathi": "tf_restricted",
    }
    assert analyses["all_gene"]["omnibus"]["p_holm"] is None


def test_existing_statistics_detect_tampering_and_recovers_after_state_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(["genie3", "spathi"], control="spathi", expected_blocks=2)
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
        )
        for dataset_index in (1, 2)
        for configuration in ("genie3", "spathi")
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1), _dataset(2)],
        rows=rows,
        monkeypatch=monkeypatch,
    )
    paths = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)

    state = load_state(run_root, "smoke")
    del state["statistics"]
    write_state(run_root, state)
    rankings = paths[1].read_bytes()
    paths[1].write_text("tampered-before-recovery\n", encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="Incomplete or stale"):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    paths[1].write_bytes(rankings)
    statistics_document = paths[0].read_text(encoding="utf-8")
    tampered_document = json.loads(statistics_document)
    tampered_document["status"] = "not_testable"
    paths[0].write_text(json.dumps(tampered_document), encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="Incomplete or stale"):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    paths[0].write_text(statistics_document, encoding="utf-8")
    assert analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root) == paths

    paths[1].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="missing or changed"):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def _completed_statistics_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, tuple[Path, Path, Path, Path]]:
    comparison = _comparison(["genie3", "spathi"], control="spathi", expected_blocks=2)
    rows = [
        _metric_row(
            configuration_id=configuration,
            dataset_id=f"dataset_{dataset_index}",
            run_id=configuration,
            context="group:a",
            score=0.8 if configuration == "spathi" else 0.5,
        )
        for dataset_index in (1, 2)
        for configuration in ("genie3", "spathi")
    ]
    repo_root, run_root = _run(
        tmp_path,
        profile=_profile(comparison),
        datasets=[_dataset(1), _dataset(2)],
        rows=rows,
        monkeypatch=monkeypatch,
    )
    paths = analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    return repo_root, run_root, paths


def test_completed_statistics_state_requires_canonical_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, run_root, paths = _completed_statistics_run(tmp_path, monkeypatch)
    state = load_state(run_root, "smoke")
    state["statistics"]["directory"] = str(run_root / "reports" / "alias" / ".." / "statistics")
    write_state(run_root, state)

    with pytest.raises(BenchmarkConfigError, match="not canonical"):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)

    state["statistics"]["directory"] = str(run_root / "reports" / "statistics")
    state["statistics"]["files"]["rankings.csv"]["path"] = str(
        paths[1].parent / "alias" / ".." / "rankings.csv"
    )
    write_state(run_root, state)
    with pytest.raises(BenchmarkConfigError, match="missing or changed"):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)


@pytest.mark.parametrize("symlink_kind", ["file", "ancestor"])
def test_completed_statistics_resume_rejects_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_kind: str,
) -> None:
    repo_root, run_root, paths = _completed_statistics_run(tmp_path, monkeypatch)
    if symlink_kind == "file":
        external = tmp_path / "external-rankings.csv"
        external.write_bytes(paths[1].read_bytes())
        paths[1].unlink()
        paths[1].symlink_to(external)
    else:
        statistics_dir = run_root / "reports" / "statistics"
        relocated = run_root / "reports" / "relocated-statistics"
        statistics_dir.rename(relocated)
        statistics_dir.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(BenchmarkConfigError, match="symlink"):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)


@pytest.mark.parametrize("state_gap", [False, True], ids=("completed", "recovery"))
def test_statistics_requires_exact_directory_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_gap: bool,
) -> None:
    repo_root, run_root, _paths = _completed_statistics_run(tmp_path, monkeypatch)
    if state_gap:
        state = load_state(run_root, "smoke")
        del state["statistics"]
        write_state(run_root, state)
    (run_root / "reports" / "statistics" / "unexpected.txt").write_text(
        "unexpected\n",
        encoding="utf-8",
    )

    message = "Incomplete or stale" if state_gap else "non-canonical members"
    with pytest.raises(BenchmarkConfigError, match=message):
        analyze_statistics(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_statistics_cli_dispatches_under_the_run_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outputs = tuple(
        tmp_path / name for name in ("statistics.json", "rankings.csv", "pairwise.csv", "report.md")
    )
    observed: dict[str, Any] = {}

    def fake_statistics(**kwargs: Any) -> tuple[Path, Path, Path, Path]:
        observed.update(kwargs)
        return outputs

    monkeypatch.setattr("spathi_bench.cli.analyze_statistics", fake_statistics)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = tmp_path / "run"
    assert (
        main(
            [
                "statistics",
                "--repo-root",
                str(repo_root),
                "--profile",
                "smoke",
                "--run-root",
                str(run_root),
            ]
        )
        == 0
    )
    assert observed == {
        "repo_root": repo_root.resolve(),
        "profile_id": "smoke",
        "run_root": run_root.resolve(),
    }
    assert "Statistical reports written" in capsys.readouterr().out
