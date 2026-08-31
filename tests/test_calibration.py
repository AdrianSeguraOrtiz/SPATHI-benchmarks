from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.calibration import (
    CalibrationError,
    CandidateConfiguration,
    build_andrea_requests,
    expand_stage_candidates,
    expand_stage_runs,
    expand_validation_runs,
    load_search_space,
    select_stage_candidates,
    select_validation_candidate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate_by_variant(
    candidates: tuple[CandidateConfiguration, ...],
    variant_id: str,
) -> CandidateConfiguration:
    return next(candidate for candidate in candidates if candidate.variant_id == variant_id)


def _summary_rows(
    candidates: tuple[CandidateConfiguration, ...],
    *,
    role: str,
    seeds: tuple[int, ...],
    datasets: tuple[str, ...] = ("dataset_a", "dataset_b"),
    aupr: dict[tuple[str, str, int], float] | None = None,
    wall_time: dict[str, float] | None = None,
    peak_memory: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    aupr = aupr or {}
    wall_time = wall_time or {}
    peak_memory = peak_memory or {}
    return [
        {
            "configuration_id": candidate.configuration_id,
            "configuration_sha256": candidate.configuration_sha256,
            "dataset_role": role,
            "dataset_id": dataset_id,
            "algorithm_seed": seed,
            "run_id": (
                f"validation__{candidate.configuration_id}__seed_{seed}"
                if role == "validation"
                else f"{candidate.configuration_id}__seed_{seed}"
            ),
            "execution_sha256": "1" * 64,
            "aupr_directed": aupr.get(
                (candidate.configuration_id, dataset_id, seed),
                0.5,
            ),
            "wall_time_seconds": wall_time.get(candidate.configuration_id, 10.0),
            "peak_memory_mb": peak_memory.get(candidate.configuration_id, 100.0),
            "n_group_contexts": 2,
            "group_contexts": ["group:a", "group:b"],
        }
        for candidate in candidates
        for dataset_id in datasets
        for seed in seeds
    ]


def test_loads_validated_search_space() -> None:
    search_space = load_search_space(REPO_ROOT)

    assert search_space["id"] == "spathi_calibration_v1"
    assert search_space["smoke_contract"]["configuration_id"] == "spathi_smoke"
    assert search_space["smoke_contract"]["eligible_for_scientific_selection"] is False


def test_expands_convergence_deterministically_with_all_seeds() -> None:
    search_space = load_search_space(REPO_ROOT)

    candidates = expand_stage_candidates(search_space, "convergence")
    runs = expand_stage_runs(search_space, "convergence")

    assert [candidate.variant_id for candidate in candidates] == [
        "reference",
        "trees_100",
        "trees_250",
        "trees_1000",
    ]
    assert len(runs) == 20
    assert runs[0].run_id == "convergence__reference__seed_1729"
    assert [run.algorithm_seed for run in runs[:5]] == [1729, 2718, 31415, 57721, 104729]
    assert all(run.dataset_role == "development" for run in runs)
    assert len({candidate.configuration_sha256 for candidate in candidates}) == 4
    assert runs[0].parameters["random_seed"] == 1729


def test_renders_calibration_runs_as_standard_andrea_requests() -> None:
    search_space = load_search_space(REPO_ROOT)
    runs = expand_stage_runs(search_space, "convergence")

    tools, custom, metadata = build_andrea_requests(runs, docker_image="spathi:locked")

    assert len(tools["runs"]) == len(custom["tools"]) == len(metadata["runs"]) == 20
    assert tools["runs"][0]["tool_id"] == f"custom_{runs[0].run_id}"
    assert tools["runs"][0]["params"]["random_seed"] == 1729
    assert custom["tools"][0]["docker_image"] == "spathi:locked"
    assert metadata["runs"][0]["algorithm_seed_parameter"] == "random_seed"


def test_later_stages_are_bounded_and_parent_order_independent() -> None:
    search_space = load_search_space(REPO_ROOT)
    convergence = expand_stage_candidates(search_space, "convergence")
    convergence_parent = _candidate_by_variant(convergence, "trees_250")
    structures = expand_stage_candidates(
        search_space,
        "structure",
        parents=(convergence_parent,),
    )

    assert len(structures) == 7
    assert len(expand_stage_runs(search_space, "structure", parents=(convergence_parent,))) == 21

    first_parent = _candidate_by_variant(structures, "inherited")
    second_parent = _candidate_by_variant(structures, "group_distance")
    forward = expand_stage_runs(
        search_space,
        "estimator",
        parents=(first_parent, second_parent),
    )
    reversed_order = expand_stage_runs(
        search_space,
        "estimator",
        parents=(second_parent, first_parent),
    )

    assert len(forward) == 30
    assert [run.run_id for run in forward] == [run.run_id for run in reversed_order]


def test_expansion_rejects_missing_wrong_or_excess_parents() -> None:
    search_space = load_search_space(REPO_ROOT)
    convergence = expand_stage_candidates(search_space, "convergence")

    with pytest.raises(CalibrationError, match="requires parent"):
        expand_stage_candidates(search_space, "structure")
    with pytest.raises(CalibrationError, match="requires parents from 'structure'"):
        expand_stage_candidates(search_space, "estimator", parents=(convergence[0],))

    structure = expand_stage_candidates(search_space, "structure", parents=(convergence[0],))
    with pytest.raises(CalibrationError, match="at most 2 parents"):
        expand_stage_candidates(search_space, "estimator", parents=tuple(structure[:3]))


def test_stage_selection_uses_hierarchical_aupr_before_tie_breakers() -> None:
    search_space = load_search_space(REPO_ROOT)
    candidates = expand_stage_candidates(search_space, "convergence")
    seeds = (1729, 2718, 31415, 57721, 104729)
    winner = _candidate_by_variant(candidates, "trees_100")
    aupr = {(winner.configuration_id, "dataset_a", seed): 0.9 for seed in seeds} | {
        (winner.configuration_id, "dataset_b", seed): 0.3 for seed in seeds
    }
    rows = _summary_rows(
        candidates,
        role="development",
        seeds=seeds,
        aupr=aupr,
        wall_time={winner.configuration_id: 1000.0},
        peak_memory={winner.configuration_id: 1000.0},
    )

    selected = select_stage_candidates(search_space, "convergence", candidates, rows)

    assert len(selected) == 1
    assert selected[0].candidate == winner
    assert selected[0].hierarchical_aupr == pytest.approx(0.6)
    assert selected[0].dataset_count == 2
    assert selected[0].algorithm_seed_count == 5


def test_tie_breakers_apply_stability_then_cost_then_simplicity() -> None:
    search_space = load_search_space(REPO_ROOT)
    candidates = expand_stage_candidates(search_space, "convergence")
    seeds = (1729, 2718, 31415, 57721, 104729)
    reference = _candidate_by_variant(candidates, "reference")
    unstable = _candidate_by_variant(candidates, "trees_100")
    aupr = {
        (unstable.configuration_id, dataset_id, seed): value
        for dataset_id in ("dataset_a", "dataset_b")
        for seed, value in zip(seeds, (0.3, 0.4, 0.5, 0.6, 0.7), strict=True)
    }

    stable_selection = select_stage_candidates(
        search_space,
        "convergence",
        candidates,
        _summary_rows(
            candidates,
            role="development",
            seeds=seeds,
            aupr=aupr,
            wall_time={
                **{candidate.configuration_id: 100.0 for candidate in candidates},
                unstable.configuration_id: 1.0,
            },
        ),
    )
    assert stable_selection[0].candidate == reference

    cheap = _candidate_by_variant(candidates, "trees_250")
    cost_selection = select_stage_candidates(
        search_space,
        "convergence",
        candidates,
        _summary_rows(
            candidates,
            role="development",
            seeds=seeds,
            wall_time={cheap.configuration_id: 1.0},
        ),
    )
    assert cost_selection[0].candidate == cheap

    simplicity_selection = select_stage_candidates(
        search_space,
        "convergence",
        candidates,
        _summary_rows(candidates, role="development", seeds=seeds),
    )
    assert simplicity_selection[0].candidate == reference


def test_selection_rejects_role_mixing_and_unbalanced_rows() -> None:
    search_space = load_search_space(REPO_ROOT)
    candidates = expand_stage_candidates(search_space, "convergence")
    seeds = (1729, 2718, 31415, 57721, 104729)
    rows = _summary_rows(candidates, role="development", seeds=seeds)
    rows[0]["dataset_role"] = "test"

    with pytest.raises(CalibrationError, match="expected 'development'"):
        select_stage_candidates(search_space, "convergence", candidates, rows)

    unbalanced = _summary_rows(candidates, role="development", seeds=seeds)
    unbalanced.pop()
    with pytest.raises(CalibrationError, match="same algorithmic seeds"):
        select_stage_candidates(search_space, "convergence", candidates, unbalanced)

    inconsistent_contexts = _summary_rows(candidates, role="development", seeds=seeds)
    inconsistent_contexts[0]["group_contexts"] = ["group:a"]
    inconsistent_contexts[0]["n_group_contexts"] = 1
    with pytest.raises(CalibrationError, match="same group contexts"):
        select_stage_candidates(
            search_space,
            "convergence",
            candidates,
            inconsistent_contexts,
        )


def test_validation_selects_only_from_bounded_estimator_finalists() -> None:
    search_space = load_search_space(REPO_ROOT)
    convergence = expand_stage_candidates(search_space, "convergence")
    structures = expand_stage_candidates(
        search_space,
        "structure",
        parents=(convergence[0],),
    )
    estimators = expand_stage_candidates(
        search_space,
        "estimator",
        parents=(structures[0],),
    )
    finalists = tuple(estimators[:3])
    validation_runs = expand_validation_runs(search_space, finalists)
    assert len(validation_runs) == 9
    assert all(run.dataset_role == "validation" for run in validation_runs)
    seeds = (1729, 31415, 104729)
    best = finalists[1]
    aupr = {
        (best.configuration_id, dataset_id, seed): 0.7
        for dataset_id in ("dataset_a", "dataset_b")
        for seed in seeds
    }

    selected = select_validation_candidate(
        search_space,
        finalists,
        _summary_rows(finalists, role="validation", seeds=seeds, aupr=aupr),
    )
    assert selected.candidate == best

    with pytest.raises(CalibrationError, match="expected 'validation'"):
        select_validation_candidate(
            search_space,
            finalists,
            _summary_rows(finalists, role="test", seeds=seeds, aupr=aupr),
        )
    with pytest.raises(CalibrationError, match="between 2 and 3"):
        select_validation_candidate(
            search_space,
            tuple(estimators[:4]),
            _summary_rows(tuple(estimators[:4]), role="validation", seeds=seeds),
        )


def test_candidate_role_cannot_be_relabelled_for_search() -> None:
    search_space = load_search_space(REPO_ROOT)
    candidates = expand_stage_candidates(search_space, "convergence")
    contaminated = (replace(candidates[0], dataset_role="validation"), *candidates[1:])

    with pytest.raises(CalibrationError, match="must use development"):
        select_stage_candidates(search_space, "convergence", contaminated, [])


def test_candidate_parameters_cannot_be_mutated_after_expansion() -> None:
    search_space = load_search_space(REPO_ROOT)
    candidates = expand_stage_candidates(search_space, "convergence")
    mutated_parameters = dict(candidates[0].parameters)
    mutated_parameters["n_estimators"] = 1
    contaminated = (
        replace(candidates[0], parameters=mutated_parameters),
        *candidates[1:],
    )

    with pytest.raises(CalibrationError, match="parameter digest is inconsistent"):
        select_stage_candidates(search_space, "convergence", contaminated, [])
