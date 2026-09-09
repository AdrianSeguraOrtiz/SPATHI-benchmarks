from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.calibration_materialization as calibration_materialization_module
from spathi_bench.calibration import (
    CalibrationError,
    CandidateConfiguration,
    build_andrea_requests,
    build_stage_profile,
    expand_stage_candidates,
    expand_stage_runs,
    expand_validation_runs,
    load_search_space,
    materialize_group_policy,
    rank_stage_candidates,
    rank_validation_candidates,
    resolve_group_policy,
    stage_profile_id,
    validate_calibration_panel,
)
from spathi_bench.calibration_design import calibration_ledger, calibration_phase
from spathi_bench.calibration_materialization import materialize_calibration_stage
from spathi_bench.config import (
    BenchmarkConfigError,
    load_json_object,
    validate_profile_references,
)
from spathi_bench.pipeline import _parameter_hashes

REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate_by_variant(
    candidates: tuple[CandidateConfiguration, ...], variant_id: str
) -> CandidateConfiguration:
    return next(candidate for candidate in candidates if candidate.variant_id == variant_id)


def _seeds(search: dict[str, Any], budget: str, stage: str) -> tuple[int, ...]:
    return tuple(search["calibration_budgets"][budget]["algorithmic_seeds"][stage])


def _phase_seeds(search: dict[str, Any], budget: str, stage: str, fidelity: str) -> tuple[int, ...]:
    seeds = _seeds(search, budget, stage)
    phase = calibration_phase(search, budget, stage, fidelity)
    return tuple(seeds[index - 1] for index in phase.algorithm_seed_indices)


def _dataset_ids(count: int) -> tuple[str, ...]:
    return tuple(f"dataset_{index:03d}" for index in range(1, count + 1))


def _copy_calibration_repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for directory in ("schemas", "experiments"):
        shutil.copytree(REPO_ROOT / directory, repo / directory)
    for filename in ("images.lock.json", "repos.lock.json"):
        shutil.copyfile(REPO_ROOT / filename, repo / filename)
    return repo


def _summary_rows(
    candidates: tuple[CandidateConfiguration, ...],
    *,
    role: str,
    seeds: tuple[int, ...],
    fidelity: str,
    dataset_ids: tuple[str, ...] = ("dataset_a", "dataset_b"),
    aupr: dict[tuple[str, str, int], float] | None = None,
    wall_time: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    aupr = aupr or {}
    wall_time = wall_time or {}
    return [
        {
            "configuration_id": candidate.configuration_id,
            "configuration_sha256": candidate.configuration_sha256,
            "dataset_role": role,
            "dataset_id": dataset_id,
            "algorithm_seed": seed,
            "run_id": (
                f"validation__{candidate.configuration_id}__final__seed_{seed}"
                if role == "validation"
                else f"{candidate.configuration_id}__{fidelity}__seed_{seed}"
            ),
            "execution_sha256": "1" * 64,
            "aupr_directed": aupr.get((candidate.configuration_id, dataset_id, seed), 0.5),
            "wall_time_seconds": wall_time.get(candidate.configuration_id, 10.0),
            "peak_memory_mb": 100.0,
            "n_group_contexts": 2,
            "group_contexts": ["group:a", "group:b"],
        }
        for candidate in candidates
        for dataset_id in dataset_ids
        for seed in seeds
    ]


def _stage_chain(search: dict[str, Any]) -> dict[str, tuple[CandidateConfiguration, ...]]:
    geometry = expand_stage_candidates(search, "geometry")
    estimator = expand_stage_candidates(search, "estimator", parents=geometry[:3])
    estimator_parents = tuple(
        next(
            candidate
            for candidate in estimator
            if candidate.parent_configuration_id == parent.configuration_id
            and candidate.variant_id == "inherited"
        )
        for parent in geometry[:3]
    )
    interaction = expand_stage_candidates(search, "interaction", parents=estimator_parents)
    interaction_parents = tuple(
        next(
            candidate
            for candidate in interaction
            if candidate.parent_configuration_id == parent.configuration_id
            and candidate.variant_id == "extra_trees_sqrt_leaf_2"
        )
        for parent in estimator_parents
    )
    scale_transfer = expand_stage_candidates(
        search, "estimator_scale_transfer", parents=interaction_parents
    )
    convergence = expand_stage_candidates(search, "convergence", parents=scale_transfer[:2])
    return {
        "geometry": geometry,
        "estimator": estimator,
        "interaction": interaction,
        "estimator_scale_transfer": scale_transfer,
        "convergence": convergence,
    }


def _panel_for_rows(
    rows: list[dict[str, Any]],
    *,
    contexts: tuple[str, ...] = ("group:a", "group:b"),
) -> dict[str, Any]:
    dataset_ids = sorted({row["dataset_id"] for row in rows})
    run_ids = sorted({row["run_id"] for row in rows})
    return {
        "schema_version": "1.0",
        "dataset_role": rows[0]["dataset_role"],
        "dataset_ids": dataset_ids,
        "run_ids": run_ids,
        "n_expected_executions": len(dataset_ids) * len(run_ids),
        "handoff_validation_sha256": "a" * 64,
        "datasets": [
            {
                "dataset_id": dataset_id,
                "expected_group_contexts": list(contexts),
                "n_group_contexts": len(contexts),
                "dataset_manifest_sha256": "b" * 64,
                "group_assignments_sha256": "c" * 64,
                "ground_truth_manifest_sha256": "d" * 64,
                "truth_network_sha256": "e" * 64,
            }
            for dataset_id in dataset_ids
        ],
    }


def test_calibration_panel_rejects_systematic_context_omission() -> None:
    search = load_search_space(REPO_ROOT)
    candidate = expand_stage_candidates(search, "geometry")[:1]
    rows = _summary_rows(
        candidate,
        role="development",
        seeds=_seeds(search, "full", "geometry"),
        fidelity="scout",
    )
    panel = _panel_for_rows(rows, contexts=("group:a", "group:b", "group:c"))

    with pytest.raises(CalibrationError, match="does not cover its declared group contexts"):
        validate_calibration_panel(panel, rows, expected_role="development")


def test_calibration_panel_requires_the_complete_dataset_run_product() -> None:
    search = load_search_space(REPO_ROOT)
    candidate = expand_stage_candidates(search, "geometry")[:1]
    rows = _summary_rows(
        candidate,
        role="development",
        seeds=_seeds(search, "full", "geometry"),
        fidelity="scout",
    )
    panel = _panel_for_rows(rows)
    validate_calibration_panel(panel, rows, expected_role="development")

    with pytest.raises(CalibrationError, match="omits planned executions"):
        validate_calibration_panel(panel, rows[:-1], expected_role="development")


def test_search_is_bounded_staged_and_covers_current_scientific_parameters() -> None:
    search = load_search_space(REPO_ROOT)

    assert [stage["id"] for stage in search["stages"]] == [
        "geometry",
        "estimator",
        "interaction",
        "estimator_scale_transfer",
        "convergence",
    ]
    assert search["stages"][-1]["strategy"] == "fixed_ceiling_sweep"
    geometry_overrides = [item["parameter_overrides"] for item in search["stages"][0]["candidates"]]
    estimator_overrides = [
        item["parameter_overrides"] for item in search["stages"][1]["candidates"]
    ]
    assert {20, 100} <= {
        item["n_components"] for item in geometry_overrides if "n_components" in item
    }
    assert {"auto", "full", "randomized"} == {
        search["base_configuration"]["pca_svd_solver"],
        *(item["pca_svd_solver"] for item in geometry_overrides if "pca_svd_solver" in item),
    }
    assert {"none", "standard"} <= {
        search["base_configuration"]["distance_standardization"],
        *(
            item["distance_standardization"]
            for item in geometry_overrides
            if "distance_standardization" in item
        ),
    }
    observed_max_features = [
        search["base_configuration"]["max_features"],
        *[item["max_features"] for item in estimator_overrides if "max_features" in item],
    ]
    assert {"sqrt", "log2", 0.25, 0.5} <= set(observed_max_features)
    assert any(type(item) is int and item == 1 for item in observed_max_features)
    assert any(type(item) is float and item == 1.0 for item in observed_max_features)
    assert {2, 5} <= {
        item["min_samples_leaf"] for item in estimator_overrides if "min_samples_leaf" in item
    }
    assert any(item.get("max_depth") == 20 for item in estimator_overrides)
    assert any(item.get("bootstrap") is True for item in estimator_overrides)
    assert any(
        item.get("tree_method") == "random-forest" and item.get("bootstrap") is False
        for item in estimator_overrides
    )
    assert {0.5, 2.0} <= {
        item["bandwidth_scale"] for item in geometry_overrides if "bandwidth_scale" in item
    }
    assert search["base_configuration"]["distance_metric"] == "cosine"
    assert search["base_configuration"]["n_estimators"] == 250
    assert search["base_configuration"]["max_features"] == "sqrt"
    interaction_stage = search["stages"][2]
    scale_transfer_stage = search["stages"][3]
    assert interaction_stage["include_inherited_configuration"] is False
    assert interaction_stage["parent_expansion_identity"] == {
        "excluded_parameters": [
            "tree_method",
            "max_features",
            "min_samples_leaf",
            "max_depth",
            "bootstrap",
        ],
        "required_distinct_parents": 3,
    }
    assert scale_transfer_stage["parent_expansion_identity"] == {
        "excluded_parameters": ["max_features"],
        "required_distinct_parents": 3,
    }
    assert search["calibration_constraints"]["fixed_parameters"] == {
        "bandwidth": "auto",
        "target_eligibility": "all",
        "adaptive_trees": False,
    }


def test_multifidelity_ledgers_are_exact_and_prospectively_independent() -> None:
    search = load_search_space(REPO_ROOT)
    expected = {
        "reduced": (
            [64, 18, 144, 24, 72, 24, 18, 3, 32, 32],
            [64, 18, 144, 24, 72, 24, 18, 3, 128, 32],
            431,
            527,
        ),
        "full": (
            [112, 168, 288, 288, 144, 192, 54, 36, 240, 240],
            [112, 168, 288, 288, 144, 192, 54, 36, 960, 240],
            1762,
            2482,
        ),
    }
    for budget, (physical, logical, physical_total, logical_total) in expected.items():
        ledger = calibration_ledger(search, budget)
        assert [phase.expected_physical_fits for phase in ledger] == physical
        assert [phase.expected_logical_results for phase in ledger] == logical
        assert sum(physical) == physical_total
        assert sum(logical) == logical_total
        contract = search["calibration_budgets"][budget]["multifidelity"]
        assert contract["independent_profile"] is True
        assert contract["may_consume_other_profile_outcomes"] is False

    for stage in (
        "geometry",
        "estimator",
        "interaction",
        "estimator_scale_transfer",
    ):
        for budget in ("reduced", "full"):
            scout = calibration_phase(search, budget, stage, "scout")
            assert scout.candidate_source == "all_expanded"
            assert scout.candidate_count > scout.advance_limit


def test_high_tf_scout_is_hash_only_and_confirmation_uses_all_targets() -> None:
    search = load_search_space(REPO_ROOT)
    for budget in ("reduced", "full"):
        scout = calibration_phase(search, budget, "estimator_scale_transfer", "scout")
        confirmation = calibration_phase(search, budget, "estimator_scale_transfer", "confirmation")
        assert scout.target_scope == {
            "method": "sha256_gene_id",
            "size": 256,
            "salt": "spathi-calibration-high-tf-v1",
        }
        assert confirmation.target_scope == {"method": "all"}


def test_conditional_policy_has_one_total_routing_rule_and_shared_estimators() -> None:
    search = load_search_space(REPO_ROOT)
    policy = materialize_group_policy(search, search["base_configuration"])

    single = resolve_group_policy(policy, 1)
    multi = resolve_group_policy(policy, 2)
    assert (single["weight_mode"], single["distance_metric"], single["group_size_correction"]) == (
        "cell-distance",
        "euclidean",
        "none",
    )
    assert multi == search["base_configuration"]
    for parameter in policy["shared_parameters"]:
        assert single[parameter] == multi[parameter]
    with pytest.raises(CalibrationError, match="positive integer"):
        resolve_group_policy(policy, 0)


def test_geometry_expansion_and_requests_are_deterministic_and_policy_aware() -> None:
    search = load_search_space(REPO_ROOT)
    candidates = expand_stage_candidates(search, "geometry")
    runs = expand_stage_runs(search, "geometry", "full", "scout")
    seeds = _phase_seeds(search, "full", "geometry", "scout")

    assert candidates[0].variant_id == "base"
    assert len(candidates) == 16
    assert len(runs) == len(candidates) * len(seeds)
    assert [run.algorithm_seed for run in runs[: len(seeds)]] == list(seeds)
    assert runs[0].parameters["random_seed"] == seeds[0]
    assert resolve_group_policy(runs[0].parameters["group_policy"], 1)["weight_mode"] == (
        "cell-distance"
    )
    tools, custom, metadata = build_andrea_requests(
        runs,
        docker_image="spathi:locked",
        calibration_budget="full",
    )
    assert len(tools["runs"]) == len(custom["tools"]) == len(metadata["runs"]) == len(runs)
    assert tools["runs"][0]["params"] == runs[0].parameters
    assert metadata["runs"][0]["configuration_sha256"] == runs[0].configuration_sha256
    assert metadata["runs"][0]["selection_role"] == "base"


def test_materialized_geometry_profile_validates_the_complete_canonical_run(
    tmp_path: Path,
) -> None:
    repo = _copy_calibration_repository(tmp_path)
    bundle = repo / "work/calibration/reduced/geometry/requests"
    profile_path = repo / "work/calibration/reduced/geometry/profile.json"
    materialize_calibration_stage(
        repo_root=repo,
        stage_id="geometry",
        calibration_budget="reduced",
        fidelity="scout",
        parent_ranking_path=None,
        output_dir=bundle,
        profile_output=profile_path,
    )
    profile = load_json_object(profile_path, "materialized geometry profile")

    validate_profile_references(repo, profile)

    tools_path = bundle / "tools-params.json"
    tools = load_json_object(tools_path, "materialized geometry requests")
    metadata_path = bundle / "run-metadata.json"
    metadata = load_json_object(metadata_path, "materialized run metadata")
    manifest = load_json_object(bundle / "calibration-manifest.json", "calibration manifest")
    assert _parameter_hashes(tools["runs"][0], metadata["runs"][0]) == (
        manifest["runs"][0]["configuration_sha256"],
        manifest["runs"][0]["execution_sha256"],
    )

    metadata["runs"][0]["configuration_sha256"] = "f" * 64
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkConfigError, match="canonical calibration run"):
        validate_profile_references(repo, profile)
    metadata["runs"][0]["configuration_sha256"] = manifest["runs"][0]["configuration_sha256"]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tools["runs"][0]["params"]["group_policy"]["branches"]["single_group"]["parameters"][
        "n_components"
    ] += 1
    tools_path.write_text(
        json.dumps(tools, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkConfigError, match="differs from its canonical calibration run"):
        validate_profile_references(repo, profile)


def test_materialized_convergence_profile_attaches_its_plan_before_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _copy_calibration_repository(tmp_path)
    parents = _stage_chain(load_search_space(repo))["estimator_scale_transfer"][:2]
    monkeypatch.setattr(
        calibration_materialization_module,
        "load_parent_ranking",
        lambda *args, **kwargs: (parents, None),
    )
    monkeypatch.setattr(
        calibration_materialization_module,
        "validate_calibration_manifest",
        lambda *args, **kwargs: None,
    )
    bundle = repo / "work/calibration/reduced/convergence/requests"
    profile_path = repo / "work/calibration/reduced/convergence/profile.json"

    materialize_calibration_stage(
        repo_root=repo,
        stage_id="convergence",
        calibration_budget="reduced",
        fidelity="final",
        parent_ranking_path=repo / "work/parent-ranking.json",
        output_dir=bundle,
        profile_output=profile_path,
    )

    profile = load_json_object(profile_path, "materialized convergence profile")
    assert profile["inference_batches"][0]["convergence_plans"] == (
        "work/calibration/reduced/convergence/requests/convergence-plans.json"
    )
    plans = load_json_object(bundle / "convergence-plans.json", "convergence plans")
    assert plans["artifact_type"] == "spathi_calibration_convergence_plans"
    assert plans["physical_plans_per_dataset"] == 4
    assert plans["logical_results_per_dataset"] == 16


def test_reduced_and_full_algorithmic_seed_domains_are_disjoint() -> None:
    search = load_search_space(REPO_ROOT)

    for stage in (
        "geometry",
        "estimator",
        "estimator_scale_transfer",
        "interaction",
        "convergence",
        "validation",
    ):
        assert set(_seeds(search, "reduced", stage)).isdisjoint(_seeds(search, "full", stage))
    assert stage_profile_id("interaction", "reduced", "confirmation") == (
        "calibration-reduced-interaction-confirmation"
    )
    assert stage_profile_id("validation", "full", "final") == ("selection-full-validation-final")


def test_stage_profile_cannot_mix_budget_and_template() -> None:
    template = json.loads(
        (REPO_ROOT / "experiments/v1/profiles/calibration-reduced.json").read_text()
    )
    profile = build_stage_profile(
        template,
        "geometry",
        "reduced",
        "scout",
        bundle_path="work/calibration/reduced/geometry/plan",
        search_space=load_search_space(REPO_ROOT),
    )
    assert profile["id"] == "calibration-reduced-geometry-scout"
    assert {job["split_id"] for job in profile["generation_jobs"]} == {"development_v1"}
    scale_profile = build_stage_profile(
        template,
        "estimator_scale_transfer",
        "reduced",
        "scout",
        bundle_path="work/calibration/reduced/estimator-scale-transfer/plan",
        search_space=load_search_space(REPO_ROOT),
    )
    assert [job["condition_id"] for job in scale_profile["generation_jobs"]] == [
        "scmultisim_estimator_scale_transfer"
    ]
    assert scale_profile["generation_jobs"][0]["split_id"] == (
        "development_estimator_scale_transfer_v1"
    )
    with pytest.raises(CalibrationError, match="full.*template"):
        build_stage_profile(
            template,
            "geometry",
            "full",
            "scout",
            bundle_path="work/calibration/full/geometry/plan",
            search_space=load_search_space(REPO_ROOT),
        )


def test_later_stages_are_bounded_and_parent_order_independent() -> None:
    search = load_search_space(REPO_ROOT)
    geometry = expand_stage_candidates(search, "geometry")
    parents = (geometry[0], geometry[1], geometry[2])
    forward = expand_stage_candidates(search, "estimator", parents=parents)
    reverse = expand_stage_candidates(search, "estimator", parents=tuple(reversed(parents)))

    assert forward == reverse
    assert len(forward) == 3 * 12
    estimator_parents = tuple(
        next(
            candidate
            for candidate in forward
            if candidate.parent_configuration_id == parent.configuration_id
            and candidate.variant_id == "inherited"
        )
        for parent in parents
    )
    interaction = expand_stage_candidates(search, "interaction", parents=estimator_parents)
    interaction_parents = tuple(
        next(
            candidate
            for candidate in interaction
            if candidate.parent_configuration_id == parent.configuration_id
            and candidate.variant_id == "extra_trees_sqrt_leaf_2"
        )
        for parent in estimator_parents
    )
    scale_transfer = expand_stage_candidates(
        search, "estimator_scale_transfer", parents=interaction_parents
    )
    expected_scale_variants = {
        "high_tf_features_sqrt",
        "high_tf_features_log2",
        "high_tf_features_one",
        "high_tf_features_fraction_025",
        "high_tf_features_fraction_050",
        "high_tf_features_fraction_100",
    }
    assert len(scale_transfer) == 3 * len(expected_scale_variants)
    for parent in interaction_parents:
        assert {
            candidate.variant_id
            for candidate in scale_transfer
            if candidate.parent_configuration_id == parent.configuration_id
        } == expected_scale_variants
    assert all(candidate.variant_id != "inherited" for candidate in scale_transfer)
    convergence = expand_stage_candidates(search, "convergence", parents=scale_transfer[:2])
    assert len(interaction) == 3 * 6
    assert all(candidate.variant_id != "inherited" for candidate in interaction)
    assert len(convergence) == 2 * 4
    assert {candidate.parameters["n_estimators"] for candidate in convergence} == {
        250,
        500,
        750,
        1000,
    }


def test_scale_transfer_rejects_equivalent_parent_expansions() -> None:
    search = load_search_space(REPO_ROOT)
    geometry = expand_stage_candidates(search, "geometry")[0]
    estimator = next(
        candidate
        for candidate in expand_stage_candidates(search, "estimator", parents=(geometry,))
        if candidate.variant_id == "inherited"
    )
    interactions = expand_stage_candidates(search, "interaction", parents=(estimator,))
    parents = tuple(
        candidate
        for candidate in interactions
        if candidate.variant_id in {"extra_trees_sqrt_leaf_2", "extra_trees_fraction_050_leaf_2"}
    )

    assert len(parents) == 2
    with pytest.raises(CalibrationError, match="duplicate effective configurations"):
        expand_stage_candidates(
            search,
            "estimator_scale_transfer",
            parents=parents,
        )


def test_interaction_rejects_parents_that_only_differ_in_overwritten_parameters() -> None:
    search = load_search_space(REPO_ROOT)
    geometry = expand_stage_candidates(search, "geometry")[0]
    estimators = expand_stage_candidates(search, "estimator", parents=(geometry,))
    parents = tuple(
        candidate
        for candidate in estimators
        if candidate.variant_id in {"extra_trees_features_log2", "extra_trees_leaf_2"}
    )

    assert len(parents) == 2
    with pytest.raises(CalibrationError, match="duplicate effective configurations"):
        expand_stage_candidates(search, "interaction", parents=parents)


def test_scout_run_expansion_requires_the_declared_candidate_cardinality() -> None:
    search = load_search_space(REPO_ROOT)
    geometry = expand_stage_candidates(search, "geometry")
    with pytest.raises(CalibrationError, match="requires exactly 36 distinct logical candidates"):
        expand_stage_runs(
            search,
            "estimator",
            "full",
            "scout",
            parents=(geometry[0],),
        )


def test_expansion_rejects_missing_wrong_or_excess_parents() -> None:
    search = load_search_space(REPO_ROOT)
    geometry = expand_stage_candidates(search, "geometry")
    with pytest.raises(CalibrationError, match="requires parent"):
        expand_stage_candidates(search, "estimator")
    with pytest.raises(CalibrationError, match="requires parents from 'estimator'"):
        expand_stage_candidates(search, "interaction", parents=(geometry[0],))
    with pytest.raises(CalibrationError, match="at most 3 parents"):
        expand_stage_candidates(search, "estimator", parents=geometry[:4])


def test_stage_selection_uses_aupr_before_stability_and_simplicity() -> None:
    search = load_search_space(REPO_ROOT)
    candidates = expand_stage_candidates(search, "geometry")
    phase = calibration_phase(search, "full", "geometry", "scout")
    seeds = _phase_seeds(search, "full", "geometry", "scout")
    dataset_ids = _dataset_ids(phase.dataset_count)
    winner = candidates[1]
    aupr = {
        (winner.configuration_id, dataset_id, seed): score
        for dataset_id, score in zip(
            dataset_ids,
            (0.9 if index % 2 == 0 else 0.3 for index in range(len(dataset_ids))),
            strict=True,
        )
        for seed in seeds
    }
    ranking = rank_stage_candidates(
        search,
        "geometry",
        candidates,
        _summary_rows(
            candidates,
            role="development",
            seeds=seeds,
            fidelity="scout",
            dataset_ids=dataset_ids,
            aupr=aupr,
            wall_time={winner.configuration_id: 1000.0},
        ),
        calibration_budget="full",
        fidelity="scout",
    )
    assert ranking[0].candidate == winner
    assert ranking[0].hierarchical_aupr == pytest.approx(
        sum(0.9 if index % 2 == 0 else 0.3 for index in range(len(dataset_ids))) / len(dataset_ids)
    )


def test_calibration_does_not_use_runtime_as_a_selection_tie_breaker() -> None:
    search = load_search_space(REPO_ROOT)
    candidates = expand_stage_candidates(search, "geometry")
    by_simplicity: dict[int, list[CandidateConfiguration]] = {}
    for candidate in candidates:
        by_simplicity.setdefault(candidate.base_parameter_changes, []).append(candidate)
    tied = next(items[:2] for items in by_simplicity.values() if len(items) >= 2)
    expected = min(tied, key=lambda candidate: candidate.configuration_id)
    phase = calibration_phase(search, "full", "geometry", "scout")
    seeds = _phase_seeds(search, "full", "geometry", "scout")
    dataset_ids = _dataset_ids(phase.dataset_count)
    aupr = {
        (candidate.configuration_id, dataset_id, seed): (0.8 if candidate in tied else 0.1)
        for candidate in candidates
        for dataset_id in dataset_ids
        for seed in seeds
    }
    wall_time = {candidate.configuration_id: 1.0 for candidate in candidates}
    wall_time[expected.configuration_id] = 10_000.0

    ranking = rank_stage_candidates(
        search,
        "geometry",
        candidates,
        _summary_rows(
            candidates,
            role="development",
            seeds=seeds,
            fidelity="scout",
            dataset_ids=dataset_ids,
            aupr=aupr,
            wall_time=wall_time,
        ),
        calibration_budget="full",
        fidelity="scout",
    )

    assert ranking[0].candidate == expected


def test_convergence_promotes_the_smallest_noninferior_stable_ceiling_per_parent() -> None:
    search = load_search_space(REPO_ROOT)
    candidates = _stage_chain(search)["convergence"]
    phase = calibration_phase(search, "reduced", "convergence", "final")
    seeds = _phase_seeds(search, "reduced", "convergence", "final")
    dataset_ids = _dataset_ids(phase.dataset_count)
    parents = sorted({candidate.parent_configuration_id for candidate in candidates})
    parent_quality = dict(zip(parents, (0.80, 0.75), strict=True))
    aupr: dict[tuple[str, str, int], float] = {}
    for candidate in candidates:
        ceiling = int(candidate.parameters["n_estimators"])
        score = parent_quality[candidate.parent_configuration_id] - (
            0.001 if ceiling == 250 else 0.0
        )
        for dataset_id in dataset_ids:
            for seed in seeds:
                aupr[(candidate.configuration_id, dataset_id, seed)] = score

    ranking = rank_stage_candidates(
        search,
        "convergence",
        candidates,
        _summary_rows(
            candidates,
            role="development",
            seeds=seeds,
            fidelity="final",
            dataset_ids=dataset_ids,
            aupr=aupr,
        ),
        calibration_budget="reduced",
        fidelity="final",
    )

    promoted = ranking[:2]
    assert len({item.candidate.parent_configuration_id for item in promoted}) == 2
    assert {item.candidate.parameters["n_estimators"] for item in promoted} == {250}


def test_validation_accepts_only_bounded_convergence_finalists() -> None:
    search = load_search_space(REPO_ROOT)
    finalists = _stage_chain(search)["convergence"][:2]
    seeds = _seeds(search, "full", "validation")
    dataset_ids = _dataset_ids(
        calibration_phase(search, "full", "validation", "final").dataset_count
    )
    runs = expand_validation_runs(search, finalists, "full")
    assert len(runs) == len(finalists) * len(seeds)
    best = finalists[1]
    aupr = {
        (best.configuration_id, dataset_id, seed): 0.7
        for dataset_id in dataset_ids
        for seed in seeds
    }
    selected = rank_validation_candidates(
        search,
        finalists,
        _summary_rows(
            finalists,
            role="validation",
            seeds=seeds,
            fidelity="final",
            dataset_ids=dataset_ids,
            aupr=aupr,
        ),
        calibration_budget="full",
    )
    assert selected[0].candidate == best
    with pytest.raises(CalibrationError, match="exactly 2"):
        expand_validation_runs(search, finalists[:1], "full")
    with pytest.raises(CalibrationError, match="exactly 2"):
        expand_validation_runs(search, _stage_chain(search)["convergence"][:3], "full")
    geometry = expand_stage_candidates(search, "geometry")
    with pytest.raises(CalibrationError, match="convergence stage"):
        expand_validation_runs(search, geometry[:2], "full")


def test_candidate_integrity_rejects_role_and_parameter_mutation() -> None:
    search = load_search_space(REPO_ROOT)
    candidates = expand_stage_candidates(search, "geometry")
    phase = calibration_phase(search, "full", "geometry", "scout")
    seeds = _phase_seeds(search, "full", "geometry", "scout")
    dataset_ids = _dataset_ids(phase.dataset_count)
    relabelled = (replace(candidates[0], dataset_role="validation"), *candidates[1:])
    with pytest.raises(CalibrationError, match="must use development"):
        rank_stage_candidates(
            search,
            "geometry",
            relabelled,
            [],
            calibration_budget="full",
            fidelity="scout",
        )

    parameters = dict(candidates[0].parameters)
    parameters["adaptive_trees"] = True
    mutated = (replace(candidates[0], parameters=parameters), *candidates[1:])
    with pytest.raises(CalibrationError, match="parameter digest is inconsistent"):
        rank_stage_candidates(
            search,
            "geometry",
            mutated,
            [],
            calibration_budget="full",
            fidelity="scout",
        )

    rows = _summary_rows(
        candidates,
        role="development",
        seeds=seeds,
        fidelity="scout",
        dataset_ids=dataset_ids,
    )
    rows.pop()
    with pytest.raises(CalibrationError, match="same datasets"):
        rank_stage_candidates(
            search,
            "geometry",
            candidates,
            rows,
            calibration_budget="full",
            fidelity="scout",
        )
