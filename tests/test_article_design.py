from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import pytest

from spathi_bench.design import (
    ARTICLE_TOOL_IDS,
    DesignError,
    calibration_instances,
    comparison_instances,
    eligible_routes,
    load_design,
    parse_design,
    profile_view,
    resolve_thread_points,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = REPO_ROOT / "experiments" / "article-study.json"


def test_committed_article_design_is_the_single_valid_three_phase_source() -> None:
    from_root = load_design(REPO_ROOT)
    from_file = load_design(DESIGN_PATH)

    assert from_root.fingerprint == from_file.fingerprint
    assert set(from_root.profiles) == {"smoke", "reduced", "full"}
    assert len(from_root.candidates) == 27
    assert from_root.validation_prefixes == (25, 50, 100, 250)
    assert from_root.screening_trees == 50
    assert from_root.finalists == 4
    assert from_root.validation_fit_trees == 250


def test_calibration_candidates_are_equal_explicit_and_cover_every_domain() -> None:
    design = load_design(REPO_ROOT)
    domains = design.raw["calibration"]["parameter_domains"]
    candidate_keys = set(design.candidates[0].parameters)

    assert all(set(candidate.parameters) == candidate_keys for candidate in design.candidates)
    assert candidate_keys == set(domains)
    assert design.raw["calibration"]["protocol"]["advancement"]["privileged_candidates"] == []
    for parameter, allowed in domains.items():
        values = [candidate.parameters[parameter] for candidate in design.candidates]
        observed = {json.dumps(value) for value in values}
        assert observed == {json.dumps(value) for value in allowed}
        counts = [values.count(value) for value in allowed]
        assert max(counts) - min(counts) <= 1

    for left, right in itertools.combinations(domains, 2):
        observed_pairs = {
            (
                json.dumps(candidate.parameters[left]),
                json.dumps(candidate.parameters[right]),
            )
            for candidate in design.candidates
        }
        expected_pairs = {
            (json.dumps(left_value), json.dumps(right_value))
            for left_value in domains[left]
            for right_value in domains[right]
        }
        assert observed_pairs == expected_pairs


def test_current_spathi_configuration_is_one_unprivileged_candidate() -> None:
    design = load_design(REPO_ROOT)
    current = {
        "weight_mode": "cell-distance-group-anchored",
        "distance_space": "pca",
        "n_components": 50,
        "distance_standardization": "none",
        "distance_metric": "cosine",
        "kernel": "gaussian",
        "bandwidth_scale": 1.0,
        "group_size_correction": "none",
        "tree_method": "extra-trees",
        "max_features": 0.5,
        "min_samples_leaf": 2,
        "max_depth": None,
        "min_weight_fraction_leaf": 0.1,
    }

    assert sum(candidate.parameters == current for candidate in design.candidates) == 1
    assert design.raw["calibration"]["protocol"]["advancement"]["privileged_candidates"] == []


def test_calibration_budgets_match_the_frozen_physical_fit_ledger() -> None:
    design = load_design(REPO_ROOT)

    assert design.calibration_fit_count("smoke") == 31
    assert design.calibration_fit_count("reduced") == 155
    assert design.calibration_fit_count("full") == 390
    assert len(calibration_instances(design, "reduced", split="development")) == 5
    assert len(calibration_instances(design, "full", split="development")) == 10
    assert len(calibration_instances(design, "full", split="validation")) == 10


def test_profile_expansion_is_nested_and_seeded_by_scientific_coordinates() -> None:
    design = load_design(REPO_ROOT)
    reduced = profile_view(design, "reduced")
    full = profile_view(design, "full")

    assert {phase: set(roles) for phase, roles in reduced.datasets.items()} == {
        "calibration": {"development", "validation"},
        "comparison": {"test"},
        "scalability": {"ladder", "thread_scaling"},
    }
    reduced_seeds = {
        (dataset.condition_id, dataset.role, dataset.replicate): dataset.seed
        for roles in reduced.datasets.values()
        for datasets in roles.values()
        for dataset in datasets
    }
    full_seeds = {
        (dataset.condition_id, dataset.role, dataset.replicate): dataset.seed
        for roles in full.datasets.values()
        for datasets in roles.values()
        for dataset in datasets
    }
    assert reduced_seeds.items() <= full_seeds.items()
    assert (
        reduced.datasets["scalability"]["ladder"][1].seed
        == reduced.datasets["scalability"]["thread_scaling"][0].seed
    )
    assert (
        reduced.datasets["scalability"]["ladder"][1].scenario_request
        == reduced.datasets["scalability"]["thread_scaling"][0].scenario_request
    )


def test_comparison_panel_is_held_out_and_has_declared_replicate_counts() -> None:
    design = load_design(REPO_ROOT)

    assert design.comparison_dataset_count("smoke") == 1
    assert design.comparison_dataset_count("reduced") == 8
    assert design.comparison_dataset_count("full") == 24
    assert len(comparison_instances(design, "full")) == 24
    assert set(design.profile("full").comparison.replicates.values()) == {3}
    assert design.profile("smoke").comparison.task_timeout_seconds == 1_800
    assert design.profile("reduced").comparison.task_timeout_seconds == 3_600
    assert design.profile("full").comparison.task_timeout_seconds == 7_200
    calibration_ids = set(design.calibration_conditions)
    comparison_ids = set(design.comparison_datasets)
    assert calibration_ids.isdisjoint(comparison_ids)


def test_competitor_families_enforce_information_and_output_contracts() -> None:
    design = load_design(REPO_ROOT)
    families = {family.id: family for family in design.competitor_families}

    assert set(families) == {
        "a-information-matched",
        "b-order-aware",
        "c-lineage-aware",
        "d-directed-all-gene",
        "e-undirected-topology",
    }
    assert families["a-information-matched"].statistical_role == "primary_friedman_holm"
    assert families["a-information-matched"].route_ids == (
        "spathi",
        "genie3",
        "grnboost2",
        "infercsn",
        "tigress",
        "scmtni-independent",
    )
    assert all(route.tool_id in ARTICLE_TOOL_IDS for route in design.tool_routes.values())
    assert design.tool_routes["kscreni"].tool_id == "kscreni"
    assert design.tool_routes["scmtni-independent"].execution_mode == "group_emulated"
    assert design.tool_routes["scmtni-independent"].seed_parameter is None
    assert design.tool_routes["spathi"].default_parameter_overrides == {}


def test_dataset_contract_drives_eligibility_without_manual_tool_panels() -> None:
    design = load_design(REPO_ROOT)
    linear_routes = {route.id for route in eligible_routes(design, "t1-linear-clean")}
    branching_routes = {route.id for route in eligible_routes(design, "t3-branching-clean")}

    assert "simic-ordered" in linear_routes
    assert "scmtni-lineage" in linear_routes
    assert "simic-ordered" not in branching_routes
    assert "scmtni-lineage" in branching_routes


def test_calibration_and_comparison_exclude_single_group_datasets() -> None:
    design = load_design(REPO_ROOT)

    assert all(
        condition.dimensions.groups > 1 for condition in design.calibration_conditions.values()
    )
    assert all(dataset.dimensions.groups > 1 for dataset in design.comparison_datasets.values())
    assert all(
        "single_group" not in dataset.attributes.values()
        for dataset in (
            *design.calibration_conditions.values(),
            *design.comparison_datasets.values(),
        )
    )

    high_regulator = design.calibration_conditions["c5-high-regulator"].dimensions
    assert (high_regulator.cells, high_regulator.genes) == (600, 600)
    assert high_regulator.transcription_factors == high_regulator.genes // 2


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ("calibration", "Calibration may contain only multi-group conditions"),
        ("comparison", "cannot be applied to single-group comparison datasets"),
    ],
)
def test_design_rejects_single_group_scientific_panels(section: str, message: str) -> None:
    raw = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    key = "conditions" if section == "calibration" else "datasets"
    raw[section][key][0]["dimensions"]["groups"] = 1

    with pytest.raises(DesignError, match=message):
        parse_design(raw)


def test_simulator_materialization_matches_declared_group_contracts() -> None:
    design = load_design(REPO_ROOT)
    view = profile_view(design, "full")
    datasets = [
        dataset for phase in view.datasets.values() for role in phase.values() for dataset in role
    ]

    for planned in datasets:
        declared = (
            design.calibration_conditions.get(planned.condition_id)
            or design.comparison_datasets.get(planned.condition_id)
            or design.scalability_scales[planned.condition_id]
        )
        if planned.generator == "dyngen":
            backbone = planned.simulator_parameters["backbone_template"]
            expected_groups = 4 if backbone == "linear" else 7
            assert declared.dimensions.groups == expected_groups
        if planned.generator == "scmultisim":
            assert planned.simulator_parameters["tree_preset"] == "input_newick"
            assert planned.generated_inputs["tree_newick"]["groups"] == declared.dimensions.groups
            assert sum(planned.simulator_parameters["discrete_population"]["pop_size"]) == (
                declared.dimensions.cells
            )
            assert (
                planned.generated_inputs["regulatory_network"]["transcription_factors"]
                == declared.dimensions.transcription_factors
            )


def test_scalability_is_breadth_first_portable_and_contains_a_real_scale_synthetic_case() -> None:
    design = load_design(REPO_ROOT)
    full = design.profile("full")
    execution = design.raw["scalability"]["execution"]

    assert full.scalability.scale_ids == ("p-small", "p-medium", "p-large", "p-xlarge")
    assert execution["ordering"] == "breadth_first_by_scale"
    assert execution["advance_rule"] == "previous_scale_completed"
    assert design.raw["scalability"]["route_rule"] == ("all_contract_eligible_comparison_routes")
    assert design.scalability_scales["p-xlarge"].dimensions.cells == 10_000
    assert design.scalability_scales["p-xlarge"].dimensions.genes == 20_000
    assert {dataset.simulator for dataset in design.scalability_scales.values()} == {"scale-stress"}
    assert resolve_thread_points(("one", "standard", "all"), 4) == (1, 4)
    assert resolve_thread_points(("one", "standard", "all"), 64) == (1, 8, 64)
    assert {route.id for route in eligible_routes(design, "p-xlarge")} == set(design.tool_routes)


def test_invalid_privilege_unknown_tool_or_duplicate_candidate_is_rejected() -> None:
    raw = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))

    privileged = copy.deepcopy(raw)
    privileged["calibration"]["protocol"]["advancement"]["privileged_candidates"] = ["candidate-01"]
    with pytest.raises(DesignError, match="privileged"):
        parse_design(privileged)

    unknown_tool = copy.deepcopy(raw)
    unknown_tool["comparison"]["routes"][0]["tool_id"] = "unknown-tool"
    with pytest.raises(DesignError, match="unsupported tool"):
        parse_design(unknown_tool)

    duplicate = copy.deepcopy(raw)
    duplicate["calibration"]["candidates"][1]["parameters"] = copy.deepcopy(
        duplicate["calibration"]["candidates"][0]["parameters"]
    )
    with pytest.raises(DesignError, match="duplicates"):
        parse_design(duplicate)


def test_design_rejects_unknown_information_and_execution_contracts() -> None:
    raw = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))

    unknown_capability = copy.deepcopy(raw)
    unknown_capability["comparison"]["datasets"][0]["capabilities"].append("pseudotime")
    with pytest.raises(DesignError, match="unsupported values"):
        parse_design(unknown_capability)

    invalid_delivery = copy.deepcopy(raw)
    invalid_delivery["comparison"]["routes"][0]["delivery"] = "global"
    with pytest.raises(DesignError, match="unsupported delivery"):
        parse_design(invalid_delivery)

    invalid_semantics = copy.deepcopy(raw)
    invalid_semantics["comparison"]["routes"][0]["output_semantics"] = "mixed"
    with pytest.raises(DesignError, match="unsupported output semantics"):
        parse_design(invalid_semantics)

    missing_spathi = copy.deepcopy(raw)
    missing_spathi["comparison"]["families"][0]["route_ids"].remove("spathi")
    with pytest.raises(DesignError, match="must include SPATHI"):
        parse_design(missing_spathi)
