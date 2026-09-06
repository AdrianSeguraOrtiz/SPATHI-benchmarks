from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from spathi_bench.benchmark_materialization import (
    _load_catalog_plan,
    _validate_baseline_inputs,
    build_runnable_benchmark_profile,
)
from spathi_bench.catalog_contracts import validate_catalog_plan_coverage
from spathi_bench.config import BenchmarkConfigError, load_json_object, load_profile

REPO_ROOT = Path(__file__).resolve().parents[1]


def _matrix() -> dict:
    return load_json_object(REPO_ROOT / "experiments/v1/inference/catalog-matrix.json")


def _route(matrix: dict, tool_id: str, variant: str) -> dict:
    entry = next(item for item in matrix["entries"] if item["tool_id"] == tool_id)
    return next(item for item in entry["routes"] if item["variant"] == variant)


def test_catalog_full_plan_covers_every_executable_non_assisted_route() -> None:
    validate_catalog_plan_coverage(REPO_ROOT, _matrix())


def test_all_gene_inference_stratum_classifies_all_ten_technical_competitors() -> None:
    matrix = _matrix()
    expected = {
        "cespgrn",
        "clr",
        "kscreni",
        "lioness",
        "metasem",
        "pidc",
        "ppcor",
        "scgenerai",
        "scing",
        "scsgl",
    }
    candidates = {
        entry["tool_id"]
        for entry in matrix["entries"]
        if entry["tool_id"] in expected and entry["disposition"] == "candidate"
    }
    blocked = {
        entry["tool_id"]
        for entry in matrix["entries"]
        if entry["tool_id"] in expected and entry["disposition"] == "blocked"
    }

    assert candidates == expected - {"scgenerai", "scing", "scsgl"}
    assert blocked == {"scgenerai", "scing", "scsgl"}


def test_every_v1_catalog_tool_has_a_closed_reproducibility_classification() -> None:
    matrix = _matrix()
    assert len(matrix["entries"]) == 24
    assert {entry["reproducibility"]["class"] for entry in matrix["entries"]} <= {
        "caller_seeded",
        "deterministic",
        "fixed_internal_seed",
        "uncontrolled_stochastic",
    }


def test_uncontrolled_stochastic_tool_cannot_be_promoted_to_candidate() -> None:
    matrix = deepcopy(_matrix())
    entry = next(item for item in matrix["entries"] if item["tool_id"] == "scgenerai")
    entry["disposition"] = "candidate"
    entry["routes"] = [
        {
            "regime": "group_only",
            "execution_mode": "group_aggregated",
            "variant": "default",
            "role": "secondary",
            "status": "candidate",
            "configuration_id": "scgenerai_default",
            "evaluation_levels": ["topology"],
        }
    ]

    with pytest.raises(BenchmarkConfigError, match="uncontrolled randomness"):
        validate_catalog_plan_coverage(REPO_ROOT, matrix)


@pytest.mark.parametrize(
    ("comparison_id", "expected_configurations"),
    [
        (
            "full-all-gene-inference-directed-accuracy",
            {
                "selected_spathi",
                "kscreni_default_fixed_seed_100",
                "metasem_unlabeled_all_genes",
            },
        ),
        (
            "full-all-gene-inference-topology-accuracy",
            {
                "selected_spathi",
                "cespgrn_expression_no_prior",
                "clr_default",
                "kscreni_default_fixed_seed_100",
                "lioness_default",
                "metasem_unlabeled_all_genes",
                "pidc_default",
                "ppcor_default",
            },
        ),
    ],
)
def test_full_all_gene_statistics_include_every_level_compatible_configuration(
    comparison_id: str,
    expected_configurations: set[str],
) -> None:
    plan = load_json_object(REPO_ROOT / "experiments/v1/inference/plans/full-v1.json")
    comparison = next(
        item for item in plan["statistics"]["comparisons"] if item["id"] == comparison_id
    )

    assert set(comparison["configuration_ids"]) == expected_configurations
    assert set(comparison["algorithm_seeds_by_configuration"]) == expected_configurations
    assert comparison["candidate_space"] == "tf_restricted"
    scopes = comparison["inference_regulator_scope_by_configuration"]
    assert set(scopes) == expected_configurations
    assert scopes["selected_spathi"] == "tf_restricted"
    assert {
        scope for configuration, scope in scopes.items() if configuration != "selected_spathi"
    } == {"all_gene"}


def test_descriptive_catalog_runs_without_public_seeds_are_materializable() -> None:
    plan, _relative = _load_catalog_plan(
        REPO_ROOT,
        REPO_ROOT / "experiments/v1/inference/plans/full-v1.json",
        template_profile_id="full",
    )
    profile = build_runnable_benchmark_profile(
        load_profile(REPO_ROOT, "full"),
        plan,
        profile_id="full-v1",
        bundle_path="work/test/catalog-bundle",
        policy_id="spathi_conditional_v1",
        production_config_path="work/test/production-config.json",
    )

    _validate_baseline_inputs(REPO_ROOT, profile)


def test_fixed_internal_seed_is_one_null_caller_seed_execution() -> None:
    metadata = load_json_object(
        REPO_ROOT / "experiments/v1/inference/group_only/full-all-gene-metadata.json"
    )
    runs = [
        run
        for run in metadata["runs"]
        if run["configuration_id"] == "kscreni_default_fixed_seed_100"
    ]

    assert len(runs) == 1
    assert runs[0]["algorithm_seed"] is None
    assert runs[0]["algorithm_seed_parameter"] is None


def test_catalog_candidate_route_requires_a_configuration_identity() -> None:
    matrix = deepcopy(_matrix())
    _route(matrix, "clr", "default").pop("configuration_id")

    with pytest.raises(BenchmarkConfigError, match="has no configuration_id"):
        validate_catalog_plan_coverage(REPO_ROOT, matrix)


def test_all_gene_catalog_route_requires_declared_evaluation_levels() -> None:
    matrix = deepcopy(_matrix())
    _route(matrix, "clr", "default").pop("evaluation_levels")

    with pytest.raises(BenchmarkConfigError, match="has no declared evaluation level"):
        validate_catalog_plan_coverage(REPO_ROOT, matrix)


def test_catalog_blocked_route_cannot_claim_a_runnable_configuration() -> None:
    matrix = deepcopy(_matrix())
    _route(
        matrix,
        "simic",
        "similarity=true with a fingerprinted permuted-order dataset view",
    )["configuration_id"] = "simic_permuted_order"

    with pytest.raises(BenchmarkConfigError, match="cannot claim a runnable configuration_id"):
        validate_catalog_plan_coverage(REPO_ROOT, matrix)


def test_catalog_plan_rejects_a_matrix_identity_not_used_by_the_requests() -> None:
    matrix = deepcopy(_matrix())
    _route(matrix, "clr", "default")["configuration_id"] = "clr_unplanned"

    with pytest.raises(BenchmarkConfigError, match="executes undeclared catalog configuration"):
        validate_catalog_plan_coverage(REPO_ROOT, matrix)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("similarity", False), ("normalization_factor", True)],
)
def test_catalog_plan_rejects_nondefault_and_type_coerced_parameters(
    monkeypatch: pytest.MonkeyPatch,
    parameter: str,
    value: object,
) -> None:
    original = load_json_object

    def mutated_loader(path: Path, label: str | None = None) -> dict:
        payload = original(path, label)
        if path.name == "full-simic.json":
            payload = deepcopy(payload)
            payload["runs"][0]["params"][parameter] = value
        return payload

    monkeypatch.setattr("spathi_bench.catalog_contracts.load_json_object", mutated_loader)
    with pytest.raises(BenchmarkConfigError, match="away from the locked ANDREA ToolSpec default"):
        validate_catalog_plan_coverage(REPO_ROOT, _matrix())


def test_all_gene_statistics_require_an_exact_per_configuration_scope_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = load_json_object

    def mutated_loader(path: Path, label: str | None = None) -> dict:
        payload = original(path, label)
        if path.name == "full-v1.json":
            payload = deepcopy(payload)
            comparison = next(
                item
                for item in payload["statistics"]["comparisons"]
                if item["id"] == "full-all-gene-inference-directed-accuracy"
            )
            comparison["inference_regulator_scope_by_configuration"].pop(
                "metasem_unlabeled_all_genes"
            )
        return payload

    monkeypatch.setattr("spathi_bench.catalog_contracts.load_json_object", mutated_loader)
    with pytest.raises(BenchmarkConfigError, match="inference-scope keys differ"):
        validate_catalog_plan_coverage(REPO_ROOT, _matrix())
