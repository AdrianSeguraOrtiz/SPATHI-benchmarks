from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def _load_json(relative_path: str) -> dict[str, Any]:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize(
    ("schema_path", "document_path"),
    [
        (
            "schemas/spathi-search-space.schema.json",
            "experiments/v1/spathi/search-space.json",
        ),
        (
            "schemas/spathi-selected-config.schema.json",
            "experiments/v1/spathi/selected-config.json",
        ),
    ],
)
def test_calibration_document_matches_valid_schema(
    schema_path: str,
    document_path: str,
) -> None:
    schema = _load_json(schema_path)
    document = _load_json(document_path)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER).validate(
        document
    )


def test_search_stages_cannot_use_validation_or_test_data() -> None:
    schema = _load_json("schemas/spathi-search-space.schema.json")
    document = copy.deepcopy(_load_json("experiments/v1/spathi/search-space.json"))
    document["stages"][0]["dataset_role"] = "validation"

    errors = list(Draft202012Validator(schema).iter_errors(document))

    assert errors


def test_unselected_configuration_cannot_claim_a_policy() -> None:
    schema = _load_json("schemas/spathi-selected-config.schema.json")
    document = copy.deepcopy(_load_json("experiments/v1/spathi/selected-config.json"))
    search_space = _load_json("experiments/v1/spathi/search-space.json")
    multi = search_space["base_configuration"]
    single = {
        **multi,
        "weight_mode": "cell-distance",
        "distance_metric": "euclidean",
        "group_size_correction": "none",
    }
    document["selected_policy"] = {
        "policy_id": "spathi_conditional_v1",
        "source_candidate_id": "invented_winner",
        "spathi_commit": "0" * 40,
        "source_configuration_sha256": "0" * 64,
        "policy_sha256": "0" * 64,
        "group_policy": {
            "routing_parameter": "group_count",
            "shared_parameters": search_space["conditional_group_policy"]["shared_parameters"],
            "branches": {
                "single_group": {
                    "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                    "parameters": single,
                },
                "multi_group": {
                    "condition": {"minimum_group_count": 2},
                    "parameters": multi,
                },
            },
        },
    }

    errors = list(Draft202012Validator(schema).iter_errors(document))

    assert errors


def test_schema_accepts_a_fully_evidenced_future_frozen_selection() -> None:
    schema = _load_json("schemas/spathi-selected-config.schema.json")
    document = copy.deepcopy(_load_json("experiments/v1/spathi/selected-config.json"))
    search_space = _load_json("experiments/v1/spathi/search-space.json")
    multi = search_space["base_configuration"]
    single = {
        **multi,
        "weight_mode": "cell-distance",
        "distance_metric": "euclidean",
        "group_size_correction": "none",
    }
    document.update(
        {
            "status": "selected",
            "phase": "frozen",
            "calibration_budget": "full",
            "selected_policy": {
                "policy_id": "spathi_conditional_v1",
                "source_candidate_id": "future_evidenced_candidate",
                "spathi_commit": "a" * 40,
                "source_configuration_sha256": "b" * 64,
                "policy_sha256": "2" * 64,
                "group_policy": {
                    "routing_parameter": "group_count",
                    "shared_parameters": search_space["conditional_group_policy"][
                        "shared_parameters"
                    ],
                    "branches": {
                        "single_group": {
                            "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                            "parameters": single,
                        },
                        "multi_group": {
                            "condition": {"minimum_group_count": 2},
                            "parameters": multi,
                        },
                    },
                },
            },
            "selection_evidence": {
                "calibration_budget": "full",
                "validation_ranking": "experiments/v1/spathi/evidence/validation-ranking.json",
                "validation_ranking_sha256": "c" * 64,
                "search_space_sha256": "d" * 64,
                "validation_manifest_sha256": "c" * 64,
                "benchmark_summary_sha256": "e" * 64,
                "spathi_build_fingerprint": "f" * 64,
                "spathi_image_id": "sha256:" + "1" * 64,
                "primary_metric": "aupr_directed",
                "hierarchical_score": 0.5,
                "winner_candidate_id": "future_evidenced_candidate",
                "winner_configuration_sha256": "b" * 64,
                "winner_policy_sha256": "2" * 64,
                "tie_breakers_applied": ["stability", "simplicity", "canonical_configuration_id"],
                "frozen_at": "2026-08-28T12:00:00Z",
            },
        }
    )

    Draft202012Validator(
        schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(document)


def test_search_budget_is_bounded_and_test_is_reserved() -> None:
    document = _load_json("experiments/v1/spathi/search-space.json")

    assert [stage["id"] for stage in document["stages"]] == [
        "geometry",
        "estimator",
        "interaction",
        "estimator_scale_transfer",
        "convergence",
    ]
    assert all(stage["candidate_budget_per_parent"] <= 14 for stage in document["stages"])
    assert document["selection_protocol"]["maximum_validation_finalists"] == 3
    assert document["selection_protocol"]["test_results_can_change_selection"] is False
