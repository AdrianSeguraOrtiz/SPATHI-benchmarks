from __future__ import annotations

from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.scalability_boundary import (
    decide_scalability_attempt,
    load_scalability_boundary,
    target_scout_scope,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scout(dataset_id: str, **overrides: object) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "stage": "scout",
        "status": "completed",
        "elapsed_seconds": 10.0,
        "peak_memory_bytes": 100,
        "target_count": 1000,
        "scout_target_count": 64,
        **overrides,
    }


def _full(dataset_id: str, **overrides: object) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "stage": "all_targets",
        "status": "completed",
        "elapsed_seconds": 100.0,
        "peak_memory_bytes": 200,
        "target_count": 1000,
        "scout_target_count": 64,
        **overrides,
    }


def test_contract_covers_public_panels_and_materializes_hash_scope() -> None:
    contract = load_scalability_boundary(REPO_ROOT)
    assert target_scout_scope(
        contract,
        budget="full",
        dataset_id="scale_stress_full_tf_heavy",
    ) == {
        "method": "sha256_gene_id",
        "size": 128,
        "salt": "spathi-scalability-scout-v1",
        "dataset_id": "scale_stress_full_tf_heavy",
    }


def test_all_target_attempt_uses_scout_projection_only() -> None:
    contract = load_scalability_boundary(REPO_ROOT)
    dataset = "scale_stress_reduced_baseline"
    accepted = decide_scalability_attempt(
        contract,
        budget="reduced",
        lane_id="reduced_progression",
        dataset_id=dataset,
        stage="all_targets",
        prior_results=[_scout(dataset)],
        allocation_memory_bytes=1000,
    )
    rejected = decide_scalability_attempt(
        contract,
        budget="reduced",
        lane_id="reduced_progression",
        dataset_id=dataset,
        stage="all_targets",
        prior_results=[_scout(dataset, elapsed_seconds=3_000.0)],
        allocation_memory_bytes=1000,
    )
    assert accepted.decision == "attempt"
    assert rejected.decision == "not_attempted_after_preregistered_resource_boundary"
    assert rejected.reason == "scout_projection_exceeds_wall_time_boundary"
    assert rejected.lane_id == "reduced_progression"


def test_later_scout_stops_after_resource_failure() -> None:
    contract = load_scalability_boundary(REPO_ROOT)
    first = "scale_stress_reduced_baseline"
    second = "scale_stress_reduced_moderate"
    result = decide_scalability_attempt(
        contract,
        budget="reduced",
        lane_id="reduced_progression",
        dataset_id=second,
        stage="scout",
        prior_results=[_scout(first), _full(first, status="oom")],
        allocation_memory_bytes=1000,
    )
    assert result.decision == "not_attempted_after_preregistered_resource_boundary"
    assert result.reason == "preceding_oom"


def test_boundary_rejects_nonresource_or_nonprefix_evidence() -> None:
    contract = load_scalability_boundary(REPO_ROOT)
    dataset = "scale_stress_reduced_baseline"
    with pytest.raises(BenchmarkConfigError, match="unexpected"):
        decide_scalability_attempt(
            contract,
            budget="reduced",
            lane_id="reduced_progression",
            dataset_id=dataset,
            stage="all_targets",
            prior_results=[{**_scout(dataset), "aupr": 0.9}],
            allocation_memory_bytes=1000,
        )
    with pytest.raises(BenchmarkConfigError, match="exact preregistered prefix"):
        decide_scalability_attempt(
            contract,
            budget="reduced",
            lane_id="reduced_progression",
            dataset_id="scale_stress_reduced_moderate",
            stage="scout",
            prior_results=[_scout(dataset)],
            allocation_memory_bytes=1000,
        )


def test_preregistered_skip_propagates_to_the_next_stage_in_its_lane() -> None:
    contract = load_scalability_boundary(REPO_ROOT)
    first = "scale_stress_reduced_baseline"
    second = "scale_stress_reduced_moderate"
    skipped = {
        "dataset_id": first,
        "stage": "all_targets",
        "status": "not_attempted_after_preregistered_resource_boundary",
        "elapsed_seconds": None,
        "peak_memory_bytes": None,
        "target_count": None,
        "scout_target_count": None,
    }
    decision = decide_scalability_attempt(
        contract,
        budget="reduced",
        lane_id="reduced_progression",
        dataset_id=second,
        stage="scout",
        prior_results=[_scout(first), skipped],
        allocation_memory_bytes=1_000,
    )

    assert decision.decision == "not_attempted_after_preregistered_resource_boundary"
    assert decision.reason == "preceding_preregistered_resource_boundary"
