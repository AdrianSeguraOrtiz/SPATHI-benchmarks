from __future__ import annotations

from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.reporting.common import central_report_path, render_html
from spathi_bench.reporting.efficiency import _validate_production_transition, resource_summaries


def _measurement(*, time: float, memory: int, output: int, targets: int, trees: int) -> dict:
    return {
        "run_completed": True,
        "wall_time_seconds": time,
        "peak_memory_bytes": memory,
        "output_bytes": output,
        "targets_requested": 100,
        "targets_processed": targets,
        "models_requested": targets * 2,
        "models_completed": targets * 2,
        "models_trained": targets * 2,
        "trees_fitted": trees,
    }


def test_resource_summary_deduplicates_runs_repeated_by_context() -> None:
    reference = _measurement(time=10.0, memory=1_000, output=500, targets=100, trees=20_000)
    candidate = _measurement(time=4.0, memory=800, output=300, targets=60, trees=6_000)
    shared = {
        "candidate_id": "eligible_targets",
        "dataset_id": "dataset_1",
        "algorithm_seed": 42,
        "reference_measurements": reference,
        "candidate_measurements": candidate,
    }
    summaries = resource_summaries(
        {"rows": [{**shared, "context_id": "group:a"}, {**shared, "context_id": "group:b"}]}
    )

    assert summaries == [
        {
            "candidate_id": "eligible_targets",
            "reference_run_count": 1,
            "reference_wall_time_seconds_median": 10.0,
            "reference_peak_memory_bytes_median": 1_000,
            "reference_output_bytes_median": 500,
            "reference_targets_processed_median": 100,
            "reference_models_requested_median": 200,
            "reference_trees_fitted_median": 20_000,
            "candidate_run_count": 1,
            "candidate_wall_time_seconds_median": 4.0,
            "candidate_peak_memory_bytes_median": 800,
            "candidate_output_bytes_median": 300,
            "candidate_targets_processed_median": 60,
            "candidate_models_requested_median": 120,
            "candidate_trees_fitted_median": 6_000,
        }
    ]


def test_resource_summary_rejects_conflicting_context_measurements() -> None:
    reference = _measurement(time=10.0, memory=1_000, output=500, targets=100, trees=20_000)
    candidate = _measurement(time=4.0, memory=800, output=300, targets=60, trees=6_000)
    changed = {**candidate, "wall_time_seconds": 5.0}
    rows = [
        {
            "candidate_id": "eligible_targets",
            "dataset_id": "dataset_1",
            "algorithm_seed": 42,
            "reference_measurements": reference,
            "candidate_measurements": candidate,
        },
        {
            "candidate_id": "eligible_targets",
            "dataset_id": "dataset_1",
            "algorithm_seed": 42,
            "reference_measurements": reference,
            "candidate_measurements": changed,
        },
    ]

    with pytest.raises(BenchmarkConfigError, match="conflicting measurements"):
        resource_summaries({"rows": rows})


def test_efficiency_renderer_is_english_interactive_and_self_contained() -> None:
    payload = {
        "schema_version": "1.0",
        "report_type": "efficiency",
        "workflow": {
            "flow_id": "efficiency-fixture",
            "status": "complete",
            "profile": "reduced",
            "analysis_role": "descriptive",
            "identity": {"fingerprint": "f" * 64},
        },
        "datasets": {
            "summary": {"dataset_count": 0, "total_cells": 0},
            "rows": [],
        },
        "study": {
            "id": "study",
            "interpretation_limit": "Do not inject </script><script>alert(1)</script>.",
        },
        "policies": [],
        "comparisons": [],
        "resources": [],
        "decision": {},
        "evidence": {"dataset_role": "efficiency_validation", "algorithm_seed_count": 0},
        "provenance": {},
        "links": [],
    }

    html = render_html(kind="efficiency", payload=payload)

    assert "Can SPATHI run faster without a relevant loss of accuracy?" in html
    assert "Datasets in the efficiency panel" in html
    assert "Policies and parameter values evaluated" in html
    assert "Accuracy and non-inferiority" in html
    assert "Runtime, memory, output, targets, and trees" in html
    assert "plotly.js" in html.lower()
    assert "<script src=" not in html.lower()
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e" in html


def test_efficiency_report_has_one_canonical_location(tmp_path: Path) -> None:
    version = tmp_path / "work/v1"
    expected = version / "reports/full/flow-01/efficiency.html"

    assert (
        central_report_path(
            version_root=version,
            report_type="efficiency",
            scope="full",
            flow_id="flow-01",
        )
        == expected.resolve()
    )


@pytest.mark.parametrize(
    ("safety_status", "production_decision", "selected_id", "selected_digest"),
    (
        ("candidate_authorized", "candidate_adopted", "optimized", "a" * 64),
        ("reference_retained_safety_veto", "reference_retained", "exact", "b" * 64),
    ),
)
def test_production_transition_accepts_safety_authorization_or_veto(
    safety_status: str,
    production_decision: str,
    selected_id: str,
    selected_digest: str,
) -> None:
    adoption = {
        "decision": "candidate_adopted",
        "selected_candidate_id": "optimized",
        "selected_policy_id": "optimized",
        "selected_policy_sha256": "a" * 64,
        "selected_group_policy": {"mode": "optimized"},
    }
    safety = {
        "provisional_candidate_id": "optimized",
        "provisional_candidate_policy_sha256": "a" * 64,
        "decision": {
            "status": safety_status,
            "selected_policy_id": selected_id,
            "selected_policy_sha256": selected_digest,
            "selected_group_policy": {"mode": selected_id},
        },
    }
    production = {
        "evidence_role": "confirmatory",
        "decision": production_decision,
        "policy": {
            "policy_id": selected_id,
            "policy_sha256": selected_digest,
            "group_policy": {"mode": selected_id},
        },
    }

    _validate_production_transition(
        adoption=adoption,
        results={"analysis_role": "confirmatory"},
        production=production,
        safety=safety,
    )


def test_production_transition_rejects_a_different_safety_candidate() -> None:
    with pytest.raises(BenchmarkConfigError, match="different provisional candidate"):
        _validate_production_transition(
            adoption={
                "decision": "candidate_adopted",
                "selected_candidate_id": "planned",
                "selected_policy_sha256": "a" * 64,
            },
            results={"analysis_role": "confirmatory"},
            production={
                "evidence_role": "confirmatory",
                "decision": "reference_retained",
                "policy": {
                    "policy_id": "exact",
                    "policy_sha256": "b" * 64,
                    "group_policy": {"mode": "exact"},
                },
            },
            safety={
                "provisional_candidate_id": "other",
                "provisional_candidate_policy_sha256": "a" * 64,
                "decision": {},
            },
        )
