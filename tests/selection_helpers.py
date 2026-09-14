"""Schema-valid calibration selections for downstream contract tests.

These fixtures do not manufacture calibration evidence: tests exercising the
ranking chain use the calibration fixtures, while downstream unit tests mock
only that independent verification boundary.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from spathi_bench.artifacts import canonical_json_sha256
from spathi_bench.calibration import load_search_space, materialize_group_policy


def base_policy(*, spathi_commit: str = "1" * 40, n_estimators: int = 250) -> dict[str, Any]:
    """Build an executable conditional policy from the committed calibration base."""

    search = load_search_space(Path(__file__).resolve().parents[1])
    group_policy = materialize_group_policy(
        search, {**search["base_configuration"], "n_estimators": n_estimators}
    )
    return {
        "policy_id": "spathi_conditional_v1",
        "policy_sha256": canonical_json_sha256(group_policy),
        "group_policy": group_policy,
        "spathi_commit": spathi_commit,
    }


def selected_document(
    policy: dict[str, Any],
    *,
    budget: str = "reduced",
    ranking_path: str = "work/calibration/validation-ranking.json",
    ranking_sha256: str = "9" * 64,
) -> dict[str, Any]:
    policy = deepcopy(policy)
    parameters = policy["group_policy"]["branches"]["multi_group"]["parameters"]
    policy.setdefault("source_candidate_id", "calibration_winner")
    policy.setdefault("source_configuration_sha256", canonical_json_sha256(parameters))
    policy["policy_sha256"] = canonical_json_sha256(policy["group_policy"])
    return {
        "schema_version": "1.0",
        "id": "spathi_selected_v1",
        "search_space": "experiments/v1/spathi/search-space.json",
        "calibration_budget": budget,
        "status": "selected",
        "phase": "frozen",
        "selected_policy": policy,
        "selection_evidence": {
            "calibration_budget": budget,
            "validation_ranking": ranking_path,
            "validation_ranking_sha256": ranking_sha256,
            "search_space_sha256": "1" * 64,
            "validation_manifest_sha256": "2" * 64,
            "benchmark_summary_sha256": "3" * 64,
            "spathi_build_fingerprint": "4" * 64,
            "spathi_image_id": "sha256:" + "5" * 64,
            "primary_metric": "aupr_directed",
            "hierarchical_score": 0.5,
            "winner_candidate_id": policy["source_candidate_id"],
            "winner_configuration_sha256": policy["source_configuration_sha256"],
            "winner_policy_sha256": policy["policy_sha256"],
            "tie_breakers_applied": ["stability", "simplicity", "canonical_configuration_id"],
            "frozen_at": "2026-09-14T00:00:00Z",
        },
        "test_accessed_before_freeze": False,
        "notes": ["Frozen selection fixture for downstream contract tests."],
    }
