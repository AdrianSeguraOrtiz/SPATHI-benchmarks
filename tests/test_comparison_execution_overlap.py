"""Real comparison panels must reuse executions without inflating observations."""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.artifacts import canonical_json_sha256
from spathi_bench.benchmark_materialization import (
    SELECTED_PLACEHOLDER,
    build_runnable_benchmark_profile,
    build_selected_spathi_requests,
)
from spathi_bench.calibration import materialize_group_policy
from spathi_bench.config import load_json_object
from spathi_bench.execution_records import unique_execution_records
from spathi_bench.pipeline import _parameter_hashes

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "spathi_conditional_v1"


def _comparison_records(
    budget: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[str]]]:
    """Expand committed panels as inference does, without running simulators/tools."""
    template = load_json_object(REPO_ROOT / f"experiments/v1/profiles/{budget}.json")
    plan = load_json_object(REPO_ROOT / f"experiments/v1/inference/plans/{budget}-v1.json")
    search = load_json_object(REPO_ROOT / "experiments/v1/spathi/search-space.json")
    policy = materialize_group_policy(search, copy.deepcopy(search["base_configuration"]))
    seeds = plan["statistics"]["comparisons"][0]["algorithm_seeds_by_configuration"][
        SELECTED_PLACEHOLDER
    ]
    selected, _custom_tools, selected_metadata = build_selected_spathi_requests(
        {
            "policy_id": POLICY_ID,
            "group_policy": policy,
            "policy_sha256": canonical_json_sha256(policy),
        },
        seeds,
        docker_image="example/spathi:test",
        seed_namespace=f"{budget}.test",
    )
    profile = build_runnable_benchmark_profile(
        template,
        plan,
        profile_id=f"{budget}-v1",
        bundle_path=f"experiments/v1/spathi/benchmark/v1/{budget}",
        policy_id=POLICY_ID,
        production_config_path="work/test-production-config.json",
    )
    datasets: dict[str, list[str]] = {}
    for job in profile["generation_jobs"]:
        scenario = load_json_object(REPO_ROOT / job["scenario"])
        runs = load_json_object(REPO_ROOT / job["simulator_runs"])["runs"]
        datasets[job["id"]] = [
            f"{scenario['id']}__{run['run_id']}__r{replicate:02d}"
            for run in runs
            for replicate in range(1, run["replicates"] + 1)
        ]

    records: list[dict[str, Any]] = []
    physical_seen: set[str] = set()
    for batch in profile["inference_batches"]:
        if batch["kind"] == "spathi":
            requests, metadata = selected, selected_metadata
        else:
            requests = load_json_object(REPO_ROOT / batch["tools_params"])
            metadata = load_json_object(REPO_ROOT / batch["run_metadata"])
        by_run = {row["run_id"]: row for row in metadata["runs"]}
        for job_id in batch["generation_jobs"]:
            for dataset_id in datasets[job_id]:
                for request in requests["runs"]:
                    run_metadata = by_run[request["run_id"]]
                    configuration_hash, execution_hash = _parameter_hashes(request, run_metadata)
                    digest = canonical_json_sha256(
                        {
                            "dataset_id": dataset_id,
                            "regime": batch["regime"],
                            "request": request,
                        }
                    )
                    path = f"/synthetic-test/artifacts/inference/{digest}"
                    records.append(
                        {
                            "dataset_id": dataset_id,
                            "batch_id": batch["id"],
                            "batch_kind": batch["kind"],
                            "regime": batch["regime"],
                            "run_id": request["run_id"],
                            "tool_id": request["tool_id"],
                            "tool_origin": "custom" if batch["kind"] == "spathi" else "catalog",
                            "configuration_id": run_metadata["configuration_id"],
                            "selection_role": run_metadata["selection_role"],
                            "algorithm_seed": run_metadata["algorithm_seed"],
                            "algorithm_seed_parameter": run_metadata["algorithm_seed_parameter"],
                            "family": run_metadata.get("family"),
                            "configuration_sha256": configuration_hash,
                            "execution_sha256": execution_hash,
                            "dataset_fingerprint": canonical_json_sha256(dataset_id),
                            "tool_image": {"image_id": canonical_json_sha256(request["tool_id"])},
                            "resources": {"threads": 1},
                            "run_report": f"{path}/output/run_report.json",
                            "artifact": {
                                "kind": "inference",
                                "key": f"sha256:{digest}",
                                "digest": digest,
                                "path": path,
                                "cache_hit": digest in physical_seen,
                            },
                        }
                    )
                    physical_seen.add(digest)
    return profile, records, datasets


@pytest.mark.parametrize("budget", ["reduced", "full"])
def test_real_comparison_panels_project_aliases_without_changing_frozen_inputs(budget: str) -> None:
    profile, records, datasets = _comparison_records(budget)
    frozen_profile = copy.deepcopy(profile)
    frozen_records = copy.deepcopy(records)
    projected = unique_execution_records(records)

    def scientific_identity(row: dict[str, Any]) -> tuple[Any, ...]:
        return row["dataset_id"], row["regime"], row["configuration_id"], row["algorithm_seed"]

    # This deliberately reproduces the committed primary/all-gene overlap.
    original_counts = Counter(scientific_identity(row) for row in records)
    assert max(original_counts.values()) > 1
    projected_counts = Counter(scientific_identity(row) for row in projected)
    assert set(projected_counts) == set(original_counts)
    assert set(projected_counts.values()) == {1}
    assert profile == frozen_profile
    assert records == frozen_records
    assert len(projected) < len(records)

    for comparison in profile["statistics"]["comparisons"]:
        selected_datasets = {
            dataset_id
            for job_id in comparison["generation_jobs"]
            for dataset_id in datasets[job_id]
        }
        for configuration in comparison["configuration_ids"]:
            expected = {
                (dataset_id, comparison["regime"], configuration, seed)
                for dataset_id in selected_datasets
                for seed in comparison["algorithm_seeds_by_configuration"][configuration]
            }
            observed = {
                scientific_identity(row)
                for row in projected
                if row["dataset_id"] in selected_datasets
                and row["regime"] == comparison["regime"]
                and row["configuration_id"] == configuration
            }
            assert observed == expected, comparison["id"]

    selected_rows = [row for row in projected if row["configuration_id"] == POLICY_ID]
    assert {row["regime"] for row in selected_rows} == (
        {"group_only", "order_aware"} if budget == "full" else {"group_only"}
    )
    if budget == "full":
        ordered_keys = {
            (row["dataset_id"], row["algorithm_seed"])
            for row in selected_rows
            if row["regime"] == "order_aware"
        }
        primary_keys = {
            (row["dataset_id"], row["algorithm_seed"])
            for row in selected_rows
            if row["regime"] == "group_only"
        }
        assert ordered_keys
        assert ordered_keys < primary_keys

    all_gene_panels = [
        comparison
        for comparison in profile["statistics"]["comparisons"]
        if "all-gene" in comparison["id"]
    ]
    assert all_gene_panels
    for comparison in all_gene_panels:
        scopes = comparison["inference_regulator_scope_by_configuration"]
        assert scopes[POLICY_ID] == "tf_restricted"
        assert "all_gene" in scopes.values()
