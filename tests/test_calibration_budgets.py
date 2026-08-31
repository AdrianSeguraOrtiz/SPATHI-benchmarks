from __future__ import annotations

from pathlib import Path

from spathi_bench.config import load_json_object, load_profile, validate_all_profiles
from spathi_bench.dataset_partitions import build_dataset_partition_contract

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_calibration_budgets_freeze_full_and_reduced_algorithmic_seeds() -> None:
    search = load_json_object(REPO_ROOT / "experiments/v1/spathi/search-space.json")

    assert search["calibration_budgets"] == {
        "full": {
            "algorithmic_seeds": {
                "convergence": [1729, 2718, 31415, 57721, 104729],
                "structure": [1729, 31415, 104729],
                "estimator": [1729, 31415, 104729],
                "validation": [1729, 31415, 104729],
            }
        },
        "reduced": {
            "algorithmic_seeds": {
                "convergence": [1729, 31415],
                "structure": [1729, 31415],
                "estimator": [1729, 31415],
                "validation": [1729, 31415],
            }
        },
    }
    assert all("algorithmic_seeds" not in stage for stage in search["stages"])


def test_budgeted_profiles_have_symmetric_conditions_and_expected_dataset_counts() -> None:
    profiles = validate_all_profiles(REPO_ROOT)
    profiles_by_id = {profile["id"]: profile for profile in profiles}

    for budget, replicates, dataset_count in (("full", 3, 15), ("reduced", 1, 5)):
        for prefix, purpose, role in (
            ("calibration", "calibration", "development"),
            ("selection", "selection", "validation"),
        ):
            profile = profiles_by_id[f"{prefix}-{budget}"]
            assert profile["calibration_budget"] == budget
            assert profile["purpose"] == purpose
            assert profile["planned_design"]["replicates_per_condition"] == replicates
            assert {job["dataset_role"] for job in profile["generation_jobs"]} == {role}
            assert (
                sum(
                    run["replicates"]
                    for job in profile["generation_jobs"]
                    for run in load_json_object(REPO_ROOT / job["simulator_runs"])["runs"]
                )
                == dataset_count
            )


def test_reduced_calibration_panels_are_exact_task_subsets_of_full() -> None:
    profile_ids = (
        "calibration-full",
        "calibration-reduced",
        "selection-full",
        "selection-reduced",
    )
    profiles = [load_profile(REPO_ROOT, profile_id) for profile_id in profile_ids]
    contract = build_dataset_partition_contract(REPO_ROOT, profiles)
    task_digests: dict[str, set[str]] = {profile_id: set() for profile_id in profile_ids}
    for record in contract["records"]:
        if record["identity_type"] == "generation_task":
            task_digests[record["profile_id"]].add(record["identity_sha256"])

    assert len(task_digests["calibration-full"]) == 15
    assert len(task_digests["calibration-reduced"]) == 5
    assert task_digests["calibration-reduced"] < task_digests["calibration-full"]
    assert len(task_digests["selection-full"]) == 15
    assert len(task_digests["selection-reduced"]) == 5
    assert task_digests["selection-reduced"] < task_digests["selection-full"]
    assert task_digests["calibration-full"].isdisjoint(task_digests["selection-full"])
