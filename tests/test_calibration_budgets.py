from __future__ import annotations

from pathlib import Path

from spathi_bench.config import load_json_object, load_profile, validate_all_profiles
from spathi_bench.dataset_partitions import build_dataset_partition_contract

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_calibration_budgets_freeze_full_and_reduced_algorithmic_seeds() -> None:
    search = load_json_object(REPO_ROOT / "experiments/v1/spathi/search-space.json")

    expected_seeds = {
        "full": {
            "algorithmic_seeds": {
                "geometry": [661002162, 737908398, 361869425],
                "estimator": [1078468867, 1606970393, 1420105189],
                "estimator_scale_transfer": [1564593931, 1494888927, 1246618629],
                "interaction": [1068542066, 1674017269, 135662027],
                "convergence": [1699595498, 909275566, 1245131614, 1379909982, 1544350695],
                "validation": [2114967450, 1335083472, 976486246, 330429731, 1228553048],
            }
        },
        "reduced": {
            "algorithmic_seeds": {
                "geometry": [465212547, 966823479],
                "estimator": [518138185, 1607047126],
                "estimator_scale_transfer": [1405804663, 1713637447],
                "interaction": [1844980629, 1542315938],
                "convergence": [1865227637, 272793142],
                "validation": [1153622556, 1342615056],
            }
        },
    }
    assert {
        budget: {"algorithmic_seeds": specification["algorithmic_seeds"]}
        for budget, specification in search["calibration_budgets"].items()
    } == expected_seeds
    assert search["calibration_budgets"]["reduced"]["expected_physical_fits"] == 431
    assert search["calibration_budgets"]["reduced"]["expected_logical_results"] == 527
    assert search["calibration_budgets"]["full"]["expected_physical_fits"] == 1762
    assert search["calibration_budgets"]["full"]["expected_logical_results"] == 2482
    for budget, specification in search["calibration_budgets"].items():
        fidelity = specification["multifidelity"]
        assert fidelity["independent_profile"] is True
        assert fidelity["may_consume_other_profile_outcomes"] is False
        assert budget + "_outcomes" not in fidelity["forbidden_scout_inputs"]
    assert all("algorithmic_seeds" not in stage for stage in search["stages"])


def test_budgeted_profiles_have_symmetric_conditions_and_expected_dataset_counts() -> None:
    profiles = validate_all_profiles(REPO_ROOT)
    profiles_by_id = {profile["id"]: profile for profile in profiles}

    for budget, replicates, calibration_count, selection_count in (
        ("full", 3, 27, 24),
        ("reduced", 1, 9, 8),
    ):
        for prefix, purpose, role in (
            ("calibration", "calibration", "development"),
            ("selection", "selection", "validation"),
        ):
            profile = profiles_by_id[f"{prefix}-{budget}"]
            assert profile["calibration_budget"] == budget
            assert profile["purpose"] == purpose
            assert profile["planned_design"]["replicates_per_condition"] == replicates
            assert profile["execution_class"] == (
                "bulk_canonical" if purpose == "calibration" else "bulk_full"
            )
            assert {job["dataset_role"] for job in profile["generation_jobs"]} == {role}
            expected_count = calibration_count if purpose == "calibration" else selection_count
            assert (
                sum(
                    run["replicates"]
                    for job in profile["generation_jobs"]
                    for run in load_json_object(REPO_ROOT / job["simulator_runs"])["runs"]
                )
                == expected_count
            )


def test_reduced_and_full_calibration_panels_have_disjoint_tasks() -> None:
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

    assert len(task_digests["calibration-full"]) == 27
    assert len(task_digests["calibration-reduced"]) == 9
    assert task_digests["calibration-reduced"].isdisjoint(task_digests["calibration-full"])
    assert len(task_digests["selection-full"]) == 24
    assert len(task_digests["selection-reduced"]) == 8
    assert task_digests["selection-reduced"].isdisjoint(task_digests["selection-full"])
    assert task_digests["calibration-full"].isdisjoint(task_digests["selection-full"])


def test_estimator_scale_transfer_crosses_every_feature_rule_on_independent_high_tf_data() -> None:
    search = load_json_object(REPO_ROOT / "experiments/v1/spathi/search-space.json")
    stage = next(item for item in search["stages"] if item["id"] == "estimator_scale_transfer")
    assert stage["inherits_from"] == "shortlist:interaction"
    assert stage["include_inherited_configuration"] is False
    assert stage["candidate_budget_per_parent"] == 6
    assert len(stage["candidates"]) == 6
    values = [candidate["parameter_overrides"]["max_features"] for candidate in stage["candidates"]]
    assert {"sqrt", "log2", 0.25, 0.5} <= set(values)
    assert any(type(value) is int and value == 1 for value in values)
    assert any(type(value) is float and value == 1.0 for value in values)
    assert stage["maximum_parent_configurations"] == 3
    assert stage["advance_limit"] == 2

    expected = {
        "reduced": (1, 111929143),
        "full": (3, 1319366688),
    }
    for budget, (replicates, base_seed) in expected.items():
        profile = load_profile(REPO_ROOT, f"calibration-{budget}")
        job = next(
            item
            for item in profile["generation_jobs"]
            if item["condition_id"] == "scmultisim_estimator_scale_transfer"
        )
        assert job["split_id"] == "development_estimator_scale_transfer_v1"
        run = load_json_object(REPO_ROOT / job["simulator_runs"])["runs"][0]
        assert (run["replicates"], run["base_seed"]) == (replicates, base_seed)
        assert (run["params"]["num_cells"], run["params"]["num_genes"]) == (300, 2_200)

    network = (
        REPO_ROOT
        / "experiments/v1/simulations/scmultisim/estimator_scale_transfer/regulatory-network.tsv"
    )
    regulators = {
        line.split("\t")[1] for line in network.read_text(encoding="utf-8").splitlines()[1:]
    }
    assert len(regulators) == 1_000
