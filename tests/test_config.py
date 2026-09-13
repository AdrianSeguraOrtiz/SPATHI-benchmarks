from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from spathi_bench.config import (
    _PROFILE_DATASET_ROLES,
    BenchmarkConfigError,
    _validate_run_identity,
    load_image_lock,
    load_json_object,
    load_profile,
    load_profile_file,
    load_repo_lock,
    profile_path,
    resolve_repo_path,
    validate_all_profiles,
    validate_design_contracts,
    validate_locked_repository,
    validate_profile_references,
)
from spathi_bench.seeds import generation_seed_for_consumer, load_seed_plan

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_repository_contracts_and_all_profiles_are_valid() -> None:
    repo_lock = load_repo_lock(REPO_ROOT)
    image_lock = load_image_lock(REPO_ROOT)
    profiles = validate_all_profiles(REPO_ROOT)
    validate_design_contracts(REPO_ROOT)

    assert (
        repo_lock["repositories"]["spathi"]["commit"]
        == image_lock["images"]["spathi"]["source_commit"]
    )
    assert set(image_lock["images"]["spathi"]["build_inputs"]) == {
        "containers/spathi/Dockerfile",
        "containers/spathi/constraints.txt",
        "containers/spathi/run_convergence.py",
        "containers/spathi/run_tool.py",
    }
    profile_ids = {profile["id"] for profile in profiles}
    assert {
        "smoke",
        "calibration-full",
        "selection-full",
        "calibration-reduced",
        "selection-reduced",
        "reduced",
        "full",
    } <= profile_ids
    status_by_id = {profile["id"]: profile["status"] for profile in profiles}
    assert status_by_id["smoke"] == "runnable"
    assert status_by_id["calibration-full"] == "planned"
    assert status_by_id["selection-full"] == "planned"
    assert status_by_id["calibration-reduced"] == "planned"
    assert status_by_id["selection-reduced"] == "planned"
    assert status_by_id["reduced"] == "planned"
    assert status_by_id["full"] == "planned"


def test_every_schema_profile_purpose_has_an_explicit_dataset_role_contract() -> None:
    schema = load_json_object(REPO_ROOT / "schemas/profile.schema.json")

    assert set(schema["properties"]["purpose"]["enum"]) == set(_PROFILE_DATASET_ROLES)


@pytest.mark.parametrize(
    "profile_id",
    ("efficiency-development-reduced", "efficiency-development-full"),
)
def test_materialized_efficiency_development_profile_is_role_valid(
    profile_id: str,
) -> None:
    profile = deepcopy(load_profile(REPO_ROOT, profile_id))
    profile["status"] = "runnable"

    validate_profile_references(REPO_ROOT, profile)


def test_design_validation_treats_seed_audit_as_a_mandatory_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spathi_bench.reproducibility import ReproducibilityAuditError

    def fail(_repo_root: Path) -> dict[str, object]:
        raise ReproducibilityAuditError("stale seed fixture")

    monkeypatch.setattr("spathi_bench.reproducibility.audit_reproducibility", fail)
    with pytest.raises(BenchmarkConfigError, match="Reproducibility audit failed"):
        validate_design_contracts(REPO_ROOT)


def test_full_efficiency_safety_profile_freezes_six_disjoint_conditions() -> None:
    profile = load_profile(REPO_ROOT, "efficiency-safety-full")
    jobs = [job for job in profile["generation_jobs"] if job["dataset_role"] == "efficiency_safety"]
    expected = {
        "g3000_tf100_c300": (3_000, 100, 300, [100, 100, 100]),
        "g100_tf10_c500_single_group": (100, 10, 500, [500]),
        "g100_tf10_c300_groups_10_40_250": (100, 10, 300, [10, 40, 250]),
        "g100_tf10_c10000": (100, 10, 10_000, [3_333, 3_333, 3_334]),
        "g3000_tf1500_c300": (3_000, 1_500, 300, [100, 100, 100]),
        "g15000_tf100_c300": (15_000, 100, 300, [100, 100, 100]),
    }
    assert {job["condition_id"] for job in jobs} == set(expected)

    def directed_edges(path: Path) -> set[tuple[str, str]]:
        edges: set[tuple[str, str]] = set()
        for row in path.read_text(encoding="utf-8").splitlines()[1:]:
            target, regulator, _effect = row.split("\t")
            edges.add((target, regulator))
        return edges

    safety_root = REPO_ROOT / "experiments/v1/simulations/scmultisim/efficiency_safety"
    seed_plan = load_seed_plan(REPO_ROOT / "experiments/v1/seeds/seed-plan.json")
    generation_seeds: set[int] = set()
    for job in jobs:
        run = load_json_object(REPO_ROOT / job["simulator_runs"])["runs"][0]
        genes, transcription_factors, cells, group_sizes = expected[job["condition_id"]]
        assert job["split_id"] == "efficiency_safety_v1"
        assert run["replicates"] == 3
        assert run["params"]["num_genes"] == genes
        assert run["params"]["num_cells"] == cells
        assert run["params"]["discrete_population"]["pop_size"] == group_sizes
        assert run["base_seed"] == generation_seed_for_consumer(
            seed_plan,
            path=job["simulator_runs"],
            run_id=run["run_id"],
        )
        scenario = load_json_object(REPO_ROOT / job["scenario"])
        topology = safety_root / scenario["inputs"]["regulatory_network"]["path"]
        assert len({regulator for _target, regulator in directed_edges(topology)}) == (
            transcription_factors
        )
        generation_seeds.update(run["base_seed"] + offset for offset in range(run["replicates"]))
    assert len(generation_seeds) == 18

    transfer_root = REPO_ROOT / "experiments/v1/simulations/scmultisim/dimensional_transfer"
    safety_topologies = [
        directed_edges(path) for path in sorted(safety_root.glob("regulatory-network*.tsv"))
    ]
    assert len(safety_topologies) == 4
    for index, left in enumerate(safety_topologies):
        for right in safety_topologies[index + 1 :]:
            assert left.isdisjoint(right)
    transfer_edges = set().union(
        *(directed_edges(path) for path in transfer_root.glob("regulatory-network*.tsv"))
    )
    assert set().union(*safety_topologies).isdisjoint(transfer_edges)


def test_formal_efficiency_profile_excludes_conditional_safety_generation() -> None:
    formal = load_profile(REPO_ROOT, "efficiency-full")

    assert len(formal["generation_jobs"]) == 5
    assert {job["dataset_role"] for job in formal["generation_jobs"]} == {"efficiency_validation"}
    assert "replicate_exceptions" not in formal["planned_design"]


@pytest.mark.parametrize("budget", ["reduced", "full"])
def test_boolode_panels_cover_the_native_sampling_grid(budget: str) -> None:
    profile = load_profile(REPO_ROOT, budget)
    panels = [
        load_json_object(REPO_ROOT / job["simulator_runs"])["runs"][0]
        for job in profile["generation_jobs"]
        if job["id"].startswith("boolode_")
    ]
    assert len(panels) == 2
    for run in panels:
        params = run["params"]
        # The pinned upstream sampler selects one index in [1, tmax * 100)
        # per trajectory. Its grid must cover exactly that time interval.
        assert params["integration_step_size"] == 0.01
        assert int(params["simulation_time"] / params["integration_step_size"]) == (
            params["simulation_time"] * 100
        )
        assert params["sample_cells"] is False
        assert params["num_cells"] == 300
        assert params["n_clusters"] == 3


def test_full_panel_freezes_supported_robustness_conditions() -> None:
    profile = load_profile(REPO_ROOT, "full")
    jobs = {job["id"]: job for job in profile["generation_jobs"]}
    runs_by_job = {
        job_id: load_json_object(REPO_ROOT / job["simulator_runs"])["runs"][0]
        for job_id, job in jobs.items()
    }

    assert sum(run["replicates"] for run in runs_by_job.values()) == 60
    seed_plan = load_seed_plan(REPO_ROOT / "experiments/v1/seeds/seed-plan.json")
    allocated: set[int] = set()
    for job_id, job in jobs.items():
        run = runs_by_job[job_id]
        assert run["base_seed"] == generation_seed_for_consumer(
            seed_plan,
            path=job["simulator_runs"],
            run_id=run["run_id"],
        )
        allocated.update(run["base_seed"] + offset for offset in range(run["replicates"]))
    assert len(allocated) == 60

    low = runs_by_job["scmultisim_test_dynamic_low_balanced"]["params"]
    high = runs_by_job["scmultisim_test_dynamic_high_imbalanced"]["params"]
    assert low["dynamic_grn"]["num_changing_edges"] == 2
    assert low["population_mode"] == "continuous"
    assert high["dynamic_grn"]["num_changing_edges"] == 12
    assert high["population_mode"] == "discrete"
    assert high["discrete_population"]["min_pop_size"] == 12
    assert high["technical_noise"] == {
        "enabled": True,
        "protocol": "UMI",
        "alpha_mean": 0.1,
        "alpha_sd": 0.02,
        "depth_mean": 100000,
        "depth_sd": 3000,
        "n_pcr1": 16,
        "n_pcr2": 10,
        "atac_obs_prob": 0.3,
        "atac_sd_frac": 0.5,
    }

    clean = runs_by_job["boolode_test_invariant_clean"]["params"]["dropout"]
    dropout = runs_by_job["boolode_test_invariant_dropout"]["params"]["dropout"]
    assert clean == {"enabled": False}
    assert dropout == {"enabled": True, "drop_cutoff": 0.6, "drop_prob": 0.6}

    dyngen = [run["params"] for job_id, run in runs_by_job.items() if job_id.startswith("dyngen_")]
    assert {params["num_cells"] for params in dyngen} == {200, 300, 400, 500}
    assert {params["num_tfs"] + params["num_targets"] + params["num_hks"] for params in dyngen} == {
        70,
        100,
        130,
    }

    plan = load_json_object(REPO_ROOT / "experiments/v1/inference/plans/full-v1.json")
    group_batches = [batch for batch in plan["baseline_batches"] if batch["regime"] == "group_only"]
    assert len(group_batches) == 3
    primary_batches = [batch for batch in group_batches if "all-gene" not in batch["id"]]
    all_gene_batch = next(batch for batch in group_batches if "all-gene" in batch["id"])
    assert set().union(*(set(batch["generation_jobs"]) for batch in primary_batches)) == set(jobs)
    assert set(primary_batches[0]["generation_jobs"]).isdisjoint(
        primary_batches[1]["generation_jobs"]
    )
    assert set(all_gene_batch["generation_jobs"]) <= set(jobs)
    group_comparisons = [
        comparison
        for comparison in plan["statistics"]["comparisons"]
        if comparison["regime"] == "group_only"
    ]
    assert {comparison["analysis_kind"] for comparison in group_comparisons} == {
        "network_accuracy",
        "invariant_network_stability",
    }
    assert set().union(
        *(set(comparison["generation_jobs"]) for comparison in group_comparisons)
    ) == set(jobs)


def test_load_json_object_rejects_invalid_json_and_non_object(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="Cannot read test payload"):
        load_json_object(invalid, "test payload")

    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="must contain a JSON object"):
        load_json_object(array)


@pytest.mark.parametrize("profile_id", ["../smoke", "Smoke", "", "smoke.json"])
def test_profile_path_rejects_unsafe_or_noncanonical_identifiers(profile_id: str) -> None:
    with pytest.raises(BenchmarkConfigError, match="Invalid profile identifier"):
        profile_path(REPO_ROOT, profile_id)


def test_load_profile_checks_filename_identity_and_runnable_status(tmp_path: Path) -> None:
    profiles = tmp_path / "experiments" / "v1" / "profiles"
    profiles.mkdir(parents=True)
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    (schemas / "profile.schema.json").write_text(
        (REPO_ROOT / "schemas" / "profile.schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    payload = json.loads(
        (REPO_ROOT / "experiments" / "v1" / "profiles" / "reduced.json").read_text(encoding="utf-8")
    )
    payload["id"] = "different"
    (profiles / "reduced.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="does not match filename"):
        load_profile(tmp_path, "reduced")

    payload["id"] = "reduced"
    (profiles / "reduced.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="planned but not runnable"):
        load_profile(tmp_path, "reduced", require_runnable=True)


def test_resolve_repo_path_confines_paths_and_requires_a_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    member = root / "member.json"
    member.write_text("{}", encoding="utf-8")

    assert resolve_repo_path(root, "member.json", "member") == member.resolve()
    with pytest.raises(BenchmarkConfigError, match="escapes repository root"):
        resolve_repo_path(root, "../outside.json", "member")
    with pytest.raises(BenchmarkConfigError, match="does not exist"):
        resolve_repo_path(root, "missing.json", "member")


def test_resolve_repo_path_rejects_symlinked_file_and_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    external_file = outside / "member.json"
    external_file.write_text("{}", encoding="utf-8")

    (root / "linked-file.json").symlink_to(external_file)
    with pytest.raises(BenchmarkConfigError, match="symlink"):
        resolve_repo_path(root, "linked-file.json", "member")

    (root / "linked-directory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(BenchmarkConfigError, match="symlink"):
        resolve_repo_path(root, "linked-directory/member.json", "member")


def test_load_profile_file_rejects_symlinked_file_and_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    work = root / "work"
    work.mkdir(parents=True)
    schemas = root / "schemas"
    schemas.mkdir()
    (schemas / "profile.schema.json").write_text(
        (REPO_ROOT / "schemas" / "profile.schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    external_profile = outside / "smoke.json"
    external_profile.write_text(
        (REPO_ROOT / "experiments/v1/profiles/smoke.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    linked_file = work / "linked-profile.json"
    linked_file.symlink_to(external_profile)
    with pytest.raises(BenchmarkConfigError, match="symlink"):
        load_profile_file(root, linked_file, "smoke", require_runtime=True)

    linked_directory = work / "linked-directory"
    linked_directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(BenchmarkConfigError, match="symlink"):
        load_profile_file(
            root,
            linked_directory / "smoke.json",
            "smoke",
            require_runtime=True,
        )


def test_custom_tool_run_identity_must_match_definition() -> None:
    tools_params = {
        "runs": [
            {"run_id": "catalog", "tool_id": "genie3"},
            {"run_id": "spathi_primary", "tool_id": "custom_spathi_primary"},
        ]
    }
    custom_tools = {"tools": [{"run_id": "spathi_primary"}]}
    _validate_run_identity(
        tools_params=tools_params,
        custom_tools=custom_tools,
        label="test tools",
    )

    custom_tools["tools"][0]["run_id"] = "other"
    with pytest.raises(BenchmarkConfigError, match="Custom tool identity mismatch"):
        _validate_run_identity(
            tools_params=tools_params,
            custom_tools=custom_tools,
            label="test tools",
        )


def test_custom_run_id_must_derive_its_tool_id() -> None:
    with pytest.raises(BenchmarkConfigError, match="tool_id=custom_<run_id>"):
        _validate_run_identity(
            tools_params={"runs": [{"run_id": "spathi", "tool_id": "custom_spathi_primary"}]},
            custom_tools={"tools": [{"run_id": "spathi"}]},
            label="test tools",
        )


def test_run_ids_must_be_present_and_unique() -> None:
    with pytest.raises(BenchmarkConfigError, match="duplicate run IDs"):
        _validate_run_identity(
            tools_params={
                "runs": [
                    {"run_id": "same", "tool_id": "genie3"},
                    {"run_id": "same", "tool_id": "grnboost2"},
                ]
            },
            custom_tools=None,
            label="test tools",
        )


def test_profile_job_and_batch_ids_must_be_unique() -> None:
    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["generation_jobs"].append(deepcopy(profile["generation_jobs"][0]))
    with pytest.raises(BenchmarkConfigError, match="duplicate generation job IDs"):
        validate_profile_references(REPO_ROOT, profile)

    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["inference_batches"].append(deepcopy(profile["inference_batches"][0]))
    with pytest.raises(BenchmarkConfigError, match="duplicate inference batch IDs"):
        validate_profile_references(REPO_ROOT, profile)


def test_inference_batch_generation_jobs_must_be_non_empty_and_unique() -> None:
    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["inference_batches"][0]["generation_jobs"] = []
    with pytest.raises(BenchmarkConfigError, match="must be a non-empty array"):
        validate_profile_references(REPO_ROOT, profile)

    profile["inference_batches"][0]["generation_jobs"] = ["dyngen_smoke", "dyngen_smoke"]
    with pytest.raises(BenchmarkConfigError, match="duplicate generation job IDs"):
        validate_profile_references(REPO_ROOT, profile)


def test_inference_batch_generation_jobs_must_exist_and_match_role() -> None:
    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["inference_batches"][0]["generation_jobs"] = ["missing"]
    with pytest.raises(BenchmarkConfigError, match="unknown generation jobs"):
        validate_profile_references(REPO_ROOT, profile)

    development_job = deepcopy(profile["generation_jobs"][0])
    development_job.update({"id": "dyngen_development", "dataset_role": "development"})
    profile["generation_jobs"].append(development_job)
    profile["inference_batches"][0]["generation_jobs"] = ["dyngen_development"]
    with pytest.raises(BenchmarkConfigError, match="outside dataset role 'smoke'"):
        validate_profile_references(REPO_ROOT, profile)


def test_inference_batch_generation_jobs_accept_matching_subset() -> None:
    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["inference_batches"][0]["generation_jobs"] = ["dyngen_smoke"]

    validate_profile_references(REPO_ROOT, profile)


def test_statistics_generation_jobs_must_be_non_empty_unique_existing_and_in_role() -> None:
    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    comparison = profile["statistics"]["comparisons"][0]

    comparison["generation_jobs"] = []
    with pytest.raises(BenchmarkConfigError, match="must be a non-empty array"):
        validate_profile_references(REPO_ROOT, profile)

    comparison["generation_jobs"] = ["dyngen_smoke", "dyngen_smoke"]
    with pytest.raises(BenchmarkConfigError, match="duplicate generation job IDs"):
        validate_profile_references(REPO_ROOT, profile)

    comparison["generation_jobs"] = ["missing"]
    with pytest.raises(BenchmarkConfigError, match="unknown generation jobs"):
        validate_profile_references(REPO_ROOT, profile)

    development_job = deepcopy(profile["generation_jobs"][0])
    development_job.update({"id": "dyngen_development", "dataset_role": "development"})
    profile["generation_jobs"].append(development_job)
    comparison["generation_jobs"] = ["dyngen_development"]
    with pytest.raises(BenchmarkConfigError, match="outside dataset role 'smoke'"):
        validate_profile_references(REPO_ROOT, profile)


def test_statistics_configuration_batches_must_cover_selected_generation_jobs() -> None:
    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    second_job = deepcopy(profile["generation_jobs"][0])
    second_job["id"] = "dyngen_smoke_second"
    profile["generation_jobs"].append(second_job)
    comparison = profile["statistics"]["comparisons"][0]
    comparison["generation_jobs"] = ["dyngen_smoke_second"]

    validate_profile_references(REPO_ROOT, profile)

    spathi_batch = next(
        batch for batch in profile["inference_batches"] if batch["kind"] == "spathi"
    )
    spathi_batch["generation_jobs"] = ["dyngen_smoke"]
    with pytest.raises(
        BenchmarkConfigError,
        match="spathi_smoke.*coverage.*dyngen_smoke_second",
    ):
        validate_profile_references(REPO_ROOT, profile)


def test_runnable_profiles_cannot_mix_dataset_partitions() -> None:
    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["purpose"] = "calibration"

    with pytest.raises(BenchmarkConfigError, match="must contain exactly dataset roles"):
        validate_profile_references(REPO_ROOT, profile)

    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["inference_batches"][0]["dataset_role"] = "development"
    with pytest.raises(BenchmarkConfigError, match="no generation job provides it"):
        validate_profile_references(REPO_ROOT, profile)


def test_test_profile_requires_an_efficiency_gated_production_policy() -> None:
    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["purpose"] = "benchmark"
    profile["generation_jobs"][0]["dataset_role"] = "test"
    for batch in profile["inference_batches"]:
        batch["dataset_role"] = "test"

    with pytest.raises(BenchmarkConfigError, match="requires an efficiency-gated"):
        validate_profile_references(REPO_ROOT, profile)


def test_locked_repository_checks_commit_origin_and_cleanliness(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Tests")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "remote", "add", "origin", "git@github.com:example/project.git")
    head = _git(repo, "rev-parse", "HEAD")
    lock = {"commit": head, "url": "https://github.com/example/project.git"}

    result = validate_locked_repository(
        name="fixture",
        path=repo,
        lock_entry=lock,
        require_clean=True,
    )
    assert result["commit"] == head
    assert result["clean"] is True

    linked_repo = tmp_path / "linked-repo"
    linked_repo.symlink_to(repo, target_is_directory=True)
    with pytest.raises(BenchmarkConfigError, match="symlink"):
        validate_locked_repository(
            name="fixture",
            path=linked_repo,
            lock_entry=lock,
            require_clean=True,
        )

    (repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="worktree is not clean"):
        validate_locked_repository(
            name="fixture",
            path=repo,
            lock_entry=lock,
            require_clean=True,
        )


def test_locked_repository_rejects_wrong_head(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(BenchmarkConfigError, match="is missing"):
        validate_locked_repository(
            name="fixture",
            path=missing,
            lock_entry={"commit": "0" * 40, "url": "https://example.invalid/repo.git"},
            require_clean=False,
        )
