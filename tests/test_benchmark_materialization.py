from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.benchmark_materialization as materialization
from selection_helpers import selected_document
from spathi_bench.artifacts import canonical_json_sha256, sha256_file
from spathi_bench.benchmark_materialization import (
    SELECTED_PLACEHOLDER,
    build_runnable_benchmark_profile,
    build_selected_spathi_requests,
    load_frozen_selected_policy,
    materialize_benchmark_plan,
)
from spathi_bench.calibration import materialize_group_policy
from spathi_bench.config import (
    BenchmarkConfigError,
    load_json_object,
    load_profile,
    validate_all_profiles,
    validate_json,
    validate_profile_references,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _stub_calibration_reconciliation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep downstream tests separate from the independently tested ranking chain."""
    import spathi_bench.calibration as calibration
    monkeypatch.setattr(calibration, "reconcile_frozen_selection", lambda *_args: None)


def _selected_policy(repo: Path, *, candidate_id: str = "selected_winner") -> dict[str, Any]:
    search = load_json_object(repo / "experiments/v1/spathi/search-space.json")
    parameters = copy.deepcopy(search["base_configuration"])
    group_policy = materialize_group_policy(search, parameters)
    commit = load_json_object(repo / "repos.lock.json")["repositories"]["spathi"]["commit"]
    return {
        "policy_id": "spathi_conditional_v1",
        "source_candidate_id": candidate_id,
        "spathi_commit": commit,
        "source_configuration_sha256": canonical_json_sha256(parameters),
        "policy_sha256": canonical_json_sha256(group_policy),
        "group_policy": group_policy,
    }


def _copy_materialization_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for directory in ("schemas", "experiments"):
        shutil.copytree(REPO_ROOT / directory, repo / directory)
    for filename in ("images.lock.json", "repos.lock.json"):
        shutil.copyfile(REPO_ROOT / filename, repo / filename)
    return repo


def _write_selected_config(repo: Path, budget: str) -> Path:
    root = repo / "work/v1/workflows/calibration" / budget / "test-flow"
    root.mkdir(parents=True, exist_ok=True)
    ranking_path = root / "ranking.json"
    ranking_path.write_text(json.dumps({"parent_ranking": None}) + "\n", encoding="utf-8")
    selection = selected_document(
        _selected_policy(repo), budget=budget,
        ranking_path=ranking_path.relative_to(repo).as_posix(),
        ranking_sha256=sha256_file(ranking_path),
    )
    output = root / "selected-config.json"
    output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return output


def _catalog_plan(repo: Path) -> Path:
    template = load_json_object(repo / "experiments/v1/profiles/reduced.json")
    jobs = [job["id"] for job in template["generation_jobs"]]
    algorithm_seeds = (1107144475, 512584353, 238683595)
    request_root = repo / "experiments/v1/inference/materialization-test"
    request_root.mkdir(parents=True)
    (request_root / "tools-params.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": f"genie3_reduced_seed_{seed}",
                        "tool_id": "genie3",
                        "params": {"seed": seed},
                        "execution": {"mode": "group_emulated"},
                    }
                    for seed in algorithm_seeds
                ]
            }
        ),
        encoding="utf-8",
    )
    (request_root / "run-metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "runs": [
                    {
                        "run_id": f"genie3_reduced_seed_{seed}",
                        "configuration_id": "genie3_reduced",
                        "selection_role": "baseline",
                        "algorithm_seed": seed,
                        "algorithm_seed_parameter": "seed",
                        "family": "catalog_baseline",
                        "seed_derivation": {
                            "namespace": "reduced.test",
                            "stream": "inference",
                            "coordinates": {
                                "purpose": "algorithm_replicate",
                                "replicate_index": replicate_index,
                            },
                        },
                    }
                    for replicate_index, seed in enumerate(algorithm_seeds, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = {
        "schema_version": "1.0",
        "template_profile_id": "reduced",
        "baseline_batches": [
            {
                "id": "group-only-baselines",
                "kind": "baseline",
                "dataset_role": "test",
                "generation_jobs": jobs,
                "regime": "group_only",
                "tools_params": ("experiments/v1/inference/materialization-test/tools-params.json"),
                "run_metadata": ("experiments/v1/inference/materialization-test/run-metadata.json"),
            }
        ],
        "statistics": {
            "status": "descriptive",
            "comparisons": [
                {
                    "id": "group-only-primary",
                    "comparison_role": "exploratory",
                    "analysis_kind": "network_accuracy",
                    "truth_scope": "group_specific",
                    "aggregation": {
                        "block_unit": "dataset_id",
                        "hierarchy": ["simulator_id", "condition_id", "dataset_id"],
                        "weighting": "equal_dataset",
                    },
                    "candidate_space": "tf_restricted",
                    "inference_regulator_scope_by_configuration": {
                        "genie3_reduced": "tf_restricted",
                        SELECTED_PLACEHOLDER: "tf_restricted",
                    },
                    "dataset_roles": ["test"],
                    "generation_jobs": jobs,
                    "regime": "group_only",
                    "levels": ["directed", "topology"],
                    "metrics": ["aupr", "auroc"],
                    "primary_level": "directed",
                    "primary_metric": "aupr",
                    "configuration_ids": ["genie3_reduced", SELECTED_PLACEHOLDER],
                    "algorithm_seeds_by_configuration": {
                        "genie3_reduced": list(algorithm_seeds),
                        SELECTED_PLACEHOLDER: list(algorithm_seeds),
                    },
                    "control_configuration_id": SELECTED_PLACEHOLDER,
                    "stratify_by": ["dataset_role", "regime", "level"],
                    "expected_blocks": 8,
                }
            ],
        },
    }
    plan_path = repo / "experiments/v1/inference/reduced-catalog-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path


def test_selected_requests_reuse_predeclared_seeds_without_mutating_policy() -> None:
    policy = _selected_policy(REPO_ROOT)
    before = copy.deepcopy(policy)

    tools, custom, metadata = build_selected_spathi_requests(
        policy,
        (1729, 31415, 104729),
        docker_image="spathi:test",
        seed_namespace="full.test",
    )

    assert policy == before
    assert [item["algorithm_seed"] for item in metadata["runs"]] == [1729, 31415, 104729]
    assert {item["configuration_id"] for item in metadata["runs"]} == {policy["policy_id"]}
    assert [item["params"]["random_seed"] for item in tools["runs"]] == [
        1729,
        31415,
        104729,
    ]
    assert all(item["params"]["group_policy"] == policy["group_policy"] for item in tools["runs"])
    assert {item["run_id"] for item in custom["tools"]} == {
        item["run_id"] for item in tools["runs"]
    }


def test_profile_builder_creates_selected_spathi_route_for_each_unique_selector() -> None:
    template = load_json_object(REPO_ROOT / "experiments/v1/profiles/reduced.json")
    jobs = [job["id"] for job in template["generation_jobs"]]
    plan = {
        "baseline_batches": [
            {
                "id": "baseline",
                "kind": "baseline",
                "dataset_role": "test",
                "generation_jobs": jobs,
                "regime": "group_only",
                "tools_params": "tools.json",
                "run_metadata": "metadata.json",
            }
        ],
        "statistics": {
            "status": "descriptive",
            "comparisons": [
                {
                    "id": comparison_id,
                    "dataset_roles": ["test"],
                    "generation_jobs": jobs,
                    "regime": regime,
                    "configuration_ids": ["baseline", SELECTED_PLACEHOLDER],
                    "algorithm_seeds_by_configuration": {
                        "baseline": [1],
                        SELECTED_PLACEHOLDER: [1],
                    },
                    "inference_regulator_scope_by_configuration": {
                        "baseline": "tf_restricted",
                        SELECTED_PLACEHOLDER: "tf_restricted",
                    },
                    "control_configuration_id": SELECTED_PLACEHOLDER,
                }
                for comparison_id, regime in (
                    ("primary", "group_only"),
                    ("ordered", "order_aware"),
                )
            ],
        },
    }

    profile = build_runnable_benchmark_profile(
        template,
        plan,
        profile_id="reduced-v1",
        bundle_path="experiments/v1/spathi/benchmark/v1/reduced",
        policy_id="spathi_conditional_v1",
        selected_config_path="work/v1/workflows/calibration/reduced/run/selected-config.json",
    )

    spathi_batches = [batch for batch in profile["inference_batches"] if batch["kind"] == "spathi"]
    assert {batch["regime"] for batch in spathi_batches} == {"group_only", "order_aware"}
    assert all(batch["generation_jobs"] == jobs for batch in spathi_batches)
    assert all(
        comparison["control_configuration_id"] == "spathi_conditional_v1"
        for comparison in profile["statistics"]["comparisons"]
    )


def test_selected_policy_loader_validates_budget_commit_and_provenance(
    tmp_path: Path,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    path = _write_selected_config(repo, "reduced")
    policy, seeds = load_frozen_selected_policy(repo, path, budget="reduced")

    assert policy["policy_id"] == "spathi_conditional_v1"
    assert seeds == (1107144475, 512584353, 238683595)

    with pytest.raises(BenchmarkConfigError, match="uses budget"):
        load_frozen_selected_policy(repo, path, budget="full")

    production = load_json_object(path)
    production["selected_policy"]["spathi_commit"] = "0" * 40
    path.write_text(json.dumps(production), encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="different repos.lock.json commit"):
        load_frozen_selected_policy(repo, path, budget="reduced")


def test_selected_policy_requires_immutable_evidence_reverification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    path = _write_selected_config(repo, "reduced")

    def reject_evidence(*_args: Any, **_kwargs: Any) -> tuple[Path, Path, Path]:
        raise BenchmarkConfigError("immutable evidence differs")

    monkeypatch.setattr(materialization, "validate_calibration_contracts", reject_evidence)
    with pytest.raises(BenchmarkConfigError, match="immutable evidence differs"):
        load_frozen_selected_policy(repo, path, budget="reduced")


def test_materializer_writes_reconciled_profile_and_resumes_idempotently(
    tmp_path: Path,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    plan_path = _catalog_plan(repo)
    production_path = _write_selected_config(repo, "reduced")
    policy = load_json_object(production_path)["selected_policy"]
    output_dir = repo / "experiments/v1/spathi/benchmark/v1/reduced-v1"
    profile_output = repo / "experiments/v1/profiles/reduced-v1.json"

    materialize_benchmark_plan(
        repo_root=repo,
        template_profile_id="reduced",
        baseline_plan_path=plan_path,
        output_dir=output_dir,
        profile_output=profile_output,
        selected_config_path=production_path,
    )

    profile = load_json_object(profile_output)
    validate_profile_references(repo, profile)
    tools = load_json_object(output_dir / "tools-params.json")
    manifest = load_json_object(output_dir / "benchmark-manifest.json")
    assert profile["status"] == "runnable"
    assert len(tools["runs"]) == 3
    assert manifest["selected_config"]["policy_id"] == policy["policy_id"]
    assert manifest["selected_config"]["budget"] == "reduced"
    assert manifest["algorithm_seeds"]["values"] == [1107144475, 512584353, 238683595]

    materialize_benchmark_plan(
        repo_root=repo,
        template_profile_id="reduced",
        baseline_plan_path=plan_path,
        output_dir=output_dir,
        profile_output=profile_output,
        selected_config_path=production_path,
    )
    (output_dir / "tools-params.json").write_text('{"runs": []}', encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="differs from this workflow"):
        materialize_benchmark_plan(
            repo_root=repo,
            template_profile_id="reduced",
            baseline_plan_path=plan_path,
            output_dir=output_dir,
            profile_output=profile_output,
            selected_config_path=production_path,
        )


@pytest.mark.parametrize("missing", ["bundle", "profile"])
def test_benchmark_materializer_recovers_one_missing_materialization_member(
    tmp_path: Path,
    missing: str,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    plan_path = _catalog_plan(repo)
    production_path = _write_selected_config(repo, "reduced")
    output_dir = repo / "work/comparison/reduced/requests"
    profile_output = repo / "work/comparison/reduced/reduced-v1.json"
    arguments = {
        "repo_root": repo,
        "template_profile_id": "reduced",
        "baseline_plan_path": plan_path,
        "output_dir": output_dir,
        "profile_output": profile_output,
        "selected_config_path": production_path,
    }
    materialize_benchmark_plan(**arguments)
    expected_bundle = {
        path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
    }
    expected_profile = profile_output.read_bytes()

    if missing == "bundle":
        shutil.rmtree(output_dir)
    else:
        profile_output.unlink()

    materialize_benchmark_plan(**arguments)
    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == expected_bundle
    assert profile_output.read_bytes() == expected_profile


@pytest.mark.parametrize("survivor", ["bundle", "profile"])
def test_benchmark_materializer_rejects_a_divergent_partial_materialization(
    tmp_path: Path,
    survivor: str,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    plan_path = _catalog_plan(repo)
    production_path = _write_selected_config(repo, "reduced")
    output_dir = repo / "work/comparison/reduced/requests"
    profile_output = repo / "work/comparison/reduced/reduced-v1.json"
    arguments = {
        "repo_root": repo,
        "template_profile_id": "reduced",
        "baseline_plan_path": plan_path,
        "output_dir": output_dir,
        "profile_output": profile_output,
        "selected_config_path": production_path,
    }
    materialize_benchmark_plan(**arguments)

    if survivor == "bundle":
        profile_output.unlink()
        (output_dir / "tools-params.json").write_text('{"runs": []}', encoding="utf-8")
        missing_path = profile_output
    else:
        shutil.rmtree(output_dir)
        payload = load_json_object(profile_output)
        payload["description"] = "divergent profile"
        profile_output.write_text(json.dumps(payload), encoding="utf-8")
        missing_path = output_dir

    with pytest.raises(BenchmarkConfigError, match="differs from this workflow"):
        materialize_benchmark_plan(**arguments)
    assert not missing_path.exists()


def test_materializer_accepts_selected_policy_and_profile_below_work(
    tmp_path: Path,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    plan_path = _catalog_plan(repo)
    runtime = repo / "work/comparison/reduced"
    production_path = _write_selected_config(repo, "reduced")

    bundle, profile_path = materialize_benchmark_plan(
        repo_root=repo,
        template_profile_id="reduced",
        baseline_plan_path=plan_path,
        output_dir=runtime / "requests",
        profile_output=runtime / "reduced-v1.json",
        selected_config_path=production_path,
    )

    profile = load_json_object(profile_path)
    manifest = load_json_object(bundle / "benchmark-manifest.json")
    assert profile["selected_config"] == production_path.relative_to(repo).as_posix()
    assert (
        profile["materialization_manifest"]
        == (bundle / "benchmark-manifest.json").relative_to(repo).as_posix()
    )
    assert manifest["selected_config"]["path"] == profile["selected_config"]


@pytest.mark.parametrize(
    ("template_id", "expected_batches", "expected_comparisons"),
    [("reduced", 6, 4), ("full", 8, 7)],
)
def test_committed_benchmark_plans_build_valid_runnable_profiles(
    template_id: str,
    expected_batches: int,
    expected_comparisons: int,
) -> None:
    plan, _relative = materialization._load_catalog_plan(
        REPO_ROOT,
        REPO_ROOT / f"experiments/v1/inference/plans/{template_id}-v1.json",
        template_profile_id=template_id,
    )
    profile = build_runnable_benchmark_profile(
        load_profile(REPO_ROOT, template_id),
        plan,
        profile_id=f"{template_id}-v1",
        bundle_path=f"experiments/v1/spathi/benchmark/v1/{template_id}-v1",
        policy_id="spathi_conditional_v1",
        selected_config_path=(
            f"work/v1/workflows/calibration/{template_id}/run/selected-config.json"
        ),
    )

    validate_json(
        profile,
        REPO_ROOT / "schemas/profile.schema.json",
        f"profile {template_id}-v1 fixture",
    )
    materialization._validate_baseline_inputs(REPO_ROOT, profile)
    assert len(profile["inference_batches"]) == expected_batches
    assert len(profile["statistics"]["comparisons"]) == expected_comparisons


@pytest.mark.parametrize("template_id", ["reduced", "full"])
def test_committed_plan_materializes_with_a_reconciled_selection_fixture(
    tmp_path: Path,
    template_id: str,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    production_path = _write_selected_config(repo, template_id)

    output_dir = repo / f"experiments/v1/spathi/benchmark/v1/{template_id}-v1"
    profile_output = repo / f"experiments/v1/profiles/{template_id}-v1.json"
    materialize_benchmark_plan(
        repo_root=repo,
        template_profile_id=template_id,
        baseline_plan_path=(repo / f"experiments/v1/inference/plans/{template_id}-v1.json"),
        output_dir=output_dir,
        profile_output=profile_output,
        selected_config_path=production_path,
    )

    profile = load_json_object(profile_output)
    assert profile["status"] == "runnable"
    assert all(batch.get("generation_jobs") for batch in profile["inference_batches"])
    assert f"{template_id}-v1" in {item["id"] for item in validate_all_profiles(repo)}
