from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.benchmark_materialization as materialization
import spathi_bench.config as config_module
from spathi_bench.artifacts import canonical_json_sha256
from spathi_bench.benchmark_materialization import (
    SELECTED_PLACEHOLDER,
    build_runnable_benchmark_profile,
    build_selected_spathi_requests,
    load_frozen_selected_candidate,
    materialize_benchmark_plan,
)
from spathi_bench.config import (
    BenchmarkConfigError,
    load_json_object,
    load_profile,
    validate_all_profiles,
    validate_json,
    validate_profile_references,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _selected_candidate(repo: Path, *, candidate_id: str = "selected_winner") -> dict[str, Any]:
    search = load_json_object(repo / "experiments/v1/spathi/search-space.json")
    parameters = copy.deepcopy(search["reference_configuration"])
    commit = load_json_object(repo / "repos.lock.json")["repositories"]["spathi"]["commit"]
    return {
        "candidate_id": candidate_id,
        "spathi_commit": commit,
        "configuration_sha256": canonical_json_sha256(parameters),
        "parameters": parameters,
    }


def _copy_materialization_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for directory in ("schemas", "experiments"):
        shutil.copytree(REPO_ROOT / directory, repo / directory)
    for filename in ("images.lock.json", "repos.lock.json"):
        shutil.copyfile(REPO_ROOT / filename, repo / filename)
    return repo


def _catalog_plan(repo: Path) -> Path:
    template = load_json_object(repo / "experiments/v1/profiles/reduced.json")
    jobs = [job["id"] for job in template["generation_jobs"]]
    request_root = repo / "experiments/v1/inference/materialization-test"
    request_root.mkdir(parents=True)
    (request_root / "tools-params.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "genie3_reduced",
                        "tool_id": "genie3",
                        "params": {"seed": 1729},
                        "execution": {"mode": "group_emulated"},
                    }
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
                        "run_id": "genie3_reduced",
                        "configuration_id": "genie3_reduced",
                        "selection_role": "baseline",
                        "algorithm_seed": 1729,
                        "algorithm_seed_parameter": "seed",
                        "family": "catalog_baseline",
                    }
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
                "dataset_role": "development",
                "generation_jobs": jobs,
                "regime": "group_only",
                "tools_params": ("experiments/v1/inference/materialization-test/tools-params.json"),
                "run_metadata": ("experiments/v1/inference/materialization-test/run-metadata.json"),
            }
        ],
        "statistics": {
            "status": "enabled",
            "alpha": 0.05,
            "random_seed": 1729,
            "permutations": 99999,
            "bootstrap_resamples": 20000,
            "comparisons": [
                {
                    "id": "group-only-primary",
                    "comparison_role": "confirmatory",
                    "candidate_space": "tf_restricted",
                    "dataset_roles": ["development"],
                    "generation_jobs": jobs,
                    "regime": "group_only",
                    "levels": ["directed", "topology"],
                    "metrics": ["aupr", "auroc"],
                    "primary_level": "directed",
                    "primary_metric": "aupr",
                    "configuration_ids": ["genie3_reduced", SELECTED_PLACEHOLDER],
                    "control_configuration_id": SELECTED_PLACEHOLDER,
                    "stratify_by": ["dataset_role", "regime", "level"],
                    "min_blocks": 8,
                    "posthoc_scope": "control_vs_rest",
                    "correction": "holm",
                }
            ],
        },
    }
    plan_path = repo / "experiments/v1/inference/reduced-catalog-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return plan_path


def _patch_frozen_selection(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    search = load_json_object(repo / "experiments/v1/spathi/search-space.json")
    selected = {
        "status": "selected",
        "phase": "frozen",
        "calibration_budget": "reduced",
        "selected_candidate": _selected_candidate(repo),
    }

    def selection_fixture(
        _root: Path, _path: Path | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return search, selected

    monkeypatch.setattr(materialization, "validate_calibration_contracts", selection_fixture)
    monkeypatch.setattr(config_module, "validate_calibration_contracts", selection_fixture)


def test_selected_requests_reuse_predeclared_seeds_without_mutating_candidate() -> None:
    candidate = _selected_candidate(REPO_ROOT)
    before = copy.deepcopy(candidate)

    tools, custom, metadata = build_selected_spathi_requests(
        candidate,
        (1729, 31415, 104729),
        docker_image="spathi:test",
    )

    assert candidate == before
    assert [item["algorithm_seed"] for item in metadata["runs"]] == [1729, 31415, 104729]
    assert {item["configuration_id"] for item in metadata["runs"]} == {candidate["candidate_id"]}
    assert [item["params"]["random_seed"] for item in tools["runs"]] == [
        1729,
        31415,
        104729,
    ]
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
                "dataset_role": "development",
                "generation_jobs": jobs,
                "regime": "group_only",
                "tools_params": "tools.json",
                "run_metadata": "metadata.json",
            }
        ],
        "statistics": {
            "status": "enabled",
            "alpha": 0.05,
            "random_seed": 1729,
            "permutations": 99999,
            "bootstrap_resamples": 20000,
            "comparisons": [
                {
                    "id": comparison_id,
                    "dataset_roles": ["development"],
                    "generation_jobs": jobs,
                    "regime": regime,
                    "configuration_ids": ["baseline", SELECTED_PLACEHOLDER],
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
        candidate_id="selected_winner",
    )

    spathi_batches = [batch for batch in profile["inference_batches"] if batch["kind"] == "spathi"]
    assert {batch["regime"] for batch in spathi_batches} == {"group_only", "order_aware"}
    assert all(batch["generation_jobs"] == jobs for batch in spathi_batches)
    assert all(
        comparison["control_configuration_id"] == "selected_winner"
        for comparison in profile["statistics"]["comparisons"]
    )


def test_frozen_candidate_loader_rejects_unfrozen_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = load_json_object(REPO_ROOT / "experiments/v1/spathi/search-space.json")
    selected = {
        "status": "unselected",
        "phase": "planned",
        "calibration_budget": None,
        "selected_candidate": None,
    }
    monkeypatch.setattr(
        materialization,
        "validate_calibration_contracts",
        lambda _root: (search, selected),
    )
    with pytest.raises(BenchmarkConfigError, match="requires a frozen"):
        load_frozen_selected_candidate(REPO_ROOT, calibration_budget="reduced")


def test_frozen_candidate_loader_rejects_stale_repository_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _selected_candidate(REPO_ROOT)
    candidate["spathi_commit"] = "0" * 40
    selected = {
        "status": "selected",
        "phase": "frozen",
        "calibration_budget": "reduced",
        "selected_candidate": candidate,
    }
    search = load_json_object(REPO_ROOT / "experiments/v1/spathi/search-space.json")
    monkeypatch.setattr(
        materialization,
        "validate_calibration_contracts",
        lambda _root: (search, selected),
    )

    with pytest.raises(BenchmarkConfigError, match="different repos.lock.json commit"):
        load_frozen_selected_candidate(REPO_ROOT, calibration_budget="reduced")


def test_frozen_candidate_loader_rejects_another_calibration_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = load_json_object(REPO_ROOT / "experiments/v1/spathi/search-space.json")
    selected = {
        "status": "selected",
        "phase": "frozen",
        "calibration_budget": "reduced",
        "selected_candidate": _selected_candidate(REPO_ROOT),
    }
    monkeypatch.setattr(
        materialization,
        "validate_calibration_contracts",
        lambda _root: (search, selected),
    )

    with pytest.raises(BenchmarkConfigError, match="uses calibration budget"):
        load_frozen_selected_candidate(REPO_ROOT, calibration_budget="full")


def test_materializer_writes_reconciled_profile_and_resumes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    plan_path = _catalog_plan(repo)
    search = load_json_object(repo / "experiments/v1/spathi/search-space.json")
    candidate = _selected_candidate(repo)
    selected = {
        "status": "selected",
        "phase": "frozen",
        "calibration_budget": "reduced",
        "selected_candidate": candidate,
    }
    monkeypatch.setattr(
        materialization,
        "validate_calibration_contracts",
        lambda _root, _path=None: (search, selected),
    )
    monkeypatch.setattr(
        config_module,
        "validate_calibration_contracts",
        lambda _root, _path=None: (search, selected),
    )
    output_dir = repo / "experiments/v1/spathi/benchmark/v1/reduced-v1"
    profile_output = repo / "experiments/v1/profiles/reduced-v1.json"

    materialize_benchmark_plan(
        repo_root=repo,
        template_profile_id="reduced",
        baseline_plan_path=plan_path,
        output_dir=output_dir,
        profile_output=profile_output,
    )

    profile = load_json_object(profile_output)
    validate_profile_references(repo, profile)
    tools = load_json_object(output_dir / "tools-params.json")
    manifest = load_json_object(output_dir / "benchmark-manifest.json")
    assert profile["status"] == "runnable"
    assert len(tools["runs"]) == 2
    assert manifest["selected_config"]["candidate_id"] == candidate["candidate_id"]
    assert manifest["selected_config"]["calibration_budget"] == "reduced"
    assert manifest["algorithm_seeds"]["values"] == [1729, 31415]

    materialize_benchmark_plan(
        repo_root=repo,
        template_profile_id="reduced",
        baseline_plan_path=plan_path,
        output_dir=output_dir,
        profile_output=profile_output,
    )
    (output_dir / "tools-params.json").write_text('{"runs": []}', encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="differs from this workflow"):
        materialize_benchmark_plan(
            repo_root=repo,
            template_profile_id="reduced",
            baseline_plan_path=plan_path,
            output_dir=output_dir,
            profile_output=profile_output,
        )


@pytest.mark.parametrize("missing", ["bundle", "profile"])
def test_benchmark_materializer_recovers_one_missing_materialization_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    plan_path = _catalog_plan(repo)
    _patch_frozen_selection(repo, monkeypatch)
    output_dir = repo / "work/comparison/reduced/requests"
    profile_output = repo / "work/comparison/reduced/reduced-v1.json"
    arguments = {
        "repo_root": repo,
        "template_profile_id": "reduced",
        "baseline_plan_path": plan_path,
        "output_dir": output_dir,
        "profile_output": profile_output,
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
    monkeypatch: pytest.MonkeyPatch,
    survivor: str,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    plan_path = _catalog_plan(repo)
    _patch_frozen_selection(repo, monkeypatch)
    output_dir = repo / "work/comparison/reduced/requests"
    profile_output = repo / "work/comparison/reduced/reduced-v1.json"
    arguments = {
        "repo_root": repo,
        "template_profile_id": "reduced",
        "baseline_plan_path": plan_path,
        "output_dir": output_dir,
        "profile_output": profile_output,
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


def test_materializer_accepts_explicit_selection_and_profile_below_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    plan_path = _catalog_plan(repo)
    search = load_json_object(repo / "experiments/v1/spathi/search-space.json")
    candidate = _selected_candidate(repo)
    selected = {
        "status": "selected",
        "phase": "frozen",
        "calibration_budget": "reduced",
        "selected_candidate": candidate,
    }

    def selection_fixture(
        _root: Path, _path: Path | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return search, selected

    monkeypatch.setattr(materialization, "validate_calibration_contracts", selection_fixture)
    monkeypatch.setattr(config_module, "validate_calibration_contracts", selection_fixture)
    runtime = repo / "work/comparison/reduced"
    selected_path = runtime / "selected-config.json"
    selected_path.parent.mkdir(parents=True)
    shutil.copyfile(repo / "experiments/v1/spathi/selected-config.json", selected_path)

    bundle, profile_path = materialize_benchmark_plan(
        repo_root=repo,
        template_profile_id="reduced",
        baseline_plan_path=plan_path,
        output_dir=runtime / "requests",
        profile_output=runtime / "reduced-v1.json",
        selected_config_path=selected_path,
    )

    profile = load_json_object(profile_path)
    manifest = load_json_object(bundle / "benchmark-manifest.json")
    assert profile["selected_config"] == selected_path.relative_to(repo).as_posix()
    assert (
        profile["materialization_manifest"]
        == (bundle / "benchmark-manifest.json").relative_to(repo).as_posix()
    )
    assert manifest["selected_config"]["path"] == profile["selected_config"]


@pytest.mark.parametrize(
    ("template_id", "expected_batches", "expected_comparisons"),
    [("reduced", 2, 1), ("full", 6, 4)],
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
        candidate_id="selected_fixture",
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
    monkeypatch: pytest.MonkeyPatch,
    template_id: str,
) -> None:
    repo = _copy_materialization_repo(tmp_path)
    search = load_json_object(repo / "experiments/v1/spathi/search-space.json")
    candidate = _selected_candidate(repo)
    selected = {
        "status": "selected",
        "phase": "frozen",
        "calibration_budget": template_id,
        "selected_candidate": candidate,
    }

    def selection_fixture(
        _root: Path, _path: Path | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return search, selected

    monkeypatch.setattr(materialization, "validate_calibration_contracts", selection_fixture)
    monkeypatch.setattr(config_module, "validate_calibration_contracts", selection_fixture)

    output_dir = repo / f"experiments/v1/spathi/benchmark/v1/{template_id}-v1"
    profile_output = repo / f"experiments/v1/profiles/{template_id}-v1.json"
    materialize_benchmark_plan(
        repo_root=repo,
        template_profile_id=template_id,
        baseline_plan_path=(repo / f"experiments/v1/inference/plans/{template_id}-v1.json"),
        output_dir=output_dir,
        profile_output=profile_output,
    )

    profile = load_json_object(profile_output)
    assert profile["status"] == "runnable"
    assert all(batch.get("generation_jobs") for batch in profile["inference_batches"])
    assert f"{template_id}-v1" in {item["id"] for item in validate_all_profiles(repo)}
