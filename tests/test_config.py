from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from spathi_bench.config import (
    BenchmarkConfigError,
    _validate_run_identity,
    load_image_lock,
    load_json_object,
    load_profile,
    load_repo_lock,
    profile_path,
    resolve_repo_path,
    validate_all_profiles,
    validate_design_contracts,
    validate_locked_repository,
    validate_profile_references,
)

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
        "containers/spathi/run_tool.py",
    }
    assert {profile["id"] for profile in profiles} == {
        "smoke",
        "calibration",
        "selection",
        "reduced",
        "full",
    }
    assert {profile["id"] for profile in profiles if profile["status"] == "runnable"} == {"smoke"}


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


def test_runnable_profiles_cannot_mix_dataset_partitions() -> None:
    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["purpose"] = "calibration"

    with pytest.raises(BenchmarkConfigError, match="must contain exactly dataset roles"):
        validate_profile_references(REPO_ROOT, profile)

    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["inference_batches"][0]["dataset_role"] = "development"
    with pytest.raises(BenchmarkConfigError, match="no generation job provides it"):
        validate_profile_references(REPO_ROOT, profile)


def test_test_profile_requires_a_frozen_spathi_selection() -> None:
    profile = deepcopy(load_profile(REPO_ROOT, "smoke"))
    profile["purpose"] = "benchmark"
    profile["generation_jobs"][0]["dataset_role"] = "test"
    for batch in profile["inference_batches"]:
        batch["dataset_role"] = "test"

    with pytest.raises(BenchmarkConfigError, match="cannot become runnable before"):
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
