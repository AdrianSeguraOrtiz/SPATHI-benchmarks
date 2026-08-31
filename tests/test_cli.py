from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import spathi_bench.cli as cli
from spathi_bench.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _planning_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for relative in (
        "images.lock.json",
        "schemas/calibration-manifest.schema.json",
        "schemas/calibration-ranking.schema.json",
        "schemas/profile.schema.json",
        "schemas/spathi-search-space.schema.json",
        "experiments/v1/profiles/calibration-full.json",
        "experiments/v1/spathi/search-space.json",
    ):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    template_path = repo / "experiments/v1/profiles/calibration-full.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["generation_jobs"] = [
        {
            "id": "development-panel",
            "scenario": "experiments/v1/simulations/development/scenario.json",
            "simulator_runs": "experiments/v1/simulations/development/runs.json",
            "dataset_role": "development",
            "split_id": "development",
            "condition_id": "calibration",
        }
    ]
    template_path.write_text(json.dumps(template), encoding="utf-8")
    return repo


def test_calibration_plan_and_rank_are_cli_executable(tmp_path: Path) -> None:
    repo = _planning_repo(tmp_path)
    plan = repo / "experiments/v1/spathi/generated/convergence"
    profile = repo / "experiments/v1/profiles/calibration-full-convergence.json"
    assert (
        main(
            [
                "calibration-plan",
                "--repo-root",
                str(repo),
                "--stage",
                "convergence",
                "--calibration-budget",
                "full",
                "--output-dir",
                str(plan),
                "--profile-output",
                str(profile),
            ]
        )
        == 0
    )
    manifest = json.loads((plan / "calibration-manifest.json").read_text(encoding="utf-8"))
    schema = json.loads(
        (repo / "schemas/calibration-manifest.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(manifest)
    tools = json.loads((plan / "tools-params.json").read_text(encoding="utf-8"))
    metadata = json.loads((plan / "run-metadata.json").read_text(encoding="utf-8"))
    assert len(manifest["candidates"]) == 4
    assert len(tools["runs"]) == len(metadata["runs"]) == 20
    digests = {
        candidate["configuration_id"]: candidate["configuration_sha256"]
        for candidate in manifest["candidates"]
    }

    summary_rows = [
        {
            "configuration_id": run["configuration_id"],
            "configuration_sha256": digests[run["configuration_id"]],
            "dataset_role": "development",
            "dataset_id": dataset_id,
            "algorithm_seed": run["algorithm_seed"],
            "run_id": run["run_id"],
            "execution_sha256": next(
                item["execution_sha256"]
                for item in manifest["runs"]
                if item["run_id"] == run["run_id"]
            ),
            "aupr_directed": 0.5,
            "wall_time_seconds": 10.0,
            "peak_memory_mb": None,
            "n_group_contexts": 2,
            "group_contexts": ["group:a", "group:b"],
            "spathi_commit": manifest["spathi_commit"],
            "spathi_build_fingerprint": manifest["spathi_build_fingerprint"],
            "spathi_image_id": "sha256:" + "1" * 64,
        }
        for run in metadata["runs"]
        for dataset_id in ("dataset_a", "dataset_b")
    ]
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "profile_id": "calibration-full-convergence",
                "purpose": "calibration",
                "benchmark_commit": "1" * 40,
                "calibration_rows": summary_rows,
            }
        ),
        encoding="utf-8",
    )
    ranking = repo / "experiments/v1/spathi/evidence/convergence-ranking.json"
    assert (
        main(
            [
                "calibration-rank",
                "--repo-root",
                str(repo),
                "--stage",
                "convergence",
                "--calibration-budget",
                "full",
                "--manifest",
                str(plan / "calibration-manifest.json"),
                "--summary",
                str(summary),
                "--output",
                str(ranking),
            ]
        )
        == 0
    )
    result = json.loads(ranking.read_text(encoding="utf-8"))
    assert len(result["selected_candidates"]) == 1
    assert result["selected_candidates"][0]["variant_id"] == "reference"
    assert result["test_data_accessed"] is False


def test_calibration_plan_is_idempotent_below_work(tmp_path: Path) -> None:
    repo = _planning_repo(tmp_path)
    bundle = repo / "work/calibration/convergence/requests"
    profile = repo / "work/calibration/convergence/calibration-full-convergence.json"
    arguments = [
        "calibration-plan",
        "--repo-root",
        str(repo),
        "--stage",
        "convergence",
        "--calibration-budget",
        "full",
        "--output-dir",
        str(bundle),
        "--profile-output",
        str(profile),
    ]

    assert main(arguments) == 0
    first = profile.read_bytes()
    assert main(arguments) == 0
    assert profile.read_bytes() == first


@pytest.mark.parametrize("missing", ["bundle", "profile"])
def test_calibration_plan_recovers_one_missing_materialization_member(
    tmp_path: Path,
    missing: str,
) -> None:
    repo = _planning_repo(tmp_path)
    bundle = repo / "work/calibration/convergence/requests"
    profile = repo / "work/calibration/convergence/calibration-full-convergence.json"
    arguments = [
        "calibration-plan",
        "--repo-root",
        str(repo),
        "--stage",
        "convergence",
        "--calibration-budget",
        "full",
        "--output-dir",
        str(bundle),
        "--profile-output",
        str(profile),
    ]
    assert main(arguments) == 0
    expected_bundle = {path.name: path.read_bytes() for path in bundle.iterdir() if path.is_file()}
    expected_profile = profile.read_bytes()

    if missing == "bundle":
        shutil.rmtree(bundle)
    else:
        profile.unlink()

    assert main(arguments) == 0
    assert {path.name: path.read_bytes() for path in bundle.iterdir()} == expected_bundle
    assert profile.read_bytes() == expected_profile


@pytest.mark.parametrize("survivor", ["bundle", "profile"])
def test_calibration_plan_rejects_a_divergent_partial_materialization(
    tmp_path: Path,
    survivor: str,
) -> None:
    repo = _planning_repo(tmp_path)
    bundle = repo / "work/calibration/convergence/requests"
    profile = repo / "work/calibration/convergence/calibration-full-convergence.json"
    arguments = [
        "calibration-plan",
        "--repo-root",
        str(repo),
        "--stage",
        "convergence",
        "--calibration-budget",
        "full",
        "--output-dir",
        str(bundle),
        "--profile-output",
        str(profile),
    ]
    assert main(arguments) == 0

    if survivor == "bundle":
        profile.unlink()
        (bundle / "tools-params.json").write_text('{"runs": []}', encoding="utf-8")
        missing_path = profile
    else:
        shutil.rmtree(bundle)
        payload = json.loads(profile.read_text(encoding="utf-8"))
        payload["description"] = "divergent profile"
        profile.write_text(json.dumps(payload), encoding="utf-8")
        missing_path = bundle

    assert main(arguments) == 2
    assert not missing_path.exists()


def test_high_level_calibrate_cli_uses_default_work_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "run_calibration_workflow",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )
    repo = tmp_path / "repo"

    assert (
        main(
            [
                "calibrate",
                "--repo-root",
                str(repo),
                "--andrea-repo",
                str(tmp_path / "ANDREA"),
                "--spathi-repo",
                str(tmp_path / "SPATHI"),
            ]
        )
        == 0
    )
    assert calls[0]["work_root"] == repo.resolve() / "work"
    assert calls[0]["calibration_budget"] == "full"
    assert calls[0]["run_id"] is None


def test_high_level_calibrate_cli_forwards_reduced_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "run_calibration_workflow",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )

    assert (
        main(
            [
                "calibrate",
                "--repo-root",
                str(tmp_path / "repo"),
                "--andrea-repo",
                str(tmp_path / "ANDREA"),
                "--spathi-repo",
                str(tmp_path / "SPATHI"),
                "--profile",
                "reduced",
            ]
        )
        == 0
    )
    assert calls[0]["calibration_budget"] == "reduced"


def test_high_level_compare_cli_defaults_to_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "run_comparison_workflow",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )

    assert (
        main(
            [
                "compare",
                "--repo-root",
                str(tmp_path / "repo"),
                "--andrea-repo",
                str(tmp_path / "ANDREA"),
                "--spathi-repo",
                str(tmp_path / "SPATHI"),
            ]
        )
        == 0
    )
    assert calls[0]["profile"] == "full"
    assert calls[0]["calibration_run_id"] is None
