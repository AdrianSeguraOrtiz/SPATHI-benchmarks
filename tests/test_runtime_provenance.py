from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.provenance import (
    freeze_run_inputs,
    frozen_profile,
    validate_run_inputs,
)

SOURCE_REPOSITORY = Path(__file__).resolve().parents[1]
PROFILE_ID = "calibration-runtime"
RUNTIME_MEMBERS = (
    "work/materialized/tools-params.json",
    "work/materialized/custom-tools.json",
    "work/materialized/run-metadata.json",
)


def _write_json(root: Path, relative: str, payload: dict[str, Any]) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _runtime_profile() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "id": PROFILE_ID,
        "status": "runnable",
        "purpose": "calibration",
        "calibration_stage": "convergence",
        "description": "Runtime-derived calibration profile fixture.",
        "generation_jobs": [
            {
                "id": "fixture_development",
                "scenario": "experiments/v1/simulations/fixture/scenario.json",
                "simulator_runs": "experiments/v1/simulations/fixture/runs.json",
                "dataset_role": "development",
                "split_id": "development_fixture",
                "condition_id": "fixture",
            }
        ],
        "inference_batches": [
            {
                "id": "spathi-runtime",
                "kind": "spathi",
                "dataset_role": "development",
                "regime": "group_only",
                "tools_params": RUNTIME_MEMBERS[0],
                "custom_tools": RUNTIME_MEMBERS[1],
                "run_metadata": RUNTIME_MEMBERS[2],
            }
        ],
        "resources": {
            "generate_data": {
                "max_parallel_tasks": 1,
                "max_cores": 1,
                "max_ram_gb": 1,
            },
            "infer_network": {
                "max_parallel_runs": 1,
                "max_cores": 1,
                "max_ram_gb": 1,
                "planner": "heuristic",
                "planner_time_limit_seconds": 1,
            },
        },
        "planned_design": {
            "simulators": ["dyngen"],
            "replicates_per_condition": 1,
            "regimes": ["group_only"],
            "factors": ["runtime provenance"],
            "notes": ["Test fixture."],
        },
        "statistics": {
            "status": "not_applicable",
            "alpha": 0.05,
            "random_seed": 1,
            "permutations": 9999,
            "bootstrap_resamples": 1000,
            "comparisons": [],
        },
    }


def _repository(
    tmp_path: Path,
    *,
    scientific_runtime_input: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Tests")
    _git(
        root,
        "remote",
        "add",
        "origin",
        "https://example.invalid/spathi-benchmarks.git",
    )

    (root / ".gitignore").write_text("work/\nruntime-profile.json\n", encoding="utf-8")
    schema = root / "schemas/profile.schema.json"
    schema.parent.mkdir(parents=True)
    schema.write_text(
        (SOURCE_REPOSITORY / "schemas/profile.schema.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _write_json(root, "repos.lock.json", {"schema_version": "1.0"})
    build_input = root / "containers/spathi/Dockerfile"
    build_input.parent.mkdir(parents=True)
    build_input.write_text("FROM scratch\n", encoding="utf-8")
    _write_json(
        root,
        "images.lock.json",
        {
            "schema_version": "1.0",
            "catalog_images": {},
            "images": {
                "spathi": {
                    "tag": "spathi:test",
                    "base_image": "python@test",
                    "source_repository": "spathi",
                    "source_commit": "1" * 40,
                    "build_inputs": {"containers/spathi/Dockerfile": "2" * 64},
                    "python_dependencies": {"spathi": "1.0"},
                }
            },
        },
    )
    _write_json(root, "experiments/v1/spathi/search-space.json", {})
    _write_json(
        root,
        "experiments/v1/spathi/selected-config.json",
        {"selection_evidence": None},
    )
    inputs: dict[str, Any] = {}
    if scientific_runtime_input:
        inputs["regulatory_network"] = {
            "path": str((root / "work/scientific-network.tsv").resolve())
        }
    _write_json(
        root,
        "experiments/v1/simulations/fixture/scenario.json",
        {"schema_version": "1.0", "id": "fixture", "inputs": inputs},
    )
    _write_json(
        root,
        "experiments/v1/simulations/fixture/runs.json",
        {"runs": [{"run_id": "fixture", "simulator_id": "dyngen"}]},
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "fixture")

    for relative in RUNTIME_MEMBERS:
        _write_json(root, relative, {"runs": []})
    if scientific_runtime_input:
        scientific = root / "work/scientific-network.tsv"
        scientific.parent.mkdir(parents=True, exist_ok=True)
        scientific.write_text("target\tregulator\nG2\tG1\n", encoding="utf-8")
    profile_path = _write_json(
        root,
        "work/materialized/profile.json",
        _runtime_profile(),
    )
    assert _git(root, "status", "--porcelain") == ""
    return root, profile_path


def test_runtime_profile_freezes_schema_two_with_source_kinds(tmp_path: Path) -> None:
    root, profile_path = _repository(tmp_path)
    run_root = root / "work/v1/calibration-runtime/run-a"

    manifest = freeze_run_inputs(
        root,
        PROFILE_ID,
        run_root,
        profile_file=profile_path,
    )

    assert manifest["schema_version"] == "2.0"
    assert manifest["profile_source"] == next(
        {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "kind": entry["kind"],
        }
        for entry in manifest["files"]
        if entry["path"] == "work/materialized/profile.json"
    )
    assert manifest["profile_source"]["kind"] == "runtime-derived"
    entries = {entry["path"]: entry for entry in manifest["files"]}
    assert all(entries[relative]["kind"] == "runtime-derived" for relative in RUNTIME_MEMBERS)
    assert entries["experiments/v1/simulations/fixture/scenario.json"]["kind"] == "versioned"
    assert entries["experiments/v1/simulations/fixture/runs.json"]["kind"] == "versioned"
    assert {entry["kind"] for entry in manifest["files"]} == {
        "runtime-derived",
        "versioned",
    }


def test_runtime_profile_must_be_below_work(tmp_path: Path) -> None:
    root, profile_path = _repository(tmp_path)
    outside = root / "runtime-profile.json"
    outside.write_bytes(profile_path.read_bytes())

    with pytest.raises(BenchmarkConfigError, match=r"stored below work/"):
        freeze_run_inputs(
            root,
            PROFILE_ID,
            root / "work/v1/calibration-runtime/outside",
            profile_file=outside,
        )


def test_runtime_profile_symlink_is_rejected(tmp_path: Path) -> None:
    root, profile_path = _repository(tmp_path)
    alias = root / "work/alias-profile.json"
    alias.symlink_to(profile_path)

    with pytest.raises(BenchmarkConfigError, match="symlink"):
        freeze_run_inputs(
            root,
            PROFILE_ID,
            root / "work/v1/calibration-runtime/symlink",
            profile_file=alias,
        )


def test_untracked_scientific_input_is_not_promoted_to_runtime_derived(
    tmp_path: Path,
) -> None:
    root, profile_path = _repository(tmp_path, scientific_runtime_input=True)

    with pytest.raises(
        BenchmarkConfigError,
        match=r"not tracked canonically: work/scientific-network.tsv",
    ):
        freeze_run_inputs(
            root,
            PROFILE_ID,
            root / "work/v1/calibration-runtime/scientific-input",
            profile_file=profile_path,
        )


def test_resume_uses_snapshot_after_dynamic_sources_are_removed(tmp_path: Path) -> None:
    root, profile_path = _repository(tmp_path)
    run_root = root / "work/v1/calibration-runtime/resume"
    first = freeze_run_inputs(
        root,
        PROFILE_ID,
        run_root,
        profile_file=profile_path,
    )
    frozen_description = frozen_profile(run_root, PROFILE_ID)["description"]

    profile_path.unlink()
    for relative in RUNTIME_MEMBERS:
        (root / relative).unlink()

    resumed = freeze_run_inputs(
        root,
        PROFILE_ID,
        run_root,
        profile_file=profile_path,
    )

    assert resumed == first
    assert validate_run_inputs(root, PROFILE_ID, run_root) == first
    assert frozen_profile(run_root, PROFILE_ID)["description"] == frozen_description


def test_resume_rejects_a_different_runtime_profile_source(tmp_path: Path) -> None:
    root, profile_path = _repository(tmp_path)
    run_root = root / "work/v1/calibration-runtime/resume-mismatch"
    freeze_run_inputs(root, PROFILE_ID, run_root, profile_file=profile_path)
    other_profile = profile_path.with_name("other-profile.json")
    other_profile.write_bytes(profile_path.read_bytes())

    with pytest.raises(BenchmarkConfigError, match="different profile source"):
        freeze_run_inputs(
            root,
            PROFILE_ID,
            run_root,
            profile_file=other_profile,
        )
