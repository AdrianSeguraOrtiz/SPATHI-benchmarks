from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.provenance import _snapshot_source_records


def _write_json(root: Path, relative: str, payload: dict[str, Any]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tracked_repository(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)

    build_input = "containers/spathi/Dockerfile"
    (root / build_input).parent.mkdir(parents=True)
    (root / build_input).write_text("FROM scratch\n", encoding="utf-8")
    _write_json(root, "repos.lock.json", {})
    _write_json(
        root,
        "images.lock.json",
        {
            "schema_version": "1.0",
            "catalog_images": {},
            "images": {
                "spathi": {
                    "tag": "spathi:test",
                    "base_image": "python:test",
                    "source_repository": "spathi",
                    "source_commit": "1" * 40,
                    "build_inputs": {build_input: "2" * 64},
                    "python_dependencies": {"spathi": "0.1.0"},
                }
            },
        },
    )
    _write_json(root, "schemas/profile.json", {})
    _write_json(root, "schemas/execution-policy.schema.json", {})
    _write_json(root, "experiments/v1/execution/policy.json", {})
    _write_json(root, "experiments/v1/seeds/seed-plan.json", {"schema_version": "1.0"})
    _write_json(root, "experiments/v1/spathi/search-space.json", {})
    topology = root / "experiments/v1/simulations/input/network.tsv"
    topology.parent.mkdir(parents=True)
    topology.write_text("target\tregulator\teffect\nG2\tG1\t1\n", encoding="utf-8")
    _write_json(
        root,
        "experiments/v1/simulations/input/scenario.json",
        {
            "inputs": {
                "regulatory_network": {
                    "path": "network.tsv",
                }
            }
        },
    )
    _write_json(root, "experiments/v1/simulations/input/runs.json", {"runs": []})
    profile = {
        "id": "calibration-convergence",
        "generation_jobs": [
            {
                "id": "input-panel",
                "scenario": "experiments/v1/simulations/input/scenario.json",
                "simulator_runs": "experiments/v1/simulations/input/runs.json",
            }
        ],
        "inference_batches": [],
    }
    _write_json(
        root,
        "experiments/v1/profiles/calibration-convergence.json",
        profile,
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    return root, profile


def test_snapshot_includes_external_inputs_and_seed_contract(tmp_path: Path) -> None:
    root, profile = _tracked_repository(tmp_path)

    sources = {
        source.path.relative_to(root).as_posix()
        for source in _snapshot_source_records(root, profile)
    }

    assert "experiments/v1/simulations/input/network.tsv" in sources
    assert "experiments/v1/seeds/seed-plan.json" in sources


def test_snapshot_rejects_symlinked_scenario_input(tmp_path: Path) -> None:
    root, profile = _tracked_repository(tmp_path)
    topology = root / "experiments/v1/simulations/input/network.tsv"
    topology.unlink()
    replacement = root / "experiments/v1/simulations/input/network-target.tsv"
    replacement.write_text("target\tregulator\teffect\nG2\tG1\t1\n", encoding="utf-8")
    topology.symlink_to(replacement)

    with pytest.raises(BenchmarkConfigError, match="symlink"):
        _snapshot_source_records(root, profile)


def test_snapshot_includes_scale_stress_generator_request(tmp_path: Path) -> None:
    root, profile = _tracked_repository(tmp_path)
    request = "experiments/v1/simulations/scale_stress/scale-reduced.json"
    _write_json(
        root,
        request,
        {
            "scale_grid": {
                "provenance": "author_defined_public_synthetic_grid",
                "cells": [1000],
                "genes": [5000],
                "transcription_factors": [250],
            }
        },
    )
    profile["generation_jobs"] = [
        {
            "id": "scale-stress-panel",
            "generator_request": request,
        }
    ]
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)

    sources = {
        source.path.relative_to(root).as_posix()
        for source in _snapshot_source_records(root, profile)
    }

    assert request in sources


def test_snapshot_includes_runtime_convergence_plan(tmp_path: Path) -> None:
    root, profile = _tracked_repository(tmp_path)
    paths = {
        "tools_params": "work/calibration/tools-params.json",
        "run_metadata": "work/calibration/run-metadata.json",
        "custom_tools": "work/calibration/custom-tools.json",
        "convergence_plans": "work/calibration/convergence-plans.json",
    }
    for relative in paths.values():
        _write_json(root, relative, {})
    profile["inference_batches"] = [
        {
            "id": "convergence",
            "kind": "spathi",
            **paths,
        }
    ]

    records = _snapshot_source_records(root, profile)
    by_relative = {record.relative: record.kind for record in records}

    assert by_relative[paths["convergence_plans"]] == "runtime-derived"


def test_snapshot_freezes_selected_configuration_and_entire_ranking_chain(tmp_path: Path) -> None:
    root, profile = _tracked_repository(tmp_path)
    selection_path = "work/calibration/selected-config.json"
    validation_path = "work/calibration/validation-ranking.json"
    development_path = "work/calibration/development-ranking.json"
    _write_json(root, development_path, {"parent_ranking": None})
    _write_json(root, validation_path, {"parent_ranking": {"path": development_path}})
    _write_json(
        root,
        selection_path,
        {
            "selection_evidence": {"validation_ranking": validation_path},
        },
    )
    profile["selected_config"] = selection_path

    records = _snapshot_source_records(root, profile)
    by_relative = {record.relative: record.kind for record in records}

    assert {
        path: by_relative[path]
        for path in (
            selection_path,
            validation_path,
            development_path,
        )
    } == {
        path: "runtime-derived"
        for path in (
            selection_path,
            validation_path,
            development_path,
        )
    }
