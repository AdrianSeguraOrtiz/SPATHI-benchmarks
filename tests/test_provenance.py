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


def _add_production_policy(
    root: Path,
    profile: dict[str, Any],
    *,
    safety: bool,
) -> tuple[str, set[str]]:
    production_path = "work/efficiency/production-config.json"
    artifact_paths: set[str] = set()

    def artifact(name: str) -> dict[str, str]:
        relative = f"work/efficiency/{name}.json"
        _write_json(root, relative, {"artifact": name})
        artifact_paths.add(relative)
        return {"path": relative, "sha256": "9" * 64}

    provenance: dict[str, Any] = {
        name: artifact(name)
        for name in (
            "calibration_selection",
            "efficiency_plan",
            "efficiency_evidence",
            "efficiency_results",
        )
    }
    if safety:
        provenance["efficiency_safety"] = {
            name: artifact(f"safety-{name}") for name in ("panel", "plan", "evidence", "results")
        }
    _write_json(root, production_path, {"provenance": provenance})
    profile["production_config"] = production_path
    return production_path, artifact_paths


def test_snapshot_includes_every_nested_safety_provenance_artifact(tmp_path: Path) -> None:
    root, profile = _tracked_repository(tmp_path)
    production_path, artifact_paths = _add_production_policy(root, profile, safety=True)

    sources = {
        source.path.relative_to(root).as_posix()
        for source in _snapshot_source_records(root, profile)
    }

    assert production_path in sources
    assert artifact_paths <= sources
    assert len(artifact_paths) == 8


@pytest.mark.parametrize(
    "malformed_safety",
    (
        {"panel": {"path": "work/panel.json", "sha256": "9" * 64}},
        {
            name: {"path": f"work/{name}.json", "sha256": "not-a-digest"}
            for name in ("panel", "plan", "evidence", "results")
        },
    ),
)
def test_snapshot_rejects_malformed_nested_safety_provenance(
    tmp_path: Path,
    malformed_safety: dict[str, Any],
) -> None:
    root, profile = _tracked_repository(tmp_path)
    production_path, _artifact_paths = _add_production_policy(root, profile, safety=False)
    production = json.loads((root / production_path).read_text(encoding="utf-8"))
    production["provenance"]["efficiency_safety"] = malformed_safety
    _write_json(root, production_path, production)

    with pytest.raises(BenchmarkConfigError, match="provenance is invalid"):
        _snapshot_source_records(root, profile)
