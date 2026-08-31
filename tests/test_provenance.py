from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from spathi_bench.provenance import _snapshot_sources


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
    _write_json(root, "experiments/v1/spathi/search-space.json", {})
    _write_json(
        root,
        "experiments/v1/spathi/selected-config.json",
        {
            "selection_evidence": {
                "validation_ranking": "experiments/v1/spathi/evidence/validation.json"
            }
        },
    )
    _write_json(
        root,
        "experiments/v1/spathi/evidence/validation.json",
        {
            "parent_ranking": {
                "path": "experiments/v1/spathi/evidence/estimator.json",
                "sha256": "3" * 64,
                "stage_id": "estimator",
            }
        },
    )
    _write_json(
        root,
        "experiments/v1/spathi/evidence/estimator.json",
        {"parent_ranking": None},
    )
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


def test_snapshot_includes_external_inputs_and_complete_ranking_chain(tmp_path: Path) -> None:
    root, profile = _tracked_repository(tmp_path)

    sources = {path.relative_to(root).as_posix() for path in _snapshot_sources(root, profile)}

    assert "experiments/v1/simulations/input/network.tsv" in sources
    assert "experiments/v1/spathi/evidence/validation.json" in sources
    assert "experiments/v1/spathi/evidence/estimator.json" in sources
