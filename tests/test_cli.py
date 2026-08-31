from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from spathi_bench.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_calibration_plan_and_rank_are_cli_executable(tmp_path: Path) -> None:
    plan = tmp_path / "convergence-plan"
    assert (
        main(
            [
                "calibration-plan",
                "--repo-root",
                str(REPO_ROOT),
                "--stage",
                "convergence",
                "--output-dir",
                str(plan),
            ]
        )
        == 0
    )
    manifest = json.loads((plan / "calibration-manifest.json").read_text(encoding="utf-8"))
    schema = json.loads(
        (REPO_ROOT / "schemas/calibration-manifest.schema.json").read_text(encoding="utf-8")
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
    summary.write_text(json.dumps({"calibration_rows": summary_rows}), encoding="utf-8")
    ranking = tmp_path / "ranking.json"
    assert (
        main(
            [
                "calibration-rank",
                "--repo-root",
                str(REPO_ROOT),
                "--stage",
                "convergence",
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
