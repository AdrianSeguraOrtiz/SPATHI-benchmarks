from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.selected_policy as selection
from selection_helpers import base_policy, selected_document
from spathi_bench.artifacts import sha256_file
from spathi_bench.config import BenchmarkConfigError


def _completed(root: Path, budget: str, flow_id: str) -> Path:
    directory = root / "work/v1/workflows/calibration" / budget / flow_id
    directory.mkdir(parents=True)
    path = directory / "selected-config.json"
    path.write_text(json.dumps(selected_document(base_policy(), budget=budget)))
    state = {
        "workflow_type": "calibration",
        "calibration_budget": budget,
        "status": "complete",
        "selected_config": path.relative_to(root).as_posix(),
        "selected_config_sha256": sha256_file(path),
    }
    (directory / "workflow.json").write_text(json.dumps(state))
    return path


def test_explicit_calibration_identity_selects_only_its_own_budget_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _completed(tmp_path, "reduced", "owning-study")
    _completed(tmp_path, "reduced", "other-study")
    _completed(tmp_path, "full", "owning-study")
    verified = []

    def verify(root: Path, path: Path, *, budget: str) -> tuple[dict[str, Any], tuple[int, ...]]:
        verified.append((root, path, budget))
        return json.loads(path.read_text())["selected_policy"], (123,)

    monkeypatch.setattr(selection, "load_frozen_selected_policy", verify)
    path, document = selection.completed_selected_config(
        repo_root=tmp_path,
        work_root=tmp_path / "work",
        budget="reduced",
        calibration_run_id="owning-study",
    )

    assert path == expected
    assert document["calibration_budget"] == "reduced"
    assert verified == [(tmp_path, expected, "reduced")]


def test_ambiguous_discovery_does_not_choose_a_calibration_by_timestamp(tmp_path: Path) -> None:
    _completed(tmp_path, "reduced", "first")
    _completed(tmp_path, "reduced", "second")
    with pytest.raises(BenchmarkConfigError, match="resume the owning study"):
        selection.completed_selected_config(
            repo_root=tmp_path,
            work_root=tmp_path / "work",
            budget="reduced",
            calibration_run_id=None,
        )


def test_completed_calibration_rejects_changed_selected_bytes(tmp_path: Path) -> None:
    path = _completed(tmp_path, "reduced", "study")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(BenchmarkConfigError, match="missing or changed"):
        selection.completed_selected_config(
            repo_root=tmp_path,
            work_root=tmp_path / "work",
            budget="reduced",
            calibration_run_id="study",
        )
