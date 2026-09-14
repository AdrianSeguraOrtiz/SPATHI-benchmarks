"""Resolve the frozen calibration winner consumed by every subsequent phase."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .benchmark_materialization import load_frozen_selected_policy
from .config import BenchmarkConfigError, load_json_object
from .safe_io import path_below_no_symlinks

_WORKFLOW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def completed_selected_config(
    *,
    repo_root: Path,
    work_root: Path,
    budget: str,
    calibration_run_id: str | None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a completed calibration and verify its immutable selection evidence."""
    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError(f"Invalid calibration budget: {budget!r}")
    if calibration_run_id is not None and _WORKFLOW_ID.fullmatch(calibration_run_id) is None:
        raise BenchmarkConfigError(f"Invalid calibration run ID: {calibration_run_id!r}")
    base = work_root / "v1/workflows/calibration" / budget
    candidates = (
        [base / calibration_run_id]
        if calibration_run_id is not None
        else sorted(path.parent for path in base.glob("*/workflow.json"))
    )
    completed: list[tuple[Path, dict[str, Any]]] = []
    for candidate in candidates:
        state_path = path_below_no_symlinks(
            candidate / "workflow.json", work_root, "Calibration workflow state"
        )
        if not state_path.is_file():
            continue
        workflow = load_json_object(state_path, "calibration workflow")
        if (
            workflow.get("workflow_type") != "calibration"
            or workflow.get("calibration_budget") != budget
            or workflow.get("status") != "complete"
        ):
            continue
        raw_selection = workflow.get("selected_config")
        if not isinstance(raw_selection, str):
            raise BenchmarkConfigError("Completed calibration has no selected configuration")
        selected_path = path_below_no_symlinks(
            repo_root / raw_selection, candidate, "Frozen calibration selection", require_file=True
        )
        if selected_path != (candidate / "selected-config.json").resolve() or sha256_file(
            selected_path
        ) != workflow.get("selected_config_sha256"):
            raise BenchmarkConfigError("Completed calibration selection is missing or changed")
        selected = load_json_object(selected_path, "frozen calibration selection")
        if selected.get("calibration_budget") != budget:
            raise BenchmarkConfigError("Completed calibration selection uses a different budget")
        completed.append((selected_path, selected))
    if len(completed) != 1:
        if not completed:
            raise BenchmarkConfigError(
                f"No completed {budget} calibration is available; run spathi-bench run {budget}"
            )
        raise BenchmarkConfigError(
            "Several completed calibrations are available; resume the owning study with "
            "spathi-bench run --id STUDY_ID"
        )
    selected_path, selected = completed[0]
    policy, _seeds = load_frozen_selected_policy(repo_root, selected_path, budget=budget)
    if policy != selected["selected_policy"]:
        raise BenchmarkConfigError("Calibration policy verification changed its content")
    return selected_path, selected


__all__ = ["completed_selected_config"]
