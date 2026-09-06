"""Resolution of immutable production policies emitted by efficiency workflows."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .benchmark_materialization import load_frozen_production_policy
from .config import BenchmarkConfigError, load_json_object, validate_json

_WORKFLOW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def completed_production_config(
    *,
    repo_root: Path,
    work_root: Path,
    budget: str,
    efficiency_run_id: str | None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve exactly one verified policy from a completed efficiency workflow."""

    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError(f"Invalid production-policy budget: {budget!r}")
    if efficiency_run_id is not None and _WORKFLOW_ID.fullmatch(efficiency_run_id) is None:
        raise BenchmarkConfigError(f"Invalid efficiency run ID: {efficiency_run_id!r}")
    base = work_root / "v1/workflows/efficiency" / budget
    candidates = (
        [base / efficiency_run_id]
        if efficiency_run_id is not None
        else sorted(path.parent for path in base.glob("*/workflow.json"))
    )
    completed: list[tuple[Path, dict[str, Any]]] = []
    for candidate in candidates:
        if candidate.is_symlink():
            raise BenchmarkConfigError("Efficiency workflow directory must not be a symlink")
        workflow_path = candidate / "workflow.json"
        if workflow_path.is_symlink():
            raise BenchmarkConfigError("Efficiency workflow state must not be a symlink")
        if not workflow_path.is_file():
            continue
        workflow = load_json_object(workflow_path, "efficiency workflow")
        raw_production = workflow.get("production_config")
        if (
            workflow.get("workflow_type") != "efficiency"
            or workflow.get("profile") != budget
            or workflow.get("status") != "complete"
            or not isinstance(raw_production, str)
        ):
            continue
        production_reference = repo_root / raw_production
        expected_reference = candidate / "production-config.json"
        production_path = production_reference.resolve()
        if (
            Path(os.path.abspath(production_reference)) != Path(os.path.abspath(expected_reference))
            or production_reference.is_symlink()
            or production_path.parent != candidate.resolve()
            or not production_path.is_file()
            or sha256_file(production_path) != workflow.get("production_config_sha256")
        ):
            raise BenchmarkConfigError(
                "Completed efficiency production policy is missing or changed"
            )
        production = load_json_object(production_path, "SPATHI production policy")
        validate_json(
            production,
            repo_root / "schemas/spathi-production-config.schema.json",
            "SPATHI production policy",
        )
        if production["budget"] != budget:
            raise BenchmarkConfigError("Efficiency production policy uses a different budget")
        completed.append((production_path, production))
    if len(completed) != 1:
        if not completed:
            raise BenchmarkConfigError(
                f"No completed {budget} efficiency study is available; "
                f"run scripts/justify-efficiency.sh --profile {budget} first"
            )
        raise BenchmarkConfigError(
            "Several completed efficiency studies are available; pass --efficiency-run-id"
        )
    production_path, production = completed[0]
    verified_policy, _comparison_seeds = load_frozen_production_policy(
        repo_root,
        production_path,
        budget=budget,
    )
    if verified_policy != production["policy"]:
        raise BenchmarkConfigError("Efficiency production policy verification changed its content")
    return production_path, production


__all__ = ["completed_production_config"]
