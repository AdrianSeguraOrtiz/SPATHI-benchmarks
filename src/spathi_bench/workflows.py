"""High-level calibration and comparison workflows.

The public workflows keep adaptive artifacts below ``work/`` and rely on the
normal pipeline snapshot to freeze them by content.  Git remains responsible
only for source inputs; no intermediate commit is required.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .benchmark_materialization import materialize_benchmark_plan
from .calibration import stage_profile_id
from .calibration_materialization import (
    freeze_calibration_selection,
    materialize_calibration_stage,
    rank_calibration_stage,
)
from .config import (
    BenchmarkConfigError,
    load_json_object,
    load_repo_lock,
    validate_calibration_contracts,
)
from .pipeline import (
    build_spathi_image,
    evaluate,
    export_analysis,
    generate_data,
    integrate_inference,
    preflight,
    run_inference,
    validate_handoffs,
)
from .provenance import observe_benchmark_repository
from .state import run_lock
from .statistics import analyze_statistics
from .summarize import summarize

CALIBRATION_STAGES = ("convergence", "structure", "estimator", "validation")
CALIBRATION_BUDGETS = ("reduced", "full")
_WORKFLOW_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _workflow_identity(repo_root: Path) -> dict[str, str]:
    root = repo_root.resolve()
    benchmark = observe_benchmark_repository(root, require_clean=True)
    repository_lock = load_repo_lock(root)
    search_path = root / "experiments/v1/spathi/search-space.json"
    identity = {
        "benchmark_commit": str(benchmark["commit"]),
        "benchmark_origin": str(benchmark["origin"]),
        "spathi_commit": str(repository_lock["repositories"]["spathi"]["commit"]),
        "andrea_commit": str(repository_lock["repositories"]["andrea"]["commit"]),
        "search_space_sha256": sha256_file(search_path),
    }
    identity["fingerprint"] = canonical_json_sha256(identity)
    return identity


def _flow_id(identity: dict[str, str], requested: str | None) -> str:
    if requested is None:
        return f"spathi-{identity['spathi_commit'][:12]}-{identity['fingerprint'][:10]}"
    if not _WORKFLOW_ID.fullmatch(requested):
        raise BenchmarkConfigError(f"Invalid workflow run ID: {requested!r}")
    return requested


def _calibration_identity(identity: dict[str, str], calibration_budget: str) -> dict[str, str]:
    """Bind a calibration identity to its evidence budget."""

    scoped = {key: value for key, value in identity.items() if key != "fingerprint"}
    scoped["calibration_budget"] = calibration_budget
    scoped["fingerprint"] = canonical_json_sha256(scoped)
    return scoped


def _adopt_workflow(
    path: Path,
    *,
    workflow_type: str,
    flow_id: str,
    identity: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = {
        "schema_version": "1.0",
        "workflow_type": workflow_type,
        "flow_id": flow_id,
        "identity": identity,
        **(extra or {}),
    }
    if not path.exists():
        return {**expected, "status": "running", "stages": {}}
    observed = load_json_object(path, f"{workflow_type} workflow state")
    for key, value in expected.items():
        if observed.get(key) != value:
            raise BenchmarkConfigError(
                f"Existing {workflow_type} workflow {flow_id!r} has incompatible {key}"
            )
    if observed.get("status") not in {"running", "complete"}:
        raise BenchmarkConfigError(f"Existing {workflow_type} workflow has invalid status")
    if not isinstance(observed.get("stages"), dict):
        raise BenchmarkConfigError(f"Existing {workflow_type} workflow has invalid stages")
    return observed


def run_profile_pipeline(
    *,
    repo_root: Path,
    profile_id: str,
    profile_file: Path,
    run_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    statistics: bool,
) -> dict[str, Any]:
    """Run or resume every official ANDREA-compatible stage for one profile."""

    common = {
        "repo_root": repo_root,
        "profile_id": profile_id,
        "run_root": run_root,
    }
    with run_lock(run_root):
        preflight(
            **common,
            andrea_repo=andrea_repo,
            spathi_repo=spathi_repo,
            profile_file=profile_file,
        )
        build_spathi_image(**common, spathi_repo=spathi_repo)
        generate_data(**common, andrea_repo=andrea_repo)
        run_inference(**common, andrea_repo=andrea_repo)
        validate_handoffs(**common)
        integrate_inference(**common, andrea_repo=andrea_repo)
        evaluate(**common, andrea_repo=andrea_repo)
        export_analysis(**common, andrea_repo=andrea_repo)
        report_paths = summarize(**common)
        statistics_paths = analyze_statistics(**common) if statistics else ()
    return {
        "run_root": str(run_root.resolve()),
        "summary": str((run_root / "reports/summary.json").resolve()),
        "reports": [str(path.resolve()) for path in report_paths],
        "statistics": [str(path.resolve()) for path in statistics_paths],
    }


def run_calibration_workflow(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    calibration_budget: str = "full",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run or resume the complete four-stage adaptive SPATHI calibration."""

    if calibration_budget not in CALIBRATION_BUDGETS:
        raise BenchmarkConfigError("Calibration profile must be 'reduced' or 'full'")
    root = repo_root.resolve()
    work = work_root.resolve()
    if not work.is_relative_to(root / "work"):
        raise BenchmarkConfigError("High-level workflow work root must be below repository work/")
    identity = _calibration_identity(_workflow_identity(root), calibration_budget)
    flow_id = _flow_id(identity, run_id)
    workflow_root = work / "v1/workflows/calibration" / calibration_budget / flow_id
    state_path = workflow_root / "workflow.json"
    with run_lock(workflow_root):
        state = _adopt_workflow(
            state_path,
            workflow_type="calibration",
            flow_id=flow_id,
            identity=identity,
            extra={"calibration_budget": calibration_budget},
        )
        _write_json(state_path, state)
        parent_ranking: Path | None = None
        for stage_id in CALIBRATION_STAGES:
            profile_id = stage_profile_id(stage_id, calibration_budget)
            stage_root = workflow_root / "stages" / stage_id
            bundle = stage_root / "requests"
            profile_file = stage_root / f"{profile_id}.json"
            ranking = stage_root / "ranking.json"
            run_root = work / "v1" / profile_id / f"{flow_id}-{stage_id}"
            materialize_calibration_stage(
                repo_root=root,
                stage_id=stage_id,
                calibration_budget=calibration_budget,
                parent_ranking_path=parent_ranking,
                output_dir=bundle,
                profile_output=profile_file,
            )
            pipeline = run_profile_pipeline(
                repo_root=root,
                profile_id=profile_id,
                profile_file=profile_file,
                run_root=run_root,
                andrea_repo=andrea_repo,
                spathi_repo=spathi_repo,
                statistics=False,
            )
            rank_calibration_stage(
                repo_root=root,
                stage_id=stage_id,
                calibration_budget=calibration_budget,
                manifest_path=bundle / "calibration-manifest.json",
                summary_path=run_root / "reports/summary.json",
                output_path=ranking,
            )
            state["stages"][stage_id] = {
                "profile_id": profile_id,
                "profile": str(profile_file.relative_to(root)),
                "run_root": str(run_root.relative_to(root)),
                "ranking": str(ranking.relative_to(root)),
                "ranking_sha256": sha256_file(ranking),
                "status": "ranked",
                "pipeline": pipeline,
            }
            _write_json(state_path, state)
            parent_ranking = ranking

        if parent_ranking is None:
            raise AssertionError("Calibration workflow produced no validation ranking")
        selected_config = workflow_root / "selected-config.json"
        freeze_calibration_selection(
            repo_root=root,
            ranking_path=parent_ranking,
            output_path=selected_config,
        )
        state.update(
            {
                "status": "complete",
                "selected_config": str(selected_config.relative_to(root)),
                "selected_config_sha256": sha256_file(selected_config),
            }
        )
        _write_json(state_path, state)
    return state


def _calibration_selection(
    *,
    repo_root: Path,
    work_root: Path,
    identity: dict[str, str],
    calibration_budget: str,
    calibration_run_id: str | None,
) -> tuple[str, Path]:
    calibration_identity = _calibration_identity(identity, calibration_budget)
    calibration_id = _flow_id(calibration_identity, calibration_run_id)
    workflow_root = work_root / "v1/workflows/calibration" / calibration_budget / calibration_id
    state_path = workflow_root / "workflow.json"
    if not state_path.is_file():
        raise BenchmarkConfigError(
            f"No completed {calibration_budget} calibration matches the current SPATHI "
            f"and benchmark inputs; run scripts/calibrate.sh --profile {calibration_budget} "
            "first"
        )
    state = load_json_object(state_path, "calibration workflow state")
    if (
        state.get("status") != "complete"
        or state.get("identity") != calibration_identity
        or state.get("calibration_budget") != calibration_budget
    ):
        raise BenchmarkConfigError(
            f"No completed {calibration_budget} calibration matches the current SPATHI "
            f"and benchmark inputs; run scripts/calibrate.sh --profile {calibration_budget} "
            "first"
        )
    raw_selected = state.get("selected_config")
    if not isinstance(raw_selected, str):
        raise BenchmarkConfigError("Completed calibration has no selected configuration")
    selected = (repo_root / raw_selected).resolve()
    if not selected.is_relative_to(workflow_root.resolve()) or not selected.is_file():
        raise BenchmarkConfigError("Calibration selected configuration escapes its workflow")
    if sha256_file(selected) != state.get("selected_config_sha256"):
        raise BenchmarkConfigError("Calibration selected configuration changed after freezing")
    _search, selection = validate_calibration_contracts(repo_root, selected)
    if selection.get("calibration_budget") != calibration_budget:
        raise BenchmarkConfigError(
            f"Calibration {calibration_id!r} uses budget "
            f"{selection.get('calibration_budget')!r}, not {calibration_budget!r}"
        )
    return calibration_id, selected


def run_comparison_workflow(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    profile: str = "full",
    calibration_run_id: str | None = None,
) -> dict[str, Any]:
    """Run or resume one selected-SPATHI comparison with all declared baselines."""

    if profile not in {"reduced", "full"}:
        raise BenchmarkConfigError("Comparison profile must be 'reduced' or 'full'")
    root = repo_root.resolve()
    work = work_root.resolve()
    if not work.is_relative_to(root / "work"):
        raise BenchmarkConfigError("High-level workflow work root must be below repository work/")
    identity = _workflow_identity(root)
    calibration_id, selected_config = _calibration_selection(
        repo_root=root,
        work_root=work,
        identity=identity,
        calibration_budget=profile,
        calibration_run_id=calibration_run_id,
    )
    flow_id = f"{profile}-{calibration_id}"
    workflow_root = work / "v1/workflows/comparison" / profile / flow_id
    state_path = workflow_root / "workflow.json"
    profile_id = f"{profile}-v1"
    with run_lock(workflow_root):
        state = _adopt_workflow(
            state_path,
            workflow_type="comparison",
            flow_id=flow_id,
            identity=identity,
            extra={
                "profile": profile,
                "calibration_budget": profile,
                "calibration_flow_id": calibration_id,
            },
        )
        _write_json(state_path, state)
        bundle = workflow_root / "requests"
        profile_file = workflow_root / f"{profile_id}.json"
        materialize_benchmark_plan(
            repo_root=root,
            template_profile_id=profile,
            baseline_plan_path=root / f"experiments/v1/inference/plans/{profile}-v1.json",
            output_dir=bundle,
            profile_output=profile_file,
            selected_config_path=selected_config,
        )
        run_root = work / "v1" / profile_id / flow_id
        pipeline = run_profile_pipeline(
            repo_root=root,
            profile_id=profile_id,
            profile_file=profile_file,
            run_root=run_root,
            andrea_repo=andrea_repo,
            spathi_repo=spathi_repo,
            statistics=True,
        )
        state.update(
            {
                "status": "complete",
                "stages": {"comparison": {"status": "complete", "pipeline": pipeline}},
                "profile_file": str(profile_file.relative_to(root)),
                "profile_sha256": sha256_file(profile_file),
            }
        )
        _write_json(state_path, state)
    return state
