"""High-level calibration and comparison workflows.

The public workflows keep staged-search artifacts below ``work/`` and rely on the
normal pipeline snapshot to freeze them by content.  Git remains responsible
only for source inputs; no intermediate commit is required.
"""

from __future__ import annotations

import re
from collections.abc import Callable
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
from .production_policy import completed_production_config
from .provenance import observe_benchmark_repository
from .report_publication import publish_workflow_report
from .reporting import render_calibration_report, render_comparison_report
from .reporting.common import central_report_path
from .runtime import (
    andrea_host_environment,
    benchmark_host_environment,
    docker_runtime_environment,
)
from .safe_io import atomic_write_json, path_below_no_symlinks
from .state import run_lock
from .statistics import analyze_statistics
from .summarize import summarize

CALIBRATION_STAGES = (
    "geometry",
    "estimator",
    "interaction",
    "estimator_scale_transfer",
    "convergence",
    "validation",
)
CALIBRATION_BUDGETS = ("reduced", "full")
_WORKFLOW_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _workflow_identity(repo_root: Path) -> dict[str, str]:
    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    benchmark = observe_benchmark_repository(root, require_clean=True)
    repository_lock = load_repo_lock(root)
    search_path = root / "experiments/v1/spathi/search-space.json"
    identity = {
        "benchmark_commit": str(benchmark["commit"]),
        "benchmark_origin": str(benchmark["origin"]),
        "spathi_commit": str(repository_lock["repositories"]["spathi"]["commit"]),
        "andrea_commit": str(repository_lock["repositories"]["andrea"]["commit"]),
        "search_space_sha256": sha256_file(search_path),
        "andrea_environment": str(andrea_host_environment()["fingerprint"]),
        "benchmark_environment": str(benchmark_host_environment()["fingerprint"]),
        "docker_runtime_environment": str(docker_runtime_environment()["fingerprint"]),
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


def _workflow_work_root(repo_root: Path, work_root: Path) -> tuple[Path, Path]:
    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    work = path_below_no_symlinks(
        work_root,
        root / "work",
        "High-level workflow work root",
    )
    return root, work


def _reportable_workflow_root(
    *,
    base: Path,
    workflow_type: str,
    requested_id: str | None,
    required_fields: dict[str, Any] | None = None,
) -> Path:
    if requested_id is not None:
        if not _WORKFLOW_ID.fullmatch(requested_id):
            raise BenchmarkConfigError(f"Invalid workflow run ID: {requested_id!r}")
        candidate = base / requested_id
        candidates = [candidate]
    else:
        candidates = sorted(
            (path.parent for path in base.glob("*/workflow.json")),
            key=lambda path: path.name,
        )
    reportable: list[Path] = []
    for candidate in candidates:
        state_path = candidate / "workflow.json"
        if not state_path.is_file():
            continue
        safe_state_path = path_below_no_symlinks(
            state_path,
            base,
            f"{workflow_type} workflow state",
            require_file=True,
        )
        state = load_json_object(safe_state_path, f"{workflow_type} workflow state")
        if (
            state.get("workflow_type") == workflow_type
            and state.get("status") in {"publishing", "complete"}
            and all(state.get(key) == value for key, value in (required_fields or {}).items())
        ):
            reportable.append(safe_state_path.parent)
    if requested_id is not None and not reportable:
        raise BenchmarkConfigError(
            f"No reportable {workflow_type} workflow named {requested_id!r} exists below {base}"
        )
    if not reportable:
        raise BenchmarkConfigError(f"No reportable {workflow_type} workflow exists below {base}")
    if len(reportable) > 1:
        option = "--run-id" if workflow_type == "calibration" else "--efficiency-run-id"
        identifiers = ", ".join(path.name for path in reportable)
        raise BenchmarkConfigError(
            f"Multiple reportable {workflow_type} workflows exist ({identifiers}); select one with "
            f"{option}"
        )
    return reportable[0]


def _report_paths(
    *, version_root: Path, workflow_type: str, scope: str, flow_id: str
) -> tuple[Path, Path]:
    html_path = central_report_path(
        version_root=version_root,
        report_type=workflow_type,
        scope=scope,
        flow_id=flow_id,
    )
    return html_path, html_path.parent / f"{workflow_type}-report.manifest.json"


def _publishing_state(state: dict[str, Any], *, workflow_type: str) -> dict[str, Any]:
    if state.get("status") not in {"publishing", "complete"}:
        raise BenchmarkConfigError(
            f"{workflow_type.title()} report publication requires frozen workflow outputs"
        )
    publishing = dict(state)
    publishing["status"] = "publishing"
    publishing.pop("reports", None)
    return publishing


def _publish_report(
    *,
    repo_root: Path,
    version_root: Path,
    workflow_root: Path,
    state: dict[str, Any],
    workflow_type: str,
    scope: str,
    render: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    html_path, manifest_path = _report_paths(
        version_root=version_root,
        workflow_type=workflow_type,
        scope=scope,
        flow_id=str(state.get("flow_id", "")),
    )
    expected_report = html_path.relative_to(repo_root.resolve()).as_posix()
    if state.get("interactive_report") != expected_report:
        raise BenchmarkConfigError(
            f"{workflow_type.title()} workflow records a non-canonical report path"
        )
    return publish_workflow_report(
        repo_root=repo_root,
        workflow_path=workflow_root / "workflow.json",
        publishing_state=_publishing_state(state, workflow_type=workflow_type),
        html_path=html_path,
        manifest_path=manifest_path,
        render=render,
        label=workflow_type,
    )


def rebuild_calibration_report(
    *,
    repo_root: Path,
    work_root: Path,
    calibration_budget: str,
    run_id: str | None,
) -> dict[str, Any]:
    """Rebuild only the presentation layer of one completed calibration."""

    if calibration_budget not in CALIBRATION_BUDGETS:
        raise BenchmarkConfigError("Calibration profile must be 'reduced' or 'full'")
    root, work = _workflow_work_root(repo_root, work_root)
    workflow_root = _reportable_workflow_root(
        base=work / "v1/workflows/calibration" / calibration_budget,
        workflow_type="calibration",
        requested_id=run_id,
    )
    with run_lock(workflow_root):
        state = load_json_object(workflow_root / "workflow.json", "calibration workflow state")
        _state, report = _publish_report(
            repo_root=root,
            version_root=work / "v1",
            workflow_root=workflow_root,
            state=state,
            workflow_type="calibration",
            scope=calibration_budget,
            render=lambda: render_calibration_report(
                repo_root=root,
                version_root=work / "v1",
                workflow_root=workflow_root,
            ),
        )
    return report


def rebuild_comparison_report(
    *,
    repo_root: Path,
    work_root: Path,
    profile: str,
    efficiency_run_id: str | None,
) -> dict[str, Any]:
    """Rebuild only the presentation layer of one completed comparison."""

    if profile not in {"reduced", "full"}:
        raise BenchmarkConfigError("Comparison profile must be 'reduced' or 'full'")
    root, work = _workflow_work_root(repo_root, work_root)
    if efficiency_run_id is not None and not _WORKFLOW_ID.fullmatch(efficiency_run_id):
        raise BenchmarkConfigError(f"Invalid efficiency run ID: {efficiency_run_id!r}")
    workflow_root = _reportable_workflow_root(
        base=work / "v1/workflows/comparison" / profile,
        workflow_type="comparison",
        requested_id=None,
        required_fields=(
            {"efficiency_flow_id": efficiency_run_id} if efficiency_run_id is not None else None
        ),
    )
    with run_lock(workflow_root):
        state = load_json_object(workflow_root / "workflow.json", "comparison workflow state")
        _state, report = _publish_report(
            repo_root=root,
            version_root=work / "v1",
            workflow_root=workflow_root,
            state=state,
            workflow_type="comparison",
            scope=profile,
            render=lambda: render_comparison_report(
                repo_root=root,
                version_root=work / "v1",
                workflow_root=workflow_root,
            ),
        )
    return report


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
    if observed.get("status") not in {"running", "publishing", "complete"}:
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
        if statistics_paths:
            statistics_report = load_json_object(
                statistics_paths[0], "comparison statistical report"
            )
            contract = statistics_report.get("contract")
            analysis_mode = contract.get("status") if isinstance(contract, dict) else None
            if analysis_mode == "inferential":
                confirmatory = [
                    analysis
                    for analysis in statistics_report.get("analyses", [])
                    if isinstance(analysis, dict)
                    and analysis.get("analysis_role") == "confirmatory"
                ]
                if not confirmatory or any(
                    analysis.get("status") != "tested" for analysis in confirmatory
                ):
                    raise BenchmarkConfigError(
                        "Comparison cannot complete unless every confirmatory primary analysis "
                        "is tested on its complete predeclared panel"
                    )
            else:
                analyses = statistics_report.get("analyses")
                descriptive_statuses = {"contract_only", "descriptive_only"}
                if (
                    analysis_mode != "descriptive"
                    or statistics_report.get("status") not in descriptive_statuses
                    or not isinstance(analyses, list)
                    or not analyses
                    or any(
                        not isinstance(analysis, dict)
                        or analysis.get("status") not in descriptive_statuses
                        for analysis in analyses
                    )
                ):
                    raise BenchmarkConfigError(
                        "Descriptive comparison cannot complete unless every predeclared paired "
                        "analysis is complete"
                    )
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
    """Run or resume the complete bounded six-stage SPATHI calibration."""

    if calibration_budget not in CALIBRATION_BUDGETS:
        raise BenchmarkConfigError("Calibration profile must be 'reduced' or 'full'")
    root, work = _workflow_work_root(repo_root, work_root)
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
        if state["status"] == "running":
            atomic_write_json(
                state_path,
                state,
                "calibration workflow state",
                boundary=work,
            )
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
                atomic_write_json(
                    state_path,
                    state,
                    "calibration workflow state",
                    boundary=work,
                )
                parent_ranking = ranking

            if parent_ranking is None:
                raise AssertionError("Calibration workflow produced no validation ranking")
            selected_config = workflow_root / "selected-config.json"
            freeze_calibration_selection(
                repo_root=root,
                ranking_path=parent_ranking,
                output_path=selected_config,
            )
            report_path, _manifest_path = _report_paths(
                version_root=work / "v1",
                workflow_type="calibration",
                scope=calibration_budget,
                flow_id=flow_id,
            )
            state.update(
                {
                    "status": "publishing",
                    "selected_config": str(selected_config.relative_to(root)),
                    "selected_config_sha256": sha256_file(selected_config),
                    "interactive_report": report_path.relative_to(root).as_posix(),
                }
            )
            atomic_write_json(
                state_path,
                state,
                "calibration workflow state",
                boundary=work,
            )
        state, _report = _publish_report(
            repo_root=root,
            version_root=work / "v1",
            workflow_root=workflow_root,
            state=state,
            workflow_type="calibration",
            scope=calibration_budget,
            render=lambda: render_calibration_report(
                repo_root=root,
                version_root=work / "v1",
                workflow_root=workflow_root,
            ),
        )
    return state


def run_comparison_workflow(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    profile: str = "full",
    efficiency_run_id: str | None = None,
) -> dict[str, Any]:
    """Run or resume one selected-SPATHI comparison with all declared baselines."""

    if profile not in {"reduced", "full"}:
        raise BenchmarkConfigError("Comparison profile must be 'reduced' or 'full'")
    root, work = _workflow_work_root(repo_root, work_root)
    identity = _workflow_identity(root)
    production_config, _production = completed_production_config(
        repo_root=root,
        work_root=work,
        budget=profile,
        efficiency_run_id=efficiency_run_id,
    )
    efficiency_id = production_config.parent.name
    flow_id = f"{profile}-{efficiency_id}-{identity['fingerprint'][:10]}"
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
                "efficiency_budget": profile,
                "efficiency_flow_id": efficiency_id,
                "production_config": str(production_config.relative_to(root)),
                "production_config_sha256": sha256_file(production_config),
            },
        )
        if state["status"] == "running":
            atomic_write_json(
                state_path,
                state,
                "comparison workflow state",
                boundary=work,
            )
            bundle = workflow_root / "requests"
            profile_file = workflow_root / f"{profile_id}.json"
            materialize_benchmark_plan(
                repo_root=root,
                template_profile_id=profile,
                baseline_plan_path=root / f"experiments/v1/inference/plans/{profile}-v1.json",
                output_dir=bundle,
                profile_output=profile_file,
                production_config_path=production_config,
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
            report_path, _manifest_path = _report_paths(
                version_root=work / "v1",
                workflow_type="comparison",
                scope=profile,
                flow_id=flow_id,
            )
            state.update(
                {
                    "status": "publishing",
                    "stages": {"comparison": {"status": "complete", "pipeline": pipeline}},
                    "profile_file": str(profile_file.relative_to(root)),
                    "profile_sha256": sha256_file(profile_file),
                    "interactive_report": report_path.relative_to(root).as_posix(),
                }
            )
            atomic_write_json(
                state_path,
                state,
                "comparison workflow state",
                boundary=work,
            )
        state, _report = _publish_report(
            repo_root=root,
            version_root=work / "v1",
            workflow_root=workflow_root,
            state=state,
            workflow_type="comparison",
            scope=profile,
            render=lambda: render_comparison_report(
                repo_root=root,
                version_root=work / "v1",
                workflow_root=workflow_root,
            ),
        )
    return state
