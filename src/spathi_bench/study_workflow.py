"""One named, crash-resumable study from scientific qualification to reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import BenchmarkConfigError, load_json_object
from .dimensional_transfer_workflow import run_dimensional_transfer_workflow
from .efficiency_workflow import run_efficiency_workflow
from .performance_workflow import run_performance_workflow
from .resource_plan import detect_host_capacity, resolve_runtime_plan
from .scalability_workflow import run_scalability_workflow
from .state import run_lock
from .study_amendments import require_no_pending_amendments
from .study_context import study_execution
from .study_state import (
    PHASES,
    load_study,
    new_id,
    now,
    observe_revisions,
    phase_receipt,
    record_invocation,
    save_study,
    scientific_inputs,
    study_directory,
    validate_receipt,
)
from .workflows import run_calibration_workflow, run_comparison_workflow, run_profile_pipeline

logger = logging.getLogger(__name__)
PURPOSES = {
    "smoke": "Verify Docker simulation, inference, evaluation and ANDREA exports on tiny data.",
    "calibration": "Select a bounded SPATHI configuration on development and validation data only.",
    "efficiency": "Select acceleration settings, then test accuracy loss and runtime on independent data.",
    "dimensional_transfer": "Audit the frozen policy when gene and regulator dimensionality increases.",
    "comparison": "Compare frozen SPATHI with external tools on held-out simulated datasets.",
    "scalability": "Measure the feasible size boundary of the frozen SPATHI policy.",
    "performance": "Measure isolated tool runtime and memory across dataset sizes and thread counts.",
}


def _summary(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": state["run_id"], "profile": state["profile"], "status": state["status"],
        "current_phase": state.get("current_phase"), "report_path": str(root / "reports/index.html"),
        "error": state["invocations"][-1].get("error") if state.get("invocations") else None,
        "phases": {name: {key: value for key, value in record.items() if key != "evidence"}
                   for name, record in state["phases"].items()},
        "warnings": state["warnings"], "resume_command": f"spathi-bench run --id {state['run_id']}",
    }


def _smoke(**kwargs: Any) -> dict[str, Any]:
    root, run_id = kwargs.pop("work_root"), kwargs.pop("run_id")
    pipeline = run_profile_pipeline(
        **kwargs, profile_id="smoke", profile_file=None,
        run_root=root / "v1/smoke" / run_id, statistics=True,
    )
    return {"flow_id": run_id, "status": "complete", "pipeline": pipeline}


def run_study_workflow(
    *, repo_root: Path, andrea_repo: Path, spathi_repo: Path, work_root: Path,
    profile: str | None = None, run_id: str | None = None,
    max_cores: int | None = None, max_ram_gb: float | None = None,
    reserve_cores: int = 0, reserve_ram_gb: float = 0.0,
) -> dict[str, Any]:
    """A new ID starts independently; an existing ID resumes durable phase checkpoints."""
    from .study_reporting import render_study_index

    if profile is not None and profile not in {"smoke", "reduced", "full"}:
        raise BenchmarkConfigError("Study profile must be smoke, reduced or full")
    run_id = run_id or new_id(profile or "reduced")
    root = study_directory(repo_root, work_root, run_id)
    requested = dict(max_cores=max_cores, max_ram_gb=max_ram_gb,
                     reserve_cores=reserve_cores, reserve_ram_gb=reserve_ram_gb)
    with run_lock(root):
        require_no_pending_amendments(root)
        resume = (root / "study.json").is_file()
        revisions = observe_revisions(repo_root, andrea_repo, spathi_repo)
        inputs = scientific_inputs(repo_root)
        if resume:
            state = load_study(root)
            if profile is not None and profile != state["profile"]:
                raise BenchmarkConfigError("An existing study ID cannot change its profile")
            for key, value in requested.items():
                if value not in (None, 0, 0.0) and value != state["resources"][key]:
                    raise BenchmarkConfigError("Resource ceilings are fixed per study; use a new ID to change them")
        else:
            profile = profile or "reduced"
            runtime_plan = resolve_runtime_plan(detect_host_capacity(workspace_path=work_root), **requested)
            state = {
                "schema_version": "1.0", "workflow_type": "study", "run_id": run_id,
                "profile": profile, "status": "created", "created_at": now(),
                "scientific_inputs": inputs, "resources": requested, "resource_plan": runtime_plan,
                "invocations": [], "warnings": [],
                "phases": {name: {"status": "pending"} for name in (("smoke",) if profile == "smoke" else PHASES)},
            }
        record_invocation(state, revisions, inputs)
        state["status"] = "running"
        save_study(root, state)
        logger.info("Study %s (%s). Resume: spathi-bench run --id %s", run_id, state["profile"], run_id)
        common = dict(repo_root=repo_root, andrea_repo=andrea_repo, spathi_repo=spathi_repo,
                      work_root=root, **state["resources"])
        functions = {
            "calibration": run_calibration_workflow, "efficiency": run_efficiency_workflow,
            "dimensional_transfer": run_dimensional_transfer_workflow,
            "comparison": run_comparison_workflow, "scalability": run_scalability_workflow,
            "performance": run_performance_workflow, "smoke": _smoke,
        }
        with study_execution(run_id, revisions=revisions, resume=resume):
            try:
                if state["profile"] != "smoke" and "qualification" not in state:
                    from .qualification import run_qualification

                    state["current_phase"] = "qualification"
                    save_study(root, state)
                    state["qualification"] = run_qualification(
                        repo_root=repo_root, andrea_repo=andrea_repo, spathi_repo=spathi_repo,
                        work_root=work_root, run_id=f"{run_id}-contracts", **state["resources"],
                    )
                    if state["qualification"].get("status") != "complete":
                        raise BenchmarkConfigError("Scientific contract qualification did not complete")
                    save_study(root, state)
                phase_order = ("smoke",) if state["profile"] == "smoke" else PHASES
                for name in phase_order:
                    record = state["phases"][name]
                    state["current_phase"] = name
                    if record["status"] == "complete":
                        validate_receipt(root, record)
                        logger.info("%s: evidence verified; no inference repeated", name)
                        continue
                    record.update(status="running", started_at=record.get("started_at", now()))
                    record.pop("error", None)
                    save_study(root, state)
                    render_study_index(repo_root=repo_root, study_root=root, state=state)
                    logger.info("Starting/resuming phase: %s", name)
                    arguments: dict[str, Any] = dict(common)
                    if name == "smoke":
                        arguments["run_id"] = run_id
                    elif name == "calibration":
                        arguments.update(calibration_budget=state["profile"], run_id=run_id)
                    else:
                        arguments["profile"] = state["profile"]
                        key = "calibration_run_id" if name == "efficiency" else "efficiency_run_id"
                        upstream = "calibration" if name == "efficiency" else "efficiency"
                        arguments[key] = state["phases"][upstream]["flow_id"]
                    result = functions[name](**arguments)
                    if result.get("status") != "complete" or not result.get("flow_id"):
                        raise BenchmarkConfigError(f"{name} did not return a completed workflow")
                    workflow_root = (
                        root / "v1/smoke" / run_id if name == "smoke"
                        else root / "v1/workflows" / name / state["profile"] / result["flow_id"]
                    )
                    state["phases"][name] = phase_receipt(repo_root, root, workflow_root, result)
                    save_study(root, state)
                    render_study_index(repo_root=repo_root, study_root=root, state=state)
                state.update(
                    status="complete", current_phase=None,
                    completed_at=state.get("completed_at") or now(),
                )
                state["invocations"][-1].update(status="complete", finished_at=now())
            except BaseException as exc:
                state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                state["invocations"][-1].update(status=state["status"], finished_at=now(), error=str(exc))
                active_phase = state["phases"].get(state.get("current_phase"))
                if active_phase is not None and active_phase.get("status") != "complete":
                    active_phase.update(status=state["status"], error=str(exc))
                save_study(root, state)
                render_study_index(repo_root=repo_root, study_root=root, state=state)
                logger.error("Study %s stopped in %s. Resume: spathi-bench run --id %s", run_id, state.get("current_phase"), run_id)
                raise
            save_study(root, state)
            render_study_index(repo_root=repo_root, study_root=root, state=state)
    return _summary(root, state)


def study_status(*, repo_root: Path, work_root: Path, run_id: str | None = None) -> dict[str, Any]:
    if run_id is not None:
        root = study_directory(repo_root, work_root, run_id)
        return _summary(root, load_study(root))
    roots = sorted((work_root / "studies").glob("*/study.json"))
    return {"studies": [
        _summary(root, load_study(root))
        for path in roots
        for root in (study_directory(repo_root, work_root, path.parent.name),)
    ]}


def rebuild_study_reports(*, repo_root: Path, work_root: Path, run_id: str) -> dict[str, Any]:
    from .study_reporting import rebuild_phase_reports, render_study_index

    root = study_directory(repo_root, work_root, run_id)
    with run_lock(root):
        state = load_study(root)
        rebuild_phase_reports(repo_root=repo_root, study_root=root, state=state)
        render_study_index(repo_root=repo_root, study_root=root, state=state)
    return _summary(root, state)


def plan_study(
    *, repo_root: Path, andrea_repo: Path, spathi_repo: Path, work_root: Path,
    profile: str = "reduced", **resources: Any,
) -> dict[str, Any]:
    if profile not in {"smoke", "reduced", "full"}:
        raise BenchmarkConfigError("Study profile must be smoke, reduced or full")
    plan = resolve_runtime_plan(detect_host_capacity(workspace_path=work_root), **resources)
    records = []
    templates = {"comparison": profile, "calibration": f"calibration-{profile}"}
    for phase in (("smoke",) if profile == "smoke" else PHASES):
        template = templates.get(phase, "smoke" if phase == "smoke" else f"{phase.replace('_', '-')}-{profile}")
        definition = load_json_object(repo_root / "experiments/v1/profiles" / f"{template}.json", "phase profile")
        records.append({
            "phase": phase, "purpose": PURPOSES[phase], "template": template,
            "generation_jobs": definition.get("generation_jobs", []),
            "planned_design": definition.get("planned_design", {}),
        })
    return {
        "profile": profile, "status": "planned", "resource_plan": plan["capacity"], "phases": records,
        "estimate": "Not yet measured for this revision and host. Isolated timings remain serial; CPU count alone is not a wall-time prediction.",
        "warnings": ["Reduced and smoke cannot establish publication-level significance."] if profile != "full" else [],
    }


__all__ = ["run_study_workflow", "study_status", "rebuild_study_reports", "plan_study"]
