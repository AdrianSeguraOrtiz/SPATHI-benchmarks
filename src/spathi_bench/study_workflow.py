"""End-to-end orchestration for one complete SPATHI benchmark budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import BenchmarkConfigError
from .dimensional_transfer_workflow import run_dimensional_transfer_workflow
from .efficiency_workflow import run_efficiency_workflow
from .performance_workflow import run_performance_workflow
from .scalability_workflow import run_scalability_workflow
from .workflows import run_calibration_workflow, run_comparison_workflow


def _completed_phase(state: dict[str, Any], phase: str) -> dict[str, str]:
    flow_id = state.get("flow_id")
    if state.get("status") != "complete" or not isinstance(flow_id, str) or not flow_id:
        raise BenchmarkConfigError(f"{phase} did not return a completed workflow identity")
    return {"flow_id": flow_id, "status": "complete"}


def run_study_workflow(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    profile: str,
    max_cores: int | None = None,
    max_ram_gb: float | None = None,
    reserve_cores: int = 0,
    reserve_ram_gb: float = 0.0,
) -> dict[str, Any]:
    """Run or resume every phase while carrying exact upstream identities.

    Each constituent workflow remains independently content-bound and resumable.
    The coordinator deliberately passes the calibration and efficiency flow IDs
    returned by the current invocation, so historical workflows in ``work/`` can
    never make the natural end-to-end path ambiguous.
    """

    if profile not in {"reduced", "full"}:
        raise BenchmarkConfigError("Study profile must be 'reduced' or 'full'")
    resources = {
        "max_cores": max_cores,
        "max_ram_gb": max_ram_gb,
        "reserve_cores": reserve_cores,
        "reserve_ram_gb": reserve_ram_gb,
    }
    common = {
        "repo_root": repo_root,
        "andrea_repo": andrea_repo,
        "spathi_repo": spathi_repo,
        "work_root": work_root,
    }

    calibration = run_calibration_workflow(
        **common,
        calibration_budget=profile,
        run_id=None,
        **resources,
    )
    calibration_summary = _completed_phase(calibration, "Calibration")
    calibration_flow_id = calibration_summary["flow_id"]

    efficiency = run_efficiency_workflow(
        **common,
        profile=profile,
        calibration_run_id=calibration_flow_id,
        **resources,
    )
    efficiency_summary = _completed_phase(efficiency, "Efficiency justification")
    efficiency_flow_id = efficiency_summary["flow_id"]

    downstream = {
        "dimensional_transfer": run_dimensional_transfer_workflow,
        "comparison": run_comparison_workflow,
        "scalability": run_scalability_workflow,
        "performance": run_performance_workflow,
    }
    phase_summaries: dict[str, dict[str, str]] = {
        "calibration": calibration_summary,
        "efficiency": efficiency_summary,
    }
    for phase, workflow in downstream.items():
        state = workflow(
            **common,
            profile=profile,
            efficiency_run_id=efficiency_flow_id,
            **resources,
        )
        phase_summaries[phase] = _completed_phase(state, phase.replace("_", " ").title())

    return {
        "schema_version": "1.0",
        "workflow_type": "study",
        "profile": profile,
        "status": "complete",
        "calibration_run_id": calibration_flow_id,
        "efficiency_run_id": efficiency_flow_id,
        "phases": phase_summaries,
    }


__all__ = ["run_study_workflow"]
