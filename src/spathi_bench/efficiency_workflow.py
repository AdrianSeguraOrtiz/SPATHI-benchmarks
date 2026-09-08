"""High-level, resumable SPATHI efficiency-justification workflow."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any

from .artifacts import ArtifactKey, ArtifactStore, canonical_json_sha256, sha256_file
from .config import (
    BenchmarkConfigError,
    load_json_object,
    load_profile,
    validate_calibration_contracts,
    validate_json,
)
from .efficiency_development import (
    materialize_efficiency_development,
    select_efficiency_components,
    select_efficiency_development,
)
from .efficiency_safety import (
    SAFETY_CONDITION_SCALES,
    analyze_safety,
    materialize_safety_plan,
    provisional_candidate,
)
from .execution_policy import resolve_execution_class
from .noninferiority import analyze_noninferiority, materialize_noninferiority_plan
from .noninferiority_evidence import write_noninferiority_evidence
from .noninferiority_run import run_noninferiority_artifacts
from .pipeline import build_spathi_image, generate_data, preflight
from .provenance import observe_benchmark_repository
from .report_publication import publish_workflow_report
from .reporting import render_efficiency_report
from .resource_plan import detect_host_capacity, load_or_create_runtime_plan
from .resource_scheduler import host_execution_lease
from .runtime import (
    andrea_host_environment,
    benchmark_host_environment,
    docker_runtime_environment,
)
from .safe_io import write_idempotent_json
from .scenario_inputs import scenario_external_input_records
from .state import run_lock, state_path

_WORKFLOW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_IDENTITY_INPUTS = (
    "experiments/v1/execution/policy.json",
    "experiments/v1/spathi/efficiency-design-sensitivity.json",
    "experiments/v1/spathi/efficiency-development.json",
    "experiments/v1/spathi/efficiency-study.json",
    "images.lock.json",
    "repos.lock.json",
    "schemas/efficiency-design-sensitivity.schema.json",
    "schemas/execution-policy.schema.json",
    "schemas/efficiency-development-selection.schema.json",
    "schemas/efficiency-development.schema.json",
    "schemas/noninferiority-artifact-index.schema.json",
    "schemas/noninferiority-evidence.schema.json",
    "schemas/noninferiority-panel.schema.json",
    "schemas/noninferiority-plan.schema.json",
    "schemas/noninferiority-results.schema.json",
    "schemas/noninferiority-study.schema.json",
    "schemas/efficiency-safety-panel.schema.json",
    "schemas/efficiency-safety-plan.schema.json",
    "schemas/efficiency-safety-results.schema.json",
    "schemas/profile.schema.json",
    "schemas/spathi-production-config.schema.json",
    "schemas/spathi-selected-config.schema.json",
    "src/spathi_bench/design_sensitivity.py",
    "src/spathi_bench/efficiency_development.py",
    "src/spathi_bench/efficiency_safety.py",
    "src/spathi_bench/noninferiority.py",
    "src/spathi_bench/noninferiority_evidence.py",
    "src/spathi_bench/noninferiority_run.py",
    "src/spathi_bench/efficiency_order.py",
    "src/spathi_bench/efficiency_workflow.py",
    "src/spathi_bench/execution_policy.py",
    "src/spathi_bench/report_publication.py",
    "src/spathi_bench/scenario_inputs.py",
    "src/spathi_bench/tool_contract.py",
)


def _completed_selection(
    *, work: Path, repo_root: Path, budget: str, calibration_run_id: str | None
) -> tuple[Path, dict[str, Any]]:
    base = work / "v1/workflows/calibration" / budget
    if calibration_run_id is not None and not _WORKFLOW_ID.fullmatch(calibration_run_id):
        raise BenchmarkConfigError(f"Invalid calibration run ID: {calibration_run_id!r}")
    candidates = (
        [base / calibration_run_id]
        if calibration_run_id is not None
        else sorted(path.parent for path in base.glob("*/workflow.json"))
    )
    completed: list[tuple[Path, dict[str, Any]]] = []
    for candidate in candidates:
        if candidate.is_symlink():
            raise BenchmarkConfigError("Calibration workflow directory must not be a symlink")
        workflow_path = candidate / "workflow.json"
        if workflow_path.is_symlink():
            raise BenchmarkConfigError("Calibration workflow state must not be a symlink")
        if not workflow_path.is_file():
            continue
        workflow = load_json_object(workflow_path, "calibration workflow")
        raw_selected = workflow.get("selected_config")
        if (
            workflow.get("workflow_type") != "calibration"
            or workflow.get("status") != "complete"
            or workflow.get("calibration_budget") != budget
            or not isinstance(raw_selected, str)
        ):
            continue
        selected_reference = repo_root / raw_selected
        expected_reference = candidate / "selected-config.json"
        selected_path = selected_reference.resolve()
        if (
            Path(os.path.abspath(selected_reference)) != Path(os.path.abspath(expected_reference))
            or selected_reference.is_symlink()
            or selected_path.parent != candidate.resolve()
            or not selected_path.is_file()
            or sha256_file(selected_path) != workflow.get("selected_config_sha256")
        ):
            raise BenchmarkConfigError("Completed calibration selection is missing or changed")
        selected = load_json_object(selected_path, "frozen SPATHI selection")
        validate_json(
            selected,
            repo_root / "schemas/spathi-selected-config.schema.json",
            "frozen SPATHI selection",
        )
        completed.append((selected_path, selected))
    if len(completed) != 1:
        if not completed:
            raise BenchmarkConfigError(
                f"No completed {budget} calibration is available; run calibrate.sh first"
            )
        raise BenchmarkConfigError(
            "Several completed calibrations are available; pass --calibration-run-id"
        )
    return completed[0]


def _artifact_member(artifact_files: list[dict[str, Any]], suffix: str) -> str:
    matches = [
        item["path"]
        for item in artifact_files
        if item["path"] == suffix or item["path"].endswith("/" + suffix)
    ]
    if len(matches) != 1:
        raise BenchmarkConfigError(f"Generated dataset artifact lacks exactly one {suffix}")
    return matches[0]


def _panel_from_generation(
    *,
    repo_root: Path,
    run_root: Path,
    selected: dict[str, Any],
    budget: str,
) -> dict[str, Any]:
    state = load_json_object(state_path(run_root), "efficiency generation state")
    if state.get("status") != "generated":
        raise BenchmarkConfigError("Efficiency datasets were not generated completely")
    store = ArtifactStore(Path(state["artifact_store"]))
    datasets: list[dict[str, Any]] = []
    for item in sorted(state["datasets"], key=lambda row: row["dataset_id"]):
        if item.get("dataset_role") != "efficiency_validation":
            continue
        reference = item["dataset_artifact"]
        artifact = store.validate(
            ArtifactKey(kind=str(reference["kind"]), digest=str(reference["digest"]))
        )
        groups_relative = _artifact_member(
            artifact.manifest["files"],
            f"datasets/{item['dataset_id']}/extras/groups.tsv",
        )
        with artifact.file(groups_relative).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        contexts = sorted({f"group:{row['cluster']}" for row in rows if row.get("cluster")})
        if not contexts:
            raise BenchmarkConfigError(f"Generated dataset {item['dataset_id']} has no groups")
        datasets.append(
            {
                "dataset_id": item["dataset_id"],
                "simulator_id": item["simulator_id"],
                "condition_id": item["condition_id"],
                "generation_seed": item["seed"],
                "contexts": contexts,
                "dataset_artifact": {
                    "kind": "dataset",
                    "digest": artifact.key.digest,
                },
            }
        )
    evidence = selected["selection_evidence"]
    ranking_path = (repo_root / evidence["validation_ranking"]).resolve()
    if sha256_file(ranking_path) != evidence["validation_ranking_sha256"]:
        raise BenchmarkConfigError("Frozen selection ranking has changed")
    return {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_predeclared_panel",
        "selection_validation_manifest_sha256": evidence["validation_manifest_sha256"],
        "selection_validation_ranking": {
            "path": ranking_path.relative_to(repo_root).as_posix(),
            "sha256": evidence["validation_ranking_sha256"],
        },
        "budget": budget,
        "dataset_role": "efficiency_validation",
        "split_id": "efficiency_validation_v1",
        "datasets": datasets,
    }


def _safety_panel_from_generation(
    *,
    repo_root: Path,
    run_root: Path,
    selected: dict[str, Any],
    profile_payload: dict[str, Any],
) -> dict[str, Any]:
    state = load_json_object(state_path(run_root), "efficiency generation state")
    if state.get("status") != "generated":
        raise BenchmarkConfigError("Efficiency datasets were not generated completely")
    store = ArtifactStore(Path(state["artifact_store"]))
    datasets: list[dict[str, Any]] = []
    for item in sorted(state["datasets"], key=lambda row: row["dataset_id"]):
        if item.get("dataset_role") != "efficiency_safety":
            continue
        reference = item["dataset_artifact"]
        artifact = store.validate(
            ArtifactKey(kind=str(reference["kind"]), digest=str(reference["digest"]))
        )
        groups_relative = _artifact_member(
            artifact.manifest["files"],
            f"datasets/{item['dataset_id']}/extras/groups.tsv",
        )
        with artifact.file(groups_relative).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        contexts = sorted({f"group:{row['cluster']}" for row in rows if row.get("cluster")})
        if not contexts:
            raise BenchmarkConfigError(
                f"Generated safety dataset {item['dataset_id']} has no groups"
            )
        datasets.append(
            {
                "dataset_id": item["dataset_id"],
                "simulator_id": item["simulator_id"],
                "condition_id": item["condition_id"],
                "generation_seed": item["seed"],
                "contexts": contexts,
                "dataset_artifact": {"kind": "dataset", "digest": artifact.key.digest},
            }
        )
    raw_safety_jobs = [
        job
        for job in profile_payload["generation_jobs"]
        if job["dataset_role"] == "efficiency_safety"
    ]
    safety_jobs = {str(job["condition_id"]): job for job in raw_safety_jobs}
    if (
        len(profile_payload["generation_jobs"]) != len(SAFETY_CONDITION_SCALES)
        or len(raw_safety_jobs) != len(SAFETY_CONDITION_SCALES)
        or len(safety_jobs) != len(SAFETY_CONDITION_SCALES)
        or set(safety_jobs) != set(SAFETY_CONDITION_SCALES)
    ):
        raise BenchmarkConfigError(
            "Efficiency safety profile does not declare the exact condition panel"
        )
    condition_scales: dict[str, Any] = {}
    for condition_id, dimensions in SAFETY_CONDITION_SCALES.items():
        job = safety_jobs[condition_id]
        scenario_path = repo_root / job["scenario"]
        runs_path = repo_root / job["simulator_runs"]
        condition_scales[condition_id] = {
            "generation_job": job["id"],
            "simulator": "scmultisim",
            **dimensions,
            "scenario": {
                "path": scenario_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(scenario_path),
            },
            "simulator_runs": {
                "path": runs_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(runs_path),
            },
            "scenario_inputs": scenario_external_input_records(scenario_path, repo_root),
        }
    payload = {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_safety_panel",
        "selection_validation_manifest_sha256": selected["selection_evidence"][
            "validation_manifest_sha256"
        ],
        "budget": "full",
        "dataset_role": "efficiency_safety",
        "split_id": "efficiency_safety_v1",
        "condition_scales": condition_scales,
        "datasets": datasets,
    }
    validate_json(
        payload,
        repo_root / "schemas/efficiency-safety-panel.schema.json",
        "efficiency safety panel",
    )
    return payload


def _generate_profile(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    profile_id: str,
    run_root: Path,
    expected_spathi_image_id: str,
    runtime_plan: dict[str, Any],
) -> dict[str, Any]:
    """Materialize one content-bound generation profile and verify its SPATHI image."""

    common = {"repo_root": repo_root, "profile_id": profile_id, "run_root": run_root}
    with run_lock(run_root):
        preflight(
            **common,
            andrea_repo=andrea_repo,
            spathi_repo=spathi_repo,
            runtime_plan=runtime_plan,
        )
        build_spathi_image(**common, spathi_repo=spathi_repo)
        generate_data(**common, andrea_repo=andrea_repo)
    state = load_json_object(state_path(run_root), f"{profile_id} generation state")
    if state.get("status") != "generated" or state.get("profile_id") != profile_id:
        raise BenchmarkConfigError(f"{profile_id} datasets were not generated completely")
    image = state.get("spathi_image") or {}
    if image.get("id") != expected_spathi_image_id:
        raise BenchmarkConfigError(
            "Calibration and every efficiency study run must use the same SPATHI image; recalibrate"
        )
    return state


def _run_efficiency_development(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    workflow_root: Path,
    flow_id: str,
    budget: str,
    selected_path: Path,
    runtime_plan: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Run the finite development grid and freeze its outcome-bound selection."""

    development_root = workflow_root / "efficiency-development"

    component_step = "component_scout"
    component_root = development_root / "component-scout"
    component_profile_id = f"efficiency-development-{budget}-component-scout"
    component_profile_path = component_root / "profile.json"
    materialize_efficiency_development(
        repo_root=repo_root,
        selected_config_path=selected_path,
        budget=budget,
        step=component_step,
        component_selection_path=None,
        output_dir=component_root / "bundle",
        profile_output=component_profile_path,
    )
    component_run_root = work_root / "v1" / component_profile_id / flow_id
    # Local import avoids a module-level cycle: workflows also exposes this workflow.
    from .workflows import run_profile_pipeline

    run_profile_pipeline(
        repo_root=repo_root,
        profile_id=component_profile_id,
        profile_file=component_profile_path,
        run_root=component_run_root,
        andrea_repo=andrea_repo,
        spathi_repo=spathi_repo,
        statistics=False,
        runtime_plan=runtime_plan,
    )
    component_selection_path = development_root / "component-selection.json"
    select_efficiency_components(
        repo_root=repo_root,
        selected_config_path=selected_path,
        budget=budget,
        scout_metrics_path=component_run_root / "reports/metrics.csv",
        output_path=component_selection_path,
    )

    confirmation_step = "joint_confirmation"
    confirmation_root = development_root / "joint-confirmation"
    confirmation_profile_id = f"efficiency-development-{budget}-joint-confirmation"
    confirmation_profile_path = confirmation_root / "profile.json"
    materialize_efficiency_development(
        repo_root=repo_root,
        selected_config_path=selected_path,
        budget=budget,
        step=confirmation_step,
        component_selection_path=component_selection_path,
        output_dir=confirmation_root / "bundle",
        profile_output=confirmation_profile_path,
    )
    confirmation_run_root = work_root / "v1" / confirmation_profile_id / flow_id
    run_profile_pipeline(
        repo_root=repo_root,
        profile_id=confirmation_profile_id,
        profile_file=confirmation_profile_path,
        run_root=confirmation_run_root,
        andrea_repo=andrea_repo,
        spathi_repo=spathi_repo,
        statistics=False,
        runtime_plan=runtime_plan,
    )
    development_selection_path = development_root / "selection.json"
    select_efficiency_development(
        repo_root=repo_root,
        selected_config_path=selected_path,
        budget=budget,
        component_selection_path=component_selection_path,
        confirmation_metrics_path=confirmation_run_root / "reports/metrics.csv",
        output_path=development_selection_path,
    )
    return confirmation_run_root, confirmation_profile_path, development_selection_path


def _production_config(
    *,
    repo_root: Path,
    budget: str,
    selected_path: Path,
    plan_path: Path,
    evidence_path: Path,
    results_path: Path,
    safety_panel_path: Path | None = None,
    safety_plan_path: Path | None = None,
    safety_evidence_path: Path | None = None,
    safety_results_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze the only SPATHI policy that downstream comparison may execute."""

    plan = load_json_object(plan_path, "efficiency plan")
    results = load_json_object(results_path, "efficiency results")
    adoption = results.get("adoption")
    if not isinstance(adoption, dict) or not isinstance(
        adoption.get("selected_group_policy"), dict
    ):
        raise BenchmarkConfigError("Efficiency analysis produced no deployable policy decision")
    safety_paths = (
        safety_panel_path,
        safety_plan_path,
        safety_evidence_path,
        safety_results_path,
    )
    if any(path is not None for path in safety_paths) and not all(
        path is not None for path in safety_paths
    ):
        raise BenchmarkConfigError("Efficiency safety provenance must be complete or absent")
    final_decision = adoption["decision"]
    selected_policy_id = adoption["selected_policy_id"]
    group_policy = adoption["selected_group_policy"]
    selected_policy_sha256 = adoption["selected_policy_sha256"]
    safety_provenance: dict[str, Any] | None = None
    if safety_results_path is not None:
        safety = load_json_object(safety_results_path, "efficiency safety results")
        validate_json(
            safety,
            repo_root / "schemas/efficiency-safety-results.schema.json",
            "efficiency safety results",
        )
        if safety["inputs"] != {
            "plan_sha256": sha256_file(safety_plan_path),
            "evidence_sha256": sha256_file(safety_evidence_path),
        }:
            raise BenchmarkConfigError("Efficiency safety results target different inputs")
        if (
            adoption["decision"] != "candidate_adopted"
            or safety["provisional_candidate_id"] != adoption["selected_candidate_id"]
            or safety["provisional_candidate_policy_sha256"] != adoption["selected_policy_sha256"]
        ):
            raise BenchmarkConfigError(
                "Efficiency safety does not evaluate the formal provisional candidate"
            )
        safety_decision = safety["decision"]
        final_decision = (
            "candidate_adopted"
            if safety_decision["status"] == "candidate_authorized"
            else "reference_retained"
        )
        selected_policy_id = safety_decision["selected_policy_id"]
        selected_policy_sha256 = safety_decision["selected_policy_sha256"]
        group_policy = safety_decision["selected_group_policy"]
        expected_final = (
            {
                "selected_policy_id": adoption["selected_policy_id"],
                "selected_policy_sha256": adoption["selected_policy_sha256"],
                "selected_group_policy": adoption["selected_group_policy"],
            }
            if safety_decision["status"] == "candidate_authorized"
            else {
                "selected_policy_id": plan["reference"]["policy_id"],
                "selected_policy_sha256": plan["reference"]["policy_sha256"],
                "selected_group_policy": plan["reference"]["group_policy"],
            }
        )
        if any(safety_decision[key] != value for key, value in expected_final.items()):
            raise BenchmarkConfigError(
                "Efficiency safety decision does not select the authorized candidate or exact reference"
            )
        safety_provenance = {
            name: {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in (
                ("panel", safety_panel_path),
                ("plan", safety_plan_path),
                ("evidence", safety_evidence_path),
                ("results", safety_results_path),
            )
            if path is not None
        }
    elif budget == "full" and adoption["decision"] == "candidate_adopted":
        raise BenchmarkConfigError(
            "A full provisional candidate requires efficiency safety before production"
        )
    policy_sha256 = canonical_json_sha256(group_policy)
    if policy_sha256 != selected_policy_sha256:
        raise BenchmarkConfigError("Final efficiency policy has a stale digest")
    reference = plan.get("reference")
    if not isinstance(reference, dict):
        raise BenchmarkConfigError("Efficiency plan has no exact reference")
    return {
        "schema_version": "1.0",
        "artifact_type": "spathi_production_policy",
        "budget": budget,
        "evidence_role": results["analysis_role"],
        "status": "confirmed" if results["analysis_role"] == "confirmatory" else "descriptive",
        "decision": final_decision,
        "policy": {
            "policy_id": selected_policy_id,
            "policy_sha256": policy_sha256,
            "group_policy": group_policy,
            "spathi_commit": reference["spathi_commit"],
        },
        "provenance": {
            "calibration_selection": {
                "path": selected_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(selected_path),
            },
            "efficiency_plan": {
                "path": plan_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(plan_path),
            },
            "efficiency_evidence": {
                "path": evidence_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(evidence_path),
            },
            "efficiency_results": {
                "path": results_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(results_path),
            },
            **({"efficiency_safety": safety_provenance} if safety_provenance is not None else {}),
        },
    }


def run_efficiency_workflow(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    profile: str,
    calibration_run_id: str | None = None,
    max_cores: int | None = None,
    max_ram_gb: float | None = None,
    reserve_cores: int = 0,
    reserve_ram_gb: float = 0.0,
) -> dict[str, Any]:
    """Generate the independent panel, run all arms, verify evidence and analyze it."""

    if profile not in {"reduced", "full"}:
        raise BenchmarkConfigError("Efficiency profile must be 'reduced' or 'full'")
    root = repo_root.resolve()
    work = work_root.resolve()
    if not work.is_relative_to(root / "work"):
        raise BenchmarkConfigError("Efficiency work root must be below repository work/")
    selected_path, selected = _completed_selection(
        work=work,
        repo_root=root,
        budget=profile,
        calibration_run_id=calibration_run_id,
    )
    # A selected-config is not authoritative by itself: reconstruct its winner
    # from the complete ranking chain before it can seed a post-selection study.
    validate_calibration_contracts(root, selected_path)
    benchmark = observe_benchmark_repository(root, require_clean=True)
    ranking_path = root / selected["selection_evidence"]["validation_ranking"]
    ranking = load_json_object(ranking_path, "calibration validation ranking")
    if ranking.get("benchmark_commit") != benchmark["commit"]:
        raise BenchmarkConfigError(
            "Calibration evidence was produced by another benchmark revision; recalibrate"
        )
    profile_id = f"efficiency-{profile}"
    profile_payload = load_profile(root, profile_id, require_runnable=True)
    development_profile_id = f"efficiency-development-{profile}"
    load_profile(root, development_profile_id)
    safety_profile_id = "efficiency-safety-full" if profile == "full" else None
    safety_profile_payload = (
        load_profile(root, safety_profile_id, require_runnable=True)
        if safety_profile_id is not None
        else None
    )
    profile_digests = {
        "efficiency": sha256_file(root / f"experiments/v1/profiles/{profile_id}.json"),
        "efficiency_development": sha256_file(
            root / f"experiments/v1/profiles/{development_profile_id}.json"
        ),
        **(
            {
                "efficiency_safety": sha256_file(
                    root / f"experiments/v1/profiles/{safety_profile_id}.json"
                )
            }
            if safety_profile_id is not None
            else {}
        ),
    }
    identity_fingerprint = canonical_json_sha256(
        {
            "profile_sha256": profile_digests["efficiency"],
            "profile_sha256s": profile_digests,
            "selected_config_sha256": sha256_file(selected_path),
            "benchmark_repository": {
                "commit": benchmark["commit"],
                "origin": benchmark["origin"],
            },
            "andrea_environment": andrea_host_environment()["fingerprint"],
            "benchmark_environment": benchmark_host_environment()["fingerprint"],
            "docker_runtime": docker_runtime_environment()["fingerprint"],
            "implementation": {
                relative: sha256_file(root / relative) for relative in _IDENTITY_INPUTS
            },
        }
    )
    flow_id = identity_fingerprint[:20]
    workflow_root = work / "v1/workflows/efficiency" / profile / flow_id
    run_root = work / "v1" / profile_id / flow_id
    workflow_path = workflow_root / "workflow.json"
    runtime_plan = load_or_create_runtime_plan(
        workflow_root / "runtime-plan.json",
        boundary=workflow_root,
        max_cores=max_cores,
        max_ram_gb=max_ram_gb,
        reserve_cores=reserve_cores,
        reserve_ram_gb=reserve_ram_gb,
        detected=detect_host_capacity(workspace_path=work),
    )
    execution_plan = resolve_execution_class(
        repo_root=root,
        class_id=str(profile_payload["execution_class"]),
        runtime_plan=runtime_plan,
    )
    with host_execution_lease(workload=f"efficiency:{flow_id}"), run_lock(workflow_root):
        expected_image_id = selected["selection_evidence"]["spathi_image_id"]
        (
            development_run_root,
            development_profile_path,
            development_selection_path,
        ) = _run_efficiency_development(
            repo_root=root,
            andrea_repo=andrea_repo,
            spathi_repo=spathi_repo,
            work_root=work,
            workflow_root=workflow_root,
            flow_id=flow_id,
            budget=profile,
            selected_path=selected_path,
            runtime_plan=runtime_plan,
        )
        _generate_profile(
            repo_root=root,
            andrea_repo=andrea_repo,
            spathi_repo=spathi_repo,
            profile_id=profile_id,
            run_root=run_root,
            expected_spathi_image_id=expected_image_id,
            runtime_plan=runtime_plan,
        )
        panel_path = workflow_root / "panel.json"
        panel = _panel_from_generation(
            repo_root=root,
            run_root=run_root,
            selected=selected,
            budget=profile,
        )
        write_idempotent_json(
            panel_path,
            panel,
            "efficiency panel",
            boundary=root,
        )
        plan_path = workflow_root / "plan.json"
        materialize_noninferiority_plan(
            repo_root=root,
            selected_config_path=selected_path,
            panel_path=panel_path,
            development_selection_path=development_selection_path,
            budget=profile,
            output_path=plan_path,
        )
        index_path = workflow_root / "artifact-index.json"
        run_noninferiority_artifacts(
            repo_root=root,
            plan_path=plan_path,
            andrea_repo=andrea_repo,
            artifact_index_path=index_path,
            artifact_store_path=work / "v1/artifacts",
            execution_plan=execution_plan,
        )
        evidence_path = workflow_root / "evidence.json"
        write_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
            output_path=evidence_path,
        )
        results = analyze_noninferiority(
            repo_root=root,
            plan_path=plan_path,
            evidence_path=evidence_path,
            output_dir=workflow_root / "results",
        )
        safety_panel_path: Path | None = None
        safety_plan_path: Path | None = None
        safety_index_path: Path | None = None
        safety_evidence_path: Path | None = None
        safety_results_path: Path | None = None
        safety_generation_run: Path | None = None
        if (
            profile == "full"
            and provisional_candidate(
                root,
                efficiency_plan_path=plan_path,
                efficiency_results_path=results[0],
            )
            is not None
        ):
            if safety_profile_id is None or safety_profile_payload is None:
                raise BenchmarkConfigError("Full efficiency safety profile is unavailable")
            safety_generation_run = work / "v1" / safety_profile_id / flow_id
            _generate_profile(
                repo_root=root,
                andrea_repo=andrea_repo,
                spathi_repo=spathi_repo,
                profile_id=safety_profile_id,
                run_root=safety_generation_run,
                expected_spathi_image_id=expected_image_id,
                runtime_plan=runtime_plan,
            )
            safety_root = workflow_root / "efficiency-safety"
            safety_panel_path = safety_root / "panel.json"
            write_idempotent_json(
                safety_panel_path,
                _safety_panel_from_generation(
                    repo_root=root,
                    run_root=safety_generation_run,
                    selected=selected,
                    profile_payload=safety_profile_payload,
                ),
                "efficiency safety panel",
                boundary=root,
            )
            safety_plan_path = safety_root / "plan.json"
            materialize_safety_plan(
                repo_root=root,
                efficiency_plan_path=plan_path,
                efficiency_results_path=results[0],
                panel_path=safety_panel_path,
                output_path=safety_plan_path,
            )
            safety_index_path = safety_root / "artifact-index.json"
            run_noninferiority_artifacts(
                repo_root=root,
                plan_path=safety_plan_path,
                andrea_repo=andrea_repo,
                artifact_index_path=safety_index_path,
                artifact_store_path=work / "v1/artifacts",
                execution_plan=resolve_execution_class(
                    repo_root=root,
                    class_id=str(safety_profile_payload["execution_class"]),
                    runtime_plan=runtime_plan,
                ),
            )
            safety_evidence_path = safety_root / "evidence.json"
            write_noninferiority_evidence(
                repo_root=root,
                plan_path=safety_plan_path,
                artifact_index_path=safety_index_path,
                output_path=safety_evidence_path,
            )
            safety_results_path = safety_root / "results.json"
            analyze_safety(
                repo_root=root,
                plan_path=safety_plan_path,
                evidence_path=safety_evidence_path,
                output_path=safety_results_path,
            )
        production_path = workflow_root / "production-config.json"
        production = _production_config(
            repo_root=root,
            budget=profile,
            selected_path=selected_path,
            plan_path=plan_path,
            evidence_path=evidence_path,
            results_path=results[0],
            safety_panel_path=safety_panel_path,
            safety_plan_path=safety_plan_path,
            safety_evidence_path=safety_evidence_path,
            safety_results_path=safety_results_path,
        )
        validate_json(
            production,
            root / "schemas/spathi-production-config.schema.json",
            "SPATHI production policy",
        )
        write_idempotent_json(
            production_path,
            production,
            "production policy",
            boundary=root,
        )
        report_path = work / "v1/reports" / profile / flow_id / "efficiency.html"
        publishing_state = {
            "schema_version": "1.0",
            "workflow_type": "efficiency",
            "profile": profile,
            "flow_id": flow_id,
            "identity": {
                "fingerprint": identity_fingerprint,
                "spathi_commit": production["policy"]["spathi_commit"],
                "selected_config_sha256": sha256_file(selected_path),
                "profile_sha256s": profile_digests,
            },
            "status": "publishing",
            "runtime_plan": runtime_plan,
            "efficiency_development": {
                "run": development_run_root.relative_to(root).as_posix(),
                "profile": development_profile_path.relative_to(root).as_posix(),
                "selection": development_selection_path.relative_to(root).as_posix(),
                "selection_sha256": sha256_file(development_selection_path),
            },
            "generation_run": run_root.relative_to(root).as_posix(),
            "selected_config": selected_path.relative_to(root).as_posix(),
            "panel": panel_path.relative_to(root).as_posix(),
            "plan": plan_path.relative_to(root).as_posix(),
            "artifact_index": index_path.relative_to(root).as_posix(),
            "evidence": evidence_path.relative_to(root).as_posix(),
            "results": [path.relative_to(root).as_posix() for path in results],
            "efficiency_safety": (
                {
                    "status": "completed",
                    "generation_run": safety_generation_run.relative_to(root).as_posix(),
                    "panel": safety_panel_path.relative_to(root).as_posix(),
                    "plan": safety_plan_path.relative_to(root).as_posix(),
                    "artifact_index": safety_index_path.relative_to(root).as_posix(),
                    "evidence": safety_evidence_path.relative_to(root).as_posix(),
                    "results": safety_results_path.relative_to(root).as_posix(),
                }
                if all(
                    path is not None
                    for path in (
                        safety_panel_path,
                        safety_plan_path,
                        safety_index_path,
                        safety_evidence_path,
                        safety_results_path,
                    )
                )
                else {
                    "status": "not_required",
                    "reason": (
                        "reduced_descriptive_budget"
                        if profile == "reduced"
                        else "exact_reference_retained_by_confirmatory_study"
                    ),
                }
            ),
            "production_config": production_path.relative_to(root).as_posix(),
            "production_config_sha256": sha256_file(production_path),
            "interactive_report": report_path.relative_to(root).as_posix(),
        }
        state, report = publish_workflow_report(
            repo_root=root,
            workflow_path=workflow_path,
            publishing_state=publishing_state,
            html_path=report_path,
            manifest_path=report_path.parent / "efficiency-report.manifest.json",
            render=lambda: render_efficiency_report(
                repo_root=root,
                version_root=work / "v1",
                workflow_root=workflow_root,
            ),
            label="efficiency",
        )
    return state
