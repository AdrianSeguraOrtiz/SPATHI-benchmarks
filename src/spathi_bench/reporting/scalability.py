"""Interactive report for the SPATHI computational scalability workflow."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import BenchmarkConfigError, load_json_object, validate_json
from ..execution_policy import POLICY_PATH, resolve_execution_class
from ..provenance import production_provenance_artifacts
from ..resource_plan import validate_runtime_plan
from .common import (
    central_report_path,
    path_below,
    reconcile_report_bundle,
    repo_file,
    require_array,
    require_mapping,
    require_reportable_workflow,
    safe_relative_link,
    verify_sha256,
    write_report_bundle,
)


def _range(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    values = [
        int(require_mapping(row.get("dimensions"), "dataset dimensions")[field]) for row in rows
    ]
    return {"minimum": min(values), "maximum": max(values)}


def _dataset_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dimensions = [require_mapping(row.get("dimensions"), "dataset dimensions") for row in rows]
    return {
        "dataset_count": len(rows),
        "total_cells": sum(int(item["cells"]) for item in dimensions),
        "cells": _range(rows, "cells"),
        "genes": _range(rows, "genes"),
        "groups": _range(rows, "groups"),
        "transcription_factors": _range(rows, "transcription_factors"),
        "targets": _range(rows, "targets"),
    }


def _close(observed: Any, expected: float) -> bool:
    return (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
    )


def _validate_summary(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        require_mapping(item, "scalability dataset")
        for item in require_array(results.get("datasets"), "scalability datasets")
    ]
    summary = require_mapping(results.get("summary"), "scalability summary")
    boundary = require_mapping(results.get("resource_boundary"), "scalability boundary")
    scout_size = boundary.get("target_scout_size")
    identities = [row.get("dataset_id") for row in rows]
    if len(set(identities)) != len(identities):
        raise BenchmarkConfigError("Scalability results contain duplicate dataset identities")
    skipped_status = "not_attempted_after_preregistered_resource_boundary"
    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") == "failed"]
    skipped = [row for row in rows if row.get("status") == skipped_status]
    if len(completed) + len(failed) + len(skipped) != len(rows):
        raise BenchmarkConfigError("Scalability results contain an invalid dataset status")
    attempts = [
        require_mapping(raw, f"{stage} attempt for {row['dataset_id']}")
        for row in rows
        for stage in ("scout", "all_targets")
        for raw in (row.get(stage),)
        if raw is not None
    ]
    completed_attempts = [row for row in attempts if row.get("status") == "completed"]
    failed_attempts = [row for row in attempts if row.get("status") == "failed"]
    expected_failures = {
        kind: sum(
            1
            for row in failed_attempts
            if require_mapping(row.get("failure"), "failure details").get("failure_kind") == kind
        )
        for kind in ("oom", "timeout", "tool_error", "infrastructure_error")
    }
    if (
        summary.get("datasets_requested") != len(rows)
        or summary.get("datasets_completed") != len(completed)
        or summary.get("datasets_failed") != len(failed)
        or summary.get("datasets_not_attempted") != len(skipped)
        or summary.get("all_completed") != (len(completed) == len(rows))
        or summary.get("scouts_attempted") != sum(row.get("scout") is not None for row in rows)
        or summary.get("scouts_completed")
        != sum((row.get("scout") or {}).get("status") == "completed" for row in rows)
        or summary.get("all_targets_attempted")
        != sum(row.get("all_targets") is not None for row in rows)
        or summary.get("all_targets_completed") != len(completed)
        or summary.get("failure_kinds") != expected_failures
    ):
        raise BenchmarkConfigError("Scalability completion summary differs from its datasets")

    expected_times = {
        "total_generation_wall_time_seconds": sum(
            float(
                require_mapping(
                    require_mapping(row["scout"], "scout attempt")["generation"],
                    "generation measurement",
                )["wall_time_seconds"]
            )
            for row in rows
            if row.get("scout") is not None
        ),
        "total_inference_wall_time_seconds": sum(
            float(
                require_mapping(
                    row["inference"] if row.get("status") == "completed" else row["observations"],
                    "inference measurement",
                )["andrea_execution_measurement"]["wall_time_seconds"]
            )
            for row in attempts
        ),
    }
    for field, expected in expected_times.items():
        if not _close(summary.get(field), expected):
            raise BenchmarkConfigError(f"Scalability summary has an inconsistent {field}")
    observed_peaks = [
        int(value)
        for row in attempts
        for value in (
            row["inference"]["andrea_measurement"]["peak_memory_bytes"]
            if row.get("status") == "completed"
            else row["observations"]["andrea_measurement"].get("peak_memory_bytes"),
        )
        if value is not None
    ]
    expected_integers = {
        "attempts_with_peak_memory": len(observed_peaks),
        "maximum_peak_memory_bytes": max(observed_peaks, default=0),
        "total_spathi_output_bytes": sum(
            int(row["inference"]["spathi_output_bytes"]) for row in completed_attempts
        ),
        "total_trees_fitted": sum(
            int(row["inference"]["trees_fitted"]) for row in completed_attempts
        ),
    }
    for field, expected in expected_integers.items():
        if summary.get(field) != expected:
            raise BenchmarkConfigError(f"Scalability summary has an inconsistent {field}")

    for row in completed_attempts:
        dimensions = require_mapping(row["dimensions"], "attempt dimensions")
        inference = require_mapping(row["inference"], "inference measurement")
        if inference["targets_requested"] != dimensions["targets"]:
            raise BenchmarkConfigError("Requested targets differ from the dataset dimensions")
        if inference["targets_eligible"] + inference["targets_ineligible"] != dimensions["targets"]:
            raise BenchmarkConfigError("Target-eligibility accounting is incomplete")
        if (
            inference["models_completed"] != inference["models_requested"]
            or inference["models_failed"]
        ):
            raise BenchmarkConfigError("Scalability results contain an incomplete SPATHI run")
    for row in failed_attempts:
        observations = require_mapping(row["observations"], "failed inference observations")
        measurement = require_mapping(
            observations.get("andrea_measurement"), "failed ANDREA measurement"
        )
        if measurement.get("scope") != "physical_task":
            raise BenchmarkConfigError("Failed scalability telemetry has the wrong scope")
        execution_measurement = require_mapping(
            observations.get("andrea_execution_measurement"),
            "failed ANDREA execution measurement",
        )
        if execution_measurement.get("scope") != "andrea_execution":
            raise BenchmarkConfigError(
                "Failed scalability execution telemetry has the wrong scope"
            )
    for row in rows:
        decisions = require_mapping(row.get("decisions"), "scalability decisions")
        for stage in ("scout", "all_targets"):
            decision = require_mapping(decisions.get(stage), f"{stage} decision")
            if (
                decision.get("dataset_id") != row.get("dataset_id")
                or decision.get("stage") != stage
            ):
                raise BenchmarkConfigError("Scalability decision identity is inconsistent")
            attempt = row.get(stage)
            if (attempt is None) != (decision.get("decision") == skipped_status):
                raise BenchmarkConfigError(
                    "Scalability decision and attempted-stage evidence disagree"
                )
            if attempt is not None and (
                attempt.get("dataset_id") != row.get("dataset_id")
                or attempt.get("stage") != stage
                or attempt.get("algorithm_seed") != row.get("algorithm_seed")
                or attempt.get("seed_derivation") != row.get("seed_derivation")
            ):
                raise BenchmarkConfigError("Scalability attempt identity is inconsistent")
            if (
                stage == "scout"
                and attempt is not None
                and (
                    attempt.get("target_scope")
                    != {
                        "method": "sha256_gene_id",
                        "size": scout_size,
                        "salt": "spathi-scalability-scout-v1",
                        "dataset_id": row.get("dataset_id"),
                    }
                    or attempt.get("dimensions", {}).get("targets") != scout_size
                )
            ):
                raise BenchmarkConfigError("Scalability scout scope is inconsistent")
            if (
                stage == "all_targets"
                and attempt is not None
                and attempt.get("target_scope") != {"method": "all"}
            ):
                raise BenchmarkConfigError("All-target scalability scope is inconsistent")
        all_targets = row.get("all_targets")
        if row.get("status") == skipped_status:
            if all_targets is not None:
                raise BenchmarkConfigError("A skipped endpoint contains an all-target attempt")
        elif not isinstance(all_targets, Mapping) or all_targets.get("status") != row.get("status"):
            raise BenchmarkConfigError("Endpoint status differs from its all-target attempt")
    return rows


def _validate_panel(panel: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    expected: dict[str, tuple[Any, ...]] = {}
    for raw in require_array(panel.get("datasets"), "scale-stress panel datasets"):
        dataset = require_mapping(raw, "scale-stress panel dataset")
        groups = require_array(dataset.get("groups"), "scale-stress groups")
        dataset_id = dataset.get("id")
        expected[str(dataset_id)] = (
            sum(int(require_mapping(group, "scale-stress group")["cells"]) for group in groups),
            dataset.get("genes"),
            len(groups),
            dataset.get("tfs"),
            dataset.get("genes"),
            dataset.get("seed"),
            dataset.get("network_mode"),
        )
    observed = {
        str(row["dataset_id"]): (
            row["dimensions"]["cells"],
            row["dimensions"]["genes"],
            row["dimensions"]["groups"],
            row["dimensions"]["transcription_factors"],
            row["dimensions"]["targets"],
            row["generation_seed"],
            row["network_mode"],
        )
        for row in rows
    }
    if observed != expected:
        raise BenchmarkConfigError("Scalability results differ from the frozen dimensional panel")


def _production_inputs(
    *, repo_root: Path, version_root: Path, production: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[Path]]:
    links: list[dict[str, Any]] = []
    inputs: list[Path] = []
    for label, raw in production_provenance_artifacts(dict(production)):
        reference = require_mapping(raw, f"production provenance {label}")
        source = repo_file(
            repo_root,
            reference.get("path"),
            f"production provenance {label}",
            below=version_root,
        )
        verify_sha256(source, reference.get("sha256"), f"production provenance {label}")
        links.append({"label": str(label).replace("_", " ").title(), "path": source})
        inputs.append(source)
    return links, inputs


def _validate_group_policy(policy: Mapping[str, Any]) -> None:
    group_policy = require_mapping(policy.get("group_policy"), "production group policy")
    branches = require_mapping(group_policy.get("branches"), "production policy branches")
    if not branches:
        raise BenchmarkConfigError("Production group policy has no executable branches")
    for branch_id, raw in branches.items():
        branch = require_mapping(raw, f"production policy branch {branch_id}")
        require_mapping(branch.get("condition"), f"condition for production branch {branch_id}")
        require_mapping(branch.get("parameters"), f"parameters for production branch {branch_id}")


def render_scalability_report(
    *, repo_root: Path, version_root: Path, workflow_root: Path
) -> dict[str, Any]:
    """Render a completed scale study without adding accuracy or biological claims."""

    root = repo_root.resolve()
    version = path_below(version_root, root / "work", "benchmark version root", file=False)
    scalability_root = path_below(
        workflow_root,
        version / "workflows/scalability",
        "scalability workflow",
        file=False,
    )
    workflow_path = scalability_root / "workflow.json"
    workflow = require_reportable_workflow(workflow_path, "scalability")
    profile_scope = workflow.get("profile")
    if profile_scope not in {"reduced", "full"} or scalability_root.parent.name != profile_scope:
        raise BenchmarkConfigError("Scalability workflow path disagrees with its profile")
    if scalability_root.name != workflow["flow_id"]:
        raise BenchmarkConfigError("Scalability workflow path disagrees with its flow ID")

    results_path = repo_file(
        root,
        workflow.get("results"),
        "scalability results",
        below=scalability_root,
    )
    verify_sha256(results_path, workflow.get("results_sha256"), "scalability results")
    results = load_json_object(results_path, "scalability results")
    validate_json(
        results,
        root / "schemas/scalability-results.schema.json",
        "scalability results",
    )
    if results.get("budget") != profile_scope or results.get("status") != "complete":
        raise BenchmarkConfigError("Scalability results disagree with their workflow")
    if results.get("claim_scope") != "computational_scale_stress_only":
        raise BenchmarkConfigError("Scalability results overstate their scientific claim")
    rows = _validate_summary(results)

    production_reference = require_mapping(
        results.get("production_config"), "scalability production policy reference"
    )
    production_path = repo_file(
        root,
        workflow.get("production_config"),
        "SPATHI production policy",
        below=version,
    )
    if production_reference.get("path") != production_path.relative_to(root).as_posix():
        raise BenchmarkConfigError("Workflow and results refer to different production policies")
    verify_sha256(
        production_path,
        workflow.get("production_config_sha256"),
        "SPATHI production policy",
    )
    verify_sha256(
        production_path,
        production_reference.get("sha256"),
        "results SPATHI production policy",
    )
    production = load_json_object(production_path, "SPATHI production policy")
    validate_json(
        production,
        root / "schemas/spathi-production-config.schema.json",
        "SPATHI production policy",
    )
    policy = require_mapping(production.get("policy"), "SPATHI production policy")
    _validate_group_policy(policy)
    if (
        production.get("budget") != profile_scope
        or production_reference.get("policy_id") != policy.get("policy_id")
        or production_reference.get("policy_sha256") != policy.get("policy_sha256")
        or production_reference.get("decision") != production.get("decision")
    ):
        raise BenchmarkConfigError("Scalability results change the frozen production policy")

    profile_reference = require_mapping(results.get("profile"), "scalability profile reference")
    profile_path = repo_file(root, profile_reference.get("path"), "scalability profile")
    verify_sha256(profile_path, profile_reference.get("sha256"), "scalability profile")
    profile = load_json_object(profile_path, "scalability profile")
    validate_json(profile, root / "schemas/profile.schema.json", "scalability profile")
    if (
        profile.get("id") != f"scalability-{profile_scope}"
        or profile.get("purpose") != "scalability"
    ):
        raise BenchmarkConfigError("Scalability results refer to an incompatible profile")
    runtime_plan = validate_runtime_plan(
        require_mapping(workflow.get("runtime_plan"), "scalability runtime plan")
    )
    execution_plan = resolve_execution_class(
        repo_root=root,
        class_id=str(profile.get("execution_class")),
        runtime_plan=runtime_plan,
    )
    inference_resources = require_mapping(
        require_mapping(execution_plan.get("operations"), "scalability execution plan").get(
            "inference"
        ),
        "scalability inference allocation",
    )
    expected_limits = {
        "cpu_threads": inference_resources["cpu_threads"],
        "memory_bytes": inference_resources["memory_bytes"],
        "memory_gib": float(inference_resources["memory_gib"]),
        "cpuset_cpus": sorted(
            execution_plan["detected"]["cpu_allocation_order"][: inference_resources["cpu_threads"]]
        ),
        "planner": inference_resources["planner"],
        "planner_time_limit_seconds": float(inference_resources["planner_time_limit_seconds"]),
        "output_profile": execution_plan["output_profile"],
        "exclusive": execution_plan["exclusive_inference"],
        "execution_plan_fingerprint": execution_plan["fingerprint"],
        "timeout_seconds": results["resource_boundary"][
            "prospective_wall_time_limit_seconds"
        ],
    }
    attempts = [
        attempt
        for row in rows
        for attempt in (row.get("scout"), row.get("all_targets"))
        if isinstance(attempt, Mapping)
    ]
    if any(attempt.get("execution_limits") != expected_limits for attempt in attempts):
        raise BenchmarkConfigError("Per-attempt scalability limits differ from the frozen profile")
    environment = require_mapping(
        results.get("execution_environment"), "scalability execution environment"
    )
    expected_environment_resources = {
        "cpu_threads": expected_limits["cpu_threads"],
        "memory_bytes": expected_limits["memory_bytes"],
        "memory_gib": expected_limits["memory_gib"],
        "cpuset_cpus": expected_limits["cpuset_cpus"],
        "planner": expected_limits["planner"],
        "planner_time_limit_seconds": expected_limits["planner_time_limit_seconds"],
        "output_profile": expected_limits["output_profile"],
        "exclusive_inference": expected_limits["exclusive"],
        "execution_plan_fingerprint": expected_limits["execution_plan_fingerprint"],
    }
    if any(environment.get(key) != value for key, value in expected_environment_resources.items()):
        raise BenchmarkConfigError("Scalability environment differs from its frozen execution plan")
    generation_jobs = require_array(profile.get("generation_jobs"), "scalability generation jobs")
    if len(generation_jobs) != 1:
        raise BenchmarkConfigError("Scalability profile must bind exactly one dimensional panel")
    generation_job = require_mapping(generation_jobs[0], "scalability generation job")
    panel_path = repo_file(root, generation_job.get("generator_request"), "scale-stress panel")
    panel = load_json_object(panel_path, "scale-stress scalability panel")
    validate_json(panel, root / "schemas/scale-stress-panel.schema.json", "scale-stress panel")
    if results.get("expression_preparation") != panel.get("expression_preparation"):
        raise BenchmarkConfigError(
            "Scalability results changed the frozen expression preparation contract"
        )
    _validate_panel(panel, rows)
    boundary_reference = require_mapping(
        results.get("resource_boundary"), "scalability boundary reference"
    )
    boundary_path = repo_file(
        root,
        boundary_reference.get("path"),
        "scalability resource boundary",
    )
    verify_sha256(
        boundary_path,
        boundary_reference.get("sha256"),
        "scalability resource boundary",
    )
    boundary = load_json_object(boundary_path, "scalability resource boundary")
    validate_json(
        boundary,
        root / "schemas/scalability-boundary.schema.json",
        "scalability resource boundary",
    )
    expected_boundary = {
        "path": boundary_path.relative_to(root).as_posix(),
        "sha256": boundary_reference["sha256"],
        "id": boundary["id"],
        "target_scout_size": boundary["target_scout"]["sizes"][profile_scope],
        "wall_time_measurement": boundary["resource_boundary"]["wall_time_measurement"],
        "prospective_wall_time_limit_seconds": boundary["resource_boundary"][
            "wall_time_limit_seconds"
        ][profile_scope],
        "maximum_projected_wall_time_fraction": boundary["resource_boundary"][
            "maximum_projected_wall_time_fraction"
        ],
        "maximum_observed_memory_fraction": boundary["resource_boundary"][
            "maximum_observed_memory_fraction"
        ],
    }
    if dict(boundary_reference) != expected_boundary:
        raise BenchmarkConfigError("Scalability results changed the preregistered boundary")
    policy_path = repo_file(root, POLICY_PATH.as_posix(), "execution policy")

    completed_rows = [row for row in attempts if row.get("status") == "completed"]
    failed_rows = [row for row in attempts if row.get("status") == "failed"]
    metadata_paths: list[Path] = []
    for row in completed_rows:
        reference = require_mapping(row.get("run_metadata"), "SPATHI run metadata reference")
        metadata_path = repo_file(
            root,
            reference.get("path"),
            f"SPATHI run metadata for {row['dataset_id']} ({row['stage']})",
            below=version,
        )
        verify_sha256(
            metadata_path,
            reference.get("sha256"),
            f"SPATHI run metadata for {row['dataset_id']} ({row['stage']})",
        )
        metadata_paths.append(metadata_path)

    failure_log_paths: list[Path] = []
    for row in failed_rows:
        diagnostics = require_mapping(row.get("diagnostics"), "failed-run diagnostics")
        reference = require_mapping(
            diagnostics.get("command_log"), "failed inference command log reference"
        )
        log_path = repo_file(
            root,
            reference.get("path"),
            f"failed inference command log for {row['dataset_id']} ({row['stage']})",
            below=version,
        )
        verify_sha256(
            log_path,
            reference.get("sha256"),
            f"failed inference command log for {row['dataset_id']} ({row['stage']})",
        )
        if reference.get("size_bytes") != log_path.stat().st_size:
            raise BenchmarkConfigError(
                f"Failed inference command log size changed for {row['dataset_id']}"
            )
        failure_log_paths.append(log_path)

    report_path = central_report_path(
        version_root=version,
        report_type="scalability",
        scope=str(profile_scope),
        flow_id=workflow["flow_id"],
    )
    if workflow.get("interactive_report") != report_path.relative_to(root).as_posix():
        raise BenchmarkConfigError("Workflow records a non-canonical scalability report path")

    provenance_links, provenance_inputs = _production_inputs(
        repo_root=root,
        version_root=version,
        production=production,
    )
    links = [
        {"label": "Workflow state", "path": workflow_path},
        {"label": "Scalability results", "path": results_path},
        {"label": "Scalability profile", "path": profile_path},
        {"label": "Frozen dimensional panel", "path": panel_path},
        {"label": "Preregistered resource boundary", "path": boundary_path},
        {"label": "Frozen production policy", "path": production_path},
        {"label": "Execution policy", "path": policy_path},
        *provenance_links,
        *[
            {
                "label": f"Run metadata · {row['dataset_id']} · {row['stage']}",
                "path": path,
            }
            for row, path in zip(completed_rows, metadata_paths, strict=True)
        ],
        *[
            {
                "label": f"Failure log · {row['dataset_id']} · {row['stage']}",
                "path": path,
            }
            for row, path in zip(failed_rows, failure_log_paths, strict=True)
        ],
    ]
    payload = {
        "schema_version": "1.0",
        "report_type": "scalability",
        "workflow": {
            "flow_id": workflow["flow_id"],
            "status": "complete",
            "profile": profile_scope,
            "evidence_role": results["evidence_role"],
            "runtime_plan": runtime_plan,
            "execution_plan": execution_plan,
            "identity": {
                **workflow["identity"],
                "spathi_commit": results["execution_environment"]["spathi_commit"],
                "andrea_commit": results["execution_environment"]["andrea_commit"],
            },
        },
        "scope": {
            "claim": results["claim_scope"],
            "interpretation": (
                "This phase measures computational completion and resource use at synthetic "
                "dimensional shapes from a public author-defined grid. It does not model "
                "application biology and does not evaluate "
                "network-inference accuracy."
            ),
            "expression_preparation": results["expression_preparation"],
            "resource_boundary": results["resource_boundary"],
        },
        "summary": results["summary"],
        "datasets": {
            "summary": _dataset_summary(rows),
            "rows": rows,
        },
        "production": {
            "budget": production["budget"],
            "evidence_role": production["evidence_role"],
            "status": production["status"],
            "decision": production["decision"],
            "policy": policy,
        },
        "environment": results["execution_environment"],
        "notes": results["notes"],
        "links": [
            {
                "label": link["label"],
                "href": safe_relative_link(report_path, link["path"], root),
            }
            for link in links
        ],
    }
    return reconcile_report_bundle(
        workflow,
        write_report_bundle(
            repo_root=root,
            kind="scalability",
            output_path=report_path,
            payload=payload,
            input_paths=[
                results_path,
                profile_path,
                panel_path,
                boundary_path,
                production_path,
                policy_path,
                *metadata_paths,
                *failure_log_paths,
                *provenance_inputs,
            ],
            manifest_name="scalability-report.manifest.json",
            immutable=True,
        ),
        "scalability",
    )


__all__ = ["render_scalability_report"]
