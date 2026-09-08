"""Interactive report for the SPATHI efficiency-justification workflow."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Any

from ..artifacts import sha256_file
from ..config import BenchmarkConfigError, load_json_object, validate_json
from ..design_sensitivity import load_design_sensitivity_contract
from ..efficiency_safety import validate_safety_plan
from ..resource_plan import validate_runtime_plan
from ..state import STATE_SCHEMA_VERSION
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
from .datasets import build_dataset_overview

_RESOURCE_FIELDS = (
    "wall_time_seconds",
    "peak_memory_bytes",
    "output_bytes",
    "targets_processed",
    "models_requested",
    "trees_fitted",
)


def _result_paths(
    *, repo_root: Path, workflow_root: Path, workflow: Mapping[str, Any]
) -> dict[str, Path]:
    expected = {"results.json", "comparisons.csv", "manifest.json"}
    resolved: dict[str, Path] = {}
    for raw_path in require_array(workflow.get("results"), "efficiency result files"):
        path = repo_file(
            repo_root,
            raw_path,
            "efficiency result file",
            below=workflow_root / "results",
        )
        if path.name not in expected or path.name in resolved:
            raise BenchmarkConfigError("Efficiency result files do not match the canonical set")
        resolved[path.name] = path
    if set(resolved) != expected:
        raise BenchmarkConfigError("Efficiency workflow lacks its canonical result files")
    return resolved


def _validate_result_manifest(paths: Mapping[str, Path], results: Mapping[str, Any]) -> None:
    manifest = load_json_object(paths["manifest.json"], "efficiency output manifest")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("artifact_type") != "spathi_efficiency_noninferiority_output_manifest"
        or manifest.get("inputs") != results.get("inputs")
        or manifest.get("analysis_implementation_sha256")
        != results.get("analysis_implementation_sha256")
    ):
        raise BenchmarkConfigError("Efficiency output manifest differs from its results")
    recorded = {
        item.get("path"): item.get("sha256")
        for item in require_array(manifest.get("files"), "efficiency output manifest files")
        if isinstance(item, dict)
    }
    if set(recorded) != {"results.json", "comparisons.csv"}:
        raise BenchmarkConfigError("Efficiency output manifest has an unexpected file set")
    for name, digest in recorded.items():
        verify_sha256(paths[name], digest, f"efficiency {name}")


def _validate_panel_state(
    panel: Mapping[str, Any], benchmark_state: Mapping[str, Any]
) -> dict[tuple[str, str], int]:
    panel_rows = require_array(panel.get("datasets"), "efficiency panel datasets")
    state_rows = require_array(benchmark_state.get("datasets"), "generated efficiency datasets")
    expected: dict[str, tuple[Any, ...]] = {}
    group_counts: dict[tuple[str, str], int] = {}
    role = panel.get("dataset_role")
    for raw in panel_rows:
        row = require_mapping(raw, "efficiency panel dataset")
        dataset_id = row.get("dataset_id")
        contexts = require_array(row.get("contexts"), f"contexts for {dataset_id}")
        artifact = require_mapping(row.get("dataset_artifact"), f"panel artifact for {dataset_id}")
        if not isinstance(dataset_id, str) or not dataset_id or not contexts:
            raise BenchmarkConfigError("Efficiency panel contains an incomplete dataset")
        expected[dataset_id] = (
            row.get("simulator_id"),
            row.get("condition_id"),
            row.get("generation_seed"),
            artifact.get("digest"),
        )
        group_counts[(str(role), dataset_id)] = len(contexts)

    observed: dict[str, tuple[Any, ...]] = {}
    for raw in state_rows:
        row = require_mapping(raw, "generated efficiency dataset")
        if row.get("dataset_role") != role:
            continue
        artifact = require_mapping(
            row.get("dataset_artifact"), f"generated artifact for {row.get('dataset_id')}"
        )
        observed[str(row.get("dataset_id"))] = (
            row.get("simulator_id"),
            row.get("condition_id"),
            row.get("seed"),
            artifact.get("digest"),
        )
    if observed != expected:
        raise BenchmarkConfigError("Generated datasets differ from the frozen efficiency panel")
    return group_counts


def _measurement_tuple(raw: Any, label: str) -> tuple[float | int, ...]:
    measurement = require_mapping(raw, label)
    values: list[float | int] = []
    for field in _RESOURCE_FIELDS:
        value = measurement.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise BenchmarkConfigError(f"{label} has an invalid {field}")
        values.append(value)
    return tuple(values)


def resource_summaries(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Summarize run-level resources without counting one run once per group context."""

    unique: dict[tuple[str, str, int, str], tuple[float | int, ...]] = {}
    for raw in require_array(evidence.get("rows"), "efficiency evidence rows"):
        row = require_mapping(raw, "efficiency evidence row")
        candidate_id = row.get("candidate_id")
        dataset_id = row.get("dataset_id")
        algorithm_seed = row.get("algorithm_seed")
        if (
            not isinstance(candidate_id, str)
            or not isinstance(dataset_id, str)
            or isinstance(algorithm_seed, bool)
            or not isinstance(algorithm_seed, int)
        ):
            raise BenchmarkConfigError("Efficiency evidence has an incomplete run identity")
        for side in ("reference", "candidate"):
            key = (candidate_id, dataset_id, algorithm_seed, side)
            measurement = _measurement_tuple(
                row.get(f"{side}_measurements"),
                f"{side} measurements for {candidate_id}/{dataset_id}/{algorithm_seed}",
            )
            previous = unique.setdefault(key, measurement)
            if previous != measurement:
                raise BenchmarkConfigError(
                    "One efficiency run has conflicting measurements across group contexts"
                )

    grouped: dict[tuple[str, str], list[tuple[float | int, ...]]] = defaultdict(list)
    for (candidate_id, _dataset_id, _seed, side), measurement in unique.items():
        grouped[(candidate_id, side)].append(measurement)
    candidate_ids = sorted({candidate_id for candidate_id, _side in grouped})
    summaries: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        record: dict[str, Any] = {"candidate_id": candidate_id}
        for side in ("reference", "candidate"):
            observations = grouped[(candidate_id, side)]
            if not observations:
                raise BenchmarkConfigError(
                    f"Efficiency evidence has no {side} runs for {candidate_id}"
                )
            record[f"{side}_run_count"] = len(observations)
            for index, field in enumerate(_RESOURCE_FIELDS):
                record[f"{side}_{field}_median"] = median(
                    observation[index] for observation in observations
                )
        summaries.append(record)
    return summaries


def _policies(plan: Mapping[str, Any], results: Mapping[str, Any]) -> list[dict[str, Any]]:
    comparisons = {
        item.get("candidate_id"): item
        for item in require_array(results.get("comparisons"), "efficiency comparisons")
        if isinstance(item, dict)
    }
    failed_ids = {
        item.get("configuration_id")
        for item in require_array(
            results.get("execution_failures"), "efficiency execution failures"
        )
        if isinstance(item, dict)
    }
    reference = require_mapping(plan.get("reference"), "exact efficiency policy")
    policies: list[dict[str, Any]] = [
        {
            "policy_id": reference.get("policy_id"),
            "role": "exact",
            "component": "exact",
            "rationale": (
                "Frozen calibrated policy with all requested targets and the fixed calibrated "
                "tree count. It is the paired baseline for every efficiency candidate."
            ),
            "policy_sha256": reference.get("policy_sha256"),
            "execution_overrides": {},
            "group_policy": reference.get("group_policy"),
            "decision": "paired_baseline",
        }
    ]
    for raw in require_array(plan.get("candidates"), "efficiency policies"):
        candidate = require_mapping(raw, "efficiency policy")
        candidate_id = candidate.get("id")
        result = comparisons.get(candidate_id)
        if result is None and candidate_id not in failed_ids:
            raise BenchmarkConfigError(f"Efficiency results omit candidate {candidate_id!r}")
        policies.append(
            {
                "policy_id": candidate_id,
                "role": "candidate",
                "component": candidate.get("component"),
                "rationale": candidate.get("rationale"),
                "policy_sha256": candidate.get("policy_sha256"),
                "execution_overrides": candidate.get("execution_overrides"),
                "group_policy": candidate.get("group_policy"),
                "decision": ("execution_failed" if result is None else result.get("decision")),
            }
        )
    if set(comparisons) | failed_ids != {policy["policy_id"] for policy in policies[1:]}:
        raise BenchmarkConfigError("Efficiency plan and result candidate sets differ")
    return policies


def _portable_provenance(
    *, production: Mapping[str, Any], plan: Mapping[str, Any], results: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "study": plan.get("study"),
        "design_sensitivity": plan.get("design_sensitivity"),
        "panel": plan.get("panel"),
        "selected_config": require_mapping(plan.get("reference"), "efficiency reference").get(
            "selected_config"
        ),
        "spathi_commit": require_mapping(plan.get("reference"), "efficiency reference").get(
            "spathi_commit"
        ),
        "spathi_image_id": require_mapping(plan.get("reference"), "efficiency reference").get(
            "spathi_image_id"
        ),
        "analysis_implementation_sha256": results.get("analysis_implementation_sha256"),
        "production_policy": production.get("policy"),
        "production_evidence": production.get("provenance"),
    }


def _expected_reference(repo_root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(repo_root).as_posix(), "sha256": sha256_file(path)}


def _validate_production_transition(
    *,
    adoption: Mapping[str, Any],
    results: Mapping[str, Any],
    production: Mapping[str, Any],
    safety: Mapping[str, Any] | None,
) -> None:
    """Validate formal provisional selection and the optional final safety veto."""

    production_policy = require_mapping(
        production.get("policy"), "SPATHI production policy parameters"
    )
    if production.get("evidence_role") != results.get("analysis_role"):
        raise BenchmarkConfigError("Production policy uses a different efficiency evidence role")
    decision = adoption
    expected_decision = adoption.get("decision")
    if safety is not None:
        if adoption.get("decision") != "candidate_adopted":
            raise BenchmarkConfigError(
                "Efficiency safety has no formally adopted provisional candidate"
            )
        if safety.get("provisional_candidate_id") != adoption.get(
            "selected_candidate_id"
        ) or safety.get("provisional_candidate_policy_sha256") != adoption.get(
            "selected_policy_sha256"
        ):
            raise BenchmarkConfigError(
                "Efficiency safety evaluates a different provisional candidate"
            )
        decision = require_mapping(safety.get("decision"), "efficiency safety decision")
        expected_decision = (
            "candidate_adopted"
            if decision.get("status") == "candidate_authorized"
            else "reference_retained"
        )
    expected = {
        "policy_id": decision.get("selected_policy_id"),
        "policy_sha256": decision.get("selected_policy_sha256"),
        "group_policy": decision.get("selected_group_policy"),
    }
    if production.get("decision") != expected_decision or any(
        production_policy.get(key) != value for key, value in expected.items()
    ):
        raise BenchmarkConfigError("Production policy differs from the final efficiency decision")


def render_efficiency_report(
    *, repo_root: Path, version_root: Path, workflow_root: Path
) -> dict[str, Any]:
    """Render one completed efficiency study without recomputing its statistics."""

    root = repo_root.resolve()
    version = path_below(version_root, root / "work", "benchmark version root", file=False)
    efficiency_root = path_below(
        workflow_root,
        version / "workflows/efficiency",
        "efficiency workflow",
        file=False,
    )
    workflow_path = efficiency_root / "workflow.json"
    workflow = require_reportable_workflow(workflow_path, "efficiency")
    profile = workflow.get("profile")
    if profile not in {"reduced", "full"} or efficiency_root.parent.name != profile:
        raise BenchmarkConfigError("Efficiency workflow path disagrees with its profile")
    if efficiency_root.name != workflow["flow_id"]:
        raise BenchmarkConfigError("Efficiency workflow path disagrees with its flow ID")

    panel_path = repo_file(root, workflow.get("panel"), "efficiency panel", below=efficiency_root)
    plan_path = repo_file(root, workflow.get("plan"), "efficiency plan", below=efficiency_root)
    evidence_path = repo_file(
        root, workflow.get("evidence"), "efficiency evidence", below=efficiency_root
    )
    production_path = repo_file(
        root,
        workflow.get("production_config"),
        "SPATHI production policy",
        below=efficiency_root,
    )
    verify_sha256(
        production_path,
        workflow.get("production_config_sha256"),
        "SPATHI production policy",
    )
    result_paths = _result_paths(repo_root=root, workflow_root=efficiency_root, workflow=workflow)

    panel = load_json_object(panel_path, "efficiency panel")
    plan = load_json_object(plan_path, "efficiency plan")
    evidence = load_json_object(evidence_path, "efficiency evidence")
    results = load_json_object(result_paths["results.json"], "efficiency results")
    production = load_json_object(production_path, "SPATHI production policy")
    for payload, schema, label in (
        (panel, "noninferiority-panel.schema.json", "efficiency panel"),
        (plan, "noninferiority-plan.schema.json", "efficiency plan"),
        (evidence, "noninferiority-evidence.schema.json", "efficiency evidence"),
        (results, "noninferiority-results.schema.json", "efficiency results"),
        (production, "spathi-production-config.schema.json", "SPATHI production policy"),
    ):
        validate_json(payload, root / "schemas" / schema, label)

    if (
        panel.get("budget") != profile
        or plan.get("budget", {}).get("id") != profile
        or evidence.get("budget") != profile
        or results.get("budget") != profile
        or production.get("budget") != profile
    ):
        raise BenchmarkConfigError("Efficiency artifacts disagree on their evidence budget")
    if plan["panel"] != {
        "path": panel_path.relative_to(root).as_posix(),
        "sha256": sha256_file(panel_path),
    }:
        raise BenchmarkConfigError("Efficiency plan does not bind the recorded panel")
    if (
        evidence.get("plan_sha256") != sha256_file(plan_path)
        or evidence.get("panel_sha256") != sha256_file(panel_path)
        or results.get("inputs")
        != {
            "plan_sha256": sha256_file(plan_path),
            "evidence_sha256": sha256_file(evidence_path),
        }
    ):
        raise BenchmarkConfigError("Efficiency evidence chain is inconsistent")
    if any(results["analysis"].get(key) != value for key, value in plan["analysis"].items()):
        raise BenchmarkConfigError("Efficiency results change the predeclared analysis")
    _validate_result_manifest(result_paths, results)

    artifact_index_reference = require_mapping(
        evidence.get("artifact_index"), "efficiency artifact index reference"
    )
    artifact_index_path = repo_file(
        root,
        artifact_index_reference.get("path"),
        "efficiency artifact index",
        below=efficiency_root,
    )
    verify_sha256(
        artifact_index_path,
        artifact_index_reference.get("sha256"),
        "efficiency artifact index",
    )
    if artifact_index_path != repo_file(
        root,
        workflow.get("artifact_index"),
        "workflow efficiency artifact index",
        below=efficiency_root,
    ):
        raise BenchmarkConfigError("Workflow and evidence refer to different artifact indexes")

    study_reference = require_mapping(plan.get("study"), "efficiency study reference")
    study_path = repo_file(root, study_reference.get("path"), "efficiency study")
    verify_sha256(study_path, study_reference.get("sha256"), "efficiency study")
    sensitivity_reference = require_mapping(
        plan.get("design_sensitivity"), "efficiency design-sensitivity reference"
    )
    sensitivity_path = repo_file(
        root,
        sensitivity_reference.get("path"),
        "efficiency design-sensitivity contract",
    )
    verify_sha256(
        sensitivity_path,
        sensitivity_reference.get("sha256"),
        "efficiency design-sensitivity contract",
    )
    sensitivity_contract, sensitivity_results = load_design_sensitivity_contract(root)
    selected_reference = require_mapping(
        require_mapping(plan.get("reference"), "efficiency reference").get("selected_config"),
        "calibration selection reference",
    )
    selected_path = repo_file(
        root,
        selected_reference.get("path"),
        "frozen calibration selection",
        below=version / "workflows/calibration",
    )
    verify_sha256(
        selected_path,
        selected_reference.get("sha256"),
        "frozen calibration selection",
    )
    workflow_selected = repo_file(
        root,
        workflow.get("selected_config"),
        "workflow calibration selection",
        below=version / "workflows/calibration",
    )
    if workflow_selected != selected_path:
        raise BenchmarkConfigError("Workflow and plan refer to different calibration selections")

    adoption = require_mapping(results.get("adoption"), "efficiency adoption decision")
    production_provenance = require_mapping(
        production.get("provenance"), "production policy provenance"
    )
    expected_base_provenance = {
        "calibration_selection": _expected_reference(root, selected_path),
        "efficiency_plan": _expected_reference(root, plan_path),
        "efficiency_evidence": _expected_reference(root, evidence_path),
        "efficiency_results": _expected_reference(root, result_paths["results.json"]),
    }
    for label, expected_reference in expected_base_provenance.items():
        if production_provenance.get(label) != expected_reference:
            raise BenchmarkConfigError(f"Production provenance differs for {label}")

    safety_payload: dict[str, Any] | None = None
    safety_paths: list[Path] = []
    safety_dataset_inputs: list[Path] = []
    safety_state = require_mapping(
        workflow.get("efficiency_safety"), "efficiency safety workflow state"
    )
    if safety_state.get("status") == "completed":
        safety_generation_raw = safety_state.get("generation_run")
        if not isinstance(safety_generation_raw, str) or not safety_generation_raw:
            raise BenchmarkConfigError("Completed efficiency safety has no generation run")
        safety_generation_candidate = Path(safety_generation_raw)
        if not safety_generation_candidate.is_absolute():
            safety_generation_candidate = root / safety_generation_candidate
        safety_generation_root = path_below(
            safety_generation_candidate,
            version / "efficiency-safety-full",
            "efficiency safety generation run",
            file=False,
        )
        if safety_generation_root.name != workflow["flow_id"]:
            raise BenchmarkConfigError(
                "Efficiency safety generation run disagrees with the workflow flow ID"
            )
        safety_generation_state_path = safety_generation_root / "state.json"
        safety_generation_state = load_json_object(
            safety_generation_state_path,
            "efficiency safety generation benchmark state",
        )
        if (
            safety_generation_state.get("schema_version") != STATE_SCHEMA_VERSION
            or safety_generation_state.get("status") != "generated"
            or safety_generation_state.get("profile_id") != "efficiency-safety-full"
        ):
            raise BenchmarkConfigError(
                "Efficiency safety generation state is not complete for its profile"
            )
        safety_root = efficiency_root / "efficiency-safety"
        resolved: dict[str, Path] = {}
        for label in ("panel", "plan", "artifact_index", "evidence", "results"):
            resolved[label] = repo_file(
                root,
                safety_state.get(label),
                f"efficiency safety {label}",
                below=safety_root,
            )
        safety_panel = load_json_object(resolved["panel"], "efficiency safety panel")
        safety_plan = load_json_object(resolved["plan"], "efficiency safety plan")
        safety_evidence = load_json_object(resolved["evidence"], "efficiency safety evidence")
        safety_results = load_json_object(resolved["results"], "efficiency safety results")
        for payload, schema, label in (
            (
                safety_panel,
                "efficiency-safety-panel.schema.json",
                "efficiency safety panel",
            ),
            (
                safety_plan,
                "efficiency-safety-plan.schema.json",
                "efficiency safety plan",
            ),
            (
                safety_evidence,
                "noninferiority-evidence.schema.json",
                "efficiency safety evidence",
            ),
            (
                safety_results,
                "efficiency-safety-results.schema.json",
                "efficiency safety results",
            ),
        ):
            validate_json(payload, root / "schemas" / schema, label)
        validate_safety_plan(root, safety_plan)
        safety_group_counts = _validate_panel_state(safety_panel, safety_generation_state)
        safety_datasets, safety_dataset_inputs = build_dataset_overview(
            repo_root=root,
            version_root=version,
            benchmark_state=safety_generation_state,
            group_counts=safety_group_counts,
        )
        reference_image_id = require_mapping(plan.get("reference"), "efficiency reference").get(
            "spathi_image_id"
        )
        if (
            require_mapping(
                safety_generation_state.get("spathi_image"),
                "efficiency safety SPATHI image",
            ).get("id")
            != reference_image_id
        ):
            raise BenchmarkConfigError(
                "Efficiency safety used a different SPATHI image from calibration"
            )
        if (
            safety_plan.get("panel") != _expected_reference(root, resolved["panel"])
            or safety_plan.get("source_efficiency", {}).get("plan")
            != _expected_reference(root, plan_path)
            or safety_plan.get("source_efficiency", {}).get("results")
            != _expected_reference(root, result_paths["results.json"])
            or safety_evidence.get("plan_sha256") != sha256_file(resolved["plan"])
            or safety_evidence.get("panel_sha256") != sha256_file(resolved["panel"])
            or safety_evidence.get("artifact_index")
            != _expected_reference(root, resolved["artifact_index"])
            or safety_results.get("inputs")
            != {
                "plan_sha256": sha256_file(resolved["plan"]),
                "evidence_sha256": sha256_file(resolved["evidence"]),
            }
            or safety_results.get("condition_scales")
            != safety_plan.get("analysis", {}).get("condition_scales")
            or safety_results.get("thresholds") != safety_plan.get("analysis", {}).get("thresholds")
        ):
            raise BenchmarkConfigError("Efficiency safety evidence chain is inconsistent")
        nested = production_provenance.get("efficiency_safety")
        expected_nested = {
            label: _expected_reference(root, resolved[label])
            for label in ("panel", "plan", "evidence", "results")
        }
        if nested != expected_nested:
            raise BenchmarkConfigError("Production policy has stale efficiency safety provenance")
        safety_payload = {
            "status": "completed",
            "datasets": safety_datasets,
            "condition_scales": safety_panel["condition_scales"],
            "thresholds": safety_results["thresholds"],
            "paired_datasets": safety_results["paired_datasets"],
            "execution_failures": safety_results["execution_failures"],
            "summary": safety_results["summary"],
            "gates": safety_results["gates"],
            "decision": safety_results["decision"],
            "interpretation_limit": safety_results["interpretation_limit"],
        }
        safety_paths = [safety_generation_state_path, *resolved.values()]
    elif safety_state.get("status") == "not_required":
        if production_provenance.get("efficiency_safety") is not None:
            raise BenchmarkConfigError("Production policy records unexecuted efficiency safety")
        if profile == "full" and adoption.get("decision") == "candidate_adopted":
            raise BenchmarkConfigError("Full provisional candidate bypassed efficiency safety")
        safety_payload = {
            "status": "not_required",
            "reason": safety_state.get("reason"),
        }
    else:
        raise BenchmarkConfigError("Efficiency safety workflow state is invalid")
    _validate_production_transition(
        adoption=adoption,
        results=results,
        production=production,
        safety=(safety_results if safety_state.get("status") == "completed" else None),
    )

    generation_root_raw = workflow.get("generation_run")
    if not isinstance(generation_root_raw, str) or not generation_root_raw:
        raise BenchmarkConfigError("Efficiency workflow has no generation run")
    generation_candidate = Path(generation_root_raw)
    if not generation_candidate.is_absolute():
        generation_candidate = root / generation_candidate
    generation_root = path_below(
        generation_candidate,
        version,
        "efficiency generation run",
        file=False,
    )
    generation_state_path = generation_root / "state.json"
    benchmark_state = load_json_object(
        generation_state_path, "efficiency generation benchmark state"
    )
    if (
        benchmark_state.get("schema_version") != STATE_SCHEMA_VERSION
        or benchmark_state.get("status") != "generated"
        or benchmark_state.get("profile_id") != f"efficiency-{profile}"
    ):
        raise BenchmarkConfigError("Efficiency generation state is not complete for its profile")
    group_counts = _validate_panel_state(panel, benchmark_state)
    datasets, dataset_inputs = build_dataset_overview(
        repo_root=root,
        version_root=version,
        benchmark_state=benchmark_state,
        group_counts=group_counts,
    )

    report_path = central_report_path(
        version_root=version,
        report_type="efficiency",
        scope=profile,
        flow_id=workflow["flow_id"],
    )
    if workflow.get("interactive_report") != report_path.relative_to(root).as_posix():
        raise BenchmarkConfigError("Workflow records a non-canonical efficiency report path")
    links = [
        {"label": "Workflow state", "path": workflow_path},
        {"label": "Study contract", "path": study_path},
        {"label": "Prospective design sensitivity", "path": sensitivity_path},
        {"label": "Frozen panel", "path": panel_path},
        {"label": "Materialized plan", "path": plan_path},
        {"label": "Paired evidence", "path": evidence_path},
        {"label": "Analysis results", "path": result_paths["results.json"]},
        {"label": "Comparison table", "path": result_paths["comparisons.csv"]},
        {"label": "Production policy", "path": production_path},
        {"label": "Artifact index", "path": artifact_index_path},
        {"label": "Frozen calibration selection", "path": selected_path},
        *[
            {
                "label": f"Efficiency safety {path.stem.replace('-', ' ')}",
                "path": path,
            }
            for path in safety_paths
        ],
    ]
    all_resource_rows = resource_summaries(evidence)
    result_candidate_ids = {item["candidate_id"] for item in results["comparisons"]}
    resource_rows = [
        item for item in all_resource_rows if item["candidate_id"] in result_candidate_ids
    ]
    if {item["candidate_id"] for item in resource_rows} != result_candidate_ids:
        raise BenchmarkConfigError("Efficiency resource evidence and result candidates differ")
    payload = {
        "schema_version": "1.0",
        "report_type": "efficiency",
        "workflow": {
            "flow_id": workflow["flow_id"],
            "status": "complete",
            "profile": profile,
            "analysis_role": results["analysis_role"],
            "runtime_plan": validate_runtime_plan(
                require_mapping(workflow.get("runtime_plan"), "efficiency runtime plan")
            ),
            "identity": workflow["identity"],
        },
        "datasets": datasets,
        "study": {
            "id": results["study_id"],
            "purpose": load_json_object(study_path, "efficiency study").get("purpose"),
            "analysis": results["analysis"],
            "interpretation_limit": results["interpretation_limit"],
        },
        "design_sensitivity": {
            "interpretation": sensitivity_contract["interpretation"],
            "required_caveat": sensitivity_contract["reporting"]["required_caveat"],
            "expected_independent_datasets": sensitivity_contract["target_design"][
                "expected_independent_datasets"
            ],
            "expected_effective_sample_size": sensitivity_results["effective_sample_size"],
            "operating_characteristics_sha256": sensitivity_results[
                "operating_characteristics_sha256"
            ],
        },
        "policies": _policies(plan, results),
        "comparisons": results["comparisons"],
        "execution_failures": results["execution_failures"],
        "resources": resource_rows,
        "decision": {
            "global": results["global_decision"],
            "adoption": results["adoption"],
            "production": {
                "status": production["status"],
                "decision": production["decision"],
                "policy": production["policy"],
            },
            "efficiency_safety": safety_payload,
        },
        "evidence": {
            "dataset_role": evidence["dataset_role"],
            "paired_row_count": len(evidence["rows"]),
            "candidate_count": len(plan["candidates"]),
            "evaluable_candidate_count": len(results["comparisons"]),
            "execution_failure_count": len(results["execution_failures"]),
            "algorithm_seed_count": len(plan["budget"]["algorithm_seeds"]),
            "algorithm_seeds": plan["budget"]["algorithm_seeds"],
        },
        "provenance": _portable_provenance(production=production, plan=plan, results=results),
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
            kind="efficiency",
            output_path=report_path,
            payload=payload,
            input_paths=[
                generation_state_path,
                study_path,
                sensitivity_path,
                panel_path,
                plan_path,
                evidence_path,
                artifact_index_path,
                selected_path,
                production_path,
                *result_paths.values(),
                *safety_paths,
                *safety_dataset_inputs,
                *dataset_inputs,
            ],
            manifest_name="efficiency-report.manifest.json",
            immutable=True,
        ),
        "efficiency",
    )


__all__ = ["render_efficiency_report", "resource_summaries"]
