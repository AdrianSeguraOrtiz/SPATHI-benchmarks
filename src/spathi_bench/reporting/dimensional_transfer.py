"""Interactive report for post-freeze dimensional accuracy transfer."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean, median
from typing import Any

from ..config import BenchmarkConfigError, load_json_object, validate_json
from ..dimensional_transfer import condition_ids_for_budget, validate_transfer_panel
from ..execution_policy import POLICY_PATH, resolve_execution_class
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


def _close(observed: Any, expected: float) -> bool:
    return (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
    )


def _validate_results(results: Mapping[str, Any], panel: Mapping[str, Any]) -> None:
    datasets = [
        require_mapping(item, "dimensional-transfer dataset")
        for item in require_array(results.get("datasets"), "dimensional-transfer datasets")
    ]
    conditions = [
        require_mapping(item, "dimensional-transfer condition")
        for item in require_array(results.get("conditions"), "dimensional-transfer conditions")
    ]
    budget = str(results.get("budget"))
    active_condition_ids = set(condition_ids_for_budget(budget))
    expected_dimensions = {
        str(item["id"]): (
            int(item["genes"]),
            int(item["transcription_factors"]),
            int(item["cells"]),
            str(item["analysis_role"]),
            int(item["expected_group_contexts"]),
        )
        for item in panel["conditions"]
        if str(item["id"]) in active_condition_ids
    }
    if len({row.get("dataset_id") for row in datasets}) != len(datasets):
        raise BenchmarkConfigError("Dimensional-transfer results repeat a dataset")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in datasets:
        condition_id = str(row.get("condition_id"))
        if (
            condition_id not in expected_dimensions
            or (
                row.get("genes"),
                row.get("transcription_factors"),
                row.get("cells"),
            )
            != expected_dimensions[condition_id][:3]
            or row.get("analysis_role") != expected_dimensions[condition_id][3]
            or row.get("contexts") != expected_dimensions[condition_id][4]
        ):
            raise BenchmarkConfigError("Dimensional-transfer dataset differs from its panel")
        grouped[condition_id].append(row)
    if set(grouped) != set(expected_dimensions):
        raise BenchmarkConfigError("Dimensional-transfer condition coverage is incomplete")

    condition_by_id = {str(row.get("id")): row for row in conditions}
    if set(condition_by_id) != set(expected_dimensions):
        raise BenchmarkConfigError("Dimensional-transfer summaries are incomplete")
    mean_fields = (
        "directed_aupr",
        "adjusted_aupr",
        "f1_at_truth_count",
        "epr_at_truth_count",
        "candidate_prevalence",
    )
    for condition_id, rows in grouped.items():
        aggregate = condition_by_id[condition_id]
        expected_contexts = expected_dimensions[condition_id][4]
        if (
            aggregate.get("analysis_role") != expected_dimensions[condition_id][3]
            or aggregate.get("replicates") != len(rows)
            or aggregate.get("contexts") != len(rows) * expected_contexts
        ):
            raise BenchmarkConfigError("Dimensional-transfer replicate accounting is inconsistent")
        for field in mean_fields:
            if not _close(aggregate.get(field), fmean(float(row[field]) for row in rows)):
                raise BenchmarkConfigError(f"Dimensional-transfer {field} aggregate is stale")
        aurocs = [float(row["directed_auroc"]) for row in rows if row["directed_auroc"] is not None]
        expected_auroc = fmean(aurocs) if aurocs else None
        if expected_auroc is None:
            if aggregate.get("directed_auroc") is not None:
                raise BenchmarkConfigError("Dimensional-transfer AUROC aggregate is stale")
        elif not _close(aggregate.get("directed_auroc"), expected_auroc):
            raise BenchmarkConfigError("Dimensional-transfer AUROC aggregate is stale")
        if not _close(
            aggregate.get("median_wall_time_seconds"),
            median(float(row["wall_time_seconds"]) for row in rows),
        ):
            raise BenchmarkConfigError("Dimensional-transfer runtime aggregate is stale")

    expected_contrasts = (
        {
            "gene_scale_contrast": ("g3000_tf100_c300", "g15000_tf100_c300"),
            "tf_scale_contrast": ("g3000_tf100_c300", "g3000_tf1500_c300"),
            "cell_scale_contrast": ("g100_tf10_c300", "g100_tf10_c10000"),
        }
        if budget == "full"
        else {}
    )
    contrasts = {
        str(row.get("id")): row
        for row in (
            require_mapping(item, "dimensional-transfer contrast")
            for item in require_array(results.get("contrasts"), "dimensional-transfer contrasts")
        )
    }
    if set(contrasts) != set(expected_contrasts):
        raise BenchmarkConfigError("Dimensional-transfer contrasts are incomplete")
    for contrast_id, (left, right) in expected_contrasts.items():
        expected = float(condition_by_id[right]["adjusted_aupr"]) - float(
            condition_by_id[left]["adjusted_aupr"]
        )
        if not _close(contrasts[contrast_id].get("estimate"), expected):
            raise BenchmarkConfigError("Dimensional-transfer contrast is stale")



def _export_links(
    *, run_root: Path, report_path: Path, repo_root: Path
) -> tuple[list[dict[str, str]], list[Path]]:
    state_path = run_root / "state.json"
    state = load_json_object(state_path, "dimensional-transfer pipeline state")
    if state.get("purpose") != "dimensional_transfer" or state.get("status") != "completed":
        raise BenchmarkConfigError("Dimensional-transfer pipeline is not complete")
    exports_path = repo_file(
        repo_root,
        state.get("exports_manifest"),
        "dimensional-transfer exports manifest",
        below=run_root,
    )
    exports = load_json_object(exports_path, "dimensional-transfer exports manifest")
    if exports.get("status") != "complete":
        raise BenchmarkConfigError("Dimensional-transfer exports are incomplete")
    exports_root = exports_path.parent
    links: list[dict[str, str]] = [
        {
            "label": "ANDREA inter-stage export manifest",
            "href": safe_relative_link(report_path, exports_path, repo_root),
        }
    ]
    inputs = [state_path, exports_path]
    for raw in require_array(exports.get("bundles"), "dimensional-transfer export bundles"):
        bundle = require_mapping(raw, "dimensional-transfer export bundle")
        if bundle.get("scope") not in {"dataset", "joint"}:
            continue
        relative = Path(str(bundle.get("analysis_zip", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise BenchmarkConfigError("Dimensional-transfer export path is unsafe")
        archive = path_below(
            exports_root / relative,
            exports_root,
            "dimensional-transfer ANDREA analysis ZIP",
        )
        verify_sha256(
            archive,
            bundle.get("archive_sha256"),
            "dimensional-transfer ANDREA analysis ZIP",
        )
        producer = str(bundle.get("producer_command"))
        dataset = str(bundle.get("dataset_id"))
        links.append(
            {
                "label": f"{producer} · {dataset}",
                "href": safe_relative_link(report_path, archive, repo_root),
            }
        )
    return links, inputs


def render_dimensional_transfer_report(
    *, repo_root: Path, version_root: Path, workflow_root: Path
) -> dict[str, Any]:
    """Render descriptive one-axis transfer without a ranking or joint-scale claim."""

    root = repo_root.resolve()
    version = path_below(version_root, root / "work", "benchmark version root", file=False)
    transfer_root = path_below(
        workflow_root,
        version / "workflows/dimensional_transfer",
        "dimensional-transfer workflow",
        file=False,
    )
    workflow_path = transfer_root / "workflow.json"
    workflow = require_reportable_workflow(workflow_path, "dimensional_transfer")
    budget = workflow.get("profile")
    if budget not in {"reduced", "full"} or transfer_root.parent.name != budget:
        raise BenchmarkConfigError("Dimensional-transfer workflow path disagrees with its budget")
    if transfer_root.name != workflow["flow_id"]:
        raise BenchmarkConfigError("Dimensional-transfer workflow path disagrees with its flow ID")

    results_path = repo_file(
        root, workflow.get("results"), "dimensional-transfer results", below=transfer_root
    )
    verify_sha256(results_path, workflow.get("results_sha256"), "dimensional-transfer results")
    results = load_json_object(results_path, "dimensional-transfer results")
    validate_json(
        results,
        root / "schemas/dimensional-transfer-results.schema.json",
        "dimensional-transfer results",
    )
    if (
        results.get("budget") != budget
        or results.get("status") != "complete"
        or results.get("claim_scope") != "truth_bearing_synthetic_dimensional_transfer"
    ):
        raise BenchmarkConfigError("Dimensional-transfer results overstate or change their scope")

    panel_path = repo_file(root, workflow.get("panel"), "dimensional-transfer panel")
    verify_sha256(panel_path, workflow.get("panel_sha256"), "dimensional-transfer panel")
    panel = validate_transfer_panel(root, panel_path)
    _validate_results(results, panel)

    selection_path = repo_file(
        root,
        workflow.get("selected_config"),
        "dimensional-transfer selection policy",
        below=version,
    )
    verify_sha256(
        selection_path,
        workflow.get("selected_config_sha256"),
        "dimensional-transfer selection policy",
    )
    selection = load_json_object(selection_path, "dimensional-transfer selection policy")
    validate_json(
        selection,
        root / "schemas/spathi-selected-config.schema.json",
        "dimensional-transfer selection policy",
    )
    policy_reference = require_mapping(
        results.get("selected_config"), "results selection policy reference"
    )
    policy = require_mapping(selection.get("selected_policy"), "SPATHI selection policy")
    if (
        selection.get("calibration_budget") != budget
        or policy_reference.get("sha256") != workflow.get("selected_config_sha256")
        or policy_reference.get("policy_id") != policy.get("policy_id")
        or policy_reference.get("policy_sha256") != policy.get("policy_sha256")
    ):
        raise BenchmarkConfigError("Dimensional-transfer run changed its frozen SPATHI policy")

    run_root = path_below(
        root / str(workflow.get("run_root")),
        version,
        "dimensional-transfer pipeline root",
        file=False,
    )
    report_path = central_report_path(
        version_root=version,
        report_type="dimensional_transfer",
        scope=str(budget),
        flow_id=workflow["flow_id"],
    )
    if workflow.get("interactive_report") != report_path.relative_to(root).as_posix():
        raise BenchmarkConfigError("Workflow records a non-canonical transfer report path")
    export_links, export_inputs = _export_links(
        run_root=run_root, report_path=report_path, repo_root=root
    )
    summary_reference = require_mapping(
        require_mapping(results.get("summary_inputs"), "dimensional-transfer summary inputs").get(
            "summary_json"
        ),
        "dimensional-transfer pipeline summary",
    )
    summary_path = repo_file(
        root,
        summary_reference.get("path"),
        "dimensional-transfer pipeline summary",
        below=run_root,
    )
    verify_sha256(
        summary_path,
        summary_reference.get("sha256"),
        "dimensional-transfer pipeline summary",
    )
    pipeline_summary = load_json_object(summary_path, "dimensional-transfer pipeline summary")

    template_path = repo_file(root, workflow.get("template_profile"), "transfer profile")
    runtime_profile_path = repo_file(
        root,
        workflow.get("runtime_profile"),
        "transfer runtime profile",
        below=transfer_root,
    )
    verify_sha256(template_path, workflow.get("template_profile_sha256"), "transfer profile")
    verify_sha256(
        runtime_profile_path,
        workflow.get("runtime_profile_sha256"),
        "transfer runtime profile",
    )
    runtime_profile = load_json_object(runtime_profile_path, "transfer runtime profile")
    validate_json(runtime_profile, root / "schemas/profile.schema.json", "transfer runtime profile")
    runtime_plan = validate_runtime_plan(
        require_mapping(workflow.get("runtime_plan"), "dimensional-transfer runtime plan")
    )
    execution_plan = resolve_execution_class(
        repo_root=root,
        class_id=str(runtime_profile.get("execution_class")),
        runtime_plan=runtime_plan,
    )
    policy_path = repo_file(root, POLICY_PATH.as_posix(), "execution policy")
    links = [
        {"label": "Workflow state", "path": workflow_path},
        {"label": "Dimensional-transfer results", "path": results_path},
        {"label": "Frozen dimensional panel", "path": panel_path},
        {"label": "Template profile", "path": template_path},
        {"label": "Runtime profile", "path": runtime_profile_path},
        {"label": "Frozen selection policy", "path": selection_path},
        {"label": "Execution policy", "path": policy_path},
    ]
    payload = {
        "schema_version": "1.0",
        "report_type": "dimensional_transfer",
        "workflow": {
            "flow_id": workflow["flow_id"],
            "status": "complete",
            "profile": budget,
            "evidence_role": results["evidence_role"],
            "runtime_plan": runtime_plan,
            "execution_plan": execution_plan,
            "scheduling": require_mapping(
                pipeline_summary.get("scheduling"),
                "dimensional-transfer scheduling summary",
            ),
            "identity": workflow["identity"],
        },
        "scope": {
            "claim": results["claim_scope"],
            "interpretation": (
                "This post-freeze full phase reports descriptive one-axis scale contrasts "
                "for genes, TFs and cells. The TF contrast also changes the regulatory "
                "topology and which TFs regulate targets, so it does not identify a pure "
                "causal effect of candidate-TF count. It is not a joint scale validation, "
                "a tool ranking, or a parameter-selection stage."
                if budget == "full"
                else "This low-cost operational phase checks two moderate truth-bearing "
                "conditions plus single-group and extreme-imbalance routing. It omits every "
                "declared scale endpoint and therefore reports no one-axis scale contrast, tool "
                "ranking, or parameter-selection result."
            ),
            "comparison_reuse": (
                "Forbidden. Dimensional-transfer datasets and ZIPs remain isolated from every "
                "held-out external-tool comparison family, including exploratory families."
            ),
        },
        "summary": results["summary"],
        "algorithm_seed": results["algorithm_seed"],
        "datasets": results["datasets"],
        "conditions": results["conditions"],
        "contrasts": results["contrasts"],
        "design": panel["design"],
        "feasibility": panel["feasibility_basis"],
        "selection": {
            "budget": selection["calibration_budget"],
            "evidence_role": "publication" if selection["calibration_budget"] == "full" else "descriptive",
            "policy": policy,
        },
        "limitations": results["limitations"],
        "links": [
            {
                "label": link["label"],
                "href": safe_relative_link(report_path, link["path"], root),
            }
            for link in links
        ]
        + export_links,
    }
    return reconcile_report_bundle(
        workflow,
        write_report_bundle(
            repo_root=root,
            kind="dimensional_transfer",
            output_path=report_path,
            payload=payload,
            input_paths=[
                results_path,
                panel_path,
                selection_path,
                template_path,
                runtime_profile_path,
                policy_path,
                summary_path,
                *export_inputs,
            ],
            manifest_name="dimensional-transfer-report.manifest.json",
            immutable=True,
        ),
        "dimensional_transfer",
    )


__all__ = ["render_dimensional_transfer_report"]
