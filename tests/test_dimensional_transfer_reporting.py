from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.artifacts import canonical_json_sha256, sha256_file
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.report_publication import publish_workflow_report
from spathi_bench.reporting.common import central_report_path, render_html
from spathi_bench.reporting.dimensional_transfer import (
    _validate_results,
    render_dimensional_transfer_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _runtime_plan() -> dict[str, Any]:
    unsigned = {
        "schema_version": "1.0",
        "detected": {
            "allowed_cpus": list(range(8)),
            "cpu_allocation_order": list(range(8)),
            "effective_cpu_threads": 8,
            "effective_memory_bytes": 32 * 1024**3,
            "source": "report-test",
        },
        "request": {
            "max_cores": 8,
            "max_ram_gb": 32.0,
            "reserve_cores": 0,
            "reserve_ram_gb": 0.0,
        },
        "capacity": {
            "cpu_threads": 8,
            "memory_bytes": 32 * 1024**3,
            "memory_gib": 32.0,
        },
    }
    return {**unsigned, "fingerprint": canonical_json_sha256(unsigned)}


def _results_and_panel(budget: str = "reduced") -> tuple[dict[str, Any], dict[str, Any]]:
    panel = json.loads(
        (
            REPO_ROOT / "experiments/v1/simulations/scmultisim/dimensional_transfer/panel.json"
        ).read_text(encoding="utf-8")
    )
    conditions = []
    datasets = []
    values: dict[str, float] = {}
    for index, condition in enumerate(panel["conditions"], start=1):
        if budget not in condition["runs"]:
            continue
        adjusted = index / 10
        values[condition["id"]] = adjusted
        datasets.append(
            {
                "dataset_id": f"fixture_{condition['id']}",
                "condition_id": condition["id"],
                "analysis_role": condition["analysis_role"],
                "genes": condition["genes"],
                "transcription_factors": condition["transcription_factors"],
                "cells": condition["cells"],
                "generation_seed": 100 + index,
                "contexts": condition["expected_group_contexts"],
                "directed_aupr": 0.2,
                "adjusted_aupr": adjusted,
                "directed_auroc": 0.7,
                "f1_at_truth_count": 0.1,
                "epr_at_truth_count": 2.0,
                "candidate_prevalence": 0.01,
                "wall_time_seconds": 10.0 * index,
            }
        )
        conditions.append(
            {
                "id": condition["id"],
                "analysis_role": condition["analysis_role"],
                "genes": condition["genes"],
                "transcription_factors": condition["transcription_factors"],
                "cells": condition["cells"],
                "replicates": 1,
                "contexts": condition["expected_group_contexts"],
                "directed_aupr": 0.2,
                "adjusted_aupr": adjusted,
                "directed_auroc": 0.7,
                "f1_at_truth_count": 0.1,
                "epr_at_truth_count": 2.0,
                "candidate_prevalence": 0.01,
                "median_wall_time_seconds": 10.0 * index,
            }
        )
    contrasts = (
        [
            {
                "id": "gene_scale_contrast",
                "estimand": "difference_in_mean_adjusted_aupr",
                "estimate": values["g15000_tf100_c300"] - values["g3000_tf100_c300"],
            },
            {
                "id": "tf_scale_contrast",
                "estimand": "difference_in_mean_adjusted_aupr",
                "estimate": values["g3000_tf1500_c300"] - values["g3000_tf100_c300"],
            },
            {
                "id": "cell_scale_contrast",
                "estimand": "difference_in_mean_adjusted_aupr",
                "estimate": values["g100_tf10_c10000"] - values["g100_tf10_c300"],
            },
        ]
        if budget == "full"
        else []
    )
    return {
        "budget": budget,
        "datasets": datasets,
        "conditions": conditions,
        "contrasts": contrasts,
        "exact_production_diagnostic": {
            "analysis_role": "post_freeze_descriptive_diagnostic",
            "selection_feedback": False,
            "conditions": ["g3000_tf100_c300", "g100_tf10_c300"],
            "status": "not_applicable_production_is_exact",
            "pairs": [],
            "condition_summaries": [],
            "identity_delta": {
                "adjusted_aupr": 0.0,
                "target_coverage": 0.0,
                "truth_edge_coverage": 0.0,
                "wall_time_seconds": 0.0,
                "fitted_estimators": 0,
            },
        },
    }, panel


def _report_payload() -> dict[str, Any]:
    results, panel = _results_and_panel()
    group_policy = {
        "branches": {
            "single_group": {"condition": {"maximum_group_count": 1}, "parameters": {}},
            "multi_group": {"condition": {"minimum_group_count": 2}, "parameters": {}},
        }
    }
    return {
        "schema_version": "1.0",
        "report_type": "dimensional_transfer",
        "workflow": {
            "flow_id": "transfer-fixture",
            "status": "complete",
            "profile": "reduced",
            "evidence_role": "operational",
            "identity": {"fingerprint": "f" * 64},
        },
        "scope": {
            "claim": "truth_bearing_synthetic_dimensional_transfer",
            "interpretation": "Never render </script><script>alert(1)</script> as markup.",
            "comparison_reuse": "Forbidden for every external-comparison family.",
        },
        "summary": {
            "datasets_requested": 4,
            "datasets_completed": 4,
            "contexts_evaluated": 10,
            "all_completed": True,
        },
        "algorithm_seed": {"value": 123, "derivation": {}},
        **results,
        "design": panel["design"],
        "feasibility": panel["feasibility_basis"],
        "exact_production_diagnostic": {
            "analysis_role": "post_freeze_descriptive_diagnostic",
            "selection_feedback": False,
            "conditions": ["g3000_tf100_c300", "g100_tf10_c300"],
            "exact_policy": {
                "policy_id": "spathi_exact_reference",
                "source_policy_id": "spathi_production_v1",
                "policy_sha256": "1" * 64,
            },
            "production_policy": {
                "policy_id": "spathi_production_v1",
                "policy_sha256": "1" * 64,
            },
            "paired_algorithm_seed": 123,
            "status": "not_applicable_production_is_exact",
            "pairs": [],
            "condition_summaries": [],
            "identity_delta": {
                "adjusted_aupr": 0.0,
                "target_coverage": 0.0,
                "truth_edge_coverage": 0.0,
                "wall_time_seconds": 0.0,
                "fitted_estimators": 0,
            },
            "interpretation": "Production is the exact reference.",
        },
        "production": {
            "budget": "reduced",
            "evidence_role": "descriptive",
            "decision": "descriptive_only",
            "policy": {
                "policy_id": "spathi_production_v1",
                "policy_sha256": "1" * 64,
                "group_policy": group_policy,
            },
        },
        "limitations": ["Operational subset only."],
        "links": [],
    }


def test_result_reconciliation_detects_a_stale_scale_contrast() -> None:
    results, panel = _results_and_panel("full")
    _validate_results(results, panel)
    results["contrasts"][0]["estimate"] += 0.01

    with pytest.raises(BenchmarkConfigError, match="contrast is stale"):
        _validate_results(results, panel)


def test_renderer_is_english_interactive_self_contained_and_scope_safe() -> None:
    html = render_html(kind="dimensional_transfer", payload=_report_payload())

    assert "Does SPATHI preserve accuracy" in html
    assert "Budget-specific truth-bearing conditions and routing controls" in html
    assert "Exact reference versus production policy" in html
    assert "Full-budget change from each matched anchor" in html
    assert "No scale contrast in the reduced budget" in html
    assert "ANDREA inter-stage ZIPs" in html
    assert "plotly.js" in html.lower()
    assert "<script src=" not in html.lower()
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e" in html


def test_real_renderer_is_byte_identical_after_workflow_completion_and_reconstruction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "schemas", root / "schemas")
    shutil.copytree(
        REPO_ROOT / "experiments/v1/simulations/scmultisim/dimensional_transfer",
        root / "experiments/v1/simulations/scmultisim/dimensional_transfer",
    )
    (root / "experiments/v1/seeds").mkdir(parents=True)
    shutil.copyfile(
        REPO_ROOT / "experiments/v1/seeds/seed-plan.json",
        root / "experiments/v1/seeds/seed-plan.json",
    )
    (root / "experiments/v1/profiles").mkdir(parents=True)
    template_path = root / "experiments/v1/profiles/dimensional-transfer-reduced.json"
    shutil.copyfile(
        REPO_ROOT / "experiments/v1/profiles/dimensional-transfer-reduced.json",
        template_path,
    )
    shutil.copytree(
        REPO_ROOT / "experiments/v1/execution",
        root / "experiments/v1/execution",
    )

    version = root / "work/v1"
    flow_id = "dimensional-renderer-fixture"
    workflow_root = version / "workflows/dimensional_transfer/reduced" / flow_id
    workflow_path = workflow_root / "workflow.json"
    runtime_profile = json.loads(template_path.read_text(encoding="utf-8"))
    runtime_profile.update(
        {
            "id": "dimensional-transfer-reduced-runtime",
            "description": "Runtime dimensional-transfer report fixture.",
            "inference_batches": [],
        }
    )
    runtime_profile_path = _write_json(workflow_root / "materialized/profile.json", runtime_profile)
    run_root = version / "dimensional-transfer-reduced-runtime" / flow_id
    metrics_path = run_root / "reports/metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text("fixture\n", encoding="utf-8")
    summary_path = _write_json(
        run_root / "reports/summary.json",
        {"status": "complete", "scheduling": {}},
    )
    exports_path = _write_json(
        run_root / "exports/export-manifest.json",
        {"status": "complete", "bundles": []},
    )
    pipeline_state_path = _write_json(
        run_root / "state.json",
        {
            "purpose": "dimensional_transfer",
            "status": "completed",
            "exports_manifest": exports_path.relative_to(root).as_posix(),
        },
    )

    group_policy = {
        "routing_parameter": "group_count",
        "branches": {
            "single_group": {
                "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                "parameters": {},
            },
            "multi_group": {
                "condition": {"minimum_group_count": 2},
                "parameters": {},
            },
        },
    }
    policy_sha256 = canonical_json_sha256(group_policy)
    production_path = _write_json(
        workflow_root / "production-config.json",
        {
            "schema_version": "1.0",
            "artifact_type": "spathi_production_policy",
            "budget": "reduced",
            "evidence_role": "descriptive",
            "status": "descriptive",
            "decision": "descriptive_only",
            "policy": {
                "policy_id": "spathi_production_v1",
                "policy_sha256": policy_sha256,
                "group_policy": group_policy,
                "spathi_commit": "b" * 40,
            },
            "provenance": {
                label: {"path": f"work/v1/evidence/{label}.json", "sha256": "c" * 64}
                for label in (
                    "calibration_selection",
                    "efficiency_plan",
                    "efficiency_evidence",
                    "efficiency_results",
                )
            },
        },
    )
    panel_path = root / ("experiments/v1/simulations/scmultisim/dimensional_transfer/panel.json")
    result_rows, _panel = _results_and_panel()
    exact_diagnostic = {
        "analysis_role": "post_freeze_descriptive_diagnostic",
        "selection_feedback": False,
        "conditions": ["g3000_tf100_c300", "g100_tf10_c300"],
        "exact_policy": {
            "policy_id": "spathi_exact_reference",
            "source_policy_id": "spathi_production_v1",
            "policy_sha256": policy_sha256,
        },
        "production_policy": {
            "policy_id": "spathi_production_v1",
            "policy_sha256": policy_sha256,
        },
        "paired_algorithm_seed": 2078775612,
        "status": "not_applicable_production_is_exact",
        "pairs": [],
        "condition_summaries": [],
        "identity_delta": {
            "adjusted_aupr": 0.0,
            "target_coverage": 0.0,
            "truth_edge_coverage": 0.0,
            "wall_time_seconds": 0.0,
            "fitted_estimators": 0,
        },
        "interpretation": "The frozen production policy is the exact reference.",
    }
    results_path = _write_json(
        workflow_root / "dimensional-transfer-results.json",
        {
            "schema_version": "1.0",
            "artifact_type": "spathi_dimensional_transfer_results",
            "budget": "reduced",
            "evidence_role": "operational",
            "claim_scope": "truth_bearing_synthetic_dimensional_transfer",
            "status": "complete",
            "panel": {
                "path": panel_path.relative_to(root).as_posix(),
                "sha256": sha256_file(panel_path),
            },
            "profile": {
                "path": runtime_profile_path.relative_to(root).as_posix(),
                "sha256": sha256_file(runtime_profile_path),
            },
            "production_config": {
                "path": production_path.relative_to(root).as_posix(),
                "sha256": sha256_file(production_path),
                "policy_id": "spathi_production_v1",
                "policy_sha256": policy_sha256,
            },
            "algorithm_seed": {
                "value": 2078775612,
                "derivation": {
                    "namespace": "reduced.dimensional_transfer",
                    "stream": "inference",
                    "coordinates": {"purpose": "algorithm_replicate", "replicate_index": 1},
                },
            },
            "summary_inputs": {
                "metrics_csv": {
                    "path": metrics_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(metrics_path),
                },
                "summary_json": {
                    "path": summary_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(summary_path),
                },
                "pipeline_state": {
                    "path": pipeline_state_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(pipeline_state_path),
                },
            },
            "summary": {
                "datasets_requested": 4,
                "datasets_completed": 4,
                "contexts_evaluated": sum(row["contexts"] for row in result_rows["datasets"]),
                "all_completed": True,
            },
            "datasets": result_rows["datasets"],
            "conditions": result_rows["conditions"],
            "contrasts": result_rows["contrasts"],
            "exact_production_diagnostic": exact_diagnostic,
            "limitations": [
                "Synthetic dimensional shapes only.",
                "Operational subset only.",
                "One frozen policy only.",
                "No confirmatory ranking.",
            ],
        },
    )
    report_path = central_report_path(
        version_root=version,
        report_type="dimensional_transfer",
        scope="reduced",
        flow_id=flow_id,
    )
    manifest_path = report_path.parent / "dimensional-transfer-report.manifest.json"
    publishing = {
        "schema_version": "1.0",
        "workflow_type": "dimensional_transfer",
        "profile": "reduced",
        "flow_id": flow_id,
        "identity": {"fingerprint": "d" * 64},
        "runtime_plan": _runtime_plan(),
        "status": "publishing",
        "production_config": production_path.relative_to(root).as_posix(),
        "production_config_sha256": sha256_file(production_path),
        "template_profile": template_path.relative_to(root).as_posix(),
        "template_profile_sha256": sha256_file(template_path),
        "runtime_profile": runtime_profile_path.relative_to(root).as_posix(),
        "runtime_profile_sha256": sha256_file(runtime_profile_path),
        "panel": panel_path.relative_to(root).as_posix(),
        "panel_sha256": sha256_file(panel_path),
        "results": results_path.relative_to(root).as_posix(),
        "results_sha256": sha256_file(results_path),
        "run_root": run_root.relative_to(root).as_posix(),
        "interactive_report": report_path.relative_to(root).as_posix(),
        "pipeline": {},
    }

    def render() -> dict[str, Any]:
        return render_dimensional_transfer_report(
            repo_root=root,
            version_root=version,
            workflow_root=workflow_root,
        )

    completed, bundle = publish_workflow_report(
        repo_root=root,
        workflow_path=workflow_path,
        publishing_state=publishing,
        html_path=report_path,
        manifest_path=manifest_path,
        render=render,
        label="dimensional transfer",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert workflow_path.relative_to(root).as_posix() not in {
        item["path"] for item in manifest["inputs"]
    }

    manifest_path.unlink()
    resumed, rebuilt = publish_workflow_report(
        repo_root=root,
        workflow_path=workflow_path,
        publishing_state=publishing,
        html_path=report_path,
        manifest_path=manifest_path,
        render=render,
        label="dimensional transfer",
    )
    assert resumed == completed
    assert rebuilt == bundle
    assert sha256_file(manifest_path) == bundle["manifest"]["sha256"]
