from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.reporting.scalability as scalability_reporting
from spathi_bench.artifacts import canonical_json_sha256, sha256_file
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.reporting.common import central_report_path, render_html
from spathi_bench.reporting.scalability import render_scalability_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _report_payload() -> dict[str, Any]:
    row = {
        "dataset_id": "scale_stress_fixture",
        "status": "completed",
        "network_mode": "dynamic",
        "policy_branch": "multi_group",
        "generation_seed": 12,
        "algorithm_seed": 34,
        "dimensions": {
            "cells": 100,
            "genes": 500,
            "groups": 2,
            "transcription_factors": 50,
            "targets": 500,
        },
        "execution_limits": {
            "max_parallel_runs": 1,
            "max_cores": 4,
            "max_ram_gb": 4.0,
            "planner": "heuristic",
            "planner_time_limit_seconds": 60.0,
            "inference_wall_time_limit_seconds": None,
        },
        "generation": {
            "wall_time_seconds": 1.0,
            "dataset_artifact_bytes": 1_000,
            "expression_bytes": 800,
        },
        "inference": {
            "spathi_wall_time_seconds": 4.0,
            "andrea_wall_time_seconds": 4.2,
            "harness_wall_time_seconds": 4.5,
            "peak_rss_bytes": 2_000,
            "estimated_peak_heap_with_rough_trees_bytes": 1_800,
            "spathi_output_bytes": 700,
            "inference_artifact_bytes": 1_200,
            "targets_requested": 500,
            "targets_eligible": 490,
            "targets_ineligible": 10,
            "models_requested": 980,
            "models_completed": 980,
            "trees_fitted": 98_000,
            "threads_effective": 4,
            "threads_available": 8,
            "rss_samples": 9,
        },
    }
    group_policy = {
        "branches": {
            "single_group": {
                "condition": {"maximum_group_count": 1},
                "parameters": {"n_estimators": 100},
            },
            "multi_group": {
                "condition": {"minimum_group_count": 2},
                "parameters": {"n_estimators": 100},
            },
        }
    }
    return {
        "schema_version": "1.0",
        "report_type": "scalability",
        "workflow": {
            "flow_id": "scale-fixture",
            "status": "complete",
            "profile": "reduced",
            "evidence_role": "operational",
            "identity": {"fingerprint": "f" * 64},
        },
        "scope": {
            "claim": "computational_scale_stress_only",
            "interpretation": "Never render </script><script>alert(1)</script> as markup.",
        },
        "summary": {
            "datasets_requested": 1,
            "datasets_completed": 1,
            "datasets_failed": 0,
            "all_completed": True,
            "failure_kinds": {
                "oom": 0,
                "timeout": 0,
                "tool_error": 0,
                "infrastructure_error": 0,
            },
            "total_inference_wall_time_seconds": 4.5,
            "datasets_with_peak_rss": 1,
            "maximum_peak_rss_bytes": 2_000,
            "total_spathi_output_bytes": 700,
            "total_trees_fitted": 98_000,
        },
        "datasets": {
            "summary": {
                "dataset_count": 1,
                "total_cells": 100,
                "cells": {"minimum": 100, "maximum": 100},
                "genes": {"minimum": 500, "maximum": 500},
                "groups": {"minimum": 2, "maximum": 2},
                "transcription_factors": {"minimum": 50, "maximum": 50},
                "targets": {"minimum": 500, "maximum": 500},
            },
            "rows": [row],
        },
        "production": {
            "budget": "reduced",
            "evidence_role": "descriptive",
            "status": "descriptive",
            "decision": "descriptive_only",
            "policy": {
                "policy_id": "spathi_production_v1",
                "policy_sha256": canonical_json_sha256(group_policy),
                "spathi_commit": "1" * 40,
                "group_policy": group_policy,
            },
        },
        "environment": {
            "system": "Linux",
            "kernel_release": "test",
            "machine": "x86_64",
            "cpu_model": "test CPU",
            "logical_cpus": 8,
            "host_memory_bytes": 8 * 1024**3,
            "max_cores": 4,
            "max_ram_gb": 4,
            "rss_measurement_method": "linux_ps_process_tree_rss_sum",
            "rss_sample_interval_seconds": 0.1,
            "spathi_image_id": "sha256:image",
            "andrea_environment_fingerprint": "2" * 64,
            "benchmark_environment_fingerprint": "3" * 64,
            "docker_runtime_environment_fingerprint": "4" * 64,
        },
        "notes": ["Dimensional stress only.", "No accuracy conclusion."],
        "links": [],
    }


def test_scalability_renderer_is_english_interactive_and_self_contained() -> None:
    html = render_html(kind="scalability", payload=_report_payload())

    assert "Can the frozen SPATHI policy complete across the declared scale grid?" in html
    assert "Datasets and dimensions tested" in html
    assert "Memory and output footprint" in html
    assert "Targets, models, trees, threads, and completion" in html
    assert "plotly.js" in html.lower()
    assert "<script src=" not in html.lower()
    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e" in html


def _complete_report_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    for relative in (
        "schemas/scale-stress-panel.schema.json",
        "schemas/profile.schema.json",
        "schemas/scalability-results.schema.json",
        "schemas/spathi-production-config.schema.json",
        "experiments/v1/profiles/scalability-reduced.json",
        "experiments/v1/simulations/scale_stress/scale-reduced.json",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)

    version = root / "work/v1"
    evidence: dict[str, dict[str, str]] = {}
    for name in (
        "calibration_selection",
        "efficiency_plan",
        "efficiency_evidence",
        "efficiency_results",
    ):
        source = _write_json(version / "evidence" / f"{name}.json", {"id": name})
        evidence[name] = {
            "path": source.relative_to(root).as_posix(),
            "sha256": sha256_file(source),
        }

    group_policy = {
        "routing_parameter": "group_count",
        "shared_parameters": ["n_estimators"],
        "branches": {
            "single_group": {
                "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                "parameters": {"n_estimators": 100},
            },
            "multi_group": {
                "condition": {"minimum_group_count": 2},
                "parameters": {"n_estimators": 100},
            },
        },
    }
    production = {
        "schema_version": "1.0",
        "artifact_type": "spathi_production_policy",
        "budget": "reduced",
        "evidence_role": "descriptive",
        "status": "descriptive",
        "decision": "descriptive_only",
        "policy": {
            "policy_id": "spathi_production_v1",
            "policy_sha256": canonical_json_sha256(group_policy),
            "group_policy": group_policy,
            "spathi_commit": "1" * 40,
        },
        "provenance": evidence,
    }
    production_path = _write_json(
        version / "workflows/efficiency/reduced/efficiency-fixture/production-config.json",
        production,
    )

    panel = json.loads(
        (root / "experiments/v1/simulations/scale_stress/scale-reduced.json").read_text(
            encoding="utf-8"
        )
    )
    rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(panel["datasets"], start=1):
        cells = sum(group["cells"] for group in dataset["groups"])
        groups = len(dataset["groups"])
        metadata_path = _write_json(
            version / "artifacts" / f"metadata-{dataset['id']}.json",
            {"status": "complete"},
        )
        rows.append(
            {
                "dataset_id": dataset["id"],
                "status": "completed",
                "network_mode": dataset["network_mode"],
                "policy_branch": "single_group" if groups == 1 else "multi_group",
                "generation_seed": dataset["seed"],
                "algorithm_seed": 100 + index,
                "seed_derivation": {
                    "namespace": "reduced.scalability",
                    "stream": "inference",
                    "coordinates": {
                        "purpose": "scalability",
                        "dataset_id": dataset["id"],
                        "replicate_index": 1,
                    },
                },
                "dimensions": {
                    "cells": cells,
                    "genes": dataset["genes"],
                    "groups": groups,
                    "transcription_factors": dataset["tfs"],
                    "targets": dataset["genes"],
                },
                "execution_limits": {
                    "max_parallel_runs": 1,
                    "max_cores": 8,
                    "max_ram_gb": 32.0,
                    "planner": "heuristic",
                    "planner_time_limit_seconds": 60.0,
                    "inference_wall_time_limit_seconds": None,
                },
                "generation": {
                    "wall_time_seconds": float(index),
                    "cache_hit": False,
                    "dataset_artifact_bytes": 1_000 * index,
                    "expression_bytes": 800 * index,
                },
                "inference": {
                    "andrea_wall_time_seconds": 4.0 * index,
                    "spathi_wall_time_seconds": 3.0 * index,
                    "harness_wall_time_seconds": 5.0 * index,
                    "peak_rss_bytes": 2_000 * index,
                    "rss_samples": 4,
                    "cache_hit": False,
                    "inference_artifact_bytes": 1_500 * index,
                    "spathi_output_bytes": 900 * index,
                    "targets_requested": dataset["genes"],
                    "targets_eligible": dataset["genes"],
                    "targets_ineligible": 0,
                    "models_requested": dataset["genes"] * groups,
                    "models_completed": dataset["genes"] * groups,
                    "models_trained": dataset["genes"] * groups,
                    "models_failed": 0,
                    "trees_fitted": dataset["genes"] * groups * 100,
                    "threads_requested": 4,
                    "threads_effective": 4,
                    "threads_available": 8,
                    "estimated_peak_heap_with_rough_trees_bytes": 1_800 * index,
                },
                "dataset_artifact": {"kind": "dataset", "digest": f"{index}" * 64},
                "inference_artifact": {"kind": "inference", "digest": f"{index + 2}" * 64},
                "run_metadata": {
                    "path": metadata_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(metadata_path),
                },
            }
        )

    profile_path = root / "experiments/v1/profiles/scalability-reduced.json"
    results = {
        "schema_version": "1.0",
        "artifact_type": "spathi_scalability_results",
        "budget": "reduced",
        "evidence_role": "operational",
        "claim_scope": "computational_scale_stress_only",
        "status": "complete",
        "expression_preparation": {
            "input_scale": "nonnegative_synthetic_counts_like",
            "normalization": "library-size-log1p",
            "target_sum": 10000.0,
            "library_size_scope": "all_generated_genes_before_any_filtering",
            "formula": "log1p(target_sum * raw_value / per_cell_library_size)",
        },
        "profile": {
            "path": profile_path.relative_to(root).as_posix(),
            "sha256": sha256_file(profile_path),
        },
        "production_config": {
            "path": production_path.relative_to(root).as_posix(),
            "sha256": sha256_file(production_path),
            "policy_id": "spathi_production_v1",
            "policy_sha256": production["policy"]["policy_sha256"],
            "decision": "descriptive_only",
        },
        "execution_environment": {
            "spathi_commit": "1" * 40,
            "andrea_commit": "2" * 40,
            "spathi_image_id": "sha256:image",
            "andrea_environment_fingerprint": "3" * 64,
            "benchmark_environment_fingerprint": "4" * 64,
            "docker_runtime_environment_fingerprint": "5" * 64,
            "system": "Linux",
            "kernel_release": "test",
            "machine": "x86_64",
            "cpu_model": "test CPU",
            "logical_cpus": 8,
            "host_memory_bytes": 8 * 1024**3,
            "max_cores": 8,
            "max_ram_gb": 32,
            "rss_measurement_method": "linux_ps_process_tree_rss_sum",
            "rss_sample_interval_seconds": 0.1,
        },
        "summary": {
            "datasets_requested": len(rows),
            "datasets_completed": len(rows),
            "datasets_failed": 0,
            "all_completed": True,
            "failure_kinds": {
                "oom": 0,
                "timeout": 0,
                "tool_error": 0,
                "infrastructure_error": 0,
            },
            "total_generation_wall_time_seconds": sum(
                row["generation"]["wall_time_seconds"] for row in rows
            ),
            "total_inference_wall_time_seconds": sum(
                row["inference"]["harness_wall_time_seconds"] for row in rows
            ),
            "maximum_peak_rss_bytes": max(row["inference"]["peak_rss_bytes"] for row in rows),
            "datasets_with_peak_rss": len(rows),
            "total_spathi_output_bytes": sum(
                row["inference"]["spathi_output_bytes"] for row in rows
            ),
            "total_trees_fitted": sum(row["inference"]["trees_fitted"] for row in rows),
        },
        "datasets": rows,
        "notes": [
            "The panel represents synthetic dimensions only and makes no biological claim.",
            "Accuracy is evaluated in the independent comparison phase.",
        ],
    }
    workflow_root = version / "workflows/scalability/reduced/scale-fixture"
    results_path = _write_json(workflow_root / "scalability-results.json", results)
    report_path = version / "reports/reduced/scale-fixture/scalability.html"
    workflow = {
        "schema_version": "1.0",
        "workflow_type": "scalability",
        "profile": "reduced",
        "flow_id": "scale-fixture",
        "identity": {"fingerprint": "5" * 64},
        "status": "publishing",
        "production_config": production_path.relative_to(root).as_posix(),
        "production_config_sha256": sha256_file(production_path),
        "results": results_path.relative_to(root).as_posix(),
        "results_sha256": sha256_file(results_path),
        "interactive_report": report_path.relative_to(root).as_posix(),
        "dataset_runs": [],
    }
    _write_json(workflow_root / "workflow.json", workflow)
    return root, version, workflow_root


def test_completed_scalability_report_is_canonical_and_traceable(tmp_path: Path) -> None:
    root, version, workflow_root = _complete_report_fixture(tmp_path)

    bundle = render_scalability_report(
        repo_root=root,
        version_root=version,
        workflow_root=workflow_root,
    )

    report_path = root / bundle["html"]["path"]
    manifest_path = root / bundle["manifest"]["path"]
    assert report_path == (version / "reports/reduced/scale-fixture/scalability.html").resolve()
    assert report_path.is_file()
    assert manifest_path.name == "scalability-report.manifest.json"
    assert sha256_file(report_path) == bundle["html"]["sha256"]
    html = report_path.read_text(encoding="utf-8")
    assert "computational completion and resource use" in html
    assert "makes no biological claim" in html
    assert "does not evaluate network-inference accuracy" in html
    assert "No accuracy conclusion" not in html


def test_scalability_report_remains_visible_with_a_validated_failed_endpoint(
    tmp_path: Path,
) -> None:
    root, version, workflow_root = _complete_report_fixture(tmp_path)
    results_path = workflow_root / "scalability-results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    failed = results["datasets"][-1]
    inference = failed.pop("inference")
    failed.pop("inference_artifact")
    failed.pop("run_metadata")
    failed["status"] = "failed"
    failed["failure"] = {
        "failure_kind": "oom",
        "stage": "inference",
        "message": "Container was OOMKilled",
        "orchestrator_exit_code": 1,
        "tool_exit_code": 137,
    }
    failed["observations"] = {
        "harness_wall_time_seconds": inference["harness_wall_time_seconds"],
        "tool_wall_time_seconds": inference["spathi_wall_time_seconds"],
        "peak_rss_bytes": inference["peak_rss_bytes"],
        "rss_samples": inference["rss_samples"],
    }
    log_path = version / "scalability-failures" / f"{failed['dataset_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("Container was OOMKilled\n", encoding="utf-8")
    failed["diagnostics"] = {
        "command_log": {
            "path": log_path.relative_to(root).as_posix(),
            "sha256": sha256_file(log_path),
            "size_bytes": log_path.stat().st_size,
        },
        "progress": {
            "percent": 63,
            "status": "running",
            "phase": "model_inference",
            "message": "Training target models",
            "warnings": [],
        },
        "tool_log_tail": "Container was OOMKilled",
    }
    completed = [row for row in results["datasets"] if row["status"] == "completed"]
    results["summary"].update(
        {
            "datasets_completed": len(completed),
            "datasets_failed": 1,
            "all_completed": False,
            "failure_kinds": {
                "oom": 1,
                "timeout": 0,
                "tool_error": 0,
                "infrastructure_error": 0,
            },
            "total_inference_wall_time_seconds": sum(
                row["inference"]["harness_wall_time_seconds"] for row in completed
            )
            + failed["observations"]["harness_wall_time_seconds"],
            "datasets_with_peak_rss": len(completed) + 1,
            "maximum_peak_rss_bytes": max(
                [row["inference"]["peak_rss_bytes"] for row in completed]
                + [failed["observations"]["peak_rss_bytes"]]
            ),
            "total_spathi_output_bytes": sum(
                row["inference"]["spathi_output_bytes"] for row in completed
            ),
            "total_trees_fitted": sum(row["inference"]["trees_fitted"] for row in completed),
        }
    )
    _write_json(results_path, results)
    workflow_path = workflow_root / "workflow.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["results_sha256"] = sha256_file(results_path)
    _write_json(workflow_path, workflow)

    bundle = render_scalability_report(
        repo_root=root,
        version_root=version,
        workflow_root=workflow_root,
    )

    html = (root / bundle["html"]["path"]).read_text(encoding="utf-8")
    assert "Attributable failure evidence" in html
    assert "Scale study complete with failed endpoints" in html
    assert "Container was OOMKilled" in html
    assert "Tool log tail" in html
    assert "Inference limit" in html
    assert "Frozen SPATHI configuration" in html
    manifest = json.loads((root / bundle["manifest"]["path"]).read_text(encoding="utf-8"))
    assert any(item["path"] == log_path.relative_to(root).as_posix() for item in manifest["inputs"])


def test_scalability_report_has_one_canonical_location(tmp_path: Path) -> None:
    version = tmp_path / "work/v1"
    expected = version / "reports/full/flow-01/scalability.html"

    assert (
        central_report_path(
            version_root=version,
            report_type="scalability",
            scope="full",
            flow_id="flow-01",
        )
        == expected.resolve()
    )


def test_candidate_policy_provenance_flattens_the_efficiency_safety_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    version = root / "work/v1"

    def reference(name: str) -> dict[str, str]:
        path = _write_json(version / "evidence" / f"{name}.json", {"id": name})
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }

    production = {
        "decision": "candidate_adopted",
        "provenance": {
            name: reference(name)
            for name in (
                "calibration_selection",
                "efficiency_plan",
                "efficiency_evidence",
                "efficiency_results",
            )
        },
    }
    production["provenance"]["efficiency_safety"] = {
        name: reference(f"safety-{name}") for name in ("panel", "plan", "evidence", "results")
    }

    links, inputs = scalability_reporting._production_inputs(
        repo_root=root,
        version_root=version,
        production=production,
    )

    assert len(links) == len(inputs) == 8
    assert {link["label"] for link in links} >= {
        "Efficiency Safety Panel",
        "Efficiency Safety Plan",
        "Efficiency Safety Evidence",
        "Efficiency Safety Results",
    }
    assert {path.name for path in inputs} >= {
        "safety-panel.json",
        "safety-plan.json",
        "safety-evidence.json",
        "safety-results.json",
    }


def test_scalability_report_rejects_an_inconsistent_resource_summary(tmp_path: Path) -> None:
    root, version, workflow_root = _complete_report_fixture(tmp_path)
    results_path = workflow_root / "scalability-results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["summary"]["total_trees_fitted"] += 1
    _write_json(results_path, results)
    workflow_path = workflow_root / "workflow.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["results_sha256"] = sha256_file(results_path)
    _write_json(workflow_path, workflow)

    with pytest.raises(BenchmarkConfigError, match="total_trees_fitted"):
        render_scalability_report(
            repo_root=root,
            version_root=version,
            workflow_root=workflow_root,
        )
