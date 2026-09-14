from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from selection_helpers import selected_document
from spathi_bench.artifacts import canonical_json_sha256, sha256_file
from spathi_bench.calibration import load_search_space, materialize_group_policy
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.execution_policy import resolve_execution_class
from spathi_bench.reporting.common import central_report_path, render_html
from spathi_bench.reporting.scalability import render_scalability_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _runtime_plan(*, memory_gib: float = 32.0) -> dict[str, Any]:
    unsigned = {
        "schema_version": "1.0",
        "detected": {
            "allowed_cpus": list(range(8)),
            "cpu_allocation_order": list(range(8)),
            "effective_cpu_threads": 8,
            "effective_memory_bytes": round(memory_gib * 1024**3),
            "source": "scalability-report-test",
        },
        "request": {
            "max_cores": 8,
            "max_ram_gb": memory_gib,
            "reserve_cores": 0,
            "reserve_ram_gb": 0.0,
        },
        "capacity": {
            "cpu_threads": 8,
            "memory_bytes": round(memory_gib * 1024**3),
            "memory_gib": memory_gib,
        },
    }
    return {**unsigned, "fingerprint": canonical_json_sha256(unsigned)}


def _execution_limits(execution_plan: dict[str, Any]) -> dict[str, Any]:
    inference = execution_plan["operations"]["inference"]
    threads = inference["cpu_threads"]
    return {
        **inference,
        "cpuset_cpus": execution_plan["detected"]["cpu_allocation_order"][:threads],
        "output_profile": execution_plan["output_profile"],
        "exclusive": execution_plan["exclusive_inference"],
        "execution_plan_fingerprint": execution_plan["fingerprint"],
        "timeout_seconds": 43_200.0,
    }


def _andrea_measurement(
    limits: dict[str, Any],
    *,
    peak_memory_bytes: int | None,
    telemetry_status: str = "complete",
) -> dict[str, Any]:
    cpuset = list(limits["cpuset_cpus"])
    return {
        "schema_version": "1.0",
        "scope": "physical_task",
        "wall_time_seconds": 4.2,
        "container_wall_time_seconds": 4.0,
        "cpu_time_seconds": 8.1 if peak_memory_bytes is not None else None,
        "peak_memory_bytes": peak_memory_bytes,
        "output_bytes": 700,
        "io_read_bytes": None,
        "io_write_bytes": None,
        "task_started_at_utc": "2026-01-01T00:00:00Z",
        "task_finished_at_utc": "2026-01-01T00:00:05Z",
        "task_started_monotonic_ns": 1,
        "task_finished_monotonic_ns": 4_200_000_001,
        "container_started_at_utc": "2026-01-01T00:00:00Z",
        "container_finished_at_utc": "2026-01-01T00:00:05Z",
        "assigned_resources": {
            "threads": limits["cpu_threads"],
            "ram_limit_bytes": limits["memory_bytes"],
            "cpuset_cpus_requested": cpuset,
            "cpuset_cpus_configured": cpuset,
            "cpuset_cpus_effective": cpuset,
            "cpu_quota_cores_effective": float(limits["cpu_threads"]),
            "ram_limit_bytes_effective": limits["memory_bytes"],
        },
        "telemetry": {
            "status": telemetry_status,
            "wall_source": "andrea_monotonic_clock",
            "wall_semantics": (
                "physical_task_end_to_end_including_container_launch_and_exit_collection"
            ),
            "container_wall_source": "docker_state_timestamps",
            "cpu_source": "test",
            "memory_source": "test" if peak_memory_bytes is not None else None,
            "memory_semantics": (
                "container_cgroup_peak" if peak_memory_bytes is not None else "unavailable"
            ),
            "io_source": None,
            "io_semantics": "unavailable",
            "throttled_time_seconds": 0.0,
            "oom_events": 0,
            "oom_kill_events": 0,
            "memory_limit_events": 0,
            "oom_killed": peak_memory_bytes is None,
        },
    }


def _andrea_execution_measurement(
    limits: dict[str, Any],
    *,
    wall_time_seconds: float,
) -> dict[str, Any]:
    cpuset = [list(limits["cpuset_cpus"])]
    return {
        "schema_version": "1.0",
        "scope": "andrea_execution",
        "wall_time_seconds": wall_time_seconds,
        "child_wall_time_sum_seconds": max(0.0, wall_time_seconds - 0.5),
        "postprocess_wall_time_seconds": 0.5,
        "cpu_time_seconds": wall_time_seconds * 2,
        "peak_memory_bytes": 2_000,
        "output_bytes": 700,
        "io_read_bytes": None,
        "io_write_bytes": None,
        "task_started_at_utc": "2026-01-01T00:00:00Z",
        "task_finished_at_utc": "2026-01-01T00:00:05Z",
        "task_started_monotonic_ns": 1,
        "task_finished_monotonic_ns": max(2, round(wall_time_seconds * 1_000_000_000)),
        "assigned_resources": {
            "threads_peak": limits["cpu_threads"],
            "ram_limit_peak_bytes": limits["memory_bytes"],
            "cpuset_cpus_requested_sets": cpuset,
            "cpuset_cpus_effective_sets": cpuset,
        },
        "telemetry": {
            "status": "complete",
            "wall_source": "andrea_monotonic_clock",
            "wall_semantics": (
                "andrea_execution_end_to_end_including_input_verification_preparation_"
                "scheduling_merging_and_requested_output_materialization"
            ),
            "cpu_source": "sum_of_child_container_cpu",
            "cpu_semantics": "sum_of_physical_container_cpu_excludes_andrea_host_process",
            "memory_source": "child_container_cgroup_peaks",
            "memory_semantics": (
                "overlap_upper_bound_from_physical_container_peaks_excludes_andrea_host_process"
            ),
            "io_source": None,
            "io_semantics": "unavailable",
        },
    }


def _report_payload() -> dict[str, Any]:
    limits = {
        "cpu_threads": 4,
        "memory_bytes": 4 * 1024**3,
        "memory_gib": 4.0,
        "cpuset_cpus": [0, 1, 2, 3],
        "planner": "heuristic",
        "planner_time_limit_seconds": 60.0,
        "output_profile": "canonical",
        "exclusive": True,
        "execution_plan_fingerprint": "e" * 64,
        "timeout_seconds": 43_200.0,
    }
    row = {
        "dataset_id": "scale_stress_fixture",
        "stage": "all_targets",
        "target_scope": {"method": "all"},
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
        "execution_limits": limits,
        "generation": {
            "wall_time_seconds": 1.0,
            "dataset_artifact_bytes": 1_000,
            "expression_bytes": 800,
        },
        "inference": {
            "spathi_wall_time_seconds": 4.0,
            "andrea_measurement": _andrea_measurement(limits, peak_memory_bytes=2_000),
            "andrea_execution_measurement": _andrea_execution_measurement(
                limits,
                wall_time_seconds=4.2,
            ),
            "estimated_peak_heap_with_rough_trees_bytes": 1_800,
            "spathi_output_bytes": 700,
            "inference_artifact_bytes": 1_200,
            "targets_requested": 500,
            "models_requested": 980,
            "models_completed": 980,
            "trees_fitted": 98_000,
            "threads_effective": 4,
            "threads_available": 8,
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
    scout = copy.deepcopy(row)
    scout["stage"] = "scout"
    scout["target_scope"] = {
        "method": "sha256_gene_id",
        "size": 64,
        "salt": "spathi-scalability-scout-v1",
        "dataset_id": row["dataset_id"],
    }
    scout["dimensions"]["targets"] = 64
    scout["inference"]["targets_requested"] = 64
    return {
        "schema_version": "1.0",
        "report_type": "scalability",
        "workflow": {
            "flow_id": "scale-fixture",
            "status": "complete",
            "profile": "reduced",
            "evidence_role": "operational",
            "runtime_plan": _runtime_plan(memory_gib=4.0),
            "identity": {"fingerprint": "f" * 64},
        },
        "scope": {
            "claim": "computational_scale_stress_only",
            "interpretation": "Never render </script><script>alert(1)</script> as markup.",
            "resource_boundary": {
                "target_scout_size": 64,
                "wall_time_measurement": "andrea_execution.wall_time_seconds",
                "prospective_wall_time_limit_seconds": 43_200,
                "maximum_projected_wall_time_fraction": 0.8,
                "maximum_observed_memory_fraction": 0.8,
            },
        },
        "summary": {
            "datasets_requested": 1,
            "datasets_completed": 1,
            "datasets_failed": 0,
            "datasets_not_attempted": 0,
            "all_completed": True,
            "scouts_attempted": 1,
            "scouts_completed": 1,
            "all_targets_attempted": 1,
            "all_targets_completed": 1,
            "failure_kinds": {
                "oom": 0,
                "timeout": 0,
                "tool_error": 0,
                "infrastructure_error": 0,
            },
            "total_inference_wall_time_seconds": 4.5,
            "attempts_with_peak_memory": 2,
            "maximum_peak_memory_bytes": 2_000,
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
            "rows": [
                {
                    "dataset_id": row["dataset_id"],
                    "status": "completed",
                    "network_mode": row["network_mode"],
                    "policy_branch": row["policy_branch"],
                    "generation_seed": row["generation_seed"],
                    "algorithm_seed": row["algorithm_seed"],
                    "dimensions": row["dimensions"],
                    "lane_ids": ["fixture_lane"],
                    "decisions": {
                        "scout": {
                            "decision": "attempt",
                            "reason": "first_preregistered_scout",
                        },
                        "all_targets": {
                            "decision": "attempt",
                            "reason": "within_preregistered_resource_boundary",
                        },
                    },
                    "scout": scout,
                    "all_targets": row,
                }
            ],
        },
        "selection": {
            "budget": "reduced",
            "evidence_role": "descriptive",
            "status": "descriptive",
            "policy": {
                "policy_id": "spathi_conditional_v1",
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
            "cpu_threads": 4,
            "memory_bytes": 4 * 1024**3,
            "memory_gib": 4.0,
            "cpuset_cpus": [0, 1, 2, 3],
            "planner": "heuristic",
            "planner_time_limit_seconds": 60.0,
            "output_profile": "canonical",
            "exclusive_inference": True,
            "execution_plan_fingerprint": "e" * 64,
            "measurement_contract": "andrea_container_telemetry_v1",
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
        "schemas/scalability-boundary.schema.json",
        "schemas/spathi-selected-config.schema.json",
        "schemas/execution-policy.schema.json",
        "experiments/v1/execution/policy.json",
        "experiments/v1/profiles/scalability-reduced.json",
        "experiments/v1/spathi/scalability-boundary.json",
        "experiments/v1/simulations/scale_stress/scale-reduced.json",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)

    version = root / "work/v1"
    runtime_plan = _runtime_plan()
    execution_plan = resolve_execution_class(
        repo_root=root,
        class_id="scale_canonical",
        runtime_plan=runtime_plan,
    )
    limits = _execution_limits(execution_plan)
    ranking_path = _write_json(version / "evidence/validation-ranking.json", {"id": "validation"})
    search = load_search_space(REPO_ROOT)
    group_policy = materialize_group_policy(search, {**search["base_configuration"], "n_estimators": 100})
    selection = selected_document(
        {
            "policy_id": "spathi_conditional_v1",
            "group_policy": group_policy,
            "spathi_commit": "1" * 40,
        },
        ranking_path=ranking_path.relative_to(root).as_posix(),
        ranking_sha256=sha256_file(ranking_path),
    )
    selection_path = _write_json(
        version / "workflows/calibration/reduced/calibration-fixture/selected-config.json",
        selection,
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
                "execution_limits": limits,
                "generation": {
                    "wall_time_seconds": float(index),
                    "cache_hit": False,
                    "dataset_artifact_bytes": 1_000 * index,
                    "expression_bytes": 800 * index,
                },
                "inference": {
                    "spathi_wall_time_seconds": 3.0 * index,
                    "andrea_measurement": _andrea_measurement(
                        limits,
                        peak_memory_bytes=2_000 * index,
                    ),
                    "andrea_execution_measurement": _andrea_execution_measurement(
                        limits,
                        wall_time_seconds=4.0 * index,
                    ),
                    "cache_hit": False,
                    "inference_artifact_bytes": 1_500 * index,
                    "spathi_output_bytes": 900 * index,
                    "targets_requested": dataset["genes"],
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

    attempts: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for index, all_targets in enumerate(rows):
        all_targets["stage"] = "all_targets"
        all_targets["target_scope"] = {"method": "all"}
        scout = copy.deepcopy(all_targets)
        scout["stage"] = "scout"
        scout["target_scope"] = {
            "method": "sha256_gene_id",
            "size": 64,
            "salt": "spathi-scalability-scout-v1",
            "dataset_id": scout["dataset_id"],
        }
        scout["dimensions"]["targets"] = 64
        groups = scout["dimensions"]["groups"]
        scout["inference"].update(
            {
                "targets_requested": 64,
                "models_requested": 64 * groups,
                "models_completed": 64 * groups,
                "models_trained": 64 * groups,
                "trees_fitted": 64 * groups * 100,
            }
        )
        lane_id = "reduced_progression"
        evidence_id = rows[index - 1]["dataset_id"] if index else None
        dataset_rows.append(
            {
                "dataset_id": all_targets["dataset_id"],
                "status": "completed",
                "network_mode": all_targets["network_mode"],
                "policy_branch": all_targets["policy_branch"],
                "generation_seed": all_targets["generation_seed"],
                "algorithm_seed": all_targets["algorithm_seed"],
                "seed_derivation": all_targets["seed_derivation"],
                "dimensions": copy.deepcopy(all_targets["dimensions"]),
                "lane_ids": [lane_id],
                "decisions": {
                    "scout": {
                        "dataset_id": all_targets["dataset_id"],
                        "lane_id": lane_id,
                        "stage": "scout",
                        "decision": "attempt",
                        "reason": (
                            "first_preregistered_scout"
                            if index == 0
                            else "within_preregistered_resource_boundary"
                        ),
                        "evidence_dataset_id": evidence_id,
                    },
                    "all_targets": {
                        "dataset_id": all_targets["dataset_id"],
                        "lane_id": lane_id,
                        "stage": "all_targets",
                        "decision": "attempt",
                        "reason": "within_preregistered_resource_boundary",
                        "evidence_dataset_id": all_targets["dataset_id"],
                    },
                },
                "scout": scout,
                "all_targets": all_targets,
            }
        )
        attempts.extend((scout, all_targets))

    profile_path = root / "experiments/v1/profiles/scalability-reduced.json"
    boundary_path = root / "experiments/v1/spathi/scalability-boundary.json"
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
        "resource_boundary": {
            "path": boundary_path.relative_to(root).as_posix(),
            "sha256": sha256_file(boundary_path),
            "id": "spathi_scalability_boundary_v1",
            "target_scout_size": 64,
            "wall_time_measurement": "andrea_execution.wall_time_seconds",
            "prospective_wall_time_limit_seconds": 43_200,
            "maximum_projected_wall_time_fraction": 0.8,
            "maximum_observed_memory_fraction": 0.8,
        },
        "profile": {
            "path": profile_path.relative_to(root).as_posix(),
            "sha256": sha256_file(profile_path),
        },
        "selected_config": {
            "path": selection_path.relative_to(root).as_posix(),
            "sha256": sha256_file(selection_path),
            "policy_id": "spathi_conditional_v1",
            "policy_sha256": selection["selected_policy"]["policy_sha256"],
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
            "cpu_threads": limits["cpu_threads"],
            "memory_bytes": limits["memory_bytes"],
            "memory_gib": limits["memory_gib"],
            "cpuset_cpus": limits["cpuset_cpus"],
            "planner": limits["planner"],
            "planner_time_limit_seconds": limits["planner_time_limit_seconds"],
            "output_profile": limits["output_profile"],
            "exclusive_inference": limits["exclusive"],
            "execution_plan_fingerprint": limits["execution_plan_fingerprint"],
            "measurement_contract": "andrea_container_telemetry_v1",
        },
        "summary": {
            "datasets_requested": len(dataset_rows),
            "datasets_completed": len(dataset_rows),
            "datasets_failed": 0,
            "datasets_not_attempted": 0,
            "all_completed": True,
            "scouts_attempted": len(dataset_rows),
            "scouts_completed": len(dataset_rows),
            "all_targets_attempted": len(dataset_rows),
            "all_targets_completed": len(dataset_rows),
            "failure_kinds": {
                "oom": 0,
                "timeout": 0,
                "tool_error": 0,
                "infrastructure_error": 0,
            },
            "total_generation_wall_time_seconds": sum(
                row["scout"]["generation"]["wall_time_seconds"]
                for row in dataset_rows
                if row["scout"] is not None
            ),
            "total_inference_wall_time_seconds": sum(
                row["inference"]["andrea_execution_measurement"]["wall_time_seconds"]
                for row in attempts
            ),
            "maximum_peak_memory_bytes": max(
                row["inference"]["andrea_measurement"]["peak_memory_bytes"] for row in attempts
            ),
            "attempts_with_peak_memory": len(attempts),
            "total_spathi_output_bytes": sum(
                row["inference"]["spathi_output_bytes"] for row in attempts
            ),
            "total_trees_fitted": sum(row["inference"]["trees_fitted"] for row in attempts),
        },
        "datasets": dataset_rows,
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
        "runtime_plan": runtime_plan,
        "status": "publishing",
        "selected_config": selection_path.relative_to(root).as_posix(),
        "selected_config_sha256": sha256_file(selection_path),
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
    endpoint = results["datasets"][-1]
    failed = endpoint["all_targets"]
    inference = failed.pop("inference")
    failed.pop("inference_artifact")
    failed.pop("run_metadata")
    failed["status"] = "failed"
    endpoint["status"] = "failed"
    failed["failure"] = {
        "failure_kind": "oom",
        "stage": "inference",
        "message": "Container was OOMKilled",
        "orchestrator_exit_code": 1,
        "tool_exit_code": 137,
    }
    failed["observations"] = {
        "andrea_measurement": _andrea_measurement(
            failed["execution_limits"],
            peak_memory_bytes=None,
            telemetry_status="partial",
        ),
        "andrea_execution_measurement": inference["andrea_execution_measurement"],
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
    attempts = [
        attempt
        for row in results["datasets"]
        for attempt in (row["scout"], row["all_targets"])
        if attempt is not None
    ]
    completed = [row for row in attempts if row["status"] == "completed"]
    results["summary"].update(
        {
            "datasets_completed": len(results["datasets"]) - 1,
            "datasets_failed": 1,
            "all_targets_completed": len(results["datasets"]) - 1,
            "all_completed": False,
            "failure_kinds": {
                "oom": 1,
                "timeout": 0,
                "tool_error": 0,
                "infrastructure_error": 0,
            },
            "total_inference_wall_time_seconds": sum(
                row["inference"]["andrea_execution_measurement"]["wall_time_seconds"]
                for row in completed
            )
            + failed["observations"]["andrea_execution_measurement"]["wall_time_seconds"],
            "attempts_with_peak_memory": len(completed),
            "maximum_peak_memory_bytes": max(
                row["inference"]["andrea_measurement"]["peak_memory_bytes"] for row in completed
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
    assert "Scale study complete with incomplete all-target endpoints" in html
    assert "Container was OOMKilled" in html
    assert "Tool log tail" in html
    assert "Prospective wall limit" in html
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
