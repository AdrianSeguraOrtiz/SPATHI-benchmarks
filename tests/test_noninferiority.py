from __future__ import annotations

import hashlib
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from spathi_bench.artifacts import ArtifactStore, canonical_json_sha256, sha256_file
from spathi_bench.calibration import locked_spathi_identity
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.noninferiority import (
    _coverage_endpoint_result,
    _geometric_ratio_upper_bound,
    _manifest_matches,
    analyze_noninferiority,
    decide_efficiency_adoption,
    load_noninferiority_study,
    materialize_noninferiority_plan,
)
from spathi_bench.noninferiority_run import (
    _execute_missing_tasks,
    _execute_task,
)
from spathi_bench.seeds import load_seed_plan, validate_plan_seed_record

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCKED_SPATHI = locked_spathi_identity(REPO_ROOT)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_result_bundle(
    output_dir: Path,
    *,
    plan_digest: str,
    evidence_digest: str,
    implementation_digest: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    results_path = output_dir / "results.json"
    comparisons_path = output_dir / "comparisons.csv"
    results_path.write_text('{"result": true}\n', encoding="utf-8")
    comparisons_path.write_text("candidate_id,decision\na,adoptable\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_noninferiority_output_manifest",
        "inputs": {"plan_sha256": plan_digest, "evidence_sha256": evidence_digest},
        "analysis_implementation_sha256": implementation_digest,
        "files": [
            {
                "path": "results.json",
                "sha256": sha256_file(results_path),
                "size": results_path.stat().st_size,
            },
            {
                "path": "comparisons.csv",
                "sha256": sha256_file(comparisons_path),
                "size": comparisons_path.stat().st_size,
            },
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def test_noninferiority_result_manifest_requires_the_exact_canonical_contract(
    tmp_path: Path,
) -> None:
    plan_digest = "1" * 64
    evidence_digest = "2" * 64
    implementation_digest = "3" * 64
    valid = tmp_path / "valid"
    _write_result_bundle(
        valid,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        implementation_digest=implementation_digest,
    )
    assert _manifest_matches(valid, plan_digest, evidence_digest, implementation_digest)

    cases = (
        "wrong_schema",
        "wrong_artifact_type",
        "wrong_inputs",
        "wrong_implementation",
        "missing_entry",
        "extra_entry",
        "duplicate_entry",
        "noncanonical_path",
        "missing_size",
        "boolean_size",
        "wrong_size",
        "wrong_hash",
        "non_array_files",
    )
    for case in cases:
        output_dir = tmp_path / case
        manifest = _write_result_bundle(
            output_dir,
            plan_digest=plan_digest,
            evidence_digest=evidence_digest,
            implementation_digest=implementation_digest,
        )
        if case == "wrong_schema":
            manifest["schema_version"] = "invalid"
        elif case == "wrong_artifact_type":
            manifest["artifact_type"] = "other"
        elif case == "wrong_inputs":
            manifest["inputs"]["plan_sha256"] = "f" * 64
        elif case == "wrong_implementation":
            manifest["analysis_implementation_sha256"] = "f" * 64
        elif case == "missing_entry":
            manifest["files"].pop()
        elif case == "extra_entry":
            manifest["files"].append({"path": "extra.txt", "sha256": "f" * 64, "size": 0})
        elif case == "duplicate_entry":
            manifest["files"][1] = deepcopy(manifest["files"][0])
        elif case == "noncanonical_path":
            manifest["files"][0]["path"] = "./results.json"
        elif case == "missing_size":
            manifest["files"][0].pop("size")
        elif case == "boolean_size":
            manifest["files"][0]["size"] = True
        elif case == "wrong_size":
            manifest["files"][0]["size"] += 1
        elif case == "wrong_hash":
            manifest["files"][0]["sha256"] = "f" * 64
        else:
            manifest["files"] = {}
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        assert not _manifest_matches(
            output_dir,
            plan_digest,
            evidence_digest,
            implementation_digest,
        ), case


@pytest.mark.parametrize(
    "case",
    (
        "missing_manifest",
        "missing_result",
        "extra_disk_member",
        "symlink_manifest",
        "symlink_result",
        "symlink_output_directory",
    ),
)
def test_noninferiority_result_manifest_rejects_noncanonical_disk_trees(
    tmp_path: Path,
    case: str,
) -> None:
    plan_digest = "1" * 64
    evidence_digest = "2" * 64
    implementation_digest = "3" * 64
    output_dir = tmp_path / "result"
    _write_result_bundle(
        output_dir,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        implementation_digest=implementation_digest,
    )
    checked_path = output_dir
    if case == "missing_manifest":
        (output_dir / "manifest.json").unlink()
    elif case == "missing_result":
        (output_dir / "results.json").unlink()
    elif case == "extra_disk_member":
        (output_dir / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    elif case == "symlink_manifest":
        manifest_path = output_dir / "manifest.json"
        target = tmp_path / "external-manifest.json"
        target.write_bytes(manifest_path.read_bytes())
        manifest_path.unlink()
        manifest_path.symlink_to(target)
    elif case == "symlink_result":
        results_path = output_dir / "results.json"
        target = tmp_path / "external-results.json"
        target.write_bytes(results_path.read_bytes())
        results_path.unlink()
        results_path.symlink_to(target)
    else:
        checked_path = tmp_path / "result-link"
        checked_path.symlink_to(output_dir, target_is_directory=True)
    assert not _manifest_matches(
        checked_path,
        plan_digest,
        evidence_digest,
        implementation_digest,
    )


@pytest.mark.parametrize(
    ("batch_kind", "role", "published"),
    [
        ("spathi_efficiency_safety", "candidate", True),
        ("spathi_efficiency", "candidate", True),
        ("spathi_efficiency", "reference", False),
    ],
)
def test_runner_publishes_one_bound_candidate_tool_failure_as_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_kind: str,
    role: str,
    published: bool,
) -> None:
    def failed_andrea(arguments: list[str], **_kwargs: Any) -> None:
        output_root = Path(arguments[arguments.index("--output-dir") + 1])
        run = output_root / "run"
        run.mkdir(parents=True)
        (run / "run_report.json").write_text(
            json.dumps(
                {
                    "status": "executed",
                    "output_profile": "full",
                    "run_id": "spathi_candidate",
                    "dataset": {"id": "dataset_a"},
                    "execution": {
                        "measurement": {
                            "schema_version": "1.0",
                            "scope": "andrea_execution",
                            "wall_time_seconds": 0.25,
                            "cpu_time_seconds": 0.1,
                            "peak_memory_bytes": 1024,
                            "output_bytes": 0,
                            "assigned_resources": {
                                "threads_peak": 1,
                                "ram_limit_peak_bytes": 4 * 1024**3,
                            },
                            "telemetry": {
                                "status": "complete",
                                "wall_source": "andrea_monotonic_clock",
                                "wall_semantics": (
                                    "andrea_execution_end_to_end_including_input_verification_"
                                    "preparation_scheduling_merging_and_requested_output_"
                                    "materialization"
                                ),
                                "oom_killed": False,
                            },
                        }
                    },
                    "tools": {
                        "selected": ["spathi_candidate"],
                        "completed": [],
                        "status_by_tool": {"spathi_candidate": "failed"},
                        "tool_origins": {"spathi_candidate": "custom"},
                        "failed": {"spathi_candidate": "model fit failed"},
                        "results": {
                            "spathi_candidate": {
                                "measurement": {
                                    "schema_version": "1.0",
                                    "scope": "logical_run",
                                    "wall_time_seconds": 0.25,
                                    "cpu_time_seconds": 0.1,
                                    "peak_memory_bytes": 1024,
                                    "output_bytes": 0,
                                    "assigned_resources": {
                                        "threads_peak": 1,
                                        "ram_limit_peak_bytes": 4 * 1024**3,
                                    },
                                    "telemetry": {
                                        "status": "complete",
                                        "oom_killed": False,
                                    },
                                }
                            }
                        },
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("spathi_bench.noninferiority_run.andrea_command", failed_andrea)
    store = ArtifactStore(tmp_path / "artifacts")
    spec = {"schema_version": "1.0", "batch_kind": batch_kind}
    task = {
        "key": store.key_for("inference", spec),
        "spec": spec,
        "dataset_id": "dataset_a",
        "dataset_manifest": tmp_path / "dataset-manifest.json",
        "run": {
            "run_id": "spathi_candidate",
            "resources": {"threads": 1, "ram_gb": 4.0},
        },
        "role": role,
        "custom": {"docker_image": "sha256:image"},
        "resources": {
            "memory_gib": 4.0,
            "planner": "heuristic",
            "planner_time_limit_seconds": 30,
            "output_profile": "full",
        },
    }

    if not published:
        with pytest.raises(BenchmarkConfigError, match="did not complete"):
            _execute_task(
                task,
                store=store,
                repo_root=tmp_path,
                andrea_repo=tmp_path / "ANDREA",
                log_root=tmp_path / "logs",
            )
        return
    artifact = _execute_task(
        task,
        store=store,
        repo_root=tmp_path,
        andrea_repo=tmp_path / "ANDREA",
        log_root=tmp_path / "logs",
    )
    marker = json.loads(artifact.file("efficiency-execution-failure.json").read_text())
    assert marker["status"] == "tool_execution_failed"
    assert marker["dataset_id"] == "dataset_a"
    assert marker["failure_message"] == "model fit failed"


def test_serial_efficiency_execution_resumes_only_the_missing_suffix() -> None:
    tasks = [
        {"id": "reference", "artifact": None},
        {"id": "candidate_a", "artifact": None},
        {"id": "candidate_b", "artifact": None},
    ]
    first_calls: list[str] = []

    def crash_on_second(task: dict[str, Any]) -> Any:
        first_calls.append(task["id"])
        if task["id"] == "candidate_a":
            raise RuntimeError("injected process interruption")
        return f"artifact:{task['id']}"

    with pytest.raises(RuntimeError, match="injected process interruption"):
        _execute_missing_tasks(tasks, crash_on_second)
    assert first_calls == ["reference", "candidate_a"]
    assert [task["artifact"] for task in tasks] == ["artifact:reference", None, None]

    resumed_calls: list[str] = []

    def complete(task: dict[str, Any]) -> Any:
        resumed_calls.append(task["id"])
        return f"artifact:{task['id']}"

    _execute_missing_tasks(tasks, complete)
    assert resumed_calls == ["candidate_a", "candidate_b"]
    assert [task["artifact"] for task in tasks] == [
        "artifact:reference",
        "artifact:candidate_a",
        "artifact:candidate_b",
    ]

    _execute_missing_tasks(tasks, lambda _task: pytest.fail("completed resume is not idempotent"))


def _selected_config(budget: str = "reduced") -> dict[str, Any]:
    parameters = {
        "weight_mode": "cell-distance-group-anchored",
        "distance_space": "pca",
        "n_components": 50,
        "distance_standardization": "none",
        "pca_svd_solver": "auto",
        "distance_metric": "cosine",
        "kernel": "gaussian",
        "bandwidth": "auto",
        "bandwidth_scale": 1.0,
        "group_size_correction": "cap-to-target",
        "tree_method": "extra-trees",
        "n_estimators": 250,
        "target_eligibility": "all",
        "min_target_detected_cells": 20,
        "min_target_detected_fraction": 0.01,
        "min_target_weighted_detected_fraction": 0.01,
        "min_target_weighted_detected_ess": 10.0,
        "adaptive_trees": False,
        "adaptive_min_estimators": 100,
        "adaptive_tree_step": 50,
        "adaptive_tolerance": 0.01,
        "adaptive_patience": 2,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "max_depth": None,
        "bootstrap": False,
    }
    configuration_sha256 = canonical_json_sha256(parameters)
    single_group = {
        **parameters,
        "weight_mode": "cell-distance",
        "distance_metric": "euclidean",
        "group_size_correction": "none",
    }
    group_policy = {
        "routing_parameter": "group_count",
        "shared_parameters": sorted(
            set(parameters) - {"weight_mode", "distance_metric", "group_size_correction"}
        ),
        "branches": {
            "single_group": {
                "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                "parameters": single_group,
            },
            "multi_group": {
                "condition": {"minimum_group_count": 2},
                "parameters": parameters,
            },
        },
    }
    policy_sha256 = canonical_json_sha256(group_policy)
    return {
        "schema_version": "1.0",
        "id": "spathi_selected_v1",
        "search_space": "experiments/v1/spathi/search-space.json",
        "calibration_budget": budget,
        "status": "selected",
        "phase": "frozen",
        "selected_policy": {
            "policy_id": "spathi_conditional_v1",
            "source_candidate_id": "selected_spathi",
            "spathi_commit": LOCKED_SPATHI["spathi_commit"],
            "source_configuration_sha256": configuration_sha256,
            "policy_sha256": policy_sha256,
            "group_policy": group_policy,
        },
        "selection_evidence": {
            "validation_ranking": "work/validation/ranking.json",
            "calibration_budget": budget,
            "validation_ranking_sha256": "2" * 64,
            "search_space_sha256": "3" * 64,
            "validation_manifest_sha256": "4" * 64,
            "benchmark_summary_sha256": "5" * 64,
            "spathi_build_fingerprint": LOCKED_SPATHI["spathi_build_fingerprint"],
            "spathi_image_id": "sha256:" + "7" * 64,
            "primary_metric": "aupr_directed",
            "hierarchical_score": 0.2,
            "winner_candidate_id": "selected_spathi",
            "winner_configuration_sha256": configuration_sha256,
            "winner_policy_sha256": policy_sha256,
            "tie_breakers_applied": ["stability", "simplicity", "canonical_configuration_id"],
            "frozen_at": "2026-09-04T00:00:00Z",
        },
        "test_accessed_before_freeze": False,
        "notes": ["Test selection."],
    }


def _repository(tmp_path: Path, *, budget: str = "reduced") -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    for relative in (
        "schemas/noninferiority-study.schema.json",
        "schemas/noninferiority-plan.schema.json",
        "schemas/noninferiority-panel.schema.json",
        "schemas/noninferiority-artifact-index.schema.json",
        "schemas/noninferiority-evidence.schema.json",
        "schemas/noninferiority-results.schema.json",
        "schemas/efficiency-design-sensitivity.schema.json",
        "schemas/efficiency-development.schema.json",
        "schemas/efficiency-development-selection.schema.json",
        "schemas/profile.schema.json",
        "schemas/seed-plan.schema.json",
        "schemas/spathi-selected-config.schema.json",
        "experiments/v1/spathi/efficiency-study.json",
        "experiments/v1/spathi/efficiency-design-sensitivity.json",
        "experiments/v1/spathi/efficiency-development.json",
        "experiments/v1/profiles/efficiency-development-reduced.json",
        "experiments/v1/profiles/efficiency-development-full.json",
        "experiments/v1/profiles/efficiency-reduced.json",
        "experiments/v1/profiles/efficiency-full.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/linear-baseline-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/linear-noisy-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/bifurcating-baseline-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/bifurcating-noisy-reduced-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_validation/dynamic-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/linear-baseline-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/linear-noisy-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/bifurcating-baseline-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/bifurcating-noisy-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_validation/dynamic-runs.json",
        "experiments/v1/seeds/seed-plan.json",
        "images.lock.json",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    for simulator in ("dyngen", "scmultisim"):
        for source in (
            REPO_ROOT / f"experiments/v1/simulations/{simulator}/efficiency_development"
        ).glob("*runs.json"):
            destination = root / source.relative_to(REPO_ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    study_path = root / "experiments/v1/spathi/efficiency-study.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["analysis"]["bootstrap_resamples"] = 1000
    study_path.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    ranking_path = root / "work/validation/ranking.json"
    ranking_path.parent.mkdir(parents=True)
    ranking_path.write_text(
        json.dumps(
            {
                "stage_id": "validation",
                "dataset_role": "validation",
                "calibration_rows": [{"dataset_id": "selection_dataset"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    selected_path = root / "work/calibration/selected-config.json"
    selected_path.parent.mkdir(parents=True)
    selected = _selected_config(budget)
    selected["selection_evidence"]["validation_ranking_sha256"] = sha256_file(ranking_path)
    selected_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    development_contract_path = root / "experiments/v1/spathi/efficiency-development.json"
    development_contract = json.loads(development_contract_path.read_text(encoding="utf-8"))
    development_metrics = root / "work/efficiency-development/metrics.csv"
    development_metrics.parent.mkdir(parents=True)
    development_metrics.write_text("verified\n", encoding="utf-8")
    development_state = root / "work/efficiency-development/state.json"
    development_state.write_text('{"verified":true}\n', encoding="utf-8")
    component_selection_path = root / "work/efficiency-development/component-selection.json"
    component_selection_path.write_text('{"verified":true}\n', encoding="utf-8")
    chosen_ids = {
        "eligible_targets_detect1pct_min20_wdetect1pct_ess10",
        "adaptive_trees_min100_step50_tol01_patience2",
    }
    chosen = [
        candidate
        for candidate in development_contract["candidates"]
        if candidate["id"] in chosen_ids
    ]
    primitive_nominees = []
    for candidate in chosen:
        result = {
            "candidate_id": candidate["id"],
            "component": candidate["component"],
            "mean_aupr_loss": 0.001,
            "wall_time_ratio": 0.5,
            "complete": True,
            "accuracy_eligible": True,
            "dataset_count": development_contract["budgets"][budget][
                "expected_independent_datasets"
            ],
            "accuracy_seed_count": len(
                development_contract["budgets"][budget]["accuracy_seed_indices"]
            ),
            "timing_seed_count": 1,
        }
        primitive_nominees.append(
            {
                **candidate,
                "development_result": result,
                "candidate_sha256": canonical_json_sha256(candidate),
            }
        )
    combined_candidate = {
        "id": "combined_scout_nominees",
        "component": "combined",
        "execution_overrides": {
            **primitive_nominees[0]["execution_overrides"],
            **primitive_nominees[1]["execution_overrides"],
        },
    }
    combined_result = {
        "candidate_id": combined_candidate["id"],
        "component": "combined",
        "mean_aupr_loss": 0.001,
        "wall_time_ratio": 0.4,
        "complete": True,
        "accuracy_eligible": True,
        "dataset_count": development_contract["budgets"][budget]["expected_independent_datasets"],
        "accuracy_seed_count": len(
            development_contract["budgets"][budget]["accuracy_seed_indices"]
        ),
        "timing_seed_count": 1,
    }
    nominee = {
        **combined_candidate,
        "development_result": combined_result,
        "candidate_sha256": canonical_json_sha256(combined_candidate),
    }
    development_selection = {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_development_selection",
        "contract": {
            "path": "experiments/v1/spathi/efficiency-development.json",
            "sha256": sha256_file(development_contract_path),
        },
        "budget": budget,
        "analysis_role": development_contract["budgets"][budget]["analysis_role"],
        "selected_config": {
            "path": "work/calibration/selected-config.json",
            "sha256": sha256_file(selected_path),
        },
        "component_selection": {
            "path": "work/efficiency-development/component-selection.json",
            "sha256": sha256_file(component_selection_path),
        },
        "evidence": {
            "component_scout_metrics": {
                "path": "work/efficiency-development/metrics.csv",
                "sha256": sha256_file(development_metrics),
            },
            "component_scout_execution_state": {
                "path": "work/efficiency-development/state.json",
                "sha256": sha256_file(development_state),
            },
            "joint_confirmation_metrics": {
                "path": "work/efficiency-development/metrics.csv",
                "sha256": sha256_file(development_metrics),
            },
            "joint_confirmation_execution_state": {
                "path": "work/efficiency-development/state.json",
                "sha256": sha256_file(development_state),
            },
        },
        "selection_rule": development_contract["selection_rule"],
        "candidate_results": [
            *(item["development_result"] for item in primitive_nominees),
            combined_result,
        ],
        "nominee": nominee,
        "reference_retained": False,
    }
    development_selection["selection_sha256"] = canonical_json_sha256(development_selection)
    development_selection_path = root / "work/efficiency-development/selection.json"
    development_selection_path.write_text(
        json.dumps(development_selection, indent=2) + "\n", encoding="utf-8"
    )
    panel_path = root / "work/efficiency/panel.json"
    panel_path.parent.mkdir(parents=True)
    base_strata = (
        ("dyngen_a", "dyngen", "linear_baseline"),
        ("dyngen_n", "dyngen", "linear_noisy"),
        ("dyngen_b", "dyngen", "bifurcating_baseline"),
        ("dyngen_bn", "dyngen", "bifurcating_noisy"),
        ("scmultisim_a", "scmultisim", "scmultisim_dynamic"),
    )
    replicates = 1 if budget == "reduced" else 5
    datasets = [
        {
            "dataset_id": f"{prefix}{replicate}",
            "simulator_id": simulator,
            "condition_id": condition,
            "generation_seed": 1000 + stratum_index * 100 + replicate,
            "contexts": ["group:a", "group:b"],
            "dataset_artifact": {
                "kind": "dataset",
                "digest": _digest(f"{prefix}{replicate}"),
            },
        }
        for stratum_index, (prefix, simulator, condition) in enumerate(base_strata)
        for replicate in range(1, replicates + 1)
    ]
    panel_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact_type": "spathi_efficiency_predeclared_panel",
                "selection_validation_manifest_sha256": "4" * 64,
                "selection_validation_ranking": {
                    "path": "work/validation/ranking.json",
                    "sha256": sha256_file(ranking_path),
                },
                "budget": budget,
                "dataset_role": "efficiency_validation",
                "split_id": "efficiency_validation_v1",
                "datasets": datasets,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = root / "work/efficiency/plan.json"
    materialize_noninferiority_plan(
        repo_root=root,
        selected_config_path=selected_path,
        panel_path=panel_path,
        development_selection_path=development_selection_path,
        budget=budget,
        output_path=plan_path,
        study_path=study_path,
    )
    return root, selected_path, plan_path


def _space(dataset_id: str, context_id: str, *, eligible: bool) -> dict[str, Any]:
    reference_target = _digest(f"reference-targets/{dataset_id}")
    candidate_target = (
        _digest(f"eligible-targets/{dataset_id}/{context_id}") if eligible else reference_target
    )
    source = _digest(f"sources/{dataset_id}")
    evaluation_space = _digest(f"candidate-space/{context_id}/{source}/{reference_target}")
    evaluation_targets = 100
    source_count = 10
    overlaps = 10
    evaluation_candidates = source_count * evaluation_targets - overlaps
    return {
        "policy": "full_requested_universe",
        "construction": "directed_source_target_cartesian_excluding_self_edges",
        "omitted_target_prediction_policy": "minimum_score_zero",
        "reference_source_universe_sha256": source,
        "candidate_source_universe_sha256": source,
        "reference_target_universe_sha256": reference_target,
        "candidate_target_universe_sha256": candidate_target,
        "reference_evaluation_target_universe_sha256": reference_target,
        "candidate_evaluation_target_universe_sha256": reference_target,
        "reference_evaluation_candidate_space_sha256": evaluation_space,
        "candidate_evaluation_candidate_space_sha256": evaluation_space,
        "n_reference_sources": source_count,
        "n_candidate_sources": source_count,
        "n_reference_targets": 100,
        "n_candidate_targets": 60 if eligible else 100,
        "n_reference_evaluation_targets": evaluation_targets,
        "n_candidate_evaluation_targets": evaluation_targets,
        "n_reference_evaluation_source_target_overlaps": overlaps,
        "n_candidate_evaluation_source_target_overlaps": overlaps,
        "n_reference_evaluation_candidates": evaluation_candidates,
        "n_candidate_evaluation_candidates": evaluation_candidates,
        "n_truth_edges_in_candidate_space": 100,
        "n_truth_edges_outside_candidate_space": 0,
        "truth_outside_candidate_space_examples": [],
    }


def _evidence(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    candidates = {candidate["id"]: candidate for candidate in plan["candidates"]}
    reference_parameters = plan["reference"]["group_policy"]["branches"]["multi_group"][
        "parameters"
    ]
    reference_configuration_sha256 = canonical_json_sha256(reference_parameters)
    panel = json.loads((plan_path.parent / "panel.json").read_text(encoding="utf-8"))
    datasets = [
        (
            item["dataset_id"],
            item["simulator_id"],
            item["condition_id"],
            item["generation_seed"],
        )
        for item in panel["datasets"]
    ]
    differences = {"combined_scout_nominees": -0.0012}
    runtime_ratios = {"combined_scout_nominees": 0.25}
    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in candidates.items():
        candidate_parameters = candidate["group_policy"]["branches"]["multi_group"]["parameters"]
        candidate_configuration_sha256 = canonical_json_sha256(candidate_parameters)
        eligible = candidate["intersection_diagnostic"]
        for dataset_index, (dataset_id, simulator, condition, generation_seed) in enumerate(
            datasets
        ):
            for context_index, context_id in enumerate(("group:a", "group:b")):
                for algorithm_seed in plan["budget"]["algorithm_seeds"]:
                    reference_value = 0.2 + dataset_index * 0.005 + context_index * 0.001
                    coverage = (
                        {
                            "eligibility_manifest_sha256": _digest(
                                f"eligibility/{dataset_id}/{context_id}"
                            ),
                            "truth_accessed_by_eligibility": False,
                            "n_reference_truth_edges": 100,
                            "n_retained_truth_edges": 95,
                        }
                        if eligible
                        else None
                    )
                    reference_measurements = {
                        "run_completed": True,
                        "wall_time_seconds": 100.0,
                        "peak_memory_bytes": 1_000_000_000,
                        "output_bytes": 1_000_000,
                        "targets_requested": 100,
                        "targets_processed": 100,
                        "models_requested": 200,
                        "models_completed": 200,
                        "models_trained": 190,
                        "trees_fitted": 50_000,
                    }
                    candidate_measurements = {
                        "run_completed": True,
                        "wall_time_seconds": 100.0 * runtime_ratios[candidate_id],
                        "peak_memory_bytes": 800_000_000,
                        "output_bytes": 900_000,
                        "targets_requested": 100,
                        "targets_processed": 60 if eligible else 100,
                        "models_requested": 120 if eligible else 200,
                        "models_completed": 120 if eligible else 200,
                        "models_trained": 115 if eligible else 190,
                        "trees_fitted": 20_000,
                    }
                    intersection = (
                        {
                            "target_universe_sha256": _digest(
                                f"eligible-targets/{dataset_id}/{context_id}"
                            ),
                            "candidate_space_sha256": _digest(
                                f"intersection-space/{dataset_id}/{context_id}"
                            ),
                            "n_targets": 60,
                            "n_candidates": 594,
                            "n_truth_edges": 95,
                            "reference_value": reference_value + 0.01,
                            "candidate_value": reference_value + 0.01 + differences[candidate_id],
                        }
                        if eligible
                        else None
                    )
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "dataset_id": dataset_id,
                            "dataset_manifest_sha256": _digest(f"dataset/{dataset_id}"),
                            "context_id": context_id,
                            "simulator_id": simulator,
                            "condition_id": condition,
                            "generation_seed": generation_seed,
                            "algorithm_seed": algorithm_seed,
                            "reference_policy_sha256": plan["reference"]["policy_sha256"],
                            "candidate_policy_sha256": candidate["policy_sha256"],
                            "reference_configuration_sha256": reference_configuration_sha256,
                            "candidate_configuration_sha256": candidate_configuration_sha256,
                            "metric": "aupr",
                            "level": "directed",
                            "reference_value": reference_value,
                            "candidate_value": reference_value
                            + differences[candidate_id]
                            + (dataset_index - 2) * 0.00005,
                            "reference_measurements": reference_measurements,
                            "candidate_measurements": candidate_measurements,
                            "candidate_space": _space(dataset_id, context_id, eligible=eligible),
                            "target_coverage": coverage,
                            "intersection_diagnostic": intersection,
                            "source_artifacts": {
                                "dataset": _digest(f"dataset-artifact/{dataset_id}"),
                                "reference_inference": _digest(
                                    f"reference/{dataset_id}/{algorithm_seed}"
                                ),
                                "candidate_inference": _digest(
                                    f"candidate/{candidate_id}/{dataset_id}/{algorithm_seed}"
                                ),
                                "reference_network_sha256": _digest("reference-network"),
                                "candidate_network_sha256": _digest("candidate-network"),
                                "reference_target_eligibility_sha256": _digest(
                                    "reference-eligibility"
                                ),
                                "reference_model_diagnostics_sha256": _digest(
                                    "reference-diagnostics"
                                ),
                                "reference_run_metadata_sha256": _digest("reference-metadata"),
                                "reference_run_report_sha256": _digest("reference-resources"),
                                "candidate_target_eligibility_sha256": _digest("eligibility"),
                                "candidate_model_diagnostics_sha256": _digest("diagnostics"),
                                "candidate_run_metadata_sha256": _digest("metadata"),
                                "candidate_run_report_sha256": _digest("resources"),
                            },
                        }
                    )
    index_path = plan_path.parent / "artifact-index.json"
    index_path.write_text("{}\n", encoding="utf-8")
    return {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_paired_evidence",
        "study_id": "spathi_efficiency_noninferiority_v1",
        "plan_sha256": sha256_file(plan_path),
        "panel_sha256": sha256_file(plan_path.parent / "panel.json"),
        "artifact_index": {
            "path": "work/efficiency/artifact-index.json",
            "sha256": sha256_file(index_path),
        },
        "evidence_builder_sha256": "9" * 64,
        "selection_validation_manifest_sha256": plan["reference"][
            "selection_validation_manifest_sha256"
        ],
        "budget": plan["budget"]["id"],
        "dataset_role": "efficiency_validation",
        "execution_failures": [],
        "rows": rows,
    }


def _write_evidence(root: Path, payload: dict[str, Any]) -> Path:
    path = root / "work/efficiency/evidence.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _use_preverified_unit_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep statistical unit tests independent from artifact-verifier tests."""

    def load_preverified(**kwargs: Any) -> dict[str, Any]:
        return json.loads(Path(kwargs["evidence_path"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(
        "spathi_bench.noninferiority.verify_noninferiority_evidence",
        load_preverified,
    )


def test_committed_noninferiority_contracts_are_valid() -> None:
    for name in (
        "noninferiority-study.schema.json",
        "noninferiority-plan.schema.json",
        "noninferiority-panel.schema.json",
        "noninferiority-artifact-index.schema.json",
        "noninferiority-evidence.schema.json",
        "noninferiority-results.schema.json",
    ):
        schema = json.loads((REPO_ROOT / "schemas" / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    study = load_noninferiority_study(REPO_ROOT)
    assert study["analysis"]["absolute_noninferiority_margin"] == 0.002
    assert {candidate["component"] for candidate in study["validation_components"]} == {
        "target_eligibility",
        "adaptive_trees",
        "combined",
    }


def test_plan_binds_overrides_to_frozen_reference_and_is_idempotent(tmp_path: Path) -> None:
    root, selected_path, plan_path = _repository(tmp_path)
    first = plan_path.read_bytes()
    materialize_noninferiority_plan(
        repo_root=root,
        selected_config_path=selected_path,
        panel_path=root / "work/efficiency/panel.json",
        development_selection_path=root / "work/efficiency-development/selection.json",
        budget="reduced",
        output_path=plan_path,
    )
    assert plan_path.read_bytes() == first
    plan = json.loads(first)
    reference_policy = plan["reference"]["group_policy"]
    assert plan["reference"]["selection_validation_manifest_sha256"] == "4" * 64
    for candidate in plan["candidates"]:
        assert candidate["execution_overrides_sha256"] == canonical_json_sha256(
            candidate["execution_overrides"]
        )
        assert candidate["policy_sha256"] == canonical_json_sha256(candidate["group_policy"])
        for branch_id, reference_branch in reference_policy["branches"].items():
            candidate_parameters = candidate["group_policy"]["branches"][branch_id]["parameters"]
            assert candidate_parameters == {
                **reference_branch["parameters"],
                **candidate["execution_overrides"],
            }
            assert (
                candidate_parameters["n_estimators"]
                == reference_branch["parameters"]["n_estimators"]
            )
        assert candidate["binding"]["status"] == "bound"
        assert set(candidate["binding"]["parameters"]) == set(candidate["execution_overrides"])
    assert len(plan["candidates"]) == 1
    candidate = plan["candidates"][0]
    assert candidate["component"] == "combined"
    assert candidate["execution_overrides"] == {
        "target_eligibility": "automatic",
        "min_target_detected_cells": 20,
        "min_target_detected_fraction": 0.01,
        "min_target_weighted_detected_fraction": 0.01,
        "min_target_weighted_detected_ess": 10.0,
        "adaptive_trees": True,
        "adaptive_min_estimators": 100,
        "adaptive_tree_step": 50,
        "adaptive_tolerance": 0.01,
        "adaptive_patience": 2,
    }


@pytest.mark.parametrize(
    ("budget", "expected_decision"),
    (("reduced", "descriptive_only"), ("full", "reference_retained")),
)
def test_reference_only_efficiency_flow_is_explicit_and_valid(
    tmp_path: Path,
    budget: str,
    expected_decision: str,
) -> None:
    root, selected_path, plan_path = _repository(tmp_path, budget=budget)
    selection_path = root / "work/efficiency-development/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["nominee"] = None
    selection["reference_retained"] = True
    for result in selection["candidate_results"]:
        result["mean_aupr_loss"] = 0.01
        result["accuracy_eligible"] = False
    selection.pop("selection_sha256")
    selection["selection_sha256"] = canonical_json_sha256(selection)
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    plan_path.unlink()
    materialize_noninferiority_plan(
        repo_root=root,
        selected_config_path=selected_path,
        panel_path=root / "work/efficiency/panel.json",
        development_selection_path=selection_path,
        budget=budget,
        output_path=plan_path,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["candidates"] == []

    evidence = _evidence(plan_path)
    assert evidence["rows"] == []
    assert evidence["execution_failures"] == []
    evidence_path = _write_evidence(root, evidence)
    result_path, csv_path, _manifest_path = analyze_noninferiority(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=root / "work/efficiency/results",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["comparisons"] == []
    assert result["analysis"]["confirmatory_candidate_count"] == 0
    assert result["analysis"]["per_candidate_alpha"] is None
    assert result["analysis"]["one_sided_confidence_level"] is None
    assert result["adoption"]["decision"] == expected_decision
    assert result["adoption"]["selected_policy_id"] == plan["reference"]["policy_id"]
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 1


def test_analysis_uses_paired_hierarchical_one_sided_intervals(tmp_path: Path) -> None:
    root, _selected_path, plan_path = _repository(tmp_path, budget="full")
    evidence_path = _write_evidence(root, _evidence(plan_path))
    output_dir = root / "work/efficiency/results"
    paths = analyze_noninferiority(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=output_dir,
    )
    assert paths == (
        (output_dir / "results.json").resolve(),
        (output_dir / "comparisons.csv").resolve(),
        (output_dir / "manifest.json").resolve(),
    )
    output_manifest = json.loads(paths[2].read_text(encoding="utf-8"))
    output_entries = {entry["path"]: entry for entry in output_manifest["files"]}
    assert set(output_entries) == {"results.json", "comparisons.csv"}
    for name, entry in output_entries.items():
        assert entry["size"] == (output_dir / name).stat().st_size
        assert entry["sha256"] == sha256_file(output_dir / name)
    result = json.loads(paths[0].read_text(encoding="utf-8"))
    assert result["analysis_role"] == "confirmatory"
    assert result["global_decision"] == "policy_adopted"
    assert result["adoption"]["decision"] == "candidate_adopted"
    assert result["analysis"]["per_candidate_alpha"] == pytest.approx(0.05)
    assert result["analysis"]["one_sided_confidence_level"] == pytest.approx(0.95)
    assert result["analysis"]["algorithm_seed_treatment"] == (
        "fixed_predeclared_ensemble_mean_within_independent_dataset"
    )
    assert result["analysis"]["confidence_interval_method"] == (
        "one_sided_percentile_fixed_strata_hierarchical_bootstrap"
    )
    seed_plan = load_seed_plan(root / "experiments/v1/seeds/seed-plan.json")
    assert (
        validate_plan_seed_record(
            seed_plan,
            seed=result["seed_contract"]["root"]["seed"],
            seed_derivation=result["seed_contract"]["root"]["seed_derivation"],
        )
        == result["seed_contract"]["root"]
    )
    assert result["seed_contract"]["effective"]
    for record in result["seed_contract"]["effective"]:
        assert validate_plan_seed_record(
            seed_plan,
            seed=record["seed"],
            seed_derivation=record["seed_derivation"],
        ) == {"seed": record["seed"], "seed_derivation": record["seed_derivation"]}
    assert len(result["comparisons"]) == 1
    combined = result["comparisons"][0]
    assert combined["component"] == "combined"
    assert combined["coverage"]["target_fraction"]["passed"]
    assert combined["cost"]["runtime"]["hierarchical_mean_speedup"] == pytest.approx(4)
    assert combined["accuracy"]["target_universe"] == ("full_requested_target_universe")
    assert combined["intersection_diagnostic"]["interpretation"] == (
        "diagnostic_only_not_an_adoption_endpoint"
    )

    # Completed evidence is content-addressed and safely reusable.
    assert (
        analyze_noninferiority(
            repo_root=root,
            plan_path=plan_path,
            evidence_path=evidence_path,
            output_dir=output_dir,
        )
        == paths
    )


def test_bound_candidate_failure_retains_exact_without_inventing_bounds(
    tmp_path: Path,
) -> None:
    root, _selected_path, plan_path = _repository(tmp_path, budget="full")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    evidence = _evidence(plan_path)
    failed_id = plan["candidates"][0]["id"]
    failed_row = next(row for row in evidence["rows"] if row["candidate_id"] == failed_id)
    evidence["rows"] = [row for row in evidence["rows"] if row["candidate_id"] != failed_id]
    evidence["execution_failures"] = [
        {
            "dataset_id": failed_row["dataset_id"],
            "dataset_manifest_sha256": failed_row["dataset_manifest_sha256"],
            "generation_seed": failed_row["generation_seed"],
            "algorithm_seed": failed_row["algorithm_seed"],
            "configuration_id": failed_id,
            "selection_role": "candidate",
            "inference_artifact": "a" * 64,
            "failure_marker_sha256": "b" * 64,
            "run_report_sha256": "c" * 64,
            "tool_status": "failed",
            "failure_message": "model fit failed",
        }
    ]
    evidence_path = _write_evidence(root, evidence)

    result_path, csv_path, _manifest = analyze_noninferiority(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=root / "work/efficiency/results",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["global_decision"] == "reference_retained"
    assert result["adoption"]["selected_policy_id"] == plan["reference"]["policy_id"]
    assert result["execution_failures"] == evidence["execution_failures"]
    assert failed_id not in {item["candidate_id"] for item in result["comparisons"]}
    assert failed_id not in {
        record["candidate_id"] for record in result["seed_contract"]["effective"]
    }
    csv_candidate_ids = {
        line.split(",", 1)[0]
        for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]
        if line
    }
    assert failed_id not in csv_candidate_ids


def test_reduced_analysis_is_strictly_descriptive(tmp_path: Path) -> None:
    root, _selected_path, plan_path = _repository(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["budget"]["analysis_role"] == "descriptive"
    assert set(plan["budget"]) == {
        "id",
        "analysis_role",
        "algorithm_seeds",
        "execution_order_seed",
        "expected_independent_datasets",
        "replicates_per_stratum",
    }
    evidence_path = _write_evidence(root, _evidence(plan_path))
    result_path, _csv, _manifest = analyze_noninferiority(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=root / "work/efficiency/results",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["analysis_role"] == "descriptive"
    assert result["analysis"]["per_candidate_alpha"] is None
    assert result["analysis"]["one_sided_confidence_level"] is None
    assert "seed_contract" not in result
    assert result["global_decision"] == "descriptive_only"
    assert result["adoption"]["decision"] == "descriptive_only"
    for comparison in result["comparisons"]:
        assert comparison["decision"] == "descriptive_only"
        assert comparison["accuracy"]["one_sided_lower_confidence_bound"] is None
        assert comparison["accuracy"]["passed"] is None
        assert comparison["cost"]["runtime"]["one_sided_upper_confidence_bound"] is None
        assert comparison["cost"]["runtime"]["passed"] is None
        if comparison["coverage"] is not None:
            for endpoint in ("target_fraction", "truth_edge_fraction"):
                assert comparison["coverage"][endpoint]["passed"] is None
                assert comparison["coverage"][endpoint]["one_sided_lower_confidence_bound"] is None


def test_analysis_fails_noninferiority_or_coverage_without_hiding_it(tmp_path: Path) -> None:
    root, _selected_path, plan_path = _repository(tmp_path, budget="full")
    evidence = _evidence(plan_path)
    for row in evidence["rows"]:
        if row["candidate_id"] == "combined_scout_nominees":
            row["candidate_value"] = row["reference_value"] - 0.01
            row["target_coverage"]["n_retained_truth_edges"] = 70
            row["intersection_diagnostic"]["n_truth_edges"] = 70
    evidence_path = _write_evidence(root, evidence)
    result_path, _csv, _manifest = analyze_noninferiority(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=root / "work/efficiency/results",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    by_id = {item["candidate_id"]: item for item in result["comparisons"]}
    assert by_id["combined_scout_nominees"]["accuracy"]["passed"] is False
    assert by_id["combined_scout_nominees"]["coverage"]["truth_edge_fraction"]["passed"] is False
    assert result["global_decision"] == "reference_retained"


def test_empty_eligibility_is_scored_on_full_universe_and_fails_coverage(
    tmp_path: Path,
) -> None:
    root, _selected_path, plan_path = _repository(tmp_path, budget="full")
    evidence = _evidence(plan_path)
    candidate_id = "combined_scout_nominees"
    empty_targets = _digest("empty-target-universe")
    for row in evidence["rows"]:
        if row["candidate_id"] != candidate_id:
            continue
        row["candidate_space"]["candidate_target_universe_sha256"] = empty_targets
        row["candidate_space"]["n_candidate_targets"] = 0
        row["target_coverage"]["n_retained_truth_edges"] = 0
        row["intersection_diagnostic"] = None
        row["candidate_value"] = 0.0

    evidence_path = _write_evidence(root, evidence)
    result_path, _csv, _manifest = analyze_noninferiority(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=root / "work/efficiency/results",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = next(item for item in result["comparisons"] if item["candidate_id"] == candidate_id)

    assert candidate["accuracy"]["target_universe"] == "full_requested_target_universe"
    assert candidate["intersection_diagnostic"] is None
    assert candidate["coverage"]["target_fraction"]["estimate"] == 0.0
    assert candidate["coverage"]["truth_edge_fraction"]["estimate"] == 0.0
    assert candidate["coverage"]["target_fraction"]["passed"] is False
    assert candidate["decision"] == "not_adoptable"
    assert result["global_decision"] == "reference_retained"


def test_analysis_rejects_an_accurate_candidate_without_demonstrated_speedup(
    tmp_path: Path,
) -> None:
    root, _selected_path, plan_path = _repository(tmp_path, budget="full")
    evidence = _evidence(plan_path)
    candidate_id = "combined_scout_nominees"
    for row in evidence["rows"]:
        if row["candidate_id"] == candidate_id:
            row["candidate_value"] = row["reference_value"]
            row["candidate_measurements"]["wall_time_seconds"] = (
                row["reference_measurements"]["wall_time_seconds"] * 1.05
            )
    evidence_path = _write_evidence(root, evidence)
    result_path, csv_path, _manifest = analyze_noninferiority(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=root / "work/efficiency/results",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = next(item for item in result["comparisons"] if item["candidate_id"] == candidate_id)
    assert candidate["accuracy"]["passed"] is True
    assert candidate["cost"]["runtime"]["one_sided_upper_confidence_bound"] > 1
    assert candidate["cost"]["runtime"]["passed"] is False
    assert candidate["decision"] == "not_adoptable"
    assert "runtime_upper_confidence_bound" in csv_path.read_text(encoding="utf-8").splitlines()[0]


def test_primary_accuracy_cannot_be_replaced_by_better_intersection_accuracy(
    tmp_path: Path,
) -> None:
    root, _selected_path, plan_path = _repository(tmp_path, budget="full")
    evidence = _evidence(plan_path)
    candidate_id = "combined_scout_nominees"
    for row in evidence["rows"]:
        if row["candidate_id"] != candidate_id:
            continue
        row["candidate_value"] = row["reference_value"] - 0.01
        row["intersection_diagnostic"]["candidate_value"] = (
            row["intersection_diagnostic"]["reference_value"] + 0.05
        )
    evidence_path = _write_evidence(root, evidence)
    result_path, _csv, _manifest = analyze_noninferiority(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=root / "work/efficiency/results",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = next(item for item in result["comparisons"] if item["candidate_id"] == candidate_id)
    assert candidate["accuracy"]["passed"] is False
    assert candidate["intersection_diagnostic"]["estimate"] > 0
    assert candidate["decision"] == "not_adoptable"


def test_completion_is_a_mandatory_adoption_gate(tmp_path: Path) -> None:
    root, _selected_path, plan_path = _repository(tmp_path, budget="full")
    evidence = _evidence(plan_path)
    candidate_id = "combined_scout_nominees"
    for row in evidence["rows"]:
        if row["candidate_id"] == candidate_id:
            row["candidate_measurements"]["models_completed"] = 119
    evidence_path = _write_evidence(root, evidence)
    result_path, _csv, _manifest = analyze_noninferiority(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=root / "work/efficiency/results",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = next(item for item in result["comparisons"] if item["candidate_id"] == candidate_id)
    assert candidate["completion"]["observed_complete"] is False
    assert candidate["decision"] == "not_adoptable"


def test_confirmatory_adoption_is_deterministic_and_falls_back_to_reference(
    tmp_path: Path,
) -> None:
    root, _selected_path, plan_path = _repository(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["budget"]["analysis_role"] = "confirmatory"
    comparisons = [
        {
            "candidate_id": candidate["id"],
            "decision": "adoptable",
            "cost": {"runtime": {"hierarchical_mean_ratio": ratio}},
        }
        for candidate, ratio in zip(plan["candidates"], (0.4,), strict=True)
    ]
    adopted = decide_efficiency_adoption(plan, comparisons)
    tied = sorted(
        item["candidate_id"]
        for item in comparisons
        if item["cost"]["runtime"]["hierarchical_mean_ratio"] == 0.4
    )
    assert adopted["decision"] == "candidate_adopted"
    assert adopted["selected_candidate_id"] == tied[0]
    for item in comparisons:
        item["decision"] = "not_adoptable"
    retained = decide_efficiency_adoption(plan, comparisons)
    assert retained["decision"] == "reference_retained"
    assert retained["selected_policy_sha256"] == plan["reference"]["policy_sha256"]


def test_runtime_endpoint_uses_hierarchical_log_ratios() -> None:
    observations = [
        {
            "simulator_id": "sim",
            "condition_id": "condition",
            "runtime_log_ratio": math.log(ratio),
        }
        for ratio in (1.0, 4.0)
    ]
    point, _upper = _geometric_ratio_upper_bound(
        observations,
        alpha=0.05,
        resamples=1000,
        seed=123,
    )
    assert point == pytest.approx(2.0)
    assert point != pytest.approx(2.5)


def test_coverage_gate_requires_dataset_and_condition_guardrails() -> None:
    observations = [
        {
            "simulator_id": "sim",
            "condition_id": condition,
            "target_fraction": value,
        }
        for condition, value in (
            ("a", 0.49),
            ("a", 0.91),
            ("b", 0.91),
            ("b", 0.91),
        )
    ]
    endpoint = _coverage_endpoint_result(
        observations,
        field="target_fraction",
        gates={
            "minimum_target_fraction": 0.5,
            "minimum_each_dataset_target_fraction": 0.5,
            "minimum_each_condition_mean_target_fraction": 0.75,
        },
        inferential=True,
        alpha=0.05,
        resamples=1000,
        seed=123,
    )
    assert endpoint["one_sided_lower_confidence_bound"] >= 0.5
    assert endpoint["observed_dataset_guardrail_met"] is False
    assert endpoint["observed_condition_guardrail_met"] is False
    assert endpoint["passed"] is False


@pytest.mark.parametrize("drift", ["sources", "evaluation_targets", "combined_targets"])
def test_analysis_rejects_candidate_space_drift(tmp_path: Path, drift: str) -> None:
    root, _selected_path, plan_path = _repository(tmp_path)
    evidence = _evidence(plan_path)
    target = next(
        row for row in evidence["rows"] if row["candidate_id"] == "combined_scout_nominees"
    )
    if drift == "sources":
        target["candidate_space"]["candidate_source_universe_sha256"] = "f" * 64
    elif drift == "evaluation_targets":
        target["candidate_space"]["candidate_evaluation_target_universe_sha256"] = "e" * 64
    else:
        changed = _digest("different-eligible-targets")
        changed_space = _digest("different-candidate-space")
        for target in evidence["rows"]:
            if (
                target["candidate_id"] == "combined_scout_nominees"
                and target["dataset_id"] == "dyngen_a1"
            ):
                target["candidate_space"]["candidate_target_universe_sha256"] = changed
                target["candidate_space"]["reference_evaluation_target_universe_sha256"] = changed
                target["candidate_space"]["candidate_evaluation_target_universe_sha256"] = changed
                target["candidate_space"]["reference_evaluation_candidate_space_sha256"] = (
                    changed_space
                )
                target["candidate_space"]["candidate_evaluation_candidate_space_sha256"] = (
                    changed_space
                )
    evidence_path = _write_evidence(root, evidence)
    with pytest.raises(
        BenchmarkConfigError, match="universe|target|different targets|candidate spaces"
    ):
        analyze_noninferiority(
            repo_root=root,
            plan_path=plan_path,
            evidence_path=evidence_path,
            output_dir=root / "work/efficiency/results",
        )


def test_analysis_rejects_missing_algorithm_seed(tmp_path: Path) -> None:
    root, _selected_path, plan_path = _repository(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    omitted_seed = plan["budget"]["algorithm_seeds"][-1]
    evidence = _evidence(plan_path)
    evidence["rows"] = [
        row
        for row in evidence["rows"]
        if not (
            row["candidate_id"] == "combined_scout_nominees"
            and row["dataset_id"] == "dyngen_a1"
            and row["algorithm_seed"] == omitted_seed
        )
    ]
    evidence_path = _write_evidence(root, evidence)
    with pytest.raises(BenchmarkConfigError, match="algorithm seeds differ"):
        analyze_noninferiority(
            repo_root=root,
            plan_path=plan_path,
            evidence_path=evidence_path,
            output_dir=root / "work/efficiency/results",
        )


def test_analysis_rejects_an_incomplete_group_context_panel(tmp_path: Path) -> None:
    root, _selected_path, plan_path = _repository(tmp_path)
    evidence = _evidence(plan_path)
    evidence["rows"] = [
        row
        for row in evidence["rows"]
        if not (
            row["candidate_id"] == "combined_scout_nominees"
            and row["dataset_id"] == "dyngen_a1"
            and row["context_id"] == "group:b"
        )
    ]
    evidence_path = _write_evidence(root, evidence)
    with pytest.raises(BenchmarkConfigError, match="group-context panel|group-count policy branch"):
        analyze_noninferiority(
            repo_root=root,
            plan_path=plan_path,
            evidence_path=evidence_path,
            output_dir=root / "work/efficiency/results",
        )


@pytest.mark.parametrize("drift", ["validation_manifest", "incomplete_cartesian_space"])
def test_analysis_rejects_unbound_or_incomplete_evidence(tmp_path: Path, drift: str) -> None:
    root, _selected_path, plan_path = _repository(tmp_path)
    evidence = _evidence(plan_path)
    if drift == "validation_manifest":
        evidence["selection_validation_manifest_sha256"] = "f" * 64
        match = "validation manifest"
    else:
        evidence["rows"][0]["candidate_space"]["n_reference_evaluation_candidates"] -= 1
        evidence["rows"][0]["candidate_space"]["n_candidate_evaluation_candidates"] -= 1
        match = "complete directed"
    evidence_path = _write_evidence(root, evidence)
    with pytest.raises(BenchmarkConfigError, match=match):
        analyze_noninferiority(
            repo_root=root,
            plan_path=plan_path,
            evidence_path=evidence_path,
            output_dir=root / "work/efficiency/results",
        )


def test_stale_evidence_plan_digest_is_rejected(tmp_path: Path) -> None:
    root, _selected_path, plan_path = _repository(tmp_path)
    evidence = deepcopy(_evidence(plan_path))
    evidence["plan_sha256"] = "0" * 64
    evidence_path = _write_evidence(root, evidence)
    with pytest.raises(BenchmarkConfigError, match="different plan digest"):
        analyze_noninferiority(
            repo_root=root,
            plan_path=plan_path,
            evidence_path=evidence_path,
            output_dir=root / "work/efficiency/results",
        )
