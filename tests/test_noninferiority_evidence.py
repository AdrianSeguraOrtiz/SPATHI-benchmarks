from __future__ import annotations

import csv
import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.artifacts import (
    Artifact,
    ArtifactKey,
    ArtifactStore,
    canonical_json_sha256,
    sha256_file,
)
from spathi_bench.calibration import locked_spathi_identity, resolve_group_policy
from spathi_bench.config import BenchmarkConfigError, load_json_object
from spathi_bench.efficiency_order import (
    counterbalanced_execution_orders,
    runner_implementation_sha256,
)
from spathi_bench.noninferiority import (
    analyze_noninferiority,
    materialize_noninferiority_plan,
)
from spathi_bench.noninferiority_evidence import (
    construct_noninferiority_evidence,
    verify_noninferiority_evidence,
    write_noninferiority_evidence,
)
from spathi_bench.tool_contract import spathi_runtime_resources

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCKED_SPATHI = locked_spathi_identity(REPO_ROOT)
MODEL_COLUMNS = (
    "target_group",
    "target",
    "status",
    "random_seed",
    "n_samples",
    "n_positive_weight_samples",
    "weight_sum",
    "n_predictors_input",
    "n_predictors_used",
    "discarded_predictors_json",
    "constant_predictors_json",
    "n_edges",
    "importance_sum",
    "fit_seconds",
    "target_detected_cells",
    "target_detected_fraction",
    "target_weighted_detected_fraction",
    "target_weighted_detected_ess",
    "n_estimators_fitted",
    "adaptive_converged",
    "convergence_delta",
    "convergence_checks",
    "message",
)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _selected() -> dict[str, Any]:
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
    digest = canonical_json_sha256(parameters)
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
        "calibration_budget": "reduced",
        "status": "selected",
        "phase": "frozen",
        "selected_policy": {
            "policy_id": "spathi_conditional_v1",
            "source_candidate_id": "selected_spathi",
            "spathi_commit": LOCKED_SPATHI["spathi_commit"],
            "source_configuration_sha256": digest,
            "policy_sha256": policy_sha256,
            "group_policy": group_policy,
        },
        "selection_evidence": {
            "validation_ranking": "work/validation/ranking.json",
            "calibration_budget": "reduced",
            "validation_ranking_sha256": "2" * 64,
            "search_space_sha256": "3" * 64,
            "validation_manifest_sha256": "4" * 64,
            "benchmark_summary_sha256": "5" * 64,
            "spathi_build_fingerprint": LOCKED_SPATHI["spathi_build_fingerprint"],
            "spathi_image_id": "sha256:" + "7" * 64,
            "primary_metric": "aupr_directed",
            "hierarchical_score": 0.2,
            "winner_candidate_id": "selected_spathi",
            "winner_configuration_sha256": digest,
            "winner_policy_sha256": policy_sha256,
            "tie_breakers_applied": ["stability", "simplicity", "canonical_configuration_id"],
            "frozen_at": "2026-09-05T00:00:00Z",
        },
        "test_accessed_before_freeze": False,
        "notes": ["Verifier fixture."],
    }


def _publish_dataset(
    store: ArtifactStore,
    dataset_id: str,
    simulator: str,
    condition: str,
    seed: int,
) -> Artifact:
    spec = {
        "schema_version": "1.0",
        "dataset_role": "efficiency_validation",
        "condition_id": condition,
        "split_id": "efficiency_validation_v1",
    }
    # Include dataset_id in the immutable spec so fixture artifacts cannot alias.
    spec = spec | {"dataset_id": dataset_id}
    key = store.key_for("dataset", spec)
    with store.staging_directory(key) as staging:
        root = staging / "output" / "benchmark" / "datasets" / dataset_id
        (root / "extras").mkdir(parents=True)
        (root / "truth").mkdir()
        (root / "provenance").mkdir()
        cells = [f"c{i}" for i in range(1, 26)]
        with (root / "expression.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["gene", *cells])
            writer.writerow(["TF1", *(["0"] * 25)])
            writer.writerow(["G1", *([str(1 + index % 2) for index in range(25)])])
            writer.writerow(["G2", "1", *(["0"] * 24)])
            writer.writerow(["G3", *([str(2 + index % 2) for index in range(25)])])
        with (root / "extras" / "groups.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["column", "cluster"])
            for index, cell in enumerate(cells):
                writer.writerow([cell, "a" if index % 2 == 0 else "b"])
        (root / "extras" / "tf_list.txt").write_text("TF1\n", encoding="utf-8")
        (root / "truth" / "gene_universe.txt").write_text("TF1\nG1\nG2\nG3\n", encoding="utf-8")
        with (root / "truth" / "networks.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["source", "target", "score", "sign", "evidence", "context"])
            for context in ("group:a", "group:b"):
                writer.writerow(["TF1", "G1", "1", "+", "simulated_truth", context])
                writer.writerow(["TF1", "G2", "1", "+", "simulated_truth", context])
                writer.writerow(["G3", "G1", "1", "+", "simulated_truth", context])
        fingerprint = canonical_json_sha256({"dataset": dataset_id})
        _json(
            root / "dataset-manifest.json",
            {"dataset": {"spec": {"id": dataset_id}}},
        )
        _json(
            root / "ground-truth-manifest.json",
            {
                "dataset_id": dataset_id,
                "simulator_id": simulator,
                "dataset_fingerprint": {"algorithm": "sha256", "value": fingerprint},
            },
        )
        _json(
            root / "provenance" / "simulator-run.json",
            {"dataset_id": dataset_id, "simulator_id": simulator, "seed": seed},
        )
        return store.publish_from_temp(staging, kind="dataset", spec=spec)


def _write_eligibility(path: Path, *, automatic: bool) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "target",
                "mode",
                "eligible",
                "detected_cells",
                "detected_fraction",
                "expression_min",
                "expression_max",
                "required_detected_cells",
                "reason",
            ]
        )
        if not automatic:
            for target in ("TF1", "G1", "G2", "G3"):
                writer.writerow([target, "all", True, "", "", "", "", "", "unfiltered_reference"])
            return
        writer.writerow(
            ["TF1", "automatic", False, 0, 0.0, 0.0, 0.0, 20, "insufficient_detected_cells"]
        )
        writer.writerow(["G1", "automatic", True, 25, 1.0, 1.0, 2.0, 20, "eligible"])
        writer.writerow(
            ["G2", "automatic", False, 1, 0.04, 0.0, 1.0, 20, "insufficient_detected_cells"]
        )
        writer.writerow(["G3", "automatic", True, 25, 1.0, 2.0, 3.0, 20, "eligible"])


def _write_diagnostics(path: Path, *, parameters: dict[str, Any], seed: int) -> None:
    automatic = parameters["target_eligibility"] == "automatic"
    targets = ("G1", "G3") if automatic else ("TF1", "G1", "G2", "G3")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MODEL_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for group in ("a", "b"):
            for target in targets:
                fitted = 100 if parameters["adaptive_trees"] else 250
                trained = target != "TF1"
                writer.writerow(
                    {
                        "target_group": group,
                        "target": target,
                        "status": "trained" if trained else "no_predictors_after_self_exclusion",
                        "random_seed": seed,
                        "n_samples": 25,
                        "n_positive_weight_samples": 25,
                        "weight_sum": 25,
                        "n_predictors_input": 1,
                        "n_predictors_used": 1,
                        "discarded_predictors_json": "[]",
                        "constant_predictors_json": "[]",
                        "n_edges": int(trained),
                        "importance_sum": int(trained),
                        "fit_seconds": 0.1,
                        "target_detected_cells": 25,
                        "target_detected_fraction": 1.0,
                        "target_weighted_detected_fraction": 1.0 if automatic else "",
                        "target_weighted_detected_ess": 20.0 if automatic else "",
                        "n_estimators_fitted": fitted if trained else 0,
                        "adaptive_converged": parameters["adaptive_trees"] and trained,
                        "convergence_delta": (
                            0.001 if parameters["adaptive_trees"] and trained else ""
                        ),
                        "convergence_checks": 2 if parameters["adaptive_trees"] and trained else 0,
                        "message": "",
                    }
                )


def _andrea_execution_measurement(
    *, execution: dict[str, Any], runtime: float, output_bytes: int
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scope": "andrea_execution",
        "wall_time_seconds": runtime,
        "cpu_time_seconds": runtime / 2,
        "peak_memory_bytes": 100_000_000,
        "output_bytes": output_bytes,
        "assigned_resources": {
            "threads_peak": execution["resources"]["cpu_threads"],
            "ram_limit_peak_bytes": execution["resources"]["memory_bytes"],
            "cpuset_cpus_requested_sets": [],
            "cpuset_cpus_effective_sets": [],
        },
        "telemetry": {
            "status": "complete",
            "wall_source": "andrea_monotonic_clock",
            "wall_semantics": (
                "andrea_execution_end_to_end_including_input_verification_preparation_"
                "scheduling_merging_and_requested_output_materialization"
            ),
            "memory_source": "child_container_cgroup_peaks",
            "memory_semantics": (
                "overlap_upper_bound_from_physical_container_peaks_excludes_andrea_host_process"
            ),
            "oom_killed": False,
        },
    }


def _publish_inference(
    store: ArtifactStore,
    *,
    dataset: Artifact,
    dataset_id: str,
    fingerprint: str,
    contexts: list[str],
    configuration_id: str,
    policy_sha256: str,
    selection_role: str,
    parameters: dict[str, Any],
    seed: int,
    seed_derivation: dict[str, Any],
    runtime: float,
    execution: dict[str, Any],
    execution_order: list[str],
    failure: bool = False,
    truth_leak: bool = False,
    provenance_overrides: dict[str, Any] | None = None,
) -> Artifact:
    run_id = f"{configuration_id}__seed_{seed}"
    order_label = "reference" if selection_role == "reference" else configuration_id
    spec = {
        "schema_version": "1.0",
        "dataset_artifact": dataset.key.value,
        "dataset_id": dataset_id,
        "andrea_commit": execution["andrea_commit"],
        "andrea_environment": execution["andrea_environment"],
        "benchmark_environment": execution["benchmark_environment"],
        "docker_runtime": execution["docker_runtime"],
        "batch_kind": "spathi_efficiency",
        "run": {
            "run_id": run_id,
            "tool_id": f"custom_{run_id}",
            "params": {**parameters, "random_seed": seed},
            "execution": {"mode": "group_native"},
            "resources": {
                "threads": execution["resources"]["cpu_threads"],
                "ram_gb": execution["resources"]["memory_gib"],
            },
        },
        "run_metadata": {
            "configuration_id": configuration_id,
            "configuration_sha256": canonical_json_sha256(parameters),
            "policy_sha256": policy_sha256,
            "group_count": len(contexts),
            "selection_role": selection_role,
            "algorithm_seed": seed,
            "algorithm_seed_parameter": "random_seed",
            "seed_derivation": seed_derivation,
            "family": "spathi",
        },
        "custom_definition": {
            "run_id": run_id,
            "name": configuration_id,
            "docker_image": execution["spathi_image_id"],
            "execution_mode": "group_native",
            "extra_inputs": ["groups", "tf_list"],
            "outputs": {"directed": True, "sign": "none"},
            "runtime_resources": spathi_runtime_resources(),
        },
        "tool_image": {
            "kind": "spathi",
            "image_id": execution["spathi_image_id"],
            "source_commit": execution["spathi_commit"],
            "build_fingerprint": execution["spathi_build_fingerprint"],
        },
        "resources": execution["resources"],
        "measurement_protocol": execution["measurement_protocol"],
        "runner_sha256": execution["runner_sha256"],
        "execution_order_block": {
            "dataset_id": dataset_id,
            "algorithm_seed": seed,
            "execution_order_seed": execution["execution_order_seed"],
            "execution_order": execution_order,
        },
        "execution_position": execution_order.index(order_label) + 1,
    }
    if truth_leak:
        spec["audit_fixture_nonce"] = "truth-leak"
    if provenance_overrides:
        spec.update(provenance_overrides)
    key = store.key_for("inference", spec)
    with store.staging_directory(key) as staging:
        output = staging / "output" / "run"
        if failure:
            measurement = _andrea_execution_measurement(
                execution=execution,
                runtime=runtime,
                output_bytes=0,
            )
            _json(
                output / "run_report.json",
                {
                    "status": "executed",
                    "output_profile": "full",
                    "dataset": {
                        "id": dataset_id,
                        "fingerprint": {"algorithm": "sha256", "value": fingerprint},
                    },
                    "execution": {"measurement": measurement},
                    "tools": {
                        "selected": [run_id],
                        "completed": [],
                        "status_by_tool": {run_id: "failed"},
                        "tool_origins": {run_id: "custom"},
                        "failed": {run_id: "model fit failed"},
                        "results": {
                            run_id: {
                                "measurement": {
                                    "schema_version": "1.0",
                                    "scope": "logical_run",
                                    "telemetry": {"status": "complete", "oom_killed": False},
                                },
                            }
                        },
                    },
                    "run_id": run_id,
                },
            )
            report_path = output / "run_report.json"
            _json(
                staging / "efficiency-execution-failure.json",
                {
                    "schema_version": "1.0",
                    "status": "tool_execution_failed",
                    "dataset_id": dataset_id,
                    "run_id": run_id,
                    "tool_status": "failed",
                    "failure_message": "model fit failed",
                    "run_report_sha256": sha256_file(report_path),
                    "measurement_sha256": canonical_json_sha256(
                        load_json_object(report_path)["execution"]["measurement"]
                    ),
                    "wall_time_seconds": runtime,
                    "peak_memory_bytes": 100_000_000,
                    "oom_killed": False,
                },
            )
            return store.publish_from_temp(staging, kind="inference", spec=spec)
        inputs = output / "input"
        extra = inputs / "extra"
        tool_out = output / "tools" / "spathi" / "io" / "out"
        extra.mkdir(parents=True)
        tool_out.mkdir(parents=True)
        source_root = next(dataset.path.glob(f"output/*/datasets/{dataset_id}"))
        shutil.copyfile(source_root / "expression.tsv", inputs / "expression.tsv")
        shutil.copyfile(source_root / "extras/groups.tsv", extra / "groups.tsv")
        shutil.copyfile(source_root / "extras/tf_list.txt", extra / "tf_list.txt")
        with (tool_out / "network.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["source", "target", "score", "sign", "evidence", "context"])
            for context in contexts:
                writer.writerow(["TF1", "G1", 0.9, "?", "feature_importance", context])
                writer.writerow(["TF1", "G3", 0.1, "?", "feature_importance", context])
                if parameters.get("target_eligibility", "all") == "all":
                    writer.writerow(["TF1", "G2", 0.8, "?", "feature_importance", context])
        _write_eligibility(
            tool_out / "target_eligibility.tsv.gz",
            automatic=parameters["target_eligibility"] == "automatic",
        )
        _write_diagnostics(tool_out / "model_diagnostics.tsv.gz", parameters=parameters, seed=seed)
        requested = {**parameters, "random_seed": seed, "target_list": None}
        automatic = parameters["target_eligibility"] == "automatic"
        target_count = 2 if automatic else 4
        models_requested = target_count * len(contexts)
        models_trained = models_requested if automatic else 3 * len(contexts)
        trees_per_model = 100 if parameters["adaptive_trees"] else 250
        _json(
            tool_out / "run_metadata.json",
            {
                "status": "complete",
                "random_seed": seed,
                "requested_parameters": requested,
                "inputs": {
                    name: {"sha256": sha256_file(path)}
                    for name, path in {
                        "expression": inputs / "expression.tsv",
                        "groups": extra / "groups.tsv",
                        "tf_list": extra / "tf_list.txt",
                    }.items()
                },
                "input_dimensions": {"targets": 4},
                "effective_parameters": {
                    "target_eligibility": {"globally_eligible_targets": target_count}
                },
                "models": {
                    "requested": models_requested,
                    "completed": models_requested,
                    "trained": models_trained,
                    "fitted_estimators_total": models_trained * trees_per_model,
                },
            },
        )
        if truth_leak:
            (output / "truth").mkdir()
            (output / "truth" / "networks.csv").write_text("leak\n", encoding="utf-8")
        output_bytes = sum(path.stat().st_size for path in tool_out.rglob("*") if path.is_file())
        measurement = _andrea_execution_measurement(
            execution=execution,
            runtime=runtime,
            output_bytes=output_bytes,
        )
        _json(
            output / "run_report.json",
            {
                "status": "executed",
                "output_profile": "full",
                "dataset": {
                    "id": dataset_id,
                    "fingerprint": {"algorithm": "sha256", "value": fingerprint},
                },
                "execution": {"measurement": measurement},
                "tools": {
                    "selected": [run_id],
                    "completed": [run_id],
                    "status_by_tool": {run_id: "completed"},
                    "completed_contexts": {run_id: contexts},
                    "tool_origins": {run_id: "custom"},
                    "failed": {},
                    "results": {
                        run_id: {
                            "measurement": {
                                "schema_version": "1.0",
                                "scope": "logical_run",
                                "telemetry": {"status": "complete", "oom_killed": False},
                            },
                        }
                    },
                },
                "run_id": run_id,
            },
        )
        return store.publish_from_temp(staging, kind="inference", spec=spec)


def _repository(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
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
        "schemas/spathi-selected-config.schema.json",
        "experiments/v1/spathi/efficiency-study.json",
        "experiments/v1/spathi/efficiency-design-sensitivity.json",
        "experiments/v1/spathi/efficiency-development.json",
        "experiments/v1/profiles/efficiency-development-reduced.json",
        "experiments/v1/profiles/efficiency-development-full.json",
        "experiments/v1/profiles/efficiency-reduced.json",
        "experiments/v1/profiles/efficiency-full.json",
        "experiments/v1/simulations/dyngen/efficiency_development/linear-baseline-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_development/linear-noisy-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_development/bifurcating-baseline-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_development/bifurcating-noisy-reduced-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_development/dynamic-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_development/linear-baseline-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_development/linear-noisy-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_development/bifurcating-baseline-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_development/bifurcating-noisy-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_development/dynamic-runs.json",
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
    study_path = root / "experiments/v1/spathi/efficiency-study.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["analysis"]["bootstrap_resamples"] = 1000
    _json(study_path, study)
    ranking_path = root / "work/validation/ranking.json"
    _json(
        ranking_path,
        {
            "stage_id": "validation",
            "dataset_role": "validation",
            "calibration_rows": [{"dataset_id": "selection_dataset"}],
        },
    )
    selected_path = root / "work/calibration/selected-config.json"
    selected = _selected()
    selected["selection_evidence"]["validation_ranking_sha256"] = sha256_file(ranking_path)
    _json(selected_path, selected)
    development_contract_path = root / "experiments/v1/spathi/efficiency-development.json"
    development_contract = json.loads(development_contract_path.read_text(encoding="utf-8"))
    development_metrics = root / "work/efficiency-development/metrics.csv"
    development_metrics.parent.mkdir(parents=True, exist_ok=True)
    development_metrics.write_text("fixture\n", encoding="utf-8")
    development_state = root / "work/efficiency-development/state.json"
    _json(development_state, {"status": "completed"})
    component_selection_path = root / "work/efficiency-development/component-selection.json"
    _json(component_selection_path, {"status": "completed"})
    selected_development_ids = {
        "eligible_targets_detect1pct_min20_wdetect1pct_ess10",
        "adaptive_trees_min100_step50_tol01_patience2",
    }
    primitive_nominees: list[dict[str, Any]] = []
    for candidate in development_contract["candidates"]:
        selected_candidate = candidate["id"] in selected_development_ids
        if not selected_candidate:
            continue
        result = {
            "candidate_id": candidate["id"],
            "component": candidate["component"],
            "mean_aupr_loss": 0.001,
            "wall_time_ratio": 0.5,
            "complete": True,
            "accuracy_eligible": True,
            "dataset_count": development_contract["budgets"]["reduced"][
                "expected_independent_datasets"
            ],
            "accuracy_seed_count": 1,
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
        "dataset_count": development_contract["budgets"]["reduced"][
            "expected_independent_datasets"
        ],
        "accuracy_seed_count": 1,
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
        "budget": "reduced",
        "analysis_role": development_contract["budgets"]["reduced"]["analysis_role"],
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
    _json(development_selection_path, development_selection)
    store = ArtifactStore(root / "work/v1/artifacts")
    declarations: list[dict[str, Any]] = []
    artifacts: dict[str, Artifact] = {}
    for dataset_id, simulator, condition, seed in (
        ("dataset_a", "dyngen", "linear_baseline", 101),
        ("dataset_b", "dyngen", "linear_noisy", 102),
        ("dataset_c", "dyngen", "bifurcating_baseline", 103),
        ("dataset_d", "dyngen", "bifurcating_noisy", 104),
        ("dataset_e", "scmultisim", "scmultisim_dynamic", 202),
    ):
        artifact = _publish_dataset(store, dataset_id, simulator, condition, seed)
        artifacts[dataset_id] = artifact
        declarations.append(
            {
                "dataset_id": dataset_id,
                "simulator_id": simulator,
                "condition_id": condition,
                "generation_seed": seed,
                "contexts": ["group:a", "group:b"],
                "dataset_artifact": {"kind": "dataset", "digest": artifact.key.digest},
            }
        )
    panel_path = root / "work/efficiency/panel.json"
    _json(
        panel_path,
        {
            "schema_version": "1.0",
            "artifact_type": "spathi_efficiency_predeclared_panel",
            "selection_validation_manifest_sha256": "4" * 64,
            "selection_validation_ranking": {
                "path": "work/validation/ranking.json",
                "sha256": sha256_file(ranking_path),
            },
            "budget": "reduced",
            "dataset_role": "efficiency_validation",
            "split_id": "efficiency_validation_v1",
            "datasets": declarations,
        },
    )
    plan_path = root / "work/efficiency/plan.json"
    materialize_noninferiority_plan(
        repo_root=root,
        selected_config_path=selected_path,
        panel_path=panel_path,
        development_selection_path=development_selection_path,
        budget="reduced",
        output_path=plan_path,
        study_path=study_path,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    execution = {
        "andrea_commit": "a" * 40,
        "andrea_environment": "b" * 64,
        "benchmark_environment": "c" * 64,
        "docker_runtime": "d" * 64,
        "spathi_commit": plan["reference"]["spathi_commit"],
        "spathi_build_fingerprint": plan["reference"]["spathi_build_fingerprint"],
        "spathi_image_id": plan["reference"]["spathi_image_id"],
        "execution_order_seed": plan["budget"]["execution_order_seed"],
        "execution_order_seed_derivation": {
            "namespace": f"{plan['budget']['id']}.efficiency_validation",
            "stream": "execution_order",
            "coordinates": {"purpose": "counterbalanced_execution_order"},
        },
        "resources": {
            "cpu_threads": 1,
            "memory_bytes": 4 * 1024**3,
            "memory_gib": 4.0,
            "planner": "heuristic",
            "planner_time_limit_seconds": 30,
            "output_profile": "full",
            "execution_plan_fingerprint": "e" * 64,
        },
        "measurement_protocol": {
            "source": "andrea_execution_measurement",
            "scope": "andrea_execution",
            "wall_time": (
                "andrea_execution_end_to_end_including_input_verification_preparation_"
                "scheduling_merging_and_requested_output_materialization"
            ),
            "peak_memory": "physical_container_peak_overlap_upper_bound_excluding_andrea_host",
            "output_bytes": "andrea_execution_output_tree_at_measurement",
            "telemetry_status": "complete_required",
            "model_counts": "validated_spathi_run_metadata",
            "tree_counts": "validated_spathi_model_diagnostics",
        },
        "runner_sha256": runner_implementation_sha256(),
    }
    runs: list[dict[str, Any]] = []
    execution_orders = counterbalanced_execution_orders(
        (item["dataset_id"] for item in declarations),
        plan["budget"]["algorithm_seeds"],
        ["reference", *sorted(item["id"] for item in plan["candidates"])],
        execution_order_seed=plan["budget"]["execution_order_seed"],
    )
    for declaration in declarations:
        dataset_id = declaration["dataset_id"]
        dataset = artifacts[dataset_id]
        fingerprint = canonical_json_sha256({"dataset": dataset_id})
        for seed in plan["budget"]["algorithm_seeds"]:
            replicate_index = plan["budget"]["algorithm_seeds"].index(seed) + 1
            seed_derivation = {
                "namespace": f"{plan['budget']['id']}.efficiency_validation",
                "stream": "inference",
                "coordinates": {
                    "purpose": "algorithm_replicate",
                    "replicate_index": replicate_index,
                },
            }
            group_count = len(declaration["contexts"])
            reference = _publish_inference(
                store,
                dataset=dataset,
                dataset_id=dataset_id,
                fingerprint=fingerprint,
                contexts=declaration["contexts"],
                configuration_id=plan["reference"]["policy_id"],
                policy_sha256=plan["reference"]["policy_sha256"],
                selection_role="reference",
                parameters=resolve_group_policy(plan["reference"]["group_policy"], group_count),
                seed=seed,
                seed_derivation=seed_derivation,
                runtime=10.0,
                execution=execution,
                execution_order=execution_orders[(dataset_id, seed)],
            )
            candidates = []
            for candidate in plan["candidates"]:
                inferred = _publish_inference(
                    store,
                    dataset=dataset,
                    dataset_id=dataset_id,
                    fingerprint=fingerprint,
                    contexts=declaration["contexts"],
                    configuration_id=candidate["id"],
                    policy_sha256=candidate["policy_sha256"],
                    selection_role="candidate",
                    parameters=resolve_group_policy(candidate["group_policy"], group_count),
                    seed=seed,
                    seed_derivation=seed_derivation,
                    runtime=5.0,
                    execution=execution,
                    execution_order=execution_orders[(dataset_id, seed)],
                )
                candidates.append(
                    {
                        "candidate_id": candidate["id"],
                        "inference_artifact": {
                            "kind": "inference",
                            "digest": inferred.key.digest,
                        },
                    }
                )
            runs.append(
                {
                    "dataset_id": dataset_id,
                    "algorithm_seed": seed,
                    "seed_derivation": seed_derivation,
                    "reference": {"kind": "inference", "digest": reference.key.digest},
                    "candidates": candidates,
                    "execution_order": execution_orders[(dataset_id, seed)],
                }
            )
    index_path = root / "work/efficiency/artifact-index.json"
    _json(
        index_path,
        {
            "schema_version": "1.0",
            "artifact_type": "spathi_efficiency_artifact_index",
            "plan_sha256": sha256_file(plan_path),
            "panel_sha256": sha256_file(panel_path),
            "artifact_store": "work/v1/artifacts",
            "execution": execution,
            "runs": runs,
        },
    )
    return root, plan_path, panel_path, index_path


def test_builder_recalculates_metrics_spaces_hashes_and_shared_references(tmp_path: Path) -> None:
    root, plan_path, _panel, index_path = _repository(tmp_path)
    evidence = construct_noninferiority_evidence(
        repo_root=root,
        plan_path=plan_path,
        artifact_index_path=index_path,
    )

    assert len(evidence["rows"]) == 20
    assert {row["reference_value"] for row in evidence["rows"]} == {1.0}
    candidate_values = sorted({row["candidate_value"] for row in evidence["rows"]})
    assert candidate_values == pytest.approx([5 / 6])
    per_run: dict[tuple[str, int], set[str]] = {}
    for row in evidence["rows"]:
        per_run.setdefault((row["dataset_id"], row["algorithm_seed"]), set()).add(
            row["source_artifacts"]["reference_inference"]
        )
        assert row["source_artifacts"]["dataset"]
        assert row["candidate_space"]["n_reference_evaluation_candidates"] == 3
        assert row["candidate_space"]["policy"] == "full_requested_universe"
        assert row["candidate_space"]["n_truth_edges_in_candidate_space"] == 2
        assert row["candidate_space"]["n_truth_edges_outside_candidate_space"] == 1
        assert row["candidate_space"]["truth_outside_candidate_space_examples"] == [
            "G3->G1"
        ]
        assert row["reference_measurements"]["peak_memory_bytes"] == 100_000_000
        assert row["candidate_measurements"]["models_requested"] == 4
    assert all(len(references) == 1 for references in per_run.values())


def test_builder_preserves_a_bound_formal_candidate_failure_as_evidence(
    tmp_path: Path,
) -> None:
    root, plan_path, panel_path, index_path = _repository(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = index["runs"][0]
    candidate_reference = indexed["candidates"][0]
    candidate = next(
        item for item in plan["candidates"] if item["id"] == candidate_reference["candidate_id"]
    )
    declaration = next(
        item for item in panel["datasets"] if item["dataset_id"] == indexed["dataset_id"]
    )
    store = ArtifactStore(root / index["artifact_store"])
    dataset = store.validate(
        ArtifactKey(
            kind="dataset",
            digest=declaration["dataset_artifact"]["digest"],
        )
    )
    failed = _publish_inference(
        store,
        dataset=dataset,
        dataset_id=indexed["dataset_id"],
        fingerprint=canonical_json_sha256({"dataset": indexed["dataset_id"]}),
        contexts=declaration["contexts"],
        configuration_id=candidate["id"],
        policy_sha256=candidate["policy_sha256"],
        selection_role="candidate",
        parameters=resolve_group_policy(candidate["group_policy"], len(declaration["contexts"])),
        seed=indexed["algorithm_seed"],
        seed_derivation=indexed["seed_derivation"],
        runtime=0.5,
        execution=index["execution"],
        execution_order=indexed["execution_order"],
        failure=True,
        provenance_overrides={"audit_fixture_nonce": "bound-candidate-failure"},
    )
    candidate_reference["inference_artifact"] = {
        "kind": "inference",
        "digest": failed.key.digest,
    }
    _json(index_path, index)

    evidence = construct_noninferiority_evidence(
        repo_root=root,
        plan_path=plan_path,
        artifact_index_path=index_path,
    )

    assert evidence["execution_failures"] == [
        {
            "dataset_id": indexed["dataset_id"],
            "dataset_manifest_sha256": evidence["execution_failures"][0]["dataset_manifest_sha256"],
            "generation_seed": declaration["generation_seed"],
            "algorithm_seed": indexed["algorithm_seed"],
            "configuration_id": candidate["id"],
            "selection_role": "candidate",
            "inference_artifact": failed.key.digest,
            "failure_marker_sha256": evidence["execution_failures"][0]["failure_marker_sha256"],
            "run_report_sha256": evidence["execution_failures"][0]["run_report_sha256"],
            "tool_status": "failed",
            "failure_message": "model fit failed",
        }
    ]
    assert not any(
        row["candidate_id"] == candidate["id"]
        and row["dataset_id"] == indexed["dataset_id"]
        and row["algorithm_seed"] == indexed["algorithm_seed"]
        for row in evidence["rows"]
    )


def test_builder_rejects_an_algorithm_seed_with_drifted_semantic_coordinates(
    tmp_path: Path,
) -> None:
    root, plan_path, _panel, index_path = _repository(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["runs"][0]["seed_derivation"]["coordinates"]["replicate_index"] += 1
    _json(index_path, index)

    with pytest.raises(BenchmarkConfigError, match="seed derivation differs"):
        construct_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
        )


def test_builder_rejects_index_from_another_efficiency_runner(tmp_path: Path) -> None:
    root, plan_path, _panel_path, index_path = _repository(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["execution"]["runner_sha256"] = "d" * 64
    _json(index_path, index)

    with pytest.raises(BenchmarkConfigError, match="different efficiency runner"):
        construct_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
        )


@pytest.mark.parametrize(
    "tampered_field",
    [
        "benchmark_environment",
        "docker_runtime",
        "runner_sha256",
        "execution_order_block",
        "execution_position",
    ],
)
def test_builder_rejects_tampered_temporal_execution_provenance(
    tmp_path: Path,
    tampered_field: str,
) -> None:
    root, plan_path, panel_path, index_path = _repository(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = index["runs"][0]
    indexed_candidate = indexed["candidates"][0]
    candidate = next(
        item for item in plan["candidates"] if item["id"] == indexed_candidate["candidate_id"]
    )
    declaration = next(
        item for item in panel["datasets"] if item["dataset_id"] == indexed["dataset_id"]
    )
    store = ArtifactStore(root / "work/v1/artifacts")
    dataset = store.validate(
        store.key_for(
            "dataset",
            {
                "schema_version": "1.0",
                "dataset_role": "efficiency_validation",
                "condition_id": declaration["condition_id"],
                "split_id": "efficiency_validation_v1",
                "dataset_id": declaration["dataset_id"],
            },
        )
    )
    order = indexed["execution_order"]
    correct_position = order.index(candidate["id"]) + 1
    correct_block = {
        "dataset_id": declaration["dataset_id"],
        "algorithm_seed": indexed["algorithm_seed"],
        "execution_order_seed": index["execution"]["execution_order_seed"],
        "execution_order": order,
    }
    overrides: dict[str, Any]
    if tampered_field == "benchmark_environment":
        overrides = {"benchmark_environment": "e" * 64}
    elif tampered_field == "docker_runtime":
        overrides = {"docker_runtime": "f" * 64}
    elif tampered_field == "runner_sha256":
        overrides = {"runner_sha256": "d" * 64}
    elif tampered_field == "execution_order_block":
        overrides = {"execution_order_block": {**correct_block, "execution_order": order[::-1]}}
    else:
        overrides = {"execution_position": 1 if correct_position != 1 else 2}
    replacement = _publish_inference(
        store,
        dataset=dataset,
        dataset_id=declaration["dataset_id"],
        fingerprint=canonical_json_sha256({"dataset": declaration["dataset_id"]}),
        contexts=declaration["contexts"],
        configuration_id=candidate["id"],
        policy_sha256=candidate["policy_sha256"],
        selection_role="candidate",
        parameters=resolve_group_policy(candidate["group_policy"], len(declaration["contexts"])),
        seed=indexed["algorithm_seed"],
        seed_derivation=indexed["seed_derivation"],
        runtime=5.0,
        execution=index["execution"],
        execution_order=order,
        provenance_overrides=overrides,
    )
    indexed_candidate["inference_artifact"]["digest"] = replacement.key.digest
    _json(index_path, index)

    with pytest.raises(BenchmarkConfigError, match="execution provenance"):
        construct_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
        )


def test_written_evidence_is_rebuilt_before_analysis_and_rejects_manual_metrics(
    tmp_path: Path,
) -> None:
    root, plan_path, _panel, index_path = _repository(tmp_path)
    evidence_path = root / "work/efficiency/evidence.json"
    assert (
        write_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
            output_path=evidence_path,
        )
        == evidence_path.resolve()
    )
    verified = verify_noninferiority_evidence(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
    )
    assert verified["rows"]
    result_path, _csv, _manifest = analyze_noninferiority(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=root / "work/efficiency/results",
    )
    assert result_path.is_file()

    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["rows"][0]["candidate_value"] = 0.0
    _json(evidence_path, payload)
    with pytest.raises(BenchmarkConfigError, match="does not exactly match"):
        verify_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            evidence_path=evidence_path,
        )


@pytest.mark.parametrize("omission", ["run", "candidate"])
def test_builder_rejects_incomplete_predeclared_product(tmp_path: Path, omission: str) -> None:
    root, plan_path, _panel, index_path = _repository(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if omission == "run":
        index["runs"].pop()
    else:
        index["runs"][0]["candidates"].pop()
    _json(index_path, index)

    with pytest.raises(BenchmarkConfigError, match="exact predeclared|incomplete candidates"):
        construct_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
        )


def test_builder_rejects_tampered_content_addressed_artifact(tmp_path: Path) -> None:
    root, plan_path, _panel, index_path = _repository(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    digest = index["runs"][0]["reference"]["digest"]
    network = next(
        (root / "work/v1/artifacts/inference" / digest).glob("output/*/tools/*/io/out/network.csv")
    )
    network.write_text(network.read_text(encoding="utf-8") + "corrupt\n", encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="hash differs|size differs"):
        construct_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
        )


def test_builder_still_rejects_truth_genes_outside_the_gene_universe(tmp_path: Path) -> None:
    root, plan_path, panel_path, index_path = _repository(tmp_path)
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    digest = panel["datasets"][0]["dataset_artifact"]["digest"]
    artifact_root = root / "work/v1/artifacts/dataset" / digest
    truth_path = next(artifact_root.glob("output/*/datasets/*/truth/networks.csv"))
    with truth_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["UNKNOWN", "G1", 1, "+", "simulated_truth", "group:a"])

    # Repair the immutable inventory so the assertion isolates semantic truth validation.
    manifest_path = artifact_root / "artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_truth = truth_path.relative_to(artifact_root).as_posix()
    entry = next(item for item in manifest["files"] if item["path"] == relative_truth)
    entry["size"] = truth_path.stat().st_size
    entry["sha256"] = sha256_file(truth_path)
    _json(manifest_path, manifest)

    with pytest.raises(BenchmarkConfigError, match="outside the declared gene universe"):
        construct_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
        )


def test_builder_requires_exact_canonical_ram_limit_telemetry(tmp_path: Path) -> None:
    root, plan_path, _panel, index_path = _repository(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    digest = index["runs"][0]["reference"]["digest"]
    artifact_root = root / "work/v1/artifacts/inference" / digest
    report_path = next(artifact_root.glob("output/*/run_report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["execution"]["measurement"]["assigned_resources"]["ram_limit_peak_bytes"]
    _json(report_path, report)

    # Repair the immutable file inventory so this test isolates semantic telemetry validation.
    manifest_path = artifact_root / "artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_report = report_path.relative_to(artifact_root).as_posix()
    entry = next(item for item in manifest["files"] if item["path"] == relative_report)
    entry["size"] = report_path.stat().st_size
    entry["sha256"] = sha256_file(report_path)
    _json(manifest_path, manifest)

    with pytest.raises(BenchmarkConfigError, match="canonical ANDREA telemetry is invalid"):
        construct_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
        )


def test_builder_rejects_unverifiable_diagnostic_completeness(tmp_path: Path) -> None:
    root, plan_path, _panel, index_path = _repository(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    candidate = index["runs"][0]["candidates"][0]["inference_artifact"]["digest"]
    artifact_root = root / "work/v1/artifacts/inference" / candidate
    diagnostics = next(artifact_root.glob("output/*/tools/*/io/out/model_diagnostics.tsv.gz"))
    with gzip.open(diagnostics, "rt", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    with gzip.open(diagnostics, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows[:-1])
    # Repair the immutable manifest to isolate semantic completeness validation.
    manifest_path = artifact_root / "artifact.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["files"] if item["path"].endswith("model_diagnostics.tsv.gz")
    )
    entry["size"] = diagnostics.stat().st_size
    entry["sha256"] = sha256_file(diagnostics)
    _json(manifest_path, manifest)

    with pytest.raises(BenchmarkConfigError, match="exact group-target product"):
        construct_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
        )


def test_builder_rejects_truth_bearing_inference_artifacts(tmp_path: Path) -> None:
    root, plan_path, panel_path, index_path = _repository(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    declaration = panel["datasets"][0]
    indexed_candidate = index["runs"][0]["candidates"][0]
    candidate = next(
        item for item in plan["candidates"] if item["id"] == indexed_candidate["candidate_id"]
    )
    store = ArtifactStore(root / "work/v1/artifacts")
    dataset = store.validate(
        store.key_for(
            "dataset",
            {
                "schema_version": "1.0",
                "dataset_role": "efficiency_validation",
                "condition_id": declaration["condition_id"],
                "split_id": "efficiency_validation_v1",
                "dataset_id": declaration["dataset_id"],
            },
        )
    )
    leaking = _publish_inference(
        store,
        dataset=dataset,
        dataset_id=declaration["dataset_id"],
        fingerprint=canonical_json_sha256({"dataset": declaration["dataset_id"]}),
        contexts=declaration["contexts"],
        configuration_id=candidate["id"],
        policy_sha256=candidate["policy_sha256"],
        selection_role="candidate",
        parameters=resolve_group_policy(candidate["group_policy"], len(declaration["contexts"])),
        seed=index["runs"][0]["algorithm_seed"],
        seed_derivation=index["runs"][0]["seed_derivation"],
        runtime=5.0,
        truth_leak=True,
        execution=index["execution"],
        execution_order=index["runs"][0]["execution_order"],
    )
    indexed_candidate["inference_artifact"]["digest"] = leaking.key.digest
    _json(index_path, index)

    with pytest.raises(BenchmarkConfigError, match="truth-bearing"):
        construct_noninferiority_evidence(
            repo_root=root,
            plan_path=plan_path,
            artifact_index_path=index_path,
        )
