from __future__ import annotations

import csv
import gzip
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.dimensional_transfer as dimensional_transfer_module
import spathi_bench.dimensional_transfer_workflow as dimensional_workflow_module
from spathi_bench.artifacts import Artifact, ArtifactStore, canonical_json_sha256, sha256_file
from spathi_bench.config import BenchmarkConfigError, load_profile
from spathi_bench.dimensional_transfer import (
    analyze_dimensional_transfer,
    topology_rows,
    topology_text,
    validate_transfer_panel,
)
from spathi_bench.dimensional_transfer_workflow import (
    materialize_dimensional_transfer_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _copy_contract_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "schemas", root / "schemas")
    shutil.copytree(
        REPO_ROOT / "experiments/v1/simulations/scmultisim/dimensional_transfer",
        root / "experiments/v1/simulations/scmultisim/dimensional_transfer",
    )
    (root / "experiments/v1/profiles").mkdir(parents=True)
    for budget in ("reduced", "full"):
        shutil.copyfile(
            REPO_ROOT / f"experiments/v1/profiles/dimensional-transfer-{budget}.json",
            root / f"experiments/v1/profiles/dimensional-transfer-{budget}.json",
        )
    (root / "experiments/v1/seeds").mkdir(parents=True)
    shutil.copyfile(
        REPO_ROOT / "experiments/v1/seeds/seed-plan.json",
        root / "experiments/v1/seeds/seed-plan.json",
    )
    shutil.copyfile(REPO_ROOT / "images.lock.json", root / "images.lock.json")
    shutil.copyfile(REPO_ROOT / "repos.lock.json", root / "repos.lock.json")
    return root


def _production(parameters: dict[str, Any], *, budget: str = "reduced") -> dict[str, Any]:
    multi = dict(parameters)
    multi.pop("random_seed")
    single = {
        **multi,
        "weight_mode": "cell-distance",
        "distance_metric": "euclidean",
        "group_size_correction": "none",
    }
    group_policy = {
        "routing_parameter": "group_count",
        "shared_parameters": sorted(
            set(multi) - {"weight_mode", "distance_metric", "group_size_correction"}
        ),
        "branches": {
            "single_group": {
                "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                "parameters": single,
            },
            "multi_group": {
                "condition": {"minimum_group_count": 2},
                "parameters": multi,
            },
        },
    }
    spathi_commit = json.loads((REPO_ROOT / "repos.lock.json").read_text(encoding="utf-8"))[
        "repositories"
    ]["spathi"]["commit"]
    return {
        "schema_version": "1.0",
        "artifact_type": "spathi_production_policy",
        "budget": budget,
        "evidence_role": "descriptive" if budget == "reduced" else "confirmatory",
        "status": "descriptive" if budget == "reduced" else "confirmed",
        "decision": "descriptive_only" if budget == "reduced" else "reference_retained",
        "policy": {
            "policy_id": "spathi_production_v1",
            "policy_sha256": canonical_json_sha256(group_policy),
            "group_policy": group_policy,
            "spathi_commit": spathi_commit,
        },
        "provenance": {
            name: {"path": f"work/evidence/{name}.json", "sha256": "9" * 64}
            for name in (
                "calibration_selection",
                "efficiency_plan",
                "efficiency_evidence",
                "efficiency_results",
            )
        },
    }


def _same_exact_policy(production: dict[str, Any]) -> dict[str, Any]:
    return {
        **deepcopy(production["policy"]),
        "policy_id": "spathi_exact_reference",
        "source_policy_id": production["policy"]["policy_id"],
    }


def test_topologies_are_deterministic_sparse_and_cover_every_tf() -> None:
    for tf_count, target_count in ((10, 50), (100, 1_000), (1_500, 1_000)):
        rows = topology_rows(transcription_factors=tf_count)
        assert len(rows) == target_count * 2
        assert len({regulator for _target, regulator, _effect in rows}) == tf_count
        assert topology_text(transcription_factors=tf_count).startswith(
            "target\tregulator\teffect\n"
        )
        assert rows == topology_rows(transcription_factors=tf_count)


def test_committed_panel_is_closed_reproducible_and_post_freeze() -> None:
    panel = validate_transfer_panel(REPO_ROOT)

    assert [condition["id"] for condition in panel["conditions"]] == [
        "g3000_tf100_c300",
        "g15000_tf100_c300",
        "g3000_tf1500_c300",
        "g100_tf10_c300",
        "g100_tf10_c10000",
        "g100_tf10_c500_single_group",
        "g100_tf10_c300_groups_10_40_250",
    ]
    assert panel["design"]["selection_feedback"] is False
    assert panel["design"]["confirmatory_ranking_eligible"] is False
    assert panel["design"]["external_comparison_use"] == "forbidden"
    assert panel["design"]["policy_stage"] == "post_efficiency_freeze"
    assert panel["feasibility_basis"]["endpoint_admission"]
    reduced_ids = {
        condition["id"] for condition in panel["conditions"] if "reduced" in condition["runs"]
    }
    assert reduced_ids == {
        "g3000_tf100_c300",
        "g100_tf10_c300",
        "g100_tf10_c500_single_group",
        "g100_tf10_c300_groups_10_40_250",
    }


def test_profiles_use_their_serial_budget_panel_and_no_generic_statistics() -> None:
    for budget, expected_datasets in (("reduced", 4), ("full", 9)):
        profile = load_profile(REPO_ROOT, f"dimensional-transfer-{budget}", require_runnable=True)
        assert profile["purpose"] == "dimensional_transfer"
        assert profile["inference_batches"] == []
        assert profile["resources"]["generate_data"]["max_parallel_tasks"] == 1
        assert profile["resources"]["infer_network"]["max_parallel_runs"] == 1
        assert profile["statistics"] == {"status": "not_applicable", "comparisons": []}
        replicates = sum(
            json.loads((REPO_ROOT / job["simulator_runs"]).read_text(encoding="utf-8"))["runs"][0][
                "replicates"
            ]
            for job in profile["generation_jobs"]
        )
        assert replicates == expected_datasets


def test_materialization_executes_one_frozen_policy_over_every_condition(
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_contract_repository(tmp_path)
    panel = validate_transfer_panel(root)
    template = load_profile(root, "dimensional-transfer-reduced", require_runnable=True)
    production = _production(frozen_spathi_params)
    monkeypatch.setattr(
        dimensional_workflow_module,
        "exact_reference_policy",
        lambda _root, _production: _same_exact_policy(production),
    )
    production_path = _write_json(root / "work/efficiency/production-config.json", production)

    materialized = materialize_dimensional_transfer_profile(
        repo_root=root,
        budget="reduced",
        template_profile=template,
        production_config_path=production_path,
        production=production,
        panel=panel,
        output_dir=root / "work/dimensional-transfer/materialized",
    )

    profile = json.loads(Path(materialized["profile_path"]).read_text(encoding="utf-8"))
    metadata = json.loads(Path(materialized["metadata_path"]).read_text(encoding="utf-8"))
    assert len(profile["inference_batches"]) == 1
    assert profile["inference_batches"][0]["generation_jobs"] == [
        job["id"] for job in profile["generation_jobs"]
    ]
    assert metadata["runs"][0]["selection_role"] == "selected"
    assert metadata["runs"][0]["seed_derivation"]["namespace"] == ("reduced.dimensional_transfer")
    assert materialized["algorithm_seed"] == 2078775612
    assert materialized["exact_diagnostic_status"] == ("not_applicable_production_is_exact")


def test_materialization_adds_paired_exact_runs_only_when_production_differs(
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _copy_contract_repository(tmp_path)
    panel = validate_transfer_panel(root)
    template = load_profile(root, "dimensional-transfer-reduced", require_runnable=True)
    production = _production(frozen_spathi_params)
    exact = _same_exact_policy(production)
    for branch in exact["group_policy"]["branches"].values():
        branch["parameters"]["kernel"] = (
            "exponential" if branch["parameters"]["kernel"] != "exponential" else "gaussian"
        )
    exact["policy_sha256"] = canonical_json_sha256(exact["group_policy"])
    monkeypatch.setattr(
        dimensional_workflow_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )
    monkeypatch.setattr(
        dimensional_workflow_module,
        "validate_profile_references",
        lambda _root, _profile: None,
    )
    production_path = _write_json(root / "work/efficiency/production-config.json", production)

    materialized = materialize_dimensional_transfer_profile(
        repo_root=root,
        budget="reduced",
        template_profile=template,
        production_config_path=production_path,
        production=production,
        panel=panel,
        output_dir=root / "work/dimensional-transfer/materialized",
    )

    profile = json.loads(Path(materialized["profile_path"]).read_text(encoding="utf-8"))
    assert materialized["exact_diagnostic_status"] == "paired_execution_required"
    assert len(profile["inference_batches"]) == 2
    assert len(profile["inference_batches"][1]["generation_jobs"]) == 2
    assert Path(materialized["exact_metadata_path"]).is_file()


def _analysis_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    optimized_production: bool = False,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Any]]:
    root = _copy_contract_repository(tmp_path)
    panel_path = root / "experiments/v1/simulations/scmultisim/dimensional_transfer/panel.json"
    template = load_profile(root, "dimensional-transfer-reduced", require_runnable=True)
    production = _production(
        {
            "weight_mode": "cell-distance-group-anchored",
            "distance_space": "pca",
            "n_components": 3,
            "distance_standardization": "none",
            "pca_svd_solver": "auto",
            "distance_metric": "euclidean",
            "kernel": "gaussian",
            "bandwidth": "auto",
            "bandwidth_scale": 1.0,
            "group_size_correction": "cap-to-target",
            "tree_method": "extra-trees",
            "n_estimators": 25,
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
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "max_depth": None,
            "bootstrap": None,
            "random_seed": 1,
        }
    )
    exact = _same_exact_policy(production)
    if optimized_production:
        for branch in production["policy"]["group_policy"]["branches"].values():
            branch["parameters"]["target_eligibility"] = "automatic"
            branch["parameters"]["adaptive_trees"] = True
        production["policy"]["policy_sha256"] = canonical_json_sha256(
            production["policy"]["group_policy"]
        )
    production_path = _write_json(root / "work/production.json", production)
    monkeypatch.setattr(
        dimensional_workflow_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )
    monkeypatch.setattr(
        dimensional_workflow_module,
        "validate_profile_references",
        lambda _root, _profile: None,
    )
    monkeypatch.setattr(
        dimensional_transfer_module,
        "validate_profile_references",
        lambda _root, _profile: None,
    )
    materialized = materialize_dimensional_transfer_profile(
        repo_root=root,
        budget="reduced",
        template_profile=template,
        production_config_path=production_path,
        production=production,
        panel=validate_transfer_panel(root, panel_path),
        output_dir=root / "work/dimensional-transfer/materialized",
    )
    profile_path = Path(materialized["profile_path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    run_root = root / "work/v1" / profile["id"] / "transfer-fixture"
    reports = run_root / "reports"
    reports.mkdir(parents=True)
    metrics_path = reports / "metrics.csv"
    fieldnames = [
        "generation_job",
        "dataset_id",
        "dataset_role",
        "regime",
        "family",
        "configuration_id",
        "configuration_sha256",
        "algorithm_seed",
        "context",
        "level",
        "status",
        "n_candidates",
        "n_truth_edges",
        "aupr",
        "auroc",
        "f1_at_truth_count",
        "epr_at_truth_count",
        "wall_time_seconds",
    ]
    datasets = []
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, job in enumerate(profile["generation_jobs"], start=1):
            dataset_id = f"transfer_fixture_{job['condition_id']}__r01"
            run = json.loads((root / job["simulator_runs"]).read_text(encoding="utf-8"))["runs"][0]
            datasets.append(
                {
                    "generation_job": job["id"],
                    "dataset_id": dataset_id,
                    "dataset_role": "dimensional_transfer",
                    "simulator_id": "scmultisim",
                    "seed": run["base_seed"],
                }
            )
            groups = ("A",) if job["condition_id"].endswith("single_group") else ("A", "B", "C")
            for group in groups:
                writer.writerow(
                    {
                        "generation_job": job["id"],
                        "dataset_id": dataset_id,
                        "dataset_role": "dimensional_transfer",
                        "regime": "group_only",
                        "family": "spathi",
                        "configuration_id": "spathi_production_v1",
                        "configuration_sha256": canonical_json_sha256(
                            {"group_policy": production["policy"]["group_policy"]}
                        ),
                        "algorithm_seed": 2078775612,
                        "context": f"group:{group}",
                        "level": "directed",
                        "status": "ok",
                        "n_candidates": 1000,
                        "n_truth_edges": 100,
                        "aupr": 0.2 + index / 100,
                        "auroc": 0.7,
                        "f1_at_truth_count": 0.15,
                        "epr_at_truth_count": 2.0,
                        "wall_time_seconds": 10.0 * index,
                    }
                )
    summary_path = _write_json(reports / "summary.json", {"status": "complete"})
    artifact_store = root / "work/v1/artifacts"
    artifact_store.mkdir(parents=True)
    snapshot_root = run_root / "input_snapshot"
    frozen_root = snapshot_root / "repository"
    for source in (root / "schemas", root / "experiments"):
        shutil.copytree(source, frozen_root / source.relative_to(root))
    for source in (root / "images.lock.json", root / "repos.lock.json", production_path):
        destination = frozen_root / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    materialized_root = root / "work/dimensional-transfer/materialized"
    shutil.copytree(
        materialized_root,
        frozen_root / materialized_root.relative_to(root),
    )
    frozen_files = []
    for source in sorted(path for path in frozen_root.rglob("*") if path.is_file()):
        relative = source.relative_to(frozen_root).as_posix()
        frozen_files.append(
            {
                "path": relative,
                "sha256": sha256_file(source),
                "size": source.stat().st_size,
                "kind": "runtime-derived" if relative.startswith("work/") else "versioned",
            }
        )
    profile_relative = profile_path.relative_to(root).as_posix()
    profile_entry = next(item for item in frozen_files if item["path"] == profile_relative)
    benchmark_repository = {
        "path": str(root.resolve()),
        "commit": "a" * 40,
        "origin": "https://example.invalid/spathi-benchmarks.git",
        "clean": True,
    }
    _write_json(
        snapshot_root / "manifest.json",
        {
            "schema_version": "1.0",
            "profile_id": profile["id"],
            "profile_source": {
                "path": profile_relative,
                "sha256": profile_entry["sha256"],
                "kind": profile_entry["kind"],
            },
            "benchmark_repository": benchmark_repository,
            "files": frozen_files,
            "status": "complete",
        },
    )
    monkeypatch.setattr(
        "spathi_bench.provenance.observe_benchmark_repository",
        lambda _root, require_clean=True: benchmark_repository,
    )
    monkeypatch.setattr(
        dimensional_transfer_module,
        "validate_inference_attribution_contract",
        lambda **_kwargs: None,
    )
    _write_json(
        run_root / "state.json",
        {
            "schema_version": "1.0",
            "profile_id": profile["id"],
            "purpose": "dimensional_transfer",
            "status": "completed",
            "input_snapshot": str((snapshot_root / "manifest.json").resolve()),
            "artifact_store": str(artifact_store.resolve()),
            "datasets": datasets,
            "reports": {
                "metrics_csv": str(metrics_path),
                "metrics_csv_sha256": sha256_file(metrics_path),
                "summary_json": str(summary_path),
                "summary_json_sha256": sha256_file(summary_path),
            },
        },
    )
    return root, panel_path, profile_path, production_path, run_root, exact


def test_analysis_is_descriptive_complete_and_prevalence_adjusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root, exact = _analysis_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        dimensional_transfer_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )
    output = root / "work/results.json"

    analyze_dimensional_transfer(
        repo_root=root,
        budget="reduced",
        panel_path=panel_path,
        profile_path=profile_path,
        production_config_path=production_path,
        run_root=run_root,
        output_path=output,
    )
    results = json.loads(output.read_text(encoding="utf-8"))

    assert results["evidence_role"] == "operational"
    assert len(results["datasets"]) == 4
    assert len(results["conditions"]) == 4
    assert results["contrasts"] == []
    assert "confidence_interval" not in json.dumps(results)
    assert results["conditions"][0]["adjusted_aupr"] == pytest.approx((0.21 - 0.1) / 0.9)
    assert results["exact_production_diagnostic"]["status"] == (
        "not_applicable_production_is_exact"
    )
    assert results["exact_production_diagnostic"]["identity_delta"]["adjusted_aupr"] == 0


def test_analysis_reports_paired_post_freeze_exact_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root, exact = _analysis_fixture(
        tmp_path, monkeypatch, optimized_production=True
    )
    production = json.loads(production_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        dimensional_transfer_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )

    metrics_path = run_root / "reports/metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    diagnostic_conditions = {"g3000_tf100_c300", "g100_tf10_c300"}
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    condition_by_job = {job["id"]: job["condition_id"] for job in profile["generation_jobs"]}
    exact_rows = []
    for row in rows:
        if condition_by_job[row["generation_job"]] in diagnostic_conditions:
            exact_rows.append(
                {
                    **row,
                    "configuration_id": "spathi_exact_reference",
                    "configuration_sha256": canonical_json_sha256(
                        {"group_policy": exact["group_policy"]}
                    ),
                    "aupr": "0.25",
                    "wall_time_seconds": "25.0",
                }
            )
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([*rows, *exact_rows])

    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["reports"]["metrics_csv_sha256"] = sha256_file(metrics_path)
    state["artifact_store"] = str(root / "work/v1/artifacts")
    dataset_by_condition = {
        condition_by_job[item["generation_job"]]: item["dataset_id"] for item in state["datasets"]
    }
    state["evaluations"] = [
        {
            "dataset_id": dataset_by_condition[condition],
            "configuration_id": configuration,
            "configuration_sha256": (
                canonical_json_sha256({"group_policy": exact["group_policy"]})
                if configuration == "spathi_exact_reference"
                else canonical_json_sha256({"group_policy": production["policy"]["group_policy"]})
            ),
            "algorithm_seed": 2078775612,
        }
        for condition in sorted(diagnostic_conditions)
        for configuration in ("spathi_production_v1", "spathi_exact_reference")
    ]
    _write_json(state_path, state)

    def accounting(*, evaluation: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        if evaluation["configuration_id"] == "spathi_exact_reference":
            return {
                "target_coverage": 1.0,
                "truth_edge_coverage": 1.0,
                "fitted_estimators": 250,
            }
        return {
            "target_coverage": 0.8,
            "truth_edge_coverage": 0.9,
            "fitted_estimators": 100,
        }

    monkeypatch.setattr(dimensional_transfer_module, "_configuration_accounting", accounting)
    output = root / "work/paired-results.json"
    analyze_dimensional_transfer(
        repo_root=root,
        budget="reduced",
        panel_path=panel_path,
        profile_path=profile_path,
        production_config_path=production_path,
        run_root=run_root,
        output_path=output,
    )
    diagnostic = json.loads(output.read_text(encoding="utf-8"))["exact_production_diagnostic"]

    assert diagnostic["status"] == "paired_descriptive"
    assert len(diagnostic["pairs"]) == 2
    assert all(
        pair["delta_production_minus_exact"]["target_coverage"] == pytest.approx(-0.2)
        for pair in diagnostic["pairs"]
    )
    assert all(
        pair["delta_production_minus_exact"]["fitted_estimators"] == -150
        for pair in diagnostic["pairs"]
    )


def test_analysis_rejects_an_incomplete_group_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root, exact = _analysis_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        dimensional_transfer_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )
    metrics = run_root / "reports/metrics.csv"
    rows = metrics.read_text(encoding="utf-8").splitlines()
    metrics.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["reports"]["metrics_csv_sha256"] = sha256_file(metrics)
    _write_json(state_path, state)

    with pytest.raises(BenchmarkConfigError, match="declared group contexts"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            production_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_runtime_profile_changed_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root, exact = _analysis_fixture(
        tmp_path, monkeypatch
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["description"] = "Changed after the pipeline froze its inputs."
    _write_json(profile_path, profile)
    monkeypatch.setattr(
        dimensional_transfer_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )

    with pytest.raises(BenchmarkConfigError, match="differs from the frozen pipeline input"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            production_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_production_policy_changed_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root, exact = _analysis_fixture(
        tmp_path, monkeypatch
    )
    production = json.loads(production_path.read_text(encoding="utf-8"))
    production["policy"]["group_policy"]["branches"]["multi_group"]["parameters"][
        "n_estimators"
    ] += 1
    production["policy"]["policy_sha256"] = canonical_json_sha256(
        production["policy"]["group_policy"]
    )
    _write_json(production_path, production)
    monkeypatch.setattr(
        dimensional_transfer_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )

    with pytest.raises(BenchmarkConfigError, match="differs from the frozen pipeline input"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            production_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_metric_from_another_policy_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root, exact = _analysis_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        dimensional_transfer_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )
    metrics_path = run_root / "reports/metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["configuration_sha256"] = "f" * 64
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["reports"]["metrics_csv_sha256"] = sha256_file(metrics_path)
    _write_json(state_path, state)

    with pytest.raises(BenchmarkConfigError, match="missing or invalid metrics"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            production_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_noncanonical_artifact_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root, exact = _analysis_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        dimensional_transfer_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )
    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifact_store"] = str(root / "work/v1/other-artifacts")
    _write_json(state_path, state)

    with pytest.raises(BenchmarkConfigError, match="canonical shared artifact store"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            production_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_symlinked_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root, exact = _analysis_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        dimensional_transfer_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )
    linked_run = run_root.with_name("linked-dimensional-run")
    linked_run.symlink_to(run_root.name, target_is_directory=True)

    with pytest.raises(BenchmarkConfigError, match="contains a symlink"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            production_config_path=production_path,
            run_root=linked_run,
            output_path=root / "work/results.json",
        )


def _artifact_reference(artifact: Artifact) -> dict[str, str]:
    return {
        "kind": artifact.key.kind,
        "digest": artifact.key.digest,
        "key": artifact.key.value,
    }


def _accounting_fixture(
    tmp_path: Path,
    *,
    omit_last_diagnostic: bool = False,
    completed_models: int = 4,
) -> tuple[ArtifactStore, dict[str, Any], dict[str, Any]]:
    store = ArtifactStore(tmp_path / "artifacts")
    dataset_key = store.key_for("dataset", {"fixture": "dimensional-accounting"})
    with store.staging_directory(dataset_key) as staging:
        truth_path = staging / "truth/networks.csv"
        truth_path.parent.mkdir(parents=True)
        truth_path.write_text(
            "context,source,target\ngroup:A,TF1,G1\ngroup:B,TF1,G1\n",
            encoding="utf-8",
        )
        dataset_artifact = store.publish_from_temp(
            staging,
            kind="dataset",
            spec={"fixture": "dimensional-accounting"},
        )

    inference_key = store.key_for("inference", {"fixture": "dimensional-accounting"})
    with store.staging_directory(inference_key) as staging:
        output = staging / "output/tool/io/out"
        output.mkdir(parents=True)
        with gzip.open(output / "target_eligibility.tsv.gz", "wt", encoding="utf-8") as handle:
            handle.write("target\tmode\teligible\nG1\tall\ttrue\nG2\tall\ttrue\n")
        diagnostics = [(group, target) for group in ("A", "B") for target in ("G1", "G2")]
        if omit_last_diagnostic:
            diagnostics.pop()
        with gzip.open(output / "model_diagnostics.tsv.gz", "wt", encoding="utf-8") as handle:
            handle.write("target_group\ttarget\tstatus\trandom_seed\tn_estimators_fitted\n")
            for group, target in diagnostics:
                handle.write(f"{group}\t{target}\ttrained\t123\t25\n")
        _write_json(
            output / "run_metadata.json",
            {
                "status": "complete",
                "random_seed": 123,
                "requested_parameters": {
                    "target_eligibility": "all",
                    "n_estimators": 25,
                    "random_seed": 123,
                    "target_list": None,
                },
                "input_dimensions": {"targets": 2, "groups": 2},
                "effective_parameters": {"target_eligibility": {"globally_eligible_targets": 2}},
                "models": {
                    "requested": 4,
                    "completed": completed_models,
                    "trained": 4,
                    "preflight_skipped": 0,
                    "target_not_estimable": 0,
                    "fit_or_importance_failures": 0,
                    "skipped_target_records": 0,
                    "fitted_estimators_total": 100,
                },
            },
        )
        inference_artifact = store.publish_from_temp(
            staging,
            kind="inference",
            spec={"fixture": "dimensional-accounting"},
        )
    return (
        store,
        {
            "configuration_id": "spathi_production_v1",
            "configuration_sha256": "c" * 64,
            "algorithm_seed": 123,
            "inference_artifact": _artifact_reference(inference_artifact),
        },
        {"dataset_artifact": _artifact_reference(dataset_artifact)},
    )


@pytest.mark.parametrize(
    ("omit_last_diagnostic", "completed_models", "message"),
    [
        (True, 4, "exact group-target product"),
        (False, 3, "counters differ from complete diagnostics"),
    ],
)
def test_configuration_accounting_rejects_incomplete_diagnostics_or_counters(
    tmp_path: Path,
    omit_last_diagnostic: bool,
    completed_models: int,
    message: str,
) -> None:
    store, evaluation, dataset = _accounting_fixture(
        tmp_path,
        omit_last_diagnostic=omit_last_diagnostic,
        completed_models=completed_models,
    )

    with pytest.raises(BenchmarkConfigError, match=message):
        dimensional_transfer_module._configuration_accounting(
            store=store,
            evaluation=evaluation,
            dataset=dataset,
            expected_contexts=frozenset({"group:A", "group:B"}),
            expected_configuration_id="spathi_production_v1",
            expected_configuration_sha256="c" * 64,
            expected_parameters={"target_eligibility": "all", "n_estimators": 25},
            expected_algorithm_seed=123,
        )


def test_analysis_rejects_a_symlinked_evidence_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root, exact = _analysis_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        dimensional_transfer_module,
        "exact_reference_policy",
        lambda _root, _production: exact,
    )
    target = production_path.with_name("production-target.json")
    production_path.rename(target)
    production_path.symlink_to(target.name)

    with pytest.raises(BenchmarkConfigError, match="evidence input is missing"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            production_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )
