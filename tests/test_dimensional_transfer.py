from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.dimensional_transfer as dimensional_transfer_module
import spathi_bench.dimensional_transfer_workflow as dimensional_workflow_module
from selection_helpers import selected_document
from spathi_bench.artifacts import canonical_json_sha256, sha256_file
from spathi_bench.config import BenchmarkConfigError, load_profile, validate_profile_references
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
    if multi["bootstrap"] is None:
        multi["bootstrap"] = multi["tree_method"] == "random-forest"
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
    return selected_document(
        {
            "policy_id": "spathi_conditional_v1",
            "policy_sha256": canonical_json_sha256(group_policy),
            "group_policy": group_policy,
            "spathi_commit": spathi_commit,
        },
        budget=budget,
    )


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
    assert panel["design"]["policy_stage"] == "post_calibration_freeze"
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


def test_profiles_use_portable_execution_classes_and_no_generic_statistics() -> None:
    for budget, expected_datasets in (("reduced", 4), ("full", 9)):
        profile = load_profile(REPO_ROOT, f"dimensional-transfer-{budget}", require_runnable=True)
        assert profile["purpose"] == "dimensional_transfer"
        assert profile["inference_batches"] == []
        assert profile["execution_class"] == "scale_full"
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
) -> None:
    root = _copy_contract_repository(tmp_path)
    panel = validate_transfer_panel(root)
    template = load_profile(root, "dimensional-transfer-reduced", require_runnable=True)
    production = _production(frozen_spathi_params)
    production_path = _write_json(root / "work/calibration/selected-config.json", production)

    materialized = materialize_dimensional_transfer_profile(
        repo_root=root,
        budget="reduced",
        template_profile=template,
        selected_config_path=production_path,
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

    metadata["runs"][0]["selection_role"] = "base"
    _write_json(Path(materialized["metadata_path"]), metadata)
    with pytest.raises(BenchmarkConfigError, match="only a frozen selected"):
        validate_profile_references(root, profile)


def _analysis_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, Path]:
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
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "max_depth": None,
            "bootstrap": None,
            "random_seed": 1,
        }
    )
    production_path = _write_json(root / "work/calibration/selected-config.json", production)
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
        selected_config_path=production_path,
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
                        "configuration_id": "spathi_conditional_v1",
                        "configuration_sha256": canonical_json_sha256(
                            {"group_policy": production["selected_policy"]["group_policy"]}
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
    return root, panel_path, profile_path, production_path, run_root


def test_analysis_is_descriptive_complete_and_prevalence_adjusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root = _analysis_fixture(
        tmp_path, monkeypatch
    )
    output = root / "work/results.json"

    analyze_dimensional_transfer(
        repo_root=root,
        budget="reduced",
        panel_path=panel_path,
        profile_path=profile_path,
        selected_config_path=production_path,
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


def test_analysis_rejects_an_incomplete_group_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root = _analysis_fixture(
        tmp_path, monkeypatch
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
            selected_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_runtime_profile_changed_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root = _analysis_fixture(
        tmp_path, monkeypatch
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["description"] = "Changed after the pipeline froze its inputs."
    _write_json(profile_path, profile)

    with pytest.raises(BenchmarkConfigError, match="differs from the frozen pipeline input"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            selected_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_selected_policy_changed_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root = _analysis_fixture(
        tmp_path, monkeypatch
    )
    production = json.loads(production_path.read_text(encoding="utf-8"))
    production["selected_policy"]["group_policy"]["branches"]["multi_group"]["parameters"][
        "n_estimators"
    ] += 1
    production["selected_policy"]["policy_sha256"] = canonical_json_sha256(
        production["selected_policy"]["group_policy"]
    )
    _write_json(production_path, production)

    with pytest.raises(BenchmarkConfigError, match="differs from the frozen pipeline input"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            selected_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_metric_from_another_policy_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root = _analysis_fixture(
        tmp_path, monkeypatch
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
            selected_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_noncanonical_artifact_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root = _analysis_fixture(
        tmp_path, monkeypatch
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
            selected_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_symlinked_run_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root = _analysis_fixture(
        tmp_path, monkeypatch
    )
    linked_run = run_root.with_name("linked-dimensional-run")
    linked_run.symlink_to(run_root.name, target_is_directory=True)

    with pytest.raises(BenchmarkConfigError, match="contains a symlink"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            selected_config_path=production_path,
            run_root=linked_run,
            output_path=root / "work/results.json",
        )


def test_analysis_rejects_a_symlinked_evidence_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, panel_path, profile_path, production_path, run_root = _analysis_fixture(
        tmp_path, monkeypatch
    )
    target = production_path.with_name("selected-target.json")
    production_path.rename(target)
    production_path.symlink_to(target.name)

    with pytest.raises(BenchmarkConfigError, match="evidence input is missing"):
        analyze_dimensional_transfer(
            repo_root=root,
            budget="reduced",
            panel_path=panel_path,
            profile_path=profile_path,
            selected_config_path=production_path,
            run_root=run_root,
            output_path=root / "work/results.json",
        )
