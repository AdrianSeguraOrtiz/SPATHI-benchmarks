from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.artifacts import Artifact, ArtifactStore, canonical_json_sha256, sha256_file
from spathi_bench.calibration_contexts import load_dataset_group_contract
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.state import load_state, write_state
from spathi_bench.summarize import (
    _calibration_context_aupr,
    _invariant_edge_stability,
    summarize,
)

_DEFAULT_METRICS = object()


def _scheduler_fixture(run_root: Path) -> dict[str, Any]:
    capacity = {
        "cpu_threads": 4,
        "memory_bytes": 8 * 1024**3,
        "memory_gib": 8.0,
    }
    utilization = {
        "wall_seconds": 5.0,
        "cpu_thread_seconds_reserved": 5.0,
        "memory_byte_seconds_reserved": float(4 * 1024**3 * 5),
        "cpu_capacity_utilization": 0.25,
        "memory_capacity_utilization": 0.5,
        "peak_concurrent_tasks": 1,
        "peak_reserved_cpu_threads": 1,
        "peak_reserved_memory_bytes": 4 * 1024**3,
    }
    for phase in ("inference", "evaluation"):
        path = run_root / "schedules" / f"{phase}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "complete",
                    "capacity": {
                        "cpu_threads": capacity["cpu_threads"],
                        "memory_bytes": capacity["memory_bytes"],
                        "max_concurrent_tasks": None,
                    },
                    "logical_tasks": 2,
                    "physical_tasks": 2,
                    "cache_hits": 0,
                    "wall_seconds": 5.0,
                    "tasks": [],
                    "utilization": utilization,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return {"capacity": capacity}


def test_calibration_accepts_partial_only_when_directed_aupr_is_complete() -> None:
    partial = {
        "context": "group:a",
        "truth_context": "group:a",
        "status": "partial",
        "reason": "AUROC/AUPR require at least one positive and one negative candidate",
        "auroc": None,
        "aupr": 1.0,
    }
    assert _calibration_context_aupr(partial) == 1.0

    with pytest.raises(BenchmarkConfigError, match="finite number"):
        _calibration_context_aupr({**partial, "aupr": None})

    with pytest.raises(BenchmarkConfigError, match="unsupported partial"):
        _calibration_context_aupr({**partial, "reason": "incomplete output"})


def test_invariant_edge_stability_uses_truth_count_and_penalizes_missing_edges(
    tmp_path: Path,
) -> None:
    network = tmp_path / "network.csv"
    network.write_text(
        "source,target,score,sign,evidence,context\n"
        "TF1,G1,0.9,?,association,group:a\n"
        "TF2,G2,0.8,?,association,group:a\n"
        "TF1,G1,0.7,?,association,group:b\n"
        "TF3,G3,0.6,?,association,group:b\n",
        encoding="utf-8",
    )
    metrics = [
        {
            "level": "directed",
            "context": context,
            "n_truth_edges": 2,
            "n_candidates": 9,
        }
        for context in ("group:a", "group:b")
    ]
    kwargs = {
        "candidate_sources": frozenset({"TF1", "TF2", "TF3"}),
        "candidate_targets": frozenset({"G1", "G2", "G3"}),
        "allow_self_edges": False,
    }

    assert _invariant_edge_stability(network, metrics, **kwargs) == pytest.approx(0.5)

    network.write_text(
        "source,target,score,sign,evidence,context\n",
        encoding="utf-8",
    )
    assert _invariant_edge_stability(network, metrics, **kwargs) == 0.0


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ((0.5, 0.5, 0.5), 0.0),
        ((0.9, 0.5, 0.5), 0.5),
        ((0.9, 0.8, 0.1), 1.0),
    ],
    ids=("flat_has_no_stability", "boundary_ties_are_unresolved", "ranked_signal"),
)
def test_invariant_edge_stability_is_tie_robust(
    tmp_path: Path,
    scores: tuple[float, float, float],
    expected: float,
) -> None:
    network = tmp_path / "network.csv"
    rows = ["source,target,score,sign,evidence,context"]
    for context in ("group:a", "group:b"):
        rows.extend(
            f"TF1,G{index},{score},?,association,{context}"
            for index, score in enumerate(scores, start=1)
        )
    network.write_text("\n".join(rows) + "\n", encoding="utf-8")
    metrics = [
        {
            "level": "directed",
            "context": context,
            "n_truth_edges": 2,
            "n_candidates": 3,
        }
        for context in ("group:a", "group:b")
    ]

    assert _invariant_edge_stability(
        network,
        metrics,
        candidate_sources=frozenset({"TF1"}),
        candidate_targets=frozenset({"G1", "G2", "G3"}),
        allow_self_edges=False,
    ) == pytest.approx(expected)


@pytest.fixture(autouse=True)
def _validated_run_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "spathi_bench.summarize.validate_run_inputs",
        lambda _repo_root, _profile_id, _run_root: {"status": "complete"},
    )
    monkeypatch.setattr(
        "spathi_bench.summarize.validate_inference_attribution_contract",
        lambda **_kwargs: None,
    )


def _artifact_reference(artifact: Artifact) -> dict[str, Any]:
    return {
        "kind": artifact.key.kind,
        "digest": artifact.key.digest,
        "key": artifact.key.value,
        "path": str(artifact.path),
        "cache_hit": False,
    }


def _publish_json_artifact(
    store: ArtifactStore,
    *,
    kind: str,
    spec: dict[str, Any],
    relative_path: str,
    payload: dict[str, Any],
    extra_files: dict[str, str] | None = None,
) -> tuple[Artifact, Path]:
    key = store.key_for(kind, spec)
    with store.staging_directory(key) as staging:
        destination = staging / relative_path
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps(payload), encoding="utf-8")
        for extra_relative, content in (extra_files or {}).items():
            extra = staging / extra_relative
            extra.parent.mkdir(parents=True, exist_ok=True)
            extra.write_text(content, encoding="utf-8")
        artifact = store.publish_from_temp(staging, kind=kind, spec=spec)
    return artifact, artifact.file(relative_path)


def _dataset(store: ArtifactStore, dataset_id: str, seed: int) -> dict[str, Any]:
    fingerprint = {
        "algorithm": "sha256",
        "value": canonical_json_sha256({"dataset_id": dataset_id}),
    }
    spec = {"dataset_id": dataset_id, "seed": seed}
    key = store.key_for("dataset", spec)
    with store.staging_directory(key) as staging:
        dataset_root = staging / "output" / "benchmark" / "datasets" / dataset_id
        (dataset_root / "truth").mkdir(parents=True)
        (dataset_root / "extras").mkdir()
        (dataset_root / "truth" / "gene_universe.txt").write_text("TF1\nG1\nG2\n", encoding="utf-8")
        (dataset_root / "extras" / "tf_list.txt").write_text("TF1\n", encoding="utf-8")
        (dataset_root / "extras" / "groups.tsv").write_text(
            "column\tcluster\ncell_1\ta\ncell_2\tb\n",
            encoding="utf-8",
        )
        (dataset_root / "truth" / "networks.csv").write_text(
            "source,target,score,sign,evidence,context\n"
            "TF1,G1,1,?,simulation,group:a\n"
            "TF1,G1,1,?,simulation,group:b\n",
            encoding="utf-8",
        )
        dataset_manifest = dataset_root / "dataset-manifest.json"
        dataset_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "dataset": {"spec": {"id": dataset_id}},
                    "extras": {"groups": "extras/groups.tsv"},
                }
            ),
            encoding="utf-8",
        )
        manifest = dataset_root / "ground-truth-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "dataset_id": dataset_id,
                    "dataset_fingerprint": fingerprint,
                    "truth_requirements": {"contexts": ["global", "group"]},
                    "outputs": {
                        "gene_universe": "truth/gene_universe.txt",
                        "networks": "truth/networks.csv",
                    },
                    "candidate_space": {
                        "sources": "extras/tf_list.txt",
                        "targets": "truth/gene_universe.txt",
                        "allow_self_edges": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        artifact = store.publish_from_temp(staging, kind="dataset", spec=spec)
    relative_root = artifact.path / "output" / "benchmark" / "datasets" / dataset_id
    return {
        "dataset_id": dataset_id,
        "generation_job": "dyngen_smoke",
        "simulator_id": "dyngen",
        "seed": seed,
        "dataset_role": "smoke",
        "split_id": "smoke",
        "condition_id": "baseline",
        "dataset_fingerprint": fingerprint,
        "dataset_manifest": str(relative_root / "dataset-manifest.json"),
        "ground_truth_manifest": str(relative_root / "ground-truth-manifest.json"),
        "dataset_artifact": _artifact_reference(artifact),
    }


def _evaluation(
    store: ArtifactStore,
    *,
    dataset: dict[str, Any],
    run_id: str,
    tool_id: str,
    tool_origin: str,
    configuration_id: str,
    metrics: object,
    algorithm_seed: int = 1729,
    family: str = "tree_ensemble",
    selection_role: str = "baseline",
) -> dict[str, Any]:
    dataset_id = str(dataset["dataset_id"])
    inference_spec = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "configuration_id": configuration_id,
        "algorithm_seed": algorithm_seed,
    }
    inference, run_report = _publish_json_artifact(
        store,
        kind="inference",
        spec=inference_spec,
        relative_path="output/run_report.json",
        payload={
            "tools": {
                "tool_origins": {run_id: tool_origin},
                "results": {
                    run_id: {
                        "network_path": "network.csv",
                        "measurement": {
                            "schema_version": "1.0",
                            "scope": "physical_task",
                            "wall_time_seconds": 5.0,
                            "peak_memory_bytes": 1024,
                        },
                    }
                },
            }
        },
        extra_files={
            "output/network.csv": (
                "source,target,score,sign,evidence,context\n"
                "TF1,G1,0.9,?,inference,group:a\n"
                "TF1,G2,0.4,?,inference,group:a\n"
                "TF1,G1,0.8,?,inference,group:b\n"
                "TF1,G2,0.3,?,inference,group:b\n"
            )
        },
    )
    evaluation_spec = {
        "inference_artifact": inference.key.value,
        "evaluator": "andrea",
    }
    normalized_metrics = metrics
    if isinstance(metrics, list) and all(isinstance(item, dict) for item in metrics):
        normalized_metrics = []
        for item in metrics:
            row = dict(item)
            row.update(
                {
                    "n_candidates": 2,
                    "n_candidate_genes": 3,
                    "n_candidate_sources": 1,
                    "n_candidate_targets": 3,
                    "n_truth_edges": 1,
                }
            )
            normalized_metrics.append(row)
    fingerprint = dataset["dataset_fingerprint"]
    evaluation, evaluation_report = _publish_json_artifact(
        store,
        kind="evaluation",
        spec=evaluation_spec,
        relative_path="output/evaluation_report.json",
        payload={
            "inputs": {
                "inference_dataset_id": dataset_id,
                "ground_truth_dataset_id": dataset_id,
                "inference_dataset_fingerprint": fingerprint,
                "ground_truth_dataset_fingerprint": fingerprint,
            },
            "ground_truth": {
                "dataset_id": dataset_id,
                "dataset_fingerprint": fingerprint,
                "gene_universe_size": 3,
                "candidate_space": {
                    "allow_self_edges": False,
                    "n_sources": 1,
                    "n_targets": 3,
                    "n_source_target_overlap": 1,
                    "n_candidates_by_level": {
                        "topology": 2,
                        "directed": 2,
                        "signed": 4,
                    },
                },
            },
            "metrics": normalized_metrics,
        },
    )
    return {
        "dataset_id": dataset_id,
        "batch_id": "group_only",
        "batch_kind": "baseline" if selection_role == "baseline" else "spathi",
        "regime": "group_only",
        "run_id": run_id,
        "tool_id": tool_id,
        "tool_origin": tool_origin,
        "configuration_id": configuration_id,
        "selection_role": selection_role,
        "algorithm_seed": algorithm_seed,
        "family": family,
        "configuration_sha256": canonical_json_sha256({"configuration_id": configuration_id}),
        "algorithm_seed_parameter": "random_seed" if family == "spathi" else "seed",
        "execution_sha256": canonical_json_sha256(
            {"configuration_id": configuration_id, "algorithm_seed": algorithm_seed}
        ),
        "tool_image": (
            {
                "source_commit": "1" * 40,
                "build_fingerprint": "2" * 64,
                "image_id": "sha256:" + "3" * 64,
            }
            if family == "spathi"
            else f"image-{tool_id}"
        ),
        "run_report": str(run_report),
        "evaluation_report": str(evaluation_report),
        "inference_artifact": _artifact_reference(inference),
        "artifact": _artifact_reference(evaluation),
    }


def _genie3_metrics(*, provenance_conflict: bool = False) -> list[dict[str, Any]]:
    first = {
        "tool_id": "genie3_group_emulated",
        "level": "directed",
        "context": "group:a",
        "status": "ok",
        "auroc": 0.8,
        "aupr": 0.4,
        "f1_at_truth_count": 0.5,
        "epr_at_truth_count": 1.2,
    }
    if provenance_conflict:
        first["tool_origin"] = "custom"
    return [
        first,
        {
            "tool_id": "genie3_group_emulated",
            "level": "directed",
            "context": "group:b",
            "status": "partial",
            "auroc": 0.6,
            "aupr": None,
            "f1_at_truth_count": 0.3,
            "epr_at_truth_count": 0.8,
            "diagnostics": {"reason": "small truth"},
        },
    ]


def _evaluated_run(
    tmp_path: Path,
    *,
    genie3_metrics: object = _DEFAULT_METRICS,
) -> tuple[Path, Path]:
    repo_root = tmp_path / "benchmark-repository"
    repo_root.mkdir(parents=True)
    run_root = tmp_path / "run"
    store = ArtifactStore(tmp_path / "artifacts")
    dataset = _dataset(store, "dyngen-1729", 1729)
    metrics = _genie3_metrics() if genie3_metrics is _DEFAULT_METRICS else genie3_metrics
    evaluations = [
        _evaluation(
            store,
            dataset=dataset,
            run_id="genie3_group_emulated",
            tool_id="GENIE3",
            tool_origin="catalog",
            configuration_id="genie3_default",
            metrics=metrics,
        ),
        _evaluation(
            store,
            dataset=dataset,
            run_id="spathi_primary_seed_1729",
            tool_id="custom_spathi_primary_seed_1729",
            tool_origin="custom",
            configuration_id="spathi_primary",
            family="spathi",
            selection_role="selected",
            metrics=[
                {
                    "tool_id": "spathi_primary_seed_1729",
                    "level": "directed",
                    "context": "group:a",
                    "status": "not_evaluable",
                    "auroc": None,
                    "aupr": None,
                    "f1_at_truth_count": None,
                    "epr_at_truth_count": None,
                },
                {
                    "tool_id": "spathi_primary_seed_1729",
                    "level": "directed",
                    "context": "group:b",
                    "status": "not_evaluable",
                    "auroc": None,
                    "aupr": None,
                    "f1_at_truth_count": None,
                    "epr_at_truth_count": None,
                },
            ],
        ),
    ]
    runtime_plan = _scheduler_fixture(run_root)
    write_state(
        run_root,
        {
            "schema_version": "1.0",
            "profile_id": "smoke",
            "purpose": "smoke",
            "benchmark_repository": {"commit": "4" * 40},
            "status": "exported",
            "artifact_store": str(store.root),
            "runtime_plan": runtime_plan,
            "datasets": [dataset],
            "evaluations": evaluations,
            "cache": {"dataset_hits": 0, "inference_hits": 0, "evaluation_hits": 0},
        },
    )
    return repo_root, run_root


def _attach_calibration_handoff(
    *,
    run_root: Path,
    state: dict[str, Any],
) -> None:
    dataset = state["datasets"][0]
    contract = load_dataset_group_contract(
        Path(dataset["dataset_manifest"]),
        Path(dataset["ground_truth_manifest"]),
        expected_dataset_id=dataset["dataset_id"],
    )
    inference_runs = []
    results = []
    for evaluation in state["evaluations"]:
        inference_runs.append(
            {
                **{
                    field: evaluation[field]
                    for field in (
                        "dataset_id",
                        "batch_id",
                        "batch_kind",
                        "regime",
                        "run_id",
                        "configuration_id",
                        "algorithm_seed",
                    )
                },
                "artifact": evaluation["inference_artifact"],
            }
        )
        results.append(
            {
                "dataset_id": evaluation["dataset_id"],
                "run_id": evaluation["run_id"],
                "artifact_key": evaluation["inference_artifact"]["key"],
                "status": "passed",
                "errors": [],
                "calibration_context_contract": {
                    **contract.to_dict(),
                    "completed_group_contexts": list(contract.expected_group_contexts),
                },
            }
        )
    handoff_path = run_root / "handoff-validation.json"
    handoff_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "profile_id": state["profile_id"],
                "status": "passed",
                "results": results,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    state["inference_runs"] = inference_runs
    state["handoff_validation"] = str(handoff_path.resolve())


def test_summarize_preserves_rows_and_computes_descriptive_means(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)

    metrics_path, summary_path = summarize(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["profile_id"] == "smoke"
    assert rows[0]["purpose"] == "smoke"
    assert rows[0]["generation_job"] == "dyngen_smoke"
    assert rows[0]["dataset_role"] == "smoke"
    assert rows[0]["regime"] == "group_only"
    assert rows[0]["tool_origin"] == "catalog"
    assert rows[0]["catalog_tool_id"] == "GENIE3"
    assert rows[0]["configuration_id"] == "genie3_default"
    assert rows[0]["inference_artifact"].startswith("sha256:")
    assert all(
        len(rows[0][field]) == 64
        for field in (
            "dataset_fingerprint_sha256",
            "candidate_source_universe_sha256",
            "candidate_target_universe_sha256",
            "candidate_edge_universe_sha256",
            "truth_edge_universe_sha256",
        )
    )
    assert rows[1]["aupr"] == ""
    assert rows[1]["diagnostics"] == '{"reason":"small truth"}'

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "1.0"
    assert summary["benchmark_commit"] == "4" * 40
    assert summary["metric_rows"] == 4
    aggregates = {
        (item["configuration_id"], item["level"], item["regime"]): item
        for item in summary["aggregates"]
    }
    genie3 = aggregates[("genie3_default", "directed", "group_only")]
    assert genie3["tool_id"] == "GENIE3"
    assert genie3["catalog_tool_ids"] == ["GENIE3"]
    assert genie3["configuration_sha256"] == canonical_json_sha256(
        {"configuration_id": "genie3_default"}
    )
    assert genie3["n_datasets"] == 1
    assert genie3["n_executions"] == 1
    assert genie3["n_metric_rows"] == 2
    assert genie3["mean_auroc"] == pytest.approx(0.7)
    assert genie3["mean_aupr"] == pytest.approx(0.4)
    assert genie3["mean_f1_at_truth_count"] == pytest.approx(0.4)
    assert genie3["mean_epr_at_truth_count"] == pytest.approx(1.0)
    assert ("spathi_primary", "directed", "group_only") not in aggregates
    assert summary["notes"][0].startswith("The smoke profile")

    state = load_state(run_root, "smoke")
    assert state["status"] == "completed"
    assert state["reports"] == {
        "metrics_csv": str(metrics_path.resolve()),
        "metrics_csv_sha256": sha256_file(metrics_path),
        "summary_json": str(summary_path.resolve()),
        "summary_json_sha256": sha256_file(summary_path),
    }


def _overlapping_selected_spathi_batches(tmp_path: Path) -> tuple[Path, Path]:
    repo_root, run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    store = ArtifactStore(Path(state["artifact_store"]))
    run_id = "spathi_selected_seed_1729"
    selected = _evaluation(
        store,
        dataset=state["datasets"][0],
        run_id=run_id,
        tool_id=f"custom_{run_id}",
        tool_origin="custom",
        configuration_id="spathi_selected",
        family="spathi",
        selection_role="selected",
        metrics=[
            {
                "tool_id": run_id,
                "level": "directed",
                "context": context,
                "truth_context": context,
                "status": "ok",
                "auroc": 0.8,
                "aupr": aupr,
            }
            for context, aupr in (("group:a", 0.4), ("group:b", 0.6))
        ],
    )
    selected["batch_id"] = "spathi_group_only"
    alias = deepcopy(selected)
    alias["batch_id"] = "spathi_trajectory_group_only"
    alias["artifact"]["cache_hit"] = True
    alias["inference_artifact"]["cache_hit"] = True
    state["purpose"] = "comparison"
    state["evaluations"] = [state["evaluations"][0], selected, alias]
    write_state(run_root, state)
    return repo_root, run_root


def test_summarize_counts_overlapping_spathi_batches_as_one_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, run_root = _overlapping_selected_spathi_batches(tmp_path)
    original_evaluations = load_state(run_root, "smoke")["evaluations"]
    attribution_checks = []

    def validate_attribution(**_kwargs: Any) -> None:
        attribution_checks.append(load_state(run_root, "smoke")["evaluations"])

    monkeypatch.setattr(
        "spathi_bench.summarize.validate_inference_attribution_contract", validate_attribution
    )
    metrics_path, summary_path = summarize(
        repo_root=repo_root, profile_id="smoke", run_root=run_root
    )

    assert attribution_checks == [original_evaluations]
    assert load_state(run_root, "smoke")["evaluations"] == original_evaluations
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    spathi_rows = [row for row in rows if row["configuration_id"] == "spathi_selected"]
    assert len(rows) == 4
    assert len(spathi_rows) == 2
    assert {row["context"] for row in spathi_rows} == {"group:a", "group:b"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metric_rows"] == 4
    spathi = next(
        item for item in summary["aggregates"] if item["configuration_id"] == "spathi_selected"
    )
    assert spathi["n_metric_rows"] == 2
    assert spathi["n_executions"] == 1
    assert spathi["n_datasets"] == 1
    assert spathi["mean_aupr"] == pytest.approx(0.5)


@pytest.mark.parametrize("field", ["configuration_id", "execution_sha256", "algorithm_seed"])
def test_summarize_rejects_conflicting_metadata_for_overlapping_execution(
    tmp_path: Path,
    field: str,
) -> None:
    repo_root, run_root = _overlapping_selected_spathi_batches(tmp_path)
    state = load_state(run_root, "smoke")
    state["evaluations"][-1][field] = 999 if field == "algorithm_seed" else "different"
    write_state(run_root, state)

    with pytest.raises(BenchmarkConfigError, match="Conflicting execution records"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)

    assert load_state(run_root, "smoke")["status"] == "exported"
    assert not (run_root / "reports").exists()


def test_summarize_requires_exported_state_and_is_idempotent(tmp_path: Path) -> None:
    repo_root = tmp_path / "benchmark-repository"
    repo_root.mkdir()
    run_root = tmp_path / "incomplete"
    write_state(
        run_root,
        {
            "schema_version": "1.0",
            "profile_id": "smoke",
            "status": "inferred",
            "evaluations": [],
        },
    )
    with pytest.raises(BenchmarkConfigError, match="requires benchmark status 'exported'"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)

    repo_root, evaluated = _evaluated_run(tmp_path / "second")
    first = summarize(repo_root=repo_root, profile_id="smoke", run_root=evaluated)
    second = summarize(repo_root=repo_root, profile_id="smoke", run_root=evaluated)

    assert second == first


def test_summarize_rejects_a_changed_completed_report(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    metrics_path, _summary_path = summarize(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )
    metrics_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="unbound or changed"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_summarize_requires_canonical_completed_report_paths(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    state = load_state(run_root, "smoke")
    state["reports"]["metrics_csv"] = str(run_root / "reports" / ".." / "reports" / "metrics.csv")
    write_state(run_root, state)

    with pytest.raises(BenchmarkConfigError, match="canonical reports"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


@pytest.mark.parametrize("symlink_kind", ["file", "ancestor"])
def test_summarize_completed_validation_rejects_symlinks(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    metrics_path, _summary_path = summarize(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )
    if symlink_kind == "file":
        external = tmp_path / "external-metrics.csv"
        external.write_bytes(metrics_path.read_bytes())
        metrics_path.unlink()
        metrics_path.symlink_to(external)
    else:
        reports = run_root / "reports"
        relocated = run_root / "relocated-reports"
        reports.rename(relocated)
        reports.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(BenchmarkConfigError, match="symlink"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_summarize_recovers_published_current_schema_reports(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    published = summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    state = load_state(run_root, "smoke")
    state["status"] = "exported"
    state.pop("reports")
    write_state(run_root, state)

    recovered = summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)

    assert recovered == published
    assert load_state(run_root, "smoke")["status"] == "completed"


def test_summarize_recovery_requires_exact_report_members(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)
    state = load_state(run_root, "smoke")
    state["status"] = "exported"
    state.pop("reports")
    write_state(run_root, state)
    (run_root / "reports" / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="Incomplete reports directory"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_summarize_recovery_rejects_a_symlinked_report(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    metrics_path, _summary_path = summarize(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )
    state = load_state(run_root, "smoke")
    state["status"] = "exported"
    state.pop("reports")
    write_state(run_root, state)
    external = tmp_path / "external-metrics.csv"
    external.write_bytes(metrics_path.read_bytes())
    metrics_path.unlink()
    metrics_path.symlink_to(external)

    with pytest.raises(BenchmarkConfigError, match="symlink"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_summarize_refuses_a_stale_reports_directory(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    (run_root / "reports").mkdir()

    with pytest.raises(BenchmarkConfigError, match="Incomplete reports directory already exists"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_summarize_gives_datasets_equal_weight(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    store = ArtifactStore(Path(state["artifact_store"]))
    other_dataset = _dataset(store, "dyngen-1730", 1730)
    state["datasets"].append(other_dataset)
    state["evaluations"].append(
        _evaluation(
            store,
            dataset=other_dataset,
            run_id="genie3_group_emulated",
            tool_id="GENIE3",
            tool_origin="catalog",
            configuration_id="genie3_default",
            metrics=[
                {
                    "tool_id": "genie3_group_emulated",
                    "level": "directed",
                    "context": context,
                    "status": "ok",
                    "auroc": 0.1,
                    "aupr": 0.2,
                    "f1_at_truth_count": 0.1,
                    "epr_at_truth_count": 0.2,
                }
                for context in ("group:a", "group:b")
            ],
        )
    )
    write_state(run_root, state)

    _metrics_path, summary_path = summarize(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = next(
        item
        for item in summary["aggregates"]
        if item["configuration_id"] == "genie3_default" and item["level"] == "directed"
    )
    assert aggregate["n_datasets"] == 2
    assert aggregate["n_executions"] == 2
    assert aggregate["mean_auroc"] == pytest.approx(0.4)


def test_summarize_averages_algorithm_seeds_before_datasets(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    store = ArtifactStore(Path(state["artifact_store"]))
    state["evaluations"].append(
        _evaluation(
            store,
            dataset=state["datasets"][0],
            run_id="genie3_seed_1730",
            tool_id="GENIE3",
            tool_origin="catalog",
            configuration_id="genie3_default",
            algorithm_seed=1730,
            metrics=[
                {
                    "tool_id": "genie3_seed_1730",
                    "level": "directed",
                    "context": context,
                    "status": "ok",
                    "auroc": 0.1,
                    "aupr": 0.2,
                    "f1_at_truth_count": 0.1,
                    "epr_at_truth_count": 0.2,
                }
                for context in ("group:a", "group:b")
            ],
        )
    )
    write_state(run_root, state)

    _metrics_path, summary_path = summarize(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = next(
        item for item in summary["aggregates"] if item["configuration_id"] == "genie3_default"
    )
    assert aggregate["n_executions"] == 2
    assert aggregate["mean_auroc"] == pytest.approx(0.4)


def test_summarize_allows_seed_specific_tool_ids_for_one_spathi_configuration(
    tmp_path: Path,
) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    store = ArtifactStore(Path(state["artifact_store"]))
    state["purpose"] = "calibration"
    state["datasets"][0].update(
        {"dataset_role": "development", "split_id": "dev_a", "condition_id": "condition_a"}
    )
    state["evaluations"] = [
        _evaluation(
            store,
            dataset=state["datasets"][0],
            run_id=f"spathi_candidate_seed_{seed}",
            tool_id=f"custom_spathi_candidate_seed_{seed}",
            tool_origin="custom",
            configuration_id="spathi_candidate",
            algorithm_seed=seed,
            family="spathi",
            selection_role="candidate",
            metrics=[
                {
                    "tool_id": f"spathi_candidate_seed_{seed}",
                    "level": "directed",
                    "context": context,
                    "truth_context": context,
                    "status": "ok",
                    "auroc": 0.8,
                    "aupr": aupr,
                }
                for context in ("group:a", "group:b")
            ],
        )
        for seed, aupr in ((1729, 0.5), (31415, 0.9))
    ]
    _attach_calibration_handoff(run_root=run_root, state=state)
    write_state(run_root, state)

    metrics_path, summary_path = summarize(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    assert {row["catalog_tool_id"] for row in metrics} == {
        "custom_spathi_candidate_seed_1729",
        "custom_spathi_candidate_seed_31415",
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "1.0"
    assert [
        (row["algorithm_seed"], row["aupr_directed"]) for row in summary["calibration_rows"]
    ] == [
        (1729, 0.5),
        (31415, 0.9),
    ]
    assert len(summary["aggregates"]) == 1
    aggregate = summary["aggregates"][0]
    assert aggregate["tool_id"] == "SPATHI"
    assert aggregate["catalog_tool_ids"] == [
        "custom_spathi_candidate_seed_1729",
        "custom_spathi_candidate_seed_31415",
    ]
    assert aggregate["configuration_sha256"] == canonical_json_sha256(
        {"configuration_id": "spathi_candidate"}
    )
    assert aggregate["n_datasets"] == 1
    assert aggregate["n_executions"] == 2
    assert aggregate["mean_aupr"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("field", "conflicting_value"),
    [
        ("family", "different_family"),
        ("selection_role", "different_role"),
        ("configuration_sha256", "f" * 64),
    ],
)
def test_summarize_rejects_conflicting_configuration_metadata(
    tmp_path: Path,
    field: str,
    conflicting_value: str,
) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    store = ArtifactStore(Path(state["artifact_store"]))
    evaluation = _evaluation(
        store,
        dataset=state["datasets"][0],
        run_id="genie3_seed_31415",
        tool_id="GENIE3",
        tool_origin="catalog",
        configuration_id="genie3_default",
        algorithm_seed=31415,
        metrics=[
            {
                "tool_id": "genie3_seed_31415",
                "level": "directed",
                "context": context,
                "status": "ok",
                "aupr": 0.5,
            }
            for context in ("group:a", "group:b")
        ],
    )
    evaluation[field] = conflicting_value
    state["evaluations"].append(evaluation)
    write_state(run_root, state)

    with pytest.raises(BenchmarkConfigError, match="conflicting invariant summary metadata"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_summarize_emits_balanced_calibration_rows_without_mixing_partitions(
    tmp_path: Path,
) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    store = ArtifactStore(Path(state["artifact_store"]))
    state["purpose"] = "calibration"
    state["datasets"][0].update(
        {"dataset_role": "development", "split_id": "dev_a", "condition_id": "condition_a"}
    )
    state["evaluations"] = [
        _evaluation(
            store,
            dataset=state["datasets"][0],
            run_id="spathi_candidate_seed_1729",
            tool_id="custom_spathi_candidate_seed_1729",
            tool_origin="custom",
            configuration_id="spathi_candidate",
            family="spathi",
            selection_role="candidate",
            metrics=[
                {
                    "tool_id": "spathi_candidate_seed_1729",
                    "level": "directed",
                    "context": "group:a",
                    "truth_context": "group:a",
                    "status": "ok",
                    "auroc": 0.8,
                    "aupr": 0.4,
                },
                {
                    "tool_id": "spathi_candidate_seed_1729",
                    "level": "directed",
                    "context": "group:b",
                    "truth_context": "group:b",
                    "status": "ok",
                    "auroc": 0.8,
                    "aupr": 0.6,
                },
            ],
        )
    ]
    _attach_calibration_handoff(run_root=run_root, state=state)
    write_state(run_root, state)

    _metrics, summary_path = summarize(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["calibration_rows"] == [
        {
            "algorithm_seed": 1729,
            "aupr_directed": 0.5,
            "configuration_id": "spathi_candidate",
            "configuration_sha256": canonical_json_sha256({"configuration_id": "spathi_candidate"}),
            "dataset_id": "dyngen-1729",
            "dataset_role": "development",
            "execution_sha256": canonical_json_sha256(
                {"configuration_id": "spathi_candidate", "algorithm_seed": 1729}
            ),
            "group_contexts": ["group:a", "group:b"],
            "n_group_contexts": 2,
            "peak_memory_mb": 1024 / 1024**2,
            "run_id": "spathi_candidate_seed_1729",
            "spathi_build_fingerprint": "2" * 64,
            "spathi_commit": "1" * 40,
            "spathi_image_id": "sha256:" + "3" * 64,
            "wall_time_seconds": 5.0,
        }
    ]
    assert summary["aggregates"][0]["dataset_role"] == "development"
    assert summary["aggregates"][0]["split_id"] == "dev_a"
    assert summary["aggregates"][0]["condition_id"] == "condition_a"


def test_summarize_rejects_a_systematically_missing_calibration_context(
    tmp_path: Path,
) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    store = ArtifactStore(Path(state["artifact_store"]))
    state["purpose"] = "calibration"
    state["datasets"][0]["dataset_role"] = "development"
    state["evaluations"] = [
        _evaluation(
            store,
            dataset=state["datasets"][0],
            run_id="spathi_candidate_seed_1729",
            tool_id="custom_spathi_candidate_seed_1729",
            tool_origin="custom",
            configuration_id="spathi_candidate",
            family="spathi",
            selection_role="candidate",
            metrics=[
                {
                    "tool_id": "spathi_candidate_seed_1729",
                    "level": "directed",
                    "context": "group:a",
                    "truth_context": "group:a",
                    "status": "ok",
                    "auroc": 0.8,
                    "aupr": 0.4,
                }
            ],
        )
    ]
    _attach_calibration_handoff(run_root=run_root, state=state)
    write_state(run_root, state)

    with pytest.raises(BenchmarkConfigError, match="complete context contract"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_summarize_rejects_metric_provenance_conflicts(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(
        tmp_path,
        genie3_metrics=_genie3_metrics(provenance_conflict=True),
    )

    with pytest.raises(BenchmarkConfigError, match="conflicts with frozen provenance"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


@pytest.mark.parametrize("metrics", [None, {}, ["not-an-object"]])
def test_summarize_rejects_invalid_metric_collections(
    tmp_path: Path,
    metrics: object,
) -> None:
    repo_root, run_root = _evaluated_run(tmp_path, genie3_metrics=metrics)

    with pytest.raises(BenchmarkConfigError, match="metrics must be an array of objects"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_summarize_requires_each_evaluation_artifact_to_match_one_run(tmp_path: Path) -> None:
    metrics: Sequence[dict[str, Any]] = [
        {
            "tool_id": "a_different_run",
            "level": "directed",
            "context": context,
            "status": "ok",
        }
        for context in ("group:a", "group:b")
    ]
    repo_root, run_root = _evaluated_run(tmp_path, genie3_metrics=metrics)

    with pytest.raises(BenchmarkConfigError, match="differs from its single-run artifact"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)
