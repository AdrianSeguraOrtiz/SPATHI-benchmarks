from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.artifacts import Artifact, ArtifactStore, canonical_json_sha256
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.state import load_state, write_state
from spathi_bench.summarize import summarize

_DEFAULT_METRICS = object()


@pytest.fixture(autouse=True)
def _validated_run_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "spathi_bench.summarize.validate_run_inputs",
        lambda _repo_root, _profile_id, _run_root: {"status": "complete"},
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
) -> tuple[Artifact, Path]:
    key = store.key_for(kind, spec)
    with store.staging_directory(key) as staging:
        destination = staging / relative_path
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps(payload), encoding="utf-8")
        artifact = store.publish_from_temp(staging, kind=kind, spec=spec)
    return artifact, artifact.file(relative_path)


def _dataset(dataset_id: str, seed: int) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "generation_job": "dyngen_smoke",
        "simulator_id": "dyngen",
        "seed": seed,
        "dataset_role": "smoke",
        "split_id": "smoke",
        "condition_id": "baseline",
    }


def _evaluation(
    store: ArtifactStore,
    *,
    dataset_id: str,
    run_id: str,
    tool_id: str,
    tool_origin: str,
    configuration_id: str,
    metrics: object,
    algorithm_seed: int = 1729,
    family: str = "tree_ensemble",
    selection_role: str = "baseline",
) -> dict[str, Any]:
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
                "results": {run_id: {"duration_seconds": 5.0}},
            }
        },
    )
    evaluation_spec = {
        "inference_artifact": inference.key.value,
        "evaluator": "andrea",
    }
    evaluation, evaluation_report = _publish_json_artifact(
        store,
        kind="evaluation",
        spec=evaluation_spec,
        relative_path="output/evaluation_report.json",
        payload={"metrics": metrics},
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
        "level": "group",
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
            "level": "group",
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
    metrics = _genie3_metrics() if genie3_metrics is _DEFAULT_METRICS else genie3_metrics
    evaluations = [
        _evaluation(
            store,
            dataset_id="dyngen-1729",
            run_id="genie3_group_emulated",
            tool_id="GENIE3",
            tool_origin="catalog",
            configuration_id="genie3_default",
            metrics=metrics,
        ),
        _evaluation(
            store,
            dataset_id="dyngen-1729",
            run_id="spathi_primary_seed_1729",
            tool_id="custom_spathi_primary_seed_1729",
            tool_origin="custom",
            configuration_id="spathi_primary",
            family="spathi",
            selection_role="selected",
            metrics=[
                {
                    "tool_id": "spathi_primary_seed_1729",
                    "level": "group",
                    "context": "group:a",
                    "status": "not_evaluable",
                    "auroc": None,
                    "aupr": None,
                    "f1_at_truth_count": None,
                    "epr_at_truth_count": None,
                }
            ],
        ),
    ]
    write_state(
        run_root,
        {
            "schema_version": "3.0",
            "profile_id": "smoke",
            "purpose": "smoke",
            "benchmark_repository": {"commit": "4" * 40},
            "status": "exported",
            "artifact_store": str(store.root),
            "datasets": [_dataset("dyngen-1729", 1729)],
            "evaluations": evaluations,
            "cache": {"dataset_hits": 0, "inference_hits": 0, "evaluation_hits": 0},
        },
    )
    return repo_root, run_root


def test_summarize_preserves_rows_and_computes_descriptive_means(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)

    metrics_path, summary_path = summarize(
        repo_root=repo_root,
        profile_id="smoke",
        run_root=run_root,
    )

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["profile_id"] == "smoke"
    assert rows[0]["purpose"] == "smoke"
    assert rows[0]["generation_job"] == "dyngen_smoke"
    assert rows[0]["dataset_role"] == "smoke"
    assert rows[0]["regime"] == "group_only"
    assert rows[0]["tool_origin"] == "catalog"
    assert rows[0]["catalog_tool_id"] == "GENIE3"
    assert rows[0]["configuration_id"] == "genie3_default"
    assert rows[0]["inference_artifact"].startswith("sha256:")
    assert rows[1]["aupr"] == ""
    assert rows[1]["diagnostics"] == '{"reason":"small truth"}'

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "3.0"
    assert summary["benchmark_commit"] == "4" * 40
    assert summary["metric_rows"] == 3
    aggregates = {
        (item["configuration_id"], item["level"], item["regime"]): item
        for item in summary["aggregates"]
    }
    genie3 = aggregates[("genie3_default", "group", "group_only")]
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
    assert ("spathi_primary", "group", "group_only") not in aggregates
    assert summary["notes"][0].startswith("The smoke profile")

    state = load_state(run_root, "smoke")
    assert state["status"] == "completed"
    assert state["reports"] == {
        "metrics_csv": str(metrics_path.resolve()),
        "summary_json": str(summary_path.resolve()),
    }


def test_summarize_requires_exported_state_and_is_idempotent(tmp_path: Path) -> None:
    repo_root = tmp_path / "benchmark-repository"
    repo_root.mkdir()
    run_root = tmp_path / "incomplete"
    write_state(
        run_root,
        {
            "schema_version": "3.0",
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


def test_summarize_refuses_a_stale_reports_directory(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    (run_root / "reports").mkdir()

    with pytest.raises(BenchmarkConfigError, match="Incomplete reports directory already exists"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)


def test_summarize_gives_datasets_equal_weight(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    store = ArtifactStore(Path(state["artifact_store"]))
    state["datasets"].append(_dataset("dyngen-1730", 1730))
    state["evaluations"].append(
        _evaluation(
            store,
            dataset_id="dyngen-1730",
            run_id="genie3_group_emulated",
            tool_id="GENIE3",
            tool_origin="catalog",
            configuration_id="genie3_default",
            metrics=[
                {
                    "tool_id": "genie3_group_emulated",
                    "level": "group",
                    "context": "group:c",
                    "status": "ok",
                    "auroc": 0.1,
                    "aupr": 0.2,
                    "f1_at_truth_count": 0.1,
                    "epr_at_truth_count": 0.2,
                }
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
        if item["configuration_id"] == "genie3_default" and item["level"] == "group"
    )
    assert aggregate["n_datasets"] == 2
    assert aggregate["mean_auroc"] == pytest.approx(0.4)


def test_summarize_averages_algorithm_seeds_before_datasets(tmp_path: Path) -> None:
    repo_root, run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    store = ArtifactStore(Path(state["artifact_store"]))
    state["evaluations"].append(
        _evaluation(
            store,
            dataset_id="dyngen-1729",
            run_id="genie3_seed_1730",
            tool_id="GENIE3",
            tool_origin="catalog",
            configuration_id="genie3_default",
            algorithm_seed=1730,
            metrics=[
                {
                    "tool_id": "genie3_seed_1730",
                    "level": "group",
                    "context": "group:a",
                    "status": "ok",
                    "auroc": 0.1,
                    "aupr": 0.2,
                    "f1_at_truth_count": 0.1,
                    "epr_at_truth_count": 0.2,
                }
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
            dataset_id="dyngen-1729",
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
                    "context": "group:a",
                    "status": "ok",
                    "aupr": aupr,
                }
            ],
        )
        for seed, aupr in ((1729, 0.5), (31415, 0.9))
    ]
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
    assert summary["schema_version"] == "3.0"
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
        dataset_id="dyngen-1729",
        run_id="genie3_seed_31415",
        tool_id="GENIE3",
        tool_origin="catalog",
        configuration_id="genie3_default",
        algorithm_seed=31415,
        metrics=[
            {
                "tool_id": "genie3_seed_31415",
                "level": "group",
                "context": "group:a",
                "status": "ok",
                "aupr": 0.5,
            }
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
            dataset_id="dyngen-1729",
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
                    "status": "ok",
                    "aupr": 0.4,
                },
                {
                    "tool_id": "spathi_candidate_seed_1729",
                    "level": "directed",
                    "context": "group:b",
                    "status": "ok",
                    "aupr": 0.6,
                },
            ],
        )
    ]
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
            "peak_memory_mb": None,
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
            "level": "group",
            "context": "group:a",
            "status": "ok",
        }
    ]
    repo_root, run_root = _evaluated_run(tmp_path, genie3_metrics=metrics)

    with pytest.raises(BenchmarkConfigError, match="differs from its single-run artifact"):
        summarize(repo_root=repo_root, profile_id="smoke", run_root=run_root)
