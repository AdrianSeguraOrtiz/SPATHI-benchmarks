from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.pipeline as pipeline
from spathi_bench.artifacts import sha256_file
from spathi_bench.integrate import IntegrationError, integrate_inference_runs

HEADER = ["source", "target", "score", "sign", "evidence", "context", "tool_id"]
FINGERPRINT = {"algorithm": "sha256", "value": "a" * 64}
BENCHMARK_COMMIT = "b" * 40


def _integrate(**kwargs: Any) -> Path:
    return integrate_inference_runs(benchmark_commit=BENCHMARK_COMMIT, **kwargs)


def _write_network(path: Path, *, tool_run_id: str, score: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerow(["TF1", "G1", score, "?", "feature_importance", "group:a", tool_run_id])


def _write_source(
    root: Path,
    *,
    source_run_id: str,
    tool_run_id: str,
    catalog_tool_id: str,
    origin: str,
    dataset_id: str = "dataset-a",
    fingerprint: dict[str, str] | None = None,
) -> Path:
    root.mkdir(parents=True)
    _write_network(root / "merged_network_raw.csv", tool_run_id=tool_run_id, score="2.5")
    _write_network(root / "merged_network_normalized.csv", tool_run_id=tool_run_id, score="0.5")
    report = {
        "run_id": source_run_id,
        "status": "executed",
        "dataset": {
            "id": dataset_id,
            "fingerprint": fingerprint or FINGERPRINT,
            "column_kind": "cells",
            "expression_profile": "scrna",
        },
        "tools": {
            "selected": [tool_run_id],
            "catalog_tool_ids": {tool_run_id: catalog_tool_id},
            "tool_origins": {tool_run_id: origin},
            "output_capabilities": {
                tool_run_id: {
                    "tool_origin": origin,
                    "catalog_tool_id": catalog_tool_id,
                    "directed": True,
                    "sign": "none",
                }
            },
            "skipped": {},
            "status_by_tool": {tool_run_id: "completed"},
            "completed": [tool_run_id],
            "completed_contexts": {tool_run_id: ["group:a"]},
            "failed": {},
            "results": {
                tool_run_id: {
                    "tool_id": tool_run_id,
                    "status": "completed",
                    "execution": {"mode": "group_native"},
                }
            },
        },
        "outputs": {
            "merged_network_raw": "merged_network_raw.csv",
            "merged_network_normalized": "merged_network_normalized.csv",
            "rows_per_tool": {tool_run_id: 1},
        },
        "issues": [],
        "execution": {"elapsed_seconds": 1.0},
    }
    report_path = root / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def _mutate_report(path: Path, update: Any) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    update(report)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _body(path: Path) -> bytes:
    content = path.read_bytes()
    return content[content.index(b"\n") + 1 :]


def test_pipeline_projects_overlapping_batches_through_evaluation_and_analysis_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "work/v1/comparison/test-run"
    run_root.mkdir(parents=True)
    store = pipeline._artifact_store(run_root)
    dataset_spec = {"dataset_id": "dataset-a"}
    with store.staging_directory(store.key_for("dataset", dataset_spec)) as staging:
        dataset_dir = staging / "output/benchmark/datasets/dataset-a"
        dataset_dir.mkdir(parents=True)
        for name in ("dataset-manifest.json", "ground-truth-manifest.json"):
            (dataset_dir / name).write_text("{}\n", encoding="utf-8")
        dataset_artifact = store.publish_from_temp(staging, kind="dataset", spec=dataset_spec)
    dataset = {
        "dataset_id": "dataset-a",
        "dataset_artifact": pipeline._artifact_ref(dataset_artifact, cache_hit=False),
        "dataset_manifest": str(
            dataset_artifact.file("output/benchmark/datasets/dataset-a/dataset-manifest.json")
        ),
        "ground_truth_manifest": str(
            dataset_artifact.file("output/benchmark/datasets/dataset-a/ground-truth-manifest.json")
        ),
    }
    records: list[dict[str, Any]] = []
    source_files: dict[Path, bytes] = {}
    for run_id, tool_id, origin in (
        ("spathi_run", "custom_spathi_run", "custom"),
        ("genie3_run", "genie3", "catalog"),
    ):
        spec = {"run_id": run_id}
        with store.staging_directory(store.key_for("inference", spec)) as staging:
            _write_source(
                staging / "output",
                source_run_id=run_id,
                tool_run_id=run_id,
                catalog_tool_id=tool_id,
                origin=origin,
            )
            artifact = store.publish_from_temp(staging, kind="inference", spec=spec)
        source_files.update(
            {path: path.read_bytes() for path in artifact.path.rglob("*") if path.is_file()}
        )
        records.append(
            {
                "dataset_id": "dataset-a",
                "regime": "group_only",
                "batch_id": "primary",
                "batch_kind": "spathi" if origin == "custom" else "andrea",
                "run_id": run_id,
                "tool_id": tool_id,
                "tool_origin": origin,
                "configuration_id": run_id,
                "selection_role": "selected" if origin == "custom" else "baseline",
                "algorithm_seed": 1,
                "algorithm_seed_parameter": "seed",
                "family": origin,
                "configuration_sha256": "c" * 64,
                "execution_sha256": "e" * 64,
                "dataset_fingerprint": FINGERPRINT,
                "tool_image": {"image_id": tool_id},
                "resources": {"threads": 1},
                "run_report": str(artifact.file("output/run_report.json")),
                "artifact": pipeline._artifact_ref(artifact, cache_hit=False),
            }
        )
    alias = copy.deepcopy(records[0])
    alias["batch_id"] = "all-gene-comparison"
    alias["artifact"]["cache_hit"] = True
    records.append(alias)
    original_records = copy.deepcopy(records)
    state = {
        "status": "validated",
        "datasets": [dataset],
        "inference_runs": records,
        "integrated_inference_runs": [],
        "evaluations": [],
        "joint_evaluations": [],
        "cache": {},
        "benchmark_repository": {"commit": BENCHMARK_COMMIT},
        "repositories": {"observed": {"andrea": {"commit": "a" * 40}}},
        "host_environment": {"fingerprint": "h" * 64},
        "execution_plan": {"operations": {"evaluation": {"cpu_threads": 1, "memory_bytes": 1024}}},
        "runtime_plan": {"capacity": {"cpu_threads": 2, "memory_bytes": 4096}},
    }
    monkeypatch.setattr(pipeline, "_load_existing_state", lambda *_args: state)
    monkeypatch.setattr(pipeline, "_validate_inference_state_contract", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "frozen_repository_root", lambda _root: tmp_path)
    monkeypatch.setattr(pipeline, "_validate_andrea_repository", lambda *_args: {})
    monkeypatch.setattr(pipeline, "_validate_state_andrea", lambda *_args: None)
    monkeypatch.setattr(pipeline, "andrea_host_environment", lambda: {})
    monkeypatch.setattr(pipeline, "_validate_host_environment", lambda *_args: None)
    contract_calls: list[str] = []
    evaluation_calls: list[str] = []

    def contract_command(arguments: list[str], **_kwargs: Any) -> None:
        contract_calls.append(arguments[0])
        if arguments[0] == "validate-inference":
            report = json.loads(Path(arguments[arguments.index("--run-report") + 1]).read_text())
            assert report["tools"]["selected"] == ["genie3_run", "spathi_run"]
            return
        assert arguments[0] == "export-analysis"
        archive = Path(arguments[arguments.index("--output") + 1])
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("test-report.json", "{}\n")
        archive.with_suffix(".zip.manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "complete",
                    "bundle": {"id": "analysis"},
                    "archive": {
                        "filename": "analysis.zip",
                        "sha256": sha256_file(archive),
                        "size": archive.stat().st_size,
                    },
                }
            )
        )

    def evaluate_command(arguments: list[str], **_kwargs: Any) -> None:
        assert arguments[0] == "evaluate-inference"
        evaluation_calls.append(arguments[arguments.index("--run-report") + 1])
        output = Path(arguments[arguments.index("--output-dir") + 1]) / "evaluation"
        output.mkdir(parents=True)
        (output / "evaluation_report.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "andrea_contract_command", contract_command)
    monkeypatch.setattr(pipeline, "andrea_command", evaluate_command)
    arguments = {
        "repo_root": tmp_path,
        "profile_id": "comparison",
        "run_root": run_root,
        "andrea_repo": tmp_path / "andrea",
    }
    pipeline.integrate_inference(**arguments)
    integrated = state["integrated_inference_runs"]
    assert len(integrated) == 1
    assert integrated[0]["source_run_ids"] == ["genie3_run", "spathi_run"]
    assert len(integrated[0]["source_inference_artifacts"]) == 2
    integration_key = integrated[0]["artifact"]["key"]
    state["status"] = "validated"
    pipeline.integrate_inference(**arguments)
    assert state["integrated_inference_runs"][0]["artifact"]["key"] == integration_key
    assert state["cache"]["integration_hits"] == 1
    assert contract_calls == ["validate-inference"]

    pipeline.evaluate(**arguments)
    assert len(evaluation_calls) == len(set(evaluation_calls)) == 3
    assert len(state["evaluations"]) == 3
    assert len(state["joint_evaluations"]) == 1
    assert len({row["artifact"]["key"] for row in state["evaluations"]}) == 2
    schedule = json.loads((run_root / "schedules/evaluation.json").read_text())
    assert schedule["logical_tasks"] == 4
    assert schedule["physical_tasks"] == 3

    pipeline.export_analysis(**arguments)
    assert len(state["analysis_bundles"]) == 7
    assert contract_calls.count("export-analysis") == 7
    exported = json.loads((run_root / "exports/manifest.json").read_text())
    assert len({row["analysis_zip"] for row in exported["bundles"]}) == 7
    for row in exported["bundles"]:
        archive = run_root / "exports" / row["analysis_zip"]
        assert zipfile.is_zipfile(archive)
        assert sha256_file(archive) == row["archive_sha256"]
    exports_before = pipeline._tree_file_hashes(run_root / "exports")
    pipeline.export_analysis(**arguments)
    assert contract_calls.count("export-analysis") == 7
    assert pipeline._tree_file_hashes(run_root / "exports") == exports_before
    assert records == original_records
    assert {path: path.read_bytes() for path in source_files} == source_files


def test_integrates_deterministically_and_preserves_source_rows(tmp_path: Path) -> None:
    source_z = _write_source(
        tmp_path / "source-z",
        source_run_id="z-inference",
        tool_run_id="spathi_run",
        catalog_tool_id="custom_spathi_run",
        origin="custom",
    )
    source_a = _write_source(
        tmp_path / "source-a",
        source_run_id="a-inference",
        tool_run_id="genie3_run",
        catalog_tool_id="genie3",
        origin="catalog",
    )
    validated: list[Path] = []

    report_path = _integrate(
        run_report_paths=[source_z, source_a],
        output_dir=tmp_path / "joint-a",
        combined_run_id="dataset-a_group-only",
        regime="group_only",
        contract_validator=lambda path: validated.append(path),
    )
    second_report_path = _integrate(
        run_report_paths=[source_a, source_z],
        output_dir=tmp_path / "joint-b",
        combined_run_id="dataset-a_group-only",
        regime="group_only",
    )

    assert validated and validated[0].name == "run_report.json"
    assert not validated[0].parent.name.startswith("joint-a")
    assert report_path.read_bytes() == second_report_path.read_bytes()
    assert (report_path.parent / "merged_network_raw.csv").read_bytes() == (
        b",".join(column.encode() for column in HEADER)
        + b"\n"
        + _body(source_a.parent / "merged_network_raw.csv")
        + _body(source_z.parent / "merged_network_raw.csv")
    )
    assert (report_path.parent / "merged_network_normalized.csv").read_bytes() == (
        b",".join(column.encode() for column in HEADER)
        + b"\n"
        + _body(source_a.parent / "merged_network_normalized.csv")
        + _body(source_z.parent / "merged_network_normalized.csv")
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["tools"]["selected"] == ["genie3_run", "spathi_run"]
    assert report["outputs"]["rows_per_tool"] == {"genie3_run": 1, "spathi_run": 1}
    assert report["integration"]["monolithic_execution"] is False
    assert report["integration"]["producer_commit"] == BENCHMARK_COMMIT
    assert [source["source_run_id"] for source in report["integration"]["sources"]] == [
        "a-inference",
        "z-inference",
    ]
    for source in report["integration"]["sources"]:
        original = source_a if source["source_run_id"] == "a-inference" else source_z
        assert (
            source["files"]["run_report.json"]["sha256"]
            == hashlib.sha256(original.read_bytes()).hexdigest()
        )


def test_joint_view_satisfies_locked_andrea_analysis_contract(tmp_path: Path) -> None:
    source_a = _write_source(
        tmp_path / "source-a",
        source_run_id="genie3-inference",
        tool_run_id="genie3_run",
        catalog_tool_id="genie3",
        origin="catalog",
    )
    source_b = _write_source(
        tmp_path / "source-b",
        source_run_id="spathi-inference",
        tool_run_id="spathi_run",
        catalog_tool_id="custom_spathi_run",
        origin="custom",
    )
    andrea_checkout = Path(__file__).resolve().parents[2] / "ANDREA"
    report_path = _integrate(
        run_report_paths=[source_a, source_b],
        output_dir=tmp_path / "joint",
        combined_run_id="joint_group_only",
        regime="group_only",
    )
    script = """
import json
import sys
from pathlib import Path
from andrea.core.commands.evaluate_inference import validate_inference_analysis_inputs
from andrea.core.commands.infer_network.bundles import resolve_bundle

report_path = Path(sys.argv[1])
validate_inference_analysis_inputs(run_report_path=report_path)
resolution = resolve_bundle(bundle_id="analysis", run_dir=report_path.parent)
print(json.dumps({
    "available": resolution.available,
    "missing": resolution.missing_required,
    "sources": [source.virtual_path for source in resolution.sources],
}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(andrea_checkout)
    completed = subprocess.run(
        ["python3", "-c", script, str(report_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=andrea_checkout,
        env=environment,
    )
    resolution = json.loads(completed.stdout)

    assert resolution["available"] is True
    assert resolution["missing"] == []
    assert resolution["sources"] == [
        "merged_network_normalized.csv",
        "merged_network_raw.csv",
        "run_report.json",
    ]


def test_integrates_a_completed_empty_network(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path / "source",
        source_run_id="empty-inference",
        tool_run_id="empty_run",
        catalog_tool_id="genie3",
        origin="catalog",
    )
    header = ",".join(HEADER) + "\n"
    (source.parent / "merged_network_raw.csv").write_text(header, encoding="utf-8")
    (source.parent / "merged_network_normalized.csv").write_text(header, encoding="utf-8")
    _mutate_report(
        source,
        lambda report: report["outputs"]["rows_per_tool"].update({"empty_run": 0}),
    )

    report_path = _integrate(
        run_report_paths=[source],
        output_dir=tmp_path / "joint",
        combined_run_id="joint-empty",
        regime="group_only",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["outputs"]["rows_per_tool"] == {"empty_run": 0}
    assert report["tools"]["completed_contexts"] == {"empty_run": ["group:a"]}
    assert (report_path.parent / "merged_network_raw.csv").read_text(encoding="utf-8") == header


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "dataset-b"),
        ("fingerprint", {"algorithm": "sha256", "value": "b" * 64}),
    ],
)
def test_rejects_different_dataset_identity(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    source_a = _write_source(
        tmp_path / "source-a",
        source_run_id="inference-a",
        tool_run_id="run_a",
        catalog_tool_id="genie3",
        origin="catalog",
    )
    kwargs: dict[str, Any] = {"dataset_id": "dataset-a", "fingerprint": FINGERPRINT}
    kwargs["dataset_id" if field == "id" else field] = value
    source_b = _write_source(
        tmp_path / "source-b",
        source_run_id="inference-b",
        tool_run_id="run_b",
        catalog_tool_id="grnboost2",
        origin="catalog",
        **kwargs,
    )

    with pytest.raises(IntegrationError, match="same dataset ID and fingerprint"):
        _integrate(
            run_report_paths=[source_a, source_b],
            output_dir=tmp_path / "joint",
            combined_run_id="joint",
            regime="group_only",
        )
    assert not (tmp_path / "joint").exists()


def test_rejects_duplicate_source_or_tool_run_ids(tmp_path: Path) -> None:
    source_a = _write_source(
        tmp_path / "source-a",
        source_run_id="same-source",
        tool_run_id="run_a",
        catalog_tool_id="genie3",
        origin="catalog",
    )
    source_b = _write_source(
        tmp_path / "source-b",
        source_run_id="same-source",
        tool_run_id="run_b",
        catalog_tool_id="grnboost2",
        origin="catalog",
    )
    with pytest.raises(IntegrationError, match="unique run_id"):
        _integrate(
            run_report_paths=[source_a, source_b],
            output_dir=tmp_path / "joint-source",
            combined_run_id="joint",
            regime="group_only",
        )

    source_c = _write_source(
        tmp_path / "source-c",
        source_run_id="different-source",
        tool_run_id="run_a",
        catalog_tool_id="genie3",
        origin="catalog",
    )
    with pytest.raises(IntegrationError, match="Tool run IDs must be unique"):
        _integrate(
            run_report_paths=[source_a, source_c],
            output_dir=tmp_path / "joint-tool",
            combined_run_id="joint",
            regime="group_only",
        )


def test_rejects_inconsistent_output_capabilities_and_row_counts(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path / "source",
        source_run_id="inference-a",
        tool_run_id="run_a",
        catalog_tool_id="genie3",
        origin="catalog",
    )
    _mutate_report(
        source,
        lambda report: report["tools"]["output_capabilities"]["run_a"].update({"directed": "yes"}),
    )
    with pytest.raises(IntegrationError, match="directed.*must be boolean"):
        _integrate(
            run_report_paths=[source],
            output_dir=tmp_path / "joint-capability",
            combined_run_id="joint",
            regime="group_only",
        )

    _mutate_report(
        source,
        lambda report: report["tools"]["output_capabilities"]["run_a"].update({"directed": True}),
    )
    _mutate_report(
        source,
        lambda report: report["outputs"]["rows_per_tool"].update({"run_a": 2}),
    )
    with pytest.raises(IntegrationError, match="row counts do not match"):
        _integrate(
            run_report_paths=[source],
            output_dir=tmp_path / "joint-count",
            combined_run_id="joint",
            regime="group_only",
        )


def test_rejects_different_raw_and_normalized_schemas(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path / "source",
        source_run_id="inference-a",
        tool_run_id="run_a",
        catalog_tool_id="genie3",
        origin="catalog",
    )
    normalized = source.parent / "merged_network_normalized.csv"
    normalized.write_text(
        "target,source,score,sign,evidence,context,tool_id\n"
        "G1,TF1,0.5,?,feature_importance,group:a,run_a\n",
        encoding="utf-8",
    )

    with pytest.raises(IntegrationError, match="schema must be exactly"):
        _integrate(
            run_report_paths=[source],
            output_dir=tmp_path / "joint",
            combined_run_id="joint",
            regime="group_only",
        )


def test_contract_failure_is_atomic(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path / "source",
        source_run_id="inference-a",
        tool_run_id="run_a",
        catalog_tool_id="genie3",
        origin="catalog",
    )

    def reject(_: Path) -> None:
        raise ValueError("contract mismatch")

    with pytest.raises(IntegrationError, match="ANDREA rejected.*contract mismatch"):
        _integrate(
            run_report_paths=[source],
            output_dir=tmp_path / "joint",
            combined_run_id="joint",
            regime="group_only",
            contract_validator=reject,
        )
    assert not (tmp_path / "joint").exists()
    assert not list(tmp_path.glob(".joint.tmp-*"))


def test_rejects_an_invalid_benchmark_commit(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path / "source",
        source_run_id="inference-a",
        tool_run_id="run_a",
        catalog_tool_id="genie3",
        origin="catalog",
    )

    with pytest.raises(IntegrationError, match="benchmark_commit"):
        integrate_inference_runs(
            run_report_paths=[source],
            output_dir=tmp_path / "joint",
            combined_run_id="joint",
            regime="group_only",
            benchmark_commit="not-a-commit",
        )
