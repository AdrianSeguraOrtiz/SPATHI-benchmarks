from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

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
