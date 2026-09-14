from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.artifacts import sha256_file
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.report_publication import publish_workflow_report
from spathi_bench.reporting.common import write_report_bundle


def _state(*, kind: str, flow_id: str, input_path: Path, root: Path) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "workflow_type": kind,
        "profile": "reduced",
        "flow_id": flow_id,
        "identity": {"fingerprint": "a" * 64},
        "status": "publishing",
        "scientific_input": input_path.relative_to(root).as_posix(),
        "scientific_input_sha256": sha256_file(input_path),
    }


@pytest.mark.parametrize("kind", ["calibration", "comparison", "dimensional_transfer", "scalability", "performance"])
def test_report_publication_recovers_crashes_and_missing_members(
    tmp_path: Path,
    kind: str,
) -> None:
    root = tmp_path / "repo"
    flow_id = f"{kind}-fixture"
    workflow_path = root / f"work/v1/workflows/{kind}/reduced/{flow_id}/workflow.json"
    input_path = workflow_path.parent / "scientific-input.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text('{"frozen":true}\n', encoding="utf-8")
    html_path = root / f"work/v1/reports/reduced/{flow_id}/{kind}.html"
    manifest_path = html_path.parent / f"{kind}-report.manifest.json"
    publishing = _state(kind=kind, flow_id=flow_id, input_path=input_path, root=root)

    def crash() -> dict[str, Any]:
        observed = json.loads(workflow_path.read_text(encoding="utf-8"))
        assert observed["status"] == "publishing"
        assert "reports" not in observed
        raise RuntimeError("simulated report crash")

    with pytest.raises(RuntimeError, match="simulated report crash"):
        publish_workflow_report(
            repo_root=root,
            workflow_path=workflow_path,
            publishing_state=publishing,
            html_path=html_path,
            manifest_path=manifest_path,
            render=crash,
            label=kind,
        )
    assert json.loads(workflow_path.read_text(encoding="utf-8"))["status"] == "publishing"

    def render() -> dict[str, Any]:
        observed = json.loads(workflow_path.read_text(encoding="utf-8"))
        assert observed["status"] in {"publishing", "complete"}
        return write_report_bundle(
            repo_root=root,
            kind=kind,
            output_path=html_path,
            payload={
                "workflow": {
                    "flow_id": flow_id,
                    "identity": publishing["identity"],
                    "status": "complete",
                }
            },
            input_paths=[input_path],
            manifest_name=manifest_path.name,
            immutable=True,
        )

    state, bundle = publish_workflow_report(
        repo_root=root,
        workflow_path=workflow_path,
        publishing_state=publishing,
        html_path=html_path,
        manifest_path=manifest_path,
        render=render,
        label=kind,
    )
    assert state["status"] == "complete"
    assert state["reports"] == bundle
    assert html_path.is_file() and manifest_path.is_file()

    manifest_path.unlink()
    resumed, rebuilt = publish_workflow_report(
        repo_root=root,
        workflow_path=workflow_path,
        publishing_state=publishing,
        html_path=html_path,
        manifest_path=manifest_path,
        render=render,
        label=kind,
    )
    assert resumed == state
    assert rebuilt == bundle
    assert sha256_file(manifest_path) == bundle["manifest"]["sha256"]


@pytest.mark.parametrize("kind", ["calibration", "comparison", "dimensional_transfer", "scalability", "performance"])
def test_report_publication_rejects_mutated_published_bytes(
    tmp_path: Path,
    kind: str,
) -> None:
    root = tmp_path / "repo"
    flow_id = f"{kind}-fixture"
    workflow_path = root / f"work/v1/workflows/{kind}/reduced/{flow_id}/workflow.json"
    input_path = workflow_path.parent / "scientific-input.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text('{"frozen":true}\n', encoding="utf-8")
    html_path = root / f"work/v1/reports/reduced/{flow_id}/{kind}.html"
    manifest_path = html_path.parent / f"{kind}-report.manifest.json"
    publishing = _state(kind=kind, flow_id=flow_id, input_path=input_path, root=root)

    def render() -> dict[str, Any]:
        return write_report_bundle(
            repo_root=root,
            kind=kind,
            output_path=html_path,
            payload={
                "workflow": {
                    "flow_id": flow_id,
                    "identity": publishing["identity"],
                    "status": "complete",
                }
            },
            input_paths=[input_path],
            manifest_name=manifest_path.name,
            immutable=True,
        )

    publish_workflow_report(
        repo_root=root,
        workflow_path=workflow_path,
        publishing_state=publishing,
        html_path=html_path,
        manifest_path=manifest_path,
        render=render,
        label=kind,
    )
    html_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="changed after publication"):
        publish_workflow_report(
            repo_root=root,
            workflow_path=workflow_path,
            publishing_state=publishing,
            html_path=html_path,
            manifest_path=manifest_path,
            render=render,
            label=kind,
        )


def test_report_publication_rejects_a_symlinked_report_parent(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    workflow_path = root / "work/v1/workflows/dimensional_transfer/reduced/flow/workflow.json"
    input_path = workflow_path.parent / "input.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("{}\n", encoding="utf-8")
    reports_root = root / "work/v1/reports"
    reports_root.parent.mkdir(parents=True, exist_ok=True)
    reports_root.symlink_to(outside, target_is_directory=True)
    html_path = reports_root / "reduced/flow/dimensional_transfer.html"
    manifest_path = html_path.parent / "dimensional_transfer-report.manifest.json"

    with pytest.raises(BenchmarkConfigError, match="contains a symlink"):
        publish_workflow_report(
            repo_root=root,
            workflow_path=workflow_path,
            publishing_state=_state(
                kind="dimensional_transfer",
                flow_id="flow",
                input_path=input_path,
                root=root,
            ),
            html_path=html_path,
            manifest_path=manifest_path,
            render=lambda: {},
            label="dimensional_transfer",
        )
    assert not list(outside.iterdir())


def test_report_publication_rejects_a_workflow_outside_work(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    input_path = root / "input.json"
    input_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="escapes"):
        publish_workflow_report(
            repo_root=root,
            workflow_path=tmp_path / "outside/workflow.json",
            publishing_state=_state(
                kind="dimensional_transfer",
                flow_id="flow",
                input_path=input_path,
                root=root,
            ),
            html_path=root / "work/reports/dimensional_transfer.html",
            manifest_path=root / "work/reports/dimensional_transfer.manifest.json",
            render=lambda: {},
            label="dimensional_transfer",
        )


def test_report_writer_does_not_follow_a_precreated_static_temporary_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    input_path = root / "work/input.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text("{}\n", encoding="utf-8")
    output = root / "work/reports/dimensional_transfer.html"
    output.parent.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched\n", encoding="utf-8")
    (output.parent / ".dimensional_transfer.html.tmp").symlink_to(outside)

    write_report_bundle(
        repo_root=root,
        kind="dimensional_transfer",
        output_path=output,
        payload={"workflow": {"flow_id": "flow", "identity": {}}},
        input_paths=[input_path],
        manifest_name="dimensional_transfer.manifest.json",
    )

    assert output.is_file()
    assert outside.read_text(encoding="utf-8") == "untouched\n"


def test_report_writer_rejects_a_symlinked_scientific_input(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    target = root / "work/target.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    source = root / "work/input.json"
    source.symlink_to(target)

    with pytest.raises(BenchmarkConfigError, match="contains a symlink"):
        write_report_bundle(
            repo_root=root,
            kind="dimensional_transfer",
            output_path=root / "work/reports/dimensional_transfer.html",
            payload={"workflow": {"flow_id": "flow", "identity": {}}},
            input_paths=[source],
            manifest_name="dimensional_transfer.manifest.json",
        )
