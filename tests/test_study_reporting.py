from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import spathi_bench.study_reporting as reporting
from spathi_bench.artifacts import sha256_file
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.study_reporting import rebuild_phase_reports, render_study_index


def _study(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    repo = tmp_path / "repo"
    root = repo / "work/studies/pilot"
    root.mkdir(parents=True)
    (root / "study.json").write_text("{}", encoding="utf-8")
    state = {
        "run_id": "pilot", "profile": "reduced", "status": "failed",
        "phases": {"dimensional_transfer": {"status": "failed", "error": "candidate <invalid>"}},
        "resource_plan": {"capacity": {"cpu_threads": 16, "memory_bytes": 68719476736}},
        "warnings": ["Revision <changed>; prior evidence retained."],
        "invocations": [{"started_at": "2026-01-01", "revisions": {
            name: {"commit": name + "-commit"} for name in ("benchmark", "andrea", "spathi")
        }}],
    }
    return repo, root, state


def test_study_index_exposes_scope_failure_resources_and_revision_history(tmp_path: Path) -> None:
    repo, root, state = _study(tmp_path)
    output = render_study_index(repo, root, state)
    content = output.read_text(encoding="utf-8")
    assert output == root / "reports/index.html"
    assert content.count('<article class="phase">') == 5
    assert "Descriptive development study" in content
    assert "candidate &lt;invalid&gt;" in content
    assert "Revision &lt;changed&gt;" in content
    assert 'href="../study.json"' in content
    assert "benchmark-commit" in content
    assert "68719476736" in content


def test_smoke_index_links_actual_pipeline_outputs(tmp_path: Path) -> None:
    repo, root, state = _study(tmp_path)
    phase = root / "v1/smoke/pilot"
    (phase / "reports").mkdir(parents=True)
    (phase / "reports/summary.json").write_text("{}", encoding="utf-8")
    (phase / "exports").mkdir()
    (phase / "exports/manifest.json").write_text("{}", encoding="utf-8")
    state.update(profile="smoke", phases={"smoke": {
        "status": "complete", "workflow_root": "v1/smoke/pilot",
    }})
    content = render_study_index(repo, root, state).read_text(encoding="utf-8")
    assert content.count('<article class="phase">') == 1
    assert 'href="../v1/smoke/pilot/reports/summary.json"' in content
    assert "ANDREA analysis archives" in content
    assert "Operational integration check" in content


def test_report_link_cannot_escape_the_study_directory(tmp_path: Path) -> None:
    repo, root, state = _study(tmp_path)
    state["phases"]["dimensional_transfer"]["report"] = "../../outside.html"
    with pytest.raises(BenchmarkConfigError, match="escapes"):
        render_study_index(repo, root, state)


def test_report_recovery_reconstructs_missing_html_then_verifies_frozen_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, root, state = _study(tmp_path)
    workflow = root / "v1/workflows/dimensional_transfer/reduced/pilot"
    workflow.mkdir(parents=True)
    science = workflow / "results.json"
    science.write_text('{"aupr": 0.25}', encoding="utf-8")
    report = root / "v1/reports/reduced/pilot/dimensional_transfer.html"
    report.parent.mkdir(parents=True)
    report.write_text("frozen HTML", encoding="utf-8")
    evidence = {str(path.relative_to(root)): sha256_file(path) for path in (science, report)}
    report.unlink()
    state["phases"] = {"dimensional_transfer": {
        "status": "complete", "workflow_root": str(workflow.relative_to(root)),
        "report": str(report.relative_to(root)), "evidence": evidence,
    }}
    calls: list[dict[str, Any]] = []

    def renderer(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        report.write_text("frozen HTML", encoding="utf-8")
        return {"html": {"path": str(report.relative_to(repo)), "sha256": sha256_file(report)}}

    monkeypatch.setitem(reporting._RENDERERS, "dimensional_transfer", renderer)
    result = rebuild_phase_reports(repo, root, state)
    assert result["dimensional_transfer"]["html"]["sha256"] == evidence[str(report.relative_to(root))]
    assert calls == [{"repo_root": repo, "version_root": root / "v1", "workflow_root": workflow}]

    science.write_text('{"aupr": 0.9}', encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="missing or changed"):
        rebuild_phase_reports(repo, root, state)
    assert len(calls) == 1
