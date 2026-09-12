from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import spathi_bench.qualification as qualification
import spathi_bench.study_workflow as study
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.study_context import phase_flow_id, study_execution
from spathi_bench.study_state import PHASES, load_study, new_id, study_directory


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def setup_study(tmp_path: Path, monkeypatch):
    repositories = {}
    for name in ("repo", "ANDREA", "SPATHI"):
        root = tmp_path / name
        root.mkdir()
        _git(root, "init", "-q")
        _git(root, "config", "user.name", "Test")
        _git(root, "config", "user.email", "test@example.invalid")
        _git(root, "remote", "add", "origin", f"https://example.invalid/{name}")
        (root / ".gitignore").write_text("work/\n")
        (root / "README.md").write_text("Initial revision\n")
        if name == "repo":
            (root / "experiments").mkdir()
            (root / "experiments/design.json").write_text('{"seed":42}')
        _git(root, "add", ".")
        _git(root, "commit", "-qm", "Initial")
        repositories[name] = root
    events = []
    fail = {"phase": None}
    qualification_calls = []

    def qualify(**kwargs):
        qualification_calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(qualification, "run_qualification", qualify)
    monkeypatch.setattr(study, "detect_host_capacity", lambda **_: {
        "effective_cpu_threads": 8, "effective_memory_bytes": 16 * 1024**3,
    })

    def workflow(name):
        def execute(**kwargs):
            events.append((name, kwargs))
            if fail["phase"] == name:
                raise RuntimeError("Injected interruption")
            flow = phase_flow_id("content-dependent-fallback")
            budget = kwargs.get("profile", kwargs.get("calibration_budget", "smoke"))
            phase_root = kwargs["work_root"] / "v1/workflows" / name / budget / flow
            if name == "smoke":
                phase_root = kwargs["work_root"] / "v1/smoke" / flow
            phase_root.mkdir(parents=True, exist_ok=True)
            result = {"flow_id": flow, "status": "complete"}
            (phase_root / "workflow.json").write_text(json.dumps(result))
            return result
        return execute

    for phase in PHASES:
        monkeypatch.setattr(study, f"run_{phase}_workflow", workflow(phase))
    monkeypatch.setattr(study, "_smoke", workflow("smoke"))
    common = dict(repo_root=repositories["repo"], andrea_repo=repositories["ANDREA"],
                  spathi_repo=repositories["SPATHI"], work_root=repositories["repo"] / "work")
    return common, events, fail, qualification_calls


def test_named_study_carries_upstream_ids_and_completed_resume_runs_nothing(setup_study, monkeypatch):
    common, events, _, qualifications = setup_study
    first = study.run_study_workflow(**common, profile="reduced", run_id="pilot", max_cores=4)
    assert first["status"] == "complete"
    assert [name for name, _ in events] == list(PHASES)
    assert len(qualifications) == 1
    assert all(args["work_root"] == common["work_root"] / "studies/pilot" for _, args in events)
    assert events[1][1]["calibration_run_id"] == "pilot"
    assert all(args["efficiency_run_id"] == "pilot" for _, args in events[2:])
    completed_at = load_study(common["work_root"] / "studies/pilot")["completed_at"]
    monkeypatch.setattr(study, "now", lambda: "2099-01-01T00:00:00+00:00")
    second = study.run_study_workflow(**common, run_id="pilot")
    assert second["status"] == "complete"
    assert len(events) == 6
    assert len(qualifications) == 1
    journal = load_study(common["work_root"] / "studies/pilot")
    assert len(journal["invocations"]) == 2
    assert journal["completed_at"] == completed_at
    assert journal["resources"]["max_cores"] == 4
    assert Path(second["report_path"]).is_file()


def test_interrupted_study_resumes_failed_phase_not_completed_calibration(setup_study):
    common, events, fail, _ = setup_study
    fail["phase"] = "efficiency"
    with pytest.raises(RuntimeError, match="Injected"):
        study.run_study_workflow(**common, profile="reduced", run_id="pilot")
    journal = load_study(common["work_root"] / "studies/pilot")
    assert journal["status"] == "failed"
    assert journal["current_phase"] == "efficiency"
    assert journal["phases"]["calibration"]["status"] == "complete"
    fail["phase"] = None
    study.run_study_workflow(**common, run_id="pilot")
    assert [name for name, _ in events] == ["calibration", "efficiency", *PHASES[1:]]


def test_commit_drift_warns_and_preserves_completed_phase(setup_study, caplog):
    common, events, fail, _ = setup_study
    fail["phase"] = "efficiency"
    with pytest.raises(RuntimeError):
        study.run_study_workflow(**common, profile="reduced", run_id="pilot")
    root = common["repo_root"]
    (root / "README.md").write_text("Updated documentation\n")
    _git(root, "add", "README.md")
    _git(root, "commit", "-qm", "Documentation")
    fail["phase"] = None
    result = study.run_study_workflow(**common, run_id="pilot")
    assert "revision changed" in caplog.text
    assert result["warnings"]
    assert sum(name == "calibration" for name, _ in events) == 1
    journal = load_study(common["work_root"] / "studies/pilot")
    assert journal["invocations"][0]["revisions"] != journal["invocations"][1]["revisions"]


def test_changed_seed_or_tampered_receipt_is_not_revision_drift(setup_study):
    common, _, _, _ = setup_study
    study.run_study_workflow(**common, profile="reduced", run_id="pilot")
    root = common["repo_root"]
    (root / "experiments/design.json").write_text('{"seed":99}')
    _git(root, "add", "experiments/design.json")
    _git(root, "commit", "-qm", "Change seed")
    with pytest.raises(BenchmarkConfigError, match="scientific inputs changed"):
        study.run_study_workflow(**common, run_id="pilot")


def test_completed_evidence_corruption_prevents_resume(setup_study):
    common, _, _, _ = setup_study
    study.run_study_workflow(**common, profile="reduced", run_id="pilot")
    path = common["work_root"] / "studies/pilot/v1/workflows/calibration/reduced/pilot/workflow.json"
    path.write_text("{}")
    with pytest.raises(BenchmarkConfigError, match="missing or changed"):
        study.run_study_workflow(**common, run_id="pilot")


def test_full_and_reduced_ids_coexist_without_discovery(setup_study):
    common, _, _, _ = setup_study
    study.run_study_workflow(**common, profile="reduced", run_id="pilot")
    study.run_study_workflow(**common, profile="full", run_id="paper")
    statuses = study.study_status(repo_root=common["repo_root"], work_root=common["work_root"])
    assert {item["profile"] for item in statuses["studies"]} == {"reduced", "full"}
    with pytest.raises(BenchmarkConfigError, match="cannot change its profile"):
        study.run_study_workflow(**common, profile="full", run_id="pilot")


def test_smoke_runs_only_docker_contract_pipeline(setup_study):
    common, events, _, qualifications = setup_study
    result = study.run_study_workflow(**common, profile="smoke", run_id="check")
    assert result["status"] == "complete"
    assert [name for name, _ in events] == ["smoke"]
    assert not qualifications


def test_generated_ids_obey_every_workflow_contract(tmp_path):
    from spathi_bench.workflows import _flow_id

    generated = new_id("reduced")
    assert study_directory(tmp_path, tmp_path / "work", generated).name == generated
    assert _flow_id({}, generated) == generated


@pytest.mark.parametrize("identity", ["../outside", "Upper", "has.dot", "", "a" * 97])
def test_invalid_study_identity_is_rejected(tmp_path, identity):
    with pytest.raises(BenchmarkConfigError, match="Study ID"):
        study_directory(tmp_path, tmp_path / "work", identity)


def test_revision_context_is_local_to_one_invocation():
    assert phase_flow_id("content") == "content"
    with study_execution("pilot", revisions={}, resume=True):
        assert phase_flow_id("content") == "pilot"
    assert phase_flow_id("content") == "content"
