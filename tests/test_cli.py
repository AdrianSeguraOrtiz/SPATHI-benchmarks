from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.cli as cli
from spathi_bench.config import BenchmarkConfigError


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "SPATHI-benchmarks"
    (root / "experiments/v1").mkdir(parents=True)
    (root / "repos.lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def test_discovers_checkout_from_nested_directory(checkout: Path) -> None:
    nested = checkout / "experiments/v1"
    assert cli.discover_repository(nested) == checkout


def test_cli_path_conversion_preserves_symlink_evidence(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    assert cli._path(str(linked)) == linked
    assert cli._path(str(linked)).is_symlink()


@pytest.mark.parametrize(
    ("arguments", "profile", "run_id"),
    [(["run"], None, None), (["run", "reduced", "--id", "pilot"], "reduced", "pilot"),
     (["run", "--id", "pilot"], None, "pilot"), (["run", "smoke"], "smoke", None)],
)
def test_run_uses_one_identity_and_detected_layout(
    checkout: Path, monkeypatch: pytest.MonkeyPatch,
    arguments: list[str], profile: str | None, run_id: str | None,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli, "run_study_workflow", lambda **kwargs: calls.append(kwargs) or {"status": "complete"}
    )
    assert cli.main(arguments) == 0
    assert calls == [{
        "repo_root": checkout, "work_root": checkout / "work",
        "andrea_repo": checkout.parent / "ANDREA", "spathi_repo": checkout.parent / "SPATHI",
        "profile": profile, "run_id": run_id, "max_cores": None, "max_ram_gb": None,
        "reserve_cores": 0, "reserve_ram_gb": 0.0,
    }]


def test_plan_forwards_optional_machine_ceilings(
    checkout: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "plan_study", lambda **kwargs: calls.append(kwargs) or {})
    assert cli.main(["plan", "full", "--cores", "16", "--ram-gb", "64"]) == 0
    assert calls[0]["profile"] == "full"
    assert calls[0]["max_cores"] == 16
    assert calls[0]["max_ram_gb"] == 64.0


def test_status_prints_structured_saved_evidence(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    result = {"run_id": "pilot", "status": "failed", "current_phase": "calibration"}
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "study_status", lambda **kwargs: calls.append(kwargs) or result)
    assert cli.main(["status", "pilot", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == result
    assert calls == [{"repo_root": checkout, "work_root": checkout / "work", "run_id": "pilot"}]


def test_report_dispatches_existing_study_without_repositories(
    checkout: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "rebuild_study_reports", lambda **kwargs: calls.append(kwargs) or {})
    assert cli.main(["report", "pilot"]) == 0
    assert calls == [{"repo_root": checkout, "work_root": checkout / "work", "run_id": "pilot"}]


def test_amend_inputs_requires_a_reason_and_dispatches_explicit_study(checkout, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "amend_study_inputs", lambda **kwargs: calls.append(kwargs) or {})
    assert cli.main(["amend-inputs", "pilot", "--reason", "Correct simulator integration step"]) == 0
    assert calls == [{
        "repo_root": checkout, "work_root": checkout / "work", "run_id": "pilot",
        "andrea_repo": checkout.parent / "ANDREA", "spathi_repo": checkout.parent / "SPATHI",
        "reason": "Correct simulator integration step",
    }]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["amend-inputs", "pilot"])


def test_qualification_uses_same_project_and_resource_contract(
    checkout: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "run_qualification", lambda **kwargs: calls.append(kwargs) or {})
    assert cli.main(["qualify", "--id", "check", "--cores", "8"]) == 0
    assert calls[0]["run_id"] == "check"
    assert calls[0]["max_cores"] == 8
    assert calls[0]["repo_root"] == checkout


def test_doctor_renders_actionable_failure_and_returns_nonzero(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "run_doctor", lambda **kwargs: {
        "ok": False, "checks": [{"id": "docker", "status": "fail", "message": "unavailable",
                                  "remediation": "Start Docker."}],
    })
    assert cli.main(["doctor"]) == 2
    output = capsys.readouterr().out
    assert "FAIL" in output
    assert "Start Docker." in output


def test_cli_reports_contract_failure_without_traceback(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(**kwargs: Any) -> None:
        raise BenchmarkConfigError("Study pilot has a different saved profile")
    monkeypatch.setattr(cli, "run_study_workflow", fail)
    assert cli.main(["run", "full", "--id", "pilot"]) == 2
    assert "different saved profile" in capsys.readouterr().err
