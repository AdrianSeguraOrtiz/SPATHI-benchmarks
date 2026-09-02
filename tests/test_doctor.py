from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from spathi_bench.doctor import run_doctor, write_doctor_report


def _repositories(tmp_path: Path) -> tuple[Path, Path]:
    andrea = tmp_path / "ANDREA"
    spathi = tmp_path / "SPATHI"
    for repository in (andrea, spathi):
        repository.mkdir()
        (repository / ".git").mkdir()
    (andrea / "andrea").mkdir()
    (andrea / "andrea" / "__init__.py").write_text("", encoding="utf-8")
    return andrea, spathi


def _successful_runner(
    andrea_repo: Path,
    spathi_repo: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "/usr/bin/docker" and command[1:] == ["--version"]:
            stdout = "Docker version 28.0.0\n"
        elif command[0] == "/usr/bin/docker" and command[1:3] == ["info", "--format"]:
            stdout = "28.0.0\n"
        elif command[0] == "/usr/bin/docker" and command[1:] == ["buildx", "version"]:
            stdout = "github.com/docker/buildx v0.24.0\n"
        elif command[0] == "/venv/bin/python" and command[1:] == ["-m", "andrea", "--help"]:
            stdout = "usage: andrea\n"
        elif (
            command[0] == "/venv/bin/python"
            and command[1] == "-c"
            and Path(command[-1]) == spathi_repo.resolve()
        ):
            stdout = json.dumps(
                {
                    "module": str(spathi_repo / "src" / "spathi" / "__init__.py"),
                    "missing_exports": [],
                    "missing_config_fields": [],
                    "unsupported_required_config_fields": [],
                    "missing_progress_members": [],
                    "missing_result_fields": [],
                    "infer_signature_ok": True,
                }
            )
        elif command[0] == "/venv/bin/python" and command[1] == "-c":
            stdout = f"{andrea_repo / 'andrea' / '__init__.py'}\n"
        else:  # pragma: no cover - makes unexpected subprocesses immediately visible
            raise AssertionError(f"Unexpected command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    return run


def _resolver(executable: str) -> str | None:
    return {"docker": "/usr/bin/docker"}.get(executable)


def _by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {check["id"]: check for check in report["checks"]}  # type: ignore[index, misc]


def test_run_doctor_reports_a_complete_healthy_environment(tmp_path: Path) -> None:
    andrea, spathi = _repositories(tmp_path)
    output = tmp_path / "reports" / "doctor.json"

    report = run_doctor(
        andrea_repo=andrea,
        spathi_repo=spathi,
        python_executable="/venv/bin/python",
        python_version=(3, 11, 9),
        report_path=output,
        runner=_successful_runner(andrea, spathi),
        resolver=_resolver,
    )

    assert report["ok"] is True
    assert report["summary"] == {"pass": 9, "fail": 0, "skip": 0}
    assert {check["status"] for check in report["checks"]} == {"pass"}
    assert json.loads(output.read_text(encoding="utf-8")) == report
    checks = _by_id(report)
    assert checks["andrea_cli"]["details"]["python_executable"] == "/venv/bin/python"  # type: ignore[index]
    assert checks["andrea_import"]["details"]["repository"] == str(andrea.resolve())  # type: ignore[index]
    assert checks["spathi_api"]["details"]["repository"] == str(spathi.resolve())  # type: ignore[index]


def test_missing_docker_fails_cli_and_skips_dependent_checks(tmp_path: Path) -> None:
    andrea, spathi = _repositories(tmp_path)

    report = run_doctor(
        andrea_repo=andrea,
        spathi_repo=spathi,
        python_executable="/venv/bin/python",
        runner=_successful_runner(andrea, spathi),
        resolver=lambda _executable: None,
    )

    checks = _by_id(report)
    assert report["ok"] is False
    assert checks["docker_cli"]["status"] == "fail"
    assert checks["docker_daemon"]["status"] == "skip"
    assert checks["docker_buildx"]["status"] == "skip"
    assert "Install Docker" in checks["docker_cli"]["remediation"]  # type: ignore[operator]


def test_old_python_and_missing_repositories_have_clear_failures(tmp_path: Path) -> None:
    andrea = tmp_path / "missing-ANDREA"
    spathi = tmp_path / "not-a-repository"
    spathi.mkdir()

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[0] == "/usr/bin/docker":
            return subprocess.CompletedProcess(command, 0, stdout="available\n", stderr="")
        if command == ["/venv/bin/python", "-m", "andrea", "--help"]:
            return subprocess.CompletedProcess(command, 0, stdout="usage\n", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    report = run_doctor(
        andrea_repo=andrea,
        spathi_repo=spathi,
        python_version=(3, 10, 14),
        runner=runner,
        resolver=_resolver,
    )

    checks = _by_id(report)
    assert "Python 3.11 or newer" in checks["python_version"]["message"]  # type: ignore[operator]
    assert "missing" in checks["andrea_repository"]["message"]  # type: ignore[operator]
    assert "not a Git checkout" in checks["spathi_repository"]["message"]  # type: ignore[operator]
    assert checks["andrea_import"]["status"] == "skip"


def test_andrea_import_must_resolve_inside_expected_checkout(tmp_path: Path) -> None:
    andrea, spathi = _repositories(tmp_path)
    installed = tmp_path / "site-packages" / "andrea" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        outcome = _successful_runner(andrea, spathi)(command, **kwargs)
        if (
            command[0] == "/venv/bin/python"
            and command[1] == "-c"
            and Path(command[-1]) == andrea.resolve()
        ):
            return subprocess.CompletedProcess(command, 0, stdout=f"{installed}\n", stderr="")
        return outcome

    report = run_doctor(
        andrea_repo=andrea,
        spathi_repo=spathi,
        python_executable="/venv/bin/python",
        runner=runner,
        resolver=_resolver,
    )

    check = _by_id(report)["andrea_import"]
    assert report["ok"] is False
    assert check["status"] == "fail"
    assert check["details"]["imported_module"] == str(installed.resolve())  # type: ignore[index]


def test_spathi_api_check_reports_missing_public_exports(tmp_path: Path) -> None:
    andrea, spathi = _repositories(tmp_path)

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        outcome = _successful_runner(andrea, spathi)(command, **kwargs)
        if (
            command[0] == "/venv/bin/python"
            and command[1] == "-c"
            and Path(command[-1]) == spathi.resolve()
        ):
            payload = {
                "module": str(spathi / "src/spathi/__init__.py"),
                "missing_exports": ["infer"],
                "missing_config_fields": [],
                "unsupported_required_config_fields": [],
                "missing_progress_members": [],
                "missing_result_fields": [],
                "infer_signature_ok": True,
            }
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(payload),
                stderr="",
            )
        return outcome

    report = run_doctor(
        andrea_repo=andrea,
        spathi_repo=spathi,
        python_executable="/venv/bin/python",
        runner=runner,
        resolver=_resolver,
    )

    check = _by_id(report)["spathi_api"]
    assert report["ok"] is False
    assert check["status"] == "fail"
    assert check["details"]["missing_exports"] == ["infer"]  # type: ignore[index]


def test_spathi_api_check_reports_an_incompatible_adapter_contract(
    tmp_path: Path,
) -> None:
    andrea, spathi = _repositories(tmp_path)

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        outcome = _successful_runner(andrea, spathi)(command, **kwargs)
        if (
            command[0] == "/venv/bin/python"
            and command[1] == "-c"
            and Path(command[-1]) == spathi.resolve()
        ):
            payload = {
                "module": str(spathi / "src/spathi/__init__.py"),
                "missing_exports": [],
                "missing_config_fields": ["report"],
                "unsupported_required_config_fields": ["new_required_option"],
                "missing_progress_members": ["model_fraction"],
                "missing_result_fields": ["warnings"],
                "infer_signature_ok": False,
            }
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(payload),
                stderr="",
            )
        return outcome

    report = run_doctor(
        andrea_repo=andrea,
        spathi_repo=spathi,
        python_executable="/venv/bin/python",
        runner=runner,
        resolver=_resolver,
    )

    check = _by_id(report)["spathi_api"]
    assert report["ok"] is False
    assert check["status"] == "fail"
    assert check["details"]["missing_config_fields"] == ["report"]  # type: ignore[index]
    assert check["details"]["unsupported_required_config_fields"] == [  # type: ignore[index]
        "new_required_option"
    ]
    assert check["details"]["missing_progress_members"] == [  # type: ignore[index]
        "model_fraction"
    ]
    assert check["details"]["missing_result_fields"] == ["warnings"]  # type: ignore[index]
    assert check["details"]["infer_signature_ok"] is False  # type: ignore[index]


def test_command_errors_are_reported_instead_of_raised(tmp_path: Path) -> None:
    andrea, spathi = _repositories(tmp_path)

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["/usr/bin/docker", "info"]:
            raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])
        return _successful_runner(andrea, spathi)(command, **kwargs)

    report = run_doctor(
        andrea_repo=andrea,
        spathi_repo=spathi,
        python_executable="/venv/bin/python",
        runner=runner,
        resolver=_resolver,
    )

    check = _by_id(report)["docker_daemon"]
    assert check["status"] == "fail"
    assert "TimeoutExpired" in check["details"]["error"]  # type: ignore[operator, index]


def test_write_doctor_report_is_valid_json_and_validates_timeout(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"
    report = {"schema_version": "1.0", "ok": True, "checks": []}

    assert write_doctor_report(report, target) == target.resolve()
    assert json.loads(target.read_text(encoding="utf-8")) == report

    with pytest.raises(ValueError, match="greater than zero"):
        run_doctor(
            andrea_repo=tmp_path / "ANDREA",
            spathi_repo=tmp_path / "SPATHI",
            timeout_seconds=0,
        )
