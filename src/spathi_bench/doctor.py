"""Non-destructive environment diagnostics for SPATHI benchmarks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .config import (
    BenchmarkConfigError,
    load_repo_lock,
    validate_all_profiles,
    validate_design_contracts,
    validate_locked_repository,
)
from .provenance import observe_benchmark_repository

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ExecutableResolver = Callable[[str], str | None]

_ANDREA_IMPORT_PROBE = """
import importlib
import pathlib
import sys

repository = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repository))
module = importlib.import_module("andrea")
module_file = pathlib.Path(module.__file__).resolve()
print(module_file)
""".strip()


def _result(
    check_id: str,
    status: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"id": check_id, "status": status, "message": message}
    if details:
        result["details"] = details
    if remediation:
        result["remediation"] = remediation
    return result


def _passed(
    check_id: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _result(check_id, "pass", message, details=details)


def _failed(
    check_id: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    remediation: str,
) -> dict[str, Any]:
    return _result(
        check_id,
        "fail",
        message,
        details=details,
        remediation=remediation,
    )


def _skipped(check_id: str, message: str, *, remediation: str) -> dict[str, Any]:
    return _result(check_id, "skip", message, remediation=remediation)


def _command(
    command: Sequence[str],
    *,
    runner: CommandRunner,
    timeout_seconds: float,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str] | Exception:
    try:
        return runner(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
        )
    except Exception as exc:  # A diagnostic must report command failures, not abort midway.
        return exc


def _command_failure_details(
    command: Sequence[str],
    outcome: subprocess.CompletedProcess[str] | Exception,
) -> dict[str, Any]:
    details: dict[str, Any] = {"command": list(command)}
    if isinstance(outcome, Exception):
        details["error"] = f"{type(outcome).__name__}: {outcome}"
        return details
    details["returncode"] = outcome.returncode
    if (stdout := outcome.stdout or "").strip():
        details["stdout"] = stdout.strip()
    if (stderr := outcome.stderr or "").strip():
        details["stderr"] = stderr.strip()
    return details


def _repository_check(check_id: str, name: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        return _failed(
            check_id,
            f"{name} repository is missing: {resolved}",
            details={"path": str(resolved)},
            remediation=f"Clone {name} at {resolved} or pass the correct repository path.",
        )
    if not resolved.is_dir():
        return _failed(
            check_id,
            f"{name} repository path is not a directory: {resolved}",
            details={"path": str(resolved)},
            remediation=f"Pass the directory containing the {name} Git checkout.",
        )
    if not (resolved / ".git").exists():
        return _failed(
            check_id,
            f"{name} path exists but is not a Git checkout: {resolved}",
            details={"path": str(resolved)},
            remediation=f"Use a Git checkout of {name}, not an unpacked source directory.",
        )
    return _passed(check_id, f"{name} repository exists", details={"path": str(resolved)})


def _docker_checks(
    *,
    resolver: ExecutableResolver,
    runner: CommandRunner,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    docker = resolver("docker")
    if not docker:
        remediation = "Install Docker and make the docker executable available on PATH."
        return [
            _failed("docker_cli", "Docker CLI was not found", remediation=remediation),
            _skipped(
                "docker_daemon",
                "Docker daemon check requires the Docker CLI",
                remediation=remediation,
            ),
            _skipped(
                "docker_buildx",
                "Docker Buildx check requires the Docker CLI",
                remediation=remediation,
            ),
        ]

    cli_command = [docker, "--version"]
    cli_outcome = _command(cli_command, runner=runner, timeout_seconds=timeout_seconds)
    cli_ok = not isinstance(cli_outcome, Exception) and cli_outcome.returncode == 0
    if cli_ok:
        cli = _passed(
            "docker_cli",
            "Docker CLI is available",
            details={"executable": docker, "version": (cli_outcome.stdout or "").strip()},
        )
    else:
        remediation = "Repair the Docker CLI installation or select a working executable."
        return [
            _failed(
                "docker_cli",
                "Docker CLI could not be executed",
                details=_command_failure_details(cli_command, cli_outcome),
                remediation=remediation,
            ),
            _skipped("docker_daemon", "Docker CLI check failed", remediation=remediation),
            _skipped("docker_buildx", "Docker CLI check failed", remediation=remediation),
        ]

    daemon_command = [docker, "info", "--format", "{{.ServerVersion}}"]
    daemon_outcome = _command(daemon_command, runner=runner, timeout_seconds=timeout_seconds)
    if not isinstance(daemon_outcome, Exception) and daemon_outcome.returncode == 0:
        daemon = _passed(
            "docker_daemon",
            "Docker daemon is reachable",
            details={"server_version": (daemon_outcome.stdout or "").strip()},
        )
    else:
        daemon = _failed(
            "docker_daemon",
            "Docker daemon is not reachable",
            details=_command_failure_details(daemon_command, daemon_outcome),
            remediation="Start Docker and ensure the current user can access its daemon.",
        )

    buildx_command = [docker, "buildx", "version"]
    buildx_outcome = _command(buildx_command, runner=runner, timeout_seconds=timeout_seconds)
    if not isinstance(buildx_outcome, Exception) and buildx_outcome.returncode == 0:
        buildx = _passed(
            "docker_buildx",
            "Docker Buildx is available",
            details={"version": (buildx_outcome.stdout or "").strip()},
        )
    else:
        buildx = _failed(
            "docker_buildx",
            "Docker Buildx is unavailable",
            details=_command_failure_details(buildx_command, buildx_outcome),
            remediation="Install or enable the Docker Buildx plugin.",
        )
    return [cli, daemon, buildx]


def _andrea_cli_check(
    andrea_repo: Path,
    python_executable: str,
    *,
    runner: CommandRunner,
    timeout_seconds: float,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(andrea_repo.resolve()) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [python_executable, "-m", "andrea", "--help"]
    outcome = _command(
        command,
        runner=runner,
        timeout_seconds=timeout_seconds,
        environment=environment,
    )
    if not isinstance(outcome, Exception) and outcome.returncode == 0:
        return _passed(
            "andrea_cli",
            "ANDREA CLI runs in the selected Python environment",
            details={"python_executable": python_executable},
        )
    return _failed(
        "andrea_cli",
        "ANDREA CLI could not run in the selected Python environment",
        details=_command_failure_details(command, outcome),
        remediation="Install ANDREA dependencies into the benchmark Python environment.",
    )


def _benchmark_contract_check(
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
) -> dict[str, Any]:
    try:
        benchmark = observe_benchmark_repository(repo_root, require_clean=True)
        validate_all_profiles(repo_root)
        validate_design_contracts(repo_root, andrea_repo)
        lock = load_repo_lock(repo_root)["repositories"]
        andrea = validate_locked_repository(
            name="andrea",
            path=andrea_repo,
            lock_entry=lock["andrea"],
            require_clean=True,
        )
        spathi = validate_locked_repository(
            name="spathi",
            path=spathi_repo,
            lock_entry=lock["spathi"],
            require_clean=True,
        )
        from .pipeline import _validate_spathi_image_contract

        _validate_spathi_image_contract(repo_root)
    except (BenchmarkConfigError, OSError, ValueError) as exc:
        return _failed(
            "benchmark_contracts",
            f"Benchmark locks or committed inputs are inconsistent: {exc}",
            remediation="Commit the benchmark inputs and synchronize repos.lock.json and images.lock.json.",
        )
    return _passed(
        "benchmark_contracts",
        "Benchmark inputs and locked repositories are consistent",
        details={
            "benchmark_commit": benchmark["commit"],
            "andrea_commit": andrea["commit"],
            "spathi_commit": spathi["commit"],
        },
    )


def _andrea_import_check(
    andrea_repo: Path,
    python_executable: str,
    *,
    repository_available: bool,
    runner: CommandRunner,
    timeout_seconds: float,
) -> dict[str, Any]:
    expected_repo = andrea_repo.resolve()
    if not repository_available:
        return _skipped(
            "andrea_import",
            "ANDREA import check requires an existing Git checkout",
            remediation="Provide the expected ANDREA repository and rerun the diagnostic.",
        )
    command = [python_executable, "-c", _ANDREA_IMPORT_PROBE, str(expected_repo)]
    outcome = _command(command, runner=runner, timeout_seconds=timeout_seconds)
    if isinstance(outcome, Exception) or outcome.returncode != 0:
        return _failed(
            "andrea_import",
            "ANDREA could not be imported from the expected repository",
            details=_command_failure_details(command, outcome),
            remediation="Install ANDREA's dependencies and verify that its checkout is importable.",
        )
    stdout = (outcome.stdout or "").strip()
    raw_module_path = stdout.splitlines()[-1] if stdout else ""
    if not raw_module_path:
        return _failed(
            "andrea_import",
            "ANDREA import probe did not report the imported module path",
            details=_command_failure_details(command, outcome),
            remediation="Verify that the expected ANDREA package exposes a normal module file.",
        )
    module_path = Path(raw_module_path).resolve()
    if not module_path.is_relative_to(expected_repo):
        return _failed(
            "andrea_import",
            "ANDREA resolved outside the expected repository",
            details={
                "expected_repository": str(expected_repo),
                "imported_module": str(module_path),
            },
            remediation="Remove the conflicting installation or select the matching ANDREA checkout.",
        )
    return _passed(
        "andrea_import",
        "ANDREA imports from the expected repository",
        details={"repository": str(expected_repo), "module": str(module_path)},
    )


def write_doctor_report(report: dict[str, Any], path: Path) -> Path:
    """Atomically write a doctor report as UTF-8 JSON."""

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def run_doctor(
    *,
    andrea_repo: Path,
    spathi_repo: Path,
    repo_root: Path | None = None,
    python_executable: str | None = None,
    python_version: Sequence[int] | None = None,
    report_path: Path | None = None,
    runner: CommandRunner | None = None,
    resolver: ExecutableResolver | None = None,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Run all read-only environment checks and optionally persist their report.

    ``runner``, ``resolver`` and ``python_version`` are injectable so callers can
    test diagnostics without invoking host commands.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    command_runner = runner or subprocess.run
    executable_resolver = resolver or shutil.which
    selected_python = python_executable or sys.executable
    detected_version = tuple(python_version or sys.version_info[:3])

    if detected_version >= (3, 11):
        python_check = _passed(
            "python_version",
            "Python version is supported",
            details={
                "detected": ".".join(map(str, detected_version)),
                "minimum": "3.11",
                "executable": selected_python,
            },
        )
    else:
        python_check = _failed(
            "python_version",
            f"Python 3.11 or newer is required; found {'.'.join(map(str, detected_version))}",
            details={"detected": list(detected_version), "minimum": [3, 11]},
            remediation="Run SPATHI-benchmarks in a Python 3.11+ environment.",
        )

    andrea_repository = _repository_check("andrea_repository", "ANDREA", Path(andrea_repo))
    spathi_repository = _repository_check("spathi_repository", "SPATHI", Path(spathi_repo))
    checks = [python_check, andrea_repository, spathi_repository]
    checks.extend(
        _docker_checks(
            resolver=executable_resolver,
            runner=command_runner,
            timeout_seconds=timeout_seconds,
        )
    )
    checks.append(
        _andrea_cli_check(
            Path(andrea_repo),
            selected_python,
            runner=command_runner,
            timeout_seconds=timeout_seconds,
        )
    )
    checks.append(
        _andrea_import_check(
            Path(andrea_repo),
            selected_python,
            repository_available=andrea_repository["status"] == "pass",
            runner=command_runner,
            timeout_seconds=timeout_seconds,
        )
    )
    if repo_root is not None:
        checks.append(
            _benchmark_contract_check(
                Path(repo_root),
                Path(andrea_repo),
                Path(spathi_repo),
            )
        )

    counts = {
        status: sum(check["status"] == status for check in checks)
        for status in ("pass", "fail", "skip")
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "ok": counts["fail"] == 0,
        "summary": counts,
        "checks": checks,
    }
    if report_path is not None:
        write_doctor_report(report, report_path)
    return report
