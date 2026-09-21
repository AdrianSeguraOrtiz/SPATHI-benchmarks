"""Read-only prerequisite diagnostics for the article benchmark."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from .design import DesignError, load_design

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ExecutableResolver = Callable[[str], str | None]
DiskUsageReader = Callable[[Path], Any]

_GIB = 1024**3
_DISK_FAIL_BYTES = 5 * _GIB
_DISK_WARN_BYTES = 100 * _GIB

_IMPORT_PROBE = """
import importlib
import pathlib
import sys

repository = pathlib.Path(sys.argv[1]).resolve()
source = (repository / sys.argv[2]).resolve()
sys.path.insert(0, str(source))
module = importlib.import_module(sys.argv[3])
print(pathlib.Path(module.__file__).resolve())
""".strip()


class _Repository(NamedTuple):
    name: str
    path: Path
    source_subdirectory: str
    package: str


def _check(
    check_id: str,
    status: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": check_id,
        "status": status,
        "message": message,
    }
    if details:
        result["details"] = details
    if remediation:
        result["remediation"] = remediation
    return result


def _run(
    command: Sequence[str],
    *,
    runner: CommandRunner,
    timeout_seconds: float,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str] | Exception:
    try:
        return runner(
            [str(part) for part in command],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
        )
    except Exception as exc:  # Diagnostics report each unavailable dependency.
        return exc


def _failure_details(
    command: Sequence[str],
    outcome: subprocess.CompletedProcess[str] | Exception,
) -> dict[str, Any]:
    details: dict[str, Any] = {"command": [str(part) for part in command]}
    if isinstance(outcome, Exception):
        details["error"] = f"{type(outcome).__name__}: {outcome}"
        return details
    details["returncode"] = outcome.returncode
    if stdout := (outcome.stdout or "").strip():
        details["stdout"] = stdout
    if stderr := (outcome.stderr or "").strip():
        details["stderr"] = stderr
    return details


def _repository_check(repository: _Repository) -> dict[str, Any]:
    path = repository.path.resolve()
    package_root = path / repository.source_subdirectory / repository.package
    if not path.is_dir():
        return _check(
            f"{repository.name.lower()}_repository",
            "fail",
            f"{repository.name} repository is missing: {path}",
            details={"path": str(path)},
            remediation=f"Clone {repository.name} beside SPATHI-benchmarks or pass its path.",
        )
    if not package_root.is_dir():
        return _check(
            f"{repository.name.lower()}_repository",
            "fail",
            f"{path} does not contain the {repository.name} source package",
            details={"expected_package": str(package_root)},
            remediation=f"Use the root of the {repository.name} source checkout.",
        )
    return _check(
        f"{repository.name.lower()}_repository",
        "pass",
        f"{repository.name} source checkout is present",
        details={"path": str(path)},
    )


def _import_check(
    repository: _Repository,
    *,
    python_executable: str,
    repository_available: bool,
    runner: CommandRunner,
    timeout_seconds: float,
) -> dict[str, Any]:
    check_id = f"{repository.name.lower()}_import"
    if not repository_available:
        return _check(
            check_id,
            "fail",
            f"Cannot import {repository.name} because its source checkout is unavailable",
            remediation=f"Fix the {repository.name} repository path and rerun doctor.",
        )
    command = [
        python_executable,
        "-c",
        _IMPORT_PROBE,
        str(repository.path.resolve()),
        repository.source_subdirectory or ".",
        repository.package,
    ]
    outcome = _run(command, runner=runner, timeout_seconds=timeout_seconds)
    if isinstance(outcome, Exception) or outcome.returncode != 0:
        return _check(
            check_id,
            "fail",
            f"{repository.name} cannot be imported with the current interpreter",
            details=_failure_details(command, outcome),
            remediation=(
                f"Install {repository.name}'s dependencies in the active environment "
                "and keep using the source checkout."
            ),
        )
    reported = (outcome.stdout or "").strip().splitlines()
    imported = Path(reported[-1]).resolve() if reported else None
    expected = repository.path.resolve()
    if imported is None or not imported.is_relative_to(expected):
        return _check(
            check_id,
            "fail",
            f"{repository.name} resolved outside the requested source checkout",
            details={
                "expected_repository": str(expected),
                "imported_module": str(imported) if imported is not None else None,
            },
            remediation="Remove the conflicting installation or select the intended checkout.",
        )
    return _check(
        check_id,
        "pass",
        f"{repository.name} imports from the requested checkout",
        details={"module": str(imported)},
    )


def _docker_checks(
    *,
    resolver: ExecutableResolver,
    runner: CommandRunner,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    executable = resolver("docker")
    if executable is None:
        remediation = "Install Docker with Buildx and expose docker on PATH."
        return [
            _check("docker_cli", "fail", "Docker CLI is unavailable", remediation=remediation),
            _check(
                "docker_daemon",
                "fail",
                "Docker daemon could not be checked",
                remediation=remediation,
            ),
            _check(
                "docker_buildx",
                "fail",
                "Docker Buildx could not be checked",
                remediation=remediation,
            ),
        ]
    probes = (
        ("docker_cli", [executable, "--version"], "Docker CLI"),
        (
            "docker_daemon",
            [executable, "info", "--format", "{{.ServerVersion}}"],
            "Docker daemon",
        ),
        ("docker_buildx", [executable, "buildx", "version"], "Docker Buildx"),
    )
    checks = []
    for check_id, command, label in probes:
        outcome = _run(command, runner=runner, timeout_seconds=timeout_seconds)
        if not isinstance(outcome, Exception) and outcome.returncode == 0:
            checks.append(
                _check(
                    check_id,
                    "pass",
                    f"{label} is available",
                    details={"version": (outcome.stdout or "").strip()},
                )
            )
        else:
            checks.append(
                _check(
                    check_id,
                    "fail",
                    f"{label} is unavailable",
                    details=_failure_details(command, outcome),
                    remediation="Install/enable Docker and ensure this user can access the daemon.",
                )
            )
    return checks


def _design_check(repo_root: Path) -> dict[str, Any]:
    path = repo_root.resolve() / "experiments" / "article-study.json"
    try:
        design = load_design(path)
    except (DesignError, OSError, ValueError) as exc:
        return _check(
            "article_design",
            "fail",
            f"Article study design is invalid: {exc}",
            details={"path": str(path)},
            remediation="Fix experiments/article-study.json before starting a study.",
        )
    return _check(
        "article_design",
        "pass",
        "Article study design is valid",
        details={
            "path": str(path),
            "study_id": design.id,
            "fingerprint": design.fingerprint,
            "profiles": sorted(design.profiles),
        },
    )


def _catalog_contract_check(repo_root: Path, andrea_repo: Path) -> dict[str, Any]:
    """Check that frozen external routes still match the local ANDREA catalog."""

    design_path = repo_root.resolve() / "experiments" / "article-study.json"
    catalog_root = andrea_repo.resolve() / "andrea" / "catalog_inference_tools" / "tools"
    try:
        design = load_design(design_path)
    except (DesignError, OSError, ValueError) as exc:
        return _check(
            "andrea_catalog_contract",
            "fail",
            f"Cannot validate ANDREA routes because the article design is invalid: {exc}",
            details={"design": str(design_path), "catalog": str(catalog_root)},
            remediation="Fix experiments/article-study.json and rerun doctor.",
        )

    routes = tuple(route for route in design.tool_routes.values() if route.tool_id != "spathi")
    loaded: dict[str, tuple[dict[str, Any] | None, tuple[str, ...], Path]] = {}
    route_results: list[dict[str, Any]] = []

    for route in routes:
        toolspec_path = catalog_root / route.tool_id / "toolspec.json"
        if route.tool_id not in loaded:
            specification: dict[str, Any] | None = None
            specification_issues: list[str] = []
            if not toolspec_path.is_file():
                specification_issues.append(f"ToolSpec is missing: {toolspec_path}")
            else:
                try:
                    raw = json.loads(toolspec_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    specification_issues.append(
                        f"ToolSpec is not valid JSON: {type(exc).__name__}: {exc}"
                    )
                else:
                    if not isinstance(raw, dict):
                        specification_issues.append("ToolSpec JSON root must be an object")
                    else:
                        specification = raw
            loaded[route.tool_id] = (
                specification,
                tuple(specification_issues),
                toolspec_path,
            )

        toolspec, specification_issues, toolspec_path = loaded[route.tool_id]
        issues = list(specification_issues)
        if toolspec is not None:
            if toolspec.get("id") != route.tool_id:
                issues.append(f"ToolSpec id is {toolspec.get('id')!r}, expected {route.tool_id!r}")

            capabilities = toolspec.get("execution_capabilities")
            if not (
                isinstance(capabilities, list)
                and all(isinstance(value, str) and value for value in capabilities)
            ):
                issues.append("execution_capabilities must be a list of non-empty strings")
            elif route.delivery not in capabilities:
                issues.append(f"delivery {route.delivery!r} is not in execution_capabilities")

            parameters = toolspec.get("params")
            if not isinstance(parameters, dict):
                issues.append("params must be an object")
            else:
                undeclared = sorted(set(route.parameters) - set(parameters))
                if undeclared:
                    issues.append(
                        "route parameters are not declared in params: " + ", ".join(undeclared)
                    )
                if route.seed_parameter is not None and route.seed_parameter not in parameters:
                    issues.append(
                        f"seed_parameter {route.seed_parameter!r} is not declared in params"
                    )

            extra_inputs = toolspec.get("extra_inputs")
            declared_inputs: set[str] = set()
            if not isinstance(extra_inputs, dict):
                issues.append("extra_inputs must be an object")
            else:
                for field in ("required", "optional", "conditional_required"):
                    entries = extra_inputs.get(field, [])
                    if not isinstance(entries, list):
                        issues.append(f"extra_inputs.{field} must be a list")
                        continue
                    for index, entry in enumerate(entries):
                        if not isinstance(entry, dict) or not isinstance(entry.get("input"), str):
                            issues.append(
                                f"extra_inputs.{field}[{index}] must declare a string input"
                            )
                            continue
                        declared_inputs.add(entry["input"])
                missing_inputs = sorted(set(route.inputs) - {"expression"} - declared_inputs)
                if missing_inputs:
                    issues.append(
                        "route inputs are absent from extra_inputs: " + ", ".join(missing_inputs)
                    )

            threading = toolspec.get("runtime_resources")
            threading = threading.get("threading") if isinstance(threading, dict) else None
            if not isinstance(threading, dict) or "max_threads" not in threading:
                issues.append("runtime_resources.threading.max_threads is missing")
            else:
                catalog_max_threads = threading["max_threads"]
                valid_max_threads = catalog_max_threads is None or (
                    isinstance(catalog_max_threads, int)
                    and not isinstance(catalog_max_threads, bool)
                    and catalog_max_threads >= 1
                )
                if not valid_max_threads:
                    issues.append(
                        "runtime_resources.threading.max_threads must be null or a positive integer"
                    )
                elif route.max_threads != catalog_max_threads:
                    issues.append(
                        f"max_threads is {route.max_threads!r}, but the ToolSpec declares "
                        f"{catalog_max_threads!r}"
                    )

            outputs = toolspec.get("outputs")
            if not isinstance(outputs, dict):
                issues.append("outputs must be an object")
            else:
                directed = outputs.get("directed")
                sign = outputs.get("sign")
                if not isinstance(directed, bool):
                    issues.append("outputs.directed must be boolean")
                elif (
                    route.output_semantics in {"directed_tf_to_target", "directed_all_gene"}
                    and not directed
                ):
                    issues.append(
                        f"output_semantics {route.output_semantics!r} requires directed=true"
                    )
                elif route.output_semantics == "undirected_topology" and directed:
                    issues.append("output_semantics 'undirected_topology' requires directed=false")
                if not isinstance(sign, str) or not sign:
                    issues.append("outputs.sign must be a non-empty string")

        route_results.append(
            {
                "route_id": route.id,
                "tool_id": route.tool_id,
                "toolspec": str(toolspec_path),
                "status": "fail" if issues else "pass",
                "issues": issues,
            }
        )

    failed = [result for result in route_results if result["status"] == "fail"]
    details = {
        "design": str(design_path),
        "catalog": str(catalog_root),
        "external_routes_checked": len(route_results),
        "tools_checked": sorted(loaded),
        "routes": route_results,
    }
    if failed:
        return _check(
            "andrea_catalog_contract",
            "fail",
            f"ANDREA catalog drift affects {len(failed)} external article route(s)",
            details=details,
            remediation=(
                "Align experiments/article-study.json with the current ANDREA ToolSpecs "
                "before starting or resuming the study."
            ),
        )
    return _check(
        "andrea_catalog_contract",
        "pass",
        f"All {len(route_results)} external article routes match the ANDREA catalog",
        details=details,
    )


def _git_state(
    repositories: Sequence[tuple[str, Path]],
    *,
    runner: CommandRunner,
    timeout_seconds: float,
) -> dict[str, Any]:
    states: dict[str, Any] = {}
    dirty: list[str] = []
    unavailable: list[str] = []
    for name, repository in repositories:
        path = repository.resolve()
        head_command = ["git", "-C", str(path), "rev-parse", "HEAD"]
        status_command = [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ]
        head = _run(head_command, runner=runner, timeout_seconds=timeout_seconds)
        status = _run(status_command, runner=runner, timeout_seconds=timeout_seconds)
        if (
            isinstance(head, Exception)
            or head.returncode != 0
            or isinstance(status, Exception)
            or status.returncode != 0
        ):
            states[name] = {"available": False, "path": str(path)}
            unavailable.append(name)
            continue
        changes = [line for line in (status.stdout or "").splitlines() if line]
        states[name] = {
            "available": True,
            "path": str(path),
            "commit": (head.stdout or "").strip(),
            "dirty": bool(changes),
            "changed_paths": len(changes),
        }
        if changes:
            dirty.append(name)
    if dirty or unavailable:
        facts = []
        if dirty:
            facts.append("dirty: " + ", ".join(dirty))
        if unavailable:
            facts.append("Git metadata unavailable: " + ", ".join(unavailable))
        return _check(
            "source_provenance",
            "warn",
            "; ".join(facts),
            details={"repositories": states},
            remediation=(
                "Results remain runnable and record source identities; review mixed or "
                "uncommitted revisions before publication."
            ),
        )
    return _check(
        "source_provenance",
        "pass",
        "Source revisions are identifiable and worktrees are clean",
        details={"repositories": states},
    )


def _disk_check(repo_root: Path, reader: DiskUsageReader) -> dict[str, Any]:
    try:
        usage = reader(repo_root.resolve())
        total = int(usage.total)
        free = int(usage.free)
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        return _check(
            "disk_space",
            "fail",
            f"Cannot determine available disk space: {exc}",
            remediation="Mount a writable filesystem and rerun doctor.",
        )
    details = {
        "path": str(repo_root.resolve()),
        "total_bytes": total,
        "free_bytes": free,
        "free_gib": round(free / _GIB, 2),
    }
    if free < _DISK_FAIL_BYTES:
        return _check(
            "disk_space",
            "fail",
            "Less than 5 GiB of disk space is available",
            details=details,
            remediation="Free disk space before executing any benchmark profile.",
        )
    if free < _DISK_WARN_BYTES:
        return _check(
            "disk_space",
            "warn",
            "Disk space is sufficient for smoke but may be insufficient for reduced/full",
            details=details,
            remediation="Use a filesystem with at least 100 GiB free for article runs.",
        )
    return _check(
        "disk_space",
        "pass",
        "Disk capacity is suitable for benchmark execution",
        details=details,
    )


def write_doctor_report(report: dict[str, Any], path: Path) -> Path:
    """Write the explicitly requested diagnostic report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def run_doctor(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    python_executable: str | None = None,
    python_version: Sequence[int] | None = None,
    report_path: Path | None = None,
    runner: CommandRunner | None = None,
    resolver: ExecutableResolver | None = None,
    disk_usage: DiskUsageReader | None = None,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Inspect prerequisites without installing, building, or changing sources."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    root = Path(repo_root).resolve()
    current_python = python_executable or sys.executable
    version = tuple(python_version or sys.version_info[:3])
    command_runner = runner or subprocess.run
    executable_resolver = resolver or shutil.which
    disk_reader = disk_usage or shutil.disk_usage

    if version >= (3, 11):
        python_check = _check(
            "python_version",
            "pass",
            "Python version is supported",
            details={
                "detected": ".".join(map(str, version)),
                "minimum": "3.11",
                "executable": current_python,
            },
        )
    else:
        python_check = _check(
            "python_version",
            "fail",
            f"Python 3.11 or newer is required; found {'.'.join(map(str, version))}",
            remediation="Run SPATHI-benchmarks in a Python 3.11+ environment.",
        )

    repositories = (
        _Repository("ANDREA", Path(andrea_repo), "", "andrea"),
        _Repository("SPATHI", Path(spathi_repo), "src", "spathi"),
    )
    repository_checks = [_repository_check(repository) for repository in repositories]
    checks = [python_check, *repository_checks]
    checks.extend(
        _docker_checks(
            resolver=executable_resolver,
            runner=command_runner,
            timeout_seconds=timeout_seconds,
        )
    )
    for repository, available in zip(repositories, repository_checks, strict=True):
        checks.append(
            _import_check(
                repository,
                python_executable=current_python,
                repository_available=available["status"] == "pass",
                runner=command_runner,
                timeout_seconds=timeout_seconds,
            )
        )
    checks.extend(
        (
            _design_check(root),
            _catalog_contract_check(root, Path(andrea_repo)),
            _git_state(
                (
                    ("benchmark", root),
                    ("andrea", Path(andrea_repo)),
                    ("spathi", Path(spathi_repo)),
                ),
                runner=command_runner,
                timeout_seconds=timeout_seconds,
            ),
            _disk_check(root, disk_reader),
        )
    )

    counts = {
        status: sum(check["status"] == status for check in checks)
        for status in ("pass", "warn", "fail")
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "ok": counts["fail"] == 0,
        "summary": counts,
        "checks": checks,
    }
    if report_path is not None:
        write_doctor_report(report, report_path)
    return report
