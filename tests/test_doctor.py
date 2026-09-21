from __future__ import annotations

import json
import shutil
import subprocess
from collections import namedtuple
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.doctor import run_doctor, write_doctor_report

REPO_ROOT = Path(__file__).resolve().parents[1]
DiskUsage = namedtuple("DiskUsage", "total used free")


def _checkout(tmp_path: Path) -> tuple[Path, Path, Path]:
    benchmark = tmp_path / "SPATHI-benchmarks"
    andrea = tmp_path / "ANDREA"
    spathi = tmp_path / "SPATHI"
    (benchmark / "experiments").mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "experiments" / "article-study.json",
        benchmark / "experiments" / "article-study.json",
    )
    (andrea / "andrea").mkdir(parents=True)
    (andrea / "andrea" / "__init__.py").write_text("", encoding="utf-8")
    design = json.loads(
        (benchmark / "experiments" / "article-study.json").read_text(encoding="utf-8")
    )
    external_routes = [
        route for route in design["comparison"]["routes"] if route["tool_id"] != "spathi"
    ]
    for tool_id in sorted({route["tool_id"] for route in external_routes}):
        tool_routes = [route for route in external_routes if route["tool_id"] == tool_id]
        inputs = sorted(
            {
                input_id
                for route in tool_routes
                for input_id in route["inputs"]
                if input_id != "expression"
            }
        )
        parameter_ids = {
            parameter_id for route in tool_routes for parameter_id in route["parameters"]
        }
        parameter_ids.update(
            route["seed_parameter"] for route in tool_routes if route["seed_parameter"] is not None
        )
        directed = tool_routes[0]["output_semantics"] != "undirected_topology"
        toolspec = {
            "id": tool_id,
            "execution_capabilities": sorted({route["delivery"] for route in tool_routes}),
            "params": {parameter_id: {} for parameter_id in sorted(parameter_ids)},
            "extra_inputs": {
                "required": [],
                "optional": [
                    {"input": input_id, "usage": "test fixture", "delivery": "runtime"}
                    for input_id in inputs
                ],
                "conditional_required": [],
            },
            "outputs": {"directed": directed, "sign": "none"},
            "runtime_resources": {"threading": {"max_threads": tool_routes[0]["max_threads"]}},
        }
        path = andrea / "andrea" / "catalog_inference_tools" / "tools" / tool_id
        path.mkdir(parents=True)
        (path / "toolspec.json").write_text(
            json.dumps(toolspec, indent=2) + "\n",
            encoding="utf-8",
        )
    (spathi / "src" / "spathi").mkdir(parents=True)
    (spathi / "src" / "spathi" / "__init__.py").write_text("", encoding="utf-8")
    return benchmark, andrea, spathi


def _runner(
    andrea: Path,
    spathi: Path,
    *,
    dirty: str | None = None,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "/usr/bin/docker":
            return subprocess.CompletedProcess(command, 0, stdout="28.0.0\n", stderr="")
        if command[0] == "git":
            repository = Path(command[2])
            if command[3:] == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n", stderr="")
            if command[3:5] == ["status", "--porcelain"]:
                output = " M source.py\n" if dirty == repository.name else ""
                return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")
        if command[1] == "-c" and command[-1] in {"andrea", "spathi"}:
            package = command[-1]
            module = (
                andrea / "andrea" / "__init__.py"
                if package == "andrea"
                else spathi / "src" / "spathi" / "__init__.py"
            )
            return subprocess.CompletedProcess(command, 0, stdout=str(module) + "\n", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    return run


def _resolver(name: str) -> str | None:
    return "/usr/bin/docker" if name == "docker" else None


def _disk(free_gib: int = 500) -> Callable[[Path], DiskUsage]:
    def usage(_path: Path) -> DiskUsage:
        total = 1000 * 1024**3
        free = free_gib * 1024**3
        return DiskUsage(total, total - free, free)

    return usage


def _by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {check["id"]: check for check in report["checks"]}


def test_doctor_reports_a_healthy_environment(tmp_path: Path) -> None:
    benchmark, andrea, spathi = _checkout(tmp_path)

    report = run_doctor(
        repo_root=benchmark,
        andrea_repo=andrea,
        spathi_repo=spathi,
        python_executable="/venv/bin/python",
        python_version=(3, 13, 7),
        runner=_runner(andrea, spathi),
        resolver=_resolver,
        disk_usage=_disk(),
    )

    assert report["ok"] is True
    assert report["summary"] == {"pass": 12, "warn": 0, "fail": 0}
    checks = _by_id(report)
    assert checks["article_design"]["details"]["profiles"] == [
        "full",
        "reduced",
        "smoke",
    ]
    assert checks["andrea_import"]["details"]["module"].startswith(str(andrea))
    assert checks["spathi_import"]["details"]["module"].startswith(str(spathi))
    catalog = checks["andrea_catalog_contract"]
    assert catalog["details"]["external_routes_checked"] == 15
    assert all(route["status"] == "pass" for route in catalog["details"]["routes"])


def _toolspec(andrea: Path, tool_id: str) -> Path:
    return andrea / "andrea" / "catalog_inference_tools" / "tools" / tool_id / "toolspec.json"


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        ("missing", "ToolSpec is missing"),
        ("invalid_json", "ToolSpec is not valid JSON"),
        ("delivery", "is not in execution_capabilities"),
        ("parameter", "route parameters are not declared in params"),
        ("seed", "seed_parameter 'seed' is not declared in params"),
        ("input", "route inputs are absent from extra_inputs"),
        ("threads", "but the ToolSpec declares 4"),
        ("direction", "requires directed=true"),
    ],
)
def test_doctor_rejects_andrea_catalog_drift(
    tmp_path: Path,
    mutation: str,
    expected_issue: str,
) -> None:
    benchmark, andrea, spathi = _checkout(tmp_path)
    path = _toolspec(andrea, "genie3")
    if mutation == "missing":
        path.unlink()
    elif mutation == "invalid_json":
        path.write_text("{", encoding="utf-8")
    else:
        toolspec = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "delivery":
            toolspec["execution_capabilities"] = ["global"]
        elif mutation == "parameter":
            toolspec["params"] = {"seed": {}}
            design_path = benchmark / "experiments" / "article-study.json"
            design = json.loads(design_path.read_text(encoding="utf-8"))
            genie3 = next(
                route for route in design["comparison"]["routes"] if route["id"] == "genie3"
            )
            genie3["parameters"] = {"regressor_type": "RF"}
            design_path.write_text(json.dumps(design, indent=2) + "\n", encoding="utf-8")
        elif mutation == "seed":
            toolspec["params"] = {}
        elif mutation == "input":
            toolspec["extra_inputs"]["optional"] = []
        elif mutation == "threads":
            toolspec["runtime_resources"]["threading"]["max_threads"] = 4
        elif mutation == "direction":
            toolspec["outputs"]["directed"] = False
        path.write_text(json.dumps(toolspec, indent=2) + "\n", encoding="utf-8")

    report = run_doctor(
        repo_root=benchmark,
        andrea_repo=andrea,
        spathi_repo=spathi,
        runner=_runner(andrea, spathi),
        resolver=_resolver,
        disk_usage=_disk(),
    )

    check = _by_id(report)["andrea_catalog_contract"]
    assert report["ok"] is False
    assert check["status"] == "fail"
    affected = next(route for route in check["details"]["routes"] if route["route_id"] == "genie3")
    assert any(expected_issue in issue for issue in affected["issues"])


def test_dirty_checkout_is_warning_not_failure(tmp_path: Path) -> None:
    benchmark, andrea, spathi = _checkout(tmp_path)

    report = run_doctor(
        repo_root=benchmark,
        andrea_repo=andrea,
        spathi_repo=spathi,
        runner=_runner(andrea, spathi, dirty="SPATHI"),
        resolver=_resolver,
        disk_usage=_disk(),
    )

    check = _by_id(report)["source_provenance"]
    assert report["ok"] is True
    assert check["status"] == "warn"
    assert check["details"]["repositories"]["spathi"]["dirty"] is True


def test_doctor_reports_missing_docker_without_invoking_it(tmp_path: Path) -> None:
    benchmark, andrea, spathi = _checkout(tmp_path)

    report = run_doctor(
        repo_root=benchmark,
        andrea_repo=andrea,
        spathi_repo=spathi,
        runner=_runner(andrea, spathi),
        resolver=lambda _name: None,
        disk_usage=_disk(),
    )

    checks = _by_id(report)
    assert report["ok"] is False
    assert checks["docker_cli"]["status"] == "fail"
    assert checks["docker_daemon"]["status"] == "fail"
    assert checks["docker_buildx"]["status"] == "fail"


def test_doctor_fails_import_outside_requested_checkout(tmp_path: Path) -> None:
    benchmark, andrea, spathi = _checkout(tmp_path)
    installed = tmp_path / "site-packages" / "andrea" / "__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    base = _runner(andrea, spathi)

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        outcome = base(command, **kwargs)
        if command[0] == "/venv/bin/python" and command[-1] == "andrea":
            return subprocess.CompletedProcess(command, 0, stdout=str(installed), stderr="")
        return outcome

    report = run_doctor(
        repo_root=benchmark,
        andrea_repo=andrea,
        spathi_repo=spathi,
        python_executable="/venv/bin/python",
        runner=runner,
        resolver=_resolver,
        disk_usage=_disk(),
    )

    assert report["ok"] is False
    assert _by_id(report)["andrea_import"]["status"] == "fail"


def test_disk_budget_has_warning_and_failure_boundaries(tmp_path: Path) -> None:
    benchmark, andrea, spathi = _checkout(tmp_path)
    common = {
        "repo_root": benchmark,
        "andrea_repo": andrea,
        "spathi_repo": spathi,
        "runner": _runner(andrea, spathi),
        "resolver": _resolver,
    }

    warning = run_doctor(**common, disk_usage=_disk(20))
    failure = run_doctor(**common, disk_usage=_disk(2))

    assert warning["ok"] is True
    assert _by_id(warning)["disk_space"]["status"] == "warn"
    assert failure["ok"] is False
    assert _by_id(failure)["disk_space"]["status"] == "fail"


def test_write_doctor_report_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "report" / "doctor.json"
    report = {"schema_version": 1, "ok": True, "checks": []}

    assert write_doctor_report(report, path) == path
    assert json.loads(path.read_text(encoding="utf-8")) == report
