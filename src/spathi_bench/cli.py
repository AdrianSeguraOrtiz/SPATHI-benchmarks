"""Small public CLI for the single article-study workflow."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .config import BenchmarkConfigError
from .doctor import run_doctor
from .study_workflow import (
    plan_study,
    rebuild_study_reports,
    run_study_workflow,
    study_status,
)


def _path(value: str) -> Path:
    return Path(os.path.abspath(value))


def discover_repository(start: Path | None = None) -> Path:
    """Locate a checkout by its one canonical article-design document."""

    origin = (start or Path.cwd()).absolute()
    candidates = [origin, *origin.parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "experiments/article-study.json").is_file() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    raise BenchmarkConfigError(
        "Cannot locate SPATHI-benchmarks; run inside its checkout or pass --project"
    )


def _add_project(parser: argparse.ArgumentParser, *, repositories: bool = False) -> None:
    parser.add_argument("--project", type=_path, help="benchmark checkout (auto-detected)")
    if repositories:
        parser.add_argument("--andrea", type=_path, help="ANDREA checkout (default: ../ANDREA)")
        parser.add_argument("--spathi", type=_path, help="SPATHI checkout (default: ../SPATHI)")


def _add_resources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-cpus",
        type=int,
        help="maximum schedulable CPU threads (default: detected effective capacity)",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        help="maximum schedulable RAM in GiB (default: detected effective capacity)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spathi-bench",
        description="Run the reproducible three-phase SPATHI article study.",
        epilog=(
            "Start: spathi-bench run --profile reduced --id reduced-1. "
            "Resume: spathi-bench run --id reduced-1."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start or resume a complete study")
    run.add_argument("--profile", choices=("smoke", "reduced", "full"))
    run.add_argument("--id", dest="study_id", required=True, help="stable study identity")
    _add_project(run, repositories=True)
    _add_resources(run)

    status = commands.add_parser("status", help="list studies or inspect one")
    status.add_argument("--id", dest="study_id", help="study identity; omit to list all")
    status.add_argument("--json", action="store_true", help="print structured evidence")
    _add_project(status)

    plan = commands.add_parser("plan", help="inspect the exact DAG without executing it")
    plan.add_argument("--profile", choices=("smoke", "reduced", "full"), required=True)
    plan.add_argument("--json", action="store_true", help="print every planned task")
    _add_project(plan, repositories=True)
    _add_resources(plan)

    report = commands.add_parser("report", help="rebuild portable reports from saved evidence")
    report.add_argument("--id", dest="study_id", required=True, help="study identity")
    _add_project(report)

    doctor = commands.add_parser("doctor", help="check prerequisites without changing anything")
    doctor.add_argument("--json", action="store_true", help="print structured checks")
    _add_project(doctor, repositories=True)
    return parser


def _repositories(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    return {
        "andrea_repo": getattr(args, "andrea", None) or root.parent / "ANDREA",
        "spathi_repo": getattr(args, "spathi", None) or root.parent / "SPATHI",
    }


def _dispatch(args: argparse.Namespace) -> tuple[str, Any]:
    root = args.project or discover_repository()
    work_root = root / "work"
    if args.command == "status":
        return "Study status", study_status(work_root=work_root, study_id=args.study_id)
    if args.command == "report":
        return "Study reports", rebuild_study_reports(
            repo_root=root, work_root=work_root, study_id=args.study_id
        )
    if args.command == "plan":
        repositories = _repositories(args, root)
        return "Study plan", plan_study(
            repo_root=root,
            work_root=work_root,
            profile=args.profile,
            andrea_repo=repositories["andrea_repo"],
            max_cpus=args.max_cpus,
            max_memory_gb=args.max_memory_gb,
        )
    repositories = _repositories(args, root)
    if args.command == "doctor":
        return "Environment diagnostics", run_doctor(repo_root=root, **repositories)
    if args.command == "run":
        return "Study completed", run_study_workflow(
            repo_root=root,
            work_root=work_root,
            profile=args.profile,
            study_id=args.study_id,
            max_cpus=args.max_cpus,
            max_memory_gb=args.max_memory_gb,
            **repositories,
        )
    raise AssertionError(f"Unhandled command: {args.command}")


def _render(title: str, result: Any, *, structured: bool = False) -> None:
    if structured:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return
    print(title)
    if not isinstance(result, Mapping):
        print(result)
        return
    checks = result.get("checks")
    if isinstance(checks, list):
        for check in checks:
            print(f"  {str(check['status']).upper():4}  {check['id']}: {check['message']}")
            if check.get("status") == "fail" and check.get("remediation"):
                print(f"        {check['remediation']}")
        return
    for key in ("study_id", "profile", "status", "current_phase", "report"):
        if result.get(key) is not None:
            print(f"  {key.replace('_', ' ')}: {result[key]}")
    studies = result.get("studies")
    if isinstance(studies, list):
        if not studies:
            print("  No studies yet.")
        for study in studies:
            print(
                f"  {study.get('study_id', '?')}  {study.get('profile', '')}  "
                f"{study.get('status', '')}  {study.get('current_phase') or ''}"
            )
    phases = result.get("phases")
    if isinstance(phases, Mapping):
        for phase, value in phases.items():
            if isinstance(value, Mapping):
                detail = value.get("status", f"{value.get('task_count', 0)} tasks")
            else:
                detail = value
            print(f"  {phase}: {detail}")
    resource_plan = result.get("resource_plan")
    if isinstance(resource_plan, Mapping) and isinstance(resource_plan.get("capacity"), Mapping):
        capacity = resource_plan["capacity"]
        print(
            f"  capacity: {capacity['cpu_threads']} CPU threads, "
            f"{capacity['memory_bytes'] / 1024**3:.1f} GiB RAM"
        )
    for warning in result.get("warnings", []):
        print(f"  warning: {warning}")
    if result.get("resume_command"):
        print(f"  resume: {result['resume_command']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        title, result = _dispatch(args)
    except KeyboardInterrupt:
        print("Interrupted; resume with the same study ID.", file=sys.stderr)
        return 130
    except (BenchmarkConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _render(title, result, structured=getattr(args, "json", False))
    if args.command == "doctor" and not result.get("ok", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
