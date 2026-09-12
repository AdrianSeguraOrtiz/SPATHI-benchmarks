"""One command-line interface for planning, running and inspecting benchmark studies."""

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
from .qualification import run_qualification
from .study_workflow import (
    plan_study,
    rebuild_study_reports,
    run_study_workflow,
    study_status,
)


def _path(value: str) -> Path:
    # Preserve symlinks for the underlying provenance/path-boundary validators.
    return Path(os.path.abspath(value))


def discover_repository(start: Path | None = None) -> Path:
    """Find the checkout from the working directory, then the editable installation."""
    origin = start or Path.cwd()
    candidates = [origin, *origin.parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "repos.lock.json").is_file() and (
            candidate / "experiments" / "v1"
        ).is_dir():
            return candidate
    raise BenchmarkConfigError(
        "Cannot locate the SPATHI-benchmarks checkout. Run inside it or pass --project PATH."
    )


def _add_project(parser: argparse.ArgumentParser, *, repositories: bool = False) -> None:
    parser.add_argument("--project", type=_path, help="benchmark checkout (auto-detected)")
    if repositories:
        parser.add_argument("--andrea", type=_path, help="ANDREA checkout (default: ../ANDREA)")
        parser.add_argument("--spathi", type=_path, help="SPATHI checkout (default: ../SPATHI)")


def _add_resources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cores", type=int, help="optional CPU ceiling; default: effective machine capacity"
    )
    parser.add_argument(
        "--ram-gb", type=float, help="optional RAM ceiling in GiB; default: available memory"
    )


def _resources(args: argparse.Namespace) -> dict[str, int | float | None]:
    return {
        "max_cores": args.cores,
        "max_ram_gb": args.ram_gb,
        "reserve_cores": 0,
        "reserve_ram_gb": 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spathi-bench",
        description="Reproducible SPATHI studies, from calibration to final reports.",
        epilog="Start: spathi-bench run reduced --id pilot. Resume: spathi-bench run --id pilot.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start or resume a complete study")
    run.add_argument(
        "profile", nargs="?", choices=("smoke", "reduced", "full"),
        help="new study budget (default: reduced); omitted on resume to use the saved budget",
    )
    run.add_argument("--id", dest="run_id", help="study identity; existing IDs resume saved work")
    _add_project(run, repositories=True)
    _add_resources(run)

    status = commands.add_parser("status", help="show studies, or inspect one study")
    status.add_argument("run_id", nargs="?", help="study identity; omitted to list studies")
    _add_project(status)
    status.add_argument("--json", action="store_true", help="print structured evidence")

    report = commands.add_parser("report", help="rebuild reports from a study's saved evidence")
    report.add_argument("run_id", help="study identity")
    _add_project(report)

    plan = commands.add_parser("plan", help="inspect the phase and resource plan without execution")
    plan.add_argument("profile", nargs="?", choices=("smoke", "reduced", "full"), default="reduced")
    _add_project(plan, repositories=True)
    _add_resources(plan)
    plan.add_argument("--json", action="store_true", help="print the complete structured plan")

    doctor = commands.add_parser("doctor", help="check prerequisites without installing or running")
    _add_project(doctor, repositories=True)
    doctor.add_argument("--json", action="store_true", help="print all diagnostic details")

    qualify = commands.add_parser("qualify", help="verify local SPATHI/ANDREA scientific contracts")
    qualify.add_argument("--id", dest="run_id", help="qualification identity; existing IDs resume")
    _add_project(qualify, repositories=True)
    _add_resources(qualify)
    return parser


def _dispatch(args: argparse.Namespace) -> tuple[str, Any]:
    repo_root = args.project or discover_repository()
    common = {"repo_root": repo_root, "work_root": repo_root / "work"}
    if args.command == "status":
        return "Study status", study_status(**common, run_id=args.run_id)
    if args.command == "report":
        return "Study reports", rebuild_study_reports(**common, run_id=args.run_id)
    repositories = {
        "andrea_repo": args.andrea or repo_root.parent / "ANDREA",
        "spathi_repo": args.spathi or repo_root.parent / "SPATHI",
    }
    if args.command == "doctor":
        return "Environment diagnostics", run_doctor(
            repo_root=repo_root, **repositories, python_executable=sys.executable,
        )
    if args.command == "run":
        return "Study finished", run_study_workflow(
            **common, **repositories, profile=args.profile, run_id=args.run_id,
            **_resources(args),
        )
    if args.command == "plan":
        return "Study plan", plan_study(
            **common, **repositories, profile=args.profile, **_resources(args),
        )
    if args.command == "qualify":
        return "Qualification finished", run_qualification(
            **common, **repositories, run_id=args.run_id, **_resources(args),
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
    if "checks" in result:
        for check in result["checks"]:
            print(f"  {str(check['status']).upper():5}  {check['id']}: {check['message']}")
            if check.get("remediation") and check["status"] == "fail":
                print(f"         {check['remediation']}")
        return
    # Do not dump nested inference reports or every artifact to the terminal.
    for key in ("run_id", "study_id", "profile", "status", "current_phase", "error", "report", "report_path"):
        if result.get(key) is not None:
            print(f"  {key.replace('_', ' ')}: {result[key]}")
    runs = result.get("studies", result.get("runs"))
    if isinstance(runs, list):
        if not runs:
            print("  No studies yet. Start with: spathi-bench run reduced")
        for run in runs:
            print(
                f"  {run.get('run_id', run.get('study_id', '?'))}"
                f"  {run.get('profile', '')}  {run.get('status', '')}"
                f"  {run.get('current_phase') or ''}"
            )
    phases = result.get("phases")
    if isinstance(phases, Mapping):
        phases = [
            {"phase": name, **value} if isinstance(value, Mapping)
            else {"phase": name, "status": value}
            for name, value in phases.items()
        ]
    if isinstance(phases, list):
        for phase in phases:
            if isinstance(phase, Mapping):
                name = phase.get("phase", phase.get("name", phase.get("id", "?")))
                print(f"  {name}: {phase.get('status', phase.get('purpose', 'planned'))}")
    resources = result.get("resource_plan")
    if isinstance(resources, Mapping):
        capacity = resources.get("capacity", resources)
        ram = capacity.get("memory_gib")
        if ram is None and isinstance(capacity.get("memory_bytes"), (int, float)):
            ram = capacity["memory_bytes"] / 1024**3
        ram_label = f"{ram:.1f} GiB" if isinstance(ram, (int, float)) else "unspecified RAM"
        print(f"  capacity: {capacity.get('cpu_threads', '?')} CPU threads, {ram_label}")
    for warning in result.get("warnings", []):
        print(f"  warning: {warning}")
    for key in ("estimate", "resume_command"):
        if result.get(key):
            print(f"  {key.replace('_', ' ')}: {result[key]}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        title, result = _dispatch(args)
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed artifacts are retained; resume with the same study ID.",
            file=sys.stderr,
        )
        return 130
    except (BenchmarkConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _render(title, result, structured=getattr(args, "json", False))
    if args.command == "doctor" and isinstance(result, dict) and not result.get("ok", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
