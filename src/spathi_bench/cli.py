"""Command-line entry point for SPATHI benchmark stages."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import BenchmarkConfigError
from .dimensional_transfer_workflow import run_dimensional_transfer_workflow
from .doctor import run_doctor
from .efficiency_workflow import run_efficiency_workflow
from .pipeline import (
    build_spathi_image,
    evaluate,
    export_analysis,
    generate_data,
    integrate_inference,
    preflight,
    run_inference,
    validate_handoffs,
)
from .reproducibility import write_reproducibility_audit
from .scalability_workflow import run_scalability_workflow
from .state import run_lock
from .statistics import analyze_statistics
from .summarize import summarize
from .workflows import (
    rebuild_calibration_report,
    rebuild_comparison_report,
    run_calibration_workflow,
    run_comparison_workflow,
)


def _path(value: str) -> Path:
    return Path(os.path.abspath(value))


def _add_common(
    parser: argparse.ArgumentParser,
    *,
    andrea_repository: bool = False,
    spathi_repository: bool = False,
) -> None:
    parser.add_argument("--repo-root", type=_path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-root", type=_path, required=True)
    if andrea_repository:
        parser.add_argument("--andrea-repo", type=_path, required=True)
    if spathi_repository:
        parser.add_argument("--spathi-repo", type=_path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spathi-bench", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="diagnose benchmark prerequisites")
    doctor_parser.add_argument("--repo-root", type=_path, required=True)
    doctor_parser.add_argument("--andrea-repo", type=_path, required=True)
    doctor_parser.add_argument("--spathi-repo", type=_path, required=True)
    doctor_parser.add_argument(
        "--python", default=sys.executable, help="benchmark Python executable"
    )
    doctor_parser.add_argument("--output", type=_path)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="run or resume the bounded six-stage SPATHI calibration",
    )
    calibrate_parser.add_argument("--repo-root", type=_path, required=True)
    calibrate_parser.add_argument("--andrea-repo", type=_path, required=True)
    calibrate_parser.add_argument("--spathi-repo", type=_path, required=True)
    calibrate_parser.add_argument("--work-root", type=_path)
    calibrate_parser.add_argument("--profile", choices=("reduced", "full"), default="full")
    calibrate_parser.add_argument("--run-id")
    calibrate_parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild HTML from one completed calibration without running experiments",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="run or resume the selected SPATHI comparison with catalog baselines",
    )
    compare_parser.add_argument("--repo-root", type=_path, required=True)
    compare_parser.add_argument("--andrea-repo", type=_path, required=True)
    compare_parser.add_argument("--spathi-repo", type=_path, required=True)
    compare_parser.add_argument("--work-root", type=_path)
    compare_parser.add_argument("--profile", choices=("reduced", "full"), default="full")
    compare_parser.add_argument("--efficiency-run-id")
    compare_parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild HTML from one completed comparison without running experiments",
    )

    efficiency = subparsers.add_parser(
        "justify-efficiency",
        help="run or resume the independent SPATHI efficiency study",
    )
    efficiency.add_argument("--repo-root", type=_path, required=True)
    efficiency.add_argument("--andrea-repo", type=_path, required=True)
    efficiency.add_argument("--spathi-repo", type=_path, required=True)
    efficiency.add_argument("--work-root", type=_path)
    efficiency.add_argument("--profile", choices=("reduced", "full"), default="full")
    efficiency.add_argument("--calibration-run-id")

    scalability = subparsers.add_parser(
        "scalability",
        help="measure the frozen SPATHI policy across a public synthetic scale grid",
    )
    scalability.add_argument("--repo-root", type=_path, required=True)
    scalability.add_argument("--andrea-repo", type=_path, required=True)
    scalability.add_argument("--spathi-repo", type=_path, required=True)
    scalability.add_argument("--work-root", type=_path)
    scalability.add_argument("--profile", choices=("reduced", "full"), default="full")
    scalability.add_argument("--efficiency-run-id")

    dimensional_transfer = subparsers.add_parser(
        "dimensional-transfer",
        help="measure post-freeze SPATHI accuracy transfer along separate scale axes",
    )
    dimensional_transfer.add_argument("--repo-root", type=_path, required=True)
    dimensional_transfer.add_argument("--andrea-repo", type=_path, required=True)
    dimensional_transfer.add_argument("--spathi-repo", type=_path, required=True)
    dimensional_transfer.add_argument("--work-root", type=_path)
    dimensional_transfer.add_argument("--profile", choices=("reduced", "full"), default="full")
    dimensional_transfer.add_argument("--efficiency-run-id")

    audit = subparsers.add_parser(
        "audit-reproducibility",
        help="verify every committed stochastic seed assignment",
    )
    audit.add_argument("--repo-root", type=_path, required=True)
    audit.add_argument("--output", type=_path, required=True)

    preflight_parser = subparsers.add_parser("preflight", help="validate locks and plan generation")
    _add_common(
        preflight_parser,
        andrea_repository=True,
        spathi_repository=True,
    )
    preflight_parser.add_argument("--profile-file", type=_path)

    build_parser = subparsers.add_parser("build-image", help="build the pinned SPATHI image")
    _add_common(build_parser)
    build_parser.add_argument("--spathi-repo", type=_path, required=True)

    generate_parser = subparsers.add_parser("generate", help="run ANDREA generate-data")
    _add_common(generate_parser, andrea_repository=True)

    infer_parser = subparsers.add_parser(
        "infer", help="run independently cached ANDREA inference tasks"
    )
    _add_common(infer_parser, andrea_repository=True)

    validate_parser = subparsers.add_parser("validate", help="validate evaluation handoffs")
    _add_common(validate_parser)

    integrate_parser = subparsers.add_parser(
        "integrate", help="build joint ANDREA inference views by dataset and regime"
    )
    _add_common(integrate_parser, andrea_repository=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="run ANDREA evaluate-inference")
    _add_common(evaluate_parser, andrea_repository=True)

    export_parser = subparsers.add_parser(
        "export", help="write official ANDREA analysis ZIPs for every handoff"
    )
    _add_common(export_parser, andrea_repository=True)

    summarize_parser = subparsers.add_parser("summarize", help="combine evaluation metrics")
    _add_common(summarize_parser)

    statistics_parser = subparsers.add_parser(
        "statistics",
        help=("run predeclared paired analyses; inferential tests only when authorized"),
    )
    _add_common(statistics_parser)
    return parser


def _dispatch(
    args: argparse.Namespace,
) -> tuple[str, Path | tuple[Path, Path] | dict[str, object]]:
    if args.command == "doctor":
        report = run_doctor(
            andrea_repo=args.andrea_repo,
            spathi_repo=args.spathi_repo,
            repo_root=args.repo_root,
            python_executable=args.python,
            report_path=args.output,
        )
        return "Environment diagnostics", report
    if args.command == "calibrate":
        if args.report_only:
            return "SPATHI calibration report rebuilt", rebuild_calibration_report(
                repo_root=args.repo_root,
                work_root=args.work_root or args.repo_root / "work",
                calibration_budget=args.profile,
                run_id=args.run_id,
            )
        return "SPATHI calibration completed", run_calibration_workflow(
            repo_root=args.repo_root,
            andrea_repo=args.andrea_repo,
            spathi_repo=args.spathi_repo,
            work_root=args.work_root or args.repo_root / "work",
            calibration_budget=args.profile,
            run_id=args.run_id,
        )
    if args.command == "compare":
        if args.report_only:
            return "SPATHI comparison report rebuilt", rebuild_comparison_report(
                repo_root=args.repo_root,
                work_root=args.work_root or args.repo_root / "work",
                profile=args.profile,
                efficiency_run_id=args.efficiency_run_id,
            )
        return "SPATHI comparison completed", run_comparison_workflow(
            repo_root=args.repo_root,
            andrea_repo=args.andrea_repo,
            spathi_repo=args.spathi_repo,
            work_root=args.work_root or args.repo_root / "work",
            profile=args.profile,
            efficiency_run_id=args.efficiency_run_id,
        )
    if args.command == "justify-efficiency":
        return "SPATHI efficiency study completed", run_efficiency_workflow(
            repo_root=args.repo_root,
            andrea_repo=args.andrea_repo,
            spathi_repo=args.spathi_repo,
            work_root=args.work_root or args.repo_root / "work",
            profile=args.profile,
            calibration_run_id=args.calibration_run_id,
        )
    if args.command == "scalability":
        return "SPATHI scalability study completed", run_scalability_workflow(
            repo_root=args.repo_root,
            andrea_repo=args.andrea_repo,
            spathi_repo=args.spathi_repo,
            work_root=args.work_root or args.repo_root / "work",
            profile=args.profile,
            efficiency_run_id=args.efficiency_run_id,
        )
    if args.command == "dimensional-transfer":
        return "SPATHI dimensional transfer completed", run_dimensional_transfer_workflow(
            repo_root=args.repo_root,
            andrea_repo=args.andrea_repo,
            spathi_repo=args.spathi_repo,
            work_root=args.work_root or args.repo_root / "work",
            profile=args.profile,
            efficiency_run_id=args.efficiency_run_id,
        )
    if args.command == "audit-reproducibility":
        return "Reproducibility audit written", write_reproducibility_audit(
            args.repo_root,
            args.output,
        )
    common = {
        "repo_root": args.repo_root,
        "profile_id": args.profile,
        "run_root": args.run_root,
    }
    if args.command == "preflight":
        path = preflight(
            **common,
            andrea_repo=args.andrea_repo,
            spathi_repo=args.spathi_repo,
            profile_file=args.profile_file,
        )
        return "Preflight passed", path
    if args.command == "build-image":
        return "SPATHI image built", build_spathi_image(
            **common,
            spathi_repo=args.spathi_repo,
        )
    if args.command == "generate":
        return "Data generated", generate_data(**common, andrea_repo=args.andrea_repo)
    if args.command == "infer":
        return "Inference completed", run_inference(**common, andrea_repo=args.andrea_repo)
    if args.command == "validate":
        path = validate_handoffs(
            repo_root=args.repo_root,
            profile_id=args.profile,
            run_root=args.run_root,
        )
        return "Handoff validation passed", path
    if args.command == "integrate":
        return "Joint inference views built", integrate_inference(
            **common,
            andrea_repo=args.andrea_repo,
        )
    if args.command == "evaluate":
        return "Evaluation completed", evaluate(**common, andrea_repo=args.andrea_repo)
    if args.command == "export":
        return "ANDREA analysis bundles exported", export_analysis(
            **common,
            andrea_repo=args.andrea_repo,
        )
    if args.command == "summarize":
        paths = summarize(
            repo_root=args.repo_root,
            profile_id=args.profile,
            run_root=args.run_root,
        )
        return "Reports written", paths
    if args.command == "statistics":
        paths = analyze_statistics(
            repo_root=args.repo_root,
            profile_id=args.profile,
            run_root=args.run_root,
        )
        return "Statistical reports written", paths
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in {
            "doctor",
            "calibrate",
            "compare",
            "justify-efficiency",
            "scalability",
            "dimensional-transfer",
            "audit-reproducibility",
        }:
            message, result = _dispatch(args)
        else:
            with run_lock(args.run_root):
                message, result = _dispatch(args)
    except (BenchmarkConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if isinstance(result, dict):
        rendered = json.dumps(result, indent=2, sort_keys=True)
    elif isinstance(result, tuple):
        rendered = ", ".join(str(path) for path in result)
    else:
        rendered = str(result)
    print(f"{message}: {rendered}")
    if args.command == "doctor" and isinstance(result, dict) and not result.get("ok", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
