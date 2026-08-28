"""Command-line entry point for SPATHI benchmark stages."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import BenchmarkConfigError
from .summarize import summarize
from .workflow import (
    build_spathi_image,
    evaluate,
    generate_data,
    preflight,
    run_inference,
    validate_handoffs,
)


def _path(value: str) -> Path:
    return Path(value).resolve()


def _add_common(parser: argparse.ArgumentParser, *, repositories: bool = False) -> None:
    parser.add_argument("--repo-root", type=_path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-root", type=_path, required=True)
    if repositories:
        parser.add_argument("--andrea-repo", type=_path, required=True)
    if repositories and parser.prog.endswith("preflight"):
        parser.add_argument("--spathi-repo", type=_path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spathi-bench", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight", help="validate locks and plan generation")
    _add_common(preflight_parser, repositories=True)

    build_parser = subparsers.add_parser("build-image", help="build the pinned SPATHI image")
    _add_common(build_parser)
    build_parser.add_argument("--spathi-repo", type=_path, required=True)

    generate_parser = subparsers.add_parser("generate", help="run ANDREA generate-data")
    _add_common(generate_parser, repositories=True)

    infer_parser = subparsers.add_parser("infer", help="run the hybrid ANDREA inference")
    _add_common(infer_parser, repositories=True)

    validate_parser = subparsers.add_parser("validate", help="validate evaluation handoffs")
    _add_common(validate_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="run ANDREA evaluate-inference")
    _add_common(evaluate_parser, repositories=True)

    summarize_parser = subparsers.add_parser("summarize", help="combine evaluation metrics")
    _add_common(summarize_parser)
    return parser


def _dispatch(args: argparse.Namespace) -> tuple[str, Path | tuple[Path, Path]]:
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
        path = validate_handoffs(profile_id=args.profile, run_root=args.run_root)
        return "Handoff validation passed", path
    if args.command == "evaluate":
        return "Evaluation completed", evaluate(**common, andrea_repo=args.andrea_repo)
    if args.command == "summarize":
        paths = summarize(profile_id=args.profile, run_root=args.run_root)
        return "Reports written", paths
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        message, result = _dispatch(args)
    except (BenchmarkConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if isinstance(result, tuple):
        rendered = ", ".join(str(path) for path in result)
    else:
        rendered = str(result)
    print(f"{message}: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
