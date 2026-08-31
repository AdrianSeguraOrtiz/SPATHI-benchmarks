"""Command-line entry point for SPATHI benchmark stages."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .artifacts import sha256_file
from .calibration import (
    CandidateConfiguration,
    build_andrea_requests,
    build_calibration_ranking,
    build_frozen_selected_config,
    candidate_from_dict,
    candidate_to_dict,
    expand_stage_candidates,
    expand_stage_runs,
    expand_validation_runs,
    load_search_space,
    locked_spathi_identity,
)
from .config import BenchmarkConfigError, load_image_lock, load_json_object, validate_json
from .doctor import run_doctor
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
from .state import run_lock
from .statistics import analyze_statistics
from .summarize import summarize


def _path(value: str) -> Path:
    return Path(value).resolve()


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

    calibration_plan = subparsers.add_parser(
        "calibration-plan",
        help="materialize one bounded SPATHI calibration stage as ANDREA requests",
    )
    calibration_plan.add_argument("--repo-root", type=_path, required=True)
    calibration_plan.add_argument(
        "--stage",
        choices=("convergence", "structure", "estimator", "validation"),
        required=True,
    )
    calibration_plan.add_argument("--parents", type=_path)
    calibration_plan.add_argument("--output-dir", type=_path, required=True)

    calibration_rank = subparsers.add_parser(
        "calibration-rank",
        help="rank a complete calibration stage without accessing test data",
    )
    calibration_rank.add_argument("--repo-root", type=_path, required=True)
    calibration_rank.add_argument(
        "--stage",
        choices=("convergence", "structure", "estimator", "validation"),
        required=True,
    )
    calibration_rank.add_argument("--manifest", type=_path, required=True)
    calibration_rank.add_argument("--summary", type=_path, required=True)
    calibration_rank.add_argument("--output", type=_path, required=True)

    calibration_freeze = subparsers.add_parser(
        "calibration-freeze",
        help="freeze selected-config from a reconciled validation ranking",
    )
    calibration_freeze.add_argument("--repo-root", type=_path, required=True)
    calibration_freeze.add_argument("--ranking", type=_path, required=True)
    calibration_freeze.add_argument("--output", type=_path, required=True)

    preflight_parser = subparsers.add_parser("preflight", help="validate locks and plan generation")
    _add_common(
        preflight_parser,
        andrea_repository=True,
        spathi_repository=True,
    )

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
        help="rank predeclared tools with paired non-parametric tests",
    )
    _add_common(statistics_parser)
    return parser


def _serialized_candidates(path: Path | None) -> tuple[CandidateConfiguration, ...]:
    if path is None:
        return ()
    payload = load_json_object(path, "calibration parents")
    raw = payload.get("selected_candidates")
    if not isinstance(raw, list) or not raw:
        raise BenchmarkConfigError("Calibration parents must contain selected_candidates")
    return tuple(candidate_from_dict(item) for item in raw)


def _write_calibration_plan(args: argparse.Namespace) -> Path:
    search = load_search_space(args.repo_root)
    spathi_identity = locked_spathi_identity(args.repo_root)
    parents = _serialized_candidates(args.parents)
    if args.stage == "validation":
        if not parents:
            raise BenchmarkConfigError("Validation planning requires --parents from estimator")
        candidates = parents
        runs = expand_validation_runs(search, candidates)
        dataset_role = "validation"
    else:
        candidates = expand_stage_candidates(search, args.stage, parents=parents)
        runs = expand_stage_runs(search, args.stage, parents=parents)
        dataset_role = "development"
    image_tag = load_image_lock(args.repo_root)["images"]["spathi"]["tag"]
    tools, custom, metadata = build_andrea_requests(runs, docker_image=image_tag)
    output_dir: Path = args.output_dir
    if output_dir.exists():
        raise BenchmarkConfigError(f"Calibration plan output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        payloads = {
            "tools-params.json": tools,
            "custom-tools.json": custom,
            "run-metadata.json": metadata,
            "calibration-manifest.json": {
                "schema_version": "1.0",
                "search_space_id": search["id"],
                "stage_id": args.stage,
                "dataset_role": dataset_role,
                **spathi_identity,
                "candidates": [candidate_to_dict(candidate) for candidate in candidates],
                "run_ids": [run.run_id for run in runs],
            },
        }
        for filename, payload in payloads.items():
            (temporary / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir


def _write_calibration_ranking(args: argparse.Namespace) -> Path:
    search = load_search_space(args.repo_root)
    manifest = load_json_object(args.manifest, "calibration manifest")
    validate_json(
        manifest,
        args.repo_root / "schemas" / "calibration-manifest.schema.json",
        "calibration manifest",
    )
    if manifest.get("search_space_id") != search["id"] or manifest.get("stage_id") != args.stage:
        raise BenchmarkConfigError("Calibration manifest identity differs from the requested stage")
    spathi_identity = locked_spathi_identity(args.repo_root)
    if any(manifest.get(field) != value for field, value in spathi_identity.items()):
        raise BenchmarkConfigError("Calibration manifest was generated for another SPATHI build")
    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise BenchmarkConfigError("Calibration manifest contains no candidates")
    candidates = tuple(candidate_from_dict(item) for item in raw_candidates)
    summary = load_json_object(args.summary, "benchmark summary")
    rows = summary.get("calibration_rows")
    if not isinstance(rows, list):
        raise BenchmarkConfigError("Benchmark summary contains no calibration_rows")
    output: Path = args.output
    if output.exists():
        raise BenchmarkConfigError(f"Calibration ranking output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_calibration_ranking(
        search,
        args.stage,
        candidates,
        rows,
        search_space_sha256=sha256_file(
            args.repo_root / "experiments" / "v1" / "spathi" / "search-space.json"
        ),
        calibration_manifest_sha256=sha256_file(args.manifest),
        benchmark_summary_sha256=sha256_file(args.summary),
        **spathi_identity,
    )
    validate_json(
        payload,
        args.repo_root / "schemas" / "calibration-ranking.schema.json",
        "calibration ranking",
    )
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def _write_frozen_selection(args: argparse.Namespace) -> Path:
    output: Path = args.output
    if output.exists():
        existing = load_json_object(output, "existing selected configuration")
        if existing.get("status") != "unselected" or existing.get("phase") != "planned":
            raise BenchmarkConfigError(
                f"Refusing to replace a non-planned selected configuration: {output}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_frozen_selected_config(args.repo_root, args.ranking)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


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
    if args.command == "calibration-plan":
        return "Calibration plan written", _write_calibration_plan(args)
    if args.command == "calibration-rank":
        return "Calibration ranking written", _write_calibration_ranking(args)
    if args.command == "calibration-freeze":
        return "SPATHI selection frozen", _write_frozen_selection(args)
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
            "calibration-plan",
            "calibration-rank",
            "calibration-freeze",
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
    if isinstance(result, dict) and not result.get("ok", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
