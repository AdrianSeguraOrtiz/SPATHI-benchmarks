"""Small subprocess bridge for locked ANDREA output contracts.

The benchmark may use a lightweight orchestration virtualenv while ANDREA runs
in a different interpreter selected by ``ANDREA_PYTHON``.  This module is
therefore invoked in that exact interpreter instead of importing an arbitrary
editable ANDREA installation in the parent process.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .andrea_contracts import resolve_analysis_bundle, validate_inference_analysis_contract
from .export import export_analysis_resolution


def _path(value: str) -> Path:
    return Path(value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spathi-bench-andrea-bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-inference")
    validate.add_argument("--andrea-repo", type=_path, required=True)
    validate.add_argument("--run-report", type=_path, required=True)

    export = subparsers.add_parser("export-analysis")
    export.add_argument("--andrea-repo", type=_path, required=True)
    export.add_argument(
        "--producer-command",
        choices=("generate-data", "infer-network", "evaluate-inference"),
        required=True,
    )
    export.add_argument("--root", type=_path, required=True)
    export.add_argument("--dataset-id")
    export.add_argument("--output", type=_path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-inference":
        validate_inference_analysis_contract(args.andrea_repo, args.run_report)
        print(json.dumps({"status": "passed", "run_report": str(args.run_report)}))
        return 0

    resolution = resolve_analysis_bundle(
        args.andrea_repo,
        command=args.producer_command,
        root=args.root,
        dataset_id=args.dataset_id,
    )
    result = export_analysis_resolution(
        resolution,
        args.output,
        producer_command=args.producer_command,
        dataset_id=args.dataset_id,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "zip": str(result.zip_path),
                "zip_sha256": result.zip_sha256,
                "manifest": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
