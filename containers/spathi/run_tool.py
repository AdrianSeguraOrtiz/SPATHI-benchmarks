"""Adapt SPATHI to ANDREA's standardized external inference-tool contract."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

EXPECTED_NETWORK_COLUMNS = ("source", "target", "score", "sign", "evidence", "context")
_PHASE_PERCENT = {
    "validating_inputs": 10,
    "building_representation": 15,
    "computing_distances": 25,
    "preparing_inference": 30,
    "preparing_group": 35,
    "model_inference": 35,
    "writing_outputs": 92,
    "building_report": 94,
    "publishing": 96,
    "complete": 98,
}
ALLOWED_PARAMS = {
    "bandwidth",
    "bootstrap",
    "distance_metric",
    "distance_space",
    "distance_standardization",
    "group_size_correction",
    "kernel",
    "max_depth",
    "max_features",
    "min_samples_leaf",
    "n_components",
    "n_estimators",
    "pca_svd_solver",
    "random_seed",
    "tree_method",
    "weight_mode",
}
REQUIRED_PARAMS = {
    "bandwidth",
    "distance_metric",
    "distance_space",
    "distance_standardization",
    "group_size_correction",
    "kernel",
    "max_depth",
    "max_features",
    "min_samples_leaf",
    "n_components",
    "n_estimators",
    "pca_svd_solver",
    "bootstrap",
    "random_seed",
    "tree_method",
    "weight_mode",
}


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--extra", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=_positive_int)
    return parser


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _write_progress(
    output_dir: Path,
    *,
    percent: int,
    status: str,
    phase: str,
    message: str,
    warnings: Sequence[str] = (),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "progress.json"
    temporary = output_dir / ".progress.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "percent": percent,
                "status": status,
                "phase": phase,
                "message": message,
                "warnings": list(warnings),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _progress_callback(output_dir: Path) -> Callable[[Any], None]:
    """Translate structured SPATHI progress into ANDREA's progress contract."""

    last_percent = 5
    last_signature: tuple[int, str] | None = None

    def report(event: Any) -> None:
        nonlocal last_percent, last_signature
        phase = str(event.phase)
        candidate = _PHASE_PERCENT.get(phase, last_percent)
        if phase == "model_inference" and event.model_fraction is not None:
            candidate = 35 + int(55 * float(event.model_fraction))
        percent = max(last_percent, min(98, candidate))
        signature = (percent, phase)
        if signature == last_signature:
            return
        _write_progress(
            output_dir,
            percent=percent,
            status="running",
            phase=phase,
            message=str(event.message),
        )
        last_percent = percent
        last_signature = signature

    return report


def _validate_files(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    for label, path in (("expression", args.input), ("params", args.params)):
        if not path.is_file():
            raise ValueError(f"{label} file does not exist: {path}")
    if not args.extra.is_dir():
        raise ValueError(f"extra input directory does not exist: {args.extra}")
    if not args.output_dir.is_dir():
        raise ValueError(f"ANDREA output directory does not exist: {args.output_dir}")

    tf_list = args.extra / "tf_list.txt"
    groups = args.extra / "groups.tsv"
    for label, path in (("tf_list", tf_list), ("groups", groups)):
        if not path.is_file():
            raise ValueError(f"required {label} input does not exist: {path}")
    execution = args.params.parent / "execution.json"
    if not execution.is_file():
        raise ValueError(f"ANDREA execution contract does not exist: {execution}")
    return tf_list, groups, execution


def _validated_params(path: Path) -> dict[str, Any]:
    params = _load_json_object(path, "params")
    unknown = sorted(set(params) - ALLOWED_PARAMS)
    if unknown:
        raise ValueError(f"unsupported SPATHI parameters: {', '.join(unknown)}")
    missing = sorted(REQUIRED_PARAMS - set(params))
    if missing:
        raise ValueError(f"missing frozen SPATHI parameters: {', '.join(missing)}")
    return params


def _validate_execution(path: Path) -> None:
    execution = _load_json_object(path, "execution")
    if execution.get("mode") != "group_native":
        raise ValueError("SPATHI requires execution.mode=group_native")


def _validate_network_header(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"SPATHI completed without network.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if tuple(header or ()) != EXPECTED_NETWORK_COLUMNS:
        raise RuntimeError(
            "Unexpected SPATHI network.csv columns: "
            f"expected {EXPECTED_NETWORK_COLUMNS}, got {tuple(header or ())}"
        )


def run(args: argparse.Namespace) -> None:
    tf_list, groups, execution_path = _validate_files(args)
    params = _validated_params(args.params)
    _validate_execution(execution_path)

    published_network = args.output_dir / "network.csv"
    if published_network.exists():
        raise FileExistsError("SPATHI adapter refuses to overwrite prior outputs")

    _write_progress(
        args.output_dir,
        percent=5,
        status="running",
        phase="validating_inputs",
        message="Validated ANDREA external-tool contract",
    )

    from spathi import SpathiConfig, infer

    temporary_network: Path | None = None
    try:
        with TemporaryDirectory(prefix="spathi-adapter-") as scratch_name:
            config = SpathiConfig(
                expression=args.input,
                tf_list=tf_list,
                groups=groups,
                output_dir=Path(scratch_name) / "spathi-run",
                threads=args.threads,
                report=False,
                **params,
            )
            result = infer(
                config,
                progress_callback=_progress_callback(args.output_dir),
                checkpoint=False,
            )
            _validate_network_header(result.network_path)
            with NamedTemporaryFile(
                prefix=".network.",
                suffix=".tmp",
                dir=args.output_dir,
                delete=False,
            ) as temporary_handle:
                temporary_network = Path(temporary_handle.name)
            shutil.copyfile(result.network_path, temporary_network)
            temporary_network.chmod(0o644)
        os.replace(temporary_network, published_network)
    finally:
        if temporary_network is not None:
            temporary_network.unlink(missing_ok=True)
    _write_progress(
        args.output_dir,
        percent=100,
        status="completed",
        phase="done",
        message="Published ANDREA-compatible network.csv",
        warnings=result.warnings,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 - container boundary must record every failure
        try:
            _write_progress(
                args.output_dir,
                percent=100,
                status="failed",
                phase="failed",
                message=str(exc),
            )
        except OSError:
            pass
        print(f"SPATHI adapter failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
