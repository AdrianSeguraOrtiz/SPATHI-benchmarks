"""Adapt SPATHI to ANDREA's standardized external inference-tool contract."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

EXPECTED_NETWORK_COLUMNS = ("source", "target", "score", "sign", "evidence", "context")
AUDIT_SIDECARS = (
    "target_eligibility.tsv.gz",
    "model_diagnostics.tsv.gz",
    "run_metadata.json",
)
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
REQUIRED_PARAMS = {
    "adaptive_min_estimators",
    "adaptive_patience",
    "adaptive_tolerance",
    "adaptive_tree_step",
    "adaptive_trees",
    "bandwidth",
    "bandwidth_scale",
    "distance_metric",
    "distance_space",
    "distance_standardization",
    "group_size_correction",
    "kernel",
    "max_depth",
    "max_features",
    "min_target_detected_cells",
    "min_target_detected_fraction",
    "min_target_weighted_detected_fraction",
    "min_target_weighted_detected_ess",
    "min_samples_leaf",
    "n_components",
    "n_estimators",
    "pca_svd_solver",
    "bootstrap",
    "random_seed",
    "target_eligibility",
    "tree_method",
    "weight_mode",
}
MODEL_PARAMS = REQUIRED_PARAMS - {"random_seed"}


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
    content = (
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
        + "\n"
    )
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            prefix=".progress.json.",
            suffix=".tmp",
            dir=output_dir,
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _progress_callback(output_dir: Path) -> Callable[[Any], None]:
    """Translate structured SPATHI progress into ANDREA's progress contract."""

    last_percent = 5
    last_signature: tuple[int, str] | None = None

    def report(event: Any) -> None:
        nonlocal last_percent, last_signature
        phase = str(event.phase)
        candidate = _PHASE_PERCENT.get(phase, last_percent)
        if phase == "model_inference" and event.total_models:
            model_fraction = float(event.completed_models) / float(event.total_models)
            candidate = 35 + int(55 * model_fraction)
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


def _group_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "cluster" not in reader.fieldnames:
            raise ValueError("groups.tsv must contain a cluster column")
        groups = {str(row.get("cluster", "")).strip() for row in reader}
    groups.discard("")
    if not groups:
        raise ValueError("groups.tsv contains no non-empty group labels")
    return len(groups)


def _validated_scalar_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - REQUIRED_PARAMS)
    if unknown:
        raise ValueError(f"unsupported SPATHI parameters: {', '.join(unknown)}")
    missing = sorted(REQUIRED_PARAMS - set(params))
    if missing:
        raise ValueError(f"missing frozen SPATHI parameters: {', '.join(missing)}")
    return params


def _resolve_group_policy(params: dict[str, Any], *, group_count: int) -> dict[str, Any]:
    if set(params) != {"group_policy", "random_seed"}:
        raise ValueError(
            "a conditional SPATHI request must contain exactly group_policy and random_seed"
        )
    policy = params["group_policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "routing_parameter",
        "shared_parameters",
        "branches",
    }:
        raise ValueError("group_policy has missing or unsupported fields")
    if policy["routing_parameter"] != "group_count":
        raise ValueError("group_policy.routing_parameter must be 'group_count'")
    branches = policy["branches"]
    if not isinstance(branches, dict) or set(branches) != {"single_group", "multi_group"}:
        raise ValueError("group_policy must define exactly single_group and multi_group branches")
    for candidate_id, expected_condition in (
        ("single_group", {"minimum_group_count": 1, "maximum_group_count": 1}),
        ("multi_group", {"minimum_group_count": 2}),
    ):
        candidate = branches[candidate_id]
        if not isinstance(candidate, dict) or set(candidate) != {"condition", "parameters"}:
            raise ValueError(f"group_policy.{candidate_id} has an invalid contract")
        if candidate["condition"] != expected_condition:
            raise ValueError(f"group_policy.{candidate_id}.condition is not canonical")
        if (
            not isinstance(candidate["parameters"], dict)
            or set(candidate["parameters"]) != MODEL_PARAMS
        ):
            raise ValueError(
                f"group_policy.{candidate_id}.parameters is not a complete SPATHI model"
            )
    branch_id = "single_group" if group_count == 1 else "multi_group"
    branch = branches[branch_id]
    branch_params = branch["parameters"]
    other_branch = branches["multi_group" if branch_id == "single_group" else "single_group"]
    shared = policy["shared_parameters"]
    if (
        not isinstance(shared, list)
        or len(shared) != len(set(shared))
        or set(shared) != MODEL_PARAMS - {"weight_mode", "distance_metric", "group_size_correction"}
    ):
        raise ValueError("group_policy.shared_parameters does not match the conditional contract")
    other_params = other_branch["parameters"]
    if any(other_params.get(name) != branch_params[name] for name in shared):
        raise ValueError("group_policy branches differ in a declared shared parameter")
    resolved = {**branch_params, "random_seed": params["random_seed"]}
    return _validated_scalar_params(resolved)


def _validated_params(path: Path, *, group_count: int | None = None) -> dict[str, Any]:
    params = _load_json_object(path, "params")
    if "group_policy" in params:
        if group_count is None:
            raise ValueError("group_count is required to resolve a conditional SPATHI policy")
        return _resolve_group_policy(params, group_count=group_count)
    return _validated_scalar_params(params)


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


def _validate_audit_sidecars(output_dir: Path) -> dict[str, Path]:
    paths = {name: output_dir / name for name in AUDIT_SIDECARS}
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise RuntimeError(
            "SPATHI completed without required audit sidecars: " + ", ".join(missing)
        )
    for name in ("target_eligibility.tsv.gz", "model_diagnostics.tsv.gz"):
        try:
            with gzip.open(paths[name], "rt", encoding="utf-8", newline="") as handle:
                if next(csv.reader(handle), None) is None:
                    raise RuntimeError(f"SPATHI audit sidecar is empty: {name}")
        except (OSError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Cannot read SPATHI audit sidecar {name}: {exc}") from exc
    metadata = _load_json_object(paths["run_metadata.json"], "SPATHI run metadata")
    if metadata.get("status") != "complete":
        raise RuntimeError("SPATHI run_metadata.json does not describe a complete run")
    return paths


def _stage_outputs(
    output_dir: Path,
    sources: dict[str, Path],
) -> dict[str, Path]:
    staged: dict[str, Path] = {}
    try:
        for name, source in sources.items():
            with NamedTemporaryFile(
                prefix=f".{name}.",
                suffix=".tmp",
                dir=output_dir,
                delete=False,
            ) as temporary_handle:
                temporary = Path(temporary_handle.name)
            staged[name] = temporary
            shutil.copyfile(source, temporary)
            temporary.chmod(0o644)
    except BaseException:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise
    return staged


def run(args: argparse.Namespace) -> None:
    tf_list, groups, execution_path = _validate_files(args)
    params = _validated_params(args.params, group_count=_group_count(groups))
    _validate_execution(execution_path)

    published = {name: args.output_dir / name for name in ("network.csv", *AUDIT_SIDECARS)}
    existing = sorted(name for name, path in published.items() if path.exists())
    if existing:
        raise FileExistsError(
            "SPATHI adapter refuses to overwrite prior outputs: " + ", ".join(existing)
        )

    _write_progress(
        args.output_dir,
        percent=5,
        status="running",
        phase="validating_inputs",
        message="Validated ANDREA external-tool contract",
    )

    from spathi import SpathiConfig, infer

    staged: dict[str, Path] = {}
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
            sidecars = _validate_audit_sidecars(config.output_dir)
            staged = _stage_outputs(
                args.output_dir,
                {"network.csv": result.network_path, **sidecars},
            )
        # Sidecars become visible first and network.csv is the commit marker used by
        # ANDREA. A failed publication can therefore never expose a successful
        # inference result without all of its audit evidence.
        for name in (*AUDIT_SIDECARS, "network.csv"):
            os.replace(staged.pop(name), published[name])
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    _write_progress(
        args.output_dir,
        percent=100,
        status="completed",
        phase="done",
        message="Published ANDREA-compatible network.csv and SPATHI audit sidecars",
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
