"""Run one physical SPATHI forest and publish exact logical tree-count prefixes.

This is a benchmark-owned runner, not an ANDREA inference-tool entry point. It
preserves ANDREA's one-run/one-network contract by publishing a fan-out bundle
that SPATHI-benchmarks subsequently evaluates as four distinct logical runs.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from run_tool import (
    _group_count,
    _load_json_object,
    _resolve_group_policy,
    _validate_execution,
    _validate_files,
    _validated_scalar_params,
)


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


def _validated_plan(
    path: Path, *, group_count: int
) -> tuple[str, dict[str, Any], tuple[dict[str, Any], ...]]:
    plan = _load_json_object(path, "convergence plan")
    required = {
        "schema_version",
        "physical_run_id",
        "parent_configuration_id",
        "algorithm_seed",
        "model",
        "logical_prefixes",
    }
    if set(plan) != required or plan.get("schema_version") != "1.0":
        raise ValueError("convergence plan has missing or unsupported fields")
    physical_run_id = plan["physical_run_id"]
    if not isinstance(physical_run_id, str) or not physical_run_id:
        raise ValueError("physical_run_id must be non-empty text")
    parent_configuration_id = plan["parent_configuration_id"]
    if not isinstance(parent_configuration_id, str) or not parent_configuration_id:
        raise ValueError("parent_configuration_id must be non-empty text")
    algorithm_seed = plan["algorithm_seed"]
    if isinstance(algorithm_seed, bool) or not isinstance(algorithm_seed, int):
        raise ValueError("algorithm_seed must be an integer")
    raw_model = plan["model"]
    if not isinstance(raw_model, dict):
        raise ValueError("convergence model must be an object")
    model = (
        _resolve_group_policy(dict(raw_model), group_count=group_count)
        if "group_policy" in raw_model
        else _validated_scalar_params(dict(raw_model))
    )
    if model["random_seed"] != algorithm_seed:
        raise ValueError("convergence plan seed differs from its physical model")
    prefixes = plan["logical_prefixes"]
    if not isinstance(prefixes, list) or len(prefixes) != 4:
        raise ValueError("convergence plan must declare exactly four logical prefixes")
    expected_fields = {
        "run_id",
        "configuration_id",
        "configuration_sha256",
        "n_estimators",
    }
    for item in prefixes:
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError("each logical prefix has missing or unsupported fields")
        if not all(
            isinstance(item[field], str) and item[field] for field in ("run_id", "configuration_id")
        ):
            raise ValueError("logical prefix identifiers must be non-empty text")
        digest = item["configuration_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("logical prefix configuration_sha256 must be lowercase SHA-256")
        count = item["n_estimators"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("logical prefix n_estimators must be a positive integer")
    counts = tuple(int(item["n_estimators"]) for item in prefixes)
    if any(left >= right for left, right in zip(counts, counts[1:], strict=False)):
        raise ValueError("logical prefix n_estimators must be strictly increasing")
    if counts[-1] != model["n_estimators"]:
        raise ValueError("the final logical prefix must equal the physical forest ceiling")
    for field in ("run_id", "configuration_id", "configuration_sha256"):
        values = [item[field] for item in prefixes]
        if len(values) != len(set(values)):
            raise ValueError(f"logical prefix {field} values must be unique")
    return physical_run_id, model, tuple(dict(item) for item in prefixes)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    tf_list, groups, execution_path = _validate_files(args)
    _validate_execution(execution_path)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("convergence output directory must be empty")
    physical_run_id, model, prefixes = _validated_plan(
        args.params,
        group_count=_group_count(groups),
    )
    counts = tuple(int(item["n_estimators"]) for item in prefixes)

    from spathi import SpathiConfig, prepare_forest_prefixes
    from spathi.diagnostics import compute_weight_diagnostics
    from spathi.outputs import IncrementalRunWriter

    with TemporaryDirectory(prefix=".spathi-convergence-", dir=args.output_dir.parent) as name:
        staging = Path(name)
        output_staging = staging / "prefixes"
        output_staging.mkdir()
        config = SpathiConfig(
            expression=args.input,
            tf_list=tf_list,
            groups=groups,
            output_dir=staging / "unused-ordinary-output",
            threads=args.threads,
            report=False,
            **model,
        )
        started = perf_counter()
        prepared = prepare_forest_prefixes(config, estimator_counts=counts)
        preparation_seconds = perf_counter() - started
        prefix_dirs: dict[int, Path] = {}
        edge_counts = {count: 0 for count in counts}
        completed_models = {count: 0 for count in counts}
        trained_models = {count: 0 for count in counts}
        skipped_counts = {count: 0 for count in counts}
        inference_seconds = {count: 0.0 for count in counts}
        with ExitStack() as stack:
            writers = {}
            diagnostics = tuple(
                compute_weight_diagnostics(result, emit_warnings=False)
                for result in prepared.weight_results
            )
            for count in counts:
                prefix_dir = output_staging / str(count)
                prefix_dir.mkdir()
                prefix_dirs[count] = prefix_dir
                writer = stack.enter_context(IncrementalRunWriter(prefix_dir))
                writers[count] = writer
                for weight, diagnostic in zip(
                    prepared.weight_results,
                    diagnostics,
                    strict=True,
                ):
                    writer.write_weights(weight)
                    writer.write_weight_diagnostics(diagnostic)

            target_batch_size = max(1, 2 * args.threads)
            for batch in prepared.iter_batches(target_batch_size=target_batch_size):
                if tuple(item.n_estimators for item in batch) != counts:
                    raise RuntimeError("SPATHI prefix core returned an inconsistent schedule")
                for item in batch:
                    count = item.n_estimators
                    result = item.inference
                    writer = writers[count]
                    edge_counts[count] += writer.write_edges(result.edges)
                    writer.write_skipped_targets(result.skipped_targets)
                    writer.write_model_diagnostics(result.model_stats)
                    completed_models[count] += result.completed_models
                    trained_models[count] += result.trained_models
                    skipped_counts[count] += len(result.skipped_targets)
                    inference_seconds[count] += result.duration_seconds

        logical_outputs: list[dict[str, Any]] = []
        for logical, count in zip(prefixes, counts, strict=True):
            prefix_dir = prefix_dirs[count]
            metadata = {
                "schema_version": "1.0",
                "artifact_type": "spathi_forest_prefix",
                "status": "complete",
                "physical_run_id": physical_run_id,
                "run_id": logical["run_id"],
                "configuration_id": logical["configuration_id"],
                "configuration_sha256": logical["configuration_sha256"],
                "n_estimators": count,
                "physical_forest_ceiling": counts[-1],
                "prefix_schedule": list(counts),
                "preparation_seconds": preparation_seconds,
                "shared_inference_seconds": inference_seconds[count],
                "total_models": completed_models[count],
                "trained_models": trained_models[count],
                "skipped_target_records": skipped_counts[count],
                "n_edges": edge_counts[count],
            }
            _write_json(prefix_dir / "run_metadata.json", metadata)
            files = [
                path
                for path in sorted(prefix_dir.iterdir(), key=lambda item: item.name)
                if path.is_file()
            ]
            logical_outputs.append(
                {
                    **logical,
                    "relative_path": f"prefixes/{count}",
                    "files": [
                        {
                            "path": path.name,
                            "size": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                        for path in files
                    ],
                }
            )
        manifest = {
            "schema_version": "1.0",
            "artifact_type": "spathi_convergence_fanout",
            "status": "complete",
            "physical_run_id": physical_run_id,
            "physical_fits": 1,
            "logical_results": 4,
            "estimator_counts": list(counts),
            "logical_outputs": logical_outputs,
        }
        _write_json(staging / "fanout-manifest.json", manifest)
        os.replace(output_staging, args.output_dir / "prefixes")
        os.replace(staging / "fanout-manifest.json", args.output_dir / "fanout-manifest.json")


def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 - container boundary reports all failures
        print(f"SPATHI convergence runner failed: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
