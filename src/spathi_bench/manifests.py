"""Strict helpers for ANDREA benchmark and inference handoffs."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from .config import BenchmarkConfigError, load_json_object


def unique_direct_manifest(output_root: Path, filename: str) -> Path:
    candidates = sorted(output_root.glob(f"*/{filename}"))
    if len(candidates) != 1:
        raise BenchmarkConfigError(
            f"Expected exactly one {filename} directly below {output_root}; found {len(candidates)}"
        )
    return candidates[0].resolve()


def resolve_manifest_member(root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise BenchmarkConfigError(f"{label} must be a non-empty relative path")
    if raw_path != raw_path.strip() or "\\" in raw_path or "//" in raw_path:
        raise BenchmarkConfigError(f"{label} is not a canonical POSIX path: {raw_path!r}")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or str(relative) != raw_path
        or any(part in {".", ".."} for part in relative.parts)
    ):
        raise BenchmarkConfigError(f"{label} must be a canonical relative path: {raw_path}")
    candidate = (root / Path(*relative.parts)).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise BenchmarkConfigError(f"{label} escapes its package root: {raw_path}")
    if not candidate.is_file():
        raise BenchmarkConfigError(f"{label} does not exist: {candidate}")
    return candidate


def datasets_from_benchmark(manifest_path: Path, generation_job: str) -> list[dict[str, Any]]:
    manifest = load_json_object(manifest_path, "benchmark manifest")
    raw_datasets = manifest.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise BenchmarkConfigError("benchmark-manifest.json has no datasets")
    root = manifest_path.parent
    datasets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_datasets:
        if not isinstance(raw, dict):
            raise BenchmarkConfigError("benchmark-manifest datasets entries must be objects")
        dataset_id = raw.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id or dataset_id in seen:
            raise BenchmarkConfigError(f"Invalid or duplicate dataset_id: {dataset_id!r}")
        seen.add(dataset_id)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "generation_job": generation_job,
                "simulator_id": raw.get("simulator_id"),
                "seed": raw.get("seed"),
                "dataset_manifest": str(
                    resolve_manifest_member(root, raw.get("dataset_manifest"), "dataset manifest")
                ),
                "ground_truth_manifest": str(
                    resolve_manifest_member(
                        root, raw.get("ground_truth_manifest"), "ground-truth manifest"
                    )
                ),
            }
        )
    return datasets
