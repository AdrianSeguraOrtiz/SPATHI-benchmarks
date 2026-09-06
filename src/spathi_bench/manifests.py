"""Strict helpers for ANDREA benchmark and inference handoffs."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

from .config import BenchmarkConfigError, load_json_object
from .safe_io import path_below_no_symlinks


def unique_direct_manifest(output_root: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename or "\\" in filename:
        raise BenchmarkConfigError(f"Manifest filename is not canonical: {filename!r}")
    root = path_below_no_symlinks(
        output_root,
        output_root,
        "manifest output root",
        require_directory=True,
    )
    candidates = sorted(root.glob(f"*/{filename}"))
    if len(candidates) != 1:
        raise BenchmarkConfigError(
            f"Expected exactly one {filename} directly below {root}; found {len(candidates)}"
        )
    return path_below_no_symlinks(
        candidates[0],
        root,
        f"direct {filename}",
        require_file=True,
    )


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
    safe_root = path_below_no_symlinks(
        root,
        root,
        f"{label} package root",
        require_directory=True,
    )
    candidate = Path(os.path.abspath(safe_root / Path(*relative.parts)))
    candidate = path_below_no_symlinks(
        candidate,
        safe_root,
        label,
    )
    if not candidate.is_file():
        raise BenchmarkConfigError(f"{label} does not exist: {candidate}")
    return candidate


def datasets_from_benchmark(manifest_path: Path, generation_job: str) -> list[dict[str, Any]]:
    safe_manifest = path_below_no_symlinks(
        manifest_path,
        manifest_path.parent,
        "benchmark manifest",
        require_file=True,
    )
    manifest = load_json_object(safe_manifest, "benchmark manifest")
    raw_datasets = manifest.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise BenchmarkConfigError("benchmark-manifest.json has no datasets")
    root = safe_manifest.parent
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
