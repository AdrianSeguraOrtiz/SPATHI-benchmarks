"""Discover and content-bind every external input declared by a scenario."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import BenchmarkConfigError, load_json_object
from .safe_io import path_below_no_symlinks


def _directory_files(source: Path, root: Path, label: str) -> list[Path]:
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(source, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current = Path(directory)
        for name in directory_names:
            path_below_no_symlinks(current / name, root, label, require_directory=True)
        for name in file_names:
            files.append(
                path_below_no_symlinks(
                    current / name,
                    root,
                    label,
                    require_file=True,
                )
            )
    if not files:
        raise BenchmarkConfigError(f"{label} directory is empty")
    return files


def scenario_external_input_records(
    scenario_path: Path,
    repository_root: Path,
) -> dict[str, dict[str, Any]]:
    """Return a deterministic, symlink-free content record for scenario inputs.

    File inputs bind one file. Directory inputs recursively bind every regular
    file below the declared directory. Paths are recorded relative to the
    repository snapshot so both location and contents contribute to the
    generation artifact identity.
    """

    root = path_below_no_symlinks(
        repository_root,
        repository_root,
        "Scenario input repository",
        require_directory=True,
    )
    scenario = path_below_no_symlinks(
        scenario_path,
        root,
        "Scenario",
        require_file=True,
    )
    payload = load_json_object(scenario, "scenario")
    raw_inputs = payload.get("inputs", {})
    if not isinstance(raw_inputs, dict):
        raise BenchmarkConfigError("Scenario inputs must be an object")

    records: dict[str, dict[str, Any]] = {}
    for input_id in sorted(raw_inputs):
        input_spec = raw_inputs[input_id]
        if not isinstance(input_id, str) or not input_id or not isinstance(input_spec, dict):
            raise BenchmarkConfigError("Scenario contains an invalid external input")
        raw_path = input_spec.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise BenchmarkConfigError(f"Scenario input {input_id!r} has no path")
        declared = Path(raw_path)
        if declared.is_absolute():
            raise BenchmarkConfigError(
                f"Scenario input {input_id!r} path must be relative to its scenario"
            )
        source = path_below_no_symlinks(
            scenario.parent / declared,
            root,
            f"Scenario input {input_id!r}",
        )
        if source.is_file():
            kind = "file"
            files = [source]
        elif source.is_dir():
            kind = "directory"
            files = _directory_files(source, root, f"Scenario input {input_id!r}")
        else:
            raise BenchmarkConfigError(
                f"Scenario input {input_id!r} does not exist or is not a regular file/directory"
            )
        records[input_id] = {
            "declared_path": raw_path,
            "kind": kind,
            "files": [
                {
                    "path": member.relative_to(root).as_posix(),
                    "sha256": sha256_file(member),
                }
                for member in files
            ],
        }
    return records


def scenario_external_input_paths(
    scenario_path: Path,
    repository_root: Path,
) -> list[Path]:
    """Return the unique repository members covered by scenario inputs."""

    root = repository_root.absolute()
    records = scenario_external_input_records(scenario_path, root)
    return sorted(
        {root / member["path"] for record in records.values() for member in record["files"]}
    )


__all__ = ["scenario_external_input_paths", "scenario_external_input_records"]
