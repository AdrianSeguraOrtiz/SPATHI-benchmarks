"""Symlink-safe path validation and atomic text publication."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import BenchmarkConfigError


def _absolute(path: Path) -> Path:
    """Return a lexical absolute path without dereferencing symlinks."""

    return Path(os.path.abspath(path))


def reject_symlink_components(path: Path, label: str) -> Path:
    """Reject an existing symlink at any component of one absolute path."""

    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise BenchmarkConfigError(f"{label} contains a symlink: {current}")
    return absolute


def path_below_no_symlinks(
    path: Path,
    boundary: Path,
    label: str,
    *,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    """Validate lexical containment and reject symlinks before dereferencing."""

    if require_file and require_directory:
        raise ValueError("A safe path cannot be required to be both a file and a directory")
    safe_boundary = reject_symlink_components(boundary, f"{label} boundary")
    safe_path = reject_symlink_components(path, label)
    if not safe_path.is_relative_to(safe_boundary):
        raise BenchmarkConfigError(f"{label} escapes {safe_boundary}")
    if require_file and not safe_path.is_file():
        raise BenchmarkConfigError(f"{label} is missing: {safe_path}")
    if require_directory and not safe_path.is_dir():
        raise BenchmarkConfigError(f"{label} directory is missing: {safe_path}")
    return safe_path


def _output_path(path: Path, label: str, boundary: Path | None) -> Path:
    if boundary is None:
        return reject_symlink_components(path, label)
    return path_below_no_symlinks(path, boundary, label)


def _prepare_parent(target: Path, label: str, boundary: Path | None) -> None:
    parent = _output_path(target.parent, f"{label} parent", boundary)
    parent.mkdir(parents=True, exist_ok=True)
    parent = _output_path(target.parent, f"{label} parent", boundary)
    if not parent.is_dir():
        raise BenchmarkConfigError(f"{label} parent is not a directory: {parent}")


def _temporary_text(target: Path, content: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


@contextmanager
def unique_temporary_file(
    parent: Path,
    label: str,
    *,
    boundary: Path | None = None,
    prefix: str = ".tmp-",
    suffix: str = "",
) -> Iterator[Path]:
    """Yield an exclusively created file and remove it on every exit path.

    The file name is selected by :func:`tempfile.mkstemp`, so concurrent writers
    cannot reuse a predictable staging path.  Callers that need a disk-backed
    array may close and reopen the yielded file; ownership remains with this
    context manager until it exits.
    """

    if not prefix or Path(prefix).name != prefix or Path(suffix).name != suffix:
        raise ValueError("Temporary file prefix and suffix must be simple names")
    safe_parent = _output_path(parent, f"{label} parent", boundary)
    safe_parent.mkdir(parents=True, exist_ok=True)
    safe_parent = _output_path(parent, f"{label} parent", boundary)
    if not safe_parent.is_dir():
        raise BenchmarkConfigError(f"{label} parent is not a directory: {safe_parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=prefix,
        suffix=suffix,
        dir=safe_parent,
    )
    os.close(descriptor)
    temporary = _output_path(Path(temporary_name), label, boundary)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def atomic_write_text(
    path: Path,
    content: str,
    label: str,
    *,
    boundary: Path | None = None,
) -> Path:
    """Atomically replace a text file via a uniquely created sibling temporary."""

    target = _output_path(path, label, boundary)
    _prepare_parent(target, label, boundary)
    if target.exists() and not target.is_file():
        raise BenchmarkConfigError(f"{label} is not a regular file: {target}")

    temporary = _temporary_text(target, content)
    try:
        _output_path(target, label, boundary)
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    label: str,
    *,
    boundary: Path | None = None,
) -> Path:
    """Atomically replace one deterministic JSON document."""

    return atomic_write_text(path, _json_text(payload), label, boundary=boundary)


__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "path_below_no_symlinks",
    "reject_symlink_components",
    "unique_temporary_file",
]
