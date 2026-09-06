"""Fail-closed checks for material that can enter the public repository.

The boundary is intentionally generic: it does not contain project-specific sample
names, identifiers, dimensions, hashes, or source locations. It inspects every tracked
or non-ignored untracked path, so a newly added file is covered before it is staged.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .config import BenchmarkConfigError

_RESTRICTED_SUFFIXES = frozenset(
    {
        ".h5",
        ".h5ad",
        ".hdf5",
        ".loom",
        ".parquet",
        ".feather",
        ".rds",
        ".rdata",
        ".rda",
    }
)
_RESTRICTED_DIRECTORY_NAMES = frozenset(
    {
        "confi" + "dential",
        "private_data",
        "raw_data",
        "source_data",
    }
)
_RESTRICTED_JSON_KEYS = frozenset(
    {
        "external_source_projection",
        "individual_level_metadata",
        "source_dataset_digest",
        "source_dataset_location",
    }
)
_RESTRICTED_TEXT_MARKERS = (
    "confi" + "dential",
    "private" + " dataset",
    "private" + " data",
    "/ho" + "me/",
    "/Users" + "/",
)


def _versionable_paths(repo_root: Path) -> tuple[Path, ...]:
    outcome = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
    )
    if outcome.returncode != 0:
        detail = outcome.stderr.decode("utf-8", errors="replace").strip()
        raise BenchmarkConfigError(
            f"Cannot enumerate the publication boundary with Git: {detail or 'unknown error'}"
        )
    members: list[Path] = []
    for raw in outcome.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            relative_text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BenchmarkConfigError(
                "Publication boundary contains a path that is not valid UTF-8"
            ) from exc
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or "\\" in relative_text:
            raise BenchmarkConfigError(
                f"Publication boundary contains an unsafe repository path: {relative_text!r}"
            )
        members.append(repo_root.joinpath(*relative.parts))
    return tuple(members)


def _iter_json_members(
    value: Any, trail: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, member in value.items():
            key_text = str(key)
            member_trail = (*trail, key_text)
            yield member_trail, member
            yield from _iter_json_members(member, member_trail)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, member in enumerate(value):
            yield from _iter_json_members(member, (*trail, str(index)))


def _path_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for member in value:
            if isinstance(member, str):
                yield member


def _validate_json_payload(path: Path, payload: Any) -> None:
    for trail, value in _iter_json_members(payload):
        key = trail[-1].casefold().replace("-", "_")
        if key in _RESTRICTED_JSON_KEYS:
            raise BenchmarkConfigError(
                f"Publication boundary rejects restricted metadata key {'.'.join(trail)!r} "
                f"in {path}"
            )
        if key == "path" or key.endswith("_path") or key.endswith("_paths"):
            for raw_path in _path_values(value):
                normalized = raw_path.replace("\\", "/")
                candidate = PurePosixPath(normalized)
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise BenchmarkConfigError(
                        f"Publication boundary rejects an external path at "
                        f"{'.'.join(trail)!r} in {path}"
                    )


def validate_publication_boundary(repo_root: Path) -> dict[str, Any]:
    """Validate every file Git considers eligible for publication.

    Ignored runtime material is outside this repository-publication boundary. Symlinks,
    raw scientific containers, restricted metadata shapes, external JSON paths and
    generic disclosure markers are rejected before normal design validation proceeds.
    """

    root = Path(os.path.abspath(repo_root))
    if root.is_symlink() or not root.is_dir() or not (root / ".git").exists():
        raise BenchmarkConfigError("Publication boundary requires a canonical Git repository")

    checked = 0
    for path in _versionable_paths(root):
        if not path.exists() and not path.is_symlink():
            # A tracked path deleted in the worktree cannot be published by a subsequent
            # commit and is covered by repository-cleanliness checks before execution.
            continue
        try:
            relative = path.relative_to(root)
        except ValueError as exc:  # pragma: no cover - defended by Git path validation
            raise BenchmarkConfigError("Publication boundary path escaped its repository") from exc
        if path.is_symlink():
            raise BenchmarkConfigError(
                f"Publication boundary rejects symlinked content: {relative.as_posix()}"
            )
        if not path.is_file():
            raise BenchmarkConfigError(
                f"Publication boundary contains a non-regular path: {relative.as_posix()}"
            )
        if path.suffix.casefold() in _RESTRICTED_SUFFIXES:
            raise BenchmarkConfigError(
                f"Publication boundary rejects raw scientific data: {relative.as_posix()}"
            )
        normalized_parts = {part.casefold().replace("-", "_") for part in relative.parts[:-1]}
        if normalized_parts & _RESTRICTED_DIRECTORY_NAMES:
            raise BenchmarkConfigError(
                f"Publication boundary rejects a restricted data directory: {relative.as_posix()}"
            )

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise BenchmarkConfigError(
                "Publication boundary accepts UTF-8 source and design files only; "
                f"review binary material separately: {relative.as_posix()}"
            ) from exc
        folded = text.casefold()
        marker = next(
            (candidate for candidate in _RESTRICTED_TEXT_MARKERS if candidate.casefold() in folded),
            None,
        )
        if marker is not None:
            raise BenchmarkConfigError(
                f"Publication boundary found a restricted disclosure marker in "
                f"{relative.as_posix()}"
            )
        if path.suffix.casefold() == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise BenchmarkConfigError(
                    f"Publication JSON is invalid: {relative.as_posix()}: {exc}"
                ) from exc
            _validate_json_payload(relative, payload)
        checked += 1

    return {
        "status": "pass",
        "scope": "tracked_and_nonignored_untracked_files",
        "files_checked": checked,
    }


__all__ = ["validate_publication_boundary"]
