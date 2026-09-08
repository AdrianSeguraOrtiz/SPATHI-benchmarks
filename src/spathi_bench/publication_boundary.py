"""Fail-closed checks for material that can enter the public repository.

The boundary is intentionally generic: it contains no project-specific sample names,
identifiers, source dimensions, hashes or source locations. It validates both the exact
Git-index blobs that a commit would publish and every non-ignored worktree file, so a
staged/worktree mismatch cannot hide restricted content in either direction.
"""

from __future__ import annotations

import json
import math
import os
import re
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

_APPROVED_CONSTRAINT_FILES = frozenset(
    {
        "constraints/benchmark.txt",
        "containers/spathi/constraints.txt",
    }
)
_APPROVED_SYNTHETIC_TABLES = frozenset(
    {
        "experiments/v1/simulations/scmultisim/development/medium-regulatory-network.tsv",
        "experiments/v1/simulations/scmultisim/development/regulatory-network.tsv",
        "experiments/v1/simulations/scmultisim/dimensional_transfer/regulatory-network-tf10.tsv",
        "experiments/v1/simulations/scmultisim/dimensional_transfer/regulatory-network-tf100.tsv",
        "experiments/v1/simulations/scmultisim/dimensional_transfer/regulatory-network-tf1500.tsv",
        "experiments/v1/simulations/scmultisim/efficiency_development/regulatory-network.tsv",
        "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-g15000-tf100.tsv",
        "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-g3000-tf1500.tsv",
        "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-tf10.tsv",
        "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-tf100.tsv",
        "experiments/v1/simulations/scmultisim/efficiency_validation/regulatory-network.tsv",
        "experiments/v1/simulations/scmultisim/estimator_scale_transfer/regulatory-network.tsv",
        "experiments/v1/simulations/scmultisim/test/medium-regulatory-network.tsv",
        "experiments/v1/simulations/scmultisim/test/regulatory-network.tsv",
        "experiments/v1/simulations/scmultisim/validation/medium-regulatory-network.tsv",
        "experiments/v1/simulations/scmultisim/validation/regulatory-network.tsv",
    }
)
_TABULAR_SUFFIXES = frozenset({".csv", ".tsv", ".txt"})
_SYNTHETIC_GENE = re.compile(r"G[0-9]+\Z")
_SYNTHETIC_TF = re.compile(r"TF[0-9]+\Z")
_PINNED_REQUIREMENT = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9_,.-]+\])?"
    r"(?:===|==)[A-Za-z0-9][A-Za-z0-9.*+!._-]*\Z"
)


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    outcome = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
    )
    if outcome.returncode != 0:
        detail = outcome.stderr.decode("utf-8", errors="replace").strip()
        raise BenchmarkConfigError(
            f"Cannot inspect the publication boundary with Git: {detail or 'unknown error'}"
        )
    return outcome.stdout


def _relative_path(raw: bytes) -> PurePosixPath:
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
    return relative


def _index_payloads(repo_root: Path) -> tuple[tuple[PurePosixPath, bytes], ...]:
    entries: list[tuple[PurePosixPath, bytes]] = []
    blob_cache: dict[str, bytes] = {}
    for raw in _run_git(repo_root, "ls-files", "--stage", "-z").split(b"\0"):
        if not raw:
            continue
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            raw_mode, raw_object, raw_stage = metadata.split()
            mode = raw_mode.decode("ascii")
            object_id = raw_object.decode("ascii")
            stage = raw_stage.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise BenchmarkConfigError("Git returned an invalid publication-index entry") from exc
        relative = _relative_path(raw_path)
        if stage != "0":
            raise BenchmarkConfigError(
                f"Publication boundary rejects an unresolved index entry: {relative.as_posix()}"
            )
        if mode == "120000":
            raise BenchmarkConfigError(
                f"Publication boundary rejects symlinked content: {relative.as_posix()}"
            )
        if mode not in {"100644", "100755"}:
            raise BenchmarkConfigError(
                f"Publication boundary contains a non-regular index entry: {relative.as_posix()}"
            )
        payload = blob_cache.get(object_id)
        if payload is None:
            payload = _run_git(repo_root, "cat-file", "blob", object_id)
            blob_cache[object_id] = payload
        entries.append((relative, payload))
    return tuple(entries)


def _worktree_payloads(repo_root: Path) -> tuple[tuple[PurePosixPath, bytes], ...]:
    entries: list[tuple[PurePosixPath, bytes]] = []
    raw_listing = _run_git(
        repo_root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    for raw_path in raw_listing.split(b"\0"):
        if not raw_path:
            continue
        relative = _relative_path(raw_path)
        path = repo_root.joinpath(*relative.parts)
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            raise BenchmarkConfigError(
                f"Publication boundary rejects symlinked content: {relative.as_posix()}"
            )
        if not path.is_file():
            raise BenchmarkConfigError(
                f"Publication boundary contains a non-regular path: {relative.as_posix()}"
            )
        entries.append((relative, path.read_bytes()))
    return tuple(entries)


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


def _validate_json_payload(path: PurePosixPath, payload: Any) -> None:
    for trail, value in _iter_json_members(payload):
        key = trail[-1].casefold().replace("-", "_")
        if key in _RESTRICTED_JSON_KEYS:
            raise BenchmarkConfigError(
                f"Publication boundary rejects restricted metadata key {'.'.join(trail)!r} "
                f"in {path.as_posix()}"
            )
        if key == "path" or key.endswith("_path") or key.endswith("_paths"):
            for raw_path in _path_values(value):
                normalized = raw_path.replace("\\", "/")
                candidate = PurePosixPath(normalized)
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise BenchmarkConfigError(
                        "Publication boundary rejects an external path at "
                        f"{'.'.join(trail)!r} in {path.as_posix()}"
                    )


def _validate_synthetic_network(path: PurePosixPath, text: str) -> None:
    rows = text.splitlines()
    if not rows or rows[0] != "target\tregulator\teffect":
        raise BenchmarkConfigError(
            f"Approved synthetic network has an invalid header: {path.as_posix()}"
        )
    for line_number, row in enumerate(rows[1:], start=2):
        fields = row.split("\t")
        if len(fields) != 3:
            raise BenchmarkConfigError(
                f"Approved synthetic network has an invalid row at {path.as_posix()}:{line_number}"
            )
        target, regulator, raw_effect = fields
        try:
            effect = float(raw_effect)
        except ValueError as exc:
            raise BenchmarkConfigError(
                f"Approved synthetic network has a non-numeric effect at "
                f"{path.as_posix()}:{line_number}"
            ) from exc
        if (
            _SYNTHETIC_GENE.fullmatch(target) is None
            or _SYNTHETIC_TF.fullmatch(regulator) is None
            or not math.isfinite(effect)
        ):
            raise BenchmarkConfigError(
                f"Approved synthetic network contains a non-synthetic row at "
                f"{path.as_posix()}:{line_number}"
            )


def _validate_constraints(path: PurePosixPath, text: str) -> None:
    requirements = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not requirements or any(_PINNED_REQUIREMENT.fullmatch(line) is None for line in requirements):
        raise BenchmarkConfigError(
            f"Approved constraints file contains non-pinned content: {path.as_posix()}"
        )


def _validate_payload(relative: PurePosixPath, payload: bytes) -> None:
    suffix = relative.suffix.casefold()
    if suffix in _RESTRICTED_SUFFIXES:
        raise BenchmarkConfigError(
            f"Publication boundary rejects raw scientific data: {relative.as_posix()}"
        )
    normalized_parts = {part.casefold().replace("-", "_") for part in relative.parts[:-1]}
    if normalized_parts & _RESTRICTED_DIRECTORY_NAMES:
        raise BenchmarkConfigError(
            f"Publication boundary rejects a restricted data directory: {relative.as_posix()}"
        )
    try:
        text = payload.decode("utf-8")
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
            f"Publication boundary found a restricted disclosure marker in {relative.as_posix()}"
        )

    relative_text = relative.as_posix()
    if suffix in _TABULAR_SUFFIXES:
        if relative_text in _APPROVED_SYNTHETIC_TABLES:
            _validate_synthetic_network(relative, text)
        elif relative_text in _APPROVED_CONSTRAINT_FILES:
            _validate_constraints(relative, text)
        else:
            raise BenchmarkConfigError(
                f"Publication boundary rejects an unapproved text table: {relative_text}"
            )
    if suffix == ".json":
        try:
            json_payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BenchmarkConfigError(
                f"Publication JSON is invalid: {relative.as_posix()}: {exc}"
            ) from exc
        _validate_json_payload(relative, json_payload)


def validate_publication_boundary(repo_root: Path) -> dict[str, Any]:
    """Validate exact index blobs and non-ignored worktree representations.

    Ignored runtime material remains outside this repository-publication boundary.
    Every indexed representation is read from Git rather than from the worktree; current
    tracked and untracked worktree representations are checked independently as a local
    disclosure guard.
    """

    root = Path(os.path.abspath(repo_root))
    if root.is_symlink() or not root.is_dir() or not (root / ".git").exists():
        raise BenchmarkConfigError("Publication boundary requires a canonical Git repository")

    indexed = _index_payloads(root)
    worktree = _worktree_payloads(root)
    checked_representations: set[tuple[str, bytes]] = set()
    for relative, payload in (*indexed, *worktree):
        identity = (relative.as_posix(), payload)
        if identity in checked_representations:
            continue
        _validate_payload(relative, payload)
        checked_representations.add(identity)

    unique_paths = {relative.as_posix() for relative, _payload in (*indexed, *worktree)}
    return {
        "status": "pass",
        "scope": "index_blobs_and_nonignored_worktree_files",
        "files_checked": len(unique_paths),
        "representations_checked": len(checked_representations),
    }


__all__ = ["validate_publication_boundary"]
