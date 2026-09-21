"""Immutable, content-addressed artifacts for incremental benchmark runs.

Artifact identities are derived from a canonical JSON specification.  Results are
assembled below the store's private staging directory and become visible only
after their complete manifest has been written and the directory has been
atomically renamed into place.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_KIND_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_MANIFEST_NAME = "artifact.json"


class ArtifactError(ValueError):
    """Raised when an artifact contract or stored artifact is invalid."""


def _validate_json_value(value: Any, location: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ArtifactError(f"Non-finite number at {location} is not canonical JSON")
        return
    if isinstance(value, list):
        for index, member in enumerate(value):
            _validate_json_value(member, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, member in value.items():
            if not isinstance(key, str):
                raise ArtifactError(f"Object key at {location} is not a string: {key!r}")
            _validate_json_value(member, f"{location}.{key}")
        return
    raise ArtifactError(f"Value at {location} is not JSON-compatible: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value deterministically for hashing and persistence."""

    _validate_json_value(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 digest of canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading it wholly into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"Cannot hash artifact file {path}: {exc}") from exc
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactKey:
    """Address of an artifact, derived from its kind and immutable specification."""

    kind: str
    digest: str

    def __post_init__(self) -> None:
        if not _KIND_RE.fullmatch(self.kind):
            raise ArtifactError(f"Invalid artifact kind: {self.kind!r}")
        if not _DIGEST_RE.fullmatch(self.digest):
            raise ArtifactError(f"Invalid artifact SHA-256 digest: {self.digest!r}")

    @classmethod
    def from_spec(cls, kind: str, spec: Mapping[str, Any]) -> ArtifactKey:
        """Construct a key from a kind and canonical JSON object specification."""

        if not isinstance(spec, Mapping):
            raise ArtifactError("Artifact spec must be a JSON object")
        plain_spec = dict(spec)
        return cls(kind=kind, digest=canonical_json_sha256({"kind": kind, "spec": plain_spec}))

    @property
    def value(self) -> str:
        return f"sha256:{self.digest}"

    def __str__(self) -> str:
        return f"{self.kind}/{self.value}"


@dataclass(frozen=True, slots=True)
class Artifact:
    """A validated complete artifact."""

    key: ArtifactKey
    path: Path
    manifest: dict[str, Any]

    def file(self, relative_path: str) -> Path:
        """Return a validated manifest member, rejecting unknown paths."""

        listed = {entry["path"] for entry in self.manifest["files"]}
        if relative_path not in listed:
            raise ArtifactError(f"File is not part of artifact {self.key}: {relative_path!r}")
        return _resolve_member(self.path, relative_path)


def _canonical_relative_path(raw_path: str) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path:
        raise ArtifactError("Artifact file path must be a non-empty string")
    if raw_path != raw_path.strip() or "\\" in raw_path or "//" in raw_path:
        raise ArtifactError(f"Artifact file path is not canonical POSIX: {raw_path!r}")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or str(relative) != raw_path
        or any(part in {".", ".."} for part in relative.parts)
    ):
        raise ArtifactError(f"Artifact file path must be canonical and relative: {raw_path!r}")
    if raw_path == _MANIFEST_NAME:
        raise ArtifactError(f"{_MANIFEST_NAME} is reserved for the artifact manifest")
    return relative


def _resolve_member(root: Path, raw_path: str) -> Path:
    relative = _canonical_relative_path(raw_path)
    candidate = root.joinpath(*relative.parts)
    root_resolved = root.resolve()
    try:
        candidate_resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise ArtifactError(f"Cannot resolve artifact file {raw_path!r}: {exc}") from exc
    if not candidate_resolved.is_relative_to(root_resolved):
        raise ArtifactError(f"Artifact file path escapes its root: {raw_path!r}")

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactError(f"Artifact file path contains a symlink: {raw_path!r}")
    return candidate


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / _MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArtifactError(f"Artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactError(f"Artifact manifest must be a JSON object: {manifest_path}")
    return manifest


class ArtifactStore:
    """Filesystem artifact store with immutable, atomically published entries."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def staging_root(self) -> Path:
        return self.root / ".tmp"

    def key_for(self, kind: str, spec: Mapping[str, Any]) -> ArtifactKey:
        return ArtifactKey.from_spec(kind, spec)

    def path_for(self, key: ArtifactKey) -> Path:
        return self.root / key.kind / key.digest

    @contextmanager
    def staging_directory(self, key: ArtifactKey) -> Iterator[Path]:
        """Yield a private same-filesystem directory suitable for publication."""

        self.staging_root.mkdir(parents=True, exist_ok=True)
        path = Path(
            tempfile.mkdtemp(prefix=f"{key.kind}-{key.digest[:12]}-", dir=self.staging_root)
        )
        try:
            yield path
        finally:
            if path.exists():
                shutil.rmtree(path)

    def lookup(self, key: ArtifactKey) -> Artifact | None:
        """Return a validated complete artifact, or ``None`` when it is absent."""

        path = self.path_for(key)
        if not path.exists():
            return None
        return self.validate(key)

    def validate(self, key: ArtifactKey) -> Artifact:
        """Validate manifest identity, paths, sizes, and hashes for an artifact."""

        path = self.path_for(key)
        if path.is_symlink() or not path.is_dir():
            raise ArtifactError(f"Artifact directory is missing or invalid: {path}")
        manifest = _read_manifest(path)
        if manifest.get("schema_version") != "1.0":
            raise ArtifactError(f"Unsupported artifact manifest schema: {path / _MANIFEST_NAME}")
        if manifest.get("status") != "complete":
            raise ArtifactError(f"Artifact is not complete: {key}")
        if manifest.get("kind") != key.kind or manifest.get("key") != key.value:
            raise ArtifactError(f"Artifact manifest identity does not match its address: {key}")

        spec = manifest.get("spec")
        if not isinstance(spec, dict):
            raise ArtifactError(f"Artifact spec must be a JSON object: {key}")
        if ArtifactKey.from_spec(key.kind, spec) != key:
            raise ArtifactError(f"Artifact spec does not match its key: {key}")

        raw_files = manifest.get("files")
        if not isinstance(raw_files, list):
            raise ArtifactError(f"Artifact files must be an array: {key}")
        observed_paths: set[str] = set()
        for entry in raw_files:
            if not isinstance(entry, dict):
                raise ArtifactError(f"Artifact file entry must be an object: {key}")
            raw_path = entry.get("path")
            if not isinstance(raw_path, str):
                raise ArtifactError(f"Artifact file entry has no valid path: {key}")
            member = _resolve_member(path, raw_path)
            if raw_path in observed_paths:
                raise ArtifactError(f"Duplicate artifact file path: {raw_path!r}")
            observed_paths.add(raw_path)
            if entry.get("status") != "complete":
                raise ArtifactError(f"Artifact file is not complete: {raw_path!r}")
            expected_digest = entry.get("sha256")
            expected_size = entry.get("size")
            if not isinstance(expected_digest, str) or not _DIGEST_RE.fullmatch(expected_digest):
                raise ArtifactError(f"Invalid SHA-256 for artifact file: {raw_path!r}")
            if (
                not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size < 0
            ):
                raise ArtifactError(f"Invalid size for artifact file: {raw_path!r}")
            if not member.is_file():
                raise ArtifactError(f"Artifact file is missing: {raw_path!r}")
            if member.stat().st_size != expected_size:
                raise ArtifactError(f"Artifact file size differs from manifest: {raw_path!r}")
            if sha256_file(member) != expected_digest:
                raise ArtifactError(f"Artifact file hash differs from manifest: {raw_path!r}")

        disk_paths = self._stored_file_paths(path)
        if disk_paths != observed_paths:
            missing_from_manifest = sorted(disk_paths - observed_paths)
            missing_from_disk = sorted(observed_paths - disk_paths)
            raise ArtifactError(
                "Artifact manifest/file set differs: "
                f"unlisted={missing_from_manifest}, missing={missing_from_disk}"
            )
        return Artifact(key=key, path=path, manifest=manifest)

    def publish_from_temp(
        self,
        temp_dir: Path,
        *,
        kind: str,
        spec: Mapping[str, Any],
        files: Sequence[str] | None = None,
    ) -> Artifact:
        """Finalize and atomically publish files assembled in a staging directory."""

        key = ArtifactKey.from_spec(kind, spec)
        self._validate_staging_directory(temp_dir)
        existing = self.lookup(key)
        if existing is not None:
            return existing

        disk_paths = self._stored_file_paths(temp_dir, allow_manifest=False)
        if files is None:
            selected_paths = disk_paths
        else:
            selected_paths = set()
            for raw_path in files:
                _canonical_relative_path(raw_path)
                if raw_path in selected_paths:
                    raise ArtifactError(f"Duplicate artifact file path: {raw_path!r}")
                selected_paths.add(raw_path)
            if selected_paths != disk_paths:
                raise ArtifactError(
                    "Published file list must exactly match staging contents: "
                    f"listed_only={sorted(selected_paths - disk_paths)}, "
                    f"unlisted={sorted(disk_paths - selected_paths)}"
                )

        file_entries: list[dict[str, Any]] = []
        for raw_path in sorted(selected_paths):
            member = _resolve_member(temp_dir, raw_path)
            if not member.is_file():
                raise ArtifactError(f"Staged artifact file is missing: {raw_path!r}")
            file_entries.append(
                {
                    "path": raw_path,
                    "status": "complete",
                    "sha256": sha256_file(member),
                    "size": member.stat().st_size,
                }
            )

        plain_spec = json.loads(canonical_json_bytes(dict(spec)))
        manifest = {
            "schema_version": "1.0",
            "kind": kind,
            "key": key.value,
            "spec": plain_spec,
            "files": file_entries,
            "status": "complete",
        }
        self._write_manifest(temp_dir / _MANIFEST_NAME, manifest)

        target = self.path_for(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(temp_dir, target)
        except FileExistsError:
            return self.validate(key)
        except OSError as exc:
            # ENOTEMPTY is not mapped to FileExistsError on every platform.  A
            # concurrent publisher may have won the race after our lookup.
            if target.exists():
                return self.validate(key)
            raise ArtifactError(f"Cannot atomically publish artifact {key}: {exc}") from exc
        self._fsync_directory(target.parent)
        return Artifact(key=key, path=target, manifest=manifest)

    def _validate_staging_directory(self, temp_dir: Path) -> None:
        if temp_dir.is_symlink() or not temp_dir.is_dir():
            raise ArtifactError(f"Artifact staging directory is missing or invalid: {temp_dir}")
        try:
            parent = temp_dir.resolve().parent
            staging_root = self.staging_root.resolve()
        except OSError as exc:
            raise ArtifactError(
                f"Cannot resolve artifact staging directory {temp_dir}: {exc}"
            ) from exc
        if parent != staging_root:
            raise ArtifactError(
                f"Artifact staging directory must be a direct child of {self.staging_root}: {temp_dir}"
            )
        if (temp_dir / _MANIFEST_NAME).exists() or (temp_dir / _MANIFEST_NAME).is_symlink():
            raise ArtifactError(f"Staging directory already contains reserved {_MANIFEST_NAME}")

    @staticmethod
    def _stored_file_paths(path: Path, *, allow_manifest: bool = True) -> set[str]:
        paths: set[str] = set()
        for directory, directory_names, file_names in os.walk(path, followlinks=False):
            directory_path = Path(directory)
            for name in directory_names:
                child = directory_path / name
                if child.is_symlink():
                    relative = child.relative_to(path).as_posix()
                    raise ArtifactError(f"Artifact contains a symlink: {relative!r}")
            for name in file_names:
                child = directory_path / name
                relative = child.relative_to(path).as_posix()
                if child.is_symlink():
                    raise ArtifactError(f"Artifact contains a symlink: {relative!r}")
                if not child.is_file():
                    raise ArtifactError(f"Artifact contains a non-regular file: {relative!r}")
                if relative == _MANIFEST_NAME:
                    if allow_manifest:
                        continue
                    raise ArtifactError(f"Staging directory contains reserved {_MANIFEST_NAME}")
                _canonical_relative_path(relative)
                paths.add(relative)
        return paths

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise ArtifactError(f"Cannot write artifact manifest {path}: {exc}") from exc

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)
