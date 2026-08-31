"""Deterministic exports of ANDREA ``analysis`` handoff bundles.

ANDREA owns the bundle contract: its command-specific resolvers decide which
files belong in an analysis ZIP and which virtual paths they must have.  This
module only validates a resolved contract and serializes it reproducibly.  The
resolver imports are deliberately lazy so importing :mod:`spathi_bench` does
not require ANDREA to be installed in the benchmark orchestrator environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .artifacts import canonical_json_sha256, sha256_file

_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_COPY_CHUNK_SIZE = 1024 * 1024


class AnalysisExportError(ValueError):
    """Raised when an analysis resolution cannot be exported safely."""


class _BundleSpec(Protocol):
    id: str
    label: str
    purpose: str
    intended_downstream_commands: Sequence[str]


class _BundleSource(Protocol):
    virtual_path: str
    source_path: Path


class BundleResolutionLike(Protocol):
    """Structural type implemented by ANDREA's ``BundleResolution``."""

    spec: _BundleSpec
    root: Path
    sources: Sequence[_BundleSource]
    missing_required: Sequence[str]
    skipped_optional: Sequence[str]

    @property
    def available(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class AnalysisBundleExport:
    """Files and content identities produced by one analysis export."""

    zip_path: Path
    manifest_path: Path
    zip_sha256: str
    manifest_sha256: str
    size: int
    manifest: Mapping[str, Any]


Resolver = Callable[..., BundleResolutionLike]


def _safe_virtual_path(raw_path: object) -> str:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or raw_path != raw_path.strip()
        or "\\" in raw_path
        or any(ord(character) < 32 for character in raw_path)
    ):
        raise AnalysisExportError("Bundle virtual paths must be non-empty canonical POSIX paths")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw_path
        or any(part in {"", ".", ".."} for part in raw_path.split("/"))
    ):
        raise AnalysisExportError(f"Unsafe or non-canonical bundle virtual path: {raw_path!r}")
    return raw_path


def _validated_sources(
    resolution: BundleResolutionLike,
    *,
    output_zip: Path,
    manifest_path: Path,
) -> list[tuple[str, Path]]:
    if getattr(resolution.spec, "id", None) != "analysis":
        raise AnalysisExportError(
            f"Only ANDREA analysis resolutions can be exported, got "
            f"{getattr(resolution.spec, 'id', None)!r}"
        )
    missing = tuple(str(item) for item in resolution.missing_required)
    if not resolution.available or missing:
        details = ", ".join(missing) or "unknown required inputs"
        raise AnalysisExportError(f"ANDREA analysis bundle is unavailable: {details}")

    try:
        root = Path(resolution.root).resolve(strict=True)
    except OSError as exc:
        raise AnalysisExportError(f"ANDREA bundle root is unavailable: {resolution.root}") from exc
    if not root.is_dir():
        raise AnalysisExportError(f"ANDREA bundle root is not a directory: {root}")

    reserved_outputs = {
        output_zip.resolve(strict=False),
        manifest_path.resolve(strict=False),
    }
    observed: set[str] = set()
    sources: list[tuple[str, Path]] = []
    for source in resolution.sources:
        virtual_path = _safe_virtual_path(source.virtual_path)
        if virtual_path in observed:
            raise AnalysisExportError(
                f"Duplicate bundle virtual path from ANDREA resolver: {virtual_path!r}"
            )
        observed.add(virtual_path)

        source_path = Path(source.source_path)
        if source_path.is_symlink():
            raise AnalysisExportError(f"Bundle source must not be a symbolic link: {source_path}")
        try:
            resolved_source = source_path.resolve(strict=True)
            resolved_source.relative_to(root)
        except (OSError, ValueError) as exc:
            raise AnalysisExportError(
                f"Bundle source escapes or is missing from its resolver root: {source_path}"
            ) from exc
        if not resolved_source.is_file():
            raise AnalysisExportError(f"Bundle source is not a regular file: {source_path}")
        if resolved_source in reserved_outputs:
            raise AnalysisExportError(
                f"Bundle output collides with an input source: {resolved_source}"
            )
        sources.append((virtual_path, resolved_source))

    if not sources:
        raise AnalysisExportError("ANDREA analysis resolution contains no files")
    return sorted(sources, key=lambda item: item[0])


def _zip_info(virtual_path: str, *, source_size: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=virtual_path, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    info.file_size = source_size
    return info


def _write_deterministic_zip(
    output_zip: Path,
    sources: Sequence[tuple[str, Path]],
) -> list[dict[str, Any]]:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_zip.parent,
        prefix=f".{output_zip.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    files: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            archive.comment = b""
            for virtual_path, source_path in sources:
                expected_size = source_path.stat().st_size
                digest = hashlib.sha256()
                observed_size = 0
                info = _zip_info(virtual_path, source_size=expected_size)
                with (
                    source_path.open("rb") as source,
                    archive.open(
                        info,
                        mode="w",
                        force_zip64=expected_size >= zipfile.ZIP64_LIMIT,
                    ) as destination,
                ):
                    while chunk := source.read(_COPY_CHUNK_SIZE):
                        destination.write(chunk)
                        digest.update(chunk)
                        observed_size += len(chunk)
                if observed_size != expected_size:
                    raise AnalysisExportError(
                        f"Bundle source changed size while it was exported: {source_path}"
                    )
                files.append(
                    {
                        "path": virtual_path,
                        "sha256": digest.hexdigest(),
                        "size": observed_size,
                    }
                )
        os.replace(temporary, output_zip)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return files


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def export_analysis_resolution(
    resolution: BundleResolutionLike,
    output_zip: Path,
    *,
    producer_command: str | None = None,
    dataset_id: str | None = None,
    manifest_path: Path | None = None,
) -> AnalysisBundleExport:
    """Export one official ANDREA analysis resolution reproducibly.

    Virtual paths come exclusively from ``resolution.sources`` and are written
    at the ZIP root without adding a benchmark- or run-directory prefix.  A
    deterministic sidecar manifest records both archive and member hashes.
    """

    output_zip = Path(output_zip).resolve(strict=False)
    if output_zip.suffix.lower() != ".zip":
        raise AnalysisExportError(f"Analysis bundle output must end in .zip: {output_zip}")
    if output_zip.exists() and output_zip.is_dir():
        raise AnalysisExportError(f"Analysis bundle output is a directory: {output_zip}")
    resolved_manifest_path = (
        Path(manifest_path).resolve(strict=False)
        if manifest_path is not None
        else output_zip.with_name(f"{output_zip.name}.manifest.json")
    )
    if resolved_manifest_path == output_zip:
        raise AnalysisExportError("Analysis ZIP and manifest paths must be different")

    sources = _validated_sources(
        resolution,
        output_zip=output_zip,
        manifest_path=resolved_manifest_path,
    )
    files = _write_deterministic_zip(output_zip, sources)
    archive_sha256 = sha256_file(output_zip)
    archive_size = output_zip.stat().st_size
    bundle: dict[str, Any] = {
        "id": "analysis",
        "label": str(resolution.spec.label),
        "purpose": str(resolution.spec.purpose),
        "intended_downstream_commands": list(resolution.spec.intended_downstream_commands),
    }
    if producer_command is not None:
        bundle["producer_command"] = producer_command
    if dataset_id is not None:
        bundle["dataset_id"] = dataset_id
    manifest = {
        "schema_version": "1.0",
        "status": "complete",
        "bundle": bundle,
        "archive": {
            "filename": output_zip.name,
            "sha256": archive_sha256,
            "size": archive_size,
        },
        "files": files,
        "source_set_sha256": canonical_json_sha256(files),
        "skipped_optional": sorted(str(item) for item in resolution.skipped_optional),
    }
    manifest_sha256 = _write_manifest(resolved_manifest_path, manifest)
    return AnalysisBundleExport(
        zip_path=output_zip,
        manifest_path=resolved_manifest_path,
        zip_sha256=archive_sha256,
        manifest_sha256=manifest_sha256,
        size=archive_size,
        manifest=manifest,
    )


def _andrea_resolver(module_name: str) -> Resolver:
    try:
        module = import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise AnalysisExportError(
            "ANDREA is not importable; run with the locked ANDREA repository on "
            "PYTHONPATH or inject its official resolver"
        ) from exc
    resolver = getattr(module, "resolve_bundle", None)
    if not callable(resolver):
        raise AnalysisExportError(
            f"Locked ANDREA module has no callable resolve_bundle: {module_name}"
        )
    return resolver


def export_generate_data_analysis(
    *,
    benchmark_root: Path,
    dataset_id: str,
    output_zip: Path,
    manifest_path: Path | None = None,
    resolver: Resolver | None = None,
) -> AnalysisBundleExport:
    """Export one dataset-level ``generate-data`` analysis handoff."""

    official_resolver = resolver or _andrea_resolver("andrea.core.commands.generate_data.bundles")
    resolution = official_resolver(
        bundle_id="analysis",
        benchmark_root=Path(benchmark_root),
        dataset_id=dataset_id,
    )
    return export_analysis_resolution(
        resolution,
        output_zip,
        producer_command="generate-data",
        dataset_id=dataset_id,
        manifest_path=manifest_path,
    )


def export_infer_network_analysis(
    *,
    run_dir: Path,
    output_zip: Path,
    manifest_path: Path | None = None,
    resolver: Resolver | None = None,
) -> AnalysisBundleExport:
    """Export an ``infer-network`` analysis handoff."""

    official_resolver = resolver or _andrea_resolver("andrea.core.commands.infer_network.bundles")
    resolution = official_resolver(bundle_id="analysis", run_dir=Path(run_dir))
    return export_analysis_resolution(
        resolution,
        output_zip,
        producer_command="infer-network",
        manifest_path=manifest_path,
    )


def export_evaluate_inference_analysis(
    *,
    evaluation_dir: Path,
    output_zip: Path,
    manifest_path: Path | None = None,
    resolver: Resolver | None = None,
) -> AnalysisBundleExport:
    """Export an ``evaluate-inference`` analysis handoff."""

    official_resolver = resolver or _andrea_resolver(
        "andrea.core.commands.evaluate_inference.bundles"
    )
    resolution = official_resolver(
        bundle_id="analysis",
        evaluation_dir=Path(evaluation_dir),
    )
    return export_analysis_resolution(
        resolution,
        output_zip,
        producer_command="evaluate-inference",
        manifest_path=manifest_path,
    )


__all__ = [
    "AnalysisBundleExport",
    "AnalysisExportError",
    "BundleResolutionLike",
    "export_analysis_resolution",
    "export_evaluate_inference_analysis",
    "export_generate_data_analysis",
    "export_infer_network_analysis",
]
