from __future__ import annotations

import hashlib
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from spathi_bench.artifacts import canonical_json_sha256, sha256_file
from spathi_bench.export import (
    AnalysisExportError,
    export_analysis_resolution,
)


@dataclass(frozen=True)
class _Spec:
    id: str = "analysis"
    label: str = "Analysis Bundle"
    purpose: str = "Strict downstream handoff."
    intended_downstream_commands: tuple[str, ...] = ("evaluate-inference",)


@dataclass(frozen=True)
class _Source:
    virtual_path: str
    source_path: Path


@dataclass(frozen=True)
class _Resolution:
    root: Path
    sources: tuple[_Source, ...]
    spec: _Spec = field(default_factory=_Spec)
    missing_required: tuple[str, ...] = ()
    skipped_optional: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return not self.missing_required


def _resolution(root: Path, members: dict[str, bytes]) -> _Resolution:
    sources: list[_Source] = []
    for index, (virtual_path, content) in enumerate(members.items()):
        source = root / "physical" / f"source-{index}.bin"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
        sources.append(_Source(virtual_path, source))
    return _Resolution(root=root, sources=tuple(reversed(sources)))


def test_analysis_zip_is_reproducible_and_preserves_exact_virtual_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    resolution = _resolution(
        root,
        {
            "truth/network.csv": b"source,target\nTF1,G1\n",
            "ground-truth-manifest.json": b'{"schema_version":"1.0"}\n',
            "extras/tf_list.txt": "TF1\nTF-β\n".encode(),
        },
    )
    output = tmp_path / "exports" / "dataset-analysis.zip"

    first = export_analysis_resolution(resolution, output)
    first_bytes = output.read_bytes()
    first_manifest_bytes = first.manifest_path.read_bytes()
    for source in resolution.sources:
        os.utime(source.source_path, (1_900_000_000, 1_900_000_000))
        source.source_path.chmod(0o600)
    second = export_analysis_resolution(resolution, output)

    assert output.read_bytes() == first_bytes
    assert second.manifest_path.read_bytes() == first_manifest_bytes
    assert first.zip_sha256 == second.zip_sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert first.manifest_sha256 == sha256_file(first.manifest_path)
    assert first.manifest["source_set_sha256"] == canonical_json_sha256(first.manifest["files"])
    assert first.manifest["archive"] == {
        "filename": "dataset-analysis.zip",
        "sha256": first.zip_sha256,
        "size": len(first_bytes),
    }

    expected_paths = sorted(
        [
            "truth/network.csv",
            "ground-truth-manifest.json",
            "extras/tf_list.txt",
        ]
    )
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == expected_paths
        assert archive.read("extras/tf_list.txt") == "TF1\nTF-β\n".encode()
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_DEFLATED
            assert info.create_system == 3
            assert (info.external_attr >> 16) & 0o777 == 0o644


def test_export_rejects_an_unavailable_official_resolution(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    resolution = _Resolution(
        root=root,
        sources=(),
        missing_required=("ground-truth-manifest.json", "truth/networks.csv"),
    )
    output = tmp_path / "analysis.zip"

    with pytest.raises(
        AnalysisExportError,
        match="ground-truth-manifest.json, truth/networks.csv",
    ):
        export_analysis_resolution(resolution, output)

    assert not output.exists()
    assert not (tmp_path / "analysis.zip.manifest.json").exists()


@pytest.mark.parametrize(
    "virtual_path",
    ["../escape.txt", "/absolute.txt", "a\\b.txt", "a//b.txt", "a/./b.txt", " bad"],
)
def test_export_rejects_unsafe_virtual_paths(
    tmp_path: Path,
    virtual_path: str,
) -> None:
    root = tmp_path / "source"
    source = root / "member.txt"
    source.parent.mkdir()
    source.write_text("safe\n", encoding="utf-8")
    resolution = _Resolution(root=root, sources=(_Source(virtual_path, source),))

    with pytest.raises(AnalysisExportError, match="[Vv]irtual path|Unsafe"):
        export_analysis_resolution(resolution, tmp_path / "analysis.zip")


def test_export_rejects_duplicate_paths_and_sources_outside_the_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    inside = root / "inside.txt"
    outside = tmp_path / "outside.txt"
    inside.parent.mkdir()
    inside.write_text("inside\n", encoding="utf-8")
    outside.write_text("outside\n", encoding="utf-8")

    duplicates = _Resolution(
        root=root,
        sources=(
            _Source("same.txt", inside),
            _Source("same.txt", inside),
        ),
    )
    with pytest.raises(AnalysisExportError, match="Duplicate"):
        export_analysis_resolution(duplicates, tmp_path / "duplicates.zip")

    escaping = _Resolution(
        root=root,
        sources=(_Source("outside.txt", outside),),
    )
    with pytest.raises(AnalysisExportError, match="escapes or is missing"):
        export_analysis_resolution(escaping, tmp_path / "escaping.zip")


def test_export_rejects_non_analysis_resolution_and_empty_resolution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()

    with pytest.raises(AnalysisExportError, match="Only ANDREA analysis"):
        export_analysis_resolution(
            _Resolution(root=root, sources=(), spec=_Spec(id="full")),
            tmp_path / "full.zip",
        )
    with pytest.raises(AnalysisExportError, match="contains no files"):
        export_analysis_resolution(
            _Resolution(root=root, sources=()),
            tmp_path / "empty.zip",
        )
