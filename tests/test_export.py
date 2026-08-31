from __future__ import annotations

import hashlib
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.artifacts import canonical_json_sha256, sha256_file
from spathi_bench.export import (
    AnalysisExportError,
    export_analysis_resolution,
    export_evaluate_inference_analysis,
    export_generate_data_analysis,
    export_infer_network_analysis,
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


def test_command_wrappers_use_the_official_resolver_signatures(tmp_path: Path) -> None:
    root = tmp_path / "source"
    resolution = _resolution(root, {"handoff.json": b"{}\n"})
    calls: list[dict[str, Any]] = []

    def resolver(**kwargs: Any) -> _Resolution:
        calls.append(kwargs)
        return resolution

    generated = export_generate_data_analysis(
        benchmark_root=root,
        dataset_id="dataset_01",
        output_zip=tmp_path / "dataset-analysis.zip",
        resolver=resolver,
    )
    inferred = export_infer_network_analysis(
        run_dir=root,
        output_zip=tmp_path / "inference-analysis.zip",
        resolver=resolver,
    )
    evaluated = export_evaluate_inference_analysis(
        evaluation_dir=root,
        output_zip=tmp_path / "evaluation-analysis.zip",
        resolver=resolver,
    )

    assert calls == [
        {
            "bundle_id": "analysis",
            "benchmark_root": root,
            "dataset_id": "dataset_01",
        },
        {"bundle_id": "analysis", "run_dir": root},
        {"bundle_id": "analysis", "evaluation_dir": root},
    ]
    assert generated.manifest["bundle"]["producer_command"] == "generate-data"
    assert generated.manifest["bundle"]["dataset_id"] == "dataset_01"
    assert inferred.manifest["bundle"]["producer_command"] == "infer-network"
    assert evaluated.manifest["bundle"]["producer_command"] == "evaluate-inference"
    for result in (generated, inferred, evaluated):
        assert result.zip_path.is_file()
        assert result.manifest_path.is_file()


def test_wrapper_resolver_import_is_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import spathi_bench.export as export_module

    root = tmp_path / "source"
    resolution = _resolution(root, {"evaluation_report.json": b"{}\n"})
    imported: list[str] = []

    class _Module:
        @staticmethod
        def resolve_bundle(**kwargs: Any) -> _Resolution:
            assert kwargs == {"bundle_id": "analysis", "evaluation_dir": root}
            return resolution

    def fake_import(name: str) -> _Module:
        imported.append(name)
        return _Module()

    monkeypatch.setattr(export_module, "import_module", fake_import)

    export_evaluate_inference_analysis(
        evaluation_dir=root,
        output_zip=tmp_path / "evaluation-analysis.zip",
    )

    assert imported == ["andrea.core.commands.evaluate_inference.bundles"]
