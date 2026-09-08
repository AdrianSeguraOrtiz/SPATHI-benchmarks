from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.publication_boundary import validate_publication_boundary


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("work/\n", encoding="utf-8")
    (root / "design.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "input_path": "experiments/public/request.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_publication_boundary_accepts_safe_versionable_inputs_and_ignores_work(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    ignored = root / "work/runtime/input.h5"
    ignored.parent.mkdir(parents=True)
    ignored.write_bytes(b"runtime-only")

    report = validate_publication_boundary(root)

    assert report["status"] == "pass"
    assert report["scope"] == "index_blobs_and_nonignored_worktree_files"
    assert report["files_checked"] == 2
    assert report["representations_checked"] == 2


def test_publication_boundary_rejects_a_versionable_raw_scientific_container(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    (root / "input.h5").write_bytes(b"not-public")

    with pytest.raises(BenchmarkConfigError, match="raw scientific data"):
        validate_publication_boundary(root)


def test_publication_boundary_rejects_binary_content_with_a_disguised_suffix(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    (root / "apparently-safe.txt").write_bytes(b"\xff\xfe\x00\x01")

    with pytest.raises(BenchmarkConfigError, match="UTF-8 source and design files only"):
        validate_publication_boundary(root)


def test_publication_boundary_rejects_symlinked_content(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (root / "linked.txt").symlink_to(outside)

    with pytest.raises(BenchmarkConfigError, match="symlinked content"):
        validate_publication_boundary(root)


def test_publication_boundary_rejects_an_external_json_path(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "design.json").write_text(
        json.dumps({"input_path": ".." + "/outside.tsv"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkConfigError, match="external path"):
        validate_publication_boundary(root)


def test_publication_boundary_rejects_a_generic_disclosure_marker(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    marker = "confi" + "dential"
    (root / "notes.txt").write_text(f"{marker} material\n", encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="restricted disclosure marker"):
        validate_publication_boundary(root)


def test_publication_boundary_rejects_source_provenance_metadata(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    restricted_key = "source_dataset_" + "location"
    (root / "design.json").write_text(
        json.dumps({restricted_key: "inputs/source.tsv"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkConfigError, match="restricted metadata key"):
        validate_publication_boundary(root)


def test_publication_boundary_rejects_an_unapproved_utf8_table(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / "cell-annotations.tsv").write_text(
        "cell\ttype\tstate\ncell-1\tB\tmature\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkConfigError, match="unapproved text table"):
        validate_publication_boundary(root)


def test_publication_boundary_validates_the_staged_blob_not_only_the_worktree(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    staged_payload = {"input_path": ".." + "/outside.tsv"}
    (root / "design.json").write_text(json.dumps(staged_payload) + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "design.json"], check=True)
    (root / "design.json").write_text(
        json.dumps({"input_path": "experiments/public/request.json"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkConfigError, match="external path"):
        validate_publication_boundary(root)


def test_publication_boundary_validates_the_worktree_when_the_index_is_safe(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    subprocess.run(["git", "-C", str(root), "add", "design.json"], check=True)
    (root / "design.json").write_text(
        json.dumps({"input_path": ".." + "/outside.tsv"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkConfigError, match="external path"):
        validate_publication_boundary(root)
