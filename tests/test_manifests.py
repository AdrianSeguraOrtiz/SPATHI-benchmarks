from __future__ import annotations

import json
from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.manifests import (
    datasets_from_benchmark,
    resolve_manifest_member,
    unique_direct_manifest,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_unique_direct_manifest_requires_exactly_one_direct_child(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkConfigError, match="found 0"):
        unique_direct_manifest(tmp_path, "run_report.json")

    expected = tmp_path / "run-a" / "run_report.json"
    _write_json(expected, {})
    nested = tmp_path / "ignored" / "nested" / "run_report.json"
    _write_json(nested, {})
    assert unique_direct_manifest(tmp_path, "run_report.json") == expected.resolve()

    _write_json(tmp_path / "run-b" / "run_report.json", {})
    with pytest.raises(BenchmarkConfigError, match="found 2"):
        unique_direct_manifest(tmp_path, "run_report.json")


def test_resolve_manifest_member_confines_member_to_package(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    member = package / "dataset" / "dataset-manifest.json"
    _write_json(member, {})

    assert (
        resolve_manifest_member(package, "dataset/dataset-manifest.json", "dataset")
        == member.resolve()
    )
    with pytest.raises(BenchmarkConfigError, match="non-empty relative path"):
        resolve_manifest_member(package, "", "dataset")
    with pytest.raises(BenchmarkConfigError, match="canonical relative path"):
        resolve_manifest_member(package, "../outside.json", "dataset")
    with pytest.raises(BenchmarkConfigError, match="does not exist"):
        resolve_manifest_member(package, "missing.json", "dataset")


def test_resolve_manifest_member_rejects_symlinked_file_and_ancestor(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    external_file = outside / "manifest.json"
    _write_json(external_file, {})

    (package / "linked-file.json").symlink_to(external_file)
    with pytest.raises(BenchmarkConfigError, match="symlink"):
        resolve_manifest_member(package, "linked-file.json", "member")

    (package / "linked-directory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(BenchmarkConfigError, match="symlink"):
        resolve_manifest_member(package, "linked-directory/manifest.json", "member")


def test_unique_direct_manifest_rejects_symlinked_manifest(tmp_path: Path) -> None:
    output = tmp_path / "output"
    run = output / "run-a"
    run.mkdir(parents=True)
    external = tmp_path / "external.json"
    _write_json(external, {})
    (run / "run_report.json").symlink_to(external)

    with pytest.raises(BenchmarkConfigError, match="symlink"):
        unique_direct_manifest(output, "run_report.json")


@pytest.mark.parametrize(
    "raw_path",
    ["/absolute/member.json", "nested/./member.json", "nested\\member.json"],
)
def test_resolve_manifest_member_requires_canonical_relative_posix_path(
    tmp_path: Path,
    raw_path: str,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    with pytest.raises(BenchmarkConfigError, match="canonical|relative"):
        resolve_manifest_member(package, raw_path, "member")


def test_datasets_from_benchmark_resolves_handoff_paths(tmp_path: Path) -> None:
    package = tmp_path / "benchmark"
    dataset_manifest = package / "dataset-a" / "dataset-manifest.json"
    truth_manifest = package / "dataset-a" / "ground-truth-manifest.json"
    _write_json(dataset_manifest, {})
    _write_json(truth_manifest, {})
    manifest_path = package / "benchmark-manifest.json"
    _write_json(
        manifest_path,
        {
            "datasets": [
                {
                    "dataset_id": "dataset-a",
                    "simulator_id": "dyngen",
                    "seed": 1729,
                    "dataset_manifest": "dataset-a/dataset-manifest.json",
                    "ground_truth_manifest": "dataset-a/ground-truth-manifest.json",
                }
            ]
        },
    )

    assert datasets_from_benchmark(manifest_path, "dyngen_smoke") == [
        {
            "dataset_id": "dataset-a",
            "generation_job": "dyngen_smoke",
            "simulator_id": "dyngen",
            "seed": 1729,
            "dataset_manifest": str(dataset_manifest.resolve()),
            "ground_truth_manifest": str(truth_manifest.resolve()),
        }
    ]


def test_datasets_from_benchmark_rejects_duplicate_ids(tmp_path: Path) -> None:
    package = tmp_path / "benchmark"
    dataset_manifest = package / "dataset-manifest.json"
    truth_manifest = package / "ground-truth-manifest.json"
    _write_json(dataset_manifest, {})
    _write_json(truth_manifest, {})
    manifest_path = package / "benchmark-manifest.json"
    entry = {
        "dataset_id": "duplicate",
        "dataset_manifest": "dataset-manifest.json",
        "ground_truth_manifest": "ground-truth-manifest.json",
    }
    _write_json(manifest_path, {"datasets": [entry, entry]})

    with pytest.raises(BenchmarkConfigError, match="Invalid or duplicate dataset_id"):
        datasets_from_benchmark(manifest_path, "job")


@pytest.mark.parametrize("datasets", [None, [], ["not-an-object"]])
def test_datasets_from_benchmark_rejects_invalid_dataset_arrays(
    tmp_path: Path,
    datasets: object,
) -> None:
    manifest_path = tmp_path / "benchmark-manifest.json"
    _write_json(manifest_path, {"datasets": datasets})

    message = "has no datasets" if not datasets else "entries must be objects"
    with pytest.raises(BenchmarkConfigError, match=message):
        datasets_from_benchmark(manifest_path, "job")
