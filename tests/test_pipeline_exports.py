from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.artifacts import Artifact, ArtifactStore, sha256_file
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.pipeline import _materialize_exports


def _reference(artifact: Artifact) -> dict[str, Any]:
    return {
        "kind": artifact.key.kind,
        "digest": artifact.key.digest,
        "key": artifact.key.value,
        "path": str(artifact.path),
        "cache_hit": False,
    }


def _source_artifact(store: ArtifactStore) -> Artifact:
    spec = {"source": "dataset-a"}
    key = store.key_for("source", spec)
    with store.staging_directory(key) as staging:
        (staging / "source.txt").write_text("source\n", encoding="utf-8")
        return store.publish_from_temp(staging, kind="source", spec=spec)


def _bundle_artifact(store: ArtifactStore, identity: str) -> Artifact:
    spec = {"bundle": identity}
    key = store.key_for("analysis-bundle", spec)
    with store.staging_directory(key) as staging:
        archive = staging / "analysis.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("payload.txt", identity)
        manifest = {
            "schema_version": "1.0",
            "status": "complete",
            "bundle": {"id": "analysis"},
            "archive": {
                "filename": "analysis.zip",
                "sha256": sha256_file(archive),
                "size": archive.stat().st_size,
            },
        }
        (staging / "analysis.zip.manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        return store.publish_from_temp(staging, kind="analysis-bundle", spec=spec)


def test_materialized_exports_are_stable_idempotent_and_tamper_evident(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    store = ArtifactStore(tmp_path / "artifacts")
    source = _source_artifact(store)
    dataset_bundle = _bundle_artifact(store, "dataset")
    joint_bundle = _bundle_artifact(store, "joint-inference")
    bundles = [
        {
            "producer_command": "generate-data",
            "scope": "dataset",
            "dataset_id": "dataset-a",
            "source_artifact": _reference(source),
            "artifact": _reference(dataset_bundle),
        },
        {
            "producer_command": "infer-network",
            "scope": "joint",
            "dataset_id": "dataset-a",
            "regime": "group_only",
            "source_artifact": _reference(source),
            "artifact": _reference(joint_bundle),
        },
    ]

    manifest_path = _materialize_exports(
        run_root=run_root,
        profile_id="smoke",
        bundles=bundles,
        store=store,
    )
    first_hash = sha256_file(manifest_path)

    assert (run_root / "exports/datasets/dataset-a/analysis.zip").is_file()
    assert (
        run_root / "exports/inference/dataset-a/group_only/joint/analysis.zip"
    ).is_file()
    assert all(Path(bundle["export_zip"]).is_file() for bundle in bundles)
    assert (
        _materialize_exports(
            run_root=run_root,
            profile_id="smoke",
            bundles=bundles,
            store=store,
        )
        == manifest_path
    )
    assert sha256_file(manifest_path) == first_hash

    (run_root / "exports/datasets/dataset-a/analysis.zip").write_bytes(b"tampered")
    with pytest.raises(BenchmarkConfigError, match="differs from the immutable bundle set"):
        _materialize_exports(
            run_root=run_root,
            profile_id="smoke",
            bundles=bundles,
            store=store,
        )
