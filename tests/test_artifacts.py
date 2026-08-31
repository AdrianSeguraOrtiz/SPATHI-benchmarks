from __future__ import annotations

import errno
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from spathi_bench.artifacts import (
    ArtifactError,
    ArtifactKey,
    ArtifactStore,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
)


def test_canonical_json_hash_is_order_independent_and_strict() -> None:
    first = {"z": [1, True, None], "a": {"unicode": "á"}}
    second = {"a": {"unicode": "á"}, "z": [1, True, None]}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    assert canonical_json_sha256(first) == hashlib.sha256(canonical_json_bytes(first)).hexdigest()

    with pytest.raises(ArtifactError, match="Non-finite"):
        canonical_json_bytes({"bad": float("nan")})
    with pytest.raises(ArtifactError, match="not a string"):
        canonical_json_bytes({1: "ambiguous"})


def test_sha256_file_streams_file_contents(tmp_path: Path) -> None:
    payload = (b"benchmark\x00" * 200_000) + b"end"
    path = tmp_path / "large.bin"
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=97) == hashlib.sha256(payload).hexdigest()


def test_artifact_key_includes_kind_and_validates_identifiers() -> None:
    spec = {"dataset": "dyngen-1", "configuration": {"trees": 500}}

    assert ArtifactKey.from_spec("inference", spec) == ArtifactKey.from_spec(
        "inference", {"configuration": {"trees": 500}, "dataset": "dyngen-1"}
    )
    assert ArtifactKey.from_spec("inference", spec) != ArtifactKey.from_spec("evaluation", spec)
    assert ArtifactKey.from_spec("inference", spec).value.startswith("sha256:")
    with pytest.raises(ArtifactError, match="Invalid artifact kind"):
        ArtifactKey.from_spec("../inference", spec)


def test_publish_lookup_and_manifest_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    spec = {"dataset_fingerprint": "abc", "tool": {"id": "spathi", "seed": 1729}}
    key = store.key_for("inference", spec)

    with store.staging_directory(key) as staging:
        (staging / "nested").mkdir()
        (staging / "nested" / "network.csv").write_text(
            "source,target,weight\nTF1,G1,0.8\n", encoding="utf-8"
        )
        (staging / "metrics.json").write_text('{"runtime":1.2}\n', encoding="utf-8")
        artifact = store.publish_from_temp(staging, kind="inference", spec=spec)

    expected_path = store.path_for(key)
    assert artifact.path == expected_path
    assert artifact.key == key
    assert store.lookup(key) == artifact
    assert not list(store.staging_root.glob("*"))
    assert artifact.file("nested/network.csv").read_text(encoding="utf-8").startswith("source")

    manifest = json.loads((expected_path / "artifact.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "inference"
    assert manifest["key"] == key.value
    assert manifest["spec"] == spec
    assert manifest["status"] == "complete"
    assert [entry["path"] for entry in manifest["files"]] == [
        "metrics.json",
        "nested/network.csv",
    ]
    assert all(entry["status"] == "complete" for entry in manifest["files"])
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert all(isinstance(entry["size"], int) for entry in manifest["files"])


def test_publication_is_idempotent_for_an_existing_complete_key(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    spec = {"run": "same"}
    key = store.key_for("dataset", spec)
    with store.staging_directory(key) as staging:
        (staging / "data.txt").write_text("first", encoding="utf-8")
        first = store.publish_from_temp(staging, kind="dataset", spec=spec)

    with store.staging_directory(key) as staging:
        (staging / "data.txt").write_text("second", encoding="utf-8")
        second = store.publish_from_temp(staging, kind="dataset", spec=spec)

    assert second == first
    assert second.file("data.txt").read_text(encoding="utf-8") == "first"


def test_concurrent_atomic_publisher_winner_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    spec = {"run": "race"}
    key = store.key_for("dataset", spec)

    def winning_race(source: Path, target: Path) -> None:
        shutil.copytree(source, target)
        raise OSError(errno.ENOTEMPTY, "another publisher won")

    monkeypatch.setattr("spathi_bench.artifacts.os.rename", winning_race)
    with store.staging_directory(key) as staging:
        (staging / "data.txt").write_text("stable", encoding="utf-8")
        artifact = store.publish_from_temp(staging, kind="dataset", spec=spec)

    assert artifact == store.lookup(key)
    assert artifact.file("data.txt").read_text(encoding="utf-8") == "stable"


def test_publish_requires_store_owned_temp_and_exact_file_list(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    spec = {"run": 1}
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    with pytest.raises(ArtifactError, match="direct child"):
        store.publish_from_temp(foreign, kind="dataset", spec=spec)

    key = store.key_for("dataset", spec)
    with store.staging_directory(key) as staging:
        (staging / "a.txt").write_text("a", encoding="utf-8")
        (staging / "b.txt").write_text("b", encoding="utf-8")
        with pytest.raises(ArtifactError, match="exactly match"):
            store.publish_from_temp(staging, kind="dataset", spec=spec, files=["a.txt"])


def test_validation_rejects_path_escape(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    spec = {"run": "escape"}
    key = store.key_for("dataset", spec)
    artifact_path = store.path_for(key)
    artifact_path.mkdir(parents=True)
    (artifact_path / "artifact.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": key.kind,
                "key": key.value,
                "spec": spec,
                "files": [
                    {
                        "path": "../outside.txt",
                        "status": "complete",
                        "sha256": "0" * 64,
                        "size": 0,
                    }
                ],
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="canonical and relative"):
        store.lookup(key)


@pytest.mark.parametrize("mutation", ["missing", "content", "size"])
def test_validation_rejects_missing_or_divergent_file(tmp_path: Path, mutation: str) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    spec = {"run": mutation}
    key = store.key_for("inference", spec)
    with store.staging_directory(key) as staging:
        (staging / "network.tsv").write_text("TF1\tG1\t0.5\n", encoding="utf-8")
        artifact = store.publish_from_temp(staging, kind="inference", spec=spec)

    member = artifact.path / "network.tsv"
    if mutation == "missing":
        member.unlink()
    elif mutation == "content":
        member.write_text("TF2\tG2\t0.5\n", encoding="utf-8")
    else:
        manifest_path = artifact.path / "artifact.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][0]["size"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    expected = "missing|file set differs" if mutation == "missing" else "size differs|hash differs"
    with pytest.raises(ArtifactError, match=expected):
        store.lookup(key)


def test_lookup_returns_none_only_when_artifact_is_absent(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    key = store.key_for("evaluation", {"input": "unknown"})

    assert store.lookup(key) is None

    path = store.path_for(key)
    path.mkdir(parents=True)
    (path / "artifact.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": key.kind,
                "key": key.value,
                "spec": {"input": "unknown"},
                "files": [],
                "status": "building",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="not complete"):
        store.lookup(key)
