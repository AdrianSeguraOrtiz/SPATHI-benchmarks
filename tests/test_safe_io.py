from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.safe_io import (
    atomic_write_json,
    unique_temporary_file,
    write_idempotent_json,
)


@pytest.mark.parametrize("writer", [atomic_write_json, write_idempotent_json])
def test_json_writers_reject_a_symlinked_parent(
    tmp_path: Path,
    writer: object,
) -> None:
    boundary = tmp_path / "work"
    real_parent = boundary / "real"
    real_parent.mkdir(parents=True)
    linked_parent = boundary / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(BenchmarkConfigError, match="contains a symlink"):
        writer(  # type: ignore[operator]
            linked_parent / "result.json",
            {"status": "complete"},
            "result",
            boundary=boundary,
        )

    assert not (real_parent / "result.json").exists()


def test_atomic_json_writer_preserves_mutable_replacement_semantics(tmp_path: Path) -> None:
    target = tmp_path / "state.json"

    atomic_write_json(target, {"status": "running"}, "state", boundary=tmp_path)
    atomic_write_json(target, {"status": "complete"}, "state", boundary=tmp_path)

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "complete"}


def test_idempotent_json_publication_is_concurrency_safe(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    payload = {"status": "complete", "value": [1, 2, 3]}

    with ThreadPoolExecutor(max_workers=16) as executor:
        paths = list(
            executor.map(
                lambda _index: write_idempotent_json(
                    target,
                    payload,
                    "result",
                    boundary=tmp_path,
                ),
                range(64),
            )
        )

    assert paths == [target] * 64
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(f".{target.name}-*"))


def test_idempotent_json_publication_never_overwrites_different_content(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.json"
    write_idempotent_json(target, {"winner": "first"}, "result", boundary=tmp_path)

    with pytest.raises(BenchmarkConfigError, match="Existing result differs"):
        write_idempotent_json(target, {"winner": "second"}, "result", boundary=tmp_path)

    assert json.loads(target.read_text(encoding="utf-8")) == {"winner": "first"}


def test_unique_temporary_file_is_exclusive_and_removed_after_failure(tmp_path: Path) -> None:
    first_path: Path | None = None
    with pytest.raises(RuntimeError, match="forced failure"):
        with unique_temporary_file(
            tmp_path,
            "binary staging file",
            boundary=tmp_path,
            prefix=".raw-",
            suffix=".bin",
        ) as temporary:
            first_path = temporary
            assert temporary.is_file()
            temporary.write_bytes(b"counts")
            with unique_temporary_file(
                tmp_path,
                "second binary staging file",
                boundary=tmp_path,
                prefix=".raw-",
                suffix=".bin",
            ) as second:
                assert second != temporary
            raise RuntimeError("forced failure")

    assert first_path is not None and not first_path.exists()
    assert not list(tmp_path.glob(".raw-*.bin"))
