from __future__ import annotations

import json
from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.safe_io import (
    atomic_write_json,
    unique_temporary_file,
)


def test_json_writer_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    boundary = tmp_path / "work"
    real_parent = boundary / "real"
    real_parent.mkdir(parents=True)
    linked_parent = boundary / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(BenchmarkConfigError, match="contains a symlink"):
        atomic_write_json(
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
