from __future__ import annotations

from pathlib import Path

import pytest

from spathi_bench import environment_lock
from spathi_bench.environment_lock import (
    EnvironmentLockError,
    load_environment_lock,
    validate_environment_lock,
)


def _write_lock(root: Path, text: str) -> None:
    path = root / "constraints" / "benchmark.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_environment_lock_accepts_only_unique_exact_pins(tmp_path: Path) -> None:
    _write_lock(tmp_path, "# frozen\nNumPy==2.5.2\njsonschema==4.26.0\n")

    assert load_environment_lock(tmp_path) == {
        "jsonschema": "4.26.0",
        "numpy": "2.5.2",
    }

    _write_lock(tmp_path, "numpy>=2\n")
    with pytest.raises(EnvironmentLockError, match="exact name==version"):
        load_environment_lock(tmp_path)

    _write_lock(tmp_path, "some.package==2.5.2\nsome_package==2.5.2\n")
    with pytest.raises(EnvironmentLockError, match="Duplicate"):
        load_environment_lock(tmp_path)


def test_validate_environment_lock_rejects_missing_and_mismatched_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_lock(tmp_path, "present==1.0\nwrong==2.0\nmissing==3.0\n")

    def version(name: str) -> str:
        if name == "missing":
            raise environment_lock.metadata.PackageNotFoundError(name)
        return {"present": "1.0", "wrong": "9.0"}[name]

    monkeypatch.setattr(environment_lock.metadata, "version", version)
    with pytest.raises(EnvironmentLockError, match=r"missing missing.*wrong"):
        validate_environment_lock(tmp_path)


def test_repository_environment_matches_the_committed_lock() -> None:
    root = Path(__file__).resolve().parents[1]
    locked = validate_environment_lock(root)
    assert locked["numpy"]
    assert locked["pytest"]
