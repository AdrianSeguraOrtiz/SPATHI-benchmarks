from __future__ import annotations

import json
from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.state import load_state, state_path, write_state


def test_state_round_trip_is_atomic_and_stable(tmp_path: Path) -> None:
    state = {
        "schema_version": "1.0",
        "profile_id": "smoke",
        "status": "generated",
        "datasets": [],
    }

    written = write_state(tmp_path / "run", state)

    assert written == state_path(tmp_path / "run")
    assert load_state(tmp_path / "run", "smoke") == state
    assert not (tmp_path / "run" / ".state.json.tmp").exists()
    assert written.read_text(encoding="utf-8").endswith("\n")


def test_write_state_replaces_prior_state(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_state(
        run_root,
        {"schema_version": "1.0", "profile_id": "smoke", "status": "generated"},
    )
    replacement = {
        "schema_version": "1.0",
        "profile_id": "smoke",
        "status": "inferred",
    }
    write_state(run_root, replacement)

    assert json.loads(state_path(run_root).read_text(encoding="utf-8")) == replacement


def test_load_state_rejects_wrong_schema_and_profile(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_state(
        run_root,
        {"schema_version": "2.0", "profile_id": "smoke", "status": "generated"},
    )
    with pytest.raises(BenchmarkConfigError, match="Unsupported benchmark state schema"):
        load_state(run_root, "smoke")

    write_state(
        run_root,
        {"schema_version": "1.0", "profile_id": "reduced", "status": "generated"},
    )
    with pytest.raises(BenchmarkConfigError, match="State profile mismatch"):
        load_state(run_root, "smoke")
