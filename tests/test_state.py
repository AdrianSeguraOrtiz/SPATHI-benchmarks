from __future__ import annotations

import json
from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.state import (
    load_state,
    require_status,
    run_lock,
    state_path,
    status_at_least,
    write_state,
)


def test_state_round_trip_is_atomic_and_stable(tmp_path: Path) -> None:
    state = {
        "schema_version": "2.0",
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
        {"schema_version": "2.0", "profile_id": "smoke", "status": "generated"},
    )
    replacement = {
        "schema_version": "2.0",
        "profile_id": "smoke",
        "status": "inferred",
    }
    write_state(run_root, replacement)

    assert json.loads(state_path(run_root).read_text(encoding="utf-8")) == replacement


def test_load_state_rejects_wrong_schema_and_profile(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    write_state(
        run_root,
        {"schema_version": "1.0", "profile_id": "smoke", "status": "generated"},
    )
    with pytest.raises(BenchmarkConfigError, match="Unsupported benchmark state schema"):
        load_state(run_root, "smoke")

    write_state(
        run_root,
        {"schema_version": "2.0", "profile_id": "reduced", "status": "generated"},
    )
    with pytest.raises(BenchmarkConfigError, match="State profile mismatch"):
        load_state(run_root, "smoke")


def test_status_order_supports_idempotent_stage_checks() -> None:
    state = {"status": "validated"}

    assert status_at_least(state, "preflighted") is True
    assert status_at_least(state, "validated") is True
    assert status_at_least(state, "evaluated") is False

    with pytest.raises(BenchmarkConfigError, match="Unknown benchmark state status"):
        status_at_least({"status": "unknown"}, "generated")
    with pytest.raises(ValueError, match="Unknown expected benchmark status"):
        status_at_least(state, "unknown")


def test_require_status_requires_the_exact_predecessor() -> None:
    require_status({"status": "evaluated"}, "evaluated", "Summarization")

    with pytest.raises(
        BenchmarkConfigError,
        match="Summarization requires benchmark status 'evaluated', found 'completed'",
    ):
        require_status({"status": "completed"}, "evaluated", "Summarization")


def test_run_lock_rejects_concurrent_mutation_of_one_run(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    with run_lock(run_root):
        with pytest.raises(BenchmarkConfigError, match="already running"):
            with run_lock(run_root):
                pass

    with run_lock(run_root):
        assert (run_root / ".run.lock").is_file()
