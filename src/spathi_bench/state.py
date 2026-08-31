"""Atomic runtime state for a concrete benchmark execution."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import BenchmarkConfigError, load_json_object

STATE_SCHEMA_VERSION = "3.0"
STAGE_ORDER = {
    "preflighted": 0,
    "image_ready": 1,
    "generated": 2,
    "inferred": 3,
    "validated": 4,
    "integrated": 5,
    "evaluated": 6,
    "exported": 7,
    "completed": 8,
}


def state_path(run_root: Path) -> Path:
    return run_root / "state.json"


@contextmanager
def run_lock(run_root: Path) -> Iterator[None]:
    """Prevent two CLI stages from mutating the same concrete run concurrently."""

    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / ".run.lock"
    with lock_path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BenchmarkConfigError(
                f"Another benchmark stage is already running for {run_root}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_state(run_root: Path, profile_id: str) -> dict[str, Any]:
    path = state_path(run_root)
    state = load_json_object(path, "benchmark state")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise BenchmarkConfigError(f"Unsupported benchmark state schema at {path}")
    if state.get("profile_id") != profile_id:
        raise BenchmarkConfigError(
            f"State profile mismatch: expected {profile_id}, found {state.get('profile_id')}"
        )
    return state


def status_at_least(state: dict[str, Any], expected: str) -> bool:
    current = state.get("status")
    if current not in STAGE_ORDER:
        raise BenchmarkConfigError(f"Unknown benchmark state status: {current!r}")
    if expected not in STAGE_ORDER:
        raise ValueError(f"Unknown expected benchmark status: {expected!r}")
    return STAGE_ORDER[str(current)] >= STAGE_ORDER[expected]


def require_status(state: dict[str, Any], expected: str, stage: str) -> None:
    current = state.get("status")
    if current != expected:
        raise BenchmarkConfigError(
            f"{stage} requires benchmark status {expected!r}, found {current!r}"
        )


def write_state(run_root: Path, state: dict[str, Any]) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    target = state_path(run_root)
    temporary = run_root / ".state.json.tmp"
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target
