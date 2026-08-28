"""Atomic runtime state for a concrete benchmark execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import BenchmarkConfigError, load_json_object


def state_path(run_root: Path) -> Path:
    return run_root / "state.json"


def load_state(run_root: Path, profile_id: str) -> dict[str, Any]:
    path = state_path(run_root)
    state = load_json_object(path, "benchmark state")
    if state.get("schema_version") != "1.0":
        raise BenchmarkConfigError(f"Unsupported benchmark state schema at {path}")
    if state.get("profile_id") != profile_id:
        raise BenchmarkConfigError(
            f"State profile mismatch: expected {profile_id}, found {state.get('profile_id')}"
        )
    return state


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
