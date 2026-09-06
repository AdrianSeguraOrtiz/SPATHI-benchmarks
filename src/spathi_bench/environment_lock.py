"""Validation of the exact host-side Python dependency contract."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

from .config import BenchmarkConfigError


class EnvironmentLockError(BenchmarkConfigError):
    """Raised when the active Python environment differs from the v1 lock."""


def load_environment_lock(repo_root: Path) -> dict[str, str]:
    """Load exact ``name==version`` constraints from the canonical lock file."""

    path = repo_root.resolve() / "constraints" / "benchmark.txt"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EnvironmentLockError(f"Cannot read host dependency lock {path}: {exc}") from exc

    locked: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise EnvironmentLockError(
                f"Host dependency lock line {line_number} must use one exact name==version pin"
            )
        name, version = (part.strip() for part in line.split("==", 1))
        if not name or not version:
            raise EnvironmentLockError(
                f"Host dependency lock line {line_number} has an empty name or version"
            )
        # Distribution names follow PEP 503 normalization: runs of hyphens,
        # underscores and dots are equivalent.  Applying the complete rule is
        # important both for duplicate detection and for metadata lookup.
        canonical_name = re.sub(r"[-_.]+", "-", name).lower()
        if canonical_name in locked:
            raise EnvironmentLockError(f"Duplicate host dependency pin for {name!r}")
        locked[canonical_name] = version
    if not locked:
        raise EnvironmentLockError("Host dependency lock cannot be empty")
    return dict(sorted(locked.items()))


def validate_environment_lock(repo_root: Path) -> dict[str, str]:
    """Require every locked distribution at exactly its recorded version."""

    locked = load_environment_lock(repo_root)
    missing: list[str] = []
    mismatched: list[str] = []
    for name, expected in locked.items():
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if observed != expected:
            mismatched.append(f"{name}: expected {expected}, observed {observed}")
    if missing or mismatched:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if mismatched:
            details.append("version mismatches " + "; ".join(mismatched))
        raise EnvironmentLockError("Host Python environment is not locked: " + " | ".join(details))
    return locked
