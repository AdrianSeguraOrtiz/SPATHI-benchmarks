"""Content identities for reproducible work without commit-coupled cache invalidation."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import BenchmarkConfigError


def _git(repo: Path, *arguments: str, check: bool = True) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    if check:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise BenchmarkConfigError(f"Cannot inspect repository {repo}: {detail}")
    return None


def _tracked_and_untracked(repo: Path) -> tuple[str, ...]:
    output = _git(
        repo,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    assert output is not None
    return tuple(sorted({name for name in output.split("\0") if name}))


def content_tree(repo: Path, prefixes: Iterable[str]) -> str:
    """Hash selected source paths, including uncommitted and untracked content."""

    root = repo.resolve()
    normalized = tuple(sorted({value.strip("/") for value in prefixes if value.strip("/")}))
    if not normalized:
        raise ValueError("At least one content-tree prefix is required")
    selected = [
        name
        for name in _tracked_and_untracked(root)
        if any(name == prefix or name.startswith(prefix + "/") for prefix in normalized)
    ]
    digest = hashlib.sha256()
    for name in selected:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise BenchmarkConfigError(f"Scientific source must be a regular file: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    if not selected:
        raise BenchmarkConfigError(
            f"No repository files match scientific prefixes: {', '.join(normalized)}"
        )
    return digest.hexdigest()


def repository_revision(repo: Path, *, scientific_prefixes: Iterable[str]) -> dict[str, Any]:
    root = repo.resolve()
    if not (root / ".git").exists():
        raise BenchmarkConfigError(f"Repository has no .git directory: {root}")
    status = _git(repo, "status", "--short") or ""
    return {
        "path": str(root),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "origin": _git(repo, "remote", "get-url", "origin", check=False),
        "dirty": bool(status),
        "status": status.splitlines(),
        "scientific_tree": content_tree(root, scientific_prefixes),
    }


def observe_revisions(
    benchmark_repo: Path,
    spathi_repo: Path,
    andrea_repo: Path,
) -> dict[str, Any]:
    benchmark = repository_revision(
        benchmark_repo,
        scientific_prefixes=(
            "experiments",
            "containers/spathi",
            "src/spathi_bench",
            "pyproject.toml",
            "constraints",
        ),
    )
    benchmark_execution_tree = content_tree(
        benchmark_repo,
        (
            "experiments/article-study.json",
            "containers/spathi",
            "src/spathi_bench/article_statistics.py",
            "src/spathi_bench/design.py",
            "src/spathi_bench/executor.py",
            "src/spathi_bench/study_workflow.py",
            "src/spathi_bench/synthetic.py",
            "pyproject.toml",
            "constraints",
        ),
    )
    spathi = repository_revision(
        spathi_repo,
        scientific_prefixes=("src/spathi", "pyproject.toml"),
    )
    andrea = repository_revision(
        andrea_repo,
        scientific_prefixes=(
            "andrea/core",
            "andrea/catalog_inference_tools",
            "andrea/catalog_simulation_data_tools",
            "wrappers",
            "pyproject.toml",
        ),
    )
    return {
        "benchmark_commit": benchmark["commit"],
        "benchmark_tree": benchmark["scientific_tree"],
        "benchmark_execution_tree": benchmark_execution_tree,
        "benchmark_origin": benchmark["origin"],
        "benchmark_dirty": benchmark["dirty"],
        "spathi_commit": spathi["commit"],
        "spathi_scientific_tree": spathi["scientific_tree"],
        "spathi_origin": spathi["origin"],
        "spathi_dirty": spathi["dirty"],
        "andrea_commit": andrea["commit"],
        "andrea_scientific_tree": andrea["scientific_tree"],
        "andrea_origin": andrea["origin"],
        "andrea_dirty": andrea["dirty"],
    }


__all__ = ["content_tree", "observe_revisions", "repository_revision"]
