"""Freeze experiment inputs and benchmark-code provenance for resumable runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    load_profile,
    resolve_repo_path,
)


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise BenchmarkConfigError(f"Git provenance check failed for {repo}: {details}")
    return completed.stdout.strip()


def observe_benchmark_repository(repo_root: Path, *, require_clean: bool = True) -> dict[str, Any]:
    """Return the exact benchmark checkout used to orchestrate a run."""

    root = repo_root.resolve()
    if not (root / ".git").exists():
        raise BenchmarkConfigError(f"Benchmark repository is not a Git checkout: {root}")
    status = _git(root, "status", "--porcelain")
    if require_clean and status:
        raise BenchmarkConfigError(
            "SPATHI-benchmarks worktree must be committed and clean before an experiment"
        )
    origin = _git(root, "remote", "get-url", "origin")
    return {
        "path": str(root),
        "commit": _git(root, "rev-parse", "HEAD"),
        "origin": origin,
        "clean": not status,
    }


def _snapshot_sources(repo_root: Path, profile: dict[str, Any]) -> list[Path]:
    relative_paths = {
        "repos.lock.json",
        "images.lock.json",
        f"experiments/v1/profiles/{profile['id']}.json",
        "experiments/v1/spathi/search-space.json",
        "experiments/v1/spathi/selected-config.json",
    }
    selected = load_json_object(
        repo_root / "experiments/v1/spathi/selected-config.json",
        "SPATHI selected configuration",
    )
    evidence = selected.get("selection_evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("validation_ranking"), str):
        relative_paths.add(evidence["validation_ranking"])
    for job in profile["generation_jobs"]:
        relative_paths.update((job["scenario"], job["simulator_runs"]))
    for batch in profile["inference_batches"]:
        relative_paths.update((batch["tools_params"], batch["run_metadata"]))
        if batch.get("custom_tools"):
            relative_paths.add(batch["custom_tools"])
        if batch.get("calibration_manifest"):
            relative_paths.add(batch["calibration_manifest"])
    relative_paths.update(
        path.relative_to(repo_root).as_posix() for path in (repo_root / "schemas").glob("*.json")
    )
    image_lock = load_image_lock(repo_root)
    relative_paths.update(image_lock["images"]["spathi"]["build_inputs"])

    sources: list[Path] = []
    for relative in sorted(relative_paths):
        source = resolve_repo_path(repo_root, relative, "frozen benchmark input")
        tracked = _git(repo_root, "ls-files", "--error-unmatch", relative)
        if tracked != relative:
            raise BenchmarkConfigError(f"Benchmark input is not tracked canonically: {relative}")
        sources.append(source)
    return sources


def freeze_run_inputs(repo_root: Path, profile_id: str, run_root: Path) -> dict[str, Any]:
    """Copy all run-defining files once and return their immutable manifest."""

    snapshot_root = run_root / "input_snapshot"
    manifest_path = snapshot_root / "manifest.json"
    if manifest_path.is_file():
        return validate_run_inputs(repo_root, profile_id, run_root)
    if snapshot_root.exists():
        raise BenchmarkConfigError(f"Incomplete input snapshot already exists: {snapshot_root}")

    benchmark = observe_benchmark_repository(repo_root, require_clean=True)
    profile = load_profile(repo_root, profile_id, require_runnable=True)
    sources = _snapshot_sources(repo_root, profile)
    run_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".input-snapshot-", dir=run_root))
    try:
        repository_root = temporary / "repository"
        files: list[dict[str, Any]] = []
        for source in sources:
            relative = source.relative_to(repo_root.resolve()).as_posix()
            destination = repository_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            files.append(
                {
                    "path": relative,
                    "sha256": sha256_file(destination),
                    "size": destination.stat().st_size,
                }
            )
        manifest = {
            "schema_version": "1.0",
            "profile_id": profile_id,
            "benchmark_repository": benchmark,
            "files": files,
            "status": "complete",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, snapshot_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_run_inputs(repo_root, profile_id, run_root)


def validate_run_inputs(repo_root: Path, profile_id: str, run_root: Path) -> dict[str, Any]:
    """Verify a frozen snapshot and that the orchestrator code has not changed."""

    snapshot_root = run_root / "input_snapshot"
    manifest_path = snapshot_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkConfigError(f"Cannot read frozen input manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise BenchmarkConfigError("Frozen input snapshot is not complete")
    if manifest.get("schema_version") != "1.0" or manifest.get("profile_id") != profile_id:
        raise BenchmarkConfigError("Frozen input snapshot identity differs from this run")
    observed = observe_benchmark_repository(repo_root, require_clean=True)
    recorded = manifest.get("benchmark_repository")
    if not isinstance(recorded, dict) or any(
        recorded.get(field) != observed.get(field) for field in ("commit", "origin")
    ):
        raise BenchmarkConfigError(
            "SPATHI-benchmarks checkout changed after this run froze its inputs; use a new run ID"
        )
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise BenchmarkConfigError("Frozen input snapshot contains no files")
    observed_paths: set[str] = set()
    for entry in raw_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise BenchmarkConfigError("Frozen input snapshot has an invalid file entry")
        relative = entry["path"]
        member = (snapshot_root / "repository" / relative).resolve()
        repository_root = (snapshot_root / "repository").resolve()
        if not member.is_relative_to(repository_root) or not member.is_file():
            raise BenchmarkConfigError(f"Frozen input is missing or escapes its root: {relative}")
        if relative in observed_paths:
            raise BenchmarkConfigError(f"Frozen input is listed twice: {relative}")
        observed_paths.add(relative)
        if member.stat().st_size != entry.get("size") or sha256_file(member) != entry.get("sha256"):
            raise BenchmarkConfigError(f"Frozen input differs from its manifest: {relative}")
    return manifest


def frozen_repository_root(run_root: Path) -> Path:
    return (run_root / "input_snapshot" / "repository").resolve()


def frozen_profile(run_root: Path, profile_id: str) -> dict[str, Any]:
    return load_profile(frozen_repository_root(run_root), profile_id, require_runnable=True)


def frozen_path(run_root: Path, raw_path: str, label: str) -> Path:
    return resolve_repo_path(frozen_repository_root(run_root), raw_path, label)
