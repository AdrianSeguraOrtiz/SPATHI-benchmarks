"""Freeze experiment inputs and benchmark-code provenance for resumable runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    load_profile,
    load_profile_file,
    profile_path,
    resolve_repo_path,
)


@dataclass(frozen=True)
class SnapshotSource:
    path: Path
    relative: str
    kind: str


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


def _add_ranking_chain(
    repo_root: Path,
    relative_paths: set[str],
    first_ranking: str,
    runtime_candidates: set[str] | None = None,
) -> None:
    """Add one ranking and every linked predecessor to a frozen input snapshot."""

    pending: str | None = first_ranking
    observed: set[str] = set()
    while pending is not None:
        if pending in observed:
            raise BenchmarkConfigError("Calibration ranking provenance contains a cycle")
        observed.add(pending)
        ranking_path = resolve_repo_path(repo_root, pending, "calibration ranking evidence")
        relative_paths.add(pending)
        if runtime_candidates is not None and ranking_path.is_relative_to(
            (repo_root.resolve() / "work").resolve()
        ):
            runtime_candidates.add(pending)
        ranking = load_json_object(ranking_path, "calibration ranking evidence")
        parent = ranking.get("parent_ranking")
        if parent is None:
            pending = None
        elif isinstance(parent, dict) and isinstance(parent.get("path"), str):
            pending = parent["path"]
        else:
            raise BenchmarkConfigError("Calibration ranking has an invalid parent link")


def _runtime_source(repo_root: Path, relative: str, label: str) -> Path:
    root = repo_root.resolve()
    unresolved = root / relative
    current = unresolved
    while current != root:
        if current.is_symlink():
            raise BenchmarkConfigError(f"{label} cannot contain symlinks: {relative}")
        current = current.parent
    resolved = unresolved.resolve()
    if not resolved.is_relative_to((root / "work").resolve()) or not resolved.is_file():
        raise BenchmarkConfigError(f"{label} must be a regular file below work/: {relative}")
    return resolved


def _snapshot_source_records(
    repo_root: Path,
    profile: dict[str, Any],
    *,
    profile_source: Path | None = None,
) -> list[SnapshotSource]:
    repo_root = repo_root.resolve()
    canonical_profile = profile_path(repo_root, str(profile["id"]))
    selected_profile = (profile_source or canonical_profile).resolve()
    try:
        profile_relative = selected_profile.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise BenchmarkConfigError("Profile source escapes the benchmark repository") from exc
    relative_paths = {
        "repos.lock.json",
        "images.lock.json",
        profile_relative,
        "experiments/v1/spathi/search-space.json",
    }
    runtime_candidates: set[str] = set()
    if selected_profile.is_relative_to((repo_root / "work").resolve()):
        runtime_candidates.add(profile_relative)

    selected_relative = profile.get("selected_config", "experiments/v1/spathi/selected-config.json")
    if not isinstance(selected_relative, str):
        raise BenchmarkConfigError("Profile selected_config must be a relative path")
    selected_path = resolve_repo_path(repo_root, selected_relative, "SPATHI selected configuration")
    relative_paths.add(selected_relative)
    if selected_path.is_relative_to((repo_root / "work").resolve()):
        runtime_candidates.add(selected_relative)
    selected = load_json_object(
        selected_path,
        "SPATHI selected configuration",
    )
    evidence = selected.get("selection_evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("validation_ranking"), str):
        _add_ranking_chain(
            repo_root,
            relative_paths,
            evidence["validation_ranking"],
            runtime_candidates,
        )

    materialization_path = profile.get("materialization_manifest")
    if materialization_path is not None:
        if not isinstance(materialization_path, str):
            raise BenchmarkConfigError("Profile materialization_manifest must be a relative path")
        resolved_manifest = resolve_repo_path(
            repo_root, materialization_path, "benchmark materialization manifest"
        )
        relative_paths.add(materialization_path)
        if resolved_manifest.is_relative_to((repo_root / "work").resolve()):
            runtime_candidates.add(materialization_path)
        materialization = load_json_object(resolved_manifest, "benchmark materialization manifest")
        catalog_plan = materialization.get("catalog_plan")
        if isinstance(catalog_plan, dict) and isinstance(catalog_plan.get("path"), str):
            relative_paths.add(catalog_plan["path"])
    for job in profile["generation_jobs"]:
        relative_paths.update((job["scenario"], job["simulator_runs"]))
        scenario_path = resolve_repo_path(
            repo_root,
            job["scenario"],
            f"{job['id']} scenario",
        )
        scenario = load_json_object(scenario_path, f"{job['id']} scenario")
        inputs = scenario.get("inputs", {})
        if not isinstance(inputs, dict):
            raise BenchmarkConfigError(f"{job['id']} scenario inputs must be an object")
        for input_id, input_spec in inputs.items():
            if not isinstance(input_id, str) or not isinstance(input_spec, dict):
                raise BenchmarkConfigError(f"{job['id']} scenario contains an invalid input")
            raw_path = input_spec.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise BenchmarkConfigError(f"{job['id']} scenario input {input_id!r} has no path")
            source = Path(raw_path)
            if not source.is_absolute():
                source = scenario_path.parent / source
            source = source.resolve()
            if not source.is_relative_to(repo_root):
                raise BenchmarkConfigError(
                    f"{job['id']} scenario input {input_id!r} must be inside the benchmark repository"
                )
            if source.is_file():
                relative_paths.add(source.relative_to(repo_root).as_posix())
            elif source.is_dir():
                members = [member for member in sorted(source.rglob("*")) if member.is_file()]
                if not members:
                    raise BenchmarkConfigError(
                        f"{job['id']} scenario input {input_id!r} directory is empty"
                    )
                relative_paths.update(
                    member.relative_to(repo_root).as_posix() for member in members
                )
            else:
                raise BenchmarkConfigError(
                    f"{job['id']} scenario input {input_id!r} does not exist: {source}"
                )
    for batch in profile["inference_batches"]:
        relative_paths.update((batch["tools_params"], batch["run_metadata"]))
        derived_batch_paths = [batch["tools_params"], batch["run_metadata"]]
        if batch.get("custom_tools"):
            relative_paths.add(batch["custom_tools"])
            derived_batch_paths.append(batch["custom_tools"])
        if batch.get("calibration_manifest"):
            relative_paths.add(batch["calibration_manifest"])
            derived_batch_paths.append(batch["calibration_manifest"])
            manifest = load_json_object(
                resolve_repo_path(
                    repo_root,
                    batch["calibration_manifest"],
                    f"{batch['id']} calibration manifest",
                )
            )
            parent_ranking = manifest.get("parent_ranking")
            if isinstance(parent_ranking, dict) and isinstance(parent_ranking.get("path"), str):
                _add_ranking_chain(
                    repo_root,
                    relative_paths,
                    parent_ranking["path"],
                    runtime_candidates,
                )
        if batch.get("kind") == "spathi":
            for relative in derived_batch_paths:
                resolved = resolve_repo_path(repo_root, relative, "SPATHI runtime input")
                if resolved.is_relative_to((repo_root / "work").resolve()):
                    runtime_candidates.add(relative)
    relative_paths.update(
        path.relative_to(repo_root).as_posix() for path in (repo_root / "schemas").glob("*.json")
    )
    image_lock = load_image_lock(repo_root)
    relative_paths.update(image_lock["images"]["spathi"]["build_inputs"])

    tracked = set(_git(repo_root, "ls-files", "-z").split("\0"))
    sources: list[SnapshotSource] = []
    for relative in sorted(relative_paths):
        source = resolve_repo_path(repo_root, relative, "frozen benchmark input")
        if relative in tracked:
            kind = "versioned"
        elif relative in runtime_candidates:
            source = _runtime_source(repo_root, relative, "Runtime-derived benchmark input")
            kind = "runtime-derived"
        else:
            raise BenchmarkConfigError(f"Benchmark input is not tracked canonically: {relative}")
        sources.append(SnapshotSource(path=source, relative=relative, kind=kind))
    return sources


def _snapshot_sources(
    repo_root: Path,
    profile: dict[str, Any],
    *,
    profile_source: Path | None = None,
) -> list[Path]:
    """Compatibility helper used by focused provenance tests."""

    return [
        source.path
        for source in _snapshot_source_records(
            repo_root,
            profile,
            profile_source=profile_source,
        )
    ]


def freeze_run_inputs(
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    *,
    profile_file: Path | None = None,
) -> dict[str, Any]:
    """Copy all run-defining files once and return their immutable manifest."""

    snapshot_root = run_root / "input_snapshot"
    manifest_path = snapshot_root / "manifest.json"
    if manifest_path.is_file():
        manifest = validate_run_inputs(repo_root, profile_id, run_root)
        if profile_file is not None:
            requested = profile_file.absolute().resolve(strict=False)
            try:
                requested_relative = requested.relative_to(repo_root.resolve()).as_posix()
            except ValueError as exc:
                raise BenchmarkConfigError(
                    "Requested profile source escapes the benchmark repository"
                ) from exc
            frozen_relative = (manifest.get("profile_source") or {}).get("path")
            if requested_relative != frozen_relative:
                raise BenchmarkConfigError(
                    f"Existing run was frozen from a different profile source: {frozen_relative}"
                )
        return manifest
    if snapshot_root.exists():
        raise BenchmarkConfigError(f"Incomplete input snapshot already exists: {snapshot_root}")

    benchmark = observe_benchmark_repository(repo_root, require_clean=True)
    if profile_file is None:
        selected_profile_path = profile_path(repo_root, profile_id)
        profile = load_profile(repo_root, profile_id, require_runnable=True)
    else:
        requested_profile_path = profile_file.absolute()
        profile = load_profile_file(
            repo_root,
            requested_profile_path,
            profile_id,
            require_runnable=True,
            require_runtime=True,
        )
        selected_profile_path = requested_profile_path.resolve()
    sources = _snapshot_source_records(
        repo_root,
        profile,
        profile_source=selected_profile_path,
    )
    run_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".input-snapshot-", dir=run_root))
    try:
        repository_root = temporary / "repository"
        files: list[dict[str, Any]] = []
        for source in sources:
            relative = source.relative
            destination = repository_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.path, destination)
            files.append(
                {
                    "path": relative,
                    "sha256": sha256_file(destination),
                    "size": destination.stat().st_size,
                    "kind": source.kind,
                }
            )
        selected_profile_relative = selected_profile_path.relative_to(
            repo_root.resolve()
        ).as_posix()
        profile_entry = next(entry for entry in files if entry["path"] == selected_profile_relative)
        manifest = {
            "schema_version": "2.0",
            "profile_id": profile_id,
            "profile_source": {
                "path": profile_entry["path"],
                "sha256": profile_entry["sha256"],
                "kind": profile_entry["kind"],
            },
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
    if manifest.get("schema_version") != "2.0" or manifest.get("profile_id") != profile_id:
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
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or entry.get("kind") not in {"versioned", "runtime-derived"}
        ):
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
    profile_source = manifest.get("profile_source")
    if not isinstance(profile_source, dict):
        raise BenchmarkConfigError("Frozen input snapshot has no profile source")
    matching_profiles = [
        entry
        for entry in raw_files
        if entry.get("path") == profile_source.get("path")
        and entry.get("sha256") == profile_source.get("sha256")
        and entry.get("kind") == profile_source.get("kind")
    ]
    if len(matching_profiles) != 1:
        raise BenchmarkConfigError("Frozen profile source differs from the file manifest")
    return manifest


def frozen_repository_root(run_root: Path) -> Path:
    return (run_root / "input_snapshot" / "repository").resolve()


def frozen_profile(run_root: Path, profile_id: str) -> dict[str, Any]:
    manifest = load_json_object(
        run_root / "input_snapshot" / "manifest.json", "frozen input manifest"
    )
    source = manifest.get("profile_source")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise BenchmarkConfigError("Frozen input snapshot has no profile source")
    return load_profile_file(
        frozen_repository_root(run_root),
        frozen_repository_root(run_root) / source["path"],
        profile_id,
        require_runnable=True,
    )


def frozen_path(run_root: Path, raw_path: str, label: str) -> Path:
    return resolve_repo_path(frozen_repository_root(run_root), raw_path, label)
