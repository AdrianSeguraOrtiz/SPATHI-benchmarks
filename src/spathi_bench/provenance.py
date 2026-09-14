"""Freeze experiment inputs and benchmark-code provenance for resumable runs."""

from __future__ import annotations

import json
import os
import re
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
from .safe_io import path_below_no_symlinks, reject_symlink_components
from .scenario_inputs import scenario_external_input_paths


@dataclass(frozen=True)
class SnapshotSource:
    path: Path
    relative: str
    kind: str


_PRODUCTION_PROVENANCE_KEYS = {
    "calibration_selection",
    "efficiency_plan",
    "efficiency_evidence",
    "efficiency_results",
}
_SAFETY_PROVENANCE_KEYS = {"panel", "plan", "evidence", "results"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def production_provenance_artifacts(
    production: dict[str, Any],
) -> list[tuple[str, dict[str, str]]]:
    """Return labeled v1 production-evidence leaves after strict shape validation."""

    provenance = production.get("provenance")
    if not isinstance(provenance, dict):
        raise BenchmarkConfigError("SPATHI production policy has no provenance")
    observed_keys = set(provenance)
    allowed_keys = _PRODUCTION_PROVENANCE_KEYS | {"efficiency_safety"}
    if not _PRODUCTION_PROVENANCE_KEYS.issubset(observed_keys) or not observed_keys.issubset(
        allowed_keys
    ):
        raise BenchmarkConfigError("SPATHI production policy provenance is invalid")

    raw_artifacts: list[tuple[str, Any]] = [
        (key, provenance[key]) for key in sorted(_PRODUCTION_PROVENANCE_KEYS)
    ]
    safety = provenance.get("efficiency_safety")
    if safety is not None:
        if not isinstance(safety, dict) or set(safety) != _SAFETY_PROVENANCE_KEYS:
            raise BenchmarkConfigError(
                "SPATHI production policy efficiency-safety provenance is invalid"
            )
        raw_artifacts.extend(
            (f"efficiency_safety_{key}", safety[key]) for key in sorted(_SAFETY_PROVENANCE_KEYS)
        )

    artifacts: list[tuple[str, dict[str, str]]] = []
    for label, artifact in raw_artifacts:
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"path", "sha256"}
            or not isinstance(artifact.get("path"), str)
            or not artifact["path"]
            or not isinstance(artifact.get("sha256"), str)
            or _SHA256_RE.fullmatch(artifact["sha256"]) is None
        ):
            raise BenchmarkConfigError("SPATHI production policy provenance is invalid")
        artifacts.append((label, artifact))
    return artifacts


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
        "experiments/v1/execution/policy.json",
        "experiments/v1/spathi/search-space.json",
        "experiments/v1/seeds/seed-plan.json",
    }
    runtime_candidates: set[str] = set()
    if selected_profile.is_relative_to((repo_root / "work").resolve()):
        runtime_candidates.add(profile_relative)

    production_relative = profile.get("production_config")
    if production_relative is not None:
        if not isinstance(production_relative, str):
            raise BenchmarkConfigError("Profile production_config must be a relative path")
        production_path = resolve_repo_path(
            repo_root, production_relative, "SPATHI production policy"
        )
        relative_paths.add(production_relative)
        if production_path.is_relative_to((repo_root / "work").resolve()):
            runtime_candidates.add(production_relative)
        production = load_json_object(production_path, "SPATHI production policy")
        for _label, artifact in production_provenance_artifacts(production):
            artifact_path = resolve_repo_path(
                repo_root, artifact["path"], "production policy provenance"
            )
            relative_paths.add(artifact["path"])
            if artifact_path.is_relative_to((repo_root / "work").resolve()):
                runtime_candidates.add(artifact["path"])
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
        if "generator_request" in job:
            generator_request = job["generator_request"]
            relative_paths.add(generator_request)
            resolved_request = resolve_repo_path(
                repo_root,
                generator_request,
                f"{job['id']} generator request",
            )
            if resolved_request.is_relative_to((repo_root / "work").resolve()):
                runtime_candidates.add(generator_request)
            continue
        relative_paths.update((job["scenario"], job["simulator_runs"]))
        scenario_path = resolve_repo_path(
            repo_root,
            job["scenario"],
            f"{job['id']} scenario",
        )
        relative_paths.update(
            source.relative_to(repo_root).as_posix()
            for source in scenario_external_input_paths(scenario_path, repo_root)
        )
    for batch in profile["inference_batches"]:
        relative_paths.update((batch["tools_params"], batch["run_metadata"]))
        derived_batch_paths = [batch["tools_params"], batch["run_metadata"]]
        if batch.get("custom_tools"):
            relative_paths.add(batch["custom_tools"])
            derived_batch_paths.append(batch["custom_tools"])
        if batch.get("convergence_plans"):
            relative_paths.add(batch["convergence_plans"])
            derived_batch_paths.append(batch["convergence_plans"])
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
        # Catalog requests can also be materialized for a runtime profile.
        # Only explicit batch inputs below work/ qualify, not arbitrary sources.
        for relative in derived_batch_paths:
            resolved = resolve_repo_path(repo_root, relative, "Inference runtime input")
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


def freeze_run_inputs(
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    *,
    profile_file: Path | None = None,
) -> dict[str, Any]:
    """Copy all run-defining files once and return their immutable manifest."""

    run_root = reject_symlink_components(run_root, "run root")
    snapshot_root = run_root / "input_snapshot"
    manifest_path = snapshot_root / "manifest.json"
    if manifest_path.is_file():
        manifest = validate_run_inputs(repo_root, profile_id, run_root)
        if profile_file is not None:
            root = reject_symlink_components(repo_root, "repository root")
            requested = path_below_no_symlinks(
                Path(os.path.abspath(profile_file)),
                root,
                "requested profile source",
            )
            work_root = path_below_no_symlinks(
                root / "work",
                root,
                "runtime work root",
            )
            if not requested.is_relative_to(work_root):
                raise BenchmarkConfigError(
                    f"Runtime profile must be stored below work/: {profile_file}"
                )
            requested_relative = requested.relative_to(root).as_posix()
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
            "schema_version": "1.0",
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

    safe_run_root = reject_symlink_components(run_root, "run root")
    snapshot_root = path_below_no_symlinks(
        safe_run_root / "input_snapshot",
        safe_run_root,
        "frozen input snapshot",
        require_directory=True,
    )
    manifest_path = path_below_no_symlinks(
        snapshot_root / "manifest.json",
        snapshot_root,
        "frozen input manifest",
        require_file=True,
    )
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
    from .study_context import permit_revision_change

    if not isinstance(recorded, dict) or recorded.get("origin") != observed.get("origin"):
        raise BenchmarkConfigError(
            "SPATHI-benchmarks repository origin changed after this run froze its inputs"
        )
    if recorded.get("commit") != observed.get("commit") and not permit_revision_change(
        "SPATHI-benchmarks commit", recorded.get("commit"), observed.get("commit")
    ):
        raise BenchmarkConfigError("SPATHI-benchmarks checkout changed; resume the named study")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise BenchmarkConfigError("Frozen input snapshot contains no files")
    repository_root = path_below_no_symlinks(
        snapshot_root / "repository",
        snapshot_root,
        "frozen input repository",
        require_directory=True,
    )
    observed_paths: set[str] = set()
    for entry in raw_files:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or entry.get("kind") not in {"versioned", "runtime-derived"}
        ):
            raise BenchmarkConfigError("Frozen input snapshot has an invalid file entry")
        relative = entry["path"]
        member = path_below_no_symlinks(
            repository_root / relative,
            repository_root,
            f"frozen input {relative}",
            require_file=True,
        )
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
    safe_run_root = reject_symlink_components(run_root, "run root")
    snapshot_root = path_below_no_symlinks(
        safe_run_root / "input_snapshot",
        safe_run_root,
        "frozen input snapshot",
        require_directory=True,
    )
    return path_below_no_symlinks(
        snapshot_root / "repository",
        snapshot_root,
        "frozen input repository",
        require_directory=True,
    )


def frozen_profile(run_root: Path, profile_id: str) -> dict[str, Any]:
    safe_run_root = reject_symlink_components(run_root, "run root")
    manifest_path = path_below_no_symlinks(
        safe_run_root / "input_snapshot" / "manifest.json",
        safe_run_root,
        "frozen input manifest",
        require_file=True,
    )
    manifest = load_json_object(
        manifest_path,
        "frozen input manifest",
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
