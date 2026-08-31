"""Idempotent materialization of adaptive SPATHI calibration stages."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .calibration import (
    build_andrea_requests,
    build_calibration_ranking,
    build_frozen_selected_config,
    build_stage_profile,
    candidate_to_dict,
    canonical_run_records,
    expand_stage_candidates,
    expand_stage_runs,
    expand_validation_runs,
    load_parent_ranking,
    load_search_space,
    locked_spathi_identity,
    reconcile_summary_runs,
    stage_profile_id,
    validate_calibration_manifest,
)
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    validate_calibration_contracts,
    validate_json,
)


def _repo_relative_path(repo_root: Path, path: Path, label: str) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} must be inside the benchmark repository") from exc
    if relative == Path("."):
        raise BenchmarkConfigError(f"{label} cannot be the repository root")
    return relative.as_posix()


def _valid_profile_destination(repo_root: Path, path: Path, profile_id: str) -> str:
    relative = _repo_relative_path(repo_root, path, "Calibration profile output")
    canonical = f"experiments/v1/profiles/{profile_id}.json"
    if relative != canonical and not path.resolve().is_relative_to(
        (repo_root.resolve() / "work").resolve()
    ):
        raise BenchmarkConfigError(
            f"Calibration profile must be written to {canonical} or below work/"
        )
    return relative


def _same_json(path: Path, expected: dict[str, Any], label: str) -> None:
    observed = load_json_object(path, label)
    if observed != expected:
        raise BenchmarkConfigError(f"Existing {label} differs from the requested workflow: {path}")


def _validate_bundle(output_dir: Path, payloads: dict[str, dict[str, Any]]) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise BenchmarkConfigError(f"Existing calibration bundle is not a directory: {output_dir}")
    observed_members = {path.name for path in output_dir.iterdir()}
    if observed_members != set(payloads):
        raise BenchmarkConfigError(
            f"Existing calibration bundle has unexpected members: {sorted(observed_members)}"
        )
    for filename, payload in payloads.items():
        member = output_dir / filename
        if member.is_symlink() or not member.is_file():
            raise BenchmarkConfigError(
                f"Existing calibration bundle member is not a regular file: {member}"
            )
        _same_json(member, payload, f"calibration {filename}")


def _publish_bundle(output_dir: Path, payloads: dict[str, dict[str, Any]]) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for filename, payload in payloads.items():
            (temporary / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _publish_profile(profile_output: Path, profile: dict[str, Any]) -> None:
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{profile_output.name}-", dir=profile_output.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, profile_output)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_calibration_stage(
    *,
    repo_root: Path,
    stage_id: str,
    parent_ranking_path: Path | None,
    output_dir: Path,
    profile_output: Path,
) -> tuple[Path, Path]:
    """Create or validate one deterministic calibration request bundle."""

    root = repo_root.resolve()
    search = load_search_space(root)
    spathi_identity = locked_spathi_identity(root)
    parents, parent_ranking = load_parent_ranking(
        root,
        search,
        stage_id,
        parent_ranking_path,
    )
    if stage_id == "validation":
        candidates = parents
        runs = expand_validation_runs(search, candidates)
        dataset_role = "validation"
    else:
        candidates = expand_stage_candidates(search, stage_id, parents=parents)
        runs = expand_stage_runs(search, stage_id, parents=parents)
        dataset_role = "development"
    image_tag = load_image_lock(root)["images"]["spathi"]["tag"]
    tools, custom, metadata = build_andrea_requests(runs, docker_image=image_tag)
    bundle_path = _repo_relative_path(root, output_dir, "Calibration plan output")
    profile_id = stage_profile_id(stage_id)
    _valid_profile_destination(root, profile_output, profile_id)
    template_name = "selection.json" if stage_id == "validation" else "calibration.json"
    template = load_json_object(
        root / "experiments" / "v1" / "profiles" / template_name,
        f"{stage_id} profile template",
    )
    profile = build_stage_profile(template, stage_id, bundle_path=bundle_path)
    manifest = {
        "schema_version": "1.0",
        "profile_id": profile_id,
        "search_space_id": search["id"],
        "search_space_sha256": sha256_file(root / "experiments/v1/spathi/search-space.json"),
        "stage_id": stage_id,
        "dataset_role": dataset_role,
        **spathi_identity,
        "parent_ranking": parent_ranking,
        "candidates": [candidate_to_dict(candidate) for candidate in candidates],
        "runs": canonical_run_records(runs),
    }
    validate_calibration_manifest(root, search, manifest)
    validate_json(profile, root / "schemas/profile.schema.json", f"profile {profile_id}")
    payloads = {
        "tools-params.json": tools,
        "custom-tools.json": custom,
        "run-metadata.json": metadata,
        "calibration-manifest.json": manifest,
    }

    bundle_exists = output_dir.exists() or output_dir.is_symlink()
    profile_exists = profile_output.exists() or profile_output.is_symlink()
    if bundle_exists:
        _validate_bundle(output_dir, payloads)
    if profile_exists:
        if profile_output.is_symlink() or not profile_output.is_file():
            raise BenchmarkConfigError(
                f"Existing calibration profile is not a regular file: {profile_output}"
            )
        _same_json(profile_output, profile, f"profile {profile_id}")

    published_bundle = False
    published_profile = False
    try:
        if not bundle_exists:
            _publish_bundle(output_dir, payloads)
            published_bundle = True
        if not profile_exists:
            _publish_profile(profile_output, profile)
            published_profile = True
    except Exception:
        if published_profile and profile_output.exists():
            profile_output.unlink()
        if published_bundle and output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    return output_dir, profile_output


def rank_calibration_stage(
    *,
    repo_root: Path,
    stage_id: str,
    manifest_path: Path,
    summary_path: Path,
    output_path: Path,
) -> Path:
    """Create or validate the deterministic ranking for one completed stage."""

    root = repo_root.resolve()
    search = load_search_space(root)
    manifest = load_json_object(manifest_path, "calibration manifest")
    if manifest.get("search_space_id") != search["id"] or manifest.get("stage_id") != stage_id:
        raise BenchmarkConfigError("Calibration manifest identity differs from the requested stage")
    candidates, _runs = validate_calibration_manifest(root, search, manifest)
    summary = load_json_object(summary_path, "benchmark summary")
    if summary.get("profile_id") != manifest.get("profile_id"):
        raise BenchmarkConfigError("Benchmark summary profile_id differs from its manifest")
    expected_purpose = "selection" if stage_id == "validation" else "calibration"
    if summary.get("purpose") != expected_purpose:
        raise BenchmarkConfigError("Benchmark summary purpose differs from its calibration stage")
    benchmark_commit = summary.get("benchmark_commit")
    if not isinstance(benchmark_commit, str):
        raise BenchmarkConfigError("Benchmark summary contains no benchmark_commit")
    rows = summary.get("calibration_rows")
    if not isinstance(rows, list):
        raise BenchmarkConfigError("Benchmark summary contains no calibration_rows")
    reconciled_rows = reconcile_summary_runs(manifest, rows)
    _repo_relative_path(root, output_path, "Calibration ranking output")
    payload = build_calibration_ranking(
        search,
        stage_id,
        candidates,
        reconciled_rows,
        profile_id=str(manifest["profile_id"]),
        parent_ranking=manifest["parent_ranking"],
        search_space_sha256=sha256_file(root / "experiments/v1/spathi/search-space.json"),
        calibration_manifest_sha256=sha256_file(manifest_path),
        benchmark_summary_sha256=sha256_file(summary_path),
        benchmark_commit=benchmark_commit,
        **locked_spathi_identity(root),
    )
    validate_json(
        payload,
        root / "schemas/calibration-ranking.schema.json",
        "calibration ranking",
    )
    if output_path.exists():
        _same_json(output_path, payload, "calibration ranking")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def freeze_calibration_selection(
    *,
    repo_root: Path,
    ranking_path: Path,
    output_path: Path,
) -> Path:
    """Create or validate one frozen selection linked to validation evidence."""

    root = repo_root.resolve()
    _repo_relative_path(root, output_path, "Frozen SPATHI selection output")
    ranking_relative = _repo_relative_path(root, ranking_path, "Validation ranking")
    ranking_digest = sha256_file(ranking_path)
    if output_path.exists():
        existing = load_json_object(output_path, "existing selected configuration")
        if existing.get("status") == "selected" and existing.get("phase") == "frozen":
            validate_calibration_contracts(root, output_path)
            evidence = existing.get("selection_evidence")
            if (
                not isinstance(evidence, dict)
                or evidence.get("validation_ranking") != ranking_relative
                or evidence.get("validation_ranking_sha256") != ranking_digest
            ):
                raise BenchmarkConfigError(
                    "Existing frozen selection belongs to another validation ranking"
                )
            return output_path
        if existing.get("status") != "unselected" or existing.get("phase") != "planned":
            raise BenchmarkConfigError(
                f"Refusing to replace an incompatible selected configuration: {output_path}"
            )
    payload = build_frozen_selected_config(root, ranking_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path
