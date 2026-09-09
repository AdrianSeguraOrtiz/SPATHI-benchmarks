"""Idempotent materialization of bounded, staged SPATHI calibration."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ._materialization_io import publish_new_json_directory
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
from .convergence_fanout import build_convergence_plans
from .safe_io import atomic_write_json, path_below_no_symlinks, write_idempotent_json


def _repo_relative_path(repo_root: Path, path: Path, label: str) -> str:
    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    absolute = path_below_no_symlinks(path, root, label)
    relative = absolute.relative_to(root)
    if relative == Path("."):
        raise BenchmarkConfigError(f"{label} cannot be the repository root")
    return relative.as_posix()


def _valid_profile_destination(repo_root: Path, path: Path, profile_id: str) -> str:
    relative = _repo_relative_path(repo_root, path, "Calibration profile output")
    canonical = f"experiments/v1/profiles/{profile_id}.json"
    if relative != canonical and not Path(relative).is_relative_to("work"):
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


def materialize_calibration_stage(
    *,
    repo_root: Path,
    stage_id: str,
    calibration_budget: str,
    fidelity: str,
    parent_ranking_path: Path | None,
    output_dir: Path,
    profile_output: Path,
) -> tuple[Path, Path]:
    """Create or validate one deterministic calibration request bundle."""

    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    search = load_search_space(root)
    spathi_identity = locked_spathi_identity(root)
    parents, parent_ranking = load_parent_ranking(
        root,
        search,
        stage_id,
        calibration_budget,
        fidelity,
        parent_ranking_path,
    )
    if stage_id == "validation":
        candidates = parents
        runs = expand_validation_runs(search, candidates, calibration_budget)
        dataset_role = "validation"
    elif fidelity == "scout" or stage_id == "convergence":
        candidates = expand_stage_candidates(search, stage_id, parents=parents)
        runs = expand_stage_runs(
            search,
            stage_id,
            calibration_budget,
            fidelity,
            parents=parents,
        )
        dataset_role = "development"
    else:
        candidates = parents
        runs = expand_stage_runs(
            search,
            stage_id,
            calibration_budget,
            fidelity,
            candidates=candidates,
        )
        dataset_role = "development"
    image_tag = load_image_lock(root)["images"]["spathi"]["tag"]
    tools, custom, metadata = build_andrea_requests(
        runs,
        docker_image=image_tag,
        calibration_budget=calibration_budget,
    )
    bundle_path = _repo_relative_path(root, output_dir, "Calibration plan output")
    profile_id = stage_profile_id(stage_id, calibration_budget, fidelity)
    _valid_profile_destination(root, profile_output, profile_id)
    template_prefix = "selection" if stage_id == "validation" else "calibration"
    template_name = f"{template_prefix}-{calibration_budget}.json"
    template = load_json_object(
        root / "experiments" / "v1" / "profiles" / template_name,
        f"{stage_id} profile template",
    )
    profile = build_stage_profile(
        template,
        stage_id,
        calibration_budget,
        fidelity,
        bundle_path=bundle_path,
        search_space=search,
    )
    manifest = {
        "schema_version": "1.0",
        "profile_id": profile_id,
        "calibration_budget": calibration_budget,
        "search_space_id": search["id"],
        "search_space_sha256": sha256_file(root / "experiments/v1/spathi/search-space.json"),
        "stage_id": stage_id,
        "fidelity": fidelity,
        "dataset_role": dataset_role,
        **spathi_identity,
        "parent_ranking": parent_ranking,
        "candidates": [candidate_to_dict(candidate) for candidate in candidates],
        "runs": canonical_run_records(runs),
    }
    validate_calibration_manifest(root, search, manifest)
    payloads = {
        "tools-params.json": tools,
        "custom-tools.json": custom,
        "run-metadata.json": metadata,
        "calibration-manifest.json": manifest,
    }
    if stage_id == "convergence":
        payloads["convergence-plans.json"] = build_convergence_plans(root, candidates, runs)
        profile["inference_batches"][0]["convergence_plans"] = (
            f"{bundle_path}/convergence-plans.json"
        )
    validate_json(profile, root / "schemas/profile.schema.json", f"profile {profile_id}")

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
            publish_new_json_directory(output_dir, payloads)
            published_bundle = True
        if not profile_exists:
            write_idempotent_json(
                profile_output,
                profile,
                f"profile {profile_id}",
                boundary=root,
            )
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
    calibration_budget: str,
    fidelity: str,
    manifest_path: Path,
    summary_path: Path,
    output_path: Path,
) -> Path:
    """Create or validate the deterministic ranking for one completed stage."""

    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    manifest_path = path_below_no_symlinks(
        manifest_path,
        root,
        "Calibration manifest",
        require_file=True,
    )
    summary_path = path_below_no_symlinks(
        summary_path,
        root,
        "Benchmark summary",
        require_file=True,
    )
    search = load_search_space(root)
    manifest = load_json_object(manifest_path, "calibration manifest")
    if (
        manifest.get("search_space_id") != search["id"]
        or manifest.get("stage_id") != stage_id
        or manifest.get("fidelity") != fidelity
        or manifest.get("calibration_budget") != calibration_budget
    ):
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
    reconciled_rows, calibration_panel = reconcile_summary_runs(
        manifest,
        rows,
        summary.get("calibration_panel"),
    )
    _repo_relative_path(root, output_path, "Calibration ranking output")
    payload = build_calibration_ranking(
        search,
        stage_id,
        fidelity,
        candidates,
        reconciled_rows,
        calibration_panel=calibration_panel,
        calibration_budget=calibration_budget,
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
    return write_idempotent_json(
        output_path,
        payload,
        "calibration ranking",
        boundary=root,
    )


def freeze_calibration_selection(
    *,
    repo_root: Path,
    ranking_path: Path,
    output_path: Path,
) -> Path:
    """Create or validate one frozen selection linked to validation evidence."""

    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
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
    return atomic_write_json(
        output_path,
        payload,
        "frozen SPATHI selection",
        boundary=root,
    )
