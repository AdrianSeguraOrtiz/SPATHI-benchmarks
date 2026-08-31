"""Load and validate benchmark-owned configuration."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .artifacts import canonical_json_sha256, sha256_file


class BenchmarkConfigError(ValueError):
    """Raised when a committed benchmark contract is invalid."""


def load_json_object(path: Path, label: str | None = None) -> dict[str, Any]:
    label = label or path.name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkConfigError(f"Cannot read {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkConfigError(f"{label} must contain a JSON object: {path}")
    return payload


def validate_json(instance: dict[str, Any], schema_path: Path, label: str) -> None:
    schema = load_json_object(schema_path, f"schema {schema_path.name}")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        details = []
        for error in errors:
            location = ".".join(str(part) for part in error.path) or "<root>"
            details.append(f"{location}: {error.message}")
        raise BenchmarkConfigError(f"Invalid {label}: " + "; ".join(details))


def profile_path(repo_root: Path, profile_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", profile_id):
        raise BenchmarkConfigError(f"Invalid profile identifier: {profile_id!r}")
    return repo_root / "experiments" / "v1" / "profiles" / f"{profile_id}.json"


def resolve_repo_path(repo_root: Path, raw_path: str, label: str) -> Path:
    candidate = (repo_root / raw_path).resolve()
    root = repo_root.resolve()
    if not candidate.is_relative_to(root):
        raise BenchmarkConfigError(f"{label} escapes repository root: {raw_path}")
    if not candidate.is_file():
        raise BenchmarkConfigError(f"{label} does not exist: {raw_path}")
    return candidate


def load_profile(
    repo_root: Path,
    profile_id: str,
    *,
    require_runnable: bool = False,
) -> dict[str, Any]:
    path = profile_path(repo_root, profile_id)
    profile = load_json_object(path, f"profile {profile_id}")
    validate_json(profile, repo_root / "schemas" / "profile.schema.json", f"profile {profile_id}")
    if profile.get("id") != profile_id:
        raise BenchmarkConfigError(
            f"Profile ID {profile.get('id')!r} does not match filename {profile_id!r}"
        )
    if require_runnable and profile.get("status") != "runnable":
        raise BenchmarkConfigError(
            f"Profile {profile_id!r} is planned but not runnable; freeze its request files first"
        )
    return profile


def load_profile_file(
    repo_root: Path,
    profile_file: Path,
    profile_id: str,
    *,
    require_runnable: bool = False,
    require_runtime: bool = False,
) -> dict[str, Any]:
    """Load an explicitly addressed profile without weakening path boundaries.

    High-level workflows materialize profiles below ``work/``.  They are frozen
    by hash during preflight and never discovered implicitly.
    """

    root = repo_root.resolve()
    candidate = profile_file.absolute()
    if candidate.is_symlink():
        raise BenchmarkConfigError(f"Profile file cannot be a symlink: {profile_file}")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise BenchmarkConfigError(f"Profile file escapes repository root: {profile_file}")
    if require_runtime and not resolved.is_relative_to((root / "work").resolve()):
        raise BenchmarkConfigError(f"Runtime profile must be stored below work/: {profile_file}")
    profile = load_json_object(resolved, f"profile {profile_id}")
    validate_json(profile, root / "schemas" / "profile.schema.json", f"profile {profile_id}")
    if profile.get("id") != profile_id:
        raise BenchmarkConfigError(
            f"Profile ID {profile.get('id')!r} does not match requested ID {profile_id!r}"
        )
    if require_runnable and profile.get("status") != "runnable":
        raise BenchmarkConfigError(f"Profile {profile_id!r} is not runnable")
    return profile


def load_repo_lock(repo_root: Path) -> dict[str, Any]:
    lock = load_json_object(repo_root / "repos.lock.json", "repository lock")
    validate_json(lock, repo_root / "schemas" / "repos-lock.schema.json", "repository lock")
    return lock


def load_image_lock(repo_root: Path) -> dict[str, Any]:
    lock = load_json_object(repo_root / "images.lock.json", "image lock")
    if lock.get("schema_version") != "1.0" or not isinstance(lock.get("images"), dict):
        raise BenchmarkConfigError("images.lock.json has an unsupported structure")
    spathi = lock["images"].get("spathi")
    if not isinstance(spathi, dict):
        raise BenchmarkConfigError("images.lock.json must define images.spathi")
    for field in ("tag", "base_image", "source_repository", "source_commit"):
        if not isinstance(spathi.get(field), str) or not spathi[field]:
            raise BenchmarkConfigError(f"images.spathi.{field} must be a non-empty string")
    build_inputs = spathi.get("build_inputs")
    if not isinstance(build_inputs, dict) or not build_inputs:
        raise BenchmarkConfigError("images.spathi.build_inputs must be non-empty")
    if not all(
        isinstance(path, str)
        and path
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        for path, digest in build_inputs.items()
    ):
        raise BenchmarkConfigError(
            "images.spathi.build_inputs must map paths to lowercase SHA-256 digests"
        )
    dependencies = spathi.get("python_dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise BenchmarkConfigError("images.spathi.python_dependencies must be non-empty")
    catalog_images = lock.get("catalog_images")
    if not isinstance(catalog_images, dict):
        raise BenchmarkConfigError("images.lock.json must define catalog_images")
    for image_id, image in catalog_images.items():
        if not isinstance(image_id, str) or not re.fullmatch(
            r"(?:simulator|tool):[a-z0-9][a-z0-9_-]*", image_id
        ):
            raise BenchmarkConfigError(f"Invalid catalog image identifier: {image_id!r}")
        if not isinstance(image, dict):
            raise BenchmarkConfigError(f"catalog_images.{image_id} must be an object")
        if not isinstance(image.get("tag"), str) or not image["tag"]:
            raise BenchmarkConfigError(f"catalog_images.{image_id}.tag must be non-empty")
        if not isinstance(image.get("repo_digest"), str) or not re.fullmatch(
            r"[^@\s]+@sha256:[0-9a-f]{64}", image["repo_digest"]
        ):
            raise BenchmarkConfigError(
                f"catalog_images.{image_id}.repo_digest must pin a sha256 repository digest"
            )
    return lock


def load_run_metadata(repo_root: Path, raw_path: str, label: str) -> dict[str, Any]:
    path = resolve_repo_path(repo_root, raw_path, label)
    metadata = load_json_object(path, label)
    validate_json(
        metadata,
        repo_root / "schemas" / "run-metadata.schema.json",
        label,
    )
    run_ids = [item["run_id"] for item in metadata["runs"]]
    if len(set(run_ids)) != len(run_ids):
        raise BenchmarkConfigError(f"{label} contains duplicate run IDs")
    return metadata


def _canonical_remote(value: str) -> str:
    value = value.strip()
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    value = value.removesuffix(".git").rstrip("/")
    return value.lower()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise BenchmarkConfigError(f"Git check failed for {repo}: {details}")
    return completed.stdout.strip()


def validate_locked_repository(
    *,
    name: str,
    path: Path,
    lock_entry: dict[str, Any],
    require_clean: bool,
) -> dict[str, Any]:
    if not path.is_dir():
        raise BenchmarkConfigError(f"Locked repository {name} is missing: {path}")
    head = _git(path, "rev-parse", "HEAD")
    expected = str(lock_entry["commit"])
    if head != expected:
        raise BenchmarkConfigError(
            f"{name} HEAD mismatch: expected {expected}, found {head} at {path}"
        )
    origin = _git(path, "remote", "get-url", "origin")
    expected_origin = str(lock_entry["url"])
    if _canonical_remote(origin) != _canonical_remote(expected_origin):
        raise BenchmarkConfigError(
            f"{name} origin mismatch: expected {expected_origin}, found {origin}"
        )
    status = _git(path, "status", "--porcelain")
    if require_clean and status:
        raise BenchmarkConfigError(f"{name} worktree is not clean: {path}")
    return {"path": str(path.resolve()), "commit": head, "origin": origin, "clean": not status}


def _validate_run_identity(
    *,
    tools_params: dict[str, Any],
    custom_tools: dict[str, Any] | None,
    label: str,
) -> None:
    runs = tools_params.get("runs")
    if not isinstance(runs, list) or not runs or not all(isinstance(run, dict) for run in runs):
        raise BenchmarkConfigError(f"{label} must contain a non-empty runs array")
    run_ids = [run.get("run_id") for run in runs]
    if not all(isinstance(run_id, str) and run_id for run_id in run_ids):
        raise BenchmarkConfigError(f"{label} contains an invalid run_id")
    if len(set(run_ids)) != len(run_ids):
        raise BenchmarkConfigError(f"{label} contains duplicate run IDs")

    requested_custom_runs = {
        str(run["run_id"]): str(run.get("tool_id"))
        for run in runs
        if str(run.get("tool_id", "")).startswith("custom_")
    }
    if not requested_custom_runs:
        return
    if custom_tools is None:
        raise BenchmarkConfigError(f"{label} requests custom tools without custom_tools JSON")
    definitions = custom_tools.get("tools")
    if not isinstance(definitions, list) or not all(isinstance(item, dict) for item in definitions):
        raise BenchmarkConfigError("custom_tools must contain a tools array")
    definition_ids = [item.get("run_id") for item in definitions]
    if not all(isinstance(run_id, str) and run_id for run_id in definition_ids):
        raise BenchmarkConfigError("custom_tools contains an invalid run_id")
    if len(set(definition_ids)) != len(definition_ids):
        raise BenchmarkConfigError("custom_tools contains duplicate run IDs")
    invalid_pairs = {
        run_id: tool_id
        for run_id, tool_id in requested_custom_runs.items()
        if tool_id != f"custom_{run_id}"
    }
    if invalid_pairs:
        raise BenchmarkConfigError(
            "Custom runs must use tool_id=custom_<run_id>: " + repr(invalid_pairs)
        )
    if set(requested_custom_runs) != set(definition_ids):
        raise BenchmarkConfigError(
            f"Custom tool identity mismatch: requested={sorted(requested_custom_runs)}, "
            f"defined={sorted(definition_ids)}"
        )


def _resolve_generation_job_selector(
    *,
    owner: str,
    selected_jobs: Any,
    allowed_roles: set[str],
    generation_roles_by_id: dict[str, str],
) -> frozenset[str]:
    if selected_jobs is None:
        return frozenset(
            job_id
            for job_id, dataset_role in generation_roles_by_id.items()
            if dataset_role in allowed_roles
        )
    if not isinstance(selected_jobs, list) or not selected_jobs:
        raise BenchmarkConfigError(f"{owner} generation_jobs must be a non-empty array")
    if not all(isinstance(job_id, str) and job_id for job_id in selected_jobs):
        raise BenchmarkConfigError(f"{owner} contains an invalid generation job ID")
    if len(set(selected_jobs)) != len(selected_jobs):
        raise BenchmarkConfigError(f"{owner} contains duplicate generation job IDs")
    unknown_jobs = sorted(set(selected_jobs) - generation_roles_by_id.keys())
    if unknown_jobs:
        raise BenchmarkConfigError(f"{owner} selects unknown generation jobs: {unknown_jobs}")
    mismatched_jobs = {
        job_id: generation_roles_by_id[job_id]
        for job_id in selected_jobs
        if generation_roles_by_id[job_id] not in allowed_roles
    }
    if mismatched_jobs:
        role_label = (
            f"dataset role {next(iter(allowed_roles))!r}"
            if len(allowed_roles) == 1
            else f"dataset roles {sorted(allowed_roles)!r}"
        )
        raise BenchmarkConfigError(
            f"{owner} selects generation jobs outside {role_label}: {mismatched_jobs}"
        )
    return frozenset(selected_jobs)


def validate_profile_references(repo_root: Path, profile: dict[str, Any]) -> None:
    generation_ids = [job["id"] for job in profile["generation_jobs"]]
    if len(set(generation_ids)) != len(generation_ids):
        raise BenchmarkConfigError("Profile contains duplicate generation job IDs")
    generation_roles_by_id = {
        str(job["id"]): str(job["dataset_role"]) for job in profile["generation_jobs"]
    }
    batch_ids = [batch["id"] for batch in profile["inference_batches"]]
    if len(set(batch_ids)) != len(batch_ids):
        raise BenchmarkConfigError("Profile contains duplicate inference batch IDs")
    for batch in profile["inference_batches"]:
        _resolve_generation_job_selector(
            owner=f"Inference batch {batch['id']}",
            selected_jobs=batch.get("generation_jobs"),
            allowed_roles={str(batch["dataset_role"])},
            generation_roles_by_id=generation_roles_by_id,
        )
    for comparison in profile["statistics"]["comparisons"]:
        _resolve_generation_job_selector(
            owner=f"Statistics comparison {comparison['id']}",
            selected_jobs=comparison.get("generation_jobs"),
            allowed_roles={str(role) for role in comparison["dataset_roles"]},
            generation_roles_by_id=generation_roles_by_id,
        )
    if profile.get("status") != "runnable":
        return
    generation_roles = {str(job["dataset_role"]) for job in profile["generation_jobs"]}
    expected_roles = {
        "smoke": {"smoke"},
        "calibration": {"development"},
        "selection": {"validation"},
        "regression": {"development"},
        "benchmark": {"test"},
    }[str(profile["purpose"])]
    if generation_roles != expected_roles:
        raise BenchmarkConfigError(
            f"Runnable {profile['purpose']} profile {profile['id']} must contain exactly "
            f"dataset roles {sorted(expected_roles)}, found {sorted(generation_roles)}"
        )
    selected: dict[str, Any] | None = None
    if profile["purpose"] in {"regression", "benchmark"}:
        raw_selected = profile.get("selected_config")
        raw_manifest = profile.get("materialization_manifest")
        if not isinstance(raw_selected, str) or not isinstance(raw_manifest, str):
            raise BenchmarkConfigError(
                f"Profile {profile['id']} cannot become runnable before SPATHI selection is "
                "frozen and recorded in a materialization manifest"
            )
        selected_path = resolve_repo_path(
            repo_root, raw_selected, f"{profile['id']} selected configuration"
        )
        _search, selected = validate_calibration_contracts(repo_root, selected_path)
        if selected["status"] != "selected" or selected["phase"] != "frozen":
            raise BenchmarkConfigError(
                f"Profile {profile['id']} cannot become runnable before SPATHI selection is frozen"
            )
        manifest_path = resolve_repo_path(
            repo_root, raw_manifest, f"{profile['id']} materialization manifest"
        )
        materialization = load_json_object(manifest_path, "benchmark materialization manifest")
        manifest_selection = materialization.get("selected_config")
        if (
            materialization.get("profile_id") != profile["id"]
            or not isinstance(manifest_selection, dict)
            or manifest_selection.get("path") != raw_selected
            or manifest_selection.get("sha256") != sha256_file(selected_path)
        ):
            raise BenchmarkConfigError(
                f"Profile {profile['id']} differs from its materialization manifest"
            )
    statistics = profile["statistics"]
    if profile["purpose"] in {"calibration", "selection"}:
        if statistics["status"] != "not_applicable":
            raise BenchmarkConfigError(
                f"Runnable {profile['purpose']} profile {profile['id']} must mark the "
                "generic statistics stage not_applicable; calibration-rank owns selection"
            )
    elif statistics["status"] != "enabled":
        raise BenchmarkConfigError(
            f"Runnable profile {profile['id']} must have an enabled statistics contract"
        )
    comparison_ids = [str(item["id"]) for item in statistics["comparisons"]]
    if len(set(comparison_ids)) != len(comparison_ids):
        raise BenchmarkConfigError("Statistics contract contains duplicate comparison IDs")
    confirmatory = [
        item for item in statistics["comparisons"] if item["comparison_role"] == "confirmatory"
    ]
    if profile["purpose"] == "benchmark" and len(confirmatory) != 1:
        raise BenchmarkConfigError(
            "Runnable benchmark profiles require exactly one confirmatory comparison"
        )
    if any(item["regime"] != "group_only" for item in confirmatory):
        raise BenchmarkConfigError("Confirmatory comparisons must use the group_only regime")
    if any(item["candidate_space"] != "tf_restricted" for item in confirmatory):
        raise BenchmarkConfigError("Confirmatory comparisons must use the TF-restricted space")
    if any(
        item["primary_metric"] != "aupr" or item["primary_level"] != "directed"
        for item in confirmatory
    ):
        raise BenchmarkConfigError(
            "Confirmatory comparisons must use directed AUPR as the primary endpoint"
        )
    configuration_routes: dict[str, set[tuple[str, str, frozenset[str]]]] = {}
    configuration_families: dict[str, set[str]] = {}
    for job in profile["generation_jobs"]:
        scenario_path = resolve_repo_path(repo_root, job["scenario"], f"{job['id']} scenario")
        runs_path = resolve_repo_path(
            repo_root, job["simulator_runs"], f"{job['id']} simulator runs"
        )
        scenario = load_json_object(scenario_path)
        runs = load_json_object(runs_path)
        if scenario.get("schema_version") != "1.0" or not scenario.get("id"):
            raise BenchmarkConfigError(f"Invalid ANDREA scenario contract: {scenario_path}")
        if not isinstance(runs.get("runs"), list) or not runs["runs"]:
            raise BenchmarkConfigError(f"Invalid ANDREA simulator-runs contract: {runs_path}")

    for batch in profile["inference_batches"]:
        if batch["dataset_role"] not in generation_roles:
            raise BenchmarkConfigError(
                f"Inference batch {batch['id']} selects dataset role "
                f"{batch['dataset_role']!r}, but no generation job provides it"
            )
        params_path = resolve_repo_path(
            repo_root, batch["tools_params"], f"{batch['id']} tools params"
        )
        custom_path = None
        custom_payload = None
        if batch.get("custom_tools"):
            custom_path = resolve_repo_path(
                repo_root, batch["custom_tools"], f"{batch['id']} custom tools"
            )
            custom_payload = load_json_object(custom_path)
            image_tag = load_image_lock(repo_root)["images"]["spathi"]["tag"]
            definitions = custom_payload.get("tools")
            if isinstance(definitions, list):
                for definition in definitions:
                    if isinstance(definition, dict) and definition.get("docker_image") != image_tag:
                        raise BenchmarkConfigError(
                            "SPATHI custom tool image differs from images.lock.json"
                        )
        params_payload = load_json_object(params_path)
        _validate_run_identity(
            tools_params=params_payload,
            custom_tools=custom_payload,
            label=str(params_path),
        )
        metadata = load_run_metadata(
            repo_root,
            batch["run_metadata"],
            f"{batch['id']} run metadata",
        )
        batch_generation_jobs = _resolve_generation_job_selector(
            owner=f"Inference batch {batch['id']}",
            selected_jobs=batch.get("generation_jobs"),
            allowed_roles={str(batch["dataset_role"])},
            generation_roles_by_id=generation_roles_by_id,
        )
        for item in metadata["runs"]:
            configuration_id = str(item["configuration_id"])
            configuration_routes.setdefault(configuration_id, set()).add(
                (
                    str(batch["dataset_role"]),
                    str(batch["regime"]),
                    batch_generation_jobs,
                )
            )
            family = item.get("family")
            if isinstance(family, str) and family:
                configuration_families.setdefault(configuration_id, set()).add(family)
        requested_ids = {str(run["run_id"]) for run in params_payload["runs"]}
        metadata_ids = {str(run["run_id"]) for run in metadata["runs"]}
        if requested_ids != metadata_ids:
            raise BenchmarkConfigError(
                f"{batch['id']} run metadata mismatch: "
                f"requested={sorted(requested_ids)}, metadata={sorted(metadata_ids)}"
            )
        params_by_run = {str(run["run_id"]): run.get("params") for run in params_payload["runs"]}
        for item in metadata["runs"]:
            run_id = str(item["run_id"])
            params = params_by_run[run_id]
            if not isinstance(params, dict):
                raise BenchmarkConfigError(f"{batch['id']} run {run_id!r} has invalid params")
            seed_parameter = item["algorithm_seed_parameter"]
            if seed_parameter is not None and params.get(seed_parameter) != item["algorithm_seed"]:
                raise BenchmarkConfigError(
                    f"{batch['id']} run {run_id!r} declares algorithm seed "
                    f"{item['algorithm_seed']!r} in {seed_parameter!r}, but params contain "
                    f"{params.get(seed_parameter)!r}"
                )
        custom_ids = {
            str(run["run_id"])
            for run in params_payload["runs"]
            if str(run.get("tool_id", "")).startswith("custom_")
        }
        if batch["kind"] == "baseline" and custom_ids:
            raise BenchmarkConfigError(f"Baseline batch {batch['id']} contains custom tools")
        if batch["kind"] == "baseline" and batch.get("calibration_manifest"):
            raise BenchmarkConfigError(
                f"Baseline batch {batch['id']} cannot use a calibration manifest"
            )
        if batch["kind"] == "spathi" and custom_ids != requested_ids:
            raise BenchmarkConfigError(f"SPATHI batch {batch['id']} must contain only custom tools")
        metadata_roles = {str(item["selection_role"]) for item in metadata["runs"]}
        if batch["kind"] == "baseline" and metadata_roles != {"baseline"}:
            raise BenchmarkConfigError(
                f"Baseline batch {batch['id']} must use selection_role=baseline"
            )
        if batch["kind"] == "spathi" and "baseline" in metadata_roles:
            raise BenchmarkConfigError(
                f"SPATHI batch {batch['id']} cannot use selection_role=baseline"
            )
        if batch["dataset_role"] == "test" and batch["kind"] == "spathi":
            if metadata_roles != {"selected"}:
                raise BenchmarkConfigError(
                    f"Test batch {batch['id']} may execute only a frozen selected "
                    "SPATHI configuration"
                )
            if selected is None:
                raise BenchmarkConfigError(
                    f"Test batch {batch['id']} has no frozen selected configuration"
                )
            if selected["status"] != "selected" or selected["phase"] != "frozen":
                raise BenchmarkConfigError(
                    f"Test batch {batch['id']} requires a frozen selected-config.json"
                )
            selected_candidate = selected["selected_candidate"]
            selected_parameters = selected_candidate["parameters"]
            selected_digest = canonical_json_sha256(selected_parameters)
            if (
                selected_candidate["spathi_commit"]
                != load_repo_lock(repo_root)["repositories"]["spathi"]["commit"]
                or selected_candidate["configuration_sha256"] != selected_digest
            ):
                raise BenchmarkConfigError(
                    f"Test batch {batch['id']} selected configuration has stale provenance"
                )
            if {str(item["configuration_id"]) for item in metadata["runs"]} != {
                str(selected_candidate["candidate_id"])
            }:
                raise BenchmarkConfigError(
                    f"Test batch {batch['id']} does not request the frozen SPATHI candidate"
                )
            for item in metadata["runs"]:
                params = dict(params_by_run[str(item["run_id"])])
                seed_parameter = item["algorithm_seed_parameter"]
                if seed_parameter is not None:
                    params.pop(seed_parameter)
                if params != selected_parameters:
                    raise BenchmarkConfigError(
                        f"Test batch {batch['id']} parameters differ from the frozen selection"
                    )
        if profile["purpose"] in {"calibration", "selection"} and batch["kind"] == "spathi":
            raw_manifest = batch.get("calibration_manifest")
            if not isinstance(raw_manifest, str):
                raise BenchmarkConfigError(
                    f"SPATHI batch {batch['id']} requires calibration_manifest"
                )
            manifest = load_json_object(
                resolve_repo_path(
                    repo_root,
                    raw_manifest,
                    f"{batch['id']} calibration manifest",
                ),
                f"{batch['id']} calibration manifest",
            )
            validate_json(
                manifest,
                repo_root / "schemas" / "calibration-manifest.schema.json",
                f"{batch['id']} calibration manifest",
            )
            from .calibration import validate_calibration_manifest

            search = load_json_object(
                repo_root / "experiments/v1/spathi/search-space.json",
                "SPATHI search space",
            )
            candidates, manifest_runs = validate_calibration_manifest(
                repo_root,
                search,
                manifest,
            )
            expected_role = "development" if profile["purpose"] == "calibration" else "validation"
            if manifest.get("dataset_role") != expected_role:
                raise BenchmarkConfigError(
                    f"{batch['id']} calibration manifest has the wrong dataset role"
                )
            if manifest.get("profile_id") != profile["id"]:
                raise BenchmarkConfigError(
                    f"{batch['id']} calibration manifest belongs to another profile"
                )
            if manifest.get("calibration_budget") != profile.get("calibration_budget"):
                raise BenchmarkConfigError(
                    f"{batch['id']} calibration manifest uses another evidence budget"
                )
            if manifest.get("stage_id") != profile.get("calibration_stage"):
                raise BenchmarkConfigError(
                    f"{batch['id']} calibration manifest has the wrong profile stage"
                )
            manifest_run_ids = {run.run_id for run in manifest_runs}
            if manifest_run_ids != requested_ids:
                raise BenchmarkConfigError(
                    f"{batch['id']} calibration manifest run IDs differ from its request"
                )
            candidate_digests = {
                candidate.configuration_id: candidate.configuration_sha256
                for candidate in candidates
            }
            for item in metadata["runs"]:
                run_id = str(item["run_id"])
                configuration_id = str(item["configuration_id"])
                params = dict(params_by_run[run_id])
                seed_parameter = item["algorithm_seed_parameter"]
                if seed_parameter is not None:
                    params.pop(seed_parameter)
                if candidate_digests.get(configuration_id) != canonical_json_sha256(params):
                    raise BenchmarkConfigError(
                        f"{batch['id']} run {run_id!r} differs from its calibration candidate"
                    )

    for comparison in statistics["comparisons"]:
        configurations = [str(item) for item in comparison["configuration_ids"]]
        control = str(comparison["control_configuration_id"])
        if control not in configurations:
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']} control is not in configuration_ids"
            )
        if configuration_families.get(control) != {"spathi"}:
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']} control must be a SPATHI configuration"
            )
        if comparison["primary_metric"] not in comparison["metrics"]:
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']} primary metric is not selected"
            )
        if comparison["primary_level"] not in comparison["levels"]:
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']} primary level is not selected"
            )
        selected_generation_jobs = _resolve_generation_job_selector(
            owner=f"Statistics comparison {comparison['id']}",
            selected_jobs=comparison.get("generation_jobs"),
            allowed_roles={str(role) for role in comparison["dataset_roles"]},
            generation_roles_by_id=generation_roles_by_id,
        )
        for configuration_id in configurations:
            observed_routes = configuration_routes.get(configuration_id, set())
            covered_generation_jobs = frozenset(
                job_id
                for dataset_role, regime, generation_jobs in observed_routes
                if dataset_role in comparison["dataset_roles"] and regime == comparison["regime"]
                for job_id in generation_jobs
            )
            missing_generation_jobs = sorted(selected_generation_jobs - covered_generation_jobs)
            if missing_generation_jobs:
                raise BenchmarkConfigError(
                    f"Statistics comparison {comparison['id']} configuration "
                    f"{configuration_id!r} has no matching inference batch coverage for "
                    f"generation jobs {missing_generation_jobs} in regime "
                    f"{comparison['regime']!r}"
                )


CALIBRATION_BUDGET_PROFILE_CONTRACT = {
    "full": {
        "replicates_per_condition": 3,
        "dataset_count_per_role": 15,
    },
    "reduced": {
        "replicates_per_condition": 1,
        "dataset_count_per_role": 5,
    },
}


def _calibration_profile_runs(
    repo_root: Path,
    profile: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    runs: list[dict[str, Any]] = []
    for job in profile["generation_jobs"]:
        payload = load_json_object(
            resolve_repo_path(
                repo_root,
                str(job["simulator_runs"]),
                f"{profile['id']}:{job['id']} simulator runs",
            )
        )
        raw_runs = payload.get("runs")
        if (
            not isinstance(raw_runs, list)
            or not raw_runs
            or not all(isinstance(run, dict) for run in raw_runs)
        ):
            raise BenchmarkConfigError(
                f"Calibration profile {profile['id']} has an invalid simulator-runs request"
            )
        runs.extend(raw_runs)
    return tuple(runs)


def _validate_calibration_budget_profiles(
    repo_root: Path,
    profiles_by_id: dict[str, dict[str, Any]],
) -> None:
    expected_conditions = {
        "linear_baseline",
        "linear_noisy",
        "bifurcating_baseline",
        "bifurcating_noisy",
        "scmultisim_dynamic",
    }
    for budget, contract in CALIBRATION_BUDGET_PROFILE_CONTRACT.items():
        for purpose, role, prefix in (
            ("calibration", "development", "calibration"),
            ("selection", "validation", "selection"),
        ):
            profile_id = f"{prefix}-{budget}"
            profile = profiles_by_id[profile_id]
            if profile.get("purpose") != purpose or profile.get("calibration_budget") != budget:
                raise BenchmarkConfigError(
                    f"Profile {profile_id} must bind purpose={purpose!r} and "
                    f"calibration_budget={budget!r}"
                )
            jobs = profile["generation_jobs"]
            if len(jobs) != len(expected_conditions):
                raise BenchmarkConfigError(
                    f"Profile {profile_id} must contain exactly five calibration conditions"
                )
            if {str(job["dataset_role"]) for job in jobs} != {role}:
                raise BenchmarkConfigError(
                    f"Profile {profile_id} must contain only {role} datasets"
                )
            if {str(job["condition_id"]) for job in jobs} != expected_conditions:
                raise BenchmarkConfigError(
                    f"Profile {profile_id} does not cover the frozen calibration conditions"
                )
            expected_replicates = int(contract["replicates_per_condition"])
            if profile["planned_design"]["replicates_per_condition"] != expected_replicates:
                raise BenchmarkConfigError(
                    f"Profile {profile_id} declares the wrong replicate budget"
                )
            generation_parallelism = profile["resources"]["generate_data"]["max_parallel_tasks"]
            if generation_parallelism > expected_replicates:
                raise BenchmarkConfigError(
                    f"Profile {profile_id} generation parallelism exceeds each request's "
                    "replicate budget"
                )
            runs = _calibration_profile_runs(repo_root, profile)
            replicate_counts = [run.get("replicates") for run in runs]
            if not replicate_counts or any(
                isinstance(count, bool) or count != expected_replicates
                for count in replicate_counts
            ):
                raise BenchmarkConfigError(
                    f"Profile {profile_id} simulator requests differ from its replicate budget"
                )
            if sum(int(count) for count in replicate_counts) != int(
                contract["dataset_count_per_role"]
            ):
                raise BenchmarkConfigError(
                    f"Profile {profile_id} has the wrong total dataset budget"
                )

    for purpose, prefix in (("calibration", "calibration"), ("selection", "selection")):
        full_profile = profiles_by_id[f"{prefix}-full"]
        reduced_profile = profiles_by_id[f"{prefix}-reduced"]
        full_jobs = {str(job["id"]): job for job in full_profile["generation_jobs"]}
        reduced_jobs = {str(job["id"]): job for job in reduced_profile["generation_jobs"]}
        if full_jobs.keys() != reduced_jobs.keys():
            raise BenchmarkConfigError(
                f"Full and reduced {purpose} profiles must cover the same generation jobs"
            )
        for job_id, full_job in full_jobs.items():
            reduced_job = reduced_jobs[job_id]
            for field in ("scenario", "dataset_role", "split_id", "condition_id"):
                if full_job[field] != reduced_job[field]:
                    raise BenchmarkConfigError(
                        f"Full and reduced {purpose} job {job_id} differ in {field}"
                    )
            full_runs = load_json_object(
                resolve_repo_path(
                    repo_root,
                    str(full_job["simulator_runs"]),
                    f"full {purpose} job {job_id}",
                )
            )["runs"]
            reduced_runs = load_json_object(
                resolve_repo_path(
                    repo_root,
                    str(reduced_job["simulator_runs"]),
                    f"reduced {purpose} job {job_id}",
                )
            )["runs"]

            def without_budget_fields(run: dict[str, Any]) -> dict[str, Any]:
                return {
                    key: value for key, value in run.items() if key not in {"replicates", "notes"}
                }

            if [without_budget_fields(run) for run in full_runs] != [
                without_budget_fields(run) for run in reduced_runs
            ]:
                raise BenchmarkConfigError(
                    f"Reduced {purpose} job {job_id} is not a one-replicate subset of full"
                )


def validate_all_profiles(repo_root: Path) -> list[dict[str, Any]]:
    profiles = []
    for path in sorted((repo_root / "experiments" / "v1" / "profiles").glob("*.json")):
        profile = load_profile(repo_root, path.stem)
        validate_profile_references(repo_root, profile)
        profiles.append(profile)
    required_profiles = {
        "smoke",
        "reduced",
        "full",
        "calibration-full",
        "selection-full",
        "calibration-reduced",
        "selection-reduced",
    }
    profile_ids = {profile["id"] for profile in profiles}
    legacy_profiles = {"calibration", "selection"} & profile_ids
    if legacy_profiles:
        raise BenchmarkConfigError(
            f"Legacy unbudgeted calibration profiles are not allowed: {sorted(legacy_profiles)}"
        )
    if not required_profiles.issubset(profile_ids):
        raise BenchmarkConfigError(
            f"Required profiles are missing: {sorted(required_profiles - profile_ids)}"
        )
    _validate_calibration_budget_profiles(
        repo_root,
        {str(profile["id"]): profile for profile in profiles},
    )
    from .dataset_partitions import validate_dataset_partition_exclusivity

    validate_dataset_partition_exclusivity(repo_root, profiles)
    return profiles


REGIME_IDS = {"group_only", "order_aware", "pseudotime_aware", "prior_assisted"}


def validate_calibration_contracts(
    repo_root: Path,
    selected_config_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the bounded SPATHI search and its frozen-selection record."""

    search_path = repo_root / "experiments" / "v1" / "spathi" / "search-space.json"
    selected_path = (
        selected_config_path.resolve()
        if selected_config_path is not None
        else repo_root / "experiments" / "v1" / "spathi" / "selected-config.json"
    )
    if not selected_path.is_relative_to(repo_root.resolve()):
        raise BenchmarkConfigError("SPATHI selected configuration escapes repository root")
    search = load_json_object(search_path, "SPATHI search space")
    selected = load_json_object(selected_path, "SPATHI selected configuration")
    validate_json(
        search,
        repo_root / "schemas" / "spathi-search-space.schema.json",
        "SPATHI search space",
    )
    validate_json(
        selected,
        repo_root / "schemas" / "spathi-selected-config.schema.json",
        "SPATHI selected configuration",
    )
    expected_search = search_path.relative_to(repo_root).as_posix()
    if selected["search_space"] != expected_search:
        raise BenchmarkConfigError(
            "SPATHI selected configuration points to a different search space: "
            f"{selected['search_space']!r}"
        )
    if selected["status"] == "selected":
        from .calibration import reconcile_frozen_selection

        reconcile_frozen_selection(repo_root, search, selected)
    return search, selected


def validate_design_contracts(repo_root: Path, andrea_repo: Path | None = None) -> None:
    validate_calibration_contracts(repo_root)
    regimes: dict[str, dict[str, Any]] = {}
    regime_root = repo_root / "experiments" / "v1" / "inference" / "regimes"
    for path in sorted(regime_root.glob("*.json")):
        regime = load_json_object(path, f"regime {path.stem}")
        validate_json(regime, repo_root / "schemas" / "regime.schema.json", f"regime {path.stem}")
        if regime.get("id") != path.stem:
            raise BenchmarkConfigError(f"Regime ID does not match filename: {path}")
        variants = [
            (item["tool_id"], item["execution_mode"], item["variant"])
            for item in regime["planned_variants"]
        ]
        if len(set(variants)) != len(variants):
            raise BenchmarkConfigError(f"Regime {path.stem} contains duplicate variants")
        regimes[path.stem] = regime
    if set(regimes) != REGIME_IDS:
        raise BenchmarkConfigError(
            f"Expected regimes {sorted(REGIME_IDS)}, found {sorted(regimes)}"
        )
    if not regimes["group_only"]["primary"] or any(
        regimes[regime_id]["primary"] for regime_id in REGIME_IDS - {"group_only"}
    ):
        raise BenchmarkConfigError("group_only must be the only primary regime")

    for profile in validate_all_profiles(repo_root):
        referenced = set(profile["planned_design"]["regimes"])
        referenced.update(batch["regime"] for batch in profile["inference_batches"])
        if not referenced.issubset(regimes):
            raise BenchmarkConfigError(f"Profile {profile['id']} references an unknown regime")

    matrix_path = repo_root / "experiments" / "v1" / "inference" / "catalog-matrix.json"
    matrix = load_json_object(matrix_path, "catalog matrix")
    validate_json(
        matrix,
        repo_root / "schemas" / "catalog-matrix.schema.json",
        "catalog matrix",
    )
    locked_commit = load_repo_lock(repo_root)["repositories"]["andrea"]["commit"]
    if matrix["andrea_commit"] != locked_commit:
        raise BenchmarkConfigError("Catalog matrix ANDREA commit differs from repos.lock.json")
    matrix_ids = [entry["tool_id"] for entry in matrix["entries"]]
    if len(set(matrix_ids)) != len(matrix_ids):
        raise BenchmarkConfigError("Catalog matrix contains duplicate tool IDs")
    if andrea_repo is not None:
        catalog_root = andrea_repo / "andrea" / "catalog_inference_tools" / "tools"
        catalog_ids = {path.parent.name for path in catalog_root.glob("*/toolspec.json")}
        if set(matrix_ids) != catalog_ids:
            raise BenchmarkConfigError(
                f"Catalog matrix mismatch: missing={sorted(catalog_ids - set(matrix_ids))}, "
                f"extra={sorted(set(matrix_ids) - catalog_ids)}"
            )
        for entry in matrix["entries"]:
            toolspec = load_json_object(
                catalog_root / entry["tool_id"] / "toolspec.json",
                f"{entry['tool_id']} ToolSpec",
            )
            capabilities = set(toolspec.get("execution_capabilities", []))
            unsupported = sorted(
                {
                    route["execution_mode"]
                    for route in entry["routes"]
                    if route["execution_mode"] not in capabilities
                }
            )
            if unsupported:
                raise BenchmarkConfigError(
                    f"Catalog matrix gives {entry['tool_id']} unsupported modes: {unsupported}"
                )

    regime_routes = {
        (
            regime_id,
            variant["tool_id"],
            variant["execution_mode"],
            variant["variant"],
            variant["role"],
            variant["status"],
            variant["tf_list_handling"],
        )
        for regime_id, regime in regimes.items()
        for variant in regime["planned_variants"]
        if variant["tool_id"] != "spathi"
    }
    matrix_routes = {
        (
            route["regime"],
            entry["tool_id"],
            route["execution_mode"],
            route["variant"],
            route["role"],
            route["status"],
            entry["tf_list_handling"],
        )
        for entry in matrix["entries"]
        for route in entry["routes"]
    }
    if regime_routes != matrix_routes:
        raise BenchmarkConfigError(
            "Regime variants and catalog matrix routes differ: "
            f"only_in_regimes={sorted(regime_routes - matrix_routes)}, "
            f"only_in_matrix={sorted(matrix_routes - regime_routes)}"
        )
