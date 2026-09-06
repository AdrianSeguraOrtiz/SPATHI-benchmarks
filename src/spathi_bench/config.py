"""Load and validate benchmark-owned configuration."""

from __future__ import annotations

import json
import os
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
    from .safe_io import path_below_no_symlinks, reject_symlink_components

    root = reject_symlink_components(repo_root, "repository root")
    if not root.is_dir():
        raise BenchmarkConfigError(f"Repository root does not exist: {root}")
    candidate = Path(os.path.abspath(root / raw_path))
    if not candidate.is_relative_to(root):
        raise BenchmarkConfigError(f"{label} escapes repository root: {raw_path}")
    candidate = path_below_no_symlinks(candidate, root, label)
    if not candidate.is_file():
        raise BenchmarkConfigError(f"{label} does not exist: {raw_path}")
    return candidate


def load_profile(
    repo_root: Path,
    profile_id: str,
    *,
    require_runnable: bool = False,
) -> dict[str, Any]:
    relative_path = profile_path(Path(), profile_id)
    path = resolve_repo_path(repo_root, relative_path.as_posix(), f"profile {profile_id}")
    profile = load_json_object(path, f"profile {profile_id}")
    schema_path = resolve_repo_path(
        repo_root,
        "schemas/profile.schema.json",
        "profile schema",
    )
    validate_json(profile, schema_path, f"profile {profile_id}")
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

    from .safe_io import path_below_no_symlinks, reject_symlink_components

    root = reject_symlink_components(repo_root, "repository root")
    if not root.is_dir():
        raise BenchmarkConfigError(f"Repository root does not exist: {root}")
    candidate = Path(os.path.abspath(profile_file))
    if not candidate.is_relative_to(root):
        raise BenchmarkConfigError(f"Profile file escapes repository root: {profile_file}")
    safe_profile = path_below_no_symlinks(
        candidate,
        root,
        "profile file",
        require_file=True,
    )
    if require_runtime:
        work_root = path_below_no_symlinks(root / "work", root, "runtime work root")
        if not safe_profile.is_relative_to(work_root):
            raise BenchmarkConfigError(
                f"Runtime profile must be stored below work/: {profile_file}"
            )
    profile = load_json_object(safe_profile, f"profile {profile_id}")
    validate_json(
        profile,
        resolve_repo_path(root, "schemas/profile.schema.json", "profile schema"),
        f"profile {profile_id}",
    )
    if profile.get("id") != profile_id:
        raise BenchmarkConfigError(
            f"Profile ID {profile.get('id')!r} does not match requested ID {profile_id!r}"
        )
    if require_runnable and profile.get("status") != "runnable":
        raise BenchmarkConfigError(f"Profile {profile_id!r} is not runnable")
    return profile


def load_repo_lock(repo_root: Path) -> dict[str, Any]:
    lock_path = resolve_repo_path(repo_root, "repos.lock.json", "repository lock")
    schema_path = resolve_repo_path(
        repo_root,
        "schemas/repos-lock.schema.json",
        "repository lock schema",
    )
    lock = load_json_object(lock_path, "repository lock")
    validate_json(lock, schema_path, "repository lock")
    return lock


def load_image_lock(repo_root: Path) -> dict[str, Any]:
    lock = load_json_object(
        resolve_repo_path(repo_root, "images.lock.json", "image lock"),
        "image lock",
    )
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
    from .safe_io import reject_symlink_components

    safe_path = reject_symlink_components(path, f"locked repository {name}")
    if not safe_path.is_dir():
        raise BenchmarkConfigError(f"Locked repository {name} is missing: {path}")
    head = _git(safe_path, "rev-parse", "HEAD")
    expected = str(lock_entry["commit"])
    if head != expected:
        raise BenchmarkConfigError(
            f"{name} HEAD mismatch: expected {expected}, found {head} at {path}"
        )
    origin = _git(safe_path, "remote", "get-url", "origin")
    expected_origin = str(lock_entry["url"])
    if _canonical_remote(origin) != _canonical_remote(expected_origin):
        raise BenchmarkConfigError(
            f"{name} origin mismatch: expected {expected_origin}, found {origin}"
        )
    status = _git(safe_path, "status", "--porcelain")
    if require_clean and status:
        raise BenchmarkConfigError(f"{name} worktree is not clean: {path}")
    return {"path": str(safe_path), "commit": head, "origin": origin, "clean": not status}


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
        "regression": {"test"},
        "benchmark": {"test"},
        "efficiency": {"efficiency_validation"},
        "efficiency_safety": {"efficiency_safety"},
        "scalability": {"scalability"},
        "dimensional_transfer": {"dimensional_transfer"},
    }[str(profile["purpose"])]
    if generation_roles != expected_roles:
        raise BenchmarkConfigError(
            f"Runnable {profile['purpose']} profile {profile['id']} must contain exactly "
            f"dataset roles {sorted(expected_roles)}, found {sorted(generation_roles)}"
        )
    production: dict[str, Any] | None = None
    raw_production = profile.get("production_config")
    if raw_production is not None and profile["purpose"] not in {
        "regression",
        "benchmark",
        "scalability",
        "dimensional_transfer",
    }:
        raise BenchmarkConfigError(f"Profile {profile['id']} cannot declare a production policy")
    if profile["purpose"] in {"regression", "benchmark"} or raw_production is not None:
        raw_manifest = profile.get("materialization_manifest")
        if not isinstance(raw_production, str):
            raise BenchmarkConfigError(
                f"Profile {profile['id']} requires an efficiency-gated SPATHI production "
                "policy path"
            )
        if profile["purpose"] in {"regression", "benchmark"} and not isinstance(raw_manifest, str):
            raise BenchmarkConfigError(
                f"Profile {profile['id']} requires a production-policy materialization manifest"
            )
        production_path = resolve_repo_path(
            repo_root, raw_production, f"{profile['id']} production policy"
        )
        production = load_json_object(production_path, "SPATHI production policy")
        validate_json(
            production,
            repo_root / "schemas/spathi-production-config.schema.json",
            "SPATHI production policy",
        )
        if isinstance(raw_manifest, str):
            manifest_path = resolve_repo_path(
                repo_root, raw_manifest, f"{profile['id']} materialization manifest"
            )
            materialization = load_json_object(manifest_path, "benchmark materialization manifest")
            manifest_policy = materialization.get("production_config")
            if (
                materialization.get("profile_id") != profile["id"]
                or not isinstance(manifest_policy, dict)
                or manifest_policy.get("path") != raw_production
                or manifest_policy.get("sha256") != sha256_file(production_path)
            ):
                raise BenchmarkConfigError(
                    f"Profile {profile['id']} differs from its materialization manifest"
                )
    statistics = profile["statistics"]
    if profile["purpose"] in {"calibration", "selection"}:
        if statistics["status"] != "not_applicable":
            raise BenchmarkConfigError(
                f"Runnable {profile['purpose']} profile {profile['id']} must mark the "
                "generic statistics stage not_applicable; the calibration workflow owns selection"
            )
    elif profile["purpose"] in {
        "efficiency",
        "efficiency_safety",
        "scalability",
        "dimensional_transfer",
    }:
        if statistics["status"] != "not_applicable" or statistics["comparisons"]:
            raise BenchmarkConfigError(
                f"Runnable {profile['purpose']} profiles use their dedicated analyzer"
            )
    elif statistics["status"] != (
        "inferential" if profile["purpose"] == "benchmark" else "descriptive"
    ):
        raise BenchmarkConfigError(
            f"Runnable {profile['purpose']} profile {profile['id']} has the wrong analysis mode"
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
    from .inference_scope_contract import inference_regulator_scopes

    scope_contracts = {
        str(item["id"]): inference_regulator_scopes(item) for item in statistics["comparisons"]
    }
    if any(
        any(scope != "tf_restricted" for scope in scope_contracts[str(item["id"])].values())
        for item in confirmatory
    ):
        raise BenchmarkConfigError(
            "Confirmatory comparisons must restrict inference regulators to the common TF list"
        )
    if any(
        item["primary_metric"] != "aupr" or item["primary_level"] != "directed"
        for item in confirmatory
    ):
        raise BenchmarkConfigError(
            "Confirmatory comparisons must use directed AUPR as the primary endpoint"
        )
    configuration_routes: dict[str, set[tuple[str, str, frozenset[str]]]] = {}
    configuration_families: dict[str, set[str]] = {}
    configuration_seed_routes: dict[tuple[str, str, str], set[int | None]] = {}
    for job in profile["generation_jobs"]:
        if "generator_request" in job:
            from .scale_stress import validate_scale_stress_request

            request_path = resolve_repo_path(
                repo_root,
                job["generator_request"],
                f"{job['id']} scale-stress generator request",
            )
            validate_scale_stress_request(repo_root, request_path)
            continue
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
            configuration_seed_routes.setdefault(
                (
                    configuration_id,
                    str(batch["dataset_role"]),
                    str(batch["regime"]),
                ),
                set(),
            ).add(item["algorithm_seed"])
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
        is_calibration_batch = (
            profile["purpose"] in {"calibration", "selection"} and batch["kind"] == "spathi"
        )
        if not is_calibration_batch and any(
            item.get("configuration_sha256") is not None for item in metadata["runs"]
        ):
            raise BenchmarkConfigError(
                f"Only calibration batches may declare logical configuration digests: {batch['id']}"
            )
        if batch["kind"] == "baseline" and metadata_roles != {"baseline"}:
            raise BenchmarkConfigError(
                f"Baseline batch {batch['id']} must use selection_role=baseline"
            )
        if batch["kind"] == "spathi" and "baseline" in metadata_roles:
            raise BenchmarkConfigError(
                f"SPATHI batch {batch['id']} cannot use selection_role=baseline"
            )
        selected_production_batch = batch["dataset_role"] in {"test", "scalability"} or (
            batch["dataset_role"] == "dimensional_transfer" and metadata_roles == {"selected"}
        )
        if selected_production_batch and batch["kind"] == "spathi":
            if metadata_roles != {"selected"}:
                raise BenchmarkConfigError(
                    f"Production batch {batch['id']} may execute only a frozen selected "
                    "SPATHI configuration"
                )
            if production is None:
                raise BenchmarkConfigError(
                    f"Production batch {batch['id']} has no production policy"
                )
            selected_policy = production["policy"]
            group_policy = selected_policy["group_policy"]
            selected_digest = canonical_json_sha256(group_policy)
            if (
                selected_policy["spathi_commit"]
                != load_repo_lock(repo_root)["repositories"]["spathi"]["commit"]
                or selected_policy["policy_sha256"] != selected_digest
            ):
                raise BenchmarkConfigError(
                    f"Production batch {batch['id']} selected policy has stale provenance"
                )
            if {str(item["configuration_id"]) for item in metadata["runs"]} != {
                str(selected_policy["policy_id"])
            }:
                raise BenchmarkConfigError(
                    f"Production batch {batch['id']} does not request the frozen SPATHI policy"
                )
            for item in metadata["runs"]:
                params = dict(params_by_run[str(item["run_id"])])
                seed_parameter = item["algorithm_seed_parameter"]
                if seed_parameter is not None:
                    params.pop(seed_parameter)
                if params != {"group_policy": group_policy}:
                    raise BenchmarkConfigError(
                        f"Production batch {batch['id']} parameters differ from the frozen policy"
                    )
        if (
            batch["dataset_role"] == "dimensional_transfer"
            and batch["kind"] == "spathi"
            and metadata_roles != {"selected"}
        ):
            if metadata_roles != {"base"} or production is None:
                raise BenchmarkConfigError(
                    f"Dimensional diagnostic batch {batch['id']} must be the exact base policy"
                )
            provenance = production["provenance"]["efficiency_plan"]
            plan_path = resolve_repo_path(
                repo_root,
                str(provenance["path"]),
                f"{batch['id']} efficiency plan",
            )
            if sha256_file(plan_path) != provenance["sha256"]:
                raise BenchmarkConfigError("Dimensional exact-policy provenance has changed")
            plan = load_json_object(plan_path, "dimensional exact-policy efficiency plan")
            validate_json(
                plan,
                repo_root / "schemas/noninferiority-plan.schema.json",
                "dimensional exact-policy efficiency plan",
            )
            exact = plan["reference"]
            if {str(item["configuration_id"]) for item in metadata["runs"]} != {
                "spathi_exact_reference"
            }:
                raise BenchmarkConfigError(
                    "Dimensional diagnostic must use configuration_id=spathi_exact_reference"
                )
            for item in metadata["runs"]:
                params = dict(params_by_run[str(item["run_id"])])
                seed_parameter = item["algorithm_seed_parameter"]
                if seed_parameter is not None:
                    params.pop(seed_parameter)
                if params != {"group_policy": exact["group_policy"]}:
                    raise BenchmarkConfigError(
                        f"Dimensional diagnostic batch {batch['id']} differs from the exact policy"
                    )
        if is_calibration_batch:
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
            _, manifest_runs = validate_calibration_manifest(
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
            canonical_runs = {run.run_id: run for run in manifest_runs}
            for item in metadata["runs"]:
                run_id = str(item["run_id"])
                configuration_id = str(item["configuration_id"])
                canonical_run = canonical_runs[run_id]
                if (
                    configuration_id != canonical_run.configuration_id
                    or item.get("configuration_sha256") != canonical_run.configuration_sha256
                    or item["algorithm_seed"] != canonical_run.algorithm_seed
                    or params_by_run[run_id] != dict(canonical_run.parameters)
                ):
                    raise BenchmarkConfigError(
                        f"{batch['id']} run {run_id!r} differs from its canonical calibration run"
                    )

    from .algorithm_seed_contract import algorithm_seeds_by_configuration

    for comparison in statistics["comparisons"]:
        configurations = [str(item) for item in comparison["configuration_ids"]]
        expected_seeds = algorithm_seeds_by_configuration(comparison)
        control = str(comparison["control_configuration_id"])
        if control not in configurations:
            raise BenchmarkConfigError(
                f"Statistics comparison {comparison['id']} control is not in configuration_ids"
            )
        implementation_sensitivity = (
            comparison["id"] == "full-pyscenic-grn-implementation-sensitivity"
            and comparison["comparison_role"] == "exploratory"
            and configurations == ["grnboost2_default", "pyscenic_default"]
            and control == "grnboost2_default"
        )
        if implementation_sensitivity:
            if any(
                configuration_families.get(configuration_id) != {"catalog_tf_restricted"}
                for configuration_id in configurations
            ):
                raise BenchmarkConfigError(
                    "The pySCENIC implementation sensitivity must compare the two locked "
                    "catalog implementations"
                )
        elif configuration_families.get(control) != {"spathi"}:
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
            observed_seeds = set().union(
                *(
                    configuration_seed_routes.get(
                        (configuration_id, str(dataset_role), str(comparison["regime"])),
                        set(),
                    )
                    for dataset_role in comparison["dataset_roles"]
                )
            )
            if observed_seeds != set(expected_seeds[configuration_id]):
                raise BenchmarkConfigError(
                    f"Statistics comparison {comparison['id']} configuration "
                    f"{configuration_id!r} declares algorithm seeds "
                    f"{sorted(expected_seeds[configuration_id], key=lambda seed: -1 if seed is None else seed)}, "
                    "but its inference metadata expose "
                    f"{sorted(observed_seeds, key=lambda seed: -1 if seed is None else seed)}"
                )
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
        "ordinary_dataset_count": 24,
        "scale_transfer_dataset_count": 3,
    },
    "reduced": {
        "replicates_per_condition": 1,
        "ordinary_dataset_count": 8,
        "scale_transfer_dataset_count": 1,
    },
}


def _profile_runs(
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
                f"Profile {profile['id']} has an invalid simulator-runs request"
            )
        runs.extend(raw_runs)
    return tuple(runs)


def _validate_calibration_budget_profiles(
    repo_root: Path,
    profiles_by_id: dict[str, dict[str, Any]],
) -> None:
    ordinary_conditions = {
        "linear_baseline",
        "linear_noisy",
        "bifurcating_baseline",
        "bifurcating_noisy",
        "scmultisim_dynamic",
        "scmultisim_imbalanced",
        "scmultisim_single_group",
        "scmultisim_medium",
    }
    scale_transfer_condition = "scmultisim_estimator_scale_transfer"
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
            expected_conditions = set(ordinary_conditions)
            if purpose == "calibration":
                expected_conditions.add(scale_transfer_condition)
            if len(jobs) != len(expected_conditions):
                raise BenchmarkConfigError(
                    f"Profile {profile_id} has the wrong number of calibration conditions"
                )
            if {str(job["dataset_role"]) for job in jobs} != {role}:
                raise BenchmarkConfigError(
                    f"Profile {profile_id} must contain only {role} datasets"
                )
            if {str(job["condition_id"]) for job in jobs} != expected_conditions:
                raise BenchmarkConfigError(
                    f"Profile {profile_id} does not cover the frozen calibration conditions"
                )
            if purpose == "calibration":
                scale_jobs = [
                    job for job in jobs if job["condition_id"] == scale_transfer_condition
                ]
                ordinary_jobs = [job for job in jobs if job not in scale_jobs]
                if (
                    len(scale_jobs) != 1
                    or scale_jobs[0]["id"] != "scmultisim_development_estimator_scale_transfer"
                    or scale_jobs[0]["split_id"] != "development_estimator_scale_transfer_v1"
                    or scale_jobs[0]["scenario"]
                    != "experiments/v1/simulations/scmultisim/estimator_scale_transfer/scenario.json"
                    or any(job["split_id"] != "development_v1" for job in ordinary_jobs)
                ):
                    raise BenchmarkConfigError(
                        f"Profile {profile_id} does not isolate the estimator scale-transfer panel"
                    )
            elif any(job["split_id"] != "validation_v1" for job in jobs):
                raise BenchmarkConfigError(
                    f"Profile {profile_id} must use only the frozen validation split"
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
            runs = _profile_runs(repo_root, profile)
            replicate_counts = [run.get("replicates") for run in runs]
            if not replicate_counts or any(
                isinstance(count, bool) or count != expected_replicates
                for count in replicate_counts
            ):
                raise BenchmarkConfigError(
                    f"Profile {profile_id} simulator requests differ from its replicate budget"
                )
            expected_dataset_count = int(contract["ordinary_dataset_count"])
            if purpose == "calibration":
                expected_dataset_count += int(contract["scale_transfer_dataset_count"])
            if sum(int(count) for count in replicate_counts) != expected_dataset_count:
                raise BenchmarkConfigError(
                    f"Profile {profile_id} has the wrong total dataset budget"
                )

            if purpose == "calibration":
                scale_job = next(
                    job for job in jobs if job["condition_id"] == scale_transfer_condition
                )
                scale_scenario = load_json_object(
                    resolve_repo_path(
                        repo_root,
                        str(scale_job["scenario"]),
                        f"{profile_id} estimator scale-transfer scenario",
                    )
                )
                scale_runs = load_json_object(
                    resolve_repo_path(
                        repo_root,
                        str(scale_job["simulator_runs"]),
                        f"{profile_id} estimator scale-transfer runs",
                    )
                ).get("runs")
                scenario_inputs = scale_scenario.get("inputs")
                regulatory_network = (
                    scenario_inputs.get("regulatory_network")
                    if isinstance(scenario_inputs, dict)
                    else None
                )
                tree_newick = (
                    scenario_inputs.get("tree_newick")
                    if isinstance(scenario_inputs, dict)
                    else None
                )
                if (
                    scale_scenario.get("id") != "spathi_scmultisim_estimator_scale_transfer_v1"
                    or not isinstance(regulatory_network, dict)
                    or regulatory_network.get("path") != "regulatory-network.tsv"
                    or not isinstance(tree_newick, dict)
                    or tree_newick.get("path") != "tree-three-groups.nwk"
                    or not isinstance(scale_runs, list)
                    or len(scale_runs) != 1
                ):
                    raise BenchmarkConfigError(
                        f"Profile {profile_id} has an invalid estimator scale-transfer scenario"
                    )
                params = scale_runs[0].get("params")
                population = params.get("discrete_population") if isinstance(params, dict) else None
                if (
                    not isinstance(params, dict)
                    or params.get("num_cells") != 300
                    or params.get("num_genes") != 2_200
                    or not isinstance(population, dict)
                    or population.get("pop_size") != [100, 100, 100]
                ):
                    raise BenchmarkConfigError(
                        f"Profile {profile_id} changes the estimator scale-transfer dimensions"
                    )
                network_path = resolve_repo_path(
                    repo_root,
                    "experiments/v1/simulations/scmultisim/estimator_scale_transfer/regulatory-network.tsv",
                    f"{profile_id} estimator scale-transfer topology",
                )
                regulators = {
                    fields[1]
                    for line in network_path.read_text(encoding="utf-8").splitlines()[1:]
                    if len(fields := line.split("\t")) == 3
                }
                if len(regulators) != 1_000:
                    raise BenchmarkConfigError(
                        f"Profile {profile_id} estimator scale-transfer topology must expose 1,000 TFs"
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
                    key: value
                    for key, value in run.items()
                    if key not in {"run_id", "replicates", "base_seed", "notes"}
                }

            if [without_budget_fields(run) for run in full_runs] != [
                without_budget_fields(run) for run in reduced_runs
            ]:
                raise BenchmarkConfigError(
                    f"Full and reduced {purpose} job {job_id} differ scientifically"
                )
            full_seeds = {
                int(run["base_seed"]) + offset
                for run in full_runs
                for offset in range(int(run["replicates"]))
            }
            reduced_seeds = {
                int(run["base_seed"]) + offset
                for run in reduced_runs
                for offset in range(int(run["replicates"]))
            }
            if full_seeds & reduced_seeds:
                raise BenchmarkConfigError(
                    f"Full and reduced {purpose} job {job_id} reuse generation seeds"
                )

    scale_network = resolve_repo_path(
        repo_root,
        "experiments/v1/simulations/scmultisim/estimator_scale_transfer/regulatory-network.tsv",
        "estimator scale-transfer topology",
    )
    scale_digest = sha256_file(scale_network)
    for profile_id in ("calibration-full", "selection-full"):
        for job in profiles_by_id[profile_id]["generation_jobs"]:
            if job["condition_id"] == scale_transfer_condition:
                continue
            scenario_path = resolve_repo_path(
                repo_root,
                str(job["scenario"]),
                f"{profile_id}:{job['id']} scenario",
            )
            scenario = load_json_object(scenario_path, f"{profile_id}:{job['id']} scenario")
            inputs = scenario.get("inputs")
            network = inputs.get("regulatory_network") if isinstance(inputs, dict) else None
            raw_path = network.get("path") if isinstance(network, dict) else None
            if not isinstance(raw_path, str) or not raw_path:
                continue
            from .safe_io import path_below_no_symlinks

            try:
                candidate = path_below_no_symlinks(
                    Path(os.path.abspath(scenario_path.parent / raw_path)),
                    repo_root,
                    f"{profile_id}:{job['id']} regulatory-network input",
                    require_file=True,
                )
            except BenchmarkConfigError as exc:
                raise BenchmarkConfigError(
                    f"Profile {profile_id} has an invalid regulatory-network input"
                ) from exc
            if sha256_file(candidate) == scale_digest:
                raise BenchmarkConfigError(
                    "Estimator scale-transfer topology is reused by another calibration panel"
                )


def _validate_efficiency_budget_profiles(
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
    expected_safety: dict[str, dict[str, Any]] = {
        "g3000_tf100_c300": {
            "job_id": "scmultisim_efficiency_safety_g3000_tf100_c300",
            "scenario": "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-tf100.json",
            "runs": "experiments/v1/simulations/scmultisim/efficiency_safety/g3000-tf100-c300-runs.json",
            "scenario_id": "spathi_scmultisim_efficiency_safety_tf100_v1",
            "regulatory_network": "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-tf100.tsv",
            "cells": 300,
            "genes": 3_000,
            "transcription_factors": 100,
            "targets_in_regulatory_core": 1_000,
            "cifs": 50,
            "group_sizes": [100, 100, 100],
        },
        "g100_tf10_c500_single_group": {
            "job_id": "scmultisim_efficiency_safety_g100_tf10_c500_single_group",
            "scenario": "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-tf10-single-group.json",
            "runs": "experiments/v1/simulations/scmultisim/efficiency_safety/g100-tf10-c500-single-group-runs.json",
            "scenario_id": "spathi_scmultisim_efficiency_safety_tf10_single_group_v1",
            "regulatory_network": "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-tf10.tsv",
            "cells": 500,
            "genes": 100,
            "transcription_factors": 10,
            "targets_in_regulatory_core": 50,
            "cifs": 10,
            "group_sizes": [500],
        },
        "g100_tf10_c300_groups_10_40_250": {
            "job_id": "scmultisim_efficiency_safety_g100_tf10_c300_groups_10_40_250",
            "scenario": "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-tf10.json",
            "runs": "experiments/v1/simulations/scmultisim/efficiency_safety/g100-tf10-c300-groups-10-40-250-runs.json",
            "scenario_id": "spathi_scmultisim_efficiency_safety_tf10_v1",
            "regulatory_network": "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-tf10.tsv",
            "cells": 300,
            "genes": 100,
            "transcription_factors": 10,
            "targets_in_regulatory_core": 50,
            "cifs": 10,
            "group_sizes": [10, 40, 250],
        },
        "g100_tf10_c10000": {
            "job_id": "scmultisim_efficiency_safety_g100_tf10_c10000",
            "scenario": "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-tf10.json",
            "runs": "experiments/v1/simulations/scmultisim/efficiency_safety/g100-tf10-c10000-runs.json",
            "scenario_id": "spathi_scmultisim_efficiency_safety_tf10_v1",
            "regulatory_network": "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-tf10.tsv",
            "cells": 10_000,
            "genes": 100,
            "transcription_factors": 10,
            "targets_in_regulatory_core": 50,
            "cifs": 10,
            "group_sizes": [3_333, 3_333, 3_334],
        },
        "g3000_tf1500_c300": {
            "job_id": "scmultisim_efficiency_safety_g3000_tf1500_c300",
            "scenario": "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-g3000-tf1500.json",
            "runs": "experiments/v1/simulations/scmultisim/efficiency_safety/g3000-tf1500-c300-runs.json",
            "scenario_id": "spathi_scmultisim_efficiency_safety_g3000_tf1500_v1",
            "regulatory_network": "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-g3000-tf1500.tsv",
            "cells": 300,
            "genes": 3_000,
            "transcription_factors": 1_500,
            "targets_in_regulatory_core": 1_000,
            "cifs": 50,
            "group_sizes": [100, 100, 100],
        },
        "g15000_tf100_c300": {
            "job_id": "scmultisim_efficiency_safety_g15000_tf100_c300",
            "scenario": "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-g15000-tf100.json",
            "runs": "experiments/v1/simulations/scmultisim/efficiency_safety/g15000-tf100-c300-runs.json",
            "scenario_id": "spathi_scmultisim_efficiency_safety_g15000_tf100_v1",
            "regulatory_network": "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-g15000-tf100.tsv",
            "cells": 300,
            "genes": 15_000,
            "transcription_factors": 100,
            "targets_in_regulatory_core": 1_000,
            "cifs": 50,
            "group_sizes": [100, 100, 100],
        },
    }
    profiles: dict[str, dict[str, Any]] = {}
    for budget, expected_replicates in (("reduced", 1), ("full", 5)):
        profile_id = f"efficiency-{budget}"
        profile = profiles_by_id[profile_id]
        profiles[budget] = profile
        jobs = profile["generation_jobs"]
        validation_jobs = [job for job in jobs if job["dataset_role"] == "efficiency_validation"]
        safety_jobs = [job for job in jobs if job["dataset_role"] == "efficiency_safety"]
        if (
            profile["purpose"] != "efficiency"
            or len(validation_jobs) != len(expected_conditions)
            or {str(job["condition_id"]) for job in validation_jobs} != expected_conditions
            or safety_jobs
            or profile["planned_design"]["replicates_per_condition"] != expected_replicates
            or profile["resources"]["infer_network"]["max_parallel_runs"] != 1
            or "replicate_exceptions" in profile["planned_design"]
        ):
            raise BenchmarkConfigError(
                f"Profile {profile_id} differs from the predeclared efficiency budget"
            )
        validation_profile = {**profile, "generation_jobs": validation_jobs}
        runs = _profile_runs(repo_root, validation_profile)
        if len(runs) != len(expected_conditions) or any(
            run.get("replicates") != expected_replicates for run in runs
        ):
            raise BenchmarkConfigError(
                f"Profile {profile_id} has the wrong efficiency replicate budget"
            )
    full_jobs = {
        str(job["id"]): job
        for job in profiles["full"]["generation_jobs"]
        if job["dataset_role"] == "efficiency_validation"
    }
    reduced_jobs = {str(job["id"]): job for job in profiles["reduced"]["generation_jobs"]}
    if full_jobs.keys() != reduced_jobs.keys():
        raise BenchmarkConfigError("Full and reduced efficiency profiles cover different jobs")
    for job_id, full_job in full_jobs.items():
        reduced_job = reduced_jobs[job_id]
        for field in ("scenario", "dataset_role", "split_id", "condition_id"):
            if full_job[field] != reduced_job[field]:
                raise BenchmarkConfigError(
                    f"Full and reduced efficiency job {job_id} differ in {field}"
                )
        full_runs = load_json_object(
            resolve_repo_path(repo_root, str(full_job["simulator_runs"]), job_id)
        )["runs"]
        reduced_runs = load_json_object(
            resolve_repo_path(repo_root, str(reduced_job["simulator_runs"]), job_id)
        )["runs"]

        def strip_budget(run: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in run.items()
                if key not in {"run_id", "replicates", "base_seed", "notes"}
            }

        if [strip_budget(run) for run in full_runs] != [strip_budget(run) for run in reduced_runs]:
            raise BenchmarkConfigError(
                f"Full and reduced efficiency job {job_id} differ scientifically"
            )
        full_seeds = {
            int(run["base_seed"]) + offset
            for run in full_runs
            for offset in range(int(run["replicates"]))
        }
        reduced_seeds = {
            int(run["base_seed"]) + offset
            for run in reduced_runs
            for offset in range(int(run["replicates"]))
        }
        if full_seeds & reduced_seeds:
            raise BenchmarkConfigError(
                f"Full and reduced efficiency job {job_id} reuse generation seeds"
            )

    safety_profile = profiles_by_id["efficiency-safety-full"]
    safety_jobs = safety_profile["generation_jobs"]
    if (
        safety_profile["status"] != "runnable"
        or safety_profile["purpose"] != "efficiency_safety"
        or safety_profile["inference_batches"]
        or safety_profile["statistics"] != {"status": "not_applicable", "comparisons": []}
        or safety_profile["planned_design"]["replicates_per_condition"] != 3
        or "replicate_exceptions" in safety_profile["planned_design"]
        or safety_profile["resources"]["generate_data"]["max_parallel_tasks"] != 1
        or safety_profile["resources"]["infer_network"]["max_parallel_runs"] != 1
        or len(safety_jobs) != len(expected_safety)
        or {str(job["condition_id"]) for job in safety_jobs} != set(expected_safety)
    ):
        raise BenchmarkConfigError(
            "Profile efficiency-safety-full differs from the conditional safety budget"
        )
    for safety in safety_jobs:
        condition_id = str(safety["condition_id"])
        expected = expected_safety[condition_id]
        if (
            safety["id"] != expected["job_id"]
            or safety["scenario"] != expected["scenario"]
            or safety["simulator_runs"] != expected["runs"]
            or safety["split_id"] != "efficiency_safety_v1"
        ):
            raise BenchmarkConfigError(
                f"Efficiency-safety profile changes condition {condition_id}"
            )
        scenario = load_json_object(
            resolve_repo_path(repo_root, str(safety["scenario"]), condition_id),
            f"efficiency-safety scenario {condition_id}",
        )
        raw_runs = load_json_object(
            resolve_repo_path(repo_root, str(safety["simulator_runs"]), condition_id),
            f"efficiency-safety runs {condition_id}",
        ).get("runs")
        if (
            scenario.get("id") != expected["scenario_id"]
            or not isinstance(raw_runs, list)
            or len(raw_runs) != 1
        ):
            raise BenchmarkConfigError(f"Efficiency-safety inputs are invalid for {condition_id}")
        scenario_inputs = scenario.get("inputs")
        scenario_network = (
            scenario_inputs.get("regulatory_network") if isinstance(scenario_inputs, dict) else None
        )
        if (
            not isinstance(scenario_network, dict)
            or scenario_network.get("path") != Path(expected["regulatory_network"]).name
        ):
            raise BenchmarkConfigError(
                f"Efficiency-safety scenario changes the topology for {condition_id}"
            )
        safety_run = raw_runs[0]
        params = safety_run.get("params")
        population = params.get("discrete_population") if isinstance(params, dict) else None
        if (
            safety_run.get("run_id") != expected["job_id"]
            or safety_run.get("simulator_id") != "scmultisim"
            or safety_run.get("replicates") != 3
            or not isinstance(params, dict)
            or params.get("num_cells") != expected["cells"]
            or params.get("num_genes") != expected["genes"]
            or params.get("num_cifs") != expected["cifs"]
            or not isinstance(population, dict)
            or population.get("pop_size") != expected["group_sizes"]
        ):
            raise BenchmarkConfigError(
                f"Efficiency-safety dimensions are invalid for {condition_id}"
            )

    def directed_topology(relative: str) -> set[tuple[str, str]]:
        path = resolve_repo_path(repo_root, relative, "protected regulatory topology")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise BenchmarkConfigError(
                f"Cannot read protected regulatory topology: {path}"
            ) from exc
        if not lines or lines[0] != "target\tregulator\teffect":
            raise BenchmarkConfigError(
                f"Protected regulatory topology has an invalid header: {path}"
            )
        edges: set[tuple[str, str]] = set()
        for line in lines[1:]:
            fields = line.split("\t")
            if len(fields) != 3 or not fields[0] or not fields[1]:
                raise BenchmarkConfigError(
                    f"Protected regulatory topology contains an invalid row: {path}"
                )
            edge = (fields[0], fields[1])
            if edge in edges:
                raise BenchmarkConfigError(
                    f"Protected regulatory topology contains duplicate directed edges: {path}"
                )
            edges.add(edge)
        return edges

    safety_topologies: dict[str, set[tuple[str, str]]] = {}
    for expected in expected_safety.values():
        relative = str(expected["regulatory_network"])
        if relative not in safety_topologies:
            safety_topologies[relative] = directed_topology(relative)
        edges = safety_topologies[relative]
        targets = {target for target, _regulator in edges}
        regulators = {regulator for _target, regulator in edges}
        if (
            len(targets) != expected["targets_in_regulatory_core"]
            or len(regulators) != expected["transcription_factors"]
            or len(edges) != 2 * expected["targets_in_regulatory_core"]
        ):
            raise BenchmarkConfigError(
                f"Efficiency-safety topology dimensions are invalid: {relative}"
            )

    safety_items = sorted(safety_topologies.items())
    for index, (left_path, left_edges) in enumerate(safety_items):
        for right_path, right_edges in safety_items[index + 1 :]:
            if left_edges & right_edges:
                raise BenchmarkConfigError(
                    "Efficiency-safety topologies share directed edges: "
                    f"{left_path} and {right_path}"
                )

    transfer_paths = (
        "experiments/v1/simulations/scmultisim/dimensional_transfer/regulatory-network-tf10.tsv",
        "experiments/v1/simulations/scmultisim/dimensional_transfer/regulatory-network-tf100.tsv",
        "experiments/v1/simulations/scmultisim/dimensional_transfer/regulatory-network-tf1500.tsv",
    )
    transfer_edges = set().union(*(directed_topology(path) for path in transfer_paths))
    if set().union(*safety_topologies.values()) & transfer_edges:
        raise BenchmarkConfigError(
            "Efficiency-safety and dimensional-transfer topologies share directed edges"
        )


def _validate_dimensional_transfer_profiles(
    repo_root: Path,
    profiles_by_id: dict[str, dict[str, Any]],
) -> None:
    """Bind both transfer budgets to the declared full or operational panel."""

    from .dimensional_transfer import condition_ids_for_budget, validate_transfer_panel

    panel = validate_transfer_panel(repo_root)
    expected_conditions = {str(item["id"]): item for item in panel["conditions"]}
    profiles: dict[str, dict[str, Any]] = {}
    for budget, expected_datasets in (("reduced", 4), ("full", 9)):
        profile_id = f"dimensional-transfer-{budget}"
        profile = profiles_by_id[profile_id]
        profiles[budget] = profile
        jobs = profile["generation_jobs"]
        jobs_by_condition = {str(job["condition_id"]): job for job in jobs}
        budget_condition_ids = set(condition_ids_for_budget(budget))
        expected_exceptions = {"g3000_tf100_c300": 3} if budget == "full" else None
        if (
            profile["purpose"] != "dimensional_transfer"
            or profile["inference_batches"]
            or profile["statistics"] != {"status": "not_applicable", "comparisons": []}
            or len(jobs) != len(budget_condition_ids)
            or set(jobs_by_condition) != budget_condition_ids
            or {str(job["dataset_role"]) for job in jobs} != {"dimensional_transfer"}
            or profile["planned_design"]["replicates_per_condition"] != 1
            or profile["planned_design"].get("replicate_exceptions") != expected_exceptions
            or profile["resources"]["generate_data"]
            != {"max_parallel_tasks": 1, "max_cores": 8, "max_ram_gb": 48}
            or profile["resources"]["infer_network"]["max_parallel_runs"] != 1
        ):
            raise BenchmarkConfigError(
                f"Profile {profile_id} differs from the frozen dimensional-transfer budget"
            )
        total = 0
        for condition_id in condition_ids_for_budget(budget):
            condition = expected_conditions[condition_id]
            job = jobs_by_condition[condition_id]
            if (
                job["scenario"] != condition["scenario"]
                or job["simulator_runs"] != condition["runs"][budget]
                or job["split_id"] != f"dimensional_transfer_{budget}_v1"
            ):
                raise BenchmarkConfigError(
                    f"Profile {profile_id} condition {condition_id} differs from its panel"
                )
            runs = load_json_object(
                resolve_repo_path(repo_root, str(job["simulator_runs"]), condition_id)
            )["runs"]
            expected_replicates = int(condition["replicates"][budget])
            if len(runs) != 1 or runs[0].get("replicates") != expected_replicates:
                raise BenchmarkConfigError(
                    f"Profile {profile_id} condition {condition_id} has the wrong replication"
                )
            total += expected_replicates
        if total != expected_datasets:
            raise BenchmarkConfigError(
                f"Profile {profile_id} has {total} datasets, expected {expected_datasets}"
            )

    full_by_condition = {
        str(job["condition_id"]): job for job in profiles["full"]["generation_jobs"]
    }
    reduced_by_condition = {
        str(job["condition_id"]): job for job in profiles["reduced"]["generation_jobs"]
    }
    for condition_id in condition_ids_for_budget("reduced"):
        full_job = full_by_condition[condition_id]
        reduced_job = reduced_by_condition[condition_id]
        if full_job["scenario"] != reduced_job["scenario"]:
            raise BenchmarkConfigError(
                f"Dimensional-transfer condition {condition_id} changes scenario by budget"
            )
        full_run = load_json_object(repo_root / full_job["simulator_runs"])["runs"][0]
        reduced_run = load_json_object(repo_root / reduced_job["simulator_runs"])["runs"][0]

        def scientific_request(run: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in run.items()
                if key not in {"run_id", "replicates", "base_seed", "notes"}
            }

        if scientific_request(full_run) != scientific_request(reduced_run):
            raise BenchmarkConfigError(
                f"Dimensional-transfer condition {condition_id} differs scientifically by budget"
            )
        full_seeds = {
            int(full_run["base_seed"]) + offset for offset in range(int(full_run["replicates"]))
        }
        reduced_seeds = {
            int(reduced_run["base_seed"]) + offset
            for offset in range(int(reduced_run["replicates"]))
        }
        if full_seeds & reduced_seeds:
            raise BenchmarkConfigError(
                f"Dimensional-transfer condition {condition_id} reuses generation seeds"
            )


def validate_all_profiles(repo_root: Path) -> list[dict[str, Any]]:
    profiles = []
    for path in sorted((repo_root / "experiments" / "v1" / "profiles").glob("*.json")):
        profile = load_profile(repo_root, path.stem)
        validate_profile_references(repo_root, profile)
        inference = profile["resources"]["infer_network"]
        parallel_runs = int(inference["max_parallel_runs"])
        if parallel_runs * int(inference["max_cores"]) > 8:
            raise BenchmarkConfigError(
                f"Profile {profile['id']} inference can exceed the global eight-core ceiling"
            )
        if parallel_runs * float(inference["max_ram_gb"]) > 48:
            raise BenchmarkConfigError(
                f"Profile {profile['id']} inference can exceed the global 48 GB ceiling"
            )
        profiles.append(profile)
    required_profiles = {
        "smoke",
        "reduced",
        "full",
        "calibration-full",
        "selection-full",
        "calibration-reduced",
        "selection-reduced",
        "efficiency-full",
        "efficiency-reduced",
        "efficiency-development-full",
        "efficiency-development-reduced",
        "efficiency-safety-full",
        "scalability-full",
        "scalability-reduced",
        "dimensional-transfer-full",
        "dimensional-transfer-reduced",
    }
    profile_ids = {profile["id"] for profile in profiles}
    if not required_profiles.issubset(profile_ids):
        raise BenchmarkConfigError(
            f"Required profiles are missing: {sorted(required_profiles - profile_ids)}"
        )
    _validate_calibration_budget_profiles(
        repo_root,
        {str(profile["id"]): profile for profile in profiles},
    )
    _validate_efficiency_budget_profiles(
        repo_root,
        {str(profile["id"]): profile for profile in profiles},
    )
    _validate_dimensional_transfer_profiles(
        repo_root,
        {str(profile["id"]): profile for profile in profiles},
    )
    from .dataset_partitions import validate_dataset_partition_exclusivity

    validate_dataset_partition_exclusivity(repo_root, profiles)
    return profiles


REGIME_IDS = {"group_only", "order_aware", "prior_assisted"}


def validate_calibration_contracts(
    repo_root: Path,
    selected_config_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the bounded SPATHI search and its frozen-selection record."""

    search_path = repo_root / "experiments" / "v1" / "spathi" / "search-space.json"
    from .safe_io import path_below_no_symlinks, reject_symlink_components

    root = reject_symlink_components(repo_root, "repository root")
    selected_path = Path(
        os.path.abspath(
            selected_config_path
            if selected_config_path is not None
            else root / "experiments" / "v1" / "spathi" / "selected-config.json"
        )
    )
    if not selected_path.is_relative_to(root):
        raise BenchmarkConfigError("SPATHI selected configuration escapes repository root")
    selected_path = path_below_no_symlinks(
        selected_path,
        root,
        "SPATHI selected configuration",
        require_file=True,
    )
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
    # Scan tracked and non-ignored untracked material before interpreting any
    # scientific contract. The lazy import avoids a module-level cycle because
    # the publication boundary reports failures as BenchmarkConfigError.
    from .publication_boundary import validate_publication_boundary

    validate_publication_boundary(repo_root)
    # Seed integrity is a mandatory design gate, not merely an optional report.
    # The lazy import keeps the generic configuration primitives acyclic.
    from .reproducibility import ReproducibilityAuditError, audit_reproducibility

    try:
        audit_reproducibility(repo_root)
    except ReproducibilityAuditError as exc:
        raise BenchmarkConfigError(f"Reproducibility audit failed: {exc}") from exc
    search, _selected = validate_calibration_contracts(repo_root)
    accountability_path = (
        repo_root / "experiments" / "v1" / "spathi" / "parameter-accountability.json"
    )
    accountability = load_json_object(
        accountability_path,
        "SPATHI parameter accountability",
    )
    validate_json(
        accountability,
        repo_root / "schemas" / "spathi-parameter-accountability.schema.json",
        "SPATHI parameter accountability",
    )
    parameter_ids = [item["id"] for item in accountability["inference_parameters"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise BenchmarkConfigError("SPATHI parameter accountability contains duplicate IDs")
    expected_parameter_ids = {*search["base_configuration"], "random_seed"}
    if set(parameter_ids) != expected_parameter_ids:
        raise BenchmarkConfigError(
            "SPATHI parameter accountability differs from the executable inference contract: "
            f"missing={sorted(expected_parameter_ids - set(parameter_ids))}, "
            f"extra={sorted(set(parameter_ids) - expected_parameter_ids)}"
        )
    # Imported lazily because the study helper itself uses the generic JSON
    # loading and validation primitives from this module.
    from .noninferiority import load_noninferiority_study

    load_noninferiority_study(repo_root)
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
    from .catalog_contracts import validate_catalog_plan_coverage

    validate_catalog_plan_coverage(repo_root, matrix)
