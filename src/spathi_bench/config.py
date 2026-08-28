"""Load and validate benchmark-owned configuration."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


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
    return lock


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


def validate_profile_references(repo_root: Path, profile: dict[str, Any]) -> None:
    generation_ids = [job["id"] for job in profile["generation_jobs"]]
    if len(set(generation_ids)) != len(generation_ids):
        raise BenchmarkConfigError("Profile contains duplicate generation job IDs")
    batch_ids = [batch["id"] for batch in profile["inference_batches"]]
    if len(set(batch_ids)) != len(batch_ids):
        raise BenchmarkConfigError("Profile contains duplicate inference batch IDs")
    if profile.get("status") != "runnable":
        return
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
                    if (
                        isinstance(definition, dict)
                        and definition.get("run_id") == "spathi_primary"
                        and definition.get("docker_image") != image_tag
                    ):
                        raise BenchmarkConfigError(
                            "SPATHI custom tool image differs from images.lock.json"
                        )
        _validate_run_identity(
            tools_params=load_json_object(params_path),
            custom_tools=custom_payload,
            label=str(params_path),
        )


def validate_all_profiles(repo_root: Path) -> list[dict[str, Any]]:
    profiles = []
    for path in sorted((repo_root / "experiments" / "v1" / "profiles").glob("*.json")):
        profile = load_profile(repo_root, path.stem)
        validate_profile_references(repo_root, profile)
        profiles.append(profile)
    if {profile["id"] for profile in profiles} != {"smoke", "reduced", "full"}:
        raise BenchmarkConfigError("Exactly smoke, reduced and full profiles must be defined")
    return profiles


REGIME_IDS = {"group_only", "order_aware", "pseudotime_aware", "prior_assisted"}


def validate_design_contracts(repo_root: Path, andrea_repo: Path | None = None) -> None:
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
