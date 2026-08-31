"""Post-selection materialization of runnable regression and benchmark profiles.

Catalog requests are deliberately supplied as a separate, committed plan.  This
module owns only the selected SPATHI requests and the assembly/reconciliation of
both sides into one runnable profile.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .calibration import locked_spathi_identity
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    load_profile,
    load_repo_lock,
    load_run_metadata,
    resolve_repo_path,
    validate_calibration_contracts,
    validate_json,
    validate_profile_references,
)

SELECTED_PLACEHOLDER = "selected_spathi"
_PROFILE_IDS = {"reduced": "reduced-v1", "full": "full-v1"}
_PLAN_KEYS = {"schema_version", "template_profile_id", "baseline_batches", "statistics"}
_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _repo_relative_path(repo_root: Path, path: Path, label: str) -> str:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} must be inside the benchmark repository") from exc


def _estimator_seeds(search_space: Mapping[str, Any]) -> tuple[int, ...]:
    stages = search_space.get("stages")
    if not isinstance(stages, list):
        raise BenchmarkConfigError("SPATHI search space contains no stages")
    estimator = next(
        (
            stage
            for stage in stages
            if isinstance(stage, Mapping) and stage.get("id") == "estimator"
        ),
        None,
    )
    if estimator is None or not isinstance(estimator.get("algorithmic_seeds"), list):
        raise BenchmarkConfigError("Estimator stage contains no predeclared algorithmic seeds")
    raw_seeds = estimator["algorithmic_seeds"]
    if not raw_seeds or not all(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in raw_seeds
    ):
        raise BenchmarkConfigError("Estimator algorithmic seeds must be non-negative integers")
    seeds = tuple(raw_seeds)
    if len(set(seeds)) != len(seeds):
        raise BenchmarkConfigError("Estimator algorithmic seeds contain duplicates")
    return seeds


def load_frozen_selected_candidate(
    repo_root: Path,
    selected_config_path: Path | None = None,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Return the reconciled selected candidate and its predeclared execution seeds."""

    root = repo_root.resolve()
    if selected_config_path is None:
        search, selected = validate_calibration_contracts(root)
    else:
        search, selected = validate_calibration_contracts(root, selected_config_path)
    if selected.get("status") != "selected" or selected.get("phase") != "frozen":
        raise BenchmarkConfigError(
            "Benchmark materialization requires a frozen SPATHI selected-config.json"
        )
    candidate = selected.get("selected_candidate")
    if not isinstance(candidate, dict):
        raise BenchmarkConfigError("Frozen SPATHI selection contains no selected candidate")
    parameters = candidate.get("parameters")
    if not isinstance(parameters, dict):
        raise BenchmarkConfigError("Frozen SPATHI candidate contains no parameters")
    if candidate.get("configuration_sha256") != canonical_json_sha256(parameters):
        raise BenchmarkConfigError("Frozen SPATHI candidate parameters have a stale digest")
    locked_commit = load_repo_lock(root)["repositories"]["spathi"]["commit"]
    if candidate.get("spathi_commit") != locked_commit:
        raise BenchmarkConfigError(
            "Frozen SPATHI candidate targets a different repos.lock.json commit"
        )
    return copy.deepcopy(candidate), _estimator_seeds(search)


def build_selected_spathi_requests(
    candidate: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    docker_image: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Render one frozen candidate over the already predeclared estimator seeds."""

    candidate_id = candidate.get("candidate_id")
    parameters = candidate.get("parameters")
    if not isinstance(candidate_id, str) or not _ID_RE.fullmatch(candidate_id):
        raise BenchmarkConfigError("Selected SPATHI candidate has an invalid ID")
    if not isinstance(parameters, Mapping):
        raise BenchmarkConfigError("Selected SPATHI candidate has invalid parameters")
    if not docker_image:
        raise BenchmarkConfigError("Selected SPATHI requests require a Docker image")
    normalized_seeds = tuple(seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise BenchmarkConfigError("Selected SPATHI requests require unique predeclared seeds")

    runs = [
        {
            "run_id": f"spathi_selected__seed_{seed}",
            "tool_id": f"custom_spathi_selected__seed_{seed}",
            "params": {**copy.deepcopy(dict(parameters)), "random_seed": seed},
            "execution": {"mode": "group_native"},
        }
        for seed in normalized_seeds
    ]
    custom_tools = {
        "tools": [
            {
                "run_id": run["run_id"],
                "name": f"SPATHI selected {candidate_id} seed {seed}",
                "docker_image": docker_image,
                "execution_mode": "group_native",
                "extra_inputs": ["groups", "tf_list"],
                "outputs": {"directed": True, "sign": "none"},
            }
            for run, seed in zip(runs, normalized_seeds, strict=True)
        ]
    }
    metadata = {
        "schema_version": "1.0",
        "runs": [
            {
                "run_id": run["run_id"],
                "configuration_id": candidate_id,
                "selection_role": "selected",
                "algorithm_seed": seed,
                "algorithm_seed_parameter": "random_seed",
                "family": "spathi",
            }
            for run, seed in zip(runs, normalized_seeds, strict=True)
        ],
    }
    return {"runs": runs}, custom_tools, metadata


def _load_catalog_plan(
    repo_root: Path,
    plan_path: Path,
    *,
    template_profile_id: str,
) -> tuple[dict[str, Any], str]:
    root = repo_root.resolve()
    relative = _repo_relative_path(root, plan_path, "Catalog plan")
    resolved = resolve_repo_path(root, relative, "catalog plan")
    plan = load_json_object(resolved, "catalog benchmark plan")
    unknown = set(plan) - _PLAN_KEYS
    missing = _PLAN_KEYS - set(plan)
    if unknown or missing:
        raise BenchmarkConfigError(
            f"Catalog plan keys differ from its contract: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if plan["schema_version"] != "1.0":
        raise BenchmarkConfigError("Catalog plan schema_version must be '1.0'")
    if plan["template_profile_id"] != template_profile_id:
        raise BenchmarkConfigError("Catalog plan targets a different template profile")
    batches = plan["baseline_batches"]
    statistics = plan["statistics"]
    if (
        not isinstance(batches, list)
        or not batches
        or not all(isinstance(batch, dict) for batch in batches)
    ):
        raise BenchmarkConfigError("Catalog plan requires non-empty baseline_batches")
    if not isinstance(statistics, dict) or statistics.get("status") != "enabled":
        raise BenchmarkConfigError("Catalog plan requires an enabled statistics contract")
    comparisons = statistics.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise BenchmarkConfigError("Catalog plan requires at least one statistical comparison")
    return plan, relative


def _replace_selected_placeholder(
    statistics: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(statistics))
    comparisons = payload.get("comparisons")
    assert isinstance(comparisons, list)
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise BenchmarkConfigError("Catalog plan contains an invalid comparison")
        configurations = comparison.get("configuration_ids")
        if not isinstance(configurations, list) or SELECTED_PLACEHOLDER not in configurations:
            raise BenchmarkConfigError(
                f"Comparison {comparison.get('id')!r} must include {SELECTED_PLACEHOLDER!r}"
            )
        if comparison.get("control_configuration_id") != SELECTED_PLACEHOLDER:
            raise BenchmarkConfigError(
                f"Comparison {comparison.get('id')!r} must use SPATHI as its control"
            )
        comparison["configuration_ids"] = [
            candidate_id if item == SELECTED_PLACEHOLDER else item for item in configurations
        ]
        comparison["control_configuration_id"] = candidate_id
    return payload


def _selected_batches(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    dataset_role: str,
    bundle_path: str,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    observed: dict[tuple[str, tuple[str, ...]], str] = {}
    for comparison in comparisons:
        comparison_id = comparison.get("id")
        regime = comparison.get("regime")
        jobs = comparison.get("generation_jobs")
        roles = comparison.get("dataset_roles")
        if not isinstance(comparison_id, str) or not _ID_RE.fullmatch(comparison_id):
            raise BenchmarkConfigError("Catalog plan contains an invalid comparison ID")
        if not isinstance(regime, str):
            raise BenchmarkConfigError(f"Comparison {comparison_id!r} has no regime")
        if roles != [dataset_role]:
            raise BenchmarkConfigError(
                f"Comparison {comparison_id!r} must select only dataset role {dataset_role!r}"
            )
        if not isinstance(jobs, list) or not jobs or not all(isinstance(job, str) for job in jobs):
            raise BenchmarkConfigError(
                f"Comparison {comparison_id!r} requires explicit generation_jobs"
            )
        selector = (regime, tuple(sorted(jobs)))
        if selector in observed:
            continue
        batch_id = f"spathi-selected-{comparison_id}"
        observed[selector] = batch_id
        batches.append(
            {
                "id": batch_id,
                "kind": "spathi",
                "dataset_role": dataset_role,
                "generation_jobs": copy.deepcopy(jobs),
                "regime": regime,
                "tools_params": f"{bundle_path}/tools-params.json",
                "custom_tools": f"{bundle_path}/custom-tools.json",
                "run_metadata": f"{bundle_path}/run-metadata.json",
            }
        )
    return batches


def build_runnable_benchmark_profile(
    template: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    profile_id: str,
    bundle_path: str,
    candidate_id: str,
    selected_config_path: str = "experiments/v1/spathi/selected-config.json",
    materialization_manifest_path: str = "experiments/v1/spathi/benchmark-manifest.json",
) -> dict[str, Any]:
    """Combine a planned dataset template, catalog plan and selected SPATHI bundle."""

    template_id = template.get("id")
    if template_id not in _PROFILE_IDS or template.get("status") != "planned":
        raise BenchmarkConfigError(
            "Benchmark materialization requires reduced/full planned template"
        )
    if profile_id != _PROFILE_IDS[template_id]:
        raise BenchmarkConfigError(
            f"Template {template_id!r} must materialize as {_PROFILE_IDS[template_id]!r}"
        )
    expected_role = "development" if template_id == "reduced" else "test"
    generation_ids = {str(job["id"]) for job in template.get("generation_jobs", [])}
    baseline_batches = copy.deepcopy(plan["baseline_batches"])
    for batch in baseline_batches:
        if batch.get("kind") != "baseline":
            raise BenchmarkConfigError(f"Catalog batch {batch.get('id')!r} is not baseline")
        if batch.get("dataset_role") != expected_role:
            raise BenchmarkConfigError(
                f"Catalog batch {batch.get('id')!r} targets the wrong dataset role"
            )
        selected_jobs = batch.get("generation_jobs")
        if not isinstance(selected_jobs, list) or not selected_jobs:
            raise BenchmarkConfigError(
                f"Catalog batch {batch.get('id')!r} requires explicit generation_jobs"
            )
        unknown = sorted(set(selected_jobs) - generation_ids)
        if unknown:
            raise BenchmarkConfigError(
                f"Catalog batch {batch.get('id')!r} selects unknown generation jobs: {unknown}"
            )
    statistics = _replace_selected_placeholder(plan["statistics"], candidate_id)
    comparisons = statistics["comparisons"]
    spathi_batches = _selected_batches(
        comparisons,
        dataset_role=expected_role,
        bundle_path=bundle_path,
    )
    profile = copy.deepcopy(dict(template))
    profile.update(
        {
            "id": profile_id,
            "status": "runnable",
            "description": (
                f"{str(template.get('description', '')).strip()} "
                "Materialized after the immutable SPATHI configuration freeze."
            ).strip(),
            "inference_batches": baseline_batches + spathi_batches,
            "statistics": statistics,
            "selected_config": selected_config_path,
            "materialization_manifest": materialization_manifest_path,
        }
    )
    return profile


def _validate_baseline_inputs(repo_root: Path, profile: Mapping[str, Any]) -> None:
    """Fail before publication when a catalog bundle is incomplete or inconsistent."""

    root = repo_root.resolve()
    known_jobs = {str(job["id"]) for job in profile["generation_jobs"]}
    routes: dict[str, dict[tuple[str, str], set[str]]] = {}
    for batch in profile["inference_batches"]:
        if batch["kind"] != "baseline":
            continue
        params_path = resolve_repo_path(root, batch["tools_params"], "catalog tools params")
        params = load_json_object(params_path, "catalog tools params")
        runs = params.get("runs")
        if not isinstance(runs, list) or not runs or not all(isinstance(run, dict) for run in runs):
            raise BenchmarkConfigError(f"Catalog request {params_path} has no runs")
        run_ids = [run.get("run_id") for run in runs]
        if not all(isinstance(run_id, str) and run_id for run_id in run_ids) or len(
            set(run_ids)
        ) != len(run_ids):
            raise BenchmarkConfigError(f"Catalog request {params_path} has invalid run IDs")
        if any(str(run.get("tool_id", "")).startswith("custom_") for run in runs):
            raise BenchmarkConfigError(f"Catalog request {params_path} contains custom tools")
        metadata = load_run_metadata(root, batch["run_metadata"], "catalog run metadata")
        metadata_ids = {str(item["run_id"]) for item in metadata["runs"]}
        if metadata_ids != set(run_ids):
            raise BenchmarkConfigError(f"Catalog request {params_path} differs from run metadata")
        selected_jobs = set(batch["generation_jobs"])
        if not selected_jobs <= known_jobs:
            raise BenchmarkConfigError(f"Catalog batch {batch['id']} selects unknown datasets")
        for item in metadata["runs"]:
            if item["selection_role"] != "baseline":
                raise BenchmarkConfigError(
                    f"Catalog metadata run {item['run_id']!r} is not marked baseline"
                )
            configuration = str(item["configuration_id"])
            if configuration == SELECTED_PLACEHOLDER:
                raise BenchmarkConfigError(
                    f"Catalog configuration ID {SELECTED_PLACEHOLDER!r} is reserved"
                )
            route = (str(batch["dataset_role"]), str(batch["regime"]))
            routes.setdefault(configuration, {}).setdefault(route, set()).update(selected_jobs)

    for comparison in profile["statistics"]["comparisons"]:
        route = (str(comparison["dataset_roles"][0]), str(comparison["regime"]))
        selected_jobs = set(comparison["generation_jobs"])
        for configuration in comparison["configuration_ids"]:
            if configuration == comparison["control_configuration_id"]:
                continue
            available = routes.get(str(configuration), {}).get(route, set())
            if not selected_jobs <= available:
                missing = sorted(selected_jobs - available)
                raise BenchmarkConfigError(
                    f"Comparison {comparison['id']!r} baseline {configuration!r} lacks "
                    f"generation jobs: {missing}"
                )


def _validate_bundle(output_dir: Path, payloads: dict[str, dict[str, Any]]) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise BenchmarkConfigError(f"Existing benchmark bundle is not a directory: {output_dir}")
    observed_members = {path.name for path in output_dir.iterdir()}
    if observed_members != set(payloads):
        raise BenchmarkConfigError(
            f"Existing benchmark bundle has unexpected members: {sorted(observed_members)}"
        )
    for filename, payload in payloads.items():
        member = output_dir / filename
        if member.is_symlink() or not member.is_file():
            raise BenchmarkConfigError(
                f"Existing benchmark bundle member is not a regular file: {member}"
            )
        if load_json_object(member, filename) != payload:
            raise BenchmarkConfigError(
                f"Existing benchmark artifact differs from this workflow: {filename}"
            )


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


def materialize_benchmark_plan(
    *,
    repo_root: Path,
    template_profile_id: str,
    baseline_plan_path: Path,
    output_dir: Path,
    profile_output: Path,
    selected_config_path: Path | None = None,
) -> tuple[Path, Path]:
    """Publish or validate an idempotent selected-SPATHI benchmark profile."""

    root = repo_root.resolve()
    if template_profile_id not in _PROFILE_IDS:
        raise BenchmarkConfigError("Only reduced and full benchmark templates can be materialized")
    profile_id = _PROFILE_IDS[template_profile_id]
    profile_relative = _repo_relative_path(root, profile_output, "Benchmark profile output")
    expected_profile = f"experiments/v1/profiles/{profile_id}.json"
    if profile_relative != expected_profile and not profile_output.resolve().is_relative_to(
        (root / "work").resolve()
    ):
        raise BenchmarkConfigError(
            f"Template {template_profile_id!r} profile must be written to {expected_profile} "
            "or below work/"
        )
    bundle_relative = _repo_relative_path(root, output_dir, "Benchmark bundle output")
    resolved_selected = (
        selected_config_path.resolve()
        if selected_config_path is not None
        else root / "experiments/v1/spathi/selected-config.json"
    )
    selected_relative = _repo_relative_path(
        root, resolved_selected, "Selected SPATHI configuration"
    )
    candidate, seeds = load_frozen_selected_candidate(root, selected_config_path)
    plan, plan_relative = _load_catalog_plan(
        root,
        baseline_plan_path,
        template_profile_id=template_profile_id,
    )
    template = load_profile(root, template_profile_id)
    manifest_relative = f"{bundle_relative}/benchmark-manifest.json"
    profile = build_runnable_benchmark_profile(
        template,
        plan,
        profile_id=profile_id,
        bundle_path=bundle_relative,
        candidate_id=str(candidate["candidate_id"]),
        selected_config_path=selected_relative,
        materialization_manifest_path=manifest_relative,
    )
    tools, custom, metadata = build_selected_spathi_requests(
        candidate,
        seeds,
        docker_image=load_image_lock(root)["images"]["spathi"]["tag"],
    )
    validate_json(profile, root / "schemas/profile.schema.json", f"profile {profile_id}")
    validate_json(metadata, root / "schemas/run-metadata.schema.json", "selected run metadata")
    _validate_baseline_inputs(root, profile)

    identity = locked_spathi_identity(root)
    manifest = {
        "schema_version": "1.0",
        "profile_id": profile_id,
        "template_profile_id": template_profile_id,
        "catalog_plan": {"path": plan_relative, "sha256": sha256_file(baseline_plan_path)},
        "selected_config": {
            "path": selected_relative,
            "sha256": sha256_file(resolved_selected),
            "candidate_id": candidate["candidate_id"],
            "configuration_sha256": candidate["configuration_sha256"],
        },
        **identity,
        "algorithm_seeds": {
            "source": "experiments/v1/spathi/search-space.json#stages/estimator/algorithmic_seeds",
            "values": list(seeds),
        },
        "runs": [
            {
                "run_id": run["run_id"],
                "algorithm_seed": run["params"]["random_seed"],
                "execution_sha256": canonical_json_sha256(run["params"]),
            }
            for run in tools["runs"]
        ],
    }
    payloads = {
        "tools-params.json": tools,
        "custom-tools.json": custom,
        "run-metadata.json": metadata,
        "benchmark-manifest.json": manifest,
    }

    bundle_exists = output_dir.exists() or output_dir.is_symlink()
    profile_exists = profile_output.exists() or profile_output.is_symlink()
    if bundle_exists:
        _validate_bundle(output_dir, payloads)
    if profile_exists:
        if profile_output.is_symlink() or not profile_output.is_file():
            raise BenchmarkConfigError(
                f"Existing benchmark profile is not a regular file: {profile_output}"
            )
        if load_json_object(profile_output, f"profile {profile_id}") != profile:
            raise BenchmarkConfigError(
                f"Existing benchmark profile differs from this workflow: {profile_output}"
            )

    published_bundle = False
    published_profile = False
    try:
        if not bundle_exists:
            _publish_bundle(output_dir, payloads)
            published_bundle = True
        if not profile_exists:
            _publish_profile(profile_output, profile)
            published_profile = True
        validate_profile_references(root, profile)
    except Exception:
        if published_profile and profile_output.exists():
            profile_output.unlink()
        if published_bundle and output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    return output_dir, profile_output
