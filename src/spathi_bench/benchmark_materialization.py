"""Post-selection materialization of runnable regression and benchmark profiles.

Catalog requests are deliberately supplied as a separate, committed plan.  This
module owns only the selected SPATHI requests and the assembly/reconciliation of
both sides into one runnable profile.
"""

from __future__ import annotations

import copy
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._materialization_io import publish_new_json_directory, publish_new_json_file
from .algorithm_seed_contract import algorithm_seeds_by_configuration
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
from .efficiency_safety import analyze_safety, validate_safety_plan
from .inference_scope_contract import inference_regulator_scopes
from .noninferiority import analyze_noninferiority, resolve_efficiency_adoption
from .provenance import observe_benchmark_repository
from .seeds import derive_plan_seed, load_seed_plan

SELECTED_PLACEHOLDER = "selected_spathi"
_PROFILE_IDS = {"reduced": "reduced-v1", "full": "full-v1"}
_PLAN_KEYS = {"schema_version", "template_profile_id", "baseline_batches", "statistics"}
_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _required_mapping(payload: Mapping[str, Any], key: str, *, label: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BenchmarkConfigError(f"{label} has no valid {key!r} object")
    return value


def _verify_production_chain(
    repo_root: Path,
    production: Mapping[str, Any],
    provenance_paths: Mapping[str, Path],
) -> None:
    """Prove that one production policy is the decision authorized by its evidence."""

    selected_path = provenance_paths["calibration_selection"]
    plan_path = provenance_paths["efficiency_plan"]
    evidence_path = provenance_paths["efficiency_evidence"]
    results_path = provenance_paths["efficiency_results"]
    # Reconstruct the calibration winner instead of trusting the copied policy
    # embedded in later artifacts.  The benchmark implementation that selected
    # it must also be the implementation publishing the production decision.
    _search, selected = validate_calibration_contracts(repo_root, selected_path)
    ranking_path = repo_root / selected["selection_evidence"]["validation_ranking"]
    ranking = load_json_object(ranking_path, "production calibration ranking")
    benchmark = observe_benchmark_repository(repo_root, require_clean=True)
    if ranking.get("benchmark_commit") != benchmark["commit"]:
        raise BenchmarkConfigError(
            "Production calibration evidence belongs to another benchmark revision"
        )
    plan = load_json_object(plan_path, "production efficiency plan")
    evidence = load_json_object(evidence_path, "production efficiency evidence")
    results = load_json_object(results_path, "production efficiency results")
    validate_json(
        results,
        repo_root / "schemas/noninferiority-results.schema.json",
        "production efficiency results",
    )

    provenance = _required_mapping(production, "provenance", label="Production policy")
    plan_reference = _required_mapping(plan, "reference", label="Efficiency plan")
    selected_reference = _required_mapping(
        plan_reference, "selected_config", label="Efficiency plan reference"
    )
    selected_provenance = _required_mapping(
        provenance, "calibration_selection", label="Production policy provenance"
    )
    expected_selected_reference = {
        "path": selected_path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(selected_path),
    }
    if (
        dict(selected_reference) != expected_selected_reference
        or dict(selected_provenance) != expected_selected_reference
    ):
        raise BenchmarkConfigError(
            "Production policy and efficiency plan reference different calibration selections"
        )

    plan_provenance = _required_mapping(
        provenance, "efficiency_plan", label="Production policy provenance"
    )
    evidence_provenance = _required_mapping(
        provenance, "efficiency_evidence", label="Production policy provenance"
    )
    plan_digest = str(plan_provenance["sha256"])
    evidence_digest = str(evidence_provenance["sha256"])
    if evidence.get("plan_sha256") != plan_digest:
        raise BenchmarkConfigError("Production efficiency evidence targets a different plan")
    result_inputs = _required_mapping(results, "inputs", label="Efficiency results")
    if dict(result_inputs) != {
        "plan_sha256": plan_digest,
        "evidence_sha256": evidence_digest,
    }:
        raise BenchmarkConfigError(
            "Production efficiency results target a different plan or evidence artifact"
        )

    budget = production["budget"]
    plan_budget = _required_mapping(plan, "budget", label="Efficiency plan")
    analysis_role = plan_budget.get("analysis_role")
    if (
        plan_budget.get("id") != budget
        or evidence.get("budget") != budget
        or results.get("budget") != budget
    ):
        raise BenchmarkConfigError("Production evidence chain uses inconsistent budgets")
    if analysis_role not in {"descriptive", "confirmatory"} or any(
        value != analysis_role
        for value in (production.get("evidence_role"), results.get("analysis_role"))
    ):
        raise BenchmarkConfigError("Production evidence chain uses inconsistent analysis roles")
    if evidence.get("study_id") != results.get("study_id"):
        raise BenchmarkConfigError("Production evidence and results target different studies")
    if evidence.get("dataset_role") != results.get("dataset_role"):
        raise BenchmarkConfigError("Production evidence and results use different dataset roles")
    if evidence.get("selection_validation_manifest_sha256") != plan_reference.get(
        "selection_validation_manifest_sha256"
    ):
        raise BenchmarkConfigError(
            "Production efficiency evidence targets a different calibration validation"
        )
    panel_reference = _required_mapping(plan, "panel", label="Efficiency plan")
    if evidence.get("panel_sha256") != panel_reference.get("sha256"):
        raise BenchmarkConfigError("Production efficiency evidence targets a different panel")

    comparisons = results.get("comparisons")
    if not isinstance(comparisons, list) or not all(
        isinstance(item, Mapping) for item in comparisons
    ):
        raise BenchmarkConfigError("Efficiency results have no valid comparisons")
    try:
        expected_adoption = resolve_efficiency_adoption(
            plan,
            comparisons,
            results.get("execution_failures", []),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkConfigError(
            "Efficiency plan or results cannot reproduce the production decision"
        ) from exc
    adoption = _required_mapping(results, "adoption", label="Efficiency results")
    if dict(adoption) != expected_adoption:
        raise BenchmarkConfigError(
            "Efficiency results adoption does not match the predeclared decision rule"
        )
    expected_global_decision = {
        "candidate_adopted": "policy_adopted",
        "reference_retained": "reference_retained",
        "descriptive_only": "descriptive_only",
    }[expected_adoption["decision"]]
    if results.get("global_decision") != expected_global_decision:
        raise BenchmarkConfigError(
            "Efficiency global decision does not match its adoption decision"
        )

    expected_status = "confirmed" if analysis_role == "confirmatory" else "descriptive"
    expected_decision = expected_adoption["decision"]
    expected_policy = {
        "policy_id": expected_adoption["selected_policy_id"],
        "policy_sha256": expected_adoption["selected_policy_sha256"],
        "group_policy": expected_adoption["selected_group_policy"],
        "spathi_commit": plan_reference.get("spathi_commit"),
    }
    safety_provenance = provenance.get("efficiency_safety")
    if budget != "full" or expected_adoption["decision"] != "candidate_adopted":
        if safety_provenance is not None:
            raise BenchmarkConfigError(
                "Efficiency safety is only valid for one full provisional candidate"
            )
    else:
        if not isinstance(safety_provenance, Mapping):
            raise BenchmarkConfigError(
                "A full provisional candidate lacks the mandatory efficiency safety veto"
            )
        safety_plan_path = provenance_paths["safety_plan"]
        safety_evidence_path = provenance_paths["safety_evidence"]
        safety_results_path = provenance_paths["safety_results"]
        safety_plan = load_json_object(safety_plan_path, "production safety plan")
        validate_safety_plan(repo_root, safety_plan)
        expected_safety_panel = {
            "path": provenance_paths["safety_panel"].relative_to(repo_root).as_posix(),
            "sha256": sha256_file(provenance_paths["safety_panel"]),
        }
        if safety_plan["panel"] != expected_safety_panel:
            raise BenchmarkConfigError("Production safety provenance references a different panel")
        safety_results = load_json_object(safety_results_path, "production safety results")
        validate_json(
            safety_results,
            repo_root / "schemas/efficiency-safety-results.schema.json",
            "production safety results",
        )
        if safety_results["inputs"] != {
            "plan_sha256": sha256_file(safety_plan_path),
            "evidence_sha256": sha256_file(safety_evidence_path),
        }:
            raise BenchmarkConfigError("Production safety results target different inputs")
        verified_safety = analyze_safety(
            repo_root=repo_root,
            plan_path=safety_plan_path,
            evidence_path=safety_evidence_path,
            output_path=safety_results_path,
        )
        if verified_safety != safety_results_path:
            raise BenchmarkConfigError("Production safety results cannot be reproduced")
        decision = safety_results["decision"]
        expected_decision = (
            "candidate_adopted"
            if decision["status"] == "candidate_authorized"
            else "reference_retained"
        )
        expected_policy = {
            "policy_id": decision["selected_policy_id"],
            "policy_sha256": decision["selected_policy_sha256"],
            "group_policy": decision["selected_group_policy"],
            "spathi_commit": plan_reference.get("spathi_commit"),
        }
    if (
        production.get("status") != expected_status
        or production.get("decision") != expected_decision
        or production.get("policy") != expected_policy
    ):
        raise BenchmarkConfigError(
            "Production policy does not match the efficiency adoption decision"
        )

    verified_results, _comparisons, _manifest = analyze_noninferiority(
        repo_root=repo_root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_dir=results_path.parent,
    )
    if verified_results != results_path:
        raise BenchmarkConfigError(
            "Production provenance does not reference the verified efficiency results"
        )


def _repo_relative_path(repo_root: Path, path: Path, label: str) -> str:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} must be inside the benchmark repository") from exc


def _comparison_seeds(repo_root: Path, calibration_budget: str) -> tuple[int, ...]:
    if calibration_budget not in {"reduced", "full"}:
        raise BenchmarkConfigError(f"Unknown SPATHI calibration budget: {calibration_budget!r}")
    repetitions = 3
    plan = load_seed_plan(repo_root / "experiments/v1/seeds/seed-plan.json")
    return tuple(
        derive_plan_seed(
            plan,
            namespace=f"{calibration_budget}.test",
            stream="inference",
            coordinates={
                "purpose": "algorithm_replicate",
                "replicate_index": replicate,
            },
        )
        for replicate in range(1, repetitions + 1)
    )


def load_frozen_production_policy(
    repo_root: Path,
    production_config_path: Path,
    *,
    budget: str,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Return the efficiency-gated policy and independent comparison seeds."""

    root = repo_root.resolve()
    path = production_config_path.resolve()
    if not path.is_relative_to(root / "work") or not path.is_file() or path.is_symlink():
        raise BenchmarkConfigError("Production policy must be a regular file below work/")
    production = load_json_object(path, "SPATHI production policy")
    validate_json(
        production,
        root / "schemas/spathi-production-config.schema.json",
        "SPATHI production policy",
    )
    if production.get("budget") != budget:
        raise BenchmarkConfigError(
            f"Production policy uses budget {production.get('budget')!r}, not {budget!r}"
        )
    policy = production.get("policy")
    if not isinstance(policy, dict):
        raise BenchmarkConfigError("Production artifact contains no SPATHI policy")
    group_policy = policy.get("group_policy")
    if not isinstance(group_policy, dict):
        raise BenchmarkConfigError("Production SPATHI policy contains no group policy")
    if policy.get("policy_sha256") != canonical_json_sha256(group_policy):
        raise BenchmarkConfigError("Production SPATHI group policy has a stale digest")
    branches = group_policy.get("branches")
    if not isinstance(branches, Mapping):
        raise BenchmarkConfigError("Production SPATHI group policy contains no branches")
    multi_group = branches.get("multi_group")
    if not isinstance(multi_group, Mapping) or not isinstance(
        multi_group.get("parameters"), Mapping
    ):
        raise BenchmarkConfigError("Production SPATHI policy has no multi-group parameters")
    locked_commit = load_repo_lock(root)["repositories"]["spathi"]["commit"]
    if policy.get("spathi_commit") != locked_commit:
        raise BenchmarkConfigError(
            "Production SPATHI policy targets a different repos.lock.json commit"
        )
    provenance_paths: dict[str, Path] = {}

    def register(artifact_id: str, artifact: Mapping[str, Any]) -> None:
        artifact_path = (root / artifact["path"]).resolve()
        if (
            not artifact_path.is_relative_to(root / "work")
            or not artifact_path.is_file()
            or sha256_file(artifact_path) != artifact["sha256"]
        ):
            raise BenchmarkConfigError(
                f"Production policy provenance {artifact_id!r} is missing or changed"
            )
        provenance_paths[artifact_id] = artifact_path

    for artifact_id in (
        "calibration_selection",
        "efficiency_plan",
        "efficiency_evidence",
        "efficiency_results",
    ):
        artifact = production["provenance"].get(artifact_id)
        if not isinstance(artifact, Mapping):
            raise BenchmarkConfigError(f"Production provenance lacks {artifact_id!r}")
        register(artifact_id, artifact)
    raw_safety = production["provenance"].get("efficiency_safety")
    if raw_safety is not None:
        if not isinstance(raw_safety, Mapping) or set(raw_safety) != {
            "panel",
            "plan",
            "evidence",
            "results",
        }:
            raise BenchmarkConfigError("Efficiency safety provenance is incomplete")
        for artifact_id, artifact in raw_safety.items():
            if not isinstance(artifact, Mapping):
                raise BenchmarkConfigError("Efficiency safety reference is invalid")
            register(f"safety_{artifact_id}", artifact)
    if len(set(provenance_paths.values())) != len(provenance_paths):
        raise BenchmarkConfigError("Production policy provenance artifacts must be distinct")
    _verify_production_chain(root, production, provenance_paths)
    return copy.deepcopy(policy), _comparison_seeds(root, budget)


def build_selected_spathi_requests(
    policy: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    docker_image: str,
    seed_namespace: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Render one frozen policy over independently derived comparison seeds."""

    policy_id = policy.get("policy_id")
    group_policy = policy.get("group_policy")
    if not isinstance(policy_id, str) or not _ID_RE.fullmatch(policy_id):
        raise BenchmarkConfigError("Selected SPATHI policy has an invalid ID")
    if not isinstance(group_policy, Mapping):
        raise BenchmarkConfigError("Selected SPATHI policy has no group policy")
    if policy.get("policy_sha256") != canonical_json_sha256(dict(group_policy)):
        raise BenchmarkConfigError("Selected SPATHI policy has a stale digest")
    if not docker_image:
        raise BenchmarkConfigError("Selected SPATHI requests require a Docker image")
    normalized_seeds = tuple(seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise BenchmarkConfigError("Selected SPATHI requests require unique predeclared seeds")

    runs = [
        {
            "run_id": f"spathi_policy__seed_{seed}",
            "tool_id": f"custom_spathi_policy__seed_{seed}",
            "params": {
                "group_policy": copy.deepcopy(dict(group_policy)),
                "random_seed": seed,
            },
            "execution": {"mode": "group_native"},
        }
        for seed in normalized_seeds
    ]
    custom_tools = {
        "tools": [
            {
                "run_id": run["run_id"],
                "name": f"SPATHI policy {policy_id} seed {seed}",
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
                "configuration_id": policy_id,
                "selection_role": "selected",
                "algorithm_seed": seed,
                "algorithm_seed_parameter": "random_seed",
                "family": "spathi",
                "seed_derivation": {
                    "namespace": seed_namespace,
                    "stream": "inference",
                    "coordinates": {
                        "purpose": "algorithm_replicate",
                        "replicate_index": index,
                    },
                },
            }
            for index, (run, seed) in enumerate(zip(runs, normalized_seeds, strict=True), start=1)
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
    expected_mode = "inferential" if template_profile_id == "full" else "descriptive"
    if not isinstance(statistics, dict) or statistics.get("status") != expected_mode:
        raise BenchmarkConfigError(
            f"Catalog plan for {template_profile_id!r} requires a {expected_mode} analysis contract"
        )
    comparisons = statistics.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise BenchmarkConfigError("Catalog plan requires at least one statistical comparison")
    return plan, relative


def _replace_selected_placeholder(statistics: Mapping[str, Any], policy_id: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(statistics))
    comparisons = payload.get("comparisons")
    assert isinstance(comparisons, list)
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise BenchmarkConfigError("Catalog plan contains an invalid comparison")
        configurations = comparison.get("configuration_ids")
        if not isinstance(configurations, list):
            raise BenchmarkConfigError(
                f"Comparison {comparison.get('id')!r} has invalid configuration IDs"
            )
        if SELECTED_PLACEHOLDER not in configurations:
            if (
                comparison.get("id") == "full-pyscenic-grn-implementation-sensitivity"
                and comparison.get("control_configuration_id") == "grnboost2_default"
                and configurations == ["grnboost2_default", "pyscenic_default"]
            ):
                continue
            raise BenchmarkConfigError(
                f"Comparison {comparison.get('id')!r} must include {SELECTED_PLACEHOLDER!r}"
            )
        if comparison.get("control_configuration_id") != SELECTED_PLACEHOLDER:
            raise BenchmarkConfigError(
                f"Comparison {comparison.get('id')!r} must use SPATHI as its control"
            )
        seed_contract = algorithm_seeds_by_configuration(comparison)
        scope_contract = inference_regulator_scopes(comparison)
        comparison["configuration_ids"] = [
            policy_id if item == SELECTED_PLACEHOLDER else item for item in configurations
        ]
        comparison["algorithm_seeds_by_configuration"] = {
            (policy_id if configuration_id == SELECTED_PLACEHOLDER else configuration_id): list(
                seeds
            )
            for configuration_id, seeds in seed_contract.items()
        }
        comparison["inference_regulator_scope_by_configuration"] = {
            (policy_id if configuration_id == SELECTED_PLACEHOLDER else configuration_id): scope
            for configuration_id, scope in scope_contract.items()
        }
        comparison["control_configuration_id"] = policy_id
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
    policy_id: str,
    production_config_path: str,
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
    expected_role = "test"
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
    statistics = _replace_selected_placeholder(plan["statistics"], policy_id)
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
            "production_config": production_config_path,
            "materialization_manifest": materialization_manifest_path,
        }
    )
    return profile


def _validate_baseline_inputs(repo_root: Path, profile: Mapping[str, Any]) -> None:
    """Fail before publication when a catalog bundle is incomplete or inconsistent."""

    root = repo_root.resolve()
    known_jobs = {str(job["id"]) for job in profile["generation_jobs"]}
    routes: dict[str, dict[tuple[str, str], set[str]]] = {}
    seeds_by_route: dict[str, dict[tuple[str, str], set[int | None]]] = {}
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
            seed = item.get("algorithm_seed")
            if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
                raise BenchmarkConfigError(
                    f"Catalog metadata run {item['run_id']!r} has an invalid algorithm seed"
                )
            seeds_by_route.setdefault(configuration, {}).setdefault(route, set()).add(seed)

    for comparison in profile["statistics"]["comparisons"]:
        expected_seeds_by_configuration = algorithm_seeds_by_configuration(comparison)
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
            observed_seeds = seeds_by_route.get(str(configuration), {}).get(route, set())
            expected_seeds = set(expected_seeds_by_configuration[str(configuration)])
            if observed_seeds != expected_seeds:
                raise BenchmarkConfigError(
                    f"Comparison {comparison['id']!r} baseline {configuration!r} has algorithm "
                    "seeds "
                    f"{sorted(observed_seeds, key=lambda seed: -1 if seed is None else seed)}, "
                    "expected "
                    f"{sorted(expected_seeds, key=lambda seed: -1 if seed is None else seed)}"
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


def materialize_benchmark_plan(
    *,
    repo_root: Path,
    template_profile_id: str,
    baseline_plan_path: Path,
    output_dir: Path,
    profile_output: Path,
    production_config_path: Path,
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
    resolved_production = production_config_path.resolve()
    production_relative = _repo_relative_path(root, resolved_production, "SPATHI production policy")
    policy, seeds = load_frozen_production_policy(
        root,
        resolved_production,
        budget=template_profile_id,
    )
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
        policy_id=str(policy["policy_id"]),
        production_config_path=production_relative,
        materialization_manifest_path=manifest_relative,
    )
    for comparison in profile["statistics"]["comparisons"]:
        seeds_by_configuration = algorithm_seeds_by_configuration(comparison)
        if str(policy["policy_id"]) not in seeds_by_configuration:
            continue
        selected_seeds = seeds_by_configuration[str(policy["policy_id"])]
        if list(selected_seeds) != list(seeds):
            raise BenchmarkConfigError(
                f"Comparison {comparison['id']!r} selected SPATHI algorithm seeds differ "
                f"from the frozen {template_profile_id} production seed panel"
            )
    tools, custom, metadata = build_selected_spathi_requests(
        policy,
        seeds,
        docker_image=load_image_lock(root)["images"]["spathi"]["tag"],
        seed_namespace=f"{template_profile_id}.test",
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
        "production_config": {
            "path": production_relative,
            "sha256": sha256_file(resolved_production),
            "policy_id": policy["policy_id"],
            "policy_sha256": policy["policy_sha256"],
            "budget": template_profile_id,
        },
        **identity,
        "algorithm_seeds": {
            "source": "experiments/v1/seeds/seed-plan.json",
            "namespace": f"{template_profile_id}.test",
            "stream": "inference",
            "coordinates": {"purpose": "algorithm_replicate"},
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
            publish_new_json_directory(output_dir, payloads)
            published_bundle = True
        if not profile_exists:
            publish_new_json_file(profile_output, profile)
            published_profile = True
        validate_profile_references(root, profile)
    except Exception:
        if published_profile and profile_output.exists():
            profile_output.unlink()
        if published_bundle and output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    return output_dir, profile_output
