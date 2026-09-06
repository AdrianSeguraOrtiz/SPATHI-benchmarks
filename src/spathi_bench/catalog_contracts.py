"""Executable contracts linking the ANDREA catalog matrix to benchmark plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .algorithm_seed_contract import algorithm_seeds_by_configuration
from .artifacts import canonical_json_sha256
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    load_run_metadata,
    resolve_repo_path,
)
from .inference_scope_contract import inference_regulator_scopes

CATALOG_TOOL_IDS = frozenset(
    {
        "aracne3",
        "cespgrn",
        "clr",
        "dignet",
        "genie3",
        "grnboost2",
        "infercsn",
        "inferelator3",
        "kscreni",
        "lioness",
        "metasem",
        "miniex3",
        "pidc",
        "planet",
        "ppcor",
        "pyscenic",
        "scgenerai",
        "scing",
        "scminer",
        "scmtni",
        "scregulate",
        "scsgl",
        "simic",
        "tigress",
    }
)

_PERMITTED_EXTERNAL_OVERRIDES = frozenset(
    {
        "caller_seed_for_reproducibility",
        "execution_mode_for_common_group_contract",
        "common_tf_candidate_input",
    }
)


@dataclass(frozen=True)
class CatalogRoute:
    """One non-assisted catalog configuration promised by the design matrix."""

    configuration_id: str
    tool_id: str
    regime: str
    execution_mode: str
    evaluation_levels: tuple[str, ...]
    reproducibility_class: str
    seed_parameter: str | None


def _candidate_routes(matrix: Mapping[str, Any]) -> dict[str, CatalogRoute]:
    policy = matrix.get("external_parameter_policy")
    if not isinstance(policy, Mapping) or (
        policy.get("selection") != "upstream_library_defaults"
        or policy.get("tuning") != "none_on_benchmark_outcomes"
        or set(policy.get("permitted_overrides", ())) != _PERMITTED_EXTERNAL_OVERRIDES
    ):
        raise BenchmarkConfigError(
            "Catalog matrix must freeze upstream defaults, forbid outcome-based competitor "
            "tuning and enumerate exactly the permitted non-tuning overrides"
        )
    routes: dict[str, CatalogRoute] = {}
    tool_ids = [str(entry["tool_id"]) for entry in matrix["entries"]]
    if len(set(tool_ids)) != len(tool_ids) or set(tool_ids) != CATALOG_TOOL_IDS:
        raise BenchmarkConfigError(
            "Catalog matrix must classify each v1 ANDREA tool exactly once: "
            f"missing={sorted(CATALOG_TOOL_IDS - set(tool_ids))}, "
            f"unknown={sorted(set(tool_ids) - CATALOG_TOOL_IDS)}"
        )
    for entry in matrix["entries"]:
        tool_id = str(entry["tool_id"])
        disposition = str(entry["disposition"])
        reproducibility = entry["reproducibility"]
        reproducibility_class = str(reproducibility["class"])
        seed_parameter = reproducibility["seed_parameter"]
        if disposition == "candidate" and reproducibility_class == "uncontrolled_stochastic":
            raise BenchmarkConfigError(
                f"Catalog tool {tool_id!r} cannot be a candidate with uncontrolled randomness"
            )
        entry_routes = entry["routes"]
        if disposition == "blocked" and any(
            route["status"] == "candidate" for route in entry_routes
        ):
            raise BenchmarkConfigError(
                f"Blocked catalog tool {tool_id!r} contains an executable candidate route"
            )
        if disposition == "assisted_only" and any(
            route["role"] != "assisted" for route in entry_routes
        ):
            raise BenchmarkConfigError(
                f"Assisted-only catalog tool {tool_id!r} contains a non-assisted route"
            )
        for route in entry_routes:
            is_executable = route["status"] == "candidate" and route["role"] != "assisted"
            configuration_id = route.get("configuration_id")
            if not is_executable:
                if configuration_id is not None:
                    raise BenchmarkConfigError(
                        f"Non-executable catalog route {tool_id!r}/{route['variant']!r} "
                        "cannot claim a runnable configuration_id"
                    )
                continue
            if not isinstance(configuration_id, str) or not configuration_id:
                raise BenchmarkConfigError(
                    f"Executable catalog route {tool_id!r}/{route['variant']!r} has no "
                    "configuration_id"
                )
            if configuration_id in routes:
                raise BenchmarkConfigError(
                    f"Catalog configuration_id {configuration_id!r} identifies multiple routes"
                )
            raw_levels = route.get("evaluation_levels", [])
            if (
                not isinstance(raw_levels, list)
                or len(set(raw_levels)) != len(raw_levels)
                or any(level not in {"directed", "topology"} for level in raw_levels)
            ):
                raise BenchmarkConfigError(
                    f"Catalog configuration {configuration_id!r} has invalid evaluation levels"
                )
            if route["role"] == "secondary" and route["regime"] == "group_only" and not raw_levels:
                raise BenchmarkConfigError(
                    f"All-gene catalog configuration {configuration_id!r} has no declared "
                    "evaluation level"
                )
            routes[configuration_id] = CatalogRoute(
                configuration_id=configuration_id,
                tool_id=tool_id,
                regime=str(route["regime"]),
                execution_mode=str(route["execution_mode"]),
                evaluation_levels=tuple(str(level) for level in raw_levels),
                reproducibility_class=reproducibility_class,
                seed_parameter=None if seed_parameter is None else str(seed_parameter),
            )
    return routes


def _validate_plan(
    repo_root: Path,
    *,
    plan_id: str,
    routes: Mapping[str, CatalogRoute],
    andrea_repo: Path,
) -> tuple[set[str], set[str]]:
    plan_path = repo_root / "experiments/v1/inference/plans" / f"{plan_id}-v1.json"
    plan = load_json_object(plan_path, f"{plan_id} catalog plan")
    batches = plan.get("baseline_batches")
    if not isinstance(batches, list) or not batches:
        raise BenchmarkConfigError(f"{plan_id} catalog plan has no baseline batches")

    observed: set[str] = set()
    exact_routes: set[str] = set()
    locked_images = load_image_lock(repo_root)["catalog_images"]
    tools_root = andrea_repo / "andrea/catalog_inference_tools/tools"
    if not tools_root.is_dir():
        raise BenchmarkConfigError(f"Locked ANDREA ToolSpecs are unavailable: {tools_root}")
    defaults_by_tool: dict[str, dict[str, Any]] = {}
    for batch in batches:
        params_path = resolve_repo_path(
            repo_root,
            str(batch["tools_params"]),
            f"{plan_id}:{batch['id']} tools params",
        )
        params = load_json_object(params_path, f"{plan_id}:{batch['id']} tools params")
        metadata = load_run_metadata(
            repo_root,
            str(batch["run_metadata"]),
            f"{plan_id}:{batch['id']} run metadata",
        )
        request_by_id = {str(run["run_id"]): run for run in params.get("runs", [])}
        metadata_by_id = {str(run["run_id"]): run for run in metadata["runs"]}
        if not request_by_id or request_by_id.keys() != metadata_by_id.keys():
            raise BenchmarkConfigError(
                f"{plan_id}:{batch['id']} request and metadata run identities differ"
            )
        runs_by_configuration: dict[str, list[str]] = {}
        for run_id, run in request_by_id.items():
            configuration_id = str(metadata_by_id[run_id]["configuration_id"])
            route = routes.get(configuration_id)
            if route is None:
                raise BenchmarkConfigError(
                    f"{plan_id}:{batch['id']} executes undeclared catalog configuration "
                    f"{configuration_id!r}"
                )
            tool_id = str(run.get("tool_id", ""))
            execution = run.get("execution")
            execution_mode = execution.get("mode") if isinstance(execution, dict) else None
            if tool_id != route.tool_id or execution_mode != route.execution_mode:
                raise BenchmarkConfigError(
                    f"Catalog configuration {configuration_id!r} is requested through "
                    f"{tool_id!r}/{execution_mode!r}, expected "
                    f"{route.tool_id!r}/{route.execution_mode!r}"
                )
            if f"tool:{tool_id}" not in locked_images:
                raise BenchmarkConfigError(
                    f"Catalog configuration {configuration_id!r} has no locked Docker image"
                )
            defaults = defaults_by_tool.get(tool_id)
            if defaults is None:
                toolspec = load_json_object(
                    tools_root / tool_id / "toolspec.json",
                    f"ANDREA {tool_id} ToolSpec",
                )
                raw_definitions = toolspec.get("params")
                if not isinstance(raw_definitions, Mapping):
                    raise BenchmarkConfigError(f"ANDREA {tool_id!r} ToolSpec has no params")
                defaults = {
                    str(key): definition.get("default")
                    for key, definition in raw_definitions.items()
                    if isinstance(definition, Mapping) and "default" in definition
                }
                defaults_by_tool[tool_id] = defaults
            request_params = run.get("params")
            if not isinstance(request_params, Mapping):
                raise BenchmarkConfigError(
                    f"Catalog configuration {configuration_id!r} has invalid params"
                )
            for parameter, value in request_params.items():
                if parameter == route.seed_parameter:
                    continue
                if parameter not in defaults or canonical_json_sha256(
                    {"value": value}
                ) != canonical_json_sha256({"value": defaults[parameter]}):
                    raise BenchmarkConfigError(
                        f"Catalog configuration {configuration_id!r} changes {parameter!r} "
                        "away from the locked ANDREA ToolSpec default"
                    )
            run_metadata = metadata_by_id[run_id]
            observed_seed = run_metadata["algorithm_seed"]
            observed_parameter = run_metadata["algorithm_seed_parameter"]
            if route.reproducibility_class == "caller_seeded":
                if (
                    not isinstance(observed_seed, int)
                    or isinstance(observed_seed, bool)
                    or observed_parameter != route.seed_parameter
                    or request_params.get(str(route.seed_parameter)) != observed_seed
                ):
                    raise BenchmarkConfigError(
                        f"Caller-seeded catalog configuration {configuration_id!r} does not "
                        f"declare an integer {route.seed_parameter!r} algorithm seed"
                    )
            elif observed_seed is not None or observed_parameter is not None:
                raise BenchmarkConfigError(
                    f"Catalog configuration {configuration_id!r} must record a null caller "
                    f"seed for reproducibility class {route.reproducibility_class!r}"
                )
            runs_by_configuration.setdefault(configuration_id, []).append(run_id)
            observed.add(configuration_id)
            if batch["regime"] == route.regime:
                exact_routes.add(configuration_id)
        for configuration_id, run_ids in runs_by_configuration.items():
            route = routes[configuration_id]
            if route.reproducibility_class != "caller_seeded" and len(run_ids) != 1:
                raise BenchmarkConfigError(
                    f"Catalog configuration {configuration_id!r} must execute exactly once per "
                    f"dataset, found {len(run_ids)} request runs"
                )
    return observed, exact_routes


def _validate_full_all_gene_statistics(
    repo_root: Path,
    routes: Mapping[str, CatalogRoute],
) -> None:
    plan = load_json_object(
        repo_root / "experiments/v1/inference/plans/full-v1.json",
        "full catalog plan",
    )
    comparisons = {
        str(comparison["id"]): comparison for comparison in plan["statistics"]["comparisons"]
    }
    for level in ("directed", "topology"):
        comparison_id = f"full-all-gene-inference-{level}-accuracy"
        comparison = comparisons.get(comparison_id)
        if comparison is None:
            raise BenchmarkConfigError(
                f"Full catalog plan lacks the {level} all-gene statistical family"
            )
        scopes = inference_regulator_scopes(comparison)
        if comparison.get("candidate_space") != "tf_restricted":
            raise BenchmarkConfigError(
                f"All-gene {level} family must use TF-restricted ANDREA evaluation"
            )
        expected = {
            route.configuration_id for route in routes.values() if level in route.evaluation_levels
        } | {"selected_spathi"}
        observed = set(comparison["configuration_ids"])
        if observed != expected:
            raise BenchmarkConfigError(
                f"All-gene {level} statistical family differs from the catalog matrix: "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
        if scopes["selected_spathi"] != "tf_restricted" or any(
            scopes[configuration_id] != "all_gene"
            for configuration_id in observed - {"selected_spathi"}
        ):
            raise BenchmarkConfigError(
                f"All-gene {level} family must expose the TF list to SPATHI and the full "
                "gene regulator space to catalog competitors"
            )
        algorithm_seeds_by_configuration(comparison)


def validate_catalog_plan_coverage(
    repo_root: Path,
    matrix: Mapping[str, Any],
    *,
    andrea_repo: Path | None = None,
) -> None:
    """Require plans to implement the runnable matrix without executing blocked routes.

    The reduced plan may select a budgeted subset. The full plan is the publication
    contract and must execute every non-assisted candidate configuration through its
    declared tool, execution mode and regime. Any plan-only configuration is rejected,
    which keeps blocked or invented variants out of both budgets.
    """

    routes = _candidate_routes(matrix)
    locked_andrea = (andrea_repo or repo_root.resolve().parent / "ANDREA").resolve()
    if not routes:
        raise BenchmarkConfigError("Catalog matrix declares no executable comparison routes")
    reduced_observed, _reduced_exact = _validate_plan(
        repo_root,
        plan_id="reduced",
        routes=routes,
        andrea_repo=locked_andrea,
    )
    full_observed, full_exact = _validate_plan(
        repo_root,
        plan_id="full",
        routes=routes,
        andrea_repo=locked_andrea,
    )
    required = set(routes)
    if not reduced_observed <= required:
        raise BenchmarkConfigError("Reduced catalog plan contains an undeclared configuration")
    if full_observed != required:
        raise BenchmarkConfigError(
            "Full catalog plan and executable matrix configurations differ: "
            f"missing={sorted(required - full_observed)}, "
            f"extra={sorted(full_observed - required)}"
        )
    if full_exact != required:
        raise BenchmarkConfigError(
            "Full catalog plan does not exercise every configuration in its declared regime: "
            f"missing={sorted(required - full_exact)}"
        )
    _validate_full_all_gene_statistics(repo_root, routes)
