"""Materialize and select the finite SPATHI efficiency-development grid."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from statistics import fmean, median
from typing import Any

from ._materialization_io import publish_new_json_directory
from .artifacts import canonical_json_sha256, sha256_file
from .calibration import locked_spathi_identity
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    load_profile,
    validate_json,
)
from .noninferiority import (
    ADAPTIVE_TREE_PARAMETERS,
    TARGET_ELIGIBILITY_PARAMETERS,
    _policy_with_overrides,
    _selected_policy,
)
from .safe_io import write_idempotent_json
from .seeds import derive_plan_seed, load_seed_plan
from .state import state_path

CONTRACT_PATH = Path("experiments/v1/spathi/efficiency-development.json")


def _inside(root: Path, path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} must be inside the benchmark repository") from exc


def load_efficiency_development_contract(repo_root: Path) -> dict[str, Any]:
    """Validate the finite grid, profiles and disjoint deterministic seed domains."""

    root = repo_root.resolve()
    contract = load_json_object(root / CONTRACT_PATH, "efficiency-development contract")
    validate_json(
        contract,
        root / "schemas/efficiency-development.schema.json",
        "efficiency-development contract",
    )
    candidates = contract["candidates"]
    ids = [item["id"] for item in candidates]
    if len(ids) != len(set(ids)):
        raise BenchmarkConfigError("Efficiency-development candidate IDs are not unique")
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_parameters = {
        "target_eligibility": TARGET_ELIGIBILITY_PARAMETERS,
        "adaptive_trees": ADAPTIVE_TREE_PARAMETERS,
        "combined": TARGET_ELIGIBILITY_PARAMETERS | ADAPTIVE_TREE_PARAMETERS,
    }
    for candidate in candidates:
        component = str(candidate["component"])
        by_component[component].append(candidate)
        if set(candidate["execution_overrides"]) != expected_parameters[component]:
            raise BenchmarkConfigError(
                f"Efficiency-development candidate {candidate['id']!r} changes the wrong parameters"
            )
    if set(by_component) != set(expected_parameters) or any(
        len(items) != 3 for items in by_component.values()
    ):
        raise BenchmarkConfigError("Efficiency development requires three candidates per component")

    seed_plan = load_seed_plan(root / "experiments/v1/seeds/seed-plan.json")
    all_generation_seeds: set[int] = set()
    for budget_id, budget in contract["budgets"].items():
        namespace = f"{budget_id}.efficiency_development"
        expected_seeds = [
            derive_plan_seed(
                seed_plan,
                namespace=namespace,
                stream="inference",
                coordinates={"purpose": "algorithm_replicate", "replicate_index": index},
            )
            for index in range(1, len(budget["algorithm_seeds"]) + 1)
        ]
        if budget["algorithm_seeds"] != expected_seeds:
            raise BenchmarkConfigError(
                f"{budget_id.title()} efficiency-development algorithm seeds are stale"
            )
        expected_order_seed = derive_plan_seed(
            seed_plan,
            namespace=namespace,
            stream="execution_order",
            coordinates={"purpose": "counterbalanced_execution_order"},
        )
        if budget["execution_order_seed"] != expected_order_seed:
            raise BenchmarkConfigError(
                f"{budget_id.title()} efficiency-development execution-order seed is stale"
            )
        profile = load_profile(root, budget["profile"])
        if (
            profile["purpose"] != "efficiency_development"
            or profile["status"] != "planned"
            or profile["inference_batches"]
        ):
            raise BenchmarkConfigError("Efficiency-development template profile is not canonical")
        jobs = profile["generation_jobs"]
        if {job["dataset_role"] for job in jobs} != {budget["dataset_role"]} or {
            job["split_id"] for job in jobs
        } != {budget["split_id"]}:
            raise BenchmarkConfigError("Efficiency-development profile changes its role or split")
        dataset_count = 0
        for job in jobs:
            runs = load_json_object(root / job["simulator_runs"], f"runs for {job['id']}")["runs"]
            if len(runs) != 1:
                raise BenchmarkConfigError(
                    f"Efficiency-development job {job['id']!r} is not atomic"
                )
            run = runs[0]
            dataset_count += int(run["replicates"])
            seeds = set(
                range(int(run["base_seed"]), int(run["base_seed"]) + int(run["replicates"]))
            )
            if all_generation_seeds & seeds:
                raise BenchmarkConfigError("Efficiency-development generation seeds overlap")
            all_generation_seeds.update(seeds)
        if dataset_count != budget["expected_independent_datasets"]:
            raise BenchmarkConfigError("Efficiency-development profile has the wrong dataset count")
    return contract


def materialize_efficiency_development(
    *,
    repo_root: Path,
    selected_config_path: Path,
    budget: str,
    output_dir: Path,
    profile_output: Path,
) -> tuple[Path, Path]:
    """Bind every development candidate and the exact reference to one frozen selection."""

    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError("Efficiency-development budget must be reduced or full")
    root = repo_root.resolve()
    contract = load_efficiency_development_contract(root)
    selected_path = selected_config_path.resolve()
    selected = _selected_policy(root, selected_path, budget)
    reference = selected["selected_policy"]
    if reference["spathi_commit"] != locked_spathi_identity(root)["spathi_commit"]:
        raise BenchmarkConfigError("Efficiency development requires the current SPATHI selection")
    budget_contract = contract["budgets"][budget]
    image = load_image_lock(root)["images"]["spathi"]["tag"]
    seeds = budget_contract["algorithm_seeds"]
    definitions = [
        {
            "id": "spathi_exact_reference",
            "component": "reference",
            "group_policy": reference["group_policy"],
            "selection_role": "base",
        },
        *[
            {
                **candidate,
                "group_policy": _policy_with_overrides(
                    reference["group_policy"], candidate["execution_overrides"]
                ),
                "selection_role": "candidate",
            }
            for candidate in contract["candidates"]
        ],
    ]
    runs: list[dict[str, Any]] = []
    custom: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    scheduled = [
        (definition, index, seed)
        for definition in definitions
        for index, seed in enumerate(seeds, start=1)
    ]
    random.Random(budget_contract["execution_order_seed"]).shuffle(scheduled)
    for definition, index, seed in scheduled:
        run_id = f"effdev_{definition['id']}_seed_{seed}"
        runs.append(
            {
                "run_id": run_id,
                "tool_id": f"custom_{run_id}",
                "params": {"group_policy": definition["group_policy"], "random_seed": seed},
                "execution": {"mode": "group_native"},
            }
        )
        custom.append(
            {
                "run_id": run_id,
                "name": f"SPATHI efficiency development {definition['id']} seed {seed}",
                "docker_image": image,
                "execution_mode": "group_native",
                "extra_inputs": ["groups", "tf_list"],
                "outputs": {"directed": True, "sign": "none"},
            }
        )
        metadata.append(
            {
                "run_id": run_id,
                "configuration_id": definition["id"],
                "selection_role": definition["selection_role"],
                "algorithm_seed": seed,
                "algorithm_seed_parameter": "random_seed",
                "family": "spathi_efficiency_development",
                "seed_derivation": {
                    "namespace": f"{budget}.efficiency_development",
                    "stream": "inference",
                    "coordinates": {"purpose": "algorithm_replicate", "replicate_index": index},
                },
            }
        )
    payloads = {
        "tools-params.json": {"runs": runs},
        "custom-tools.json": {"tools": custom},
        "run-metadata.json": {"schema_version": "1.0", "runs": metadata},
    }
    if output_dir.exists():
        observed = {path.name for path in output_dir.iterdir()}
        if observed != set(payloads) or any(
            load_json_object(output_dir / name, name) != payload
            for name, payload in payloads.items()
        ):
            raise BenchmarkConfigError("Existing efficiency-development bundle is stale")
    else:
        publish_new_json_directory(output_dir, payloads)
    template = load_profile(root, budget_contract["profile"])
    profile = deepcopy(template)
    profile["status"] = "runnable"
    profile["inference_batches"] = [
        {
            "id": "spathi_efficiency_development",
            "kind": "spathi",
            "dataset_role": "efficiency_development",
            "generation_jobs": [job["id"] for job in profile["generation_jobs"]],
            "regime": "group_only",
            "tools_params": _inside(root, output_dir / "tools-params.json", "tools parameters"),
            "custom_tools": _inside(root, output_dir / "custom-tools.json", "custom tools"),
            "run_metadata": _inside(root, output_dir / "run-metadata.json", "run metadata"),
        }
    ]
    validate_json(profile, root / "schemas/profile.schema.json", "efficiency-development profile")
    if profile_output.exists():
        if load_json_object(profile_output, "efficiency-development profile") != profile:
            raise BenchmarkConfigError("Existing efficiency-development profile is stale")
    else:
        write_idempotent_json(
            profile_output, profile, "efficiency-development profile", boundary=root
        )
    return output_dir.resolve(), profile_output.resolve()


def _hierarchical_mean(values: Sequence[tuple[str, str, str, float]]) -> float:
    by_condition: dict[tuple[str, str], list[float]] = defaultdict(list)
    for simulator, condition, _dataset, value in values:
        by_condition[(simulator, condition)].append(value)
    by_simulator: dict[str, list[float]] = defaultdict(list)
    for (simulator, _condition), condition_values in by_condition.items():
        by_simulator[simulator].append(fmean(condition_values))
    return fmean(fmean(condition_values) for condition_values in by_simulator.values())


def _validated_metric_rows(
    *,
    repo_root: Path,
    metrics_file: Path,
    budget_contract: Mapping[str, Any],
    configuration_ids: set[str],
) -> tuple[list[dict[str, str]], Path]:
    """Bind every selected CSV row to the completed content-addressed pipeline state."""

    run_root = metrics_file.parent.parent
    execution_state_path = state_path(run_root)
    state = load_json_object(execution_state_path, "efficiency-development execution state")
    reports = state.get("reports")
    if (
        state.get("status") not in {"summarized", "completed"}
        or not isinstance(reports, Mapping)
        or Path(str(reports.get("metrics_csv", ""))).resolve() != metrics_file
    ):
        raise BenchmarkConfigError(
            "Efficiency-development metrics are not bound to a completed pipeline state"
        )
    split_id = str(budget_contract["split_id"])
    dataset_rows = [
        item
        for item in state.get("datasets", [])
        if item.get("dataset_role") == "efficiency_development" and item.get("split_id") == split_id
    ]
    dataset_ids = [str(item.get("dataset_id", "")) for item in dataset_rows]
    if not all(dataset_ids) or len(dataset_ids) != len(set(dataset_ids)):
        raise BenchmarkConfigError(
            "Efficiency-development execution state has duplicate or invalid dataset IDs"
        )
    datasets = {str(item["dataset_id"]): item for item in dataset_rows}
    if len(datasets) != int(budget_contract["expected_independent_datasets"]):
        raise BenchmarkConfigError(
            "Efficiency-development execution state has the wrong dataset panel"
        )
    expected_seeds = {int(seed) for seed in budget_contract["algorithm_seeds"]}
    expected_executions = {
        (configuration, dataset, seed)
        for configuration in configuration_ids
        for dataset in datasets
        for seed in expected_seeds
    }
    evaluations = [
        item
        for item in state.get("evaluations", [])
        if item.get("configuration_id") in configuration_ids and item.get("dataset_id") in datasets
    ]
    observed_executions = [
        (
            str(item.get("configuration_id")),
            str(item.get("dataset_id")),
            item.get("algorithm_seed"),
        )
        for item in evaluations
    ]
    if (
        len(observed_executions) != len(set(observed_executions))
        or set(observed_executions) != expected_executions
    ):
        raise BenchmarkConfigError(
            "Efficiency-development evaluations do not form the exact "
            "dataset-by-configuration-by-seed Cartesian product"
        )
    expected_metric_keys: set[tuple[str, str, str, str]] = set()
    contexts_by_execution: dict[tuple[str, str, int], set[str]] = {}
    for evaluation in evaluations:
        report_path = Path(str(evaluation.get("evaluation_report", ""))).resolve()
        if not report_path.is_file() or not report_path.is_relative_to(repo_root / "work"):
            raise BenchmarkConfigError(
                "Efficiency-development evaluation report is missing or outside work/"
            )
        report = load_json_object(report_path, "efficiency-development evaluation report")
        contexts = [
            metric
            for metric in report.get("metrics", [])
            if metric.get("level") == "directed"
            and str(metric.get("context", "")).startswith("group:")
        ]
        names = [str(metric["context"]) for metric in contexts]
        if (
            not names
            or len(names) != len(set(names))
            or any(metric.get("status") != "ok" for metric in contexts)
        ):
            raise BenchmarkConfigError(
                "Efficiency-development evaluation lacks unique, complete group contexts"
            )
        contexts_by_execution[
            (
                str(evaluation["configuration_id"]),
                str(evaluation["dataset_id"]),
                int(evaluation["algorithm_seed"]),
            )
        ] = set(names)
        expected_metric_keys.update(
            (
                str(evaluation["configuration_id"]),
                str(evaluation["dataset_id"]),
                str(evaluation["algorithm_seed"]),
                context,
            )
            for context in names
        )
    for dataset_id in datasets:
        for seed in expected_seeds:
            reference_contexts = contexts_by_execution[("spathi_exact_reference", dataset_id, seed)]
            if any(
                contexts_by_execution[(configuration, dataset_id, seed)] != reference_contexts
                for configuration in configuration_ids - {"spathi_exact_reference"}
            ):
                raise BenchmarkConfigError(
                    "Efficiency-development candidate and reference group contexts differ"
                )
    with metrics_file.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in all_rows
        if row.get("dataset_role") == "efficiency_development"
        and row.get("split_id") == split_id
        and row.get("level") == "directed"
        and str(row.get("context", "")).startswith("group:")
    ]
    observed_metric_keys = [
        (
            str(row.get("configuration_id")),
            str(row.get("dataset_id")),
            str(row.get("algorithm_seed")),
            str(row.get("context")),
        )
        for row in rows
    ]
    if (
        len(observed_metric_keys) != len(set(observed_metric_keys))
        or set(observed_metric_keys) != expected_metric_keys
        or any(row.get("status") != "ok" for row in rows)
    ):
        raise BenchmarkConfigError(
            "Efficiency-development metrics are incomplete, duplicated, failed or contain "
            "unexpected group evaluations"
        )
    for row in rows:
        dataset = datasets[str(row["dataset_id"])]
        if row.get("simulator_id") != dataset.get("simulator_id") or row.get(
            "condition_id"
        ) != dataset.get("condition_id"):
            raise BenchmarkConfigError(
                "Efficiency-development metric identity differs from its generated dataset"
            )
    return rows, execution_state_path


def select_efficiency_development(
    *,
    repo_root: Path,
    selected_config_path: Path,
    budget: str,
    metrics_path: Path,
    output_path: Path,
) -> Path:
    """Select one reproducible development finalist per component from metrics.csv."""

    root = repo_root.resolve()
    contract = load_efficiency_development_contract(root)
    selected_path = selected_config_path.resolve()
    _selected_policy(root, selected_path, budget)
    metrics_file = metrics_path.resolve()
    _inside(root, metrics_file, "Efficiency-development metrics")
    budget_contract = contract["budgets"][budget]
    expected_seeds = {str(seed) for seed in budget_contract["algorithm_seeds"]}
    configuration_ids = {"spathi_exact_reference", *[item["id"] for item in contract["candidates"]]}
    rows, execution_state_path = _validated_metric_rows(
        repo_root=root,
        metrics_file=metrics_file,
        budget_contract=budget_contract,
        configuration_ids=configuration_ids,
    )
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["configuration_id"], row["dataset_id"], row["algorithm_seed"])].append(row)
    reference_keys = {
        (dataset, seed)
        for configuration, dataset, seed in grouped
        if configuration == "spathi_exact_reference"
    }
    if (
        len({dataset for dataset, _seed in reference_keys})
        != budget_contract["expected_independent_datasets"]
        or {seed for _dataset, seed in reference_keys} != expected_seeds
    ):
        raise BenchmarkConfigError("Efficiency-development reference evidence is incomplete")

    summaries: dict[str, dict[tuple[str, str], tuple[str, str, float, float]]] = defaultdict(dict)
    for (configuration, dataset, seed), context_rows in grouped.items():
        try:
            aupr = fmean(float(row["aupr"]) for row in context_rows)
            wall_times = [float(row["wall_time_seconds"]) for row in context_rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkConfigError("Efficiency-development metrics are not numeric") from exc
        if not math.isfinite(aupr) or not all(
            value > 0 and math.isfinite(value) for value in wall_times
        ):
            raise BenchmarkConfigError("Efficiency-development metrics are non-finite")
        simulator = context_rows[0]["simulator_id"]
        condition = context_rows[0]["condition_id"]
        if any(
            row["simulator_id"] != simulator or row["condition_id"] != condition
            for row in context_rows
        ):
            raise BenchmarkConfigError("One development execution mixes dataset strata")
        summaries[configuration][(dataset, seed)] = (simulator, condition, aupr, median(wall_times))

    rule = contract["selection_rule"]
    results: list[dict[str, Any]] = []
    for candidate in contract["candidates"]:
        candidate_rows = summaries[candidate["id"]]
        complete = set(candidate_rows) == reference_keys
        if not complete:
            raise BenchmarkConfigError(
                f"Efficiency-development candidate {candidate['id']!r} has incomplete paired evidence"
            )
        accuracy_values: list[tuple[str, str, str, float]] = []
        runtime_values: list[tuple[str, str, str, float]] = []
        for key in sorted(reference_keys):
            simulator, condition, reference_aupr, reference_time = summaries[
                "spathi_exact_reference"
            ][key]
            candidate_simulator, candidate_condition, candidate_aupr, candidate_time = (
                candidate_rows[key]
            )
            if (candidate_simulator, candidate_condition) != (simulator, condition):
                raise BenchmarkConfigError("Candidate/reference development strata differ")
            accuracy_values.append((simulator, condition, key[0], reference_aupr - candidate_aupr))
            runtime_values.append(
                (simulator, condition, key[0], math.log(candidate_time / reference_time))
            )
        mean_loss = _hierarchical_mean(accuracy_values)
        runtime_ratio = math.exp(_hierarchical_mean(runtime_values))
        results.append(
            {
                "candidate_id": candidate["id"],
                "component": candidate["component"],
                "mean_aupr_loss": mean_loss,
                "wall_time_ratio": runtime_ratio,
                "complete": True,
                "accuracy_eligible": mean_loss <= rule["maximum_mean_absolute_aupr_loss"],
                "dataset_count": budget_contract["expected_independent_datasets"],
                "algorithm_seed_count": len(expected_seeds),
            }
        )
    result_by_id = {item["candidate_id"]: item for item in results}
    selected_candidates: list[dict[str, Any]] = []
    retained_reference_components: list[str] = []
    for component in ("target_eligibility", "adaptive_trees", "combined"):
        eligible = [
            item
            for item in results
            if item["component"] == component
            and item["accuracy_eligible"]
            and item["wall_time_ratio"] < 1.0
        ]
        if not eligible:
            retained_reference_components.append(component)
            continue
        winner = min(eligible, key=lambda item: (item["wall_time_ratio"], item["candidate_id"]))
        candidate = next(
            item for item in contract["candidates"] if item["id"] == winner["candidate_id"]
        )
        selected_candidates.append(
            {
                **deepcopy(candidate),
                "development_result": result_by_id[candidate["id"]],
                "candidate_sha256": canonical_json_sha256(candidate),
            }
        )
    payload = {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_development_selection",
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": sha256_file(root / CONTRACT_PATH)},
        "budget": budget,
        "analysis_role": budget_contract["analysis_role"],
        "selected_config": {
            "path": _inside(root, selected_path, "Selected config"),
            "sha256": sha256_file(selected_path),
        },
        "metrics": {
            "path": _inside(root, metrics_file, "Development metrics"),
            "sha256": sha256_file(metrics_file),
        },
        "execution_state": {
            "path": _inside(root, execution_state_path, "Development execution state"),
            "sha256": sha256_file(execution_state_path),
        },
        "selection_rule": rule,
        "candidate_results": sorted(results, key=lambda item: item["candidate_id"]),
        "selected_candidates": selected_candidates,
        "retained_reference_components": retained_reference_components,
    }
    payload["selection_sha256"] = canonical_json_sha256(payload)
    validate_json(
        payload,
        root / "schemas/efficiency-development-selection.schema.json",
        "efficiency-development selection",
    )
    return write_idempotent_json(
        output_path, payload, "efficiency-development selection", boundary=root
    ).resolve()


def load_efficiency_development_selection(
    repo_root: Path, path: Path, *, budget: str, selected_config_path: Path
) -> dict[str, Any]:
    """Verify an outcome-bound selection before independent validation may use it."""

    root = repo_root.resolve()
    selection = load_json_object(path, "efficiency-development selection")
    validate_json(
        selection,
        root / "schemas/efficiency-development-selection.schema.json",
        "efficiency-development selection",
    )
    digest_payload = dict(selection)
    recorded = digest_payload.pop("selection_sha256")
    if canonical_json_sha256(digest_payload) != recorded:
        raise BenchmarkConfigError("Efficiency-development selection digest is stale")
    contract = load_efficiency_development_contract(root)
    if (
        selection["budget"] != budget
        or selection["contract"]
        != {"path": CONTRACT_PATH.as_posix(), "sha256": sha256_file(root / CONTRACT_PATH)}
        or selection["selected_config"]["sha256"] != sha256_file(selected_config_path)
        or selection["execution_state"]["sha256"]
        != sha256_file(root / selection["execution_state"]["path"])
        or selection["metrics"]["sha256"] != sha256_file(root / selection["metrics"]["path"])
        or selection["selection_rule"] != contract["selection_rule"]
    ):
        raise BenchmarkConfigError("Efficiency-development selection provenance differs")
    declared = {item["id"]: item for item in contract["candidates"]}
    selected_components = [item["component"] for item in selection["selected_candidates"]]
    retained_components = selection["retained_reference_components"]
    all_components = {"target_eligibility", "adaptive_trees", "combined"}
    if (
        len(selected_components) != len(set(selected_components))
        or set(selected_components) & set(retained_components)
        or set(selected_components) | set(retained_components) != all_components
    ):
        raise BenchmarkConfigError(
            "Efficiency-development selection must resolve every component exactly once"
        )
    for item in selection["selected_candidates"]:
        candidate = declared.get(item["id"])
        if (
            candidate is None
            or item["execution_overrides"] != candidate["execution_overrides"]
            or item["candidate_sha256"] != canonical_json_sha256(candidate)
        ):
            raise BenchmarkConfigError(
                "Efficiency-development finalist differs from the finite grid"
            )
    return selection


__all__ = [
    "CONTRACT_PATH",
    "load_efficiency_development_contract",
    "load_efficiency_development_selection",
    "materialize_efficiency_development",
    "select_efficiency_development",
]
