"""Materialize and select the finite SPATHI efficiency-development grid."""

from __future__ import annotations

import csv
import math
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
from .tool_contract import spathi_runtime_resources

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
        raise BenchmarkConfigError(
            "Efficiency development requires three candidates for each primitive component"
        )

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
        timing_index = int(budget["timing_seed_index"])
        accuracy_indices = tuple(int(item) for item in budget["accuracy_seed_indices"])
        if timing_index != 1 or tuple(sorted(accuracy_indices)) != tuple(
            range(2, len(expected_seeds) + 1)
        ):
            raise BenchmarkConfigError(
                f"{budget_id.title()} must reserve seed 1 for timing and every later seed "
                "for held-back accuracy"
            )
        expected_order_seeds = {
            step: derive_plan_seed(
                seed_plan,
                namespace=namespace,
                stream="execution_order",
                coordinates={
                    "purpose": "counterbalanced_execution_order",
                    "step": step,
                },
            )
            for step in ("component_scout", "joint_confirmation")
        }
        if budget["execution_order_seeds"] != expected_order_seeds:
            raise BenchmarkConfigError(
                f"{budget_id.title()} efficiency-development execution-order seeds are stale"
            )
        profile = load_profile(root, budget["profile"])
        if (
            profile["purpose"] != "efficiency_development"
            or profile["status"] != "planned"
            or profile["inference_batches"]
            or profile["execution_class"] != "efficiency_development_canonical"
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
        scout = budget["steps"]["component_scout"]
        confirmation = budget["steps"]["joint_confirmation"]
        expected_confirmation_indices = [timing_index, *accuracy_indices]
        if (
            scout["algorithm_seed_indices"] != [timing_index]
            or confirmation["algorithm_seed_indices"] != expected_confirmation_indices
        ):
            raise BenchmarkConfigError(
                f"{budget_id.title()} efficiency-development step seed ledger is stale"
            )
        expected_scout = dataset_count * 7
        expected_confirmation = dataset_count * 4 * (1 + len(accuracy_indices))
        if (
            scout["expected_physical_fits"] != expected_scout
            or confirmation["expected_physical_fits"] != expected_confirmation
            or budget["expected_physical_fits"] != expected_scout + expected_confirmation
        ):
            raise BenchmarkConfigError(
                f"{budget_id.title()} efficiency-development physical-fit ledger is stale"
            )
    return contract


def materialize_efficiency_development(
    *,
    repo_root: Path,
    selected_config_path: Path,
    budget: str,
    step: str,
    component_selection_path: Path | None,
    output_dir: Path,
    profile_output: Path,
) -> tuple[Path, Path]:
    """Materialize one prospective efficiency-development step.

    The component scout executes the exact reference and all six primitive
    candidates with the designated timing seed.  Joint confirmation is
    outcome-bound to that scout: it reruns the exact reference, both primitive
    nominees and their combined policy with one common timing seed and every
    held-back accuracy seed. The pipeline isolates all timing blocks in a
    deterministic counterbalanced order, then runs held-back accuracy concurrently.
    """

    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError("Efficiency-development budget must be reduced or full")
    if step not in {"component_scout", "joint_confirmation"}:
        raise BenchmarkConfigError(
            "Efficiency-development step must be component_scout or joint_confirmation"
        )
    root = repo_root.resolve()
    contract = load_efficiency_development_contract(root)
    selected_path = selected_config_path.resolve()
    selected = _selected_policy(root, selected_path, budget)
    reference = selected["selected_policy"]
    if reference["spathi_commit"] != locked_spathi_identity(root)["spathi_commit"]:
        raise BenchmarkConfigError("Efficiency development requires the current SPATHI selection")
    budget_contract = contract["budgets"][budget]
    image = load_image_lock(root)["images"]["spathi"]["tag"]
    seeds = tuple(int(seed) for seed in budget_contract["algorithm_seeds"])
    reference_definition = {
        "id": "spathi_exact_reference",
        "component": "reference",
        "group_policy": reference["group_policy"],
        "selection_role": "base",
        "execution_overrides": {},
    }
    if step == "component_scout":
        if component_selection_path is not None:
            raise BenchmarkConfigError("Component scout cannot consume a prior selection")
        definitions = [
            reference_definition,
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
        scheduled = [(definition, 1, seeds[0]) for definition in definitions]
    else:
        if component_selection_path is None:
            raise BenchmarkConfigError(
                "Joint confirmation requires the content-bound component scout selection"
            )
        component_selection = load_efficiency_component_selection(
            root,
            component_selection_path,
            budget=budget,
            selected_config_path=selected_path,
        )
        nominees = {
            str(item["component"]): item for item in component_selection["component_nominees"]
        }
        if set(nominees) != {"target_eligibility", "adaptive_trees"}:
            raise BenchmarkConfigError("Component scout did not nominate both primitive policies")
        eligibility = nominees["target_eligibility"]
        adaptive = nominees["adaptive_trees"]
        combined_overrides = {
            **deepcopy(eligibility["execution_overrides"]),
            **deepcopy(adaptive["execution_overrides"]),
        }
        combined = {
            "id": "combined_scout_nominees",
            "component": "combined",
            "execution_overrides": combined_overrides,
            "group_policy": _policy_with_overrides(reference["group_policy"], combined_overrides),
            "selection_role": "candidate",
        }
        definitions = [
            reference_definition,
            *[
                {
                    **deepcopy(nominee),
                    "group_policy": _policy_with_overrides(
                        reference["group_policy"], nominee["execution_overrides"]
                    ),
                    "selection_role": "candidate",
                }
                for nominee in (eligibility, adaptive)
            ],
            combined,
        ]
        scheduled = [
            (definition, position, seeds[position - 1])
            for position in range(1, len(seeds) + 1)
            for definition in definitions
        ]

    expected_fit_count = int(budget_contract["steps"][step]["expected_physical_fits"])
    dataset_count = int(budget_contract["expected_independent_datasets"])
    if len(scheduled) * dataset_count != expected_fit_count:
        raise BenchmarkConfigError(
            f"Efficiency-development {step} schedule does not match its physical-fit ledger"
        )
    runs: list[dict[str, Any]] = []
    custom: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for definition, index, seed in scheduled:
        run_id = f"effdev_{step}_{definition['id']}_seed_{seed}"
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
                "name": (f"SPATHI efficiency development {step} {definition['id']} seed {seed}"),
                "docker_image": image,
                "execution_mode": "group_native",
                "extra_inputs": ["groups", "tf_list"],
                "outputs": {"directed": True, "sign": "none"},
                "runtime_resources": spathi_runtime_resources(),
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
                "efficiency_development_step": step,
                "efficiency_component": definition["component"],
                "seed_role": "timing" if index == 1 else "accuracy",
                "execution_order_seed": budget_contract["execution_order_seeds"][step],
                "seed_derivation": {
                    "namespace": f"{budget}.efficiency_development",
                    "stream": "inference",
                    "coordinates": {
                        "purpose": "algorithm_replicate",
                        "replicate_index": index,
                    },
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
    profile["id"] = f"efficiency-development-{budget}-{step.replace('_', '-')}"
    profile["status"] = "runnable"
    profile["inference_batches"] = [
        {
            "id": f"spathi_efficiency_development_{step}",
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
    configuration_seeds: Mapping[str, set[int]],
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
    expected_executions = {
        (configuration, dataset, seed)
        for configuration, seeds in configuration_seeds.items()
        for dataset in datasets
        for seed in seeds
    }
    evaluations = [
        item
        for item in state.get("evaluations", [])
        if item.get("configuration_id") in configuration_seeds
        and item.get("dataset_id") in datasets
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
        for seed in configuration_seeds.get("spathi_exact_reference", set()):
            reference_contexts = contexts_by_execution[("spathi_exact_reference", dataset_id, seed)]
            same_seed_candidates = {
                configuration
                for configuration, seeds in configuration_seeds.items()
                if configuration != "spathi_exact_reference" and seed in seeds
            }
            if any(
                contexts_by_execution[(configuration, dataset_id, seed)] != reference_contexts
                for configuration in same_seed_candidates
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


def _execution_summaries(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[tuple[str, int], tuple[str, str, float, float]]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["configuration_id"], row["dataset_id"], int(row["algorithm_seed"]))].append(
            row
        )
    summaries: dict[str, dict[tuple[str, int], tuple[str, str, float, float]]] = defaultdict(dict)
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
        summaries[configuration][(dataset, seed)] = (
            simulator,
            condition,
            aupr,
            median(wall_times),
        )
    return summaries


def _candidate_result(
    *,
    candidate: Mapping[str, Any],
    summaries: Mapping[str, Mapping[tuple[str, int], tuple[str, str, float, float]]],
    dataset_ids: set[str],
    accuracy_seeds: tuple[int, ...],
    timing_seed: int,
    maximum_loss: float,
) -> dict[str, Any]:
    reference = summaries.get("spathi_exact_reference", {})
    observed = summaries.get(str(candidate["id"]), {})
    accuracy_values: list[tuple[str, str, str, float]] = []
    runtime_values: list[tuple[str, str, str, float]] = []
    for dataset_id in sorted(dataset_ids):
        losses: list[float] = []
        simulator: str | None = None
        condition: str | None = None
        for seed in accuracy_seeds:
            key = (dataset_id, seed)
            if key not in reference or key not in observed:
                raise BenchmarkConfigError(
                    f"Efficiency candidate {candidate['id']!r} lacks paired accuracy evidence"
                )
            ref_simulator, ref_condition, ref_aupr, _ref_time = reference[key]
            cand_simulator, cand_condition, cand_aupr, _cand_time = observed[key]
            if (ref_simulator, ref_condition) != (cand_simulator, cand_condition):
                raise BenchmarkConfigError("Candidate/reference development strata differ")
            simulator, condition = ref_simulator, ref_condition
            losses.append(ref_aupr - cand_aupr)
        if simulator is None or condition is None:
            raise AssertionError("Accuracy seed contract produced no evidence")
        accuracy_values.append((simulator, condition, dataset_id, fmean(losses)))

        timing_key = (dataset_id, timing_seed)
        if timing_key not in reference or timing_key not in observed:
            raise BenchmarkConfigError(
                f"Efficiency candidate {candidate['id']!r} lacks paired timing evidence"
            )
        ref_simulator, ref_condition, _ref_aupr, ref_time = reference[timing_key]
        cand_simulator, cand_condition, _cand_aupr, cand_time = observed[timing_key]
        if (ref_simulator, ref_condition) != (cand_simulator, cand_condition):
            raise BenchmarkConfigError("Candidate/reference timing strata differ")
        runtime_values.append(
            (ref_simulator, ref_condition, dataset_id, math.log(cand_time / ref_time))
        )
    mean_loss = _hierarchical_mean(accuracy_values)
    runtime_ratio = math.exp(_hierarchical_mean(runtime_values))
    return {
        "candidate_id": candidate["id"],
        "component": candidate["component"],
        "mean_aupr_loss": mean_loss,
        "wall_time_ratio": runtime_ratio,
        "complete": True,
        "accuracy_eligible": mean_loss <= maximum_loss,
        "dataset_count": len(dataset_ids),
        "accuracy_seed_count": len(accuracy_seeds),
        "timing_seed_count": 1,
    }


def _reference(path: Path, root: Path, label: str) -> dict[str, str]:
    return {"path": _inside(root, path, label), "sha256": sha256_file(path)}


def _verify_selection_digest(selection: Mapping[str, Any], label: str) -> None:
    digest_payload = dict(selection)
    recorded = digest_payload.pop("selection_sha256", None)
    if canonical_json_sha256(digest_payload) != recorded:
        raise BenchmarkConfigError(f"{label} digest is stale")


def select_efficiency_components(
    *,
    repo_root: Path,
    selected_config_path: Path,
    budget: str,
    scout_metrics_path: Path,
    output_path: Path,
) -> Path:
    """Nominate one eligibility and one adaptive policy from the timing-seed scout."""

    root = repo_root.resolve()
    contract = load_efficiency_development_contract(root)
    selected_path = selected_config_path.resolve()
    _selected_policy(root, selected_path, budget)
    metrics_file = scout_metrics_path.resolve()
    _inside(root, metrics_file, "Component-scout metrics")
    budget_contract = contract["budgets"][budget]
    timing_seed = int(budget_contract["algorithm_seeds"][0])
    configuration_seeds = {
        "spathi_exact_reference": {timing_seed},
        **{item["id"]: {timing_seed} for item in contract["candidates"]},
    }
    rows, state_path_value = _validated_metric_rows(
        repo_root=root,
        metrics_file=metrics_file,
        budget_contract=budget_contract,
        configuration_seeds=configuration_seeds,
    )
    summaries = _execution_summaries(rows)
    dataset_ids = {dataset for dataset, _seed in summaries["spathi_exact_reference"]}
    rule = contract["selection_rule"]
    results = [
        _candidate_result(
            candidate=candidate,
            summaries=summaries,
            dataset_ids=dataset_ids,
            accuracy_seeds=(timing_seed,),
            timing_seed=timing_seed,
            maximum_loss=float(rule["maximum_mean_absolute_aupr_loss"]),
        )
        for candidate in contract["candidates"]
    ]
    result_by_id = {item["candidate_id"]: item for item in results}
    nominees: list[dict[str, Any]] = []
    for component in ("target_eligibility", "adaptive_trees"):
        component_results = [item for item in results if item["component"] == component]
        eligible = [item for item in component_results if item["accuracy_eligible"]]
        winner = (
            min(
                eligible,
                key=lambda item: (
                    item["wall_time_ratio"],
                    item["mean_aupr_loss"],
                    item["candidate_id"],
                ),
            )
            if eligible
            else min(
                component_results,
                key=lambda item: (
                    item["mean_aupr_loss"],
                    item["wall_time_ratio"],
                    item["candidate_id"],
                ),
            )
        )
        candidate = next(
            item for item in contract["candidates"] if item["id"] == winner["candidate_id"]
        )
        nominees.append(
            {
                **deepcopy(candidate),
                "scout_result": result_by_id[candidate["id"]],
                "candidate_sha256": canonical_json_sha256(candidate),
            }
        )
    payload = {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_component_selection",
        "contract": _reference(root / CONTRACT_PATH, root, "Efficiency contract"),
        "budget": budget,
        "analysis_role": budget_contract["analysis_role"],
        "selected_config": _reference(selected_path, root, "Selected config"),
        "scout_metrics": _reference(metrics_file, root, "Component-scout metrics"),
        "scout_execution_state": _reference(
            state_path_value, root, "Component-scout execution state"
        ),
        "selection_rule": rule,
        "candidate_results": sorted(results, key=lambda item: item["candidate_id"]),
        "component_nominees": nominees,
    }
    payload["selection_sha256"] = canonical_json_sha256(payload)
    validate_json(
        payload,
        root / "schemas/efficiency-component-selection.schema.json",
        "efficiency component selection",
    )
    return write_idempotent_json(
        output_path, payload, "efficiency component selection", boundary=root
    ).resolve()


def load_efficiency_component_selection(
    repo_root: Path, path: Path, *, budget: str, selected_config_path: Path
) -> dict[str, Any]:
    """Verify the content-bound primitive nominees used to construct confirmation."""

    root = repo_root.resolve()
    selection = load_json_object(path, "efficiency component selection")
    validate_json(
        selection,
        root / "schemas/efficiency-component-selection.schema.json",
        "efficiency component selection",
    )
    _verify_selection_digest(selection, "Efficiency component selection")
    contract = load_efficiency_development_contract(root)
    if (
        selection["budget"] != budget
        or selection["contract"] != _reference(root / CONTRACT_PATH, root, "contract")
        or selection["selected_config"]["sha256"] != sha256_file(selected_config_path)
        or selection["selection_rule"] != contract["selection_rule"]
    ):
        raise BenchmarkConfigError("Efficiency component selection provenance differs")
    for field in ("scout_metrics", "scout_execution_state"):
        reference = selection[field]
        if reference["sha256"] != sha256_file(root / reference["path"]):
            raise BenchmarkConfigError(f"Efficiency component selection {field} changed")
    declared = {item["id"]: item for item in contract["candidates"]}
    nominees = selection["component_nominees"]
    if {item["component"] for item in nominees} != {
        "target_eligibility",
        "adaptive_trees",
    }:
        raise BenchmarkConfigError("Efficiency component selection lacks one primitive nominee")
    for item in nominees:
        candidate = declared.get(item["id"])
        if (
            candidate is None
            or any(item[key] != value for key, value in candidate.items())
            or item["candidate_sha256"] != canonical_json_sha256(candidate)
        ):
            raise BenchmarkConfigError("Efficiency component nominee differs from its finite grid")
    return selection


def select_efficiency_development(
    *,
    repo_root: Path,
    selected_config_path: Path,
    budget: str,
    component_selection_path: Path,
    confirmation_metrics_path: Path,
    output_path: Path,
) -> Path:
    """Choose at most one global policy using held-back accuracy and timing-only cost."""

    root = repo_root.resolve()
    contract = load_efficiency_development_contract(root)
    selected_path = selected_config_path.resolve()
    _selected_policy(root, selected_path, budget)
    component_selection = load_efficiency_component_selection(
        root,
        component_selection_path,
        budget=budget,
        selected_config_path=selected_path,
    )
    budget_contract = contract["budgets"][budget]
    seeds = tuple(int(seed) for seed in budget_contract["algorithm_seeds"])
    timing_seed = seeds[0]
    accuracy_seeds = tuple(seeds[index - 1] for index in budget_contract["accuracy_seed_indices"])
    nominees = {str(item["component"]): item for item in component_selection["component_nominees"]}
    combined_candidate = {
        "id": "combined_scout_nominees",
        "component": "combined",
        "execution_overrides": {
            **deepcopy(nominees["target_eligibility"]["execution_overrides"]),
            **deepcopy(nominees["adaptive_trees"]["execution_overrides"]),
        },
    }
    candidates = (
        {key: deepcopy(value) for key, value in nominees.items()},
        combined_candidate,
    )
    candidate_list = [*candidates[0].values(), candidates[1]]

    scout_metrics = root / component_selection["scout_metrics"]["path"]
    _scout_rows, scout_state = _validated_metric_rows(
        repo_root=root,
        metrics_file=scout_metrics,
        budget_contract=budget_contract,
        configuration_seeds={
            "spathi_exact_reference": {timing_seed},
            **{item["id"]: {timing_seed} for item in contract["candidates"]},
        },
    )
    confirmation_metrics = confirmation_metrics_path.resolve()
    _inside(root, confirmation_metrics, "Joint-confirmation metrics")
    confirmation_rows, confirmation_state = _validated_metric_rows(
        repo_root=root,
        metrics_file=confirmation_metrics,
        budget_contract=budget_contract,
        configuration_seeds={
            "spathi_exact_reference": {timing_seed, *accuracy_seeds},
            nominees["target_eligibility"]["id"]: {timing_seed, *accuracy_seeds},
            nominees["adaptive_trees"]["id"]: {timing_seed, *accuracy_seeds},
            combined_candidate["id"]: {timing_seed, *accuracy_seeds},
        },
    )
    summaries = _execution_summaries(confirmation_rows)
    reference_keys = set(summaries["spathi_exact_reference"])
    dataset_ids = {dataset for dataset, seed in reference_keys if seed == timing_seed}
    if not dataset_ids or any(
        (dataset, seed) not in reference_keys for dataset in dataset_ids for seed in accuracy_seeds
    ):
        raise BenchmarkConfigError("Efficiency development reference panel is incomplete")
    rule = contract["selection_rule"]
    results = [
        _candidate_result(
            candidate=candidate,
            summaries=summaries,
            dataset_ids=dataset_ids,
            accuracy_seeds=accuracy_seeds,
            timing_seed=timing_seed,
            maximum_loss=float(rule["maximum_mean_absolute_aupr_loss"]),
        )
        for candidate in candidate_list
    ]
    eligible = [
        result
        for result in results
        if result["accuracy_eligible"] and result["wall_time_ratio"] < 1.0
    ]
    nominee: dict[str, Any] | None = None
    if eligible:
        winner = min(eligible, key=lambda item: (item["wall_time_ratio"], item["candidate_id"]))
        candidate = next(item for item in candidate_list if item["id"] == winner["candidate_id"])
        canonical_candidate = {
            "id": candidate["id"],
            "component": candidate["component"],
            "execution_overrides": deepcopy(candidate["execution_overrides"]),
        }
        nominee = {
            **canonical_candidate,
            "development_result": winner,
            "candidate_sha256": canonical_json_sha256(canonical_candidate),
        }
    payload = {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_development_selection",
        "contract": _reference(root / CONTRACT_PATH, root, "Efficiency contract"),
        "budget": budget,
        "analysis_role": budget_contract["analysis_role"],
        "selected_config": _reference(selected_path, root, "Selected config"),
        "component_selection": _reference(
            component_selection_path.resolve(), root, "Component selection"
        ),
        "evidence": {
            "component_scout_metrics": _reference(scout_metrics, root, "Component-scout metrics"),
            "component_scout_execution_state": _reference(
                scout_state, root, "Component-scout state"
            ),
            "joint_confirmation_metrics": _reference(
                confirmation_metrics, root, "Joint-confirmation metrics"
            ),
            "joint_confirmation_execution_state": _reference(
                confirmation_state, root, "Joint-confirmation state"
            ),
        },
        "selection_rule": rule,
        "candidate_results": sorted(results, key=lambda item: item["candidate_id"]),
        "nominee": nominee,
        "reference_retained": nominee is None,
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
    """Verify the single global nominee before independent validation may use it."""

    root = repo_root.resolve()
    selection = load_json_object(path, "efficiency-development selection")
    validate_json(
        selection,
        root / "schemas/efficiency-development-selection.schema.json",
        "efficiency-development selection",
    )
    _verify_selection_digest(selection, "Efficiency-development selection")
    contract = load_efficiency_development_contract(root)
    if (
        selection["budget"] != budget
        or selection["contract"] != _reference(root / CONTRACT_PATH, root, "contract")
        or selection["selected_config"]["sha256"] != sha256_file(selected_config_path)
        or selection["selection_rule"] != contract["selection_rule"]
    ):
        raise BenchmarkConfigError("Efficiency-development selection provenance differs")
    for reference in selection["evidence"].values():
        if reference["sha256"] != sha256_file(root / reference["path"]):
            raise BenchmarkConfigError("Efficiency-development evidence changed after selection")
    component_reference = selection["component_selection"]
    if component_reference["sha256"] != sha256_file(root / component_reference["path"]):
        raise BenchmarkConfigError("Efficiency component selection changed after confirmation")
    nominee = selection["nominee"]
    if (nominee is None) != bool(selection["reference_retained"]):
        raise BenchmarkConfigError("Efficiency global decision is inconsistent")
    if nominee is not None:
        canonical_candidate = {
            key: nominee[key] for key in ("id", "component", "execution_overrides")
        }
        if nominee["candidate_sha256"] != canonical_json_sha256(canonical_candidate):
            raise BenchmarkConfigError("Efficiency nominee digest is stale")
    return selection


__all__ = [
    "CONTRACT_PATH",
    "load_efficiency_component_selection",
    "load_efficiency_development_contract",
    "load_efficiency_development_selection",
    "materialize_efficiency_development",
    "select_efficiency_components",
    "select_efficiency_development",
]
