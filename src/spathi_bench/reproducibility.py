"""Deterministic audit of every committed benchmark seed assignment.

The audit deliberately reads only versioned design inputs.  Its result contains
logical artifact identifiers and canonical JSON hashes, never execution paths or
timestamps, so two checkouts of the same design produce byte-identical evidence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .algorithm_seed_contract import algorithm_seeds_by_configuration
from .artifacts import canonical_json_sha256, sha256_file
from .config import validate_json
from .safe_io import atomic_write_json
from .seeds import (
    MAX_SEED,
    SeedPlanError,
    derive_plan_seed,
    load_seed_plan,
    plan_seed_record,
    validate_plan_seed_record,
)

_SEED_PLAN_MEMBER = "experiments/v1/seeds/seed-plan.json"


class ReproducibilityAuditError(ValueError):
    """Raised when a committed stochastic input violates the seed contract."""


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReproducibilityAuditError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReproducibilityAuditError(f"{label} must contain a JSON object")
    return payload


def _positive_seed(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_SEED:
        raise ReproducibilityAuditError(f"{label} must be an integer between 1 and {MAX_SEED}")
    return value


def _logical_id(path: Path, root: Path) -> str:
    """Build a stable identifier without exposing a checkout-dependent path."""

    try:
        relative = path.relative_to(root / "experiments" / "v1")
    except ValueError as exc:
        raise ReproducibilityAuditError("Audited inputs must be below experiments/v1") from exc
    parts = list(relative.with_suffix("").parts)
    return ".".join(parts)


def _seed_derivation(
    value: Any,
    *,
    plan: Mapping[str, Any],
    recorded_seed: int,
    label: str,
    required_stream: str,
) -> dict[str, Any]:
    try:
        record = validate_plan_seed_record(
            plan,
            seed=recorded_seed,
            seed_derivation=value,
            required_stream=required_stream,
        )
    except (TypeError, SeedPlanError) as exc:
        raise ReproducibilityAuditError(f"Invalid {label}: {exc}") from exc
    return record["seed_derivation"]


def _generation_allocations(plan: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    consumers: dict[tuple[str, str], dict[str, Any]] = {}
    for allocation in plan["allocations"]:
        if allocation["stream"] != "generation":
            continue
        for consumer in allocation.get("consumers", []):
            key = (consumer["path"], consumer["run_id"])
            if key in consumers:
                raise ReproducibilityAuditError(
                    f"Generation consumer {consumer['run_id']!r} is declared more than once"
                )
            consumers[key] = {
                "namespace": allocation["namespace"],
                "seed": allocation["seed"] + consumer["offset"],
                "span": consumer["span"],
            }
    return consumers


def _audit_simulator_runs(
    root: Path,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int, int]:
    simulation_root = root / "experiments" / "v1" / "simulations"
    paths = sorted(simulation_root.rglob("*runs.json"))
    declared = _generation_allocations(plan)
    observed: set[tuple[str, str]] = set()
    artifacts: list[dict[str, Any]] = []
    run_count = 0
    replicate_count = 0

    for path in paths:
        payload = _read_object(path, f"simulator run manifest {path.name}")
        runs = payload.get("runs")
        if payload.get("schema_version") != "1.0" or not isinstance(runs, list) or not runs:
            raise ReproducibilityAuditError(
                f"Simulator run manifest {path.name} must be a non-empty v1 runs document"
            )
        relative = path.relative_to(root).as_posix()
        assignments: list[dict[str, Any]] = []
        manifest_ids: set[str] = set()
        for index, run in enumerate(runs):
            label = f"{path.name}.runs[{index}]"
            if not isinstance(run, Mapping):
                raise ReproducibilityAuditError(f"{label} must be an object")
            run_id = run.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ReproducibilityAuditError(f"{label}.run_id must be a non-empty string")
            if run_id in manifest_ids:
                raise ReproducibilityAuditError(f"Duplicate simulator run_id {run_id!r}")
            manifest_ids.add(run_id)
            key = (relative, run_id)
            allocation = declared.get(key)
            if allocation is None:
                raise ReproducibilityAuditError(
                    f"Simulator run {run_id!r} has no generation allocation in the seed plan"
                )
            seed = _positive_seed(run.get("base_seed"), f"{label}.base_seed")
            replicates = run.get("replicates")
            if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
                raise ReproducibilityAuditError(f"{label}.replicates must be an integer >= 1")
            if seed != allocation["seed"] or replicates != allocation["span"]:
                raise ReproducibilityAuditError(
                    f"Simulator run {run_id!r} records seed/span {seed}/{replicates}; "
                    f"the seed plan declares {allocation['seed']}/{allocation['span']}"
                )
            observed.add(key)
            run_count += 1
            replicate_count += replicates
            assignments.append(
                {
                    "run_id": run_id,
                    "namespace": allocation["namespace"],
                    "base_seed": seed,
                    "replicates": replicates,
                }
            )
        artifacts.append(
            {
                "artifact_id": _logical_id(path, root),
                "kind": "simulator_run_manifest",
                "content_sha256": canonical_json_sha256(payload),
                "seed_assignment_sha256": canonical_json_sha256(assignments),
                "stochastic_units": sum(item["replicates"] for item in assignments),
            }
        )

    missing = sorted(set(declared) - observed)
    if missing:
        missing_ids = ", ".join(run_id for _, run_id in missing)
        raise ReproducibilityAuditError(
            f"Generation allocations reference absent simulator runs: {missing_ids}"
        )
    return artifacts, run_count, replicate_count


def _audit_dimensional_transfer_inference_seeds(
    root: Path,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    from .config import BenchmarkConfigError
    from .dimensional_transfer import transfer_algorithm_seed, validate_transfer_panel

    try:
        panel = validate_transfer_panel(root)
        assignments = []
        for budget in ("reduced", "full"):
            seed, derivation = transfer_algorithm_seed(root, budget=budget)
            expected = derive_plan_seed(
                plan,
                namespace=f"{budget}.dimensional_transfer",
                stream="inference",
                coordinates={"purpose": "algorithm_replicate", "replicate_index": 1},
            )
            if seed != expected:
                raise ReproducibilityAuditError(
                    f"{budget} dimensional-transfer algorithm seed is stale"
                )
            assignments.append(
                {
                    "budget": budget,
                    "algorithm_seed": seed,
                    "seed_derivation": derivation,
                    "reuse_contract": "paired_exact_and_production_within_dataset",
                }
            )
    except BenchmarkConfigError as exc:
        raise ReproducibilityAuditError(
            f"Invalid dimensional-transfer seed contract: {exc}"
        ) from exc
    return (
        {
            "artifact_id": "spathi.dimensional-transfer-inference",
            "kind": "dimensional_transfer_seed_contract",
            "content_sha256": canonical_json_sha256(panel),
            "seed_assignment_sha256": canonical_json_sha256(assignments),
            "stochastic_units": len(assignments),
        },
        len(assignments),
    )


def _request_runs(root: Path) -> dict[str, list[tuple[Path, Mapping[str, Any]]]]:
    indexed: dict[str, list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
    roots = [
        root / "experiments" / "v1" / "inference",
        root / "experiments" / "v1" / "spathi",
    ]
    for source_root in roots:
        for path in sorted(source_root.rglob("*.json")):
            if path.name.endswith("metadata.json"):
                continue
            payload = _read_object(path, f"inference request candidate {path.name}")
            runs = payload.get("runs")
            if not isinstance(runs, list):
                continue
            observed_ids: set[str] = set()
            for run in runs:
                if (
                    isinstance(run, Mapping)
                    and isinstance(run.get("run_id"), str)
                    and isinstance(run.get("params"), Mapping)
                ):
                    if run["run_id"] in observed_ids:
                        raise ReproducibilityAuditError(
                            f"Inference request {path.name} contains duplicate run_id "
                            f"{run['run_id']!r}"
                        )
                    observed_ids.add(run["run_id"])
                    indexed[run["run_id"]].append((path, run))
    return indexed


def _direct_request_path(metadata_path: Path) -> Path:
    return metadata_path.with_name(metadata_path.name.removesuffix("-metadata.json") + ".json")


def _matching_requests(
    metadata_path: Path,
    run_id: str,
    request_index: Mapping[str, list[tuple[Path, Mapping[str, Any]]]],
) -> list[Mapping[str, Any]]:
    candidates = request_index.get(run_id, [])
    direct = _direct_request_path(metadata_path)
    direct_matches = [run for path, run in candidates if path == direct]
    if direct_matches:
        return direct_matches
    return [run for _, run in candidates]


def _audit_inference_metadata(
    root: Path,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int, int]:
    metadata_paths = sorted(
        [
            *(root / "experiments" / "v1" / "inference").rglob("*metadata.json"),
            *(root / "experiments" / "v1" / "spathi").rglob("*metadata.json"),
        ]
    )
    request_index = _request_runs(root)
    artifacts: list[dict[str, Any]] = []
    stochastic_count = 0
    deterministic_count = 0
    all_observed_ids: set[str] = set()

    for path in metadata_paths:
        payload = _read_object(path, f"inference metadata {path.name}")
        runs = payload.get("runs")
        if payload.get("schema_version") != "1.0" or not isinstance(runs, list) or not runs:
            raise ReproducibilityAuditError(
                f"Inference metadata {path.name} must be a non-empty v1 runs document"
            )
        assignments: list[dict[str, Any]] = []
        observed_ids: set[str] = set()
        for index, run in enumerate(runs):
            label = f"{path.name}.runs[{index}]"
            if not isinstance(run, Mapping):
                raise ReproducibilityAuditError(f"{label} must be an object")
            run_id = run.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ReproducibilityAuditError(f"{label}.run_id must be a non-empty string")
            if run_id in observed_ids:
                raise ReproducibilityAuditError(f"Duplicate metadata run_id {run_id!r}")
            observed_ids.add(run_id)
            all_observed_ids.add(run_id)
            requests = _matching_requests(path, run_id, request_index)
            if not requests:
                raise ReproducibilityAuditError(
                    f"Inference metadata run {run_id!r} has no committed request parameters"
                )

            seed = run.get("algorithm_seed")
            parameter = run.get("algorithm_seed_parameter")
            derivation = run.get("seed_derivation")
            if seed is None:
                if parameter is not None or derivation is not None:
                    raise ReproducibilityAuditError(
                        f"Deterministic metadata run {run_id!r} cannot declare seed fields"
                    )
                deterministic_count += 1
                assignments.append({"run_id": run_id, "algorithm_seed": None})
                continue

            validated_seed = _positive_seed(seed, f"{label}.algorithm_seed")
            if not isinstance(parameter, str) or not parameter:
                raise ReproducibilityAuditError(
                    f"{label}.algorithm_seed_parameter must name the request parameter"
                )
            validated_derivation = _seed_derivation(
                derivation,
                plan=plan,
                recorded_seed=validated_seed,
                label=f"{label}.seed_derivation",
                required_stream="inference",
            )
            for request in requests:
                params = request["params"]
                if params.get(parameter) != validated_seed:
                    raise ReproducibilityAuditError(
                        f"Request {run_id!r} does not expose metadata seed {validated_seed} "
                        f"through parameter {parameter!r}"
                    )
            stochastic_count += 1
            assignments.append(
                {
                    "run_id": run_id,
                    "configuration_id": run.get("configuration_id"),
                    "algorithm_seed": validated_seed,
                    "algorithm_seed_parameter": parameter,
                    "seed_derivation": validated_derivation,
                }
            )
        artifacts.append(
            {
                "artifact_id": _logical_id(path, root),
                "kind": "inference_run_metadata",
                "content_sha256": canonical_json_sha256(payload),
                "seed_assignment_sha256": canonical_json_sha256(assignments),
                "stochastic_units": sum(item["algorithm_seed"] is not None for item in assignments),
            }
        )
    requests_without_metadata = sorted(set(request_index) - all_observed_ids)
    if requests_without_metadata:
        raise ReproducibilityAuditError(
            "Committed inference requests lack seed metadata: "
            + ", ".join(requests_without_metadata)
        )
    return artifacts, stochastic_count, deterministic_count


def _audit_scale_stress_requests(
    root: Path,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int, int]:
    request_root = root / "experiments" / "v1" / "simulations" / "scale_stress"
    artifacts: list[dict[str, Any]] = []
    dataset_count = 0
    component_seed_count = 0
    observed_dataset_ids: set[str] = set()
    for path in sorted(request_root.glob("*.json")):
        payload = _read_object(path, f"scale-stress generator request {path.name}")
        datasets = payload.get("datasets")
        request_id = payload.get("id")
        if (
            payload.get("schema_version") != "1.0"
            or not isinstance(request_id, str)
            or not request_id
            or not isinstance(datasets, list)
            or not datasets
        ):
            raise ReproducibilityAuditError(
                f"scale-stress request {path.name} must be a non-empty named v1 request"
            )
        if payload.get("seed_plan") != _SEED_PLAN_MEMBER:
            raise ReproducibilityAuditError(
                f"scale-stress request {request_id!r} is not bound to the canonical seed plan"
            )
        assignments: list[dict[str, Any]] = []
        for index, dataset in enumerate(datasets):
            label = f"{path.name}.datasets[{index}]"
            if not isinstance(dataset, Mapping):
                raise ReproducibilityAuditError(f"{label} must be an object")
            dataset_id = dataset.get("id")
            if not isinstance(dataset_id, str) or not dataset_id:
                raise ReproducibilityAuditError(f"{label}.id must be a non-empty string")
            if dataset_id in observed_dataset_ids:
                raise ReproducibilityAuditError(f"Duplicate scale-stress dataset id {dataset_id!r}")
            observed_dataset_ids.add(dataset_id)
            seed = _positive_seed(dataset.get("seed"), f"{label}.seed")
            raw_derivation = dataset.get("seed_derivation")
            derivation = _seed_derivation(
                raw_derivation,
                plan=plan,
                recorded_seed=seed,
                label=f"{label}.seed_derivation",
                required_stream="augmentation",
            )
            budget = derivation["namespace"].split(".", 1)[0]
            expected_derivation = {
                "namespace": f"{budget}.scalability",
                "stream": "augmentation",
                "coordinates": {
                    "phase": "scalability",
                    "dataset_id": dataset_id,
                    "augmentation_contract": "scale_stress_v1",
                },
            }
            if derivation != expected_derivation:
                raise ReproducibilityAuditError(
                    f"{label}.seed_derivation does not describe the canonical "
                    "scale-stress scalability augmentation"
                )
            component_seeds = {
                component: plan_seed_record(
                    plan,
                    namespace=derivation["namespace"],
                    stream="augmentation",
                    coordinates={
                        **derivation["coordinates"],
                        "component": component,
                    },
                )
                for component in ("groups", "regulatory_model", "expression", "dropout")
            }
            assignments.append(
                {
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "seed_derivation": derivation,
                    "effective_component_seeds": component_seeds,
                }
            )
            dataset_count += 1
            component_seed_count += len(component_seeds)
        artifacts.append(
            {
                "artifact_id": request_id,
                "kind": "scale_stress_generator_request",
                "content_sha256": canonical_json_sha256(payload),
                "seed_assignment_sha256": canonical_json_sha256(assignments),
                "stochastic_units": sum(
                    1 + len(item["effective_component_seeds"]) for item in assignments
                ),
            }
        )
    return artifacts, dataset_count, component_seed_count


def _audit_calibration_seed_contract(
    root: Path,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    path = root / "experiments/v1/spathi/search-space.json"
    payload = _read_object(path, "calibration search space")
    budgets = payload.get("calibration_budgets")
    if payload.get("schema_version") != "1.0" or not isinstance(budgets, Mapping):
        raise ReproducibilityAuditError(
            "Calibration search space must contain v1 calibration budgets"
        )
    assignments: list[dict[str, Any]] = []
    observed_values: set[int] = set()
    for budget_id in ("reduced", "full"):
        budget = budgets.get(budget_id)
        seed_sets = budget.get("algorithmic_seeds") if isinstance(budget, Mapping) else None
        if not isinstance(seed_sets, Mapping):
            raise ReproducibilityAuditError(
                f"Calibration budget {budget_id!r} has no algorithmic seed contract"
            )
        expected_stages = {
            "geometry",
            "estimator",
            "interaction",
            "estimator_scale_transfer",
            "convergence",
            "validation",
        }
        if set(seed_sets) != expected_stages:
            raise ReproducibilityAuditError(
                f"Calibration budget {budget_id!r} must seed exactly {sorted(expected_stages)}"
            )
        for stage_id in sorted(expected_stages):
            seeds = seed_sets[stage_id]
            if not isinstance(seeds, list) or len(seeds) < 2:
                raise ReproducibilityAuditError(
                    f"Calibration {budget_id}.{stage_id} needs at least two algorithm seeds"
                )
            role = "validation" if stage_id == "validation" else "development"
            namespace = f"{budget_id}.{role}"
            for replicate_index, raw_seed in enumerate(seeds, start=1):
                seed = _positive_seed(
                    raw_seed,
                    f"calibration.{budget_id}.{stage_id}[{replicate_index - 1}]",
                )
                record = plan_seed_record(
                    plan,
                    namespace=namespace,
                    stream="inference",
                    coordinates={
                        "phase": "calibration",
                        "stage": stage_id,
                        "replicate": replicate_index,
                    },
                )
                if seed != record["seed"]:
                    raise ReproducibilityAuditError(
                        f"Calibration {budget_id}.{stage_id} seed {replicate_index} "
                        f"derives {record['seed']}, not {seed}"
                    )
                if seed in observed_values:
                    raise ReproducibilityAuditError(
                        f"Calibration reuses algorithm seed {seed} across semantic tasks"
                    )
                observed_values.add(seed)
                assignments.append(
                    {
                        "budget": budget_id,
                        "stage": stage_id,
                        "replicate_index": replicate_index,
                        **record,
                    }
                )
    return (
        {
            "artifact_id": "spathi.search-space",
            "kind": "calibration_seed_contract",
            "content_sha256": canonical_json_sha256(payload),
            "seed_assignment_sha256": canonical_json_sha256(assignments),
            "stochastic_units": len(assignments),
        },
        len(assignments),
    )


def _audit_statistical_seed_contracts(
    root: Path,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    sources = (("experiments/v1/inference/plans/full-v1.json", "full.test"),)
    artifacts: list[dict[str, Any]] = []
    for relative, namespace in sources:
        path = root / relative
        payload = _read_object(path, f"statistical contract {relative}")
        contract = payload.get("statistics")
        if not isinstance(contract, Mapping) or contract.get("status") != "inferential":
            raise ReproducibilityAuditError(f"Statistical contract {relative} must be inferential")
        seed = _positive_seed(contract.get("random_seed"), f"{relative}.random_seed")
        derivation = _seed_derivation(
            contract.get("seed_derivation"),
            plan=plan,
            recorded_seed=seed,
            label=f"{relative}.seed_derivation",
            required_stream="statistical",
        )
        expected_derivation = {
            "namespace": namespace,
            "stream": "statistical",
            "coordinates": {"purpose": "phase_root"},
        }
        if derivation != expected_derivation:
            raise ReproducibilityAuditError(
                f"Statistical contract {relative} uses the wrong semantic seed domain"
            )
        expected_algorithm_seeds = [
            int(
                plan_seed_record(
                    plan,
                    namespace=namespace,
                    stream="inference",
                    coordinates={
                        "purpose": "algorithm_replicate",
                        "replicate_index": replicate_index,
                    },
                )["seed"]
            )
            for replicate_index in range(1, 4)
        ]
        for comparison in contract.get("comparisons", []):
            try:
                seed_contract = algorithm_seeds_by_configuration(comparison)
            except ValueError as exc:
                raise ReproducibilityAuditError(
                    f"Invalid algorithm-seed contract in {relative}: {exc}"
                ) from exc
            for configuration_id, seeds in seed_contract.items():
                if list(seeds) not in ([None], expected_algorithm_seeds):
                    raise ReproducibilityAuditError(
                        f"Statistical comparison {comparison.get('id')!r} configuration "
                        f"{configuration_id!r} in {relative} must declare either the exact "
                        "inference seed panel or one null caller seed"
                    )
        assignment = {"seed": seed, "seed_derivation": derivation}
        artifacts.append(
            {
                "artifact_id": _logical_id(path, root),
                "kind": "statistical_seed_contract",
                "content_sha256": canonical_json_sha256(payload),
                "seed_assignment_sha256": canonical_json_sha256(assignment),
                "stochastic_units": 1,
            }
        )
    return artifacts, len(artifacts)


def _audit_comparison_seed_contract(
    root: Path,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    repetitions = {"smoke.smoke": 1, "reduced.test": 3, "full.test": 3}
    expected: dict[str, dict[int, int]] = {}
    assignments: list[dict[str, Any]] = []
    for namespace, count in repetitions.items():
        expected[namespace] = {}
        for replicate_index in range(1, count + 1):
            record = plan_seed_record(
                plan,
                namespace=namespace,
                stream="inference",
                coordinates={
                    "purpose": "algorithm_replicate",
                    "replicate_index": replicate_index,
                },
            )
            expected[namespace][replicate_index] = int(record["seed"])
            assignments.append(
                {
                    "namespace": namespace,
                    "replicate_index": replicate_index,
                    **record,
                }
            )

    observed: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    observed_by_configuration: dict[str, dict[str, set[int | None]]] = defaultdict(
        lambda: defaultdict(set)
    )
    observed_run_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    metadata_hashes: list[dict[str, str]] = []
    for path in sorted(
        [
            *(root / "experiments/v1/inference").rglob("*metadata.json"),
            *(root / "experiments/v1/spathi").rglob("*metadata.json"),
        ]
    ):
        payload = _read_object(path, f"comparison seed metadata {path.name}")
        metadata_hashes.append(
            {"artifact_id": _logical_id(path, root), "sha256": canonical_json_sha256(payload)}
        )
        runs = [run for run in payload.get("runs", []) if isinstance(run, Mapping)]
        payload_namespaces = {
            str(derivation["namespace"])
            for run in runs
            if run.get("algorithm_seed") is not None
            and isinstance((derivation := run.get("seed_derivation")), Mapping)
            and derivation.get("namespace") in expected
        }
        for run in runs:
            configuration_id = run.get("configuration_id")
            if not isinstance(configuration_id, str) or not configuration_id:
                raise ReproducibilityAuditError(
                    f"Comparison metadata {path.name} has no configuration identity"
                )
            if run.get("algorithm_seed") is None:
                if run.get("algorithm_seed_parameter") is not None:
                    raise ReproducibilityAuditError(
                        f"Comparison metadata {path.name} gives null seed run "
                        f"{run.get('run_id')!r} a seed parameter"
                    )
                if not payload_namespaces:
                    continue
                if len(payload_namespaces) != 1:
                    raise ReproducibilityAuditError(
                        f"Comparison metadata {path.name} mixes seed namespaces around a null "
                        "caller-seed execution"
                    )
                namespace = next(iter(payload_namespaces))
                observed_by_configuration[namespace][configuration_id].add(None)
                observed_run_counts[namespace][configuration_id] += 1
                continue
            derivation = run.get("seed_derivation")
            if not isinstance(derivation, Mapping):
                continue
            namespace = derivation.get("namespace")
            coordinates = derivation.get("coordinates")
            if namespace not in expected or not isinstance(coordinates, Mapping):
                continue
            if (
                set(coordinates) != {"purpose", "replicate_index"}
                or coordinates.get("purpose") != "algorithm_replicate"
            ):
                raise ReproducibilityAuditError(
                    f"Comparison metadata {path.name} has noncanonical inference coordinates"
                )
            replicate_index = coordinates.get("replicate_index")
            if not isinstance(replicate_index, int) or isinstance(replicate_index, bool):
                raise ReproducibilityAuditError(
                    f"Comparison metadata {path.name} has an invalid replicate index"
                )
            observed[str(namespace)][replicate_index].add(int(run["algorithm_seed"]))
            observed_by_configuration[str(namespace)][configuration_id].add(
                int(run["algorithm_seed"])
            )
            observed_run_counts[str(namespace)][configuration_id] += 1
    normalized_observed = {
        namespace: {
            index: next(iter(values)) for index, values in per_replicate.items() if len(values) == 1
        }
        for namespace, per_replicate in observed.items()
    }
    if (
        any(
            len(values) != 1
            for per_replicate in observed.values()
            for values in per_replicate.values()
        )
        or normalized_observed != expected
    ):
        raise ReproducibilityAuditError(
            "Committed comparison metadata do not expose the exact predeclared seed panel"
        )

    contract_sources = {
        "smoke.smoke": root / "experiments/v1/profiles/smoke.json",
        "reduced.test": root / "experiments/v1/inference/plans/reduced-v1.json",
        "full.test": root / "experiments/v1/inference/plans/full-v1.json",
    }
    for namespace, source_path in contract_sources.items():
        contract = _read_object(source_path, f"comparison contract {source_path.name}").get(
            "statistics"
        )
        if not isinstance(contract, Mapping):
            raise ReproducibilityAuditError(
                f"Comparison contract {source_path.name} has no statistics object"
            )
        expected_mode = "inferential" if namespace == "full.test" else "descriptive"
        if contract.get("status") != expected_mode:
            raise ReproducibilityAuditError(
                f"Comparison contract {source_path.name} must be {expected_mode}"
            )
        if expected_mode == "descriptive" and set(contract) != {"status", "comparisons"}:
            raise ReproducibilityAuditError(
                f"Descriptive comparison contract {source_path.name} contains inferential "
                "parameters"
            )
        for comparison in contract.get("comparisons", []):
            if not isinstance(comparison, Mapping):
                raise ReproducibilityAuditError(
                    f"Comparison contract {source_path.name} contains a non-object comparison"
                )
            try:
                seed_contract = algorithm_seeds_by_configuration(comparison)
            except ValueError as exc:
                raise ReproducibilityAuditError(
                    f"Invalid algorithm-seed contract in {source_path.name}: {exc}"
                ) from exc
            for configuration_id, declared_seeds in seed_contract.items():
                if configuration_id == "selected_spathi":
                    continue
                observed_seeds = observed_by_configuration[namespace].get(configuration_id, set())
                if observed_seeds != set(declared_seeds):
                    raise ReproducibilityAuditError(
                        f"Comparison configuration {configuration_id!r} in {namespace} exposes "
                        f"algorithm seeds {sorted(observed_seeds, key=lambda value: -1 if value is None else value)}, "
                        f"not {list(declared_seeds)}"
                    )
                if (
                    declared_seeds == (None,)
                    and observed_run_counts[namespace].get(configuration_id, 0) != 1
                ):
                    raise ReproducibilityAuditError(
                        f"Non-caller-seeded comparison configuration {configuration_id!r} in "
                        f"{namespace} must have exactly one request run"
                    )
    return (
        {
            "artifact_id": "comparison.algorithm-seed-panel",
            "kind": "comparison_seed_contract",
            "content_sha256": canonical_json_sha256(metadata_hashes),
            "seed_assignment_sha256": canonical_json_sha256(assignments),
            "stochastic_units": len(assignments),
        },
        len(assignments),
    )


def _audit_scalability_inference_seeds(
    root: Path,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    request_root = root / "experiments/v1/simulations/scale_stress"
    artifacts: list[dict[str, Any]] = []
    seed_count = 0
    for path in sorted(request_root.glob("*.json")):
        payload = _read_object(path, f"scalability seed contract {path.name}")
        assignments: list[dict[str, Any]] = []
        for dataset in payload.get("datasets", []):
            if not isinstance(dataset, Mapping) or not isinstance(dataset.get("id"), str):
                raise ReproducibilityAuditError(
                    f"Scalability request {path.name} contains an invalid dataset"
                )
            root_derivation = dataset.get("seed_derivation")
            if not isinstance(root_derivation, Mapping):
                raise ReproducibilityAuditError(
                    f"Scalability dataset {dataset['id']!r} has no seed derivation"
                )
            namespace = root_derivation.get("namespace")
            if not isinstance(namespace, str):
                raise ReproducibilityAuditError(
                    f"Scalability dataset {dataset['id']!r} has no seed namespace"
                )
            record = plan_seed_record(
                plan,
                namespace=namespace,
                stream="inference",
                coordinates={
                    "purpose": "scalability",
                    "dataset_id": dataset["id"],
                    "replicate_index": 1,
                },
            )
            assignments.append({"dataset_id": dataset["id"], **record})
            seed_count += 1
        artifacts.append(
            {
                "artifact_id": f"{payload.get('id', path.stem)}.inference-seeds",
                "kind": "scalability_inference_seed_contract",
                "content_sha256": canonical_json_sha256(payload),
                "seed_assignment_sha256": canonical_json_sha256(assignments),
                "stochastic_units": len(assignments),
            }
        )
    return artifacts, seed_count


def _audit_efficiency_study(
    root: Path,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], int, int, int]:
    path = root / "experiments/v1/spathi/efficiency-study.json"
    payload = _read_object(path, "efficiency study")
    budgets = payload.get("budgets")
    if payload.get("schema_version") != "1.0" or not isinstance(budgets, Mapping):
        raise ReproducibilityAuditError("Efficiency study must contain v1 budget definitions")
    assignments: list[dict[str, Any]] = []
    algorithm_seed_count = 0
    statistical_seed_count = 0
    execution_order_seed_count = 0
    for budget_id in ("reduced", "full"):
        budget = budgets.get(budget_id)
        if not isinstance(budget, Mapping):
            raise ReproducibilityAuditError(f"Efficiency study has no {budget_id!r} budget")
        seeds = budget.get("algorithm_seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ReproducibilityAuditError(
                f"Efficiency {budget_id} algorithm seeds must be a non-empty array"
            )
        namespace = f"{budget_id}.efficiency_validation"
        for index, value in enumerate(seeds, start=1):
            seed = _positive_seed(value, f"efficiency.{budget_id}.algorithm_seeds[{index - 1}]")
            coordinates = {
                "purpose": "algorithm_replicate",
                "replicate_index": index,
            }
            expected = derive_plan_seed(
                plan,
                namespace=namespace,
                stream="inference",
                coordinates=coordinates,
            )
            if seed != expected:
                raise ReproducibilityAuditError(
                    f"Efficiency {budget_id} algorithm seed {index} derives {expected}, "
                    f"but the committed value is {seed}"
                )
            assignments.append(
                {
                    "budget": budget_id,
                    "purpose": "algorithm_replicate",
                    "replicate_index": index,
                    **plan_seed_record(
                        plan,
                        namespace=namespace,
                        stream="inference",
                        coordinates=coordinates,
                    ),
                }
            )
            algorithm_seed_count += 1
        if budget.get("analysis_role") == "confirmatory":
            statistical_seed = _positive_seed(
                budget.get("statistical_seed"), f"efficiency.{budget_id}.statistical_seed"
            )
            statistical_coordinates = {"purpose": "noninferiority_analysis"}
            expected_statistical = derive_plan_seed(
                plan,
                namespace=namespace,
                stream="statistical",
                coordinates=statistical_coordinates,
            )
            if statistical_seed != expected_statistical:
                raise ReproducibilityAuditError(
                    f"Efficiency {budget_id} statistical seed derives {expected_statistical}, "
                    f"but the committed value is {statistical_seed}"
                )
            assignments.append(
                {
                    "budget": budget_id,
                    "purpose": "noninferiority_analysis",
                    **plan_seed_record(
                        plan,
                        namespace=namespace,
                        stream="statistical",
                        coordinates=statistical_coordinates,
                    ),
                }
            )
            statistical_seed_count += 1
        elif "statistical_seed" in budget:
            raise ReproducibilityAuditError(
                f"Efficiency {budget_id} descriptive budget reserves a statistical seed"
            )
        execution_order_seed = _positive_seed(
            budget.get("execution_order_seed"),
            f"efficiency.{budget_id}.execution_order_seed",
        )
        execution_coordinates = {"purpose": "counterbalanced_execution_order"}
        expected_execution_order = derive_plan_seed(
            plan,
            namespace=namespace,
            stream="execution_order",
            coordinates=execution_coordinates,
        )
        if execution_order_seed != expected_execution_order:
            raise ReproducibilityAuditError(
                f"Efficiency {budget_id} execution-order seed derives "
                f"{expected_execution_order}, but the committed value is "
                f"{execution_order_seed}"
            )
        assignments.append(
            {
                "budget": budget_id,
                "purpose": "counterbalanced_execution_order",
                **plan_seed_record(
                    plan,
                    namespace=namespace,
                    stream="execution_order",
                    coordinates=execution_coordinates,
                ),
            }
        )
        execution_order_seed_count += 1
    return (
        {
            "artifact_id": "spathi.efficiency-study",
            "kind": "efficiency_study",
            "content_sha256": canonical_json_sha256(payload),
            "seed_assignment_sha256": canonical_json_sha256(assignments),
            "stochastic_units": len(assignments),
        },
        algorithm_seed_count,
        statistical_seed_count,
        execution_order_seed_count,
    )


def _audit_efficiency_design_sensitivity(root: Path) -> dict[str, Any]:
    from .config import BenchmarkConfigError
    from .design_sensitivity import load_design_sensitivity_contract

    try:
        contract, operating_characteristics = load_design_sensitivity_contract(root)
    except BenchmarkConfigError as exc:
        raise ReproducibilityAuditError(
            f"Invalid prospective efficiency design sensitivity: {exc}"
        ) from exc
    return {
        "artifact_id": "spathi.efficiency-design-sensitivity",
        "kind": "efficiency_design_sensitivity_contract",
        "content_sha256": canonical_json_sha256(contract),
        "seed_assignment_sha256": operating_characteristics["operating_characteristics_sha256"],
        "stochastic_units": 0,
    }


def _audit_efficiency_safety_seed_contract(
    root: Path,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], int, int, int]:
    """Bind all pre-freeze safety conditions to disjoint data and paired execution seeds."""

    namespace = "full.efficiency_safety"

    algorithm = plan_seed_record(
        plan,
        namespace=namespace,
        stream="inference",
        coordinates={"purpose": "common_algorithm_seed"},
    )
    execution_order = plan_seed_record(
        plan,
        namespace=namespace,
        stream="execution_order",
        coordinates={"purpose": "counterbalanced_execution_order"},
    )
    json_members = (
        "experiments/v1/profiles/efficiency-safety-full.json",
        "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-tf100.json",
        "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-tf10.json",
        "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-tf10-single-group.json",
        "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-g3000-tf1500.json",
        "experiments/v1/simulations/scmultisim/efficiency_safety/scenario-g15000-tf100.json",
    )
    run_members = (
        "experiments/v1/simulations/scmultisim/efficiency_safety/g3000-tf100-c300-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_safety/g100-tf10-c500-single-group-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_safety/g100-tf10-c300-groups-10-40-250-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_safety/g100-tf10-c10000-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_safety/g3000-tf1500-c300-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_safety/g15000-tf100-c300-runs.json",
    )
    raw_members = (
        "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-tf100.tsv",
        "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-tf10.tsv",
        "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-g3000-tf1500.tsv",
        "experiments/v1/simulations/scmultisim/efficiency_safety/regulatory-network-g15000-tf100.tsv",
        "experiments/v1/simulations/scmultisim/efficiency_safety/tree-three-groups.nwk",
        "experiments/v1/simulations/scmultisim/efficiency_safety/tree-single-group.nwk",
    )
    design_inputs = {
        relative: canonical_json_sha256(
            _read_object(root / relative, f"efficiency-safety input {relative}")
        )
        for relative in (*json_members, *run_members)
    }
    design_inputs.update({relative: sha256_file(root / relative) for relative in raw_members})

    generation_blocks: list[dict[str, Any]] = []
    generation_seeds: set[int] = set()
    for relative in run_members:
        runs = _read_object(root / relative, f"efficiency-safety runs {relative}").get("runs")
        if not isinstance(runs, list) or len(runs) != 1:
            raise ReproducibilityAuditError(
                f"Efficiency-safety input {relative} must contain exactly one simulator run"
            )
        run = runs[0]
        if not isinstance(run, Mapping):
            raise ReproducibilityAuditError(
                f"Efficiency-safety input {relative} contains an invalid simulator run"
            )
        base_seed = _positive_seed(run.get("base_seed"), f"{relative}.base_seed")
        replicates = run.get("replicates")
        if replicates != 3:
            raise ReproducibilityAuditError(
                f"Efficiency-safety input {relative} must contain three replicates"
            )
        seeds = set(range(base_seed, base_seed + replicates))
        if generation_seeds & seeds:
            raise ReproducibilityAuditError("Efficiency-safety generation seeds are not unique")
        generation_seeds.update(seeds)
        generation_blocks.append(
            {
                "path": relative,
                "run_id": run.get("run_id"),
                "base_seed": base_seed,
                "replicates": replicates,
            }
        )
    if len(generation_seeds) != 18:
        raise ReproducibilityAuditError(
            "Efficiency-safety design must contain eighteen unique generation seeds"
        )
    assignments = {
        "generation_blocks": generation_blocks,
        "algorithm_seed": algorithm,
        "execution_order_seed": execution_order,
        "reuse_contract": "one_common_algorithm_seed_for_exact_and_candidate_on_all_datasets",
    }
    return (
        {
            "artifact_id": "spathi.efficiency-safety",
            "kind": "efficiency_safety_seed_contract",
            "content_sha256": canonical_json_sha256(design_inputs),
            "seed_assignment_sha256": canonical_json_sha256(assignments),
            "stochastic_units": 20,
        },
        len(generation_seeds),
        1,
        1,
    )


def audit_reproducibility(repo_root: Path) -> dict[str, Any]:
    """Validate every committed seed and return stable, hash-backed audit evidence."""

    root = repo_root.resolve()
    plan_path = root / _SEED_PLAN_MEMBER
    try:
        plan = load_seed_plan(plan_path)
    except (OSError, ValueError) as exc:
        raise ReproducibilityAuditError(f"Invalid canonical seed plan: {exc}") from exc

    try:
        simulator_artifacts, simulator_runs, simulator_replicates = _audit_simulator_runs(
            root, plan
        )
        inference_artifacts, stochastic_inference, deterministic_inference = (
            _audit_inference_metadata(root, plan)
        )
        generator_artifacts, generator_datasets, generator_component_seeds = (
            _audit_scale_stress_requests(root, plan)
        )
        calibration_artifact, calibration_seeds = _audit_calibration_seed_contract(root, plan)
        statistical_artifacts, statistical_root_seeds = _audit_statistical_seed_contracts(
            root, plan
        )
        comparison_artifact, comparison_seeds = _audit_comparison_seed_contract(root, plan)
        scalability_artifacts, scalability_seeds = _audit_scalability_inference_seeds(root, plan)
        dimensional_artifact, dimensional_seeds = _audit_dimensional_transfer_inference_seeds(
            root, plan
        )
        (
            efficiency_artifact,
            efficiency_algorithm_seeds,
            efficiency_statistical_seeds,
            efficiency_execution_order_seeds,
        ) = _audit_efficiency_study(root, plan)
        efficiency_sensitivity_artifact = _audit_efficiency_design_sensitivity(root)
        (
            efficiency_safety_artifact,
            efficiency_safety_generation_seeds,
            efficiency_safety_algorithm_seeds,
            efficiency_safety_execution_order_seeds,
        ) = _audit_efficiency_safety_seed_contract(root, plan)
    except (TypeError, SeedPlanError) as exc:
        raise ReproducibilityAuditError(f"Invalid seed assignment: {exc}") from exc
    artifacts = sorted(
        [
            *simulator_artifacts,
            *inference_artifacts,
            *generator_artifacts,
            *statistical_artifacts,
            *scalability_artifacts,
            calibration_artifact,
            comparison_artifact,
            efficiency_artifact,
            efficiency_sensitivity_artifact,
            efficiency_safety_artifact,
            dimensional_artifact,
        ],
        key=lambda item: (item["kind"], item["artifact_id"]),
    )
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "passed",
        "seed_contract": {
            "algorithm": plan["derivation"]["algorithm"],
            "domain_separator": plan["derivation"]["domain_separator"],
            "seed_plan_sha256": canonical_json_sha256(plan),
        },
        "coverage": {
            "simulator_manifests": len(simulator_artifacts),
            "simulator_runs": simulator_runs,
            "simulator_replicates": simulator_replicates,
            "inference_metadata_documents": len(inference_artifacts),
            "stochastic_inference_runs": stochastic_inference,
            "deterministic_inference_runs": deterministic_inference,
            "scale_stress_generator_requests": len(generator_artifacts),
            "scale_stress_generator_datasets": generator_datasets,
            "scale_stress_component_seeds": generator_component_seeds,
            "calibration_algorithm_seeds": calibration_seeds,
            "statistical_contract_documents": len(statistical_artifacts),
            "statistical_root_seeds": statistical_root_seeds,
            "comparison_algorithm_seeds": comparison_seeds,
            "efficiency_algorithm_seeds": efficiency_algorithm_seeds,
            "efficiency_statistical_root_seeds": efficiency_statistical_seeds,
            "efficiency_execution_order_seeds": efficiency_execution_order_seeds,
            "efficiency_design_sensitivity_contracts": 1,
            "efficiency_safety_generation_seeds": efficiency_safety_generation_seeds,
            "efficiency_safety_algorithm_seeds": efficiency_safety_algorithm_seeds,
            "efficiency_safety_execution_order_seeds": (efficiency_safety_execution_order_seeds),
            "scalability_algorithm_seeds": scalability_seeds,
            "dimensional_transfer_algorithm_seeds": dimensional_seeds,
        },
        "artifacts": artifacts,
    }
    result["audit_sha256"] = canonical_json_sha256(result)
    return result


def write_reproducibility_audit(repo_root: Path, output_path: Path) -> dict[str, Any]:
    """Audit the design and persist canonical evidence as stable pretty JSON."""

    result = audit_reproducibility(repo_root)
    validate_json(
        result,
        repo_root / "schemas/reproducibility-audit.schema.json",
        "reproducibility audit",
    )
    atomic_write_json(
        output_path,
        result,
        "reproducibility audit",
        boundary=output_path.parent,
    )
    return result
