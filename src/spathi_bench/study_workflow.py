"""Clean three-phase, crash-resumable article-study orchestration.

A study has one immutable scientific design, one frozen resource budget, one
SQLite journal and content-addressed artifacts. Git revisions are provenance;
reuse is determined by a task's scientific request and executor contract.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from .artifacts import canonical_json_sha256
from .catalog_identity import catalog_definition_identity
from .config import BenchmarkConfigError
from .design import (
    PHASE_NAMES,
    PROFILE_NAMES,
    PlannedDataset,
    ProfileView,
    StudyDesign,
    derive_seed,
    eligible_routes,
    load_design,
    profile_view,
    resolve_thread_points,
)
from .executor import (
    DATASET_MATERIALIZATION_CONTRACT,
    EXECUTOR_CONTRACT,
    INFERENCE_RESOURCE_CONTRACT,
    AndreaExecutor,
    effective_route_threads,
)
from .reporting.deliverables import build_study_deliverables
from .resource_plan import detect_host_capacity, resolve_runtime_plan
from .resource_scheduler import (
    ResourceCapacity,
    ResourceClaim,
    ResourceScheduleError,
    ScheduledTask,
    execute_resource_schedule,
    host_execution_lease,
)
from .revisions import observe_revisions
from .safe_io import atomic_write_json
from .study_store import StudyStore, TaskRecord

logger = logging.getLogger(__name__)
DESIGN_PATH = Path("experiments/article-study.json")
WORKFLOW_CONTRACT = "article-three-phase-workflow-v1"
_STUDY_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_GIB = 1024**3
_EXTERNAL_SIMULATOR_MEMORY_TIERS = tuple(value * _GIB for value in (8, 16, 32))
_PARALLEL_SIMULATOR_CPU_TIERS = (1, 2, 4, 8)
_SERIAL_SIMULATORS = frozenset({"boolode", "sergio"})


class TaskExecutor(Protocol):
    """The only execution interface understood by the orchestrator."""

    def execute_task(
        self, task: TaskRecord, dependencies: Mapping[str, Mapping[str, Any]]
    ) -> Mapping[str, Any]: ...


ExecutorFactory = Callable[..., TaskExecutor]


def build_executor(
    *, repo_root: Path, andrea_repo: Path, spathi_repo: Path, study_root: Path
) -> TaskExecutor:
    """Build the production adapter; tests inject a fake at this boundary."""

    executor = AndreaExecutor(
        repo_root=repo_root,
        andrea_repo=andrea_repo,
        spathi_repo=spathi_repo,
        study_root=study_root,
    )
    if not callable(getattr(executor, "execute_task", None)):
        raise BenchmarkConfigError(
            "The ANDREA article executor must implement execute_task(task, dependencies)"
        )
    return executor  # type: ignore[return-value]


def execute_task(
    executor: TaskExecutor,
    task: TaskRecord,
    dependencies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one request and admit only strict-JSON evidence to the journal."""

    result = executor.execute_task(task, dependencies)
    if not isinstance(result, Mapping):
        raise BenchmarkConfigError(f"Task {task.task_id!r} returned a non-object result")
    payload = dict(result)
    try:
        json.dumps(payload, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError(
            f"Task {task.task_id!r} returned non-canonical evidence: {exc}"
        ) from exc
    return payload


def _execute_wave_task(
    executor: TaskExecutor,
    dependency_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    task: TaskRecord,
) -> dict[str, Any]:
    return execute_task(executor, task, dependency_results[task.task_id])


def _study_root(work_root: Path, study_id: str) -> Path:
    if not _STUDY_ID.fullmatch(study_id):
        raise BenchmarkConfigError("Study ID must match [a-z0-9][a-z0-9._-]{0,127}")
    base = (work_root.resolve() / "studies").resolve()
    root = (base / study_id).resolve()
    if not root.is_relative_to(base):
        raise BenchmarkConfigError("Study path escapes the work directory")
    return root


def _dataset_payload(dataset: PlannedDataset) -> dict[str, Any]:
    # Newly declared materialization recipes automatically enter task identity.
    return _plain(asdict(dataset))


def _plain(value: Any) -> Any:
    """Convert dataclass containers to canonical-JSON container types."""

    if isinstance(value, Mapping):
        return {str(key): _plain(member) for key, member in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(member) for member in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(member) for member in value), key=str)
    return value


def _route_payload(route: Any) -> dict[str, Any]:
    return _plain(asdict(route))


def _task_id(phase: str, stage: str, identity: Mapping[str, Any]) -> str:
    suffix = canonical_json_sha256(identity)[:16]
    raw = re.sub(r"[^a-z0-9._-]+", "-", f"{phase}--{stage}".lower()).strip("-.")
    return f"{raw}--{suffix}"


def _claim_payload(
    capacity: Mapping[str, Any],
    *,
    cpu_threads: int,
    memory_bytes: int,
    exclusive: bool = False,
) -> dict[str, Any]:
    return {
        "cpu_threads": max(1, min(int(capacity["cpu_threads"]), cpu_threads)),
        "memory_bytes": max(1, min(int(capacity["memory_bytes"]), memory_bytes)),
        "exclusive": exclusive,
    }


def _data_bytes(dataset: PlannedDataset) -> int:
    # Declared dimensions are the canonical prospective size for every
    # generator. Eight bytes describe one dense float64 matrix and serve only
    # as a unit for admission estimates, not as a statement about file format.
    return max(1, dataset.dimensions.cells * dataset.dimensions.genes * 8)


def _largest_supported_tier(available: int, tiers: Sequence[int], resource: str) -> int:
    supported = [value for value in tiers if value <= available]
    if not supported:
        minimum = min(tiers)
        raise BenchmarkConfigError(
            f"External simulator generation requires at least {minimum} {resource}; "
            f"the frozen study capacity provides {available}"
        )
    return max(supported)


def _generation_claim(dataset: PlannedDataset, capacity: Mapping[str, Any]) -> dict[str, Any]:
    dense_bytes = _data_bytes(dataset)
    if dataset.generator == "scale-stress":
        # The in-repository generator writes its expression matrix through a
        # memmap and materializes targets in bounded row blocks.
        memory_bytes = dense_bytes * 12 + _GIB
        cpu_threads = min(4, int(capacity["cpu_threads"]))
    else:
        # These are the exact resource tiers measured by ANDREA's simulator
        # catalog. Use the largest admissible tier: the outer scheduler then
        # bounds aggregate R/Python working sets without inventing an
        # unvalidated 1--2 GiB budget for a large prospective dataset.
        memory_bytes = _largest_supported_tier(
            int(capacity["memory_bytes"]),
            _EXTERNAL_SIMULATOR_MEMORY_TIERS,
            "bytes of RAM",
        )
        cpu_threads = (
            1
            if dataset.generator in _SERIAL_SIMULATORS
            else _largest_supported_tier(
                int(capacity["cpu_threads"]),
                _PARALLEL_SIMULATOR_CPU_TIERS,
                "CPU thread(s)",
            )
        )
    return _claim_payload(
        capacity,
        cpu_threads=cpu_threads,
        memory_bytes=memory_bytes,
    )


def _analysis_claim(dataset: PlannedDataset, capacity: Mapping[str, Any]) -> dict[str, Any]:
    return _claim_payload(
        capacity,
        cpu_threads=int(capacity["cpu_threads"]),
        memory_bytes=int(capacity["memory_bytes"]),
    )


def _small_claim(capacity: Mapping[str, Any]) -> dict[str, Any]:
    return _claim_payload(capacity, cpu_threads=1, memory_bytes=_GIB)


def _isolated_claim(capacity: Mapping[str, Any]) -> dict[str, Any]:
    return _claim_payload(
        capacity,
        cpu_threads=int(capacity["cpu_threads"]),
        memory_bytes=int(capacity["memory_bytes"]),
        exclusive=True,
    )


def _task(
    *,
    phase: str,
    stage: str,
    kind: str,
    spec: Mapping[str, Any],
    dependencies: Sequence[str] = (),
) -> dict[str, Any]:
    identity = spec.get("identity")
    if not isinstance(identity, Mapping):
        raise BenchmarkConfigError("Every planned task requires an identity object")
    request = {
        "workflow_contract": WORKFLOW_CONTRACT,
        "phase": phase,
        "stage": stage,
        "kind": kind,
        "dependency_ids": list(dependencies),
        **dict(spec),
    }
    return {
        "task_id": _task_id(phase, stage, identity),
        "phase": phase,
        "stage": stage,
        "kind": kind,
        "scientific_digest": canonical_json_sha256(request),
        "executor_contract": (
            f"{EXECUTOR_CONTRACT}:{WORKFLOW_CONTRACT}:{kind}"
            + (f":{INFERENCE_RESOURCE_CONTRACT}" if kind == "comparison_dataset" else "")
        ),
        "spec": request,
        "dependencies": list(dependencies),
    }


def _generation_task(
    dataset: PlannedDataset,
    *,
    design: StudyDesign,
    capacity: Mapping[str, Any],
    dependencies: Sequence[str] = (),
) -> dict[str, Any]:
    return _task(
        phase=dataset.phase,
        stage="generate",
        kind="generate_dataset",
        spec={
            "identity": {"dataset_id": dataset.id},
            "design_digest": design.fingerprint,
            "materialization_contract": DATASET_MATERIALIZATION_CONTRACT,
            "dataset": _dataset_payload(dataset),
            "resources": _generation_claim(dataset, capacity),
        },
        dependencies=dependencies,
    )


def _algorithm_seeds(
    design: StudyDesign, *, dataset: PlannedDataset, stage: str, count: int
) -> list[int]:
    return [
        derive_seed(
            design,
            phase="calibration",
            stream="inference",
            coordinates={
                "dataset": dataset.id,
                "stage": stage,
                "algorithm_replicate": replicate,
            },
        )
        for replicate in range(1, count + 1)
    ]


def _calibration_tasks(
    design: StudyDesign, view: ProfileView, capacity: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    tasks: list[dict[str, Any]] = []
    screen_ids: list[str] = []
    budget = view.profile.calibration
    for dataset in view.datasets["calibration"]["development"]:
        generation = _generation_task(dataset, design=design, capacity=capacity)
        tasks.append(generation)
        screen = _task(
            phase="calibration",
            stage="screen",
            kind="calibration_screen",
            spec={
                "identity": {"dataset_id": dataset.id, "candidates": design.fingerprint},
                "design_digest": design.fingerprint,
                "dataset": _dataset_payload(dataset),
                "candidates": [_plain(asdict(candidate)) for candidate in view.candidates],
                "algorithm_seeds": _algorithm_seeds(
                    design,
                    dataset=dataset,
                    stage="screen",
                    count=budget.screening_algorithm_seeds,
                ),
                "n_estimators": view.screening_trees,
                "resources": _analysis_claim(dataset, capacity),
            },
            dependencies=[generation["task_id"]],
        )
        tasks.append(screen)
        screen_ids.append(screen["task_id"])

    finalists = _task(
        phase="calibration",
        stage="select-finalists",
        kind="select_calibration_finalists",
        spec={
            "identity": {"screens": sorted(screen_ids), "count": view.finalists},
            "design_digest": design.fingerprint,
            "candidate_ids": [candidate.id for candidate in view.candidates],
            "protocol": dict(design.raw["calibration"]["protocol"]["advancement"]),
            "resources": _small_claim(capacity),
        },
        dependencies=screen_ids,
    )
    tasks.append(finalists)

    validation_ids: list[str] = []
    for dataset in view.datasets["calibration"]["validation"]:
        generation = _generation_task(dataset, design=design, capacity=capacity)
        tasks.append(generation)
        validation = _task(
            phase="calibration",
            stage="validate",
            kind="calibration_validate",
            spec={
                "identity": {"dataset_id": dataset.id, "finalists": finalists["task_id"]},
                "design_digest": design.fingerprint,
                "dataset": _dataset_payload(dataset),
                "finalists_dependency": finalists["task_id"],
                "algorithm_seeds": _algorithm_seeds(
                    design,
                    dataset=dataset,
                    stage="validation",
                    count=budget.validation_algorithm_seeds,
                ),
                "fit_trees": view.validation_fit_trees,
                "exact_prefixes": list(view.validation_prefixes),
                "resources": _analysis_claim(dataset, capacity),
            },
            dependencies=[generation["task_id"], finalists["task_id"]],
        )
        tasks.append(validation)
        validation_ids.append(validation["task_id"])

    transfer_ids: list[str] = []
    for dataset in view.datasets["calibration"]["transfer"]:
        generation = _generation_task(
            dataset,
            design=design,
            capacity=capacity,
            dependencies=validation_ids,
        )
        tasks.append(generation)
        transfer = _task(
            phase="calibration",
            stage="transfer",
            kind="calibration_transfer",
            spec={
                "identity": {"dataset_id": dataset.id, "finalists": finalists["task_id"]},
                "design_digest": design.fingerprint,
                "dataset": _dataset_payload(dataset),
                "finalists_dependency": finalists["task_id"],
                "algorithm_seeds": _algorithm_seeds(
                    design,
                    dataset=dataset,
                    stage="transfer",
                    count=budget.transfer_algorithm_seeds,
                ),
                "fit_trees": view.validation_fit_trees,
                "exact_prefixes": list(view.validation_prefixes),
                "resources": _analysis_claim(dataset, capacity),
            },
            dependencies=[generation["task_id"], finalists["task_id"], *validation_ids],
        )
        tasks.append(transfer)
        transfer_ids.append(transfer["task_id"])

    winner = _task(
        phase="calibration",
        stage="select-winner",
        kind="select_calibration_winner",
        spec={
            "identity": {
                "validations": sorted(validation_ids),
                "transfers": sorted(transfer_ids),
            },
            "design_digest": design.fingerprint,
            "protocol": dict(design.raw["calibration"]["protocol"]["selection"]),
            "resources": _small_claim(capacity),
        },
        dependencies=[*validation_ids, *transfer_ids],
    )
    tasks.append(winner)
    return tasks, str(winner["task_id"])


def _comparison_tasks(
    design: StudyDesign,
    view: ProfileView,
    capacity: Mapping[str, Any],
    winner_id: str,
) -> tuple[list[dict[str, Any]], str]:
    tasks: list[dict[str, Any]] = []
    run_ids: list[str] = []
    for dataset in view.datasets["comparison"]["test"]:
        generation = _generation_task(
            dataset, design=design, capacity=capacity, dependencies=[winner_id]
        )
        tasks.append(generation)
        routes = eligible_routes(design, dataset.condition_id)
        run = _task(
            phase="comparison",
            stage="infer-and-evaluate",
            kind="comparison_dataset",
            spec={
                "identity": {
                    "dataset_id": dataset.id,
                    "route_ids": [route.id for route in routes],
                    "winner": winner_id,
                },
                "design_digest": design.fingerprint,
                "dataset": _dataset_payload(dataset),
                "routes": [_route_payload(route) for route in routes],
                "families": [_plain(asdict(family)) for family in view.families],
                "winner_dependency": winner_id,
                "timeout_seconds": view.profile.comparison.task_timeout_seconds,
                "resources": _analysis_claim(dataset, capacity),
            },
            dependencies=[generation["task_id"], winner_id],
        )
        tasks.append(run)
        run_ids.append(str(run["task_id"]))

    statistics = _task(
        phase="comparison",
        stage="statistics",
        kind="comparison_statistics",
        spec={
            "identity": {"datasets": sorted(run_ids)},
            "design_digest": design.fingerprint,
            "families": [_plain(asdict(family)) for family in view.families],
            "statistics": dict(design.raw["comparison"]["statistics"]),
            "resources": _small_claim(capacity),
        },
        dependencies=run_ids,
    )
    tasks.append(statistics)
    return tasks, str(statistics["task_id"])


def _scalability_tasks(
    design: StudyDesign,
    view: ProfileView,
    capacity: Mapping[str, Any],
    winner_id: str,
    comparison_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Plan isolated routes breadth-first over the common size ladder."""

    tasks: list[dict[str, Any]] = []
    generated: dict[str, str] = {}
    ladder_datasets = tuple(view.datasets["scalability"]["ladder"])
    ladder_by_coordinate = {
        (dataset.condition_id, dataset.replicate): dataset for dataset in ladder_datasets
    }
    thread_datasets: list[PlannedDataset] = []
    for requested in view.datasets["scalability"]["thread_scaling"]:
        try:
            thread_datasets.append(
                ladder_by_coordinate[(requested.condition_id, requested.replicate)]
            )
        except KeyError as exc:
            raise BenchmarkConfigError(
                "Every thread-scaling replicate must reuse the matching ladder dataset"
            ) from exc
    unique_datasets = {dataset.id: dataset for dataset in ladder_datasets}
    for dataset in unique_datasets.values():
        generation = _generation_task(
            dataset,
            design=design,
            capacity=capacity,
            dependencies=[comparison_id],
        )
        tasks.append(generation)
        generated[dataset.id] = str(generation["task_id"])

    budget = view.profile.scalability
    scale_order = {scale_id: index for index, scale_id in enumerate(budget.scale_ids, 1)}
    # A route is stopped globally after its first non-completed scale outcome.
    # Chaining across both replicates and scales prevents repeated predictable
    # timeouts while leaving every other route independent.
    prior: dict[str, str] = {}
    ladder_point: dict[tuple[str, int, str], str] = {}
    measurements: list[str] = []
    thread_resolution: list[dict[str, Any]] = []
    standard_threads = min(8, int(capacity["cpu_threads"]))

    # Stages carry a zero-padded scale ordinal. Each scheduling wave therefore
    # contains only one size. Every task is exclusive, so timings are isolated;
    # routes at the next size become ready only after the whole current wave was
    # durably journalled.
    for dataset in ladder_datasets:
        ordinal = scale_order[dataset.condition_id]
        for route in eligible_routes(design, dataset.condition_id):
            previous = prior.get(route.id)
            threads = effective_route_threads(standard_threads, route.max_threads)
            dependencies = [generated[dataset.id], winner_id]
            if previous is not None:
                dependencies.append(previous)
            run = _task(
                phase="scalability",
                stage=f"ladder-{ordinal:02d}",
                kind="scalability_measurement",
                spec={
                    "identity": {
                        "dataset_id": dataset.id,
                        "route_id": route.id,
                        "effective_threads": threads,
                        "measurement": "size_ladder",
                    },
                    "design_digest": design.fingerprint,
                    "dataset": _dataset_payload(dataset),
                    "route": _route_payload(route),
                    "winner_dependency": winner_id,
                    "advance_dependency": previous,
                    "measurement": "size_ladder",
                    "requested_threads": standard_threads,
                    "effective_threads": threads,
                    "timeout_seconds": budget.task_timeout_seconds,
                    "desist_policy": {
                        **dict(design.raw["scalability"]["execution"]),
                        "allowed_reasons": list(design.raw["scalability"]["desist_reasons"]),
                    },
                    "resources": _isolated_claim(capacity),
                },
                dependencies=dependencies,
            )
            tasks.append(run)
            task_id = str(run["task_id"])
            prior[route.id] = task_id
            ladder_point[(route.id, dataset.replicate, dataset.condition_id)] = task_id
            measurements.append(task_id)

    points = resolve_thread_points(budget.thread_points, int(capacity["cpu_threads"]))
    for dataset in thread_datasets:
        for route in eligible_routes(design, dataset.condition_id):
            ladder_id = ladder_point.get((route.id, dataset.replicate, dataset.condition_id))
            effective_points: set[int] = set()
            for requested_threads in points:
                threads = effective_route_threads(requested_threads, route.max_threads)
                if threads in effective_points:
                    thread_resolution.append(
                        {
                            "decision": "not_distinct",
                            "dataset_id": dataset.id,
                            "method_id": route.id,
                            "requested_threads": requested_threads,
                            "effective_threads": threads,
                        }
                    )
                    continue
                effective_points.add(threads)
                standard_effective = effective_route_threads(standard_threads, route.max_threads)
                if threads == standard_effective and ladder_id is not None:
                    # Reuse the already isolated P-M standard-thread result.
                    thread_resolution.append(
                        {
                            "decision": "reuse_size_ladder",
                            "dataset_id": dataset.id,
                            "method_id": route.id,
                            "requested_threads": requested_threads,
                            "effective_threads": threads,
                        }
                    )
                    continue
                dependencies = [generated[dataset.id], winner_id]
                if ladder_id is not None:
                    dependencies.append(ladder_id)
                run = _task(
                    phase="scalability",
                    stage="thread-scaling",
                    kind="scalability_measurement",
                    spec={
                        "identity": {
                            "dataset_id": dataset.id,
                            "route_id": route.id,
                            "effective_threads": threads,
                            "measurement": "thread_scaling",
                        },
                        "design_digest": design.fingerprint,
                        "dataset": _dataset_payload(dataset),
                        "route": _route_payload(route),
                        "winner_dependency": winner_id,
                        "advance_dependency": ladder_id,
                        "measurement": "thread_scaling",
                        "requested_threads": requested_threads,
                        "effective_threads": threads,
                        "timeout_seconds": budget.task_timeout_seconds,
                        "desist_policy": {
                            **dict(design.raw["scalability"]["execution"]),
                            "allowed_reasons": list(design.raw["scalability"]["desist_reasons"]),
                        },
                        "resources": _isolated_claim(capacity),
                    },
                    dependencies=dependencies,
                )
                tasks.append(run)
                measurements.append(str(run["task_id"]))
                thread_resolution.append(
                    {
                        "decision": "scheduled_measurement",
                        "dataset_id": dataset.id,
                        "method_id": route.id,
                        "requested_threads": requested_threads,
                        "effective_threads": threads,
                    }
                )

    summary = _task(
        phase="scalability",
        stage="summarize",
        kind="scalability_summary",
        spec={
            "identity": {"measurements": sorted(measurements)},
            "design_digest": design.fingerprint,
            "resource_capacity": dict(capacity),
            "thread_resolution": thread_resolution,
            "resources": _small_claim(capacity),
        },
        dependencies=measurements,
    )
    tasks.append(summary)
    return tasks, str(summary["task_id"])


def build_task_plan(
    design: StudyDesign,
    profile: str,
    resource_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand one profile into a deterministic three-phase DAG."""

    view = profile_view(design, profile)
    capacity = resource_plan["capacity"]
    calibration, winner = _calibration_tasks(design, view, capacity)
    comparison, comparison_receipt = _comparison_tasks(design, view, capacity, winner)
    scalability, _ = _scalability_tasks(
        design,
        view,
        capacity,
        winner,
        comparison_receipt,
    )
    tasks = [*calibration, *comparison, *scalability]
    identifiers = [str(task["task_id"]) for task in tasks]
    duplicates = sorted(name for name, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise BenchmarkConfigError(f"Task planner produced duplicate IDs: {duplicates}")
    return tasks


def bind_catalog_definitions(
    tasks: Sequence[Mapping[str, Any]], andrea_repo: Path
) -> list[dict[str, Any]]:
    """Bind tasks only to the ANDREA catalog definitions they consume.

    IDs and DAG topology remain stable. A ToolSpec or SimulatorSpec change
    changes the task digest, so the journal invalidates that task and all of
    its descendants while unrelated results survive a cross-commit resume.
    """

    cache: dict[tuple[str, str], dict[str, str]] = {}

    def resolve(kind: str, identifier: str) -> dict[str, str]:
        coordinate = (kind, identifier)
        if coordinate not in cache:
            cache[coordinate] = catalog_definition_identity(
                andrea_repo, catalog_kind=kind, catalog_id=identifier
            )
        return dict(cache[coordinate])

    bound: list[dict[str, Any]] = []
    for raw_task in tasks:
        task = dict(raw_task)
        spec = dict(task["spec"])
        definitions: dict[str, dict[str, str]] = {}
        dataset = spec.get("dataset")
        if task["kind"] == "generate_dataset" and isinstance(dataset, Mapping):
            generator = dataset.get("generator")
            if isinstance(generator, str) and generator != "scale-stress":
                definitions[f"simulator:{generator}"] = resolve("simulator", generator)
        routes = spec.get("routes")
        if isinstance(routes, list):
            for route in routes:
                if not isinstance(route, Mapping):
                    raise BenchmarkConfigError("Planned route must be an object")
                tool_id = route.get("tool_id")
                if isinstance(tool_id, str) and tool_id != "spathi":
                    definitions[f"tool:{tool_id}"] = resolve("tool", tool_id)
        route = spec.get("route")
        if isinstance(route, Mapping):
            tool_id = route.get("tool_id")
            if isinstance(tool_id, str) and tool_id != "spathi":
                definitions[f"tool:{tool_id}"] = resolve("tool", tool_id)
        spec["catalog_definitions"] = dict(sorted(definitions.items()))
        task["spec"] = spec
        task["scientific_digest"] = canonical_json_sha256(spec)
        bound.append(task)
    return bound


def _plan_topology_digest(tasks: Sequence[Mapping[str, Any]]) -> str:
    """Identify the immutable DAG without coupling reuse to Git revisions.

    Scientific request and executor-contract changes keep the same topology and
    are handled by ``StudyStore.add_tasks``, which invalidates only affected
    tasks and descendants. Adding, removing or rewiring tasks requires a new
    study ID so an old journal can never mix orphaned and current work.
    """

    topology = [
        {
            "task_id": str(task["task_id"]),
            "phase": str(task["phase"]),
            "stage": str(task["stage"]),
            "kind": str(task["kind"]),
            "dependencies": sorted(str(value) for value in task["dependencies"]),
        }
        for task in tasks
    ]
    return canonical_json_sha256(sorted(topology, key=lambda value: value["task_id"]))


def _claim(task: TaskRecord) -> ResourceClaim:
    raw = task.spec.get("resources")
    if not isinstance(raw, Mapping):
        raise BenchmarkConfigError(f"Task {task.task_id!r} has no resource claim")
    return ResourceClaim(
        cpu_threads=int(raw["cpu_threads"]),
        memory_bytes=int(raw["memory_bytes"]),
        exclusive=bool(raw.get("exclusive", False)),
    )


def _status_from_store(store: StudyStore) -> tuple[str, str | None, dict[str, str]]:
    phase_status: dict[str, str] = {}
    current: str | None = None
    for phase in PHASE_NAMES:
        tasks = store.tasks(phase=phase)
        statuses = {task.status for task in tasks}
        if tasks and statuses == {"complete"}:
            phase_status[phase] = "complete"
        elif "running" in statuses:
            phase_status[phase] = "running"
        else:
            phase_status[phase] = "pending"
        if current is None and phase_status[phase] != "complete":
            current = phase
    return ("complete" if current is None else phase_status[current]), current, phase_status


def _phase_runtime_provenance(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Extract portable runtime identities from nested stage summaries."""

    images: dict[str, dict[str, Any]] = {}
    generators: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            requested = value.get("requested_image")
            image_id = value.get("image_id")
            if isinstance(requested, str) and isinstance(image_id, str):
                observed = dict(value)
                # The same image may be governed by multiple ToolSpecs, and a
                # resumed study may legitimately contain attempts made with
                # different definitions. Key by the complete observed
                # identity so none is silently hidden by the image tag.
                images.setdefault(canonical_json_sha256(observed), observed)
            generator_id = value.get("generator_id")
            generator_contract = value.get("generator_contract")
            if isinstance(generator_id, str) and isinstance(generator_contract, str):
                generators.setdefault(f"{generator_id}:{generator_contract}", dict(value))
            for member in value.values():
                visit(member)
        elif isinstance(value, (list, tuple)):
            for member in value:
                visit(member)

    visit(summaries)
    return {
        "image_identities": [images[key] for key in sorted(images)],
        "host_generator_identities": [generators[key] for key in sorted(generators)],
    }


def _report_snapshot(store: StudyStore) -> dict[str, Any]:
    metadata = store.metadata()
    status, _current, phase_status = _status_from_store(store)
    journal = store.snapshot()
    attempts_by_task: dict[str, list[dict[str, Any]]] = {}
    for attempt in journal["task_attempts"]:
        task_id = str(attempt["task_id"])
        result = attempt.get("result")
        result_summary = result.get("summary") if isinstance(result, Mapping) else None
        runtime_provenance = _phase_runtime_provenance(
            [dict(result_summary)] if isinstance(result_summary, Mapping) else []
        )
        attempts_by_task.setdefault(task_id, []).append(
            {
                "attempt": int(attempt["attempt"]),
                "revision_sequence": int(attempt["revision_sequence"]),
                "scientific_digest": str(attempt["scientific_digest"]),
                "executor_contract": str(attempt["executor_contract"]),
                "status": str(attempt["status"]),
                "started_at": str(attempt["started_at"]),
                "finished_at": attempt.get("finished_at"),
                "result_sha256": (
                    canonical_json_sha256(result) if isinstance(result, Mapping) else None
                ),
                "error": attempt.get("error"),
                "runtime_provenance": runtime_provenance,
            }
        )
    phases: dict[str, Any] = {}
    for phase in PHASE_NAMES:
        tasks = store.tasks(phase=phase)
        rows: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        execution_coverage: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        datasets: dict[str, dict[str, Any]] = {}
        for task in tasks:
            result = task.result or {}
            dataset = task.spec.get("dataset")
            if isinstance(dataset, Mapping):
                dataset_id = str(dataset.get("id", task.task_id))
                reported = dict(dataset)
                record = result.get("record")
                observed = (
                    record.get("observed_dimensions") if isinstance(record, Mapping) else None
                )
                if isinstance(observed, Mapping):
                    materialization = record.get("materialization", {})
                    group_sizes = (
                        materialization.get("group_sizes")
                        if isinstance(materialization, Mapping)
                        else None
                    )
                    reported.update(
                        dimensions=dict(observed),
                        observed_dimensions=dict(observed),
                        requested_dimensions=dict(record.get("requested_dimensions", {})),
                        dimension_deviations=dict(record.get("dimension_deviations", {})),
                        materialization_policy=record.get("materialization_policy"),
                        dimension_policy=dict(record.get("dimension_policy", {})),
                        materialization_status=record.get("materialization_status"),
                        tf_count=observed.get("transcription_factors"),
                        group_sizes=(
                            dict(group_sizes) if isinstance(group_sizes, Mapping) else None
                        ),
                        **dict(observed),
                    )
                existing = datasets.get(dataset_id)
                if existing is None or "observed_dimensions" in reported:
                    datasets[dataset_id] = reported
            for key, target in (
                ("rows", rows),
                ("decisions", decisions),
                ("failures", failures),
                ("execution_coverage", execution_coverage),
            ):
                values = result.get(key, [])
                if isinstance(values, list):
                    target.extend(dict(value) for value in values if isinstance(value, Mapping))
            if isinstance(result.get("decision"), Mapping):
                decisions.append(dict(result["decision"]))
            if isinstance(result.get("summary"), Mapping):
                summaries.append(dict(result["summary"]))
            if task.error:
                failures.append(
                    {"task_id": task.task_id, "status": task.status, "reason": task.error}
                )
        phase_attempt_runtime = [
            attempt["runtime_provenance"]
            for task in tasks
            for attempt in attempts_by_task.get(task.task_id, [])
        ]
        phases[phase] = {
            "summary": {
                "status": phase_status[phase],
                "tasks": len(tasks),
                "tasks_complete": sum(task.status == "complete" for task in tasks),
                "datasets": list(datasets.values()),
                "stage_summaries": summaries,
                "execution_coverage": execution_coverage,
            },
            "rows": rows,
            "decisions": decisions,
            "failures": failures,
            "provenance": {
                "design_digest": metadata["design_digest"],
                **_phase_runtime_provenance(summaries),
                "all_attempt_runtime_identities": _phase_runtime_provenance(phase_attempt_runtime),
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "scientific_digest": task.scientific_digest,
                        "executor_contract": task.executor_contract,
                        "status": task.status,
                        "attempt": task.attempt,
                        "catalog_definitions": dict(task.spec.get("catalog_definitions", {})),
                        "attempts": attempts_by_task.get(task.task_id, []),
                    }
                    for task in tasks
                ],
                "revisions": journal["revisions"],
            },
        }
    return {
        "study": {
            "id": metadata["study_id"],
            "profile": metadata["profile"],
            "status": status,
            "resources": metadata["resource_plan"]["capacity"],
        },
        "phases": phases,
    }


def _summary(
    store: StudyStore,
    study_root: Path,
    report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = store.metadata()
    status, current, phases = _status_from_store(store)
    result: dict[str, Any] = {
        "study_id": metadata["study_id"],
        "profile": metadata["profile"],
        "status": status,
        "current_phase": current,
        "phases": phases,
        "task_counts": store.snapshot()["task_counts"],
        "study_root": str(study_root),
        "resume_command": f"spathi-bench run --id {metadata['study_id']}",
    }
    if report is not None:
        result["report"] = report.get("directory")
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkConfigError(f"Cannot read study manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkConfigError("Study manifest must be an object")
    return value


def _runtime_plan(
    *, work_root: Path, max_cpus: int | None, max_memory_gb: float | None
) -> dict[str, Any]:
    return resolve_runtime_plan(
        detect_host_capacity(workspace_path=work_root),
        max_cores=max_cpus,
        max_ram_gb=max_memory_gb,
    )


def run_study_workflow(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    profile: str | None,
    study_id: str,
    max_cpus: int | None = None,
    max_memory_gb: float | None = None,
    executor_factory: ExecutorFactory | None = None,
) -> dict[str, Any]:
    """Create or resume a study, preserving scientifically identical tasks."""

    root = _study_root(work_root, study_id)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    design = load_design(repo_root / DESIGN_PATH)

    new_study = not manifest_path.is_file()
    if not new_study:
        manifest = _load_manifest(manifest_path)
        saved_profile = manifest.get("profile")
        if profile is not None and profile != saved_profile:
            raise BenchmarkConfigError("An existing study ID cannot change profile")
        profile = str(saved_profile)
        if max_cpus is not None or max_memory_gb is not None:
            raise BenchmarkConfigError(
                "Resource limits are frozen at study creation; omit them when resuming"
            )
        raw_plan = manifest.get("resource_plan")
        if not isinstance(raw_plan, Mapping):
            raise BenchmarkConfigError("Study manifest has no resource plan")
        resource_plan = dict(raw_plan)
    else:
        if profile not in PROFILE_NAMES:
            raise BenchmarkConfigError("A new study requires --profile smoke, reduced or full")
        resource_plan = _runtime_plan(
            work_root=work_root,
            max_cpus=max_cpus,
            max_memory_gb=max_memory_gb,
        )
        manifest = {}

    if profile not in PROFILE_NAMES:
        raise BenchmarkConfigError("Study manifest contains an invalid profile")
    if not new_study and manifest.get("design_digest") != design.fingerprint:
        raise BenchmarkConfigError(
            "The scientific design changed; start a new study ID. Compatible "
            "implementation commits may resume the existing ID."
        )
    raw_capacity = resource_plan.get("capacity")
    if not isinstance(raw_capacity, Mapping):
        raise BenchmarkConfigError("Study resource plan has no capacity")
    capacity = ResourceCapacity(
        cpu_threads=int(raw_capacity["cpu_threads"]),
        memory_bytes=int(raw_capacity["memory_bytes"]),
    )

    revision = observe_revisions(repo_root, spathi_repo, andrea_repo)
    tasks = bind_catalog_definitions(build_task_plan(design, profile, resource_plan), andrea_repo)
    plan_topology_digest = _plan_topology_digest(tasks)
    if new_study:
        manifest = {
            "schema_version": 1,
            "study_id": study_id,
            "profile": profile,
            "design_id": design.id,
            "design_digest": design.fingerprint,
            "plan_topology_digest": plan_topology_digest,
            "resource_plan": resource_plan,
        }
        atomic_write_json(manifest_path, manifest, "Study manifest", boundary=root)
    elif manifest.get("plan_topology_digest") != plan_topology_digest:
        raise BenchmarkConfigError(
            "The study task topology changed; start a new study ID. The existing "
            "journal is left untouched so tasks from different DAGs cannot mix."
        )
    factory = executor_factory or build_executor
    with host_execution_lease(workload=f"study:{study_id}"):
        with StudyStore(root / "study.sqlite") as store:
            store.initialize(
                {
                    "study_id": study_id,
                    "profile": profile,
                    "design_digest": design.fingerprint,
                    "plan_topology_digest": plan_topology_digest,
                    "scientific_design": {
                        "id": design.id,
                        "phases": list(PHASE_NAMES),
                        "profile": profile,
                    },
                    "resource_plan": resource_plan,
                }
            )
            changed, previous = store.record_revision(revision)
            if changed:
                store.event(
                    "implementation_revision_changed",
                    level="warning",
                    details={"previous": previous, "current": revision},
                )
                logger.warning("Implementation revision changed; compatible results are preserved")
            recovered = store.recover_interrupted()
            planning = store.add_tasks(tasks)
            store.event(
                "study_resumed",
                details={"planning": planning, "recovered_running_tasks": recovered},
            )
            executor: TaskExecutor | None = None

            while True:
                ready = store.ready_tasks()
                if not ready:
                    incomplete = [task for task in store.tasks() if task.status != "complete"]
                    if incomplete:
                        examples = ", ".join(task.task_id for task in incomplete[:5])
                        raise BenchmarkConfigError(
                            f"Study DAG is blocked with {len(incomplete)} incomplete tasks: "
                            f"{examples}"
                        )
                    break

                if executor is None:
                    executor = factory(
                        repo_root=repo_root,
                        andrea_repo=andrea_repo,
                        spathi_repo=spathi_repo,
                        study_root=root,
                    )

                dependency_results = {
                    task.task_id: {
                        dependency: store.task(dependency).result or {}
                        for dependency in task.dependencies
                    }
                    for task in ready
                }
                scheduled = [
                    ScheduledTask(task_id=task.task_id, payload=task, claim=_claim(task))
                    for task in ready
                ]
                for task in ready:
                    store.start(task.task_id)

                try:
                    assert executor is not None
                    outcome = execute_resource_schedule(
                        scheduled,
                        capacity=capacity,
                        worker=partial(_execute_wave_task, executor, dependency_results),
                        thread_name_prefix="spathi-study",
                    )
                except ResourceScheduleError as exc:
                    store.event(
                        "task_wave_failed",
                        level="error",
                        task_id=exc.task_id,
                        details={"schedule": exc.manifest},
                    )
                    # Workers publish atomically. Reset the wave so a restart
                    # discovers completed artifacts by content address and never
                    # records a partial batch of worker results in SQLite.
                    store.recover_interrupted()
                    cause = exc.__cause__ or exc
                    raise RuntimeError(
                        f"Task {exc.task_id!r} failed; resume with "
                        f"spathi-bench run --id {study_id}: {cause}"
                    ) from cause
                for execution in outcome.tasks:
                    store.finish(execution.task_id, execution.result)
                store.event(
                    "task_wave_completed",
                    details={
                        "schedule": outcome.manifest(),
                        "utilization": outcome.utilization(capacity),
                    },
                )

            with store.read_transaction():
                report_snapshot = _report_snapshot(store)
            report = build_study_deliverables(report_snapshot, repo_root / "deliverables")
            store.set_metadata("deliverables", report)
            return _summary(store, root, report)


def study_status(*, work_root: Path, study_id: str | None = None) -> dict[str, Any]:
    """Inspect one journal or list all named studies without modifying them."""

    if study_id is not None:
        root = _study_root(work_root, study_id)
        if not (root / "study.sqlite").is_file():
            raise BenchmarkConfigError(f"Unknown study ID: {study_id}")
        with StudyStore(root / "study.sqlite") as store:
            return _summary(store, root, store.metadata().get("deliverables"))

    studies: list[dict[str, Any]] = []
    studies_root = work_root.resolve() / "studies"
    if studies_root.is_dir():
        for database in sorted(studies_root.glob("*/study.sqlite")):
            with StudyStore(database) as store:
                studies.append(
                    _summary(
                        store,
                        database.parent,
                        store.metadata().get("deliverables"),
                    )
                )
    return {"studies": studies}


def rebuild_study_reports(*, repo_root: Path, work_root: Path, study_id: str) -> dict[str, Any]:
    """Rebuild disposable report views without altering scientific tasks."""

    root = _study_root(work_root, study_id)
    if not (root / "study.sqlite").is_file():
        raise BenchmarkConfigError(f"Unknown study ID: {study_id}")
    with StudyStore(root / "study.sqlite") as store:
        # A report may deliberately be rebuilt for an interrupted study.  Read
        # it from one point-in-time snapshot so a concurrently finishing task
        # cannot leave counts, rows and provenance from different DB states.
        with store.read_transaction():
            report_snapshot = _report_snapshot(store)
        report = build_study_deliverables(report_snapshot, repo_root / "deliverables")
        store.set_metadata("deliverables", report)
        return _summary(store, root, report)


def plan_study(
    *,
    repo_root: Path,
    work_root: Path,
    profile: str,
    andrea_repo: Path | None = None,
    max_cpus: int | None = None,
    max_memory_gb: float | None = None,
) -> dict[str, Any]:
    """Return the exact prospective DAG and host-resolved resource budget."""

    if profile not in PROFILE_NAMES:
        raise BenchmarkConfigError("Profile must be smoke, reduced or full")
    design = load_design(repo_root / DESIGN_PATH)
    resource_plan = _runtime_plan(
        work_root=work_root,
        max_cpus=max_cpus,
        max_memory_gb=max_memory_gb,
    )
    tasks = bind_catalog_definitions(
        build_task_plan(design, profile, resource_plan),
        andrea_repo or repo_root.parent / "ANDREA",
    )
    plan_topology_digest = _plan_topology_digest(tasks)
    phases = {
        phase: {
            "task_count": sum(task["phase"] == phase for task in tasks),
            "stages": dict(
                sorted(
                    Counter(str(task["stage"]) for task in tasks if task["phase"] == phase).items()
                )
            ),
        }
        for phase in PHASE_NAMES
    }
    return {
        "status": "planned",
        "profile": profile,
        "design_id": design.id,
        "design_digest": design.fingerprint,
        "plan_topology_digest": plan_topology_digest,
        "resource_plan": resource_plan,
        "phases": phases,
        "tasks": tasks,
        "warnings": (
            ["Smoke and reduced are engineering profiles, not publication evidence."]
            if profile != "full"
            else []
        ),
    }


__all__ = [
    "DESIGN_PATH",
    "WORKFLOW_CONTRACT",
    "TaskExecutor",
    "build_executor",
    "build_task_plan",
    "bind_catalog_definitions",
    "execute_task",
    "plan_study",
    "rebuild_study_reports",
    "run_study_workflow",
    "study_status",
]
