"""Frozen design and materialization for the formal computational-performance study."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .benchmark_materialization import build_selected_spathi_requests
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    validate_json,
    validate_profile_references,
)
from .execution_policy import resolve_execution_class
from .safe_io import path_below_no_symlinks, write_idempotent_json
from .seeds import derive_plan_seed, load_seed_plan

DESIGN_PATH = Path("experiments/v1/performance/design.json")

_EXPECTED_BUDGETS = {
    "reduced": {
        "profile_id": "performance-reduced",
        "catalog_request": "experiments/v1/inference/group_only/reduced-primary.json",
        "catalog_metadata": "experiments/v1/inference/group_only/reduced-primary-metadata.json",
        "datasets": (("scmultisim_performance_medium", 1, "medium"),),
        "tools": (
            ("spathi_selected", "spathi", "spathi_selected_policy"),
            ("genie3_default", "genie3", "catalog"),
            ("grnboost2_default", "grnboost2", "catalog"),
            ("infercsn_none", "infercsn", "catalog"),
        ),
        "threads": (1, 8),
        "repetitions": 1,
        "measured": 8,
        "warmups": 8,
    },
    "full": {
        "profile_id": "performance-full",
        "catalog_request": "experiments/v1/inference/group_only/full-primary.json",
        "catalog_metadata": "experiments/v1/inference/group_only/full-primary-metadata.json",
        "datasets": (
            ("scmultisim_performance_small", 1, "small"),
            ("scmultisim_performance_medium", 1, "medium"),
        ),
        "tools": (
            ("spathi_selected", "spathi", "spathi_selected_policy"),
            ("aracne3_default", "aracne3", "catalog"),
            ("genie3_default", "genie3", "catalog"),
            ("grnboost2_default", "grnboost2", "catalog"),
            ("infercsn_none", "infercsn", "catalog"),
            ("tigress_default", "tigress", "catalog"),
        ),
        "threads": (1, 2, 4, 8),
        "repetitions": 3,
        "measured": 144,
        "warmups": 48,
    },
}


def _relative(repo_root: Path, path: Path, label: str) -> str:
    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    safe = path_below_no_symlinks(path, root, label)
    return safe.relative_to(root).as_posix()


def load_performance_design(repo_root: Path) -> dict[str, Any]:
    """Load the sole v1 performance design and enforce its publication contract."""

    root = repo_root.resolve()
    path = root / DESIGN_PATH
    design = load_json_object(path, "performance design")
    validate_json(
        design,
        root / "schemas/performance-design.schema.json",
        "performance design",
    )
    for budget, expected in _EXPECTED_BUDGETS.items():
        declared = design["budgets"][budget]
        datasets = tuple(
            (item["generation_job_id"], item["replicate_index"], item["size_class"])
            for item in declared["datasets"]
        )
        tools = tuple(
            (item["configuration_id"], item["tool_id"], item["origin"])
            for item in declared["tools"]
        )
        if (
            declared["profile_id"] != expected["profile_id"]
            or declared["catalog_request"] != expected["catalog_request"]
            or declared["catalog_metadata"] != expected["catalog_metadata"]
            or datasets != expected["datasets"]
            or tools != expected["tools"]
            or tuple(declared["thread_counts"]) != expected["threads"]
            or declared["system_repetitions"] != expected["repetitions"]
            or declared["warmups_per_condition"] != 1
            or declared["algorithm_seed_namespace"] != f"{budget}.performance"
            or declared["expected_measured_runs"] != expected["measured"]
            or declared["expected_warmup_runs"] != expected["warmups"]
        ):
            raise BenchmarkConfigError(f"{budget} performance budget differs from protocol v1")
        calculated_measured = (
            len(declared["datasets"])
            * len(declared["tools"])
            * len(declared["thread_counts"])
            * int(declared["system_repetitions"])
        )
        calculated_warmups = (
            len(declared["datasets"])
            * len(declared["tools"])
            * len(declared["thread_counts"])
            * int(declared["warmups_per_condition"])
        )
        if (
            calculated_measured != declared["expected_measured_runs"]
            or calculated_warmups != declared["expected_warmup_runs"]
        ):
            raise BenchmarkConfigError(f"{budget} performance run counts are inconsistent")
        if len({item["generation_job_id"] for item in declared["datasets"]}) != len(
            declared["datasets"]
        ):
            raise BenchmarkConfigError(f"{budget} performance datasets are not unique")
        if len({item["configuration_id"] for item in declared["tools"]}) != len(declared["tools"]):
            raise BenchmarkConfigError(f"{budget} performance tools are not unique")
    return design


def validate_performance_catalog_threading(andrea_repo: Path, design: Mapping[str, Any]) -> None:
    """Require every primary catalog method to support every declared endpoint."""

    catalog_requirements: dict[str, int] = {}
    for budget in design["budgets"].values():
        endpoint = max(int(value) for value in budget["thread_counts"])
        for tool in budget["tools"]:
            if tool["origin"] == "catalog":
                tool_id = str(tool["tool_id"])
                catalog_requirements[tool_id] = max(endpoint, catalog_requirements.get(tool_id, 0))
    for tool_id, endpoint in sorted(catalog_requirements.items()):
        spec_path = (
            andrea_repo.resolve()
            / "andrea/catalog_inference_tools/tools"
            / tool_id
            / "toolspec.json"
        )
        spec = load_json_object(spec_path, f"performance {tool_id} ToolSpec")
        runtime = spec.get("runtime_resources")
        threading = runtime.get("threading") if isinstance(runtime, Mapping) else None
        maximum = threading.get("max_threads") if isinstance(threading, Mapping) else None
        if (
            spec.get("id") != tool_id
            or not isinstance(threading, Mapping)
            or threading.get("supported") is not True
            or threading.get("default_threads") != 1
            or (maximum is not None and (type(maximum) is not int or maximum < endpoint))
            or not isinstance(threading.get("upstream_mapping"), str)
            or not threading["upstream_mapping"].strip()
        ):
            raise BenchmarkConfigError(
                f"Performance tool {tool_id!r} does not support the frozen 1–{endpoint} "
                "thread contract"
            )


def resolve_performance_assignment(runtime_plan: Mapping[str, Any], threads: int) -> dict[str, Any]:
    """Bind one thread endpoint to a deterministic subset of the runtime-plan CPUs."""

    capacity = runtime_plan.get("capacity")
    detected = runtime_plan.get("detected")
    if not isinstance(capacity, Mapping) or not isinstance(detected, Mapping):
        raise BenchmarkConfigError("Performance execution requires a valid runtime plan")
    available_threads = capacity.get("cpu_threads")
    allowed_cpus = detected.get("allowed_cpus")
    allocation_order = detected.get("cpu_allocation_order")
    if (
        not isinstance(threads, int)
        or isinstance(threads, bool)
        or threads < 1
        or not isinstance(available_threads, int)
        or isinstance(available_threads, bool)
        or not isinstance(allowed_cpus, list)
        or not isinstance(allocation_order, list)
        or any(not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0 for cpu in allowed_cpus)
        or allowed_cpus != sorted(set(allowed_cpus))
        or any(
            not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0 for cpu in allocation_order
        )
        or sorted(allocation_order) != allowed_cpus
        or len(set(allocation_order)) != len(allocation_order)
    ):
        raise BenchmarkConfigError("Performance runtime-plan CPU contract is invalid")
    if threads > available_threads or threads > len(allowed_cpus):
        raise BenchmarkConfigError(
            f"Performance endpoint requests {threads} threads but only "
            f"{available_threads} are schedulable"
        )
    # Preserve topology-aware selection while satisfying ANDREA's canonical
    # strictly-increasing representation of a CPU set.
    return {"threads": threads, "cpuset_cpus": sorted(allocation_order[:threads])}


def _catalog_scientific_requests(
    *, repo_root: Path, budget_design: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    request_path = repo_root / str(budget_design["catalog_request"])
    metadata_path = repo_root / str(budget_design["catalog_metadata"])
    requests = load_json_object(request_path, "performance catalog request")
    metadata = load_json_object(metadata_path, "performance catalog metadata")
    runs_by_id = {str(item["run_id"]): item for item in requests.get("runs", [])}
    metadata_by_configuration: dict[str, list[Mapping[str, Any]]] = {}
    for raw in metadata.get("runs", []):
        if isinstance(raw, Mapping):
            metadata_by_configuration.setdefault(str(raw.get("configuration_id")), []).append(raw)
    selected: list[dict[str, Any]] = []
    seed_namespace = str(budget_design["algorithm_seed_namespace"])
    seed_coordinates = {
        "purpose": "algorithm_replicate",
        "replicate_index": int(budget_design["algorithm_seed_replicate_index"]),
    }
    common_seed = derive_plan_seed(
        load_seed_plan(repo_root / "experiments/v1/seeds/seed-plan.json"),
        namespace=seed_namespace,
        stream="inference",
        coordinates=seed_coordinates,
    )
    for tool in budget_design["tools"]:
        if tool["origin"] != "catalog":
            continue
        configuration_id = str(tool["configuration_id"])
        candidates = [
            item
            for item in metadata_by_configuration.get(configuration_id, [])
            if item.get("selection_role") == "baseline"
            and item.get("seed_derivation", {}).get("coordinates", {}).get("replicate_index")
            == budget_design["algorithm_seed_replicate_index"]
        ]
        if len(candidates) != 1:
            raise BenchmarkConfigError(
                f"Performance configuration {configuration_id!r} has no unique default seed source"
            )
        meta = candidates[0]
        request = runs_by_id.get(str(meta["run_id"]))
        if request is None or request.get("tool_id") != tool["tool_id"]:
            raise BenchmarkConfigError(
                f"Performance configuration {configuration_id!r} differs from its catalog request"
            )
        source_seed = meta.get("algorithm_seed")
        if not isinstance(source_seed, int) or isinstance(source_seed, bool):
            raise BenchmarkConfigError(
                f"Performance configuration {configuration_id!r} has no stochastic seed"
            )
        source_params = request.get("params")
        if not isinstance(source_params, Mapping) or source_params.get("seed") != source_seed:
            raise BenchmarkConfigError(
                f"Performance configuration {configuration_id!r} has inconsistent source seed"
            )
        performance_request = copy.deepcopy(request)
        performance_request["run_id"] = f"{configuration_id}__performance_seed_{common_seed}"
        performance_request["params"]["seed"] = common_seed
        performance_metadata = copy.deepcopy(dict(meta))
        performance_metadata.update(
            {
                "run_id": performance_request["run_id"],
                "algorithm_seed": common_seed,
                "seed_derivation": {
                    "namespace": seed_namespace,
                    "stream": "inference",
                    "coordinates": seed_coordinates,
                },
            }
        )
        selected.append(
            {
                "configuration_id": configuration_id,
                "tool_id": str(tool["tool_id"]),
                "origin": "catalog",
                "algorithm_seed": common_seed,
                "request": performance_request,
                "custom_tool": None,
                "metadata": performance_metadata,
            }
        )
    return selected, common_seed


def _spathi_scientific_request(
    *,
    repo_root: Path,
    budget_design: Mapping[str, Any],
    production: Mapping[str, Any],
    algorithm_seed: int,
) -> dict[str, Any]:
    policy = production.get("selected_policy")
    if not isinstance(policy, Mapping):
        raise BenchmarkConfigError("Performance production configuration has no policy")
    tools, custom_tools, metadata = build_selected_spathi_requests(
        policy,
        [algorithm_seed],
        docker_image=load_image_lock(repo_root)["images"]["spathi"]["tag"],
        seed_namespace=str(budget_design["algorithm_seed_namespace"]),
    )
    request = tools["runs"][0]
    custom = custom_tools["tools"][0]
    run_metadata = metadata["runs"][0]
    expected = [
        item for item in budget_design["tools"] if item["origin"] == "spathi_selected_policy"
    ]
    if len(expected) != 1:
        raise BenchmarkConfigError("Performance design requires exactly one SPATHI policy entry")
    return {
        "configuration_id": str(expected[0]["configuration_id"]),
        "tool_id": str(expected[0]["tool_id"]),
        "origin": "spathi_selected_policy",
        "algorithm_seed": algorithm_seed,
        "request": request,
        "custom_tool": custom,
        "metadata": run_metadata,
    }


def _condition_key(condition: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(condition["generation_job_id"]),
            str(condition["configuration_id"]),
            str(condition["threads"]),
        )
    )


def _counterbalanced(
    conditions: Sequence[dict[str, Any]], *, salt: str, block_index: int
) -> list[dict[str, Any]]:
    base = sorted(
        conditions,
        key=lambda item: hashlib.sha256(f"{salt}|{_condition_key(item)}".encode()).hexdigest(),
    )
    if not base:
        return []
    offset = (block_index * max(1, len(base) // 3)) % len(base)
    rotated = base[offset:] + base[:offset]
    if block_index % 2:
        rotated.reverse()
    return [copy.deepcopy(item) for item in rotated]


def build_performance_tasks(
    *,
    budget_design: Mapping[str, Any],
    runtime_plan: Mapping[str, Any],
    order_salt: str,
    ram_gb: float,
) -> list[dict[str, Any]]:
    """Expand and freeze warm-up plus measured tasks in deterministic order."""

    if (
        not isinstance(ram_gb, (int, float))
        or isinstance(ram_gb, bool)
        or not math.isfinite(float(ram_gb))
        or float(ram_gb) <= 0
    ):
        raise BenchmarkConfigError("Performance RAM allocation must be positive")

    conditions = [
        {
            **copy.deepcopy(dataset),
            "configuration_id": str(tool["configuration_id"]),
            "tool_id": str(tool["tool_id"]),
            "origin": str(tool["origin"]),
            "threads": int(threads),
            "ram_gb": float(ram_gb),
            "cpuset_cpus": resolve_performance_assignment(runtime_plan, int(threads))[
                "cpuset_cpus"
            ],
        }
        for dataset in budget_design["datasets"]
        for tool in budget_design["tools"]
        for threads in budget_design["thread_counts"]
    ]
    tasks: list[dict[str, Any]] = []
    for condition in _counterbalanced(conditions, salt=order_salt, block_index=0):
        tasks.append({**condition, "phase": "warmup", "system_repetition": 0})
    block = 1
    for repetition in range(1, int(budget_design["system_repetitions"]) + 1):
        for dataset in budget_design["datasets"]:
            subset = [
                item
                for item in conditions
                if item["generation_job_id"] == dataset["generation_job_id"]
            ]
            for condition in _counterbalanced(subset, salt=order_salt, block_index=block):
                tasks.append({**condition, "phase": "measurement", "system_repetition": repetition})
            block += 1
    for sequence_index, task in enumerate(tasks, start=1):
        prefix = "warmup" if task["phase"] == "warmup" else "measure"
        task["sequence_index"] = sequence_index
        task["task_id"] = (
            f"{prefix}_{task['size_class']}_{task['configuration_id']}_"
            f"t{task['threads']}_r{task['system_repetition']}"
        )
    identifiers = [task["task_id"] for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise BenchmarkConfigError("Performance materialization produced duplicate task IDs")
    return tasks


def materialize_performance_protocol(
    *,
    repo_root: Path,
    budget: str,
    template_path: Path,
    template_profile: Mapping[str, Any],
    production_path: Path,
    production: Mapping[str, Any],
    runtime_plan_path: Path,
    runtime_plan: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Freeze requests, resource assignments, order and a runnable generation profile."""

    if budget not in _EXPECTED_BUDGETS:
        raise BenchmarkConfigError("Performance budget must be 'reduced' or 'full'")
    root = repo_root.resolve()
    design = load_performance_design(root)
    budget_design = design["budgets"][budget]
    if (
        template_profile.get("id") != budget_design["profile_id"]
        or template_profile.get("purpose") != "performance"
        or template_profile.get("status") != "runnable"
        or template_profile.get("execution_class") != "performance_full"
    ):
        raise BenchmarkConfigError("Performance template does not match its budget")
    expected_jobs = {item["generation_job_id"] for item in budget_design["datasets"]}
    observed_jobs = {str(item.get("id")) for item in template_profile.get("generation_jobs", [])}
    if expected_jobs != observed_jobs:
        raise BenchmarkConfigError("Performance template differs from its dataset panel")

    execution = resolve_execution_class(
        repo_root=root,
        class_id="performance_full",
        runtime_plan=runtime_plan,
    )
    if execution["output_profile"] != design["protocol"]["output_profile"]:
        raise BenchmarkConfigError("Performance execution class does not produce full outputs")
    if not execution["exclusive_inference"]:
        raise BenchmarkConfigError("Performance execution class is not isolated")
    max_threads = max(int(value) for value in budget_design["thread_counts"])
    resolve_performance_assignment(runtime_plan, max_threads)

    catalog, algorithm_seed = _catalog_scientific_requests(
        repo_root=root,
        budget_design=budget_design,
    )
    spathi = _spathi_scientific_request(
        repo_root=root,
        budget_design=budget_design,
        production=production,
        algorithm_seed=algorithm_seed,
    )
    by_configuration = {item["configuration_id"]: item for item in [spathi, *catalog]}
    requested_order = [item["configuration_id"] for item in budget_design["tools"]]
    if set(by_configuration) != set(requested_order):
        raise BenchmarkConfigError("Performance materialization resolved the wrong methods")
    scientific_requests = [by_configuration[item] for item in requested_order]
    tasks = build_performance_tasks(
        budget_design=budget_design,
        runtime_plan=runtime_plan,
        order_salt=str(design["protocol"]["order_salt"]),
        ram_gb=float(execution["operations"]["inference"]["memory_gib"]),
    )
    measured = sum(task["phase"] == "measurement" for task in tasks)
    warmups = sum(task["phase"] == "warmup" for task in tasks)
    if (
        measured != budget_design["expected_measured_runs"]
        or warmups != budget_design["expected_warmup_runs"]
    ):
        raise BenchmarkConfigError("Performance task expansion differs from the declared counts")

    materialized_root = path_below_no_symlinks(
        output_dir,
        root / "work",
        "Performance materialization",
    )
    materialized_root.mkdir(parents=True, exist_ok=True)
    catalog_tools = {"runs": [copy.deepcopy(item["request"]) for item in catalog]}
    catalog_metadata = {
        "schema_version": "1.0",
        "runs": [copy.deepcopy(item["metadata"]) for item in catalog],
    }
    spathi_tools = {"runs": [copy.deepcopy(spathi["request"])]}
    spathi_custom = {"tools": [copy.deepcopy(spathi["custom_tool"])]}
    spathi_metadata = {"schema_version": "1.0", "runs": [copy.deepcopy(spathi["metadata"])]}
    paths = {
        "catalog_tools": write_idempotent_json(
            materialized_root / "catalog-tools.json",
            catalog_tools,
            "performance catalog tools",
            boundary=root,
        ),
        "catalog_metadata": write_idempotent_json(
            materialized_root / "catalog-run-metadata.json",
            catalog_metadata,
            "performance catalog metadata",
            boundary=root,
        ),
        "spathi_tools": write_idempotent_json(
            materialized_root / "spathi-tools.json",
            spathi_tools,
            "performance SPATHI tools",
            boundary=root,
        ),
        "spathi_custom": write_idempotent_json(
            materialized_root / "spathi-custom-tools.json",
            spathi_custom,
            "performance SPATHI custom tool",
            boundary=root,
        ),
        "spathi_metadata": write_idempotent_json(
            materialized_root / "spathi-run-metadata.json",
            spathi_metadata,
            "performance SPATHI metadata",
            boundary=root,
        ),
    }
    runtime_profile = copy.deepcopy(dict(template_profile))
    runtime_profile_id = f"performance_{budget}_{canonical_json_sha256(tasks)[:16]}"
    runtime_profile.update(
        {
            "id": runtime_profile_id,
            "selected_config": _relative(root, production_path, "production config"),
            "inference_batches": [
                {
                    "id": f"performance_catalog_{budget}",
                    "kind": "baseline",
                    "dataset_role": "performance",
                    "generation_jobs": sorted(expected_jobs),
                    "replicate_indices": {
                        item["generation_job_id"]: [int(item["replicate_index"])]
                        for item in budget_design["datasets"]
                    },
                    "regime": "group_only",
                    "tools_params": _relative(root, paths["catalog_tools"], "catalog tools"),
                    "run_metadata": _relative(root, paths["catalog_metadata"], "catalog metadata"),
                },
                {
                    "id": f"performance_spathi_{budget}",
                    "kind": "spathi",
                    "dataset_role": "performance",
                    "generation_jobs": sorted(expected_jobs),
                    "replicate_indices": {
                        item["generation_job_id"]: [int(item["replicate_index"])]
                        for item in budget_design["datasets"]
                    },
                    "regime": "group_only",
                    "tools_params": _relative(root, paths["spathi_tools"], "SPATHI tools"),
                    "custom_tools": _relative(root, paths["spathi_custom"], "SPATHI custom tools"),
                    "run_metadata": _relative(root, paths["spathi_metadata"], "SPATHI metadata"),
                },
            ],
        }
    )
    profile_path = write_idempotent_json(
        materialized_root / "profile.json",
        runtime_profile,
        "performance runtime profile",
        boundary=root,
    )
    validate_json(
        runtime_profile,
        root / "schemas/profile.schema.json",
        "performance runtime profile",
    )
    validate_profile_references(root, runtime_profile)

    inference_resources = execution["operations"]["inference"]
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "spathi_performance_materialization",
        "budget": budget,
        "design": {
            "path": DESIGN_PATH.as_posix(),
            "sha256": sha256_file(root / DESIGN_PATH),
        },
        "template_profile": {
            "path": _relative(root, template_path, "performance template"),
            "sha256": sha256_file(template_path),
        },
        "selected_config": {
            "path": _relative(root, production_path, "production config"),
            "sha256": sha256_file(production_path),
            "policy_id": production["selected_policy"]["policy_id"],
            "policy_sha256": production["selected_policy"]["policy_sha256"],
        },
        "runtime_plan": {
            "path": _relative(root, runtime_plan_path, "runtime plan"),
            "sha256": sha256_file(runtime_plan_path),
            "fingerprint": runtime_plan["fingerprint"],
            "execution_class": "performance_full",
            "request": copy.deepcopy(dict(runtime_plan["request"])),
            "detected": copy.deepcopy(dict(runtime_plan["detected"])),
            "capacity": {
                "cpu_threads": int(runtime_plan["capacity"]["cpu_threads"]),
                "memory_bytes": int(runtime_plan["capacity"]["memory_bytes"]),
            },
        },
        "execution": {
            "output_profile": execution["output_profile"],
            "inference_cache": design["protocol"]["inference_cache"],
            "concurrency": design["protocol"]["concurrency"],
            "order_algorithm": design["protocol"]["order_algorithm"],
            "order_salt": design["protocol"]["order_salt"],
            "max_ram_gb": float(inference_resources["memory_gib"]),
            "planner": inference_resources["planner"],
            "planner_time_limit_seconds": float(inference_resources["planner_time_limit_seconds"]),
        },
        "datasets": copy.deepcopy(budget_design["datasets"]),
        "scientific_requests": [
            {key: copy.deepcopy(value) for key, value in item.items() if key != "metadata"}
            for item in scientific_requests
        ],
        "tasks": tasks,
        "schedule_sha256": canonical_json_sha256(tasks),
    }
    validate_json(
        manifest,
        root / "schemas/performance-materialization.schema.json",
        "performance materialization",
    )
    manifest_path = write_idempotent_json(
        materialized_root / "performance-materialization.json",
        manifest,
        "performance materialization",
        boundary=root,
    )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "runtime_profile": runtime_profile,
        "profile_path": profile_path,
        "profile_id": runtime_profile_id,
        "scientific_requests": {item["configuration_id"]: item for item in scientific_requests},
    }


__all__ = [
    "DESIGN_PATH",
    "build_performance_tasks",
    "load_performance_design",
    "materialize_performance_protocol",
    "resolve_performance_assignment",
    "validate_performance_catalog_threading",
]
