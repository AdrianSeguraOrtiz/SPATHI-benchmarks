"""Run the predeclared SPATHI efficiency panel into immutable artifacts.

The runner deliberately executes one ANDREA request per configuration, dataset,
and algorithmic seed.  Every request is content-addressed before execution, so
an interrupted invocation can retain complete artifacts and resume the missing
members of the predeclared counterbalanced order.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .artifacts import (
    Artifact,
    ArtifactError,
    ArtifactKey,
    ArtifactStore,
    canonical_json_sha256,
    sha256_file,
)
from .calibration import resolve_group_policy
from .config import (
    BenchmarkConfigError,
    load_json_object,
    load_repo_lock,
    validate_json,
    validate_locked_repository,
)
from .efficiency_order import (
    counterbalanced_execution_orders,
    runner_implementation_sha256,
)
from .manifests import unique_direct_manifest
from .noninferiority import _validate_plan_binding
from .noninferiority_evidence import _load_panel
from .runtime import (
    andrea_command,
    andrea_host_environment,
    benchmark_host_environment,
    docker_image_details,
    docker_runtime_environment,
)
from .safe_io import atomic_write_json, path_below_no_symlinks, write_idempotent_json
from .study_context import permit_revision_change
from .tool_contract import oom_observed, spathi_runtime_resources

_GIB = 1024**3


def _canonical_execution_measurement(
    report: Mapping[str, Any],
    *,
    expected_threads: int,
    expected_ram_gb: float,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate and return ANDREA's canonical end-to-end telemetry."""

    execution = report.get("execution")
    measurement = execution.get("measurement") if isinstance(execution, Mapping) else None
    if not isinstance(measurement, Mapping):
        raise BenchmarkConfigError("ANDREA report omits canonical end-to-end telemetry")
    wall = measurement.get("wall_time_seconds")
    cpu = measurement.get("cpu_time_seconds")
    peak = measurement.get("peak_memory_bytes")
    output = measurement.get("output_bytes")
    assigned = measurement.get("assigned_resources")
    telemetry = measurement.get("telemetry")
    if (
        measurement.get("schema_version") != "1.0"
        or measurement.get("scope") != "andrea_execution"
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0
        or (
            cpu is not None
            and (
                isinstance(cpu, bool)
                or not isinstance(cpu, (int, float))
                or not math.isfinite(float(cpu))
                or float(cpu) < 0
            )
        )
        or (peak is not None and (type(peak) is not int or peak < 0))
        or type(output) is not int
        or output < 0
        or not isinstance(assigned, Mapping)
        or type(assigned.get("threads_peak")) is not int
        or int(assigned["threads_peak"]) > expected_threads
        or type(assigned.get("ram_limit_peak_bytes")) is not int
        or int(assigned["ram_limit_peak_bytes"]) > round(expected_ram_gb * _GIB)
        or not isinstance(telemetry, Mapping)
        or telemetry.get("status") not in {"complete", "partial", "unavailable"}
        or telemetry.get("wall_source") != "andrea_monotonic_clock"
        or telemetry.get("wall_semantics")
        != "andrea_execution_end_to_end_including_input_verification_preparation_"
        "scheduling_merging_and_requested_output_materialization"
    ):
        raise BenchmarkConfigError("ANDREA end-to-end telemetry is invalid")
    if require_complete and (
        float(wall) <= 0
        or not isinstance(cpu, (int, float))
        or isinstance(cpu, bool)
        or type(peak) is not int
        or peak < 1
        or output < 1
        or assigned.get("threads_peak") != expected_threads
        or assigned.get("ram_limit_peak_bytes") != round(expected_ram_gb * _GIB)
        or telemetry.get("status") != "complete"
    ):
        raise BenchmarkConfigError(
            "Completed efficiency inference lacks complete canonical resource telemetry"
        )
    return dict(measurement)


def _inside(repo_root: Path, path: Path, label: str, *, require_file: bool = False) -> str:
    root = repo_root.resolve()
    safe = path_below_no_symlinks(
        path,
        root,
        label,
        require_file=require_file,
    )
    return safe.relative_to(root).as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload, "efficiency staging document")


def _single_dataset_manifest(artifact: Artifact, dataset_id: str) -> Path:
    suffix = f"datasets/{dataset_id}/dataset-manifest.json"
    matches = [
        entry["path"]
        for entry in artifact.manifest["files"]
        if entry["path"] == suffix or entry["path"].endswith("/" + suffix)
    ]
    if len(matches) != 1:
        raise BenchmarkConfigError(
            f"Dataset artifact {artifact.key} has {len(matches)} manifests for {dataset_id!r}"
        )
    return artifact.file(matches[0])


def _configuration_run(
    *,
    configuration_id: str,
    parameters: Mapping[str, Any],
    policy_sha256: str,
    group_count: int,
    algorithm_seed: int,
    seed_derivation: Mapping[str, Any],
    role: str,
    image_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_id = f"efficiency_{role}_{configuration_id}__seed_{algorithm_seed}"
    run = {
        "run_id": run_id,
        "tool_id": f"custom_{run_id}",
        "params": {**dict(parameters), "random_seed": algorithm_seed},
        "execution": {"mode": "group_native"},
    }
    metadata = {
        "run_id": run_id,
        "configuration_id": configuration_id,
        "configuration_sha256": canonical_json_sha256(parameters),
        "policy_sha256": policy_sha256,
        "group_count": group_count,
        "selection_role": role,
        "algorithm_seed": algorithm_seed,
        "algorithm_seed_parameter": "random_seed",
        "seed_derivation": dict(seed_derivation),
        "family": "spathi",
    }
    custom = {
        "run_id": run_id,
        "name": f"SPATHI efficiency {configuration_id} seed {algorithm_seed}",
        "docker_image": image_id,
        "execution_mode": "group_native",
        "extra_inputs": ["groups", "tf_list"],
        "outputs": {"directed": True, "sign": "none"},
        "runtime_resources": spathi_runtime_resources(),
    }
    return run, metadata, custom


def _inference_spec(
    *,
    dataset: Artifact,
    dataset_id: str,
    andrea_commit: str,
    andrea_environment: str,
    benchmark_environment: str,
    docker_runtime: str,
    run: Mapping[str, Any],
    metadata: Mapping[str, Any],
    custom: Mapping[str, Any],
    image_contract: Mapping[str, Any],
    resources: Mapping[str, Any],
    measurement_protocol: Mapping[str, Any],
    runner_sha256: str,
    execution_order_block: Mapping[str, Any],
    execution_position: int,
    batch_kind: str = "spathi_efficiency",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset_artifact": dataset.key.value,
        "dataset_id": dataset_id,
        "andrea_commit": andrea_commit,
        "andrea_environment": andrea_environment,
        "benchmark_environment": benchmark_environment,
        "docker_runtime": docker_runtime,
        "batch_kind": batch_kind,
        "run": dict(run),
        "run_metadata": dict(metadata),
        "custom_definition": dict(custom),
        "tool_image": dict(image_contract),
        "resources": dict(resources),
        "measurement_protocol": dict(measurement_protocol),
        "runner_sha256": runner_sha256,
        "execution_order_block": dict(execution_order_block),
        "execution_position": execution_position,
    }


def _validate_execution_output(
    output_root: Path,
    *,
    dataset_id: str,
    run_id: str,
    expected_threads: int,
    expected_ram_gb: float,
    output_profile: str,
) -> None:
    report_path = unique_direct_manifest(output_root, "run_report.json")
    report = load_json_object(report_path, "efficiency inference report")
    tools = report.get("tools") or {}
    if (
        report.get("status") != "executed"
        or report.get("output_profile") != output_profile
        or report.get("run_id") != run_id
        or report.get("dataset", {}).get("id") != dataset_id
        or tools.get("selected") != [run_id]
        or tools.get("completed") != [run_id]
        or tools.get("status_by_tool", {}).get(run_id)
        not in {"completed", "completed_with_warnings"}
        or tools.get("failed")
        or tools.get("tool_origins", {}).get(run_id) != "custom"
    ):
        raise BenchmarkConfigError(f"ANDREA did not complete exactly {run_id!r} on {dataset_id!r}")
    _canonical_execution_measurement(
        report,
        expected_threads=expected_threads,
        expected_ram_gb=expected_ram_gb,
        require_complete=True,
    )
    for suffix in (
        "io/out/network.csv",
        "io/out/target_eligibility.tsv.gz",
        "io/out/model_diagnostics.tsv.gz",
        "io/out/run_metadata.json",
    ):
        matches = [
            path
            for path in output_root.rglob(Path(suffix).name)
            if path.as_posix().endswith(suffix)
        ]
        if len(matches) != 1 or not matches[0].is_file():
            raise BenchmarkConfigError(
                f"Efficiency run {run_id!r} did not publish exactly one {suffix}"
            )


def _known_tool_failure(
    output_root: Path,
    *,
    dataset_id: str,
    run_id: str,
    expected_threads: int,
    expected_ram_gb: float,
    output_profile: str,
) -> dict[str, Any] | None:
    """Return an auditable tool failure, while rejecting ambiguous infrastructure output."""

    try:
        report_path = unique_direct_manifest(output_root, "run_report.json")
        report = load_json_object(report_path, "efficiency inference failure report")
    except BenchmarkConfigError:
        return None
    tools = report.get("tools")
    if not isinstance(tools, Mapping):
        return None
    failed = tools.get("failed")
    failure_message = failed.get(run_id) if isinstance(failed, Mapping) else None
    tool_status = tools.get("status_by_tool", {}).get(run_id)
    if (
        report.get("status") != "executed"
        or report.get("output_profile") != output_profile
        or report.get("run_id") != run_id
        or report.get("dataset", {}).get("id") != dataset_id
        or tools.get("selected") != [run_id]
        or run_id in (tools.get("completed") or [])
        or tools.get("tool_origins", {}).get(run_id) != "custom"
        or not isinstance(tool_status, str)
        or tool_status in {"completed", "completed_with_warnings"}
        or not isinstance(failure_message, str)
        or not failure_message.strip()
    ):
        return None
    try:
        measurement = _canonical_execution_measurement(
            report,
            expected_threads=expected_threads,
            expected_ram_gb=expected_ram_gb,
            require_complete=False,
        )
    except BenchmarkConfigError:
        return None
    result = (tools.get("results") or {}).get(run_id)
    if not isinstance(result, Mapping):
        return None

    return {
        "schema_version": "1.0",
        "status": "tool_execution_failed",
        "dataset_id": dataset_id,
        "run_id": run_id,
        "tool_status": tool_status,
        "failure_message": failure_message,
        "run_report_sha256": sha256_file(report_path),
        "measurement_sha256": canonical_json_sha256(measurement),
        "wall_time_seconds": float(measurement["wall_time_seconds"]),
        "peak_memory_bytes": measurement["peak_memory_bytes"],
        "oom_killed": oom_observed(result),
    }


def _publish_bound_tool_failure(
    *,
    task: Mapping[str, Any],
    staging: Path,
    output_root: Path,
    store: ArtifactStore,
) -> Artifact | None:
    batch_kind = task["spec"].get("batch_kind")
    if batch_kind not in {"spathi_efficiency", "spathi_efficiency_safety"}:
        return None
    # A failed exact reference makes either study inconclusive and must remain
    # a hard error. Only a bound candidate failure is valid fail-safe evidence.
    if task.get("role") != "candidate":
        return None
    failure = _known_tool_failure(
        output_root,
        dataset_id=str(task["dataset_id"]),
        run_id=str(task["run"]["run_id"]),
        expected_threads=int(task["run"]["resources"]["threads"]),
        expected_ram_gb=float(task["run"]["resources"]["ram_gb"]),
        output_profile=str(task["resources"]["output_profile"]),
    )
    if failure is None:
        return None
    _write_json(staging / "efficiency-execution-failure.json", failure)
    return store.publish_from_temp(staging, kind="inference", spec=task["spec"])


def _execute_task(
    task: dict[str, Any],
    *,
    store: ArtifactStore,
    repo_root: Path,
    andrea_repo: Path,
    log_root: Path,
) -> Artifact:
    """Execute one missing task.  Kept separate to make failure/restart tests exact."""

    key = task["key"]
    with store.staging_directory(key) as staging:
        request_root = staging / "request"
        tools_path = request_root / "tools-params.json"
        custom_path = request_root / "custom-tools.json"
        _write_json(tools_path, {"runs": [task["run"]]})
        _write_json(custom_path, {"tools": [task["custom"]]})
        output_root = staging / "output"
        resources = task["resources"]
        try:
            andrea_command(
                [
                    "infer-network",
                    "execute",
                    "--dataset-manifest",
                    str(task["dataset_manifest"]),
                    "--tools-params",
                    str(tools_path),
                    "--custom-tools",
                    str(custom_path),
                    "--output-dir",
                    str(output_root),
                    "--max-cores",
                    str(task["run"]["resources"]["threads"]),
                    "--max-ram-gb",
                    str(resources["memory_gib"]),
                    "--planner",
                    str(resources["planner"]),
                    "--planner-time-limit-seconds",
                    str(resources["planner_time_limit_seconds"]),
                    "--output-profile",
                    str(resources["output_profile"]),
                ],
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                log_path=log_root / f"{task['dataset_id']}__{task['run']['run_id']}.log",
            )
        except BaseException:
            failure_artifact = _publish_bound_tool_failure(
                task=task,
                staging=staging,
                output_root=output_root,
                store=store,
            )
            if failure_artifact is not None:
                return failure_artifact
            raise
        try:
            _validate_execution_output(
                output_root,
                dataset_id=task["dataset_id"],
                run_id=task["run"]["run_id"],
                expected_threads=int(task["run"]["resources"]["threads"]),
                expected_ram_gb=float(task["run"]["resources"]["ram_gb"]),
                output_profile=str(resources["output_profile"]),
            )
        except BenchmarkConfigError:
            failure_artifact = _publish_bound_tool_failure(
                task=task,
                staging=staging,
                output_root=output_root,
                store=store,
            )
            if failure_artifact is not None:
                return failure_artifact
            raise
        return store.publish_from_temp(staging, kind="inference", spec=task["spec"])


def _execute_missing_tasks(
    tasks: list[dict[str, Any]], execute_one: Callable[[dict[str, Any]], Artifact]
) -> None:
    """Resume a frozen serial order at its first missing immutable artifact.

    Publication is atomic, so a process interruption can leave only a completed
    prefix of the serial task list.  Retaining that prefix and executing the
    suffix preserves the counterbalanced order and never reuses a partial run.
    """

    observed_missing = False
    for task in tasks:
        if task["artifact"] is None:
            observed_missing = True
            task["artifact"] = execute_one(task)
        elif observed_missing:
            raise BenchmarkConfigError(
                "Efficiency cache is not a completed prefix of the frozen serial order"
            )


def _runner_sha256() -> str:
    return runner_implementation_sha256()


def _bind_execution_contract(
    path: Path, observed: Mapping[str, Any], *, boundary: Path
) -> dict[str, Any]:
    """Freeze requested identities before the first expensive formal inference.

    Source-only continuation retains the original cache coordinates; each newly
    published artifact separately records the invocation's actual producer.
    """

    path = path_below_no_symlinks(path, boundary, "efficiency execution contract")
    if not path.exists():
        write_idempotent_json(path, dict(observed), "efficiency execution contract", boundary=boundary)
        return dict(observed)
    recorded = load_json_object(path, "efficiency execution contract")
    if set(recorded) != set(observed):
        raise BenchmarkConfigError("Efficiency execution contract fields changed")
    for field, length in (("andrea_commit", 40), ("runner_sha256", 64)):
        if not isinstance(recorded[field], str) or re.fullmatch(
            rf"[0-9a-f]{{{length}}}", recorded[field]
        ) is None:
            raise BenchmarkConfigError(f"Efficiency execution contract has invalid {field}")
    for field, value in observed.items():
        if recorded[field] == value:
            continue
        if field in {"andrea_commit", "runner_sha256"} and permit_revision_change(
            f"Efficiency {field}", recorded[field], value
        ):
            continue
        raise BenchmarkConfigError(f"Efficiency execution contract changed: {field}")
    return recorded


def run_noninferiority_artifacts(
    *,
    repo_root: Path,
    plan_path: Path,
    andrea_repo: Path,
    artifact_index_path: Path,
    execution_plan: Mapping[str, Any],
    artifact_store_path: Path | None = None,
) -> Path:
    """Run or resume the exact reference/candidate product declared by a plan."""

    operations = execution_plan.get("operations")
    inference = operations.get("inference") if isinstance(operations, Mapping) else None
    if not isinstance(inference, Mapping):
        raise BenchmarkConfigError("Efficiency execution plan has no inference allocation")
    cpu_threads = inference.get("cpu_threads")
    memory_bytes = inference.get("memory_bytes")
    memory_gib = inference.get("memory_gib")
    planner = inference.get("planner")
    planner_time_limit_seconds = inference.get("planner_time_limit_seconds")
    if (
        not isinstance(cpu_threads, int)
        or isinstance(cpu_threads, bool)
        or cpu_threads < 1
        or type(memory_bytes) is not int
        or memory_bytes < _GIB
        or not isinstance(memory_gib, (int, float))
        or isinstance(memory_gib, bool)
        or memory_gib <= 0
        or not math.isclose(float(memory_gib), memory_bytes / _GIB, rel_tol=0, abs_tol=1e-12)
    ):
        raise BenchmarkConfigError("Efficiency execution allocation must be positive")
    if (
        planner not in {"auto", "cp_sat", "heuristic"}
        or not isinstance(planner_time_limit_seconds, (int, float))
        or planner_time_limit_seconds <= 0
    ):
        raise BenchmarkConfigError("Invalid efficiency planner resources")
    output_profile = execution_plan.get("output_profile")
    fingerprint = execution_plan.get("fingerprint")
    unsigned_execution_plan = dict(execution_plan)
    unsigned_execution_plan.pop("fingerprint", None)
    if (
        output_profile != "full"
        or execution_plan.get("class_id") != "bulk_full"
        or not isinstance(fingerprint, str)
        or fingerprint != canonical_json_sha256(unsigned_execution_plan)
    ):
        raise BenchmarkConfigError("Efficiency execution plan is invalid")
    root = repo_root.resolve()
    _inside(root, plan_path, "Non-inferiority plan", require_file=True)
    _inside(root, artifact_index_path, "Artifact index output")
    plan = load_json_object(plan_path, "efficiency policy-pair plan")
    safety = plan.get("artifact_type") == "spathi_efficiency_safety_plan"
    if safety:
        from .efficiency_safety import load_safety_panel, validate_safety_plan

        validate_json(
            plan,
            root / "schemas/efficiency-safety-plan.schema.json",
            "efficiency safety plan",
        )
        validate_safety_plan(root, plan)
        panel_path, panel = load_safety_panel(root, plan)
    else:
        validate_json(
            plan,
            root / "schemas/noninferiority-plan.schema.json",
            "non-inferiority plan",
        )
        _validate_plan_binding(root, plan)
        panel_path, panel = _load_panel(root, plan)
    namespace = (
        "full.efficiency_safety" if safety else f"{plan['budget']['id']}.efficiency_validation"
    )
    batch_kind = "spathi_efficiency_safety" if safety else "spathi_efficiency"

    repository_lock = load_repo_lock(root)
    observed = validate_locked_repository(
        name="andrea",
        path=andrea_repo,
        lock_entry=repository_lock["repositories"]["andrea"],
        require_clean=True,
    )
    environment = andrea_host_environment()
    environment_fingerprint = str(environment["fingerprint"])
    benchmark_environment_fingerprint = str(benchmark_host_environment()["fingerprint"])
    docker_runtime_fingerprint = str(docker_runtime_environment()["fingerprint"])

    image_id = str(plan["reference"]["spathi_image_id"])
    details = docker_image_details(image_id)
    if details is None or details.get("id") != image_id:
        raise BenchmarkConfigError(
            "The exact SPATHI efficiency-study image is not available locally"
        )
    labels = details.get("labels") or {}
    if (
        labels.get("org.opencontainers.image.revision") != plan["reference"]["spathi_commit"]
        or labels.get("dev.spathi.benchmark.build-fingerprint")
        != plan["reference"]["spathi_build_fingerprint"]
    ):
        raise BenchmarkConfigError("The frozen SPATHI image labels differ from the plan")
    image_contract = {
        "kind": "spathi",
        "image_id": image_id,
        "source_commit": plan["reference"]["spathi_commit"],
        "build_fingerprint": plan["reference"]["spathi_build_fingerprint"],
    }
    resources: dict[str, Any] = {
        "cpu_threads": cpu_threads,
        "memory_bytes": memory_bytes,
        "memory_gib": memory_gib,
        "planner": planner,
        "planner_time_limit_seconds": planner_time_limit_seconds,
        "output_profile": output_profile,
        "execution_plan_fingerprint": fingerprint,
    }
    measurement_protocol: dict[str, Any] = {
        "source": "andrea_execution_measurement",
        "scope": "andrea_execution",
        "wall_time": (
            "andrea_execution_end_to_end_including_input_verification_preparation_"
            "scheduling_merging_and_requested_output_materialization"
        ),
        "peak_memory": "physical_container_peak_overlap_upper_bound_excluding_andrea_host",
        "output_bytes": "andrea_execution_output_tree_at_measurement",
        "telemetry_status": "complete_required",
        "model_counts": "validated_spathi_run_metadata",
        "tree_counts": "validated_spathi_model_diagnostics",
    }
    execution_contract = _bind_execution_contract(
        artifact_index_path.with_name(f"{artifact_index_path.stem}-execution.json"),
        {
            "plan_sha256": sha256_file(plan_path),
            "panel_sha256": sha256_file(panel_path),
            "andrea_commit": str(observed["commit"]),
            "runner_sha256": _runner_sha256(),
            "andrea_environment": environment_fingerprint,
            "benchmark_environment": benchmark_environment_fingerprint,
            "docker_runtime": docker_runtime_fingerprint,
            "resources": resources,
            "measurement_protocol": measurement_protocol,
        },
        boundary=root,
    )
    runner_sha256 = str(execution_contract["runner_sha256"])
    requested_andrea_commit = str(execution_contract["andrea_commit"])
    store_path = (artifact_store_path or root / "work/v1/artifacts").resolve()
    store_relative = _inside(root, store_path, "Artifact store")
    store = ArtifactStore(store_path)
    log_root = artifact_index_path.parent / "logs"

    tasks: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    candidates = sorted(plan["candidates"], key=lambda item: item["id"])
    policies = [
        (
            "reference",
            plan["reference"]["policy_id"],
            plan["reference"]["policy_sha256"],
            plan["reference"]["group_policy"],
        ),
        *[
            ("candidate", item["id"], item["policy_sha256"], item["group_policy"])
            for item in candidates
        ],
    ]
    labels = [
        "reference" if role == "reference" else policy_id
        for role, policy_id, _policy_sha256, _group_policy in policies
    ]
    execution_orders = counterbalanced_execution_orders(
        (item["dataset_id"] for item in panel["datasets"]),
        plan["budget"]["algorithm_seeds"],
        labels,
        execution_order_seed=plan["budget"]["execution_order_seed"],
    )
    policies_by_label = {
        "reference" if role == "reference" else policy_id: (
            role,
            policy_id,
            policy_sha256,
            group_policy,
        )
        for role, policy_id, policy_sha256, group_policy in policies
    }
    if safety:
        recorded = plan["seed_contract"]["algorithm_seed"]
        seed_derivations = {int(recorded["seed"]): dict(recorded["seed_derivation"])}
    else:
        seed_derivations = {
            seed: {
                "namespace": namespace,
                "stream": "inference",
                "coordinates": {
                    "purpose": "algorithm_replicate",
                    "replicate_index": index,
                },
            }
            for index, seed in enumerate(plan["budget"]["algorithm_seeds"], start=1)
        }
    for declaration in sorted(panel["datasets"], key=lambda item: item["dataset_id"]):
        dataset_id = declaration["dataset_id"]
        reference = declaration["dataset_artifact"]
        try:
            dataset = store.validate(ArtifactKey(kind="dataset", digest=str(reference["digest"])))
        except ArtifactError as exc:
            raise BenchmarkConfigError(
                f"Cannot use efficiency dataset {dataset_id!r}: {exc}"
            ) from exc
        dataset_manifest = _single_dataset_manifest(dataset, dataset_id)
        group_count = len(declaration["contexts"])
        for seed in plan["budget"]["algorithm_seeds"]:
            order = execution_orders[(dataset_id, seed)]
            execution_order_block = {
                "dataset_id": dataset_id,
                "algorithm_seed": seed,
                "execution_order_seed": plan["budget"]["execution_order_seed"],
                "execution_order": list(order),
            }
            ordered_policies = [policies_by_label[label] for label in order]
            group = {
                "dataset_id": dataset_id,
                "algorithm_seed": seed,
                "seed_derivation": seed_derivations[seed],
                "reference": None,
                "candidates": [],
                "execution_order": order,
            }
            grouped[(dataset_id, seed)] = group
            for execution_position, (
                role,
                configuration_id,
                policy_sha256,
                group_policy,
            ) in enumerate(ordered_policies, start=1):
                parameters = resolve_group_policy(group_policy, group_count)
                run, metadata, custom = _configuration_run(
                    configuration_id=configuration_id,
                    parameters=parameters,
                    policy_sha256=policy_sha256,
                    group_count=group_count,
                    algorithm_seed=seed,
                    seed_derivation=seed_derivations[seed],
                    role=role,
                    image_id=image_id,
                )
                run["resources"] = {"threads": cpu_threads, "ram_gb": memory_gib}
                spec = _inference_spec(
                    dataset=dataset,
                    dataset_id=dataset_id,
                    andrea_commit=requested_andrea_commit,
                    andrea_environment=environment_fingerprint,
                    benchmark_environment=benchmark_environment_fingerprint,
                    docker_runtime=docker_runtime_fingerprint,
                    run=run,
                    metadata=metadata,
                    custom=custom,
                    image_contract=image_contract,
                    resources=resources,
                    measurement_protocol=measurement_protocol,
                    runner_sha256=runner_sha256,
                    execution_order_block=execution_order_block,
                    execution_position=execution_position,
                    batch_kind=batch_kind,
                )
                key = store.key_for("inference", spec)
                artifact = store.lookup(key)
                task = {
                    "dataset": dataset,
                    "dataset_id": dataset_id,
                    "dataset_manifest": dataset_manifest,
                    "configuration_id": configuration_id,
                    "role": role,
                    "run": run,
                    "metadata": metadata,
                    "custom": custom,
                    "resources": resources,
                    "spec": spec,
                    "key": key,
                    "artifact": artifact,
                    "execution_order_block": execution_order_block,
                    "execution_position": execution_position,
                }
                tasks.append(task)

    def execute_missing(prepared: list[dict[str, Any]]) -> None:
        # Executing the list directly preserves the frozen within-block order and, unlike a
        # pre-submitted executor queue, stops at the first crash.  A subsequent
        # invocation therefore reuses validated completed artifacts and starts
        # at the first missing member without manual cache deletion.
        _execute_missing_tasks(
            prepared,
            lambda task: _execute_task(
                task,
                store=store,
                repo_root=root,
                andrea_repo=andrea_repo,
                log_root=log_root,
            ),
        )

    execute_missing(tasks)

    for task in tasks:
        artifact = task["artifact"]
        if not isinstance(artifact, Artifact):
            raise RuntimeError("Efficiency inference finished without an artifact")
        reference = {"kind": "inference", "digest": artifact.key.digest}
        group = grouped[(task["dataset_id"], task["metadata"]["algorithm_seed"])]
        if task["role"] == "reference":
            group["reference"] = reference
        else:
            group["candidates"].append(
                {
                    "candidate_id": task["configuration_id"],
                    "inference_artifact": reference,
                }
            )
    if any(group["reference"] is None for group in grouped.values()):
        raise RuntimeError("Efficiency artifact index lacks a reference run")
    for group in grouped.values():
        group["candidates"].sort(key=lambda item: item["candidate_id"])

    current = docker_image_details(image_id)
    if current is None or current.get("id") != image_id:
        raise BenchmarkConfigError("The frozen SPATHI image disappeared during execution")
    if benchmark_host_environment().get("fingerprint") != benchmark_environment_fingerprint:
        raise BenchmarkConfigError(
            "The benchmark host environment changed during efficiency measurement"
        )
    if docker_runtime_environment().get("fingerprint") != docker_runtime_fingerprint:
        raise BenchmarkConfigError("The Docker runtime changed during efficiency measurement")
    index = {
        "schema_version": "1.0",
        "artifact_type": (
            "spathi_efficiency_safety_artifact_index"
            if safety
            else "spathi_efficiency_artifact_index"
        ),
        "plan_sha256": sha256_file(plan_path),
        "panel_sha256": sha256_file(panel_path),
        "artifact_store": store_relative,
        "execution": {
            "andrea_commit": requested_andrea_commit,
            "andrea_environment": environment_fingerprint,
            "benchmark_environment": benchmark_environment_fingerprint,
            "docker_runtime": docker_runtime_fingerprint,
            "spathi_commit": plan["reference"]["spathi_commit"],
            "spathi_build_fingerprint": plan["reference"]["spathi_build_fingerprint"],
            "spathi_image_id": image_id,
            "execution_order_seed": plan["budget"]["execution_order_seed"],
            "execution_order_seed_derivation": {
                "namespace": namespace,
                "stream": "execution_order",
                "coordinates": {"purpose": "counterbalanced_execution_order"},
            },
            "resources": resources,
            "measurement_protocol": measurement_protocol,
            "runner_sha256": runner_sha256,
        },
        "runs": [grouped[key] for key in sorted(grouped)],
    }
    validate_json(
        index,
        root / "schemas/noninferiority-artifact-index.schema.json",
        "efficiency artifact index",
    )
    return write_idempotent_json(
        artifact_index_path,
        index,
        "efficiency artifact index",
        boundary=root,
    )
