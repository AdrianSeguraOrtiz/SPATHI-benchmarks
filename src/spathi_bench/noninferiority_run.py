"""Run the predeclared SPATHI efficiency panel into immutable artifacts.

The runner deliberately executes one ANDREA request per configuration, dataset,
and algorithmic seed.  Every request is content-addressed before execution, so
an interrupted invocation can retain complete artifacts and resume the missing
members of the predeclared counterbalanced order.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
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

_MEMORY_SAMPLE_INTERVAL_MILLISECONDS = 100
_MEMORY_UNITS = {
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
    "tb": 1000**4,
    "tib": 1024**4,
}
_MEMORY_VALUE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b)\s*$", re.I)


def _memory_bytes(raw: str) -> int:
    """Parse Docker's current-memory value without confusing SI and IEC units."""

    value = raw.split("/", 1)[0].strip()
    match = _MEMORY_VALUE.fullmatch(value)
    if match is None:
        raise BenchmarkConfigError(f"Docker returned an unsupported memory value: {raw!r}")
    multiplier = _MEMORY_UNITS.get(match.group(2).lower())
    if multiplier is None:
        raise BenchmarkConfigError(f"Docker returned an unsupported memory unit: {raw!r}")
    measured = int(float(match.group(1)) * multiplier)
    if measured < 1:
        raise BenchmarkConfigError(f"Docker returned a non-positive memory value: {raw!r}")
    return measured


class _ContainerMemorySampler:
    """Sample the one SPATHI container mounted below an ANDREA output root."""

    def __init__(self, *, output_root: Path, image_id: str) -> None:
        self.output_root = output_root.resolve()
        self.image_id = image_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[int] = []
        self._error: BaseException | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._sample,
            name="spathi-efficiency-memory",
            daemon=True,
        )
        self._thread.start()

    def finish(self) -> tuple[int, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise BenchmarkConfigError("SPATHI container memory sampler did not terminate")
        if self._error is not None:
            raise BenchmarkConfigError(f"Cannot measure SPATHI container memory: {self._error}")
        if not self._samples:
            raise BenchmarkConfigError(
                "No SPATHI container memory sample was captured; efficiency evidence cannot "
                "be published without an observed peak"
            )
        return max(self._samples), len(self._samples)

    def _docker(self, *arguments: str) -> str:
        completed = subprocess.run(
            ["docker", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode != 0:
            return ""
        return completed.stdout.strip()

    def _belongs_to_output(self, container_id: str) -> bool:
        raw = self._docker("inspect", "--format", "{{json .Mounts}}", container_id)
        if not raw:
            return False
        try:
            mounts = json.loads(raw)
        except json.JSONDecodeError:
            return False
        for mount in mounts if isinstance(mounts, list) else []:
            source = mount.get("Source") if isinstance(mount, dict) else None
            if not isinstance(source, str):
                continue
            try:
                Path(source).resolve().relative_to(self.output_root)
            except ValueError:
                continue
            return True
        return False

    def _sample(self) -> None:
        try:
            container_id: str | None = None
            interval = _MEMORY_SAMPLE_INTERVAL_MILLISECONDS / 1000.0
            while not self._stop.is_set():
                if container_id is None:
                    ids = self._docker(
                        "ps",
                        "--no-trunc",
                        "--filter",
                        f"ancestor={self.image_id}",
                        "--format",
                        "{{.ID}}",
                    ).splitlines()
                    container_id = next(
                        (item for item in ids if item and self._belongs_to_output(item)),
                        None,
                    )
                if container_id is not None:
                    raw = self._docker(
                        "stats", "--no-stream", "--format", "{{.MemUsage}}", container_id
                    )
                    if raw:
                        self._samples.append(_memory_bytes(raw.splitlines()[-1]))
                self._stop.wait(interval)
        except BaseException as exc:  # surfaced synchronously by finish()
            self._error = exc


def _output_tree_bytes(output_root: Path) -> int:
    output_dirs = [
        path for path in output_root.rglob("out") if path.is_dir() and path.parent.name == "io"
    ]
    if len(output_dirs) != 1:
        raise BenchmarkConfigError(
            f"Efficiency run has {len(output_dirs)} SPATHI output trees; expected exactly one"
        )
    total = sum(path.stat().st_size for path in output_dirs[0].rglob("*") if path.is_file())
    if total < 1:
        raise BenchmarkConfigError("SPATHI output tree is empty")
    return total


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


def _validate_execution_output(output_root: Path, *, dataset_id: str, run_id: str) -> None:
    report_path = unique_direct_manifest(output_root, "run_report.json")
    report = load_json_object(report_path, "efficiency inference report")
    tools = report.get("tools") or {}
    if (
        report.get("status") != "executed"
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
    output_root: Path, *, dataset_id: str, run_id: str
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
    elapsed = report.get("execution", {}).get("elapsed_seconds")
    if (
        report.get("status") != "executed"
        or report.get("run_id") != run_id
        or report.get("dataset", {}).get("id") != dataset_id
        or tools.get("selected") != [run_id]
        or run_id in (tools.get("completed") or [])
        or tools.get("tool_origins", {}).get(run_id) != "custom"
        or not isinstance(tool_status, str)
        or tool_status in {"completed", "completed_with_warnings"}
        or not isinstance(failure_message, str)
        or not failure_message.strip()
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
    ):
        return None
    return {
        "schema_version": "1.0",
        "status": "tool_execution_failed",
        "dataset_id": dataset_id,
        "run_id": run_id,
        "tool_status": tool_status,
        "failure_message": failure_message,
        "run_report_sha256": sha256_file(report_path),
        "elapsed_seconds": float(elapsed),
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
        sampler = _ContainerMemorySampler(
            output_root=output_root,
            image_id=str(task["custom"]["docker_image"]),
        )
        sampler.start()
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
                    str(resources["max_cores"]),
                    "--max-ram-gb",
                    str(resources["max_ram_gb"]),
                    "--planner",
                    str(resources["planner"]),
                    "--planner-time-limit-seconds",
                    str(resources["planner_time_limit_seconds"]),
                ],
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                log_path=log_root / f"{task['dataset_id']}__{task['run']['run_id']}.log",
            )
        except BaseException as primary_error:
            sampler_failed = False
            try:
                sampler.finish()
            except Exception as sampler_error:
                sampler_failed = True
                primary_error.add_note(
                    f"The SPATHI memory sampler also failed during cleanup: {sampler_error}"
                )
            failure_artifact = (
                None
                if sampler_failed
                else _publish_bound_tool_failure(
                    task=task,
                    staging=staging,
                    output_root=output_root,
                    store=store,
                )
            )
            if failure_artifact is not None:
                return failure_artifact
            raise
        peak_memory_bytes, memory_samples = sampler.finish()
        try:
            _validate_execution_output(
                output_root,
                dataset_id=task["dataset_id"],
                run_id=task["run"]["run_id"],
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
        _write_json(
            staging / "efficiency-resource-measurement.json",
            {
                "schema_version": "1.0",
                "measurement_scope": "spathi_container_and_declared_outputs",
                "peak_memory": {
                    "bytes": peak_memory_bytes,
                    "method": "docker_stats_sampled_container_memory",
                    "sample_interval_milliseconds": _MEMORY_SAMPLE_INTERVAL_MILLISECONDS,
                    "samples": memory_samples,
                },
                "output": {
                    "bytes": _output_tree_bytes(output_root),
                    "method": "exact_spathi_output_tree",
                },
            },
        )
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


def run_noninferiority_artifacts(
    *,
    repo_root: Path,
    plan_path: Path,
    andrea_repo: Path,
    artifact_index_path: Path,
    artifact_store_path: Path | None = None,
    max_parallel_runs: int = 1,
    max_cores: int = 1,
    max_ram_gb: float = 4.0,
    planner: str = "heuristic",
    planner_time_limit_seconds: int = 30,
) -> Path:
    """Run or resume the exact reference/candidate product declared by a plan."""

    if max_parallel_runs != 1:
        raise BenchmarkConfigError(
            "Efficiency runtime gates require --max-parallel-runs=1 to avoid contention"
        )
    if max_cores < 1 or max_ram_gb <= 0:
        raise BenchmarkConfigError("Efficiency resources must be positive")
    if planner not in {"auto", "cp_sat", "heuristic"} or planner_time_limit_seconds < 1:
        raise BenchmarkConfigError("Invalid efficiency planner resources")
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
        "max_parallel_runs": max_parallel_runs,
        "max_cores": max_cores,
        "max_ram_gb": max_ram_gb,
        "planner": planner,
        "planner_time_limit_seconds": planner_time_limit_seconds,
    }
    measurement_protocol: dict[str, Any] = {
        "peak_memory": "docker_stats_sampled_container_memory",
        "sample_interval_milliseconds": _MEMORY_SAMPLE_INTERVAL_MILLISECONDS,
        "output_bytes": "exact_spathi_output_tree",
        "model_counts": "validated_spathi_run_metadata",
        "tree_counts": "validated_spathi_model_diagnostics",
    }
    runner_sha256 = _runner_sha256()
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
                spec = _inference_spec(
                    dataset=dataset,
                    dataset_id=dataset_id,
                    andrea_commit=str(observed["commit"]),
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
        # max_parallel_runs is deliberately fixed to one above.  Executing the
        # list directly preserves the frozen within-block order and, unlike a
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
            "andrea_commit": str(observed["commit"]),
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
