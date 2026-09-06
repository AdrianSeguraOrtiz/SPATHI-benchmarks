"""Post-efficiency SPATHI computational scalability workflow.

This workflow deliberately answers a narrower question than the simulator-based
accuracy experiments: can the frozen production policy complete on a rounded,
author-defined public scale grid, and what resources does it use?  The scale-stress
generator is therefore treated as a deterministic stressor, not as a biological
model or a source of confirmatory accuracy evidence.
"""

from __future__ import annotations

import copy
import math
import os
import platform
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from .artifacts import Artifact, ArtifactKey, ArtifactStore, canonical_json_sha256, sha256_file
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    load_profile,
    validate_json,
    validate_profile_references,
)
from .pipeline import (
    InferenceExecutionFailure,
    build_spathi_image,
    generate_data,
    preflight,
    run_inference,
)
from .production_policy import completed_production_config
from .provenance import observe_benchmark_repository
from .report_publication import publish_workflow_report
from .reporting import render_scalability_report
from .runtime import (
    andrea_host_environment,
    benchmark_host_environment,
    docker_runtime_environment,
)
from .safe_io import path_below_no_symlinks, write_idempotent_json
from .seeds import derive_plan_seed, load_seed_plan
from .state import load_state, run_lock, status_at_least, write_state

_IDENTITY_INPUTS = (
    "experiments/v1/seeds/seed-plan.json",
    "images.lock.json",
    "repos.lock.json",
    "schemas/scale-stress-panel.schema.json",
    "schemas/profile.schema.json",
    "schemas/scalability-results.schema.json",
    "schemas/spathi-production-config.schema.json",
    "src/spathi_bench/scale_stress.py",
    "src/spathi_bench/pipeline.py",
    "src/spathi_bench/production_policy.py",
    "src/spathi_bench/provenance.py",
    "src/spathi_bench/scalability_workflow.py",
    "src/spathi_bench/report_publication.py",
    "src/spathi_bench/safe_io.py",
    "src/spathi_bench/reporting/assets/scalability.js",
    "src/spathi_bench/reporting/scalability.py",
    "src/spathi_bench/reporting/templates/scalability.html.j2",
    "src/spathi_bench/runtime.py",
)
_RSS_SAMPLE_INTERVAL_SECONDS = 0.1
_MEASUREMENT_STATE_FIELD = "scalability_measurement"


def _host_resources() -> dict[str, Any]:
    """Return the stable host facts required to interpret resource measurements."""

    cpu_model = platform.processor().strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    if not cpu_model:
        cpu_model = "unreported"
    memory_bytes = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                fields = line.split()
                if len(fields) >= 2:
                    memory_bytes = int(fields[1]) * 1024
                break
    if memory_bytes < 1:
        raise BenchmarkConfigError("Cannot determine total host memory for scalability evidence")
    logical_cpus = os.cpu_count()
    if logical_cpus is None or logical_cpus < 1:
        raise BenchmarkConfigError("Cannot determine logical CPU count for scalability evidence")
    return {
        "system": platform.system(),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": logical_cpus,
        "host_memory_bytes": memory_bytes,
    }


def _relative(repo_root: Path, path: Path, label: str) -> str:
    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    absolute = path_below_no_symlinks(path, root, label)
    return absolute.relative_to(root).as_posix()


def _measurement_reference(repo_root: Path, measurement_path: Path) -> dict[str, str]:
    return {
        "path": _relative(repo_root, measurement_path, "scalability measurement"),
        "sha256": sha256_file(measurement_path),
    }


def _publish_scalability_measurement(
    *,
    repo_root: Path,
    run_root: Path,
    profile_id: str,
    runtime_state: dict[str, Any],
    measurement_path: Path,
    measurement: Mapping[str, Any],
    dataset_id: str,
) -> Path:
    """Publish a measurement and immediately bind its digest into runtime state."""

    published = write_idempotent_json(
        measurement_path,
        measurement,
        f"{dataset_id} measurement",
        boundary=repo_root,
    )
    reference = _measurement_reference(repo_root, published)
    if _MEASUREMENT_STATE_FIELD in runtime_state:
        if runtime_state[_MEASUREMENT_STATE_FIELD] != reference:
            raise BenchmarkConfigError(
                f"Runtime state already binds a different scalability measurement for {dataset_id}"
            )
    else:
        runtime_state[_MEASUREMENT_STATE_FIELD] = reference
    write_state(run_root, runtime_state)
    reloaded = load_state(run_root, profile_id)
    if reloaded.get(_MEASUREMENT_STATE_FIELD) != reference:
        raise BenchmarkConfigError(
            f"Runtime state did not persist the scalability measurement for {dataset_id}"
        )
    return published


def _load_bound_scalability_measurement(
    *,
    repo_root: Path,
    runtime_state: Mapping[str, Any],
    measurement_path: Path,
    dataset_id: str,
) -> dict[str, Any]:
    """Load a measurement only when runtime state binds its exact path and content."""

    reference = runtime_state.get(_MEASUREMENT_STATE_FIELD)
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise BenchmarkConfigError(
            f"Scalability measurement for {dataset_id} is orphaned from its runtime state"
        )
    expected_relative = _relative(
        repo_root,
        measurement_path,
        f"{dataset_id} scalability measurement",
    )
    if reference.get("path") != expected_relative:
        raise BenchmarkConfigError(
            f"Runtime state binds a different scalability measurement path for {dataset_id}"
        )
    safe_measurement = path_below_no_symlinks(
        measurement_path,
        repo_root,
        f"{dataset_id} scalability measurement",
        require_file=True,
    )
    if reference.get("sha256") != sha256_file(safe_measurement):
        raise BenchmarkConfigError(
            f"Runtime-bound scalability measurement changed for {dataset_id}"
        )
    return load_json_object(safe_measurement, f"{dataset_id} scalability measurement")


def _algorithm_seed(repo_root: Path, *, budget: str, dataset_id: str) -> tuple[int, dict[str, Any]]:
    derivation = {
        "namespace": f"{budget}.scalability",
        "stream": "inference",
        "coordinates": {
            "purpose": "scalability",
            "dataset_id": dataset_id,
            "replicate_index": 1,
        },
    }
    plan = load_seed_plan(repo_root / "experiments/v1/seeds/seed-plan.json")
    seed = derive_plan_seed(
        plan,
        namespace=derivation["namespace"],
        stream=derivation["stream"],
        coordinates=derivation["coordinates"],
    )
    return seed, derivation


def _runtime_profile_id(budget: str, dataset_id: str) -> str:
    profile_id = f"scalability_{budget}_{dataset_id}"
    if len(profile_id) > 120:
        profile_id = f"scalability_{budget}_{canonical_json_sha256(dataset_id)[:20]}"
    return profile_id


def materialize_scalability_dataset(
    *,
    repo_root: Path,
    budget: str,
    template_profile: Mapping[str, Any],
    production_config_path: Path,
    production: Mapping[str, Any],
    panel: Mapping[str, Any],
    dataset: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path | str | int | dict[str, Any]]:
    """Materialize one runtime profile so resources remain attributable per shape."""

    root = repo_root.resolve()
    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError("Scalability budget must be 'reduced' or 'full'")
    expected_template_id = f"scalability-{budget}"
    if (
        template_profile.get("id") != expected_template_id
        or template_profile.get("purpose") != "scalability"
        or template_profile.get("status") != "runnable"
    ):
        raise BenchmarkConfigError("Scalability template does not match the requested budget")
    generation_jobs = template_profile.get("generation_jobs")
    if not isinstance(generation_jobs, list) or len(generation_jobs) != 1:
        raise BenchmarkConfigError("Scalability template must define one generator request")
    policy = production.get("policy")
    if not isinstance(policy, Mapping) or not isinstance(policy.get("group_policy"), Mapping):
        raise BenchmarkConfigError("Production config contains no executable group policy")
    if production.get("budget") != budget:
        raise BenchmarkConfigError("Production config budget differs from scalability budget")
    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise BenchmarkConfigError("Scalability dataset has no canonical ID")
    if dataset not in panel.get("datasets", []):
        raise BenchmarkConfigError("Scalability dataset is not a member of the declared panel")

    runtime_dir = path_below_no_symlinks(
        output_dir,
        root / "work",
        "Scalability materialization",
    )
    cells = sum(int(group["cells"]) for group in dataset["groups"])
    scale_grid = {
        "provenance": "author_defined_public_synthetic_grid",
        "cells": [cells],
        "genes": [int(dataset["genes"])],
        "transcription_factors": [int(dataset["tfs"])],
    }
    request = {
        "schema_version": panel["schema_version"],
        "id": f"{panel['id']}__{dataset_id}",
        "description": (
            f"Single-shape materialization of {dataset_id} from {panel['id']}; "
            "computational scale stress only."
        ),
        "seed_plan": panel["seed_plan"],
        "scale_grid": scale_grid,
        "expression_preparation": copy.deepcopy(dict(panel["expression_preparation"])),
        "datasets": [copy.deepcopy(dict(dataset))],
    }
    request_path = write_idempotent_json(
        runtime_dir / "generator-request.json",
        request,
        "scalability generator request",
        boundary=root,
    )
    seed, seed_derivation = _algorithm_seed(root, budget=budget, dataset_id=dataset_id)
    run_id = f"spathi_scalability_{dataset_id}"
    tools_params = {
        "runs": [
            {
                "run_id": run_id,
                "tool_id": f"custom_{run_id}",
                "params": {
                    "group_policy": copy.deepcopy(dict(policy["group_policy"])),
                    "random_seed": seed,
                },
                "execution": {"mode": "group_native"},
            }
        ]
    }
    custom_tools = {
        "tools": [
            {
                "run_id": run_id,
                "name": f"SPATHI production policy on {dataset_id}",
                "docker_image": load_image_lock(root)["images"]["spathi"]["tag"],
                "execution_mode": "group_native",
                "extra_inputs": ["groups", "tf_list"],
                "outputs": {"directed": True, "sign": "none"},
            }
        ]
    }
    metadata = {
        "schema_version": "1.0",
        "runs": [
            {
                "run_id": run_id,
                "configuration_id": policy["policy_id"],
                "selection_role": "selected",
                "algorithm_seed": seed,
                "algorithm_seed_parameter": "random_seed",
                "seed_derivation": seed_derivation,
                "family": "spathi",
                "notes": ["Frozen production policy; computational scale stress only."],
            }
        ],
    }
    tools_path = write_idempotent_json(
        runtime_dir / "tools-params.json", tools_params, "tools request", boundary=root
    )
    custom_path = write_idempotent_json(
        runtime_dir / "custom-tools.json", custom_tools, "custom tools", boundary=root
    )
    metadata_path = write_idempotent_json(
        runtime_dir / "run-metadata.json", metadata, "run metadata", boundary=root
    )
    generation_job_id = f"generate_{dataset_id}"
    profile_id = _runtime_profile_id(budget, dataset_id)
    profile = copy.deepcopy(dict(template_profile))
    profile.update(
        {
            "id": profile_id,
            "description": (
                f"Runtime scale-stress profile for {dataset_id}; executes only the frozen "
                "production SPATHI policy."
            ),
            "production_config": _relative(root, production_config_path, "production config"),
            "generation_jobs": [
                {
                    "id": generation_job_id,
                    "generator_request": _relative(root, request_path, "generator request"),
                    "dataset_role": "scalability",
                    "split_id": f"scalability_{budget}_v1",
                    "condition_id": "dimensional_scale_stress",
                }
            ],
            "inference_batches": [
                {
                    "id": f"infer_{dataset_id}",
                    "kind": "spathi",
                    "dataset_role": "scalability",
                    "generation_jobs": [generation_job_id],
                    "regime": "group_only",
                    "tools_params": _relative(root, tools_path, "tools request"),
                    "custom_tools": _relative(root, custom_path, "custom tools"),
                    "run_metadata": _relative(root, metadata_path, "run metadata"),
                }
            ],
        }
    )
    profile_path = write_idempotent_json(
        runtime_dir / "profile.json", profile, "runtime profile", boundary=root
    )
    validate_json(profile, root / "schemas/profile.schema.json", "scalability runtime profile")
    validate_profile_references(root, profile)
    return {
        "profile_id": profile_id,
        "profile_path": profile_path,
        "request_path": request_path,
        "tools_path": tools_path,
        "custom_tools_path": custom_path,
        "metadata_path": metadata_path,
        "run_id": run_id,
        "algorithm_seed": seed,
        "seed_derivation": seed_derivation,
    }


def _run_command(arguments: Sequence[str], *, timeout: float = 5.0) -> str:
    completed = subprocess.run(
        list(arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _rss_from_process_table(root_pids: Sequence[int], raw_table: str) -> int:
    """Sum RSS across each container process tree from a stable ``ps`` snapshot."""

    rows: dict[int, tuple[int, int]] = {}
    children: dict[int, list[int]] = {}
    for line in raw_table.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, parent, rss_kib = (int(field) for field in fields)
        except ValueError:
            continue
        if pid < 1 or parent < 0 or rss_kib < 0:
            continue
        rows[pid] = (parent, rss_kib)
        children.setdefault(parent, []).append(pid)
    pending = list(root_pids)
    observed: set[int] = set()
    total_kib = 0
    while pending:
        pid = pending.pop()
        if pid in observed:
            continue
        observed.add(pid)
        record = rows.get(pid)
        if record is not None:
            total_kib += record[1]
        pending.extend(children.get(pid, ()))
    return total_kib * 1024


class _PeakRssSampler:
    """Measure peak host RSS of the unambiguous SPATHI container process tree."""

    def __init__(self, image_id: str) -> None:
        self.image_id = image_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[int] = []
        self._error: BaseException | None = None

    def _container_ids(self) -> tuple[str, ...]:
        raw = _run_command(
            [
                "docker",
                "ps",
                "--no-trunc",
                "--filter",
                f"ancestor={self.image_id}",
                "--format",
                "{{.ID}}",
            ]
        )
        return tuple(item for item in raw.splitlines() if item)

    def start(self) -> None:
        existing = self._container_ids()
        if existing:
            raise BenchmarkConfigError(
                "Cannot attribute scalability RSS while another SPATHI container is running"
            )
        self._thread = threading.Thread(
            target=self._sample,
            name="spathi-scalability-rss",
            daemon=True,
        )
        self._thread.start()

    def _sample(self) -> None:
        try:
            while not self._stop.is_set():
                container_ids = self._container_ids()
                if len(container_ids) > 1:
                    raise BenchmarkConfigError(
                        "Several SPATHI containers are running; peak RSS is not attributable"
                    )
                roots: list[int] = []
                for container_id in container_ids:
                    raw_pid = _run_command(
                        ["docker", "inspect", "--format", "{{.State.Pid}}", container_id]
                    )
                    try:
                        pid = int(raw_pid)
                    except ValueError:
                        continue
                    if pid > 0:
                        roots.append(pid)
                if roots:
                    table = _run_command(["ps", "-e", "-o", "pid=,ppid=,rss="])
                    rss = _rss_from_process_table(roots, table)
                    if rss > 0:
                        self._samples.append(rss)
                self._stop.wait(_RSS_SAMPLE_INTERVAL_SECONDS)
        except BaseException as exc:  # surfaced by finish()
            self._error = exc

    def finish(self, *, require_samples: bool) -> tuple[int, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise BenchmarkConfigError("SPATHI RSS sampler did not terminate")
        if self._error is not None:
            raise BenchmarkConfigError(f"Cannot measure SPATHI peak RSS: {self._error}")
        if not self._samples:
            if require_samples:
                raise BenchmarkConfigError(
                    "No SPATHI RSS sample was captured. A cached inference cannot establish "
                    "fresh scalability evidence; remove its inference artifact and resume."
                )
            return 0, 0
        return max(self._samples), len(self._samples)


def _artifact_from_reference(
    repo_root: Path, state: Mapping[str, Any], reference: Mapping[str, Any]
) -> Artifact:
    try:
        key = ArtifactKey(kind=str(reference["kind"]), digest=str(reference["digest"]))
        store_path = path_below_no_symlinks(
            Path(str(state["artifact_store"])),
            repo_root / "work",
            "Runtime artifact store",
            require_directory=True,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkConfigError("Runtime state contains an invalid artifact reference") from exc
    store = ArtifactStore(store_path)
    artifact = store.validate(key)
    recorded_path = path_below_no_symlinks(
        Path(str(reference.get("path", ""))),
        artifact.path.parent,
        "Runtime artifact reference path",
        require_directory=True,
    )
    if reference.get("key") != key.value or recorded_path != artifact.path:
        raise BenchmarkConfigError("Runtime artifact reference differs from its content address")
    return artifact


def _single_manifest_member(artifact: Artifact, suffix: str, label: str) -> tuple[str, Path]:
    matches = [
        str(entry["path"])
        for entry in artifact.manifest["files"]
        if entry["path"] == suffix or str(entry["path"]).endswith("/" + suffix)
    ]
    if len(matches) != 1:
        raise BenchmarkConfigError(
            f"Inference artifact contains {len(matches)} {label} files; expected exactly one"
        )
    return matches[0], artifact.file(matches[0])


def _tree_bytes(artifact: Artifact, prefix: str | None = None) -> int:
    total = sum(
        int(entry["size"])
        for entry in artifact.manifest["files"]
        if prefix is None or str(entry["path"]).startswith(prefix)
    )
    if total < 1:
        raise BenchmarkConfigError("Measured artifact tree is empty")
    return total


def _required_int(mapping: Mapping[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BenchmarkConfigError(f"{label}.{key} must be an integer >= 0")
    return value


def _required_time(mapping: Mapping[str, Any], key: str, label: str) -> float:
    value = mapping.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise BenchmarkConfigError(f"{label}.{key} must be a finite number >= 0")
    return float(value)


def _expected_dimensions(dataset_spec: Mapping[str, Any]) -> dict[str, int]:
    return {
        "cells": sum(int(group["cells"]) for group in dataset_spec["groups"]),
        "genes": int(dataset_spec["genes"]),
        "groups": len(dataset_spec["groups"]),
        "transcription_factors": int(dataset_spec["tfs"]),
        "targets": int(dataset_spec["genes"]),
    }


def _execution_limits(resources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "max_parallel_runs": int(resources["max_parallel_runs"]),
        "max_cores": int(resources["max_cores"]),
        "max_ram_gb": float(resources["max_ram_gb"]),
        "planner": str(resources["planner"]),
        "planner_time_limit_seconds": float(resources["planner_time_limit_seconds"]),
        "inference_wall_time_limit_seconds": None,
    }


def _file_reference(repo_root: Path, path: Path, label: str) -> dict[str, Any]:
    safe_path = path_below_no_symlinks(
        path,
        repo_root,
        label,
        require_file=True,
    )
    return {
        "path": _relative(repo_root, safe_path, label),
        "sha256": sha256_file(safe_path),
        "size_bytes": safe_path.stat().st_size,
    }


def _failure_kind(failure: InferenceExecutionFailure) -> str:
    """Classify only failure modes exposed by a validated ANDREA tool result."""

    text = "\n".join(
        item for item in (failure.tool_error, failure.tool_log_tail or "") if item
    ).lower()
    oom_markers = (
        "out of memory",
        "out-of-memory",
        "oomkilled",
        "oom killed",
        "cannot allocate memory",
        "memoryerror",
    )
    timeout_markers = (
        "timed out",
        "timeout exceeded",
        "time limit exceeded",
        "deadline exceeded",
    )
    infrastructure_markers = (
        "cannot connect to the docker daemon",
        "error response from daemon",
        "failed while collecting container outputs",
        "no such image",
        "mounts denied",
        "permission denied while trying to connect to the docker daemon",
    )
    if failure.tool_exit_code in {-9, 137} or any(marker in text for marker in oom_markers):
        return "oom"
    if failure.tool_exit_code == 124 or any(marker in text for marker in timeout_markers):
        return "timeout"
    if any(marker in text for marker in infrastructure_markers):
        return "infrastructure_error"
    return "tool_error"


def _failed_generation_evidence(
    *,
    repo_root: Path,
    runtime_state: Mapping[str, Any],
    dataset_spec: Mapping[str, Any],
    generation_wall_time_seconds: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    if runtime_state.get("status") != "generated":
        raise BenchmarkConfigError("Failed scalability evidence requires generated pipeline state")
    datasets = runtime_state.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise BenchmarkConfigError("Per-shape scalability run must contain exactly one dataset")
    dataset_record = datasets[0]
    dataset_id = str(dataset_spec["id"])
    if dataset_record.get("dataset_id") != dataset_id:
        raise BenchmarkConfigError("Failed scalability state has the wrong dataset identity")
    dataset_artifact = _artifact_from_reference(
        repo_root, runtime_state, dataset_record["dataset_artifact"]
    )
    _expression_relative, expression_path = _single_manifest_member(
        dataset_artifact,
        f"datasets/{dataset_id}/expression.tsv",
        "expression",
    )
    if not math.isfinite(float(generation_wall_time_seconds)) or generation_wall_time_seconds < 0:
        raise BenchmarkConfigError("Generation wall time must be finite and non-negative")
    return (
        {
            "wall_time_seconds": float(generation_wall_time_seconds),
            "cache_hit": bool(dataset_record["dataset_artifact"].get("cache_hit")),
            "dataset_artifact_bytes": _tree_bytes(dataset_artifact),
            "expression_bytes": expression_path.stat().st_size,
        },
        {"kind": dataset_artifact.key.kind, "digest": dataset_artifact.key.digest},
    )


def collect_failed_scalability_measurement(
    *,
    repo_root: Path,
    runtime_state: Mapping[str, Any],
    dataset_spec: Mapping[str, Any],
    run_id: str,
    algorithm_seed: int,
    seed_derivation: Mapping[str, Any],
    resources: Mapping[str, Any],
    generation_wall_time_seconds: float,
    inference_wall_time_seconds: float,
    peak_rss_bytes: int,
    rss_samples: int,
    failure: InferenceExecutionFailure,
) -> dict[str, Any]:
    """Record an attributable failed SPATHI run as scale evidence."""

    dataset_id = str(dataset_spec["id"])
    if failure.dataset_id != dataset_id or failure.run_id != run_id:
        raise BenchmarkConfigError("Failed inference attribution differs from the scale request")
    if (
        not math.isfinite(float(inference_wall_time_seconds))
        or inference_wall_time_seconds < 0
        or peak_rss_bytes < 0
        or rss_samples < 0
        or (peak_rss_bytes == 0) != (rss_samples == 0)
    ):
        raise BenchmarkConfigError("Failed inference resource measurements are inconsistent")
    generation, dataset_artifact = _failed_generation_evidence(
        repo_root=repo_root,
        runtime_state=runtime_state,
        dataset_spec=dataset_spec,
        generation_wall_time_seconds=generation_wall_time_seconds,
    )
    message = failure.tool_error.strip()[-4096:]
    return {
        "dataset_id": dataset_id,
        "status": "failed",
        "network_mode": dataset_spec["network_mode"],
        "policy_branch": "single_group" if len(dataset_spec["groups"]) == 1 else "multi_group",
        "generation_seed": dataset_spec["seed"],
        "algorithm_seed": algorithm_seed,
        "seed_derivation": dict(seed_derivation),
        "dimensions": _expected_dimensions(dataset_spec),
        "execution_limits": _execution_limits(resources),
        "generation": generation,
        "failure": {
            "failure_kind": _failure_kind(failure),
            "stage": "inference",
            "message": message,
            "orchestrator_exit_code": failure.orchestrator_return_code,
            "tool_exit_code": failure.tool_exit_code,
        },
        "observations": {
            "harness_wall_time_seconds": float(inference_wall_time_seconds),
            "tool_wall_time_seconds": failure.tool_duration_seconds,
            "peak_rss_bytes": peak_rss_bytes if peak_rss_bytes else None,
            "rss_samples": rss_samples,
        },
        "diagnostics": {
            "command_log": _file_reference(
                repo_root, failure.command_log_path, "failed inference command log"
            ),
            "progress": copy.deepcopy(failure.progress),
            "tool_log_tail": failure.tool_log_tail,
        },
        "dataset_artifact": dataset_artifact,
    }


def validate_failed_scalability_measurement(
    *,
    repo_root: Path,
    runtime_state: Mapping[str, Any],
    dataset_spec: Mapping[str, Any],
    algorithm_seed: int,
    seed_derivation: Mapping[str, Any],
    resources: Mapping[str, Any],
    recorded: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate immutable failure evidence during deterministic resume."""

    if recorded.get("status") != "failed":
        raise BenchmarkConfigError("Expected a failed scalability measurement")
    generation = recorded.get("generation")
    if not isinstance(generation, Mapping):
        raise BenchmarkConfigError("Failed scalability measurement has no generation evidence")
    observed_generation, dataset_artifact = _failed_generation_evidence(
        repo_root=repo_root,
        runtime_state=runtime_state,
        dataset_spec=dataset_spec,
        generation_wall_time_seconds=_required_time(
            generation, "wall_time_seconds", "failed generation"
        ),
    )
    expected_common = {
        "dataset_id": str(dataset_spec["id"]),
        "status": "failed",
        "network_mode": dataset_spec["network_mode"],
        "policy_branch": "single_group" if len(dataset_spec["groups"]) == 1 else "multi_group",
        "generation_seed": dataset_spec["seed"],
        "algorithm_seed": algorithm_seed,
        "seed_derivation": dict(seed_derivation),
        "dimensions": _expected_dimensions(dataset_spec),
        "execution_limits": _execution_limits(resources),
        "generation": observed_generation,
        "dataset_artifact": dataset_artifact,
    }
    for field, expected in expected_common.items():
        if recorded.get(field) != expected:
            raise BenchmarkConfigError(
                f"Recorded failed scalability measurement changed field {field!r}"
            )
    diagnostics = recorded.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise BenchmarkConfigError("Failed scalability measurement has no diagnostics")
    log_reference = diagnostics.get("command_log")
    if not isinstance(log_reference, Mapping):
        raise BenchmarkConfigError("Failed scalability measurement has no command log")
    log_path = repo_root / str(log_reference.get("path", ""))
    if _file_reference(repo_root, log_path, "failed inference command log") != log_reference:
        raise BenchmarkConfigError("Failed inference command log changed after publication")
    return dict(recorded)


def collect_scalability_measurement(
    *,
    repo_root: Path,
    runtime_state: Mapping[str, Any],
    dataset_spec: Mapping[str, Any],
    run_id: str,
    algorithm_seed: int,
    seed_derivation: Mapping[str, Any],
    resources: Mapping[str, Any],
    generation_wall_time_seconds: float,
    inference_wall_time_seconds: float,
    peak_rss_bytes: int,
    rss_samples: int,
) -> dict[str, Any]:
    """Validate one completed run and reduce its audit artifacts to resource evidence."""

    if runtime_state.get("status") != "inferred":
        raise BenchmarkConfigError("Scalability measurement requires inferred pipeline state")
    datasets = runtime_state.get("datasets")
    inferences = runtime_state.get("inference_runs")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise BenchmarkConfigError("Per-shape scalability run must contain exactly one dataset")
    if not isinstance(inferences, list) or len(inferences) != 1:
        raise BenchmarkConfigError("Per-shape scalability run must contain exactly one inference")
    dataset_record = datasets[0]
    inference_record = inferences[0]
    dataset_id = str(dataset_spec["id"])
    if (
        dataset_record.get("dataset_id") != dataset_id
        or inference_record.get("dataset_id") != dataset_id
    ):
        raise BenchmarkConfigError("Scalability runtime state has the wrong dataset identity")
    if inference_record.get("run_id") != run_id:
        raise BenchmarkConfigError("Scalability runtime state has the wrong SPATHI run identity")
    if inference_record.get("algorithm_seed") != algorithm_seed:
        raise BenchmarkConfigError("Scalability runtime state has the wrong algorithm seed")
    dataset_artifact = _artifact_from_reference(
        repo_root, runtime_state, dataset_record["dataset_artifact"]
    )
    inference_artifact = _artifact_from_reference(
        repo_root, runtime_state, inference_record["artifact"]
    )
    _expression_relative, expression_path = _single_manifest_member(
        dataset_artifact,
        f"datasets/{dataset_id}/expression.tsv",
        "expression",
    )
    metadata_relative, metadata_path = _single_manifest_member(
        inference_artifact,
        f"tools/{run_id}/io/out/run_metadata.json",
        "SPATHI run_metadata.json",
    )
    _report_relative, report_path = _single_manifest_member(
        inference_artifact, "run_report.json", "ANDREA run_report.json"
    )
    metadata = load_json_object(metadata_path, "SPATHI run metadata")
    report = load_json_object(report_path, "ANDREA inference report")
    if metadata.get("status") != "complete":
        raise BenchmarkConfigError("SPATHI run metadata is not complete")
    if report.get("status") != "executed":
        raise BenchmarkConfigError("ANDREA inference report is not executed")
    dimensions = metadata.get("input_dimensions")
    models = metadata.get("models")
    effective = metadata.get("effective_parameters")
    phases = metadata.get("phase_times_seconds")
    parallelism = metadata.get("parallelism")
    memory_estimate = metadata.get("memory_estimate_bytes")
    if not all(
        isinstance(item, Mapping)
        for item in (dimensions, models, effective, phases, parallelism, memory_estimate)
    ):
        raise BenchmarkConfigError("SPATHI run metadata lacks resource-audit sections")
    assert isinstance(dimensions, Mapping)
    assert isinstance(models, Mapping)
    assert isinstance(effective, Mapping)
    assert isinstance(phases, Mapping)
    assert isinstance(parallelism, Mapping)
    assert isinstance(memory_estimate, Mapping)
    target_eligibility = effective.get("target_eligibility")
    if not isinstance(target_eligibility, Mapping):
        raise BenchmarkConfigError("SPATHI run metadata lacks target eligibility counts")
    observed_dimensions = {
        key: _required_int(dimensions, key, "input_dimensions")
        for key in ("cells", "genes", "groups", "transcription_factors", "targets")
    }
    expected_dimensions = _expected_dimensions(dataset_spec)
    if observed_dimensions != expected_dimensions:
        raise BenchmarkConfigError(
            f"SPATHI dimensions differ from the scale request: {observed_dimensions} != "
            f"{expected_dimensions}"
        )
    targets_eligible = _required_int(
        target_eligibility, "globally_eligible_targets", "target_eligibility"
    )
    targets_ineligible = _required_int(
        target_eligibility, "globally_ineligible_targets", "target_eligibility"
    )
    if targets_eligible + targets_ineligible != observed_dimensions["targets"]:
        raise BenchmarkConfigError("SPATHI target eligibility counts are incomplete")
    models_requested = _required_int(models, "requested", "models")
    models_completed = _required_int(models, "completed", "models")
    models_trained = _required_int(models, "trained", "models")
    models_failed = _required_int(models, "fit_or_importance_failures", "models")
    trees_fitted = _required_int(models, "fitted_estimators_total", "models")
    threads_requested = _required_int(parallelism, "threads_requested", "parallelism")
    threads_effective = _required_int(parallelism, "threads_effective", "parallelism")
    threads_available = _required_int(parallelism, "threads_available", "parallelism")
    estimated_peak_heap = _required_int(
        memory_estimate,
        "estimated_peak_heap_with_rough_trees",
        "memory_estimate_bytes",
    )
    if models_completed != models_requested or models_trained > models_completed or models_failed:
        raise BenchmarkConfigError("SPATHI model completion accounting is not publishable")
    report_execution = report.get("execution")
    if not isinstance(report_execution, Mapping):
        raise BenchmarkConfigError("ANDREA report lacks execution accounting")
    andrea_seconds = _required_time(report_execution, "elapsed_seconds", "execution")
    spathi_seconds = _required_time(phases, "total", "phase_times_seconds")
    cache_hit = bool(inference_record["artifact"].get("cache_hit"))
    if cache_hit:
        raise BenchmarkConfigError("Fresh scalability resource evidence cannot come from cache")
    if peak_rss_bytes < 1 or rss_samples < 1:
        raise BenchmarkConfigError("Scalability evidence requires observed peak RSS samples")
    output_prefix = metadata_relative.removesuffix("run_metadata.json")
    policy_branch = "single_group" if observed_dimensions["groups"] == 1 else "multi_group"
    return {
        "dataset_id": dataset_id,
        "status": "completed",
        "network_mode": dataset_spec["network_mode"],
        "policy_branch": policy_branch,
        "generation_seed": dataset_spec["seed"],
        "algorithm_seed": algorithm_seed,
        "seed_derivation": dict(seed_derivation),
        "dimensions": observed_dimensions,
        "execution_limits": _execution_limits(resources),
        "generation": {
            "wall_time_seconds": float(generation_wall_time_seconds),
            "cache_hit": bool(dataset_record["dataset_artifact"].get("cache_hit")),
            "dataset_artifact_bytes": _tree_bytes(dataset_artifact),
            "expression_bytes": expression_path.stat().st_size,
        },
        "inference": {
            "andrea_wall_time_seconds": andrea_seconds,
            "spathi_wall_time_seconds": spathi_seconds,
            "harness_wall_time_seconds": float(inference_wall_time_seconds),
            "peak_rss_bytes": peak_rss_bytes,
            "rss_samples": rss_samples,
            "cache_hit": False,
            "inference_artifact_bytes": _tree_bytes(inference_artifact),
            "spathi_output_bytes": _tree_bytes(inference_artifact, output_prefix),
            "targets_requested": observed_dimensions["targets"],
            "targets_eligible": targets_eligible,
            "targets_ineligible": targets_ineligible,
            "models_requested": models_requested,
            "models_completed": models_completed,
            "models_trained": models_trained,
            "models_failed": models_failed,
            "trees_fitted": trees_fitted,
            "threads_requested": threads_requested,
            "threads_effective": threads_effective,
            "threads_available": threads_available,
            "estimated_peak_heap_with_rough_trees_bytes": estimated_peak_heap,
        },
        "dataset_artifact": {
            "kind": dataset_artifact.key.kind,
            "digest": dataset_artifact.key.digest,
        },
        "inference_artifact": {
            "kind": inference_artifact.key.kind,
            "digest": inference_artifact.key.digest,
        },
        "run_metadata": {
            "path": _relative(repo_root, metadata_path, "SPATHI run metadata"),
            "sha256": sha256_file(metadata_path),
        },
    }


def _run_one_dataset(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    budget: str,
    flow_id: str,
    template_profile: Mapping[str, Any],
    production_path: Path,
    production: Mapping[str, Any],
    panel: Mapping[str, Any],
    dataset: Mapping[str, Any],
    workflow_root: Path,
) -> dict[str, Any]:
    dataset_id = str(dataset["id"])
    materialization = materialize_scalability_dataset(
        repo_root=repo_root,
        budget=budget,
        template_profile=template_profile,
        production_config_path=production_path,
        production=production,
        panel=panel,
        dataset=dataset,
        output_dir=workflow_root / "materialized" / dataset_id,
    )
    profile_id = str(materialization["profile_id"])
    profile_path = Path(materialization["profile_path"])
    run_root = work_root / "v1" / profile_id / flow_id
    measurement_path = workflow_root / "measurements" / f"{dataset_id}.json"
    inference_resources = template_profile["resources"]["infer_network"]
    runtime_state_path = run_root / "state.json"
    measurement_present = measurement_path.exists() or measurement_path.is_symlink()
    runtime_state = load_state(run_root, profile_id) if runtime_state_path.is_file() else None
    if measurement_present:
        if runtime_state is None:
            raise BenchmarkConfigError(
                f"Scalability measurement for {dataset_id} is orphaned from its runtime state"
            )
        recorded = _load_bound_scalability_measurement(
            repo_root=repo_root,
            runtime_state=runtime_state,
            measurement_path=measurement_path,
            dataset_id=dataset_id,
        )
        if recorded.get("status") == "completed":
            observed = collect_scalability_measurement(
                repo_root=repo_root,
                runtime_state=runtime_state,
                dataset_spec=dataset,
                run_id=str(materialization["run_id"]),
                algorithm_seed=int(materialization["algorithm_seed"]),
                seed_derivation=materialization["seed_derivation"],
                resources=inference_resources,
                generation_wall_time_seconds=float(recorded["generation"]["wall_time_seconds"]),
                inference_wall_time_seconds=float(
                    recorded["inference"]["harness_wall_time_seconds"]
                ),
                peak_rss_bytes=int(recorded["inference"]["peak_rss_bytes"]),
                rss_samples=int(recorded["inference"]["rss_samples"]),
            )
        elif recorded.get("status") == "failed":
            observed = validate_failed_scalability_measurement(
                repo_root=repo_root,
                runtime_state=runtime_state,
                dataset_spec=dataset,
                algorithm_seed=int(materialization["algorithm_seed"]),
                seed_derivation=materialization["seed_derivation"],
                resources=inference_resources,
                recorded=recorded,
            )
        else:
            raise BenchmarkConfigError(
                f"Scalability measurement for {dataset_id} has an invalid status"
            )
        if observed != recorded:
            raise BenchmarkConfigError(f"Recorded scalability measurement changed for {dataset_id}")
        return recorded
    if runtime_state is not None and _MEASUREMENT_STATE_FIELD in runtime_state:
        _load_bound_scalability_measurement(
            repo_root=repo_root,
            runtime_state=runtime_state,
            measurement_path=measurement_path,
            dataset_id=dataset_id,
        )

    with run_lock(run_root):
        common = {
            "repo_root": repo_root,
            "profile_id": profile_id,
            "run_root": run_root,
        }
        preflight(
            **common,
            andrea_repo=andrea_repo,
            spathi_repo=spathi_repo,
            profile_file=profile_path,
        )
        build_spathi_image(**common, spathi_repo=spathi_repo)
        generation_started = perf_counter()
        generate_data(**common, andrea_repo=andrea_repo)
        generation_seconds = perf_counter() - generation_started
        generated_state = load_state(run_root, profile_id)
        if status_at_least(generated_state, "inferred"):
            raise BenchmarkConfigError(
                f"{dataset_id} inference completed without its RSS measurement; "
                "remove this dataset run and resume"
            )
        image = generated_state.get("spathi_image")
        if not isinstance(image, Mapping) or not isinstance(image.get("id"), str):
            raise BenchmarkConfigError("Scalability run has no SPATHI image identity")
        sampler = _PeakRssSampler(str(image["id"]))
        sampler.start()
        inference_started = perf_counter()
        try:
            run_inference(**common, andrea_repo=andrea_repo)
        except InferenceExecutionFailure as failure:
            inference_seconds = perf_counter() - inference_started
            peak_rss, rss_samples = sampler.finish(require_samples=False)
            failed_state = load_state(run_root, profile_id)
            measurement = collect_failed_scalability_measurement(
                repo_root=repo_root,
                runtime_state=failed_state,
                dataset_spec=dataset,
                run_id=str(materialization["run_id"]),
                algorithm_seed=int(materialization["algorithm_seed"]),
                seed_derivation=materialization["seed_derivation"],
                resources=inference_resources,
                generation_wall_time_seconds=generation_seconds,
                inference_wall_time_seconds=inference_seconds,
                peak_rss_bytes=peak_rss,
                rss_samples=rss_samples,
                failure=failure,
            )
            _publish_scalability_measurement(
                repo_root=repo_root,
                run_root=run_root,
                profile_id=profile_id,
                runtime_state=failed_state,
                measurement_path=measurement_path,
                measurement=measurement,
                dataset_id=dataset_id,
            )
            return measurement
        except BaseException:
            sampler.finish(require_samples=False)
            raise
        inference_seconds = perf_counter() - inference_started
        peak_rss, rss_samples = sampler.finish(require_samples=True)
        inferred_state = load_state(run_root, profile_id)
        measurement = collect_scalability_measurement(
            repo_root=repo_root,
            runtime_state=inferred_state,
            dataset_spec=dataset,
            run_id=str(materialization["run_id"]),
            algorithm_seed=int(materialization["algorithm_seed"]),
            seed_derivation=materialization["seed_derivation"],
            resources=inference_resources,
            generation_wall_time_seconds=generation_seconds,
            inference_wall_time_seconds=inference_seconds,
            peak_rss_bytes=peak_rss,
            rss_samples=rss_samples,
        )
        _publish_scalability_measurement(
            repo_root=repo_root,
            run_root=run_root,
            profile_id=profile_id,
            runtime_state=inferred_state,
            measurement_path=measurement_path,
            measurement=measurement,
            dataset_id=dataset_id,
        )
    return measurement


def _result_document(
    *,
    repo_root: Path,
    budget: str,
    template_path: Path,
    production_path: Path,
    production: Mapping[str, Any],
    image_id: str,
    runtime_environment: Mapping[str, Any],
    host_resources: Mapping[str, Any],
    resources: Mapping[str, Any],
    expression_preparation: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completed_rows = [item for item in measurements if item.get("status") == "completed"]
    failed_rows = [item for item in measurements if item.get("status") == "failed"]
    if len(completed_rows) + len(failed_rows) != len(measurements):
        raise BenchmarkConfigError("Scalability measurements contain an invalid status")
    observed_peaks = [
        int(value)
        for item in measurements
        for value in (
            item["inference"]["peak_rss_bytes"]
            if item.get("status") == "completed"
            else item["observations"].get("peak_rss_bytes"),
        )
        if value is not None
    ]
    failure_kinds = {
        kind: sum(1 for item in failed_rows if item["failure"]["failure_kind"] == kind)
        for kind in ("oom", "timeout", "tool_error", "infrastructure_error")
    }
    return {
        "schema_version": "1.0",
        "artifact_type": "spathi_scalability_results",
        "budget": budget,
        "evidence_role": "publication" if budget == "full" else "operational",
        "claim_scope": "computational_scale_stress_only",
        "status": "complete",
        "expression_preparation": dict(expression_preparation),
        "profile": {
            "path": _relative(repo_root, template_path, "scalability profile"),
            "sha256": sha256_file(template_path),
        },
        "production_config": {
            "path": _relative(repo_root, production_path, "production config"),
            "sha256": sha256_file(production_path),
            "policy_id": production["policy"]["policy_id"],
            "policy_sha256": production["policy"]["policy_sha256"],
            "decision": production["decision"],
        },
        "execution_environment": {
            "spathi_commit": production["policy"]["spathi_commit"],
            "andrea_commit": runtime_environment["andrea_commit"],
            "spathi_image_id": image_id,
            "andrea_environment_fingerprint": runtime_environment["andrea_environment_fingerprint"],
            "benchmark_environment_fingerprint": runtime_environment[
                "benchmark_environment_fingerprint"
            ],
            "docker_runtime_environment_fingerprint": runtime_environment[
                "docker_runtime_environment_fingerprint"
            ],
            **dict(host_resources),
            "max_cores": int(resources["max_cores"]),
            "max_ram_gb": float(resources["max_ram_gb"]),
            "rss_measurement_method": "linux_ps_process_tree_rss_sum",
            "rss_sample_interval_seconds": _RSS_SAMPLE_INTERVAL_SECONDS,
        },
        "summary": {
            "datasets_requested": len(measurements),
            "datasets_completed": len(completed_rows),
            "datasets_failed": len(failed_rows),
            "all_completed": not failed_rows,
            "failure_kinds": failure_kinds,
            "total_generation_wall_time_seconds": sum(
                float(item["generation"]["wall_time_seconds"]) for item in measurements
            ),
            "total_inference_wall_time_seconds": sum(
                float(
                    item["inference"]["harness_wall_time_seconds"]
                    if item.get("status") == "completed"
                    else item["observations"]["harness_wall_time_seconds"]
                )
                for item in measurements
            ),
            "datasets_with_peak_rss": len(observed_peaks),
            "maximum_peak_rss_bytes": max(observed_peaks, default=0),
            "total_spathi_output_bytes": sum(
                int(item["inference"]["spathi_output_bytes"]) for item in completed_rows
            ),
            "total_trees_fitted": sum(
                int(item["inference"]["trees_fitted"]) for item in completed_rows
            ),
        },
        "datasets": [dict(item) for item in measurements],
        "notes": [
            "The rounded synthetic grid characterizes computational scale, not biological realism.",
            "Expression supplied to SPATHI is library-size normalized to 10000 across every "
            "generated gene and then log1p transformed under the public preparation contract.",
            "Accuracy claims remain restricted to the independent simulator comparison phase.",
            "Peak RSS is sampled from the SPATHI container process tree on the recorded host.",
            "A failed endpoint is retained only when ANDREA produced a validated report for the exact requested dataset and SPATHI run.",
            "Failed endpoints are feasibility evidence, not successful completions; contract or identity corruption aborts publication.",
        ],
    }


def run_scalability_workflow(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    profile: str,
    efficiency_run_id: str | None = None,
) -> dict[str, Any]:
    """Run or resume one operational or publication computational scale study."""

    if profile not in {"reduced", "full"}:
        raise BenchmarkConfigError("Scalability profile must be 'reduced' or 'full'")
    root = repo_root.resolve()
    work = path_below_no_symlinks(
        work_root,
        root / "work",
        "Scalability work root",
    )
    production_path, production = completed_production_config(
        repo_root=root,
        work_root=work,
        budget=profile,
        efficiency_run_id=efficiency_run_id,
    )
    template_id = f"scalability-{profile}"
    template_path = root / f"experiments/v1/profiles/{template_id}.json"
    template = load_profile(root, template_id, require_runnable=True)
    request_path = path_below_no_symlinks(
        root / template["generation_jobs"][0]["generator_request"],
        root,
        "scale-stress scalability panel",
        require_file=True,
    )
    panel = load_json_object(request_path, "scale-stress scalability panel")
    validate_json(panel, root / "schemas/scale-stress-panel.schema.json", "scale-stress panel")
    host_resources = _host_resources()
    benchmark = observe_benchmark_repository(root, require_clean=True)
    identity = canonical_json_sha256(
        {
            "profile_sha256": sha256_file(template_path),
            "request_sha256": sha256_file(request_path),
            "production_config_sha256": sha256_file(production_path),
            "benchmark_repository": {
                "commit": benchmark["commit"],
                "origin": benchmark["origin"],
            },
            "andrea_environment": andrea_host_environment()["fingerprint"],
            "benchmark_environment": benchmark_host_environment()["fingerprint"],
            "docker_runtime": docker_runtime_environment()["fingerprint"],
            "execution_host": host_resources,
            "implementation": {
                relative: sha256_file(root / relative) for relative in _IDENTITY_INPUTS
            },
        }
    )
    flow_id = identity[:20]
    workflow_root = work / "v1/workflows/scalability" / profile / flow_id
    workflow_path = workflow_root / "workflow.json"
    with run_lock(workflow_root):
        measurements = [
            _run_one_dataset(
                repo_root=root,
                andrea_repo=andrea_repo,
                spathi_repo=spathi_repo,
                work_root=work,
                budget=profile,
                flow_id=flow_id,
                template_profile=template,
                production_path=production_path,
                production=production,
                panel=panel,
                dataset=dataset,
                workflow_root=workflow_root,
            )
            for dataset in panel["datasets"]
        ]
        runtime_states = [
            load_state(
                work / "v1" / _runtime_profile_id(profile, str(item["dataset_id"])) / flow_id,
                _runtime_profile_id(profile, str(item["dataset_id"])),
            )
            for item in measurements
        ]
        image_ids = {state["spathi_image"]["id"] for state in runtime_states}
        if len(image_ids) != 1:
            raise BenchmarkConfigError("Scalability datasets used different SPATHI images")
        runtime_environments = {
            (
                state["repositories"]["observed"]["andrea"]["commit"],
                state["host_environment"]["fingerprint"],
                state["benchmark_environment"]["fingerprint"],
                state["docker_runtime_environment"]["fingerprint"],
            )
            for state in runtime_states
        }
        if len(runtime_environments) != 1:
            raise BenchmarkConfigError("Scalability datasets used different host environments")
        (
            andrea_commit,
            andrea_environment,
            benchmark_environment,
            docker_environment,
        ) = next(iter(runtime_environments))
        results = _result_document(
            repo_root=root,
            budget=profile,
            template_path=template_path,
            production_path=production_path,
            production=production,
            image_id=next(iter(image_ids)),
            runtime_environment={
                "andrea_commit": andrea_commit,
                "andrea_environment_fingerprint": andrea_environment,
                "benchmark_environment_fingerprint": benchmark_environment,
                "docker_runtime_environment_fingerprint": docker_environment,
            },
            host_resources=host_resources,
            resources=template["resources"]["infer_network"],
            expression_preparation=panel["expression_preparation"],
            measurements=measurements,
        )
        validate_json(
            results,
            root / "schemas/scalability-results.schema.json",
            "scalability results",
        )
        results_path = write_idempotent_json(
            workflow_root / "scalability-results.json",
            results,
            "scalability results",
            boundary=root,
        )
        report_path = work / "v1/reports" / profile / flow_id / "scalability.html"
        publishing_state = {
            "schema_version": "1.0",
            "workflow_type": "scalability",
            "profile": profile,
            "flow_id": flow_id,
            "identity": {"fingerprint": identity},
            "status": "publishing",
            "production_config": _relative(root, production_path, "production config"),
            "production_config_sha256": sha256_file(production_path),
            "results": _relative(root, results_path, "scalability results"),
            "results_sha256": sha256_file(results_path),
            "interactive_report": _relative(root, report_path, "scalability report"),
            "dataset_runs": [
                {
                    "dataset_id": item["dataset_id"],
                    "run_root": _relative(
                        root,
                        work
                        / "v1"
                        / _runtime_profile_id(profile, str(item["dataset_id"]))
                        / flow_id,
                        "scalability run",
                    ),
                }
                for item in measurements
            ],
        }
        state, report = publish_workflow_report(
            repo_root=root,
            workflow_path=workflow_path,
            publishing_state=publishing_state,
            html_path=report_path,
            manifest_path=report_path.parent / "scalability-report.manifest.json",
            render=lambda: render_scalability_report(
                repo_root=root,
                version_root=work / "v1",
                workflow_root=workflow_root,
            ),
            label="scalability",
        )
    return state


__all__ = [
    "collect_failed_scalability_measurement",
    "collect_scalability_measurement",
    "materialize_scalability_dataset",
    "run_scalability_workflow",
    "validate_failed_scalability_measurement",
]
