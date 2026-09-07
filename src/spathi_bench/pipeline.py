"""Incremental, content-addressed SPATHI benchmark pipeline."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import (
    Artifact,
    ArtifactKey,
    ArtifactStore,
    canonical_json_sha256,
    sha256_file,
)
from .calibration_contexts import load_dataset_group_contract, require_complete_group_contexts
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    load_profile_file,
    load_repo_lock,
    load_run_metadata,
    validate_all_profiles,
    validate_design_contracts,
    validate_locked_repository,
    validate_profile_references,
)
from .environment_lock import validate_environment_lock
from .integrate import integrate_inference_runs
from .manifests import datasets_from_benchmark, resolve_manifest_member, unique_direct_manifest
from .provenance import (
    freeze_run_inputs,
    frozen_path,
    frozen_profile,
    frozen_repository_root,
    validate_run_inputs,
)
from .runtime import (
    CommandExecutionError,
    andrea_command,
    andrea_contract_command,
    andrea_host_environment,
    benchmark_host_environment,
    catalog_image_lease,
    catalog_image_leases,
    check_docker,
    docker_image_details,
    docker_runtime_environment,
    stream_command,
    validate_andrea_import,
)
from .safe_io import atomic_write_json, path_below_no_symlinks, reject_symlink_components
from .scenario_inputs import scenario_external_input_records
from .state import (
    STATE_SCHEMA_VERSION,
    load_state,
    require_status,
    state_path,
    status_at_least,
    write_state,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class InferenceExecutionFailure(RuntimeError):
    """A validated single-run ANDREA execution whose tool failed.

    This exception is deliberately narrower than a generic command failure.  It
    is emitted only after the failed ANDREA report has been checked for the
    requested dataset, run, and tool origin.  Workflows may therefore retain it
    as operational evidence without treating a missing or corrupt report as a
    scientific result.
    """

    def __init__(
        self,
        *,
        dataset_id: str,
        batch_id: str,
        run_id: str,
        orchestrator_return_code: int,
        tool_exit_code: int,
        tool_duration_seconds: float,
        tool_error: str,
        command_log_path: Path,
        progress: dict[str, Any] | None,
        tool_log_tail: str | None,
        evidence_artifact: Artifact | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.batch_id = batch_id
        self.run_id = run_id
        self.orchestrator_return_code = orchestrator_return_code
        self.tool_exit_code = tool_exit_code
        self.tool_duration_seconds = tool_duration_seconds
        self.tool_error = tool_error
        self.command_log_path = command_log_path.resolve()
        self.progress = progress
        self.tool_log_tail = tool_log_tail
        self.evidence_artifact = evidence_artifact
        evidence_suffix = (
            f"; durable evidence: {evidence_artifact.file('failure.json').resolve()}"
            if evidence_artifact is not None
            else ""
        )
        super().__init__(f"Inference run {run_id!r} failed: {tool_error}{evidence_suffix}")

    def with_evidence(self, artifact: Artifact) -> InferenceExecutionFailure:
        """Return this validated failure bound to immutable diagnostic evidence."""

        if artifact.key.kind != "inference-failure":
            raise BenchmarkConfigError("Inference failure evidence has the wrong artifact kind")
        return InferenceExecutionFailure(
            dataset_id=self.dataset_id,
            batch_id=self.batch_id,
            run_id=self.run_id,
            orchestrator_return_code=self.orchestrator_return_code,
            tool_exit_code=self.tool_exit_code,
            tool_duration_seconds=self.tool_duration_seconds,
            tool_error=self.tool_error,
            command_log_path=self.command_log_path,
            progress=self.progress,
            tool_log_tail=self.tool_log_tail,
            evidence_artifact=artifact,
        )


def _artifact_store(run_root: Path) -> ArtifactStore:
    resolved = reject_symlink_components(run_root, "Benchmark run root")
    try:
        version_root = resolved.parents[1]
    except IndexError as exc:
        raise BenchmarkConfigError(
            f"Run root is too shallow for a shared artifact store: {run_root}"
        ) from exc
    if version_root == Path(resolved.anchor):
        raise BenchmarkConfigError(
            "Run root must follow <work>/<version>/<profile>/<run-id>; "
            f"refusing an artifact store at {version_root / 'artifacts'}"
        )
    store_root = reject_symlink_components(version_root / "artifacts", "Shared artifact store")
    return ArtifactStore(store_root)


def _artifact_ref(artifact: Artifact, *, cache_hit: bool) -> dict[str, Any]:
    return {
        "kind": artifact.key.kind,
        "digest": artifact.key.digest,
        "key": artifact.key.value,
        "path": str(artifact.path.resolve()),
        "cache_hit": cache_hit,
    }


def _artifact_from_ref(store: ArtifactStore, reference: dict[str, Any]) -> Artifact:
    try:
        key = ArtifactKey(kind=str(reference["kind"]), digest=str(reference["digest"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkConfigError(f"Invalid artifact reference: {reference!r}") from exc
    if reference.get("key") != key.value:
        raise BenchmarkConfigError(f"Artifact reference key differs from its digest: {reference!r}")
    artifact = store.lookup(key)
    if artifact is None:
        raise BenchmarkConfigError(f"Referenced artifact is missing: {key}")
    recorded_path = path_below_no_symlinks(
        Path(str(reference.get("path", ""))),
        artifact.path.parent,
        "Artifact reference path",
        require_directory=True,
    )
    if recorded_path != artifact.path:
        raise BenchmarkConfigError(f"Artifact reference path differs from its content key: {key}")
    return artifact


def _artifact_member_from_state(artifact: Artifact, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise BenchmarkConfigError(f"{label} is missing from benchmark state")
    root = reject_symlink_components(artifact.path, f"{label} artifact")
    candidate = path_below_no_symlinks(
        Path(raw_path),
        root,
        label,
        require_file=True,
    )
    relative = candidate.relative_to(root).as_posix()
    try:
        member = artifact.file(relative)
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} is not declared by its artifact: {raw_path}") from exc
    return path_below_no_symlinks(member, root, label, require_file=True)


def _validated_dataset_paths(
    store: ArtifactStore,
    dataset: dict[str, Any],
) -> tuple[Path, Path]:
    artifact = _artifact_from_ref(store, dataset["dataset_artifact"])
    return (
        _artifact_member_from_state(artifact, dataset.get("dataset_manifest"), "dataset manifest"),
        _artifact_member_from_state(
            artifact,
            dataset.get("ground_truth_manifest"),
            "ground-truth manifest",
        ),
    )


def _load_existing_state(repo_root: Path, profile_id: str, run_root: Path) -> dict[str, Any] | None:
    if not state_path(run_root).is_file():
        return None
    validate_run_inputs(repo_root, profile_id, run_root)
    return load_state(run_root, profile_id)


def _validate_state_andrea(state: dict[str, Any], observed: dict[str, Any]) -> None:
    recorded = (state.get("repositories") or {}).get("observed", {}).get("andrea")
    if not isinstance(recorded, dict):
        raise BenchmarkConfigError("Benchmark state has no observed ANDREA provenance")
    for field in ("commit", "origin"):
        if recorded.get(field) != observed.get(field):
            raise BenchmarkConfigError(
                f"ANDREA {field} changed after preflight: "
                f"recorded={recorded.get(field)!r}, observed={observed.get(field)!r}"
            )


def _validate_host_environment(state: dict[str, Any], observed: dict[str, Any]) -> None:
    recorded = state.get("host_environment")
    if not isinstance(recorded, dict) or recorded.get("fingerprint") != observed.get("fingerprint"):
        raise BenchmarkConfigError(
            "The host Python environment used by ANDREA changed after preflight; use a new run ID"
        )


def _validate_docker_runtime(state: dict[str, Any], observed: dict[str, Any]) -> None:
    recorded = state.get("docker_runtime_environment")
    if not isinstance(recorded, dict) or recorded.get("fingerprint") != observed.get("fingerprint"):
        raise BenchmarkConfigError(
            "The Docker runtime used for this benchmark changed after preflight; use a new run ID"
        )


def _validate_andrea_repository(
    frozen_root: Path,
    andrea_repo: Path,
) -> dict[str, Any]:
    lock = load_repo_lock(frozen_root)["repositories"]["andrea"]
    observed = validate_locked_repository(
        name="andrea",
        path=andrea_repo,
        lock_entry=lock,
        require_clean=True,
    )
    observed["import_path"] = validate_andrea_import(andrea_repo)
    return observed


def _validate_spathi_image_contract(frozen_root: Path) -> dict[str, Any]:
    image_lock = load_image_lock(frozen_root)
    image = image_lock["images"]["spathi"]
    repository_lock = load_repo_lock(frozen_root)["repositories"]["spathi"]
    if image["source_repository"] != "spathi":
        raise BenchmarkConfigError("SPATHI image source_repository must be 'spathi'")
    if image["source_commit"] != repository_lock["commit"]:
        raise BenchmarkConfigError("SPATHI image source commit differs from repos.lock.json")
    for relative_path, expected_digest in image["build_inputs"].items():
        source = frozen_root / relative_path
        if not source.is_file() or sha256_file(source) != expected_digest:
            raise BenchmarkConfigError(
                f"Frozen SPATHI image build input differs from images.lock.json: {relative_path}"
            )
    constraints: dict[str, str] = {}
    for line in (
        (frozen_root / "containers/spathi/constraints.txt").read_text(encoding="utf-8").splitlines()
    ):
        if line and not line.startswith("#"):
            name, separator, version = line.partition("==")
            if not separator:
                raise BenchmarkConfigError(f"Unpinned SPATHI dependency: {line}")
            constraints[name] = version
    if constraints != image["python_dependencies"]:
        raise BenchmarkConfigError("Frozen SPATHI constraints differ from images.lock.json")
    return image_lock


def _catalog_spec_path(andrea_repo: Path, image_id: str) -> Path:
    kind, catalog_id = image_id.split(":", 1)
    if kind == "simulator":
        return (
            andrea_repo
            / "andrea/catalog_simulation_data_tools/simulators"
            / catalog_id
            / "simulatorspec.json"
        )
    return andrea_repo / "andrea/catalog_inference_tools/tools" / catalog_id / "toolspec.json"


def _referenced_catalog_image_ids(
    run_root: Path,
    profile: dict[str, Any],
) -> set[str]:
    identifiers: set[str] = set()
    for job in profile["generation_jobs"]:
        if "simulator_runs" not in job:
            continue
        runs = load_json_object(frozen_path(run_root, job["simulator_runs"], "simulator runs"))
        identifiers.update(f"simulator:{run['simulator_id']}" for run in runs["runs"])
    for batch in profile["inference_batches"]:
        if batch["kind"] != "baseline":
            continue
        params = load_json_object(frozen_path(run_root, batch["tools_params"], "tools params"))
        identifiers.update(f"tool:{run['tool_id']}" for run in params["runs"])
    return identifiers


def _pin_profile_catalog_images(
    *,
    frozen_root: Path,
    run_root: Path,
    profile: dict[str, Any],
    andrea_repo: Path,
) -> dict[str, Any]:
    lock = load_image_lock(frozen_root)["catalog_images"]
    observed: dict[str, Any] = {}
    for image_id in sorted(_referenced_catalog_image_ids(run_root, profile)):
        if image_id not in lock:
            raise BenchmarkConfigError(f"No immutable catalog image lock for {image_id}")
        toolspec = load_json_object(_catalog_spec_path(andrea_repo, image_id), f"{image_id} spec")
        if toolspec.get("docker_image") != lock[image_id]["tag"]:
            raise BenchmarkConfigError(
                f"{image_id} catalog tag differs from images.lock.json: "
                f"{toolspec.get('docker_image')!r}"
            )
        with catalog_image_lease(lock[image_id]) as details:
            observed[image_id] = details
    return observed


def _benchmark_repository_provenance(run_root: Path) -> dict[str, Any]:
    snapshot = load_json_object(run_root / "input_snapshot/manifest.json", "input snapshot")
    repository = snapshot.get("benchmark_repository")
    if (
        not isinstance(repository, dict)
        or not isinstance(repository.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", repository["commit"]) is None
        or not isinstance(repository.get("origin"), str)
        or not repository["origin"]
    ):
        raise BenchmarkConfigError("Input snapshot has invalid SPATHI-benchmarks provenance")
    return dict(repository)


def _initial_state(
    *,
    created_at: str,
    profile_id: str,
    profile: dict[str, Any],
    run_root: Path,
    repository_lock: dict[str, Any],
    repositories: dict[str, Any],
    catalog_images: dict[str, Any],
    host_environment: dict[str, Any],
    benchmark_environment: dict[str, Any],
    docker_runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "created_at": created_at,
        "updated_at": _utc_now(),
        "profile_id": profile_id,
        "purpose": profile["purpose"],
        "input_snapshot": str((run_root / "input_snapshot/manifest.json").resolve()),
        "benchmark_repository": _benchmark_repository_provenance(run_root),
        "repositories": {
            "lock": repository_lock["repositories"],
            "observed": repositories,
        },
        "catalog_images": catalog_images,
        "host_environment": host_environment,
        "benchmark_environment": benchmark_environment,
        "docker_runtime_environment": docker_runtime,
        "artifact_store": str(_artifact_store(run_root).root.resolve()),
        "benchmarks": [],
        "datasets": [],
        "inference_runs": [],
        "integrated_inference_runs": [],
        "evaluations": [],
        "joint_evaluations": [],
        "analysis_bundles": [],
        "cache": {
            "dataset_hits": 0,
            "inference_hits": 0,
            "integration_hits": 0,
            "evaluation_hits": 0,
            "joint_evaluation_hits": 0,
            "bundle_hits": 0,
        },
        "status": "preflighted",
    }


def preflight(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    profile_file: Path | None = None,
) -> Path:
    # Re-run the committed design gates before both fresh and resumed work.  A
    # stale seed plan must never be hidden by an already-preflighted state.
    validate_environment_lock(repo_root)
    validate_all_profiles(repo_root)
    validate_design_contracts(repo_root, andrea_repo)
    existing = _load_existing_state(repo_root, profile_id, run_root)
    if existing is not None:
        _validate_docker_runtime(existing, docker_runtime_environment())
        if not status_at_least(existing, "preflighted"):
            raise BenchmarkConfigError("Existing run has not completed preflight")
        if profile_file is not None:
            freeze_run_inputs(
                repo_root,
                profile_id,
                run_root,
                profile_file=profile_file,
            )
        path = run_root / "preflight" / "benchmark-preflight.json"
        if not path.is_file():
            raise BenchmarkConfigError("Preflight state exists but its report is missing")
        return path

    if profile_file is not None:
        runtime_profile = load_profile_file(
            repo_root,
            profile_file,
            profile_id,
            require_runnable=True,
            require_runtime=True,
        )
        validate_profile_references(repo_root, runtime_profile)
    freeze_run_inputs(repo_root, profile_id, run_root, profile_file=profile_file)
    frozen_root = frozen_repository_root(run_root)
    profile = frozen_profile(run_root, profile_id)
    validate_profile_references(frozen_root, profile)
    repository_lock = load_repo_lock(frozen_root)
    repositories = {
        "andrea": _validate_andrea_repository(frozen_root, andrea_repo),
        "spathi": validate_locked_repository(
            name="spathi",
            path=spathi_repo,
            lock_entry=repository_lock["repositories"]["spathi"],
            require_clean=True,
        ),
    }
    check_docker()
    _validate_spathi_image_contract(frozen_root)
    catalog_images = _pin_profile_catalog_images(
        frozen_root=frozen_root,
        run_root=run_root,
        profile=profile,
        andrea_repo=andrea_repo,
    )
    host_environment = andrea_host_environment()
    benchmark_environment = benchmark_host_environment()
    docker_runtime = docker_runtime_environment()

    preflight_dir = run_root / "preflight"
    if preflight_dir.exists():
        report = load_json_object(
            preflight_dir / "benchmark-preflight.json",
            "recoverable benchmark preflight",
        )
        expected = {
            "schema_version": "1.0",
            "profile_id": profile_id,
            "purpose": profile["purpose"],
            "repositories": repositories,
            "catalog_images": catalog_images,
            "host_environment": host_environment,
            "benchmark_environment": benchmark_environment,
            "docker_runtime_environment": docker_runtime,
            "status": "passed",
        }
        if any(report.get(field) != value for field, value in expected.items()):
            raise BenchmarkConfigError(
                f"Preflight output exists without state and cannot be recovered: {preflight_dir}"
            )
        for job in profile["generation_jobs"]:
            suffixes = (
                ("generation-plan.json",)
                if "generator_request" in job
                else ("catalog.json", "simulation-plan.json")
            )
            for suffix in suffixes:
                if not (preflight_dir / f"{job['id']}-{suffix}").is_file():
                    raise BenchmarkConfigError(
                        f"Preflight output exists without state and is incomplete: {preflight_dir}"
                    )
        state = _initial_state(
            created_at=str(report["created_at"]),
            profile_id=profile_id,
            profile=profile,
            run_root=run_root,
            repository_lock=repository_lock,
            repositories=repositories,
            catalog_images=catalog_images,
            host_environment=host_environment,
            benchmark_environment=benchmark_environment,
            docker_runtime=docker_runtime,
        )
        write_state(run_root, state)
        return preflight_dir / "benchmark-preflight.json"
    temporary = Path(tempfile.mkdtemp(prefix=".preflight-", dir=run_root))
    try:
        resources = profile["resources"]["generate_data"]
        for job in profile["generation_jobs"]:
            if "generator_request" in job:
                from .scale_stress import validate_scale_stress_request

                request_path = frozen_path(
                    run_root,
                    job["generator_request"],
                    "scale-stress generator request",
                )
                request = validate_scale_stress_request(frozen_root, request_path)
                atomic_write_json(
                    temporary / f"{job['id']}-generation-plan.json",
                    {
                        "schema_version": "1.0",
                        "generator": "spathi_scale_stress",
                        "generation_job": job["id"],
                        "request_sha256": sha256_file(request_path),
                        "scale_grid": request["scale_grid"],
                        "expression_preparation": request["expression_preparation"],
                        "datasets": [
                            {
                                "dataset_id": dataset["id"],
                                "design_role": dataset["design_role"],
                                "seed": dataset["seed"],
                                "cells": sum(group["cells"] for group in dataset["groups"]),
                                "genes": dataset["genes"],
                                "tfs": dataset["tfs"],
                                "groups": len(dataset["groups"]),
                            }
                            for dataset in request["datasets"]
                        ],
                    },
                    f"{job['id']} generation plan",
                    boundary=temporary,
                )
                continue
            scenario = frozen_path(run_root, job["scenario"], "scenario")
            simulator_runs = frozen_path(run_root, job["simulator_runs"], "simulator runs")
            andrea_command(
                [
                    "generate-data",
                    "preflight",
                    "--scenario",
                    str(scenario),
                    "--output-json",
                    str(temporary / f"{job['id']}-catalog.json"),
                ],
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                log_path=run_root / "logs" / f"00-{job['id']}-catalog.log",
            )
            andrea_command(
                [
                    "generate-data",
                    "plan",
                    "--scenario",
                    str(scenario),
                    "--simulator-runs",
                    str(simulator_runs),
                    "--out",
                    str(temporary / f"{job['id']}-simulation-plan.json"),
                    "--max-parallel-tasks",
                    str(resources["max_parallel_tasks"]),
                    "--max-cores",
                    str(resources["max_cores"]),
                    "--max-ram-gb",
                    str(resources["max_ram_gb"]),
                ],
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                log_path=run_root / "logs" / f"00-{job['id']}-plan.log",
            )
        report = {
            "schema_version": "1.0",
            "created_at": _utc_now(),
            "profile_id": profile_id,
            "purpose": profile["purpose"],
            "repositories": repositories,
            "catalog_images": catalog_images,
            "host_environment": host_environment,
            "benchmark_environment": benchmark_environment,
            "docker_runtime_environment": docker_runtime,
            "status": "passed",
        }
        atomic_write_json(
            temporary / "benchmark-preflight.json",
            report,
            "benchmark preflight report",
            boundary=temporary,
        )
        os.rename(temporary, preflight_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    state = _initial_state(
        created_at=str(report["created_at"]),
        profile_id=profile_id,
        profile=profile,
        run_root=run_root,
        repository_lock=repository_lock,
        repositories=repositories,
        catalog_images=catalog_images,
        host_environment=host_environment,
        benchmark_environment=benchmark_environment,
        docker_runtime=docker_runtime,
    )
    write_state(run_root, state)
    return preflight_dir / "benchmark-preflight.json"


def _copy_locked_spathi_source(spathi_repo: Path, snapshot_root: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(spathi_repo), "ls-tree", "-r", "--name-only", "HEAD", "src"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BenchmarkConfigError(
            "Cannot enumerate locked SPATHI source: "
            + (completed.stderr or completed.stdout).strip()
        )
    members = [line for line in completed.stdout.splitlines() if line]
    if not members:
        raise BenchmarkConfigError("Locked SPATHI commit contains no tracked src files")
    for member in members:
        source = spathi_repo / member
        if source.is_symlink() or not source.is_file():
            raise BenchmarkConfigError(f"Unsupported tracked SPATHI source member: {member}")
        destination = snapshot_root / member
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _spathi_build_fingerprint(image: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "source_commit": image["source_commit"],
            "base_image": image["base_image"],
            "build_inputs": image["build_inputs"],
            "python_dependencies": image["python_dependencies"],
        }
    )


def build_spathi_image(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    spathi_repo: Path,
) -> Path:
    state = _load_existing_state(repo_root, profile_id, run_root)
    if state is None:
        raise BenchmarkConfigError("Build-image requires a completed preflight")
    _validate_docker_runtime(state, docker_runtime_environment())
    provenance_path = run_root / "provenance" / "spathi-image.json"
    previous_status = str(state["status"])
    if status_at_least(state, "image_ready"):
        recorded = load_json_object(provenance_path, "SPATHI image provenance")
        if status_at_least(state, "inferred"):
            return provenance_path
        current = docker_image_details(str(recorded["tag"]))
        if current is not None and current.get("id") == recorded.get("id"):
            return provenance_path
    else:
        require_status(state, "preflighted", "Build-image")
    frozen_root = frozen_repository_root(run_root)
    repository_lock = load_repo_lock(frozen_root)
    image_lock = _validate_spathi_image_contract(frozen_root)
    validate_locked_repository(
        name="spathi",
        path=spathi_repo,
        lock_entry=repository_lock["repositories"]["spathi"],
        require_clean=True,
    )
    image = image_lock["images"]["spathi"]
    expected_fingerprint = _spathi_build_fingerprint(image)
    details = docker_image_details(image["tag"])
    reusable = (
        details is not None
        and details["labels"].get("org.opencontainers.image.revision") == image["source_commit"]
        and details["labels"].get("dev.spathi.benchmark.build-fingerprint") == expected_fingerprint
    )
    if not reusable:
        with tempfile.TemporaryDirectory(prefix="spathi-build-context-") as directory:
            source_snapshot = Path(directory)
            _copy_locked_spathi_source(spathi_repo, source_snapshot)
            stream_command(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--load",
                    "--build-context",
                    f"spathi={source_snapshot}",
                    "--build-arg",
                    f"BASE_IMAGE={image['base_image']}",
                    "--build-arg",
                    f"SPATHI_COMMIT={image['source_commit']}",
                    "--build-arg",
                    f"BENCH_BUILD_FINGERPRINT={expected_fingerprint}",
                    "--file",
                    str(frozen_root / "containers/spathi/Dockerfile"),
                    "--tag",
                    image["tag"],
                    str(frozen_root),
                ],
                cwd=repo_root,
                log_path=run_root / "logs" / "05-build-spathi-image.log",
            )
        details = docker_image_details(image["tag"])
    if details is None:
        raise RuntimeError(f"Built SPATHI image is unavailable: {image['tag']}")
    expected_commit = image["source_commit"]
    if details["labels"].get("org.opencontainers.image.revision") != expected_commit:
        raise BenchmarkConfigError("Built SPATHI image has the wrong revision label")
    if details["labels"].get("dev.spathi.benchmark.build-fingerprint") != expected_fingerprint:
        raise BenchmarkConfigError("Built SPATHI image has the wrong build fingerprint")
    details.update(
        {
            "schema_version": "1.0",
            "built_at": _utc_now(),
            "tag": image["tag"],
            "source_commit": expected_commit,
            "base_image": image["base_image"],
            "build_inputs": image["build_inputs"],
            "build_fingerprint": expected_fingerprint,
            "python_dependencies": image["python_dependencies"],
            "reused": reusable,
        }
    )
    atomic_write_json(
        provenance_path,
        details,
        "SPATHI image provenance",
        boundary=run_root,
    )
    state["spathi_image"] = {**details, "provenance": str(provenance_path.resolve())}
    state["status"] = previous_status if previous_status != "preflighted" else "image_ready"
    state["updated_at"] = _utc_now()
    write_state(run_root, state)
    return provenance_path


def _generation_spec(
    *,
    state: dict[str, Any],
    job: dict[str, Any],
    scenario: Path,
    simulator_runs: Path,
    input_root: Path,
    resources: dict[str, Any],
) -> dict[str, Any]:
    runs = load_json_object(simulator_runs, "simulator runs")
    simulator_images = {
        simulator_id: state["catalog_images"][f"simulator:{simulator_id}"]["locked_repo_digest"]
        for simulator_id in sorted({str(run["simulator_id"]) for run in runs["runs"]})
    }
    return {
        "schema_version": "1.0",
        "andrea_commit": state["repositories"]["observed"]["andrea"]["commit"],
        "andrea_environment": state["host_environment"]["fingerprint"],
        "scenario_sha256": sha256_file(scenario),
        "scenario_external_inputs": scenario_external_input_records(scenario, input_root),
        "simulator_runs_sha256": sha256_file(simulator_runs),
        "simulator_images": simulator_images,
        "resources": resources,
        "generation_job": job["id"],
        "dataset_role": job["dataset_role"],
        "split_id": job["split_id"],
        "condition_id": job["condition_id"],
    }


def _scale_stress_generation_spec(
    *,
    state: dict[str, Any],
    job: dict[str, Any],
    request: Path,
    input_root: Path,
    resources: dict[str, Any],
) -> dict[str, Any]:
    from .scale_stress import validate_scale_stress_request

    validated_request = validate_scale_stress_request(input_root, request)
    seed_plan = path_below_no_symlinks(
        input_root / str(validated_request["seed_plan"]),
        input_root,
        "scale-stress seed plan",
        require_file=True,
    )
    implementation_root = Path(__file__).parent
    return {
        "schema_version": "1.0",
        "generator": "spathi_scale_stress",
        "generator_implementation": {
            name: sha256_file(implementation_root / name)
            for name in ("scale_stress.py", "safe_io.py", "seeds.py")
        },
        "request_sha256": sha256_file(request),
        "request_dependencies": {
            "seed_plan": {
                "path": seed_plan.relative_to(input_root).as_posix(),
                "sha256": sha256_file(seed_plan),
            },
        },
        "andrea_commit": state["repositories"]["observed"]["andrea"]["commit"],
        "benchmark_environment": state["benchmark_environment"]["fingerprint"],
        "resources": resources,
        "generation_job": job["id"],
        "dataset_role": job["dataset_role"],
        "split_id": job["split_id"],
        "condition_id": job["condition_id"],
    }


def _generation_image_requirements(
    *,
    state: dict[str, Any],
    frozen_root: Path,
    simulator_runs: Path,
) -> list[tuple[dict[str, Any], str]]:
    runs = load_json_object(simulator_runs, "simulator runs")
    lock = load_image_lock(frozen_root)["catalog_images"]
    requirements: list[tuple[dict[str, Any], str]] = []
    for simulator_id in sorted({str(run["simulator_id"]) for run in runs["runs"]}):
        image_id = f"simulator:{simulator_id}"
        recorded = state["catalog_images"][image_id]
        requirements.append((lock[image_id], str(recorded["id"])))
    return requirements


def _validate_generation_images(
    *,
    state: dict[str, Any],
    frozen_root: Path,
    simulator_runs: Path,
) -> None:
    for lock_entry, expected_id in _generation_image_requirements(
        state=state,
        frozen_root=frozen_root,
        simulator_runs=simulator_runs,
    ):
        current = docker_image_details(str(lock_entry["tag"]))
        if current is None or current.get("id") != expected_id:
            raise BenchmarkConfigError(
                f"Locked catalog image changed during generation: {lock_entry['tag']}"
            )


def generate_data(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    andrea_repo: Path,
) -> Path:
    state = _load_existing_state(repo_root, profile_id, run_root)
    if state is None:
        raise BenchmarkConfigError("Generation requires a completed preflight")
    _validate_docker_runtime(state, docker_runtime_environment())
    if status_at_least(state, "generated"):
        store = _artifact_store(run_root)
        for benchmark in state["benchmarks"]:
            _artifact_from_ref(store, benchmark["artifact"])
        return state_path(run_root)
    require_status(state, "image_ready", "Generation")
    frozen_root = frozen_repository_root(run_root)
    observed_andrea = _validate_andrea_repository(frozen_root, andrea_repo)
    _validate_state_andrea(state, observed_andrea)
    _validate_host_environment(state, andrea_host_environment())
    profile = frozen_profile(run_root, profile_id)
    resources = profile["resources"]["generate_data"]
    store = _artifact_store(run_root)
    benchmarks: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    dataset_hits = 0

    for job in profile["generation_jobs"]:
        if "generator_request" in job:
            request = frozen_path(
                run_root,
                job["generator_request"],
                "scale-stress generator request",
            )
            spec = _scale_stress_generation_spec(
                state=state,
                job=job,
                request=request,
                input_root=frozen_root,
                resources=resources,
            )
            key = store.key_for("dataset", spec)
            artifact = store.lookup(key)
            cache_hit = artifact is not None
            if artifact is None:
                from .scale_stress import generate_scale_stress_panel

                with store.staging_directory(key) as staging:
                    output_root = staging / "output"
                    output_root.mkdir()
                    generate_scale_stress_panel(
                        repo_root=frozen_root,
                        request_path=request,
                        output_root=output_root,
                    )
                    unique_direct_manifest(output_root, "benchmark-manifest.json")
                    artifact = store.publish_from_temp(staging, kind="dataset", spec=spec)
            else:
                dataset_hits += 1
            manifest_path = unique_direct_manifest(
                artifact.path / "output", "benchmark-manifest.json"
            )
            job_datasets = datasets_from_benchmark(manifest_path, job["id"])
            for dataset in job_datasets:
                dataset.update(
                    {
                        "dataset_role": job["dataset_role"],
                        "split_id": job["split_id"],
                        "condition_id": job["condition_id"],
                        "dataset_artifact": _artifact_ref(artifact, cache_hit=cache_hit),
                    }
                )
            benchmarks.append(
                {
                    "generation_job": job["id"],
                    "benchmark_manifest": str(manifest_path),
                    "dataset_ids": [item["dataset_id"] for item in job_datasets],
                    "artifact": _artifact_ref(artifact, cache_hit=cache_hit),
                }
            )
            datasets.extend(job_datasets)
            continue
        scenario = frozen_path(run_root, job["scenario"], "scenario")
        simulator_runs = frozen_path(run_root, job["simulator_runs"], "simulator runs")
        spec = _generation_spec(
            state=state,
            job=job,
            scenario=scenario,
            simulator_runs=simulator_runs,
            input_root=frozen_root,
            resources=resources,
        )
        key = store.key_for("dataset", spec)
        artifact = store.lookup(key)
        cache_hit = artifact is not None
        if artifact is None:
            requirements = _generation_image_requirements(
                state=state,
                frozen_root=frozen_root,
                simulator_runs=simulator_runs,
            )
            with store.staging_directory(key) as staging:
                with catalog_image_leases(requirements):
                    output_root = staging / "output"
                    andrea_command(
                        [
                            "generate-data",
                            "execute",
                            "--scenario",
                            str(scenario),
                            "--simulator-runs",
                            str(simulator_runs),
                            "--output-dir",
                            str(output_root),
                            "--max-parallel-tasks",
                            str(resources["max_parallel_tasks"]),
                            "--max-cores",
                            str(resources["max_cores"]),
                            "--max-ram-gb",
                            str(resources["max_ram_gb"]),
                        ],
                        repo_root=repo_root,
                        andrea_repo=andrea_repo,
                        log_path=run_root / "logs" / f"10-{job['id']}.log",
                    )
                    _validate_generation_images(
                        state=state,
                        frozen_root=frozen_root,
                        simulator_runs=simulator_runs,
                    )
                unique_direct_manifest(output_root, "benchmark-manifest.json")
                artifact = store.publish_from_temp(staging, kind="dataset", spec=spec)
        else:
            dataset_hits += 1
        manifest_path = unique_direct_manifest(artifact.path / "output", "benchmark-manifest.json")
        job_datasets = datasets_from_benchmark(manifest_path, job["id"])
        for dataset in job_datasets:
            dataset.update(
                {
                    "dataset_role": job["dataset_role"],
                    "split_id": job["split_id"],
                    "condition_id": job["condition_id"],
                    "dataset_artifact": _artifact_ref(artifact, cache_hit=cache_hit),
                }
            )
        benchmarks.append(
            {
                "generation_job": job["id"],
                "benchmark_manifest": str(manifest_path),
                "dataset_ids": [item["dataset_id"] for item in job_datasets],
                "artifact": _artifact_ref(artifact, cache_hit=cache_hit),
            }
        )
        datasets.extend(job_datasets)

    dataset_ids = [item["dataset_id"] for item in datasets]
    if len(set(dataset_ids)) != len(dataset_ids):
        raise BenchmarkConfigError("Generated benchmark contains duplicate dataset IDs")
    state["benchmarks"] = benchmarks
    state["datasets"] = datasets
    state["cache"]["dataset_hits"] = dataset_hits
    state["status"] = "generated"
    state["updated_at"] = _utc_now()
    return write_state(run_root, state)


def _single_custom_definition(custom_tools: dict[str, Any], run_id: str) -> dict[str, Any]:
    matches = [item for item in custom_tools.get("tools", []) if item.get("run_id") == run_id]
    if len(matches) != 1:
        raise BenchmarkConfigError(
            f"Expected one custom tool definition for {run_id!r}, found {len(matches)}"
        )
    return matches[0]


def _validate_single_run_report(
    report_path: Path,
    *,
    run_id: str,
    dataset_id: str,
    expected_origin: str,
) -> dict[str, Any]:
    report = load_json_object(report_path, "run report")
    tools = report.get("tools") or {}
    selected = tools.get("selected")
    status_by_tool = tools.get("status_by_tool") or {}
    if report.get("run_id") != run_id or report.get("status") != "executed" or selected != [run_id]:
        raise BenchmarkConfigError(
            f"Inference artifact did not execute exactly {run_id!r}: "
            f"report_run_id={report.get('run_id')!r}, status={report.get('status')!r}, "
            f"selected={selected!r}"
        )
    if status_by_tool.get(run_id) not in {"completed", "completed_with_warnings"}:
        raise BenchmarkConfigError(
            f"Inference run {run_id!r} did not complete: {status_by_tool.get(run_id)!r}"
        )
    if tools.get("failed"):
        raise BenchmarkConfigError(f"Inference run report contains failures: {tools['failed']}")
    if (report.get("dataset") or {}).get("id") != dataset_id:
        raise BenchmarkConfigError("Inference run report has the wrong dataset identity")
    if (tools.get("tool_origins") or {}).get(run_id) != expected_origin:
        raise BenchmarkConfigError(
            f"Inference run {run_id!r} has the wrong origin; expected {expected_origin!r}"
        )
    return report


def _reported_failure_member(report_path: Path, raw_path: Any, label: str) -> Path:
    """Resolve one failed-run diagnostic without permitting staging escape."""

    if not isinstance(raw_path, str) or not raw_path:
        raise BenchmarkConfigError(f"Failed inference report has no {label}")
    raw = Path(raw_path)
    candidate = raw if raw.is_absolute() else report_path.parent / raw
    root = reject_symlink_components(report_path.parent, "Failed inference run directory")
    try:
        return path_below_no_symlinks(
            candidate,
            root,
            f"Failed inference report {label}",
            require_file=True,
        )
    except BenchmarkConfigError as exc:
        raise BenchmarkConfigError(
            f"Failed inference report {label} escapes or is missing from its run directory"
        ) from exc


def _failure_progress_snapshot(path: Path | None) -> dict[str, Any] | None:
    """Read a wrapper progress snapshot without hiding a malformed contract."""

    if path is None:
        return None
    payload = load_json_object(path, "failed inference progress")
    percent = payload.get("percent")
    status = payload.get("status")
    phase = payload.get("phase")
    message = payload.get("message")
    warnings = payload.get("warnings", [])
    if (
        not isinstance(percent, int)
        or isinstance(percent, bool)
        or not 0 <= percent <= 100
        or not isinstance(status, str)
        or not status
        or not isinstance(phase, str)
        or not phase
        or not isinstance(message, str)
        or not isinstance(warnings, list)
        or any(not isinstance(item, str) for item in warnings)
    ):
        raise BenchmarkConfigError("Failed inference progress has an invalid contract")
    return {
        "percent": percent,
        "status": status,
        "phase": phase,
        "message": message,
        "warnings": list(warnings),
    }


def _tail_text(path: Path, *, max_lines: int = 200, max_characters: int = 65_536) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    tail = "\n".join(text.splitlines()[-max_lines:]).strip()
    if not tail:
        return None
    return tail[-max_characters:]


def _validated_failed_inference(
    report_path: Path,
    *,
    run_id: str,
    dataset_id: str,
    expected_origin: str,
    command_failure: CommandExecutionError,
    batch_id: str,
) -> InferenceExecutionFailure:
    """Reduce a structurally valid failed ANDREA run to transportable evidence."""

    report = load_json_object(report_path, "failed inference run report")
    tools = report.get("tools")
    if not isinstance(tools, dict):
        raise BenchmarkConfigError("Failed inference report has no tools object")
    selected = tools.get("selected")
    status_by_tool = tools.get("status_by_tool")
    origins = tools.get("tool_origins")
    failed = tools.get("failed")
    results = tools.get("results")
    if (
        report.get("run_id") != run_id
        or report.get("status") != "executed"
        or selected != [run_id]
        or not isinstance(status_by_tool, dict)
        or status_by_tool.get(run_id) != "failed"
        or not isinstance(failed, dict)
        or set(failed) != {run_id}
        or not isinstance(results, dict)
        or set(results) != {run_id}
    ):
        raise BenchmarkConfigError(
            f"Failed inference report does not describe exactly failed run {run_id!r}"
        )
    if (report.get("dataset") or {}).get("id") != dataset_id:
        raise BenchmarkConfigError("Failed inference report has the wrong dataset identity")
    if not isinstance(origins, dict) or origins.get(run_id) != expected_origin:
        raise BenchmarkConfigError("Failed inference report has the wrong tool origin")
    result = results[run_id]
    if not isinstance(result, dict) or result.get("status") != "failed":
        raise BenchmarkConfigError("Failed inference report has invalid tool execution details")
    exit_code = result.get("exit_code")
    duration = result.get("duration_seconds")
    error = result.get("error")
    if (
        not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or float(duration) < 0
        or not isinstance(error, str)
        or not error.strip()
    ):
        raise BenchmarkConfigError("Failed inference report lacks valid failure accounting")

    progress_path: Path | None = None
    if result.get("progress_path") is not None:
        progress_path = _reported_failure_member(
            report_path, result["progress_path"], "progress_path"
        )
    log_path = _reported_failure_member(report_path, result.get("logs_path"), "logs_path")
    return InferenceExecutionFailure(
        dataset_id=dataset_id,
        batch_id=batch_id,
        run_id=run_id,
        orchestrator_return_code=command_failure.return_code,
        tool_exit_code=exit_code,
        tool_duration_seconds=float(duration),
        tool_error=error.strip(),
        command_log_path=command_failure.log_path,
        progress=_failure_progress_snapshot(progress_path),
        tool_log_tail=_tail_text(log_path),
    )


_ADAPTER_FAILURE_DIRECTORY = "failure_diagnostics"
_ADAPTER_FAILURE_MANIFEST = "failure.json"
_ADAPTER_FAILURE_SIDECARS = {
    "model_diagnostics.tsv.gz",
    "run_metadata.json",
    "target_eligibility.tsv.gz",
}


def _failed_result_paths(
    report_path: Path,
    *,
    run_id: str,
) -> tuple[Path, Path | None]:
    report = load_json_object(report_path, "failed inference evidence report")
    results = (report.get("tools") or {}).get("results")
    if not isinstance(results, dict) or not isinstance(results.get(run_id), dict):
        raise BenchmarkConfigError("Failed inference evidence has no matching tool result")
    result = results[run_id]
    log_path = _reported_failure_member(report_path, result.get("logs_path"), "logs_path")
    progress_path = (
        _reported_failure_member(report_path, result["progress_path"], "progress_path")
        if result.get("progress_path") is not None
        else None
    )
    return log_path, progress_path


def _validated_adapter_failure_files(tool_log_path: Path) -> dict[str, Path]:
    root = tool_log_path.parent / "io" / "out" / _ADAPTER_FAILURE_DIRECTORY
    if not root.exists() and not root.is_symlink():
        return {}
    safe_root = path_below_no_symlinks(
        root,
        tool_log_path.parent,
        "SPATHI adapter failure diagnostics",
        require_directory=True,
    )
    manifest_path = path_below_no_symlinks(
        safe_root / _ADAPTER_FAILURE_MANIFEST,
        safe_root,
        "SPATHI adapter failure manifest",
        require_file=True,
    )
    manifest = load_json_object(manifest_path, "SPATHI adapter failure manifest")
    if set(manifest) != {
        "schema_version",
        "artifact_type",
        "status",
        "error",
        "files",
        "model_diagnostics",
    } or any(
        (
            manifest.get("schema_version") != "1.0",
            manifest.get("artifact_type") != "spathi_adapter_failure_diagnostics",
            manifest.get("status") != "failed",
        )
    ):
        raise BenchmarkConfigError("SPATHI adapter failure manifest has an invalid contract")
    error = manifest.get("error")
    if (
        not isinstance(error, dict)
        or set(error) != {"type", "message"}
        or not isinstance(error.get("type"), str)
        or not error["type"]
        or not isinstance(error.get("message"), str)
        or not error["message"]
    ):
        raise BenchmarkConfigError("SPATHI adapter failure manifest has an invalid error")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise BenchmarkConfigError("SPATHI adapter failure manifest has no file inventory")
    validated = {_ADAPTER_FAILURE_MANIFEST: manifest_path}
    listed: set[str] = set()
    for raw in raw_files:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"path", "sha256", "size"}
            or raw.get("path") not in _ADAPTER_FAILURE_SIDECARS
            or raw["path"] in listed
            or not isinstance(raw.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", raw["sha256"]) is None
            or not isinstance(raw.get("size"), int)
            or isinstance(raw["size"], bool)
            or raw["size"] < 0
        ):
            raise BenchmarkConfigError(
                "SPATHI adapter failure manifest contains an invalid file record"
            )
        name = str(raw["path"])
        source = path_below_no_symlinks(
            safe_root / name,
            safe_root,
            f"SPATHI adapter failure file {name}",
            require_file=True,
        )
        if source.stat().st_size != raw["size"] or sha256_file(source) != raw["sha256"]:
            raise BenchmarkConfigError(f"SPATHI adapter failure file changed: {name}")
        listed.add(name)
        validated[name] = source
    disk_names: set[str] = set()
    for child in safe_root.iterdir():
        if child.is_symlink() or not child.is_file():
            raise BenchmarkConfigError(
                "SPATHI adapter failure diagnostics contain a non-regular member"
            )
        disk_names.add(child.name)
    if disk_names != listed | {_ADAPTER_FAILURE_MANIFEST}:
        raise BenchmarkConfigError(
            "SPATHI adapter failure manifest differs from its diagnostic files"
        )
    return validated


def _copy_failure_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o644)


def _publish_inference_failure_evidence(
    *,
    store: ArtifactStore,
    task: dict[str, Any],
    run_root: Path,
    report_path: Path,
    failure: InferenceExecutionFailure,
) -> Artifact:
    """Publish an immutable failed-attempt artifact before inference staging is removed."""

    tool_log_path, progress_path = _failed_result_paths(report_path, run_id=failure.run_id)
    command_log_path = path_below_no_symlinks(
        failure.command_log_path,
        run_root,
        "Failed inference command log",
        require_file=True,
    )
    sources: dict[str, Path] = {
        "andrea/command.log": command_log_path,
        "andrea/run_report.json": report_path,
        "tool/container.log": tool_log_path,
    }
    if progress_path is not None:
        sources["tool/io/out/progress.json"] = progress_path
    for name, source in _validated_adapter_failure_files(tool_log_path).items():
        sources[f"tool/io/out/{_ADAPTER_FAILURE_DIRECTORY}/{name}"] = source

    source_records = {
        name: {"sha256": sha256_file(source), "size": source.stat().st_size}
        for name, source in sorted(sources.items())
    }
    evidence_spec = {
        "schema_version": "1.0",
        "evidence_contract": "andrea_inference_failure",
        "inference_status": "failed",
        "source_inference": {
            "kind": task["key"].kind,
            "digest": task["key"].digest,
            "key": task["key"].value,
            "spec_sha256": canonical_json_sha256(task["spec"]),
        },
        "attribution": {
            "dataset_id": failure.dataset_id,
            "batch_id": failure.batch_id,
            "run_id": failure.run_id,
        },
        "execution": {
            "orchestrator_return_code": failure.orchestrator_return_code,
            "tool_exit_code": failure.tool_exit_code,
            "tool_duration_seconds": failure.tool_duration_seconds,
            "tool_error": failure.tool_error,
        },
        "source_files": source_records,
    }
    key = store.key_for("inference-failure", evidence_spec)
    existing = store.lookup(key)
    if existing is not None:
        return existing
    with store.staging_directory(key) as evidence_staging:
        for name, source in sources.items():
            _copy_failure_file(source, evidence_staging / name)
        failure_manifest = {
            "schema_version": "1.0",
            "artifact_type": "andrea_inference_failure_evidence",
            "inference_status": "failed",
            "source_inference": evidence_spec["source_inference"],
            "attribution": evidence_spec["attribution"],
            "execution": evidence_spec["execution"],
            "diagnostics": sorted(sources),
        }
        atomic_write_json(
            evidence_staging / "failure.json",
            failure_manifest,
            "inference failure evidence",
            boundary=evidence_staging,
        )
        return store.publish_from_temp(
            evidence_staging,
            kind="inference-failure",
            spec=evidence_spec,
        )


def _validate_single_run_preflight(
    report_path: Path,
    *,
    run_id: str,
    dataset_id: str,
    expected_origin: str,
) -> dict[str, Any]:
    """Require ANDREA to accept exactly the requested run on this dataset."""

    report = load_json_object(report_path, "inference preflight report")
    runs = report.get("runs")
    if not isinstance(runs, dict):
        raise BenchmarkConfigError("Inference preflight report has no runs object")
    inputs = report.get("inputs")
    dataset = report.get("dataset")
    if (
        not isinstance(inputs, dict)
        or inputs.get("dataset_id") != dataset_id
        or not isinstance(dataset, dict)
        or dataset.get("dataset_id") != dataset_id
    ):
        raise BenchmarkConfigError(
            f"Dataset-specific preflight has the wrong dataset identity for {dataset_id!r}"
        )
    selected = runs.get("selected")
    skipped = runs.get("skipped")
    origins = runs.get("tool_origins")
    if runs.get("requested_total") != 1 or selected != [run_id]:
        raise BenchmarkConfigError(
            f"Dataset-specific preflight did not select exactly {run_id!r}: {selected!r}"
        )
    if skipped != {}:
        raise BenchmarkConfigError(
            f"Dataset-specific preflight skipped requested run {run_id!r}: {skipped!r}"
        )
    if origins != {run_id: expected_origin}:
        raise BenchmarkConfigError(
            f"Dataset-specific preflight assigned the wrong origin to {run_id!r}"
        )
    issues = runs.get("issues")
    if not isinstance(issues, dict):
        raise BenchmarkConfigError("Inference preflight report has no run issue inventory")
    blocking = [
        issue
        for issue in issues.get(run_id, [])
        if isinstance(issue, dict) and issue.get("severity") == "block"
    ]
    if blocking:
        raise BenchmarkConfigError(f"Dataset-specific preflight blocked {run_id!r}: {blocking!r}")
    global_issues = report.get("issues")
    if not isinstance(global_issues, list):
        raise BenchmarkConfigError("Inference preflight report has no global issue inventory")
    global_blocking = [
        issue
        for issue in global_issues
        if isinstance(issue, dict) and issue.get("severity") == "block"
    ]
    if global_blocking:
        raise BenchmarkConfigError(
            f"Dataset-specific preflight reported blocking dataset issues: {global_blocking!r}"
        )
    return report


def _runtime_image_contract(
    *,
    state: dict[str, Any],
    frozen_root: Path,
    batch_kind: str,
    run: dict[str, Any],
) -> dict[str, Any]:
    if batch_kind == "spathi":
        recorded = state.get("spathi_image")
        if not isinstance(recorded, dict):
            raise BenchmarkConfigError("Run state has no SPATHI image provenance")
        return {
            "kind": "spathi",
            "image_id": recorded["id"],
            "source_commit": recorded["source_commit"],
            "base_image": recorded["base_image"],
            "build_inputs": recorded["build_inputs"],
            "python_dependencies": recorded["python_dependencies"],
            "build_fingerprint": recorded["build_fingerprint"],
        }

    image_id = f"tool:{run['tool_id']}"
    image_lock = load_image_lock(frozen_root)["catalog_images"].get(image_id)
    recorded = state.get("catalog_images", {}).get(image_id)
    if not isinstance(image_lock, dict) or not isinstance(recorded, dict):
        raise BenchmarkConfigError(f"Missing catalog image provenance for {image_id}")
    return {
        "kind": "catalog",
        "image_id": image_id,
        "repo_digest": image_lock["repo_digest"],
    }


def _ensure_runtime_image(
    *,
    state: dict[str, Any],
    frozen_root: Path,
    batch_kind: str,
    run: dict[str, Any],
) -> None:
    if batch_kind == "spathi":
        recorded = state["spathi_image"]
        current = docker_image_details(str(recorded["id"]))
        if current is None or current.get("id") != recorded.get("id"):
            raise BenchmarkConfigError(
                "Exact SPATHI image built for this run is unavailable; rerun build-image"
            )
        return
    image_id = f"tool:{run['tool_id']}"
    image_lock = load_image_lock(frozen_root)["catalog_images"][image_id]
    recorded = state["catalog_images"][image_id]
    current = docker_image_details(str(image_lock["tag"]))
    if current is None or current.get("id") != recorded.get("id"):
        raise BenchmarkConfigError(f"Locked catalog image changed during inference: {image_id}")


def _inference_spec(
    *,
    state: dict[str, Any],
    dataset: dict[str, Any],
    batch: dict[str, Any],
    run: dict[str, Any],
    metadata: dict[str, Any],
    custom_definition: dict[str, Any] | None,
    tool_image: dict[str, Any],
    resources: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "preflight_contract": "andrea_infer_network_single_run",
        "dataset_artifact": dataset["dataset_artifact"]["key"],
        "dataset_id": dataset["dataset_id"],
        "andrea_commit": state["repositories"]["observed"]["andrea"]["commit"],
        "andrea_environment": state["host_environment"]["fingerprint"],
        "benchmark_environment": state["benchmark_environment"]["fingerprint"],
        "docker_runtime_environment": state["docker_runtime_environment"]["fingerprint"],
        "batch_kind": batch["kind"],
        "run": run,
        "run_metadata": metadata,
        "custom_definition": custom_definition,
        "tool_image": tool_image,
        "resources": resources,
    }


def _parameter_hashes(run: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, str]:
    params = run.get("params")
    if not isinstance(params, dict):
        raise BenchmarkConfigError(f"Inference run {run.get('run_id')!r} has invalid params")
    configuration = dict(params)
    seed_parameter = metadata.get("algorithm_seed_parameter")
    if seed_parameter is not None:
        if configuration.get(seed_parameter) != metadata.get("algorithm_seed"):
            raise BenchmarkConfigError(
                f"Inference run {run.get('run_id')!r} does not match its declared algorithm seed"
            )
        del configuration[seed_parameter]
    configuration_sha256 = canonical_json_sha256(configuration)
    declared_configuration_sha256 = metadata.get("configuration_sha256")
    if declared_configuration_sha256 is not None:
        if (
            not isinstance(declared_configuration_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", declared_configuration_sha256) is None
        ):
            raise BenchmarkConfigError(
                f"Inference run {run.get('run_id')!r} has an invalid logical configuration digest"
            )
        configuration_sha256 = declared_configuration_sha256
    return configuration_sha256, canonical_json_sha256(params)


def _log_token(value: Any) -> str:
    raw = str(value)
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-") or "item"
    return sanitized if sanitized == raw else f"{sanitized}-{canonical_json_sha256(raw)[:8]}"


def _execute_inference_task(
    task: dict[str, Any],
    *,
    store: ArtifactStore,
    repo_root: Path,
    andrea_repo: Path,
    run_root: Path,
    resources: dict[str, Any],
    state: dict[str, Any],
    frozen_root: Path,
) -> Artifact:
    run = task["run"]
    run_id = str(run["run_id"])
    with store.staging_directory(task["key"]) as staging:
        request_root = staging / "request"
        request_root.mkdir()
        tools_params = request_root / "tools-params.json"
        atomic_write_json(
            tools_params,
            {"runs": [run]},
            "inference tools request",
            boundary=staging,
        )
        custom_tools: Path | None = None
        if task["custom_definition"] is not None:
            custom_tools = request_root / "custom-tools.json"
            atomic_write_json(
                custom_tools,
                {"tools": [task["custom_definition"]]},
                "inference custom-tool request",
                boundary=staging,
            )
        output_root = staging / "output"
        command = [
            "infer-network",
            "execute",
            "--dataset-manifest",
            str(task["dataset_manifest"]),
            "--tools-params",
            str(tools_params),
            "--output-dir",
            str(output_root),
            "--max-cores",
            str(resources["max_cores"]),
            "--max-ram-gb",
            str(resources["max_ram_gb"]),
            "--planner",
            resources["planner"],
            "--planner-time-limit-seconds",
            str(resources["planner_time_limit_seconds"]),
        ]
        if custom_tools is not None:
            command.extend(["--custom-tools", str(custom_tools)])
        batch_kind = task["batch"]["kind"]

        preflight_path = request_root / "preflight.json"
        preflight_command = [
            "infer-network",
            "preflight",
            "--dataset-manifest",
            str(task["dataset_manifest"]),
            "--tools-params",
            str(tools_params),
            "--output-json",
            str(preflight_path),
        ]
        if custom_tools is not None:
            preflight_command.extend(["--custom-tools", str(custom_tools)])

        def execute_and_validate() -> None:
            _ensure_runtime_image(
                state=state,
                frozen_root=frozen_root,
                batch_kind=batch_kind,
                run=run,
            )
            andrea_command(
                preflight_command,
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                log_path=run_root
                / "logs"
                / (
                    f"19-{_log_token(task['dataset']['dataset_id'])}-"
                    f"{_log_token(task['batch']['id'])}-{_log_token(run_id)}.log"
                ),
            )
            _validate_single_run_preflight(
                preflight_path,
                run_id=run_id,
                dataset_id=str(task["dataset"]["dataset_id"]),
                expected_origin=task["expected_origin"],
            )
            try:
                andrea_command(
                    command,
                    repo_root=repo_root,
                    andrea_repo=andrea_repo,
                    log_path=run_root
                    / "logs"
                    / (
                        f"20-{_log_token(task['dataset']['dataset_id'])}-"
                        f"{_log_token(task['batch']['id'])}-{_log_token(run_id)}.log"
                    ),
                )
            except CommandExecutionError as exc:
                failed_reports = sorted(output_root.glob("*/run_report.json"))
                if not failed_reports:
                    # A command crash with no attributable ANDREA report is an
                    # infrastructure/contract failure, not scalability evidence.
                    raise
                if len(failed_reports) != 1:
                    raise BenchmarkConfigError(
                        "Failed inference produced an ambiguous run_report.json set"
                    ) from exc
                failure = _validated_failed_inference(
                    failed_reports[0].resolve(),
                    run_id=run_id,
                    dataset_id=str(task["dataset"]["dataset_id"]),
                    expected_origin=str(task["expected_origin"]),
                    command_failure=exc,
                    batch_id=str(task["batch"]["id"]),
                )
                evidence = _publish_inference_failure_evidence(
                    store=store,
                    task=task,
                    run_root=run_root,
                    report_path=failed_reports[0].resolve(),
                    failure=failure,
                )
                raise failure.with_evidence(evidence) from exc
            _ensure_runtime_image(
                state=state,
                frozen_root=frozen_root,
                batch_kind=batch_kind,
                run=run,
            )

        if batch_kind == "baseline":
            image_id = f"tool:{run['tool_id']}"
            image_lock = load_image_lock(frozen_root)["catalog_images"][image_id]
            recorded = state["catalog_images"][image_id]
            with catalog_image_lease(image_lock, expected_id=str(recorded["id"])):
                execute_and_validate()
        else:
            execute_and_validate()
        report_path = unique_direct_manifest(output_root, "run_report.json")
        _validate_single_run_report(
            report_path,
            run_id=run_id,
            dataset_id=task["dataset"]["dataset_id"],
            expected_origin=task["expected_origin"],
        )
        return store.publish_from_temp(
            staging,
            kind="inference",
            spec=task["spec"],
        )


def _batch_selects_dataset(batch: dict[str, Any], dataset: dict[str, Any]) -> bool:
    if dataset["dataset_role"] != batch["dataset_role"]:
        return False
    selected_jobs = batch.get("generation_jobs")
    return selected_jobs is None or dataset.get("generation_job") in selected_jobs


_INFERENCE_ATTRIBUTION_FIELDS = (
    "dataset_id",
    "batch_id",
    "batch_kind",
    "regime",
    "run_id",
    "tool_id",
    "tool_origin",
    "configuration_id",
    "selection_role",
    "algorithm_seed",
    "algorithm_seed_parameter",
    "family",
    "configuration_sha256",
    "execution_sha256",
    "tool_image",
)


def _validate_state_artifact_reference(
    store: ArtifactStore,
    raw_reference: Any,
    *,
    expected_kind: str,
    label: str,
    validated: dict[tuple[str, str], Artifact] | None = None,
) -> Artifact:
    if not isinstance(raw_reference, dict) or set(raw_reference) != {
        "kind",
        "digest",
        "key",
        "path",
        "cache_hit",
    }:
        raise BenchmarkConfigError(f"{label} has an invalid artifact reference")
    if raw_reference.get("kind") != expected_kind or not isinstance(
        raw_reference.get("cache_hit"), bool
    ):
        raise BenchmarkConfigError(f"{label} has an invalid artifact identity")
    cache_key = (expected_kind, str(raw_reference.get("key")))
    artifact = validated.get(cache_key) if validated is not None else None
    if artifact is None:
        artifact = _artifact_from_ref(store, raw_reference)
        if validated is not None:
            validated[cache_key] = artifact
    elif raw_reference.get("digest") != artifact.key.digest:
        raise BenchmarkConfigError(f"{label} artifact reference differs from its content key")
    else:
        recorded_path = path_below_no_symlinks(
            Path(str(raw_reference.get("path", ""))),
            artifact.path.parent,
            f"{label} artifact reference path",
            require_directory=True,
        )
        if recorded_path != artifact.path:
            raise BenchmarkConfigError(f"{label} artifact reference differs from its content key")
    if artifact.key.kind != expected_kind:
        raise BenchmarkConfigError(f"{label} references a non-{expected_kind} artifact")
    return artifact


def _validate_generated_state_contract(
    *,
    state: dict[str, Any],
    profile: dict[str, Any],
    run_root: Path,
    store: ArtifactStore,
) -> None:
    """Reconstruct generated-dataset attribution from immutable artifacts and inputs."""

    raw_benchmarks = state.get("benchmarks")
    raw_datasets = state.get("datasets")
    if not isinstance(raw_benchmarks, list) or not isinstance(raw_datasets, list):
        raise BenchmarkConfigError("Generated benchmark state has invalid collections")
    jobs = {str(job["id"]): job for job in profile["generation_jobs"]}
    if len(jobs) != len(profile["generation_jobs"]):
        raise BenchmarkConfigError("Frozen profile contains duplicate generation jobs")
    if len(raw_benchmarks) != len(jobs):
        raise BenchmarkConfigError("Generated benchmark state does not cover every generation job")

    datasets_by_id: dict[str, dict[str, Any]] = {}
    for raw_dataset in raw_datasets:
        if not isinstance(raw_dataset, dict):
            raise BenchmarkConfigError("Generated benchmark state contains an invalid dataset")
        dataset_id = raw_dataset.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id or dataset_id in datasets_by_id:
            raise BenchmarkConfigError(
                f"Generated benchmark state has duplicate dataset {dataset_id!r}"
            )
        datasets_by_id[dataset_id] = raw_dataset

    observed_jobs: set[str] = set()
    expected_dataset_ids: set[str] = set()
    dataset_hits = 0
    validated_artifacts: dict[tuple[str, str], Artifact] = {}
    generation_resources = profile["resources"]["generate_data"]
    for raw_benchmark in raw_benchmarks:
        if not isinstance(raw_benchmark, dict) or set(raw_benchmark) != {
            "generation_job",
            "benchmark_manifest",
            "dataset_ids",
            "artifact",
        }:
            raise BenchmarkConfigError("Generated benchmark state has an invalid benchmark record")
        job_id = raw_benchmark.get("generation_job")
        if not isinstance(job_id, str) or job_id not in jobs or job_id in observed_jobs:
            raise BenchmarkConfigError(f"Generated benchmark state has invalid job {job_id!r}")
        observed_jobs.add(job_id)
        job = jobs[job_id]
        artifact = _validate_state_artifact_reference(
            store,
            raw_benchmark.get("artifact"),
            expected_kind="dataset",
            label=f"Generation job {job_id!r}",
            validated=validated_artifacts,
        )
        if raw_benchmark["artifact"]["cache_hit"]:
            dataset_hits += 1

        if "generator_request" in job:
            request = frozen_path(run_root, job["generator_request"], "generator request")
            expected_spec = _scale_stress_generation_spec(
                state=state,
                job=job,
                request=request,
                input_root=frozen_repository_root(run_root),
                resources=generation_resources,
            )
        else:
            scenario = frozen_path(run_root, job["scenario"], "scenario")
            simulator_runs = frozen_path(run_root, job["simulator_runs"], "simulator runs")
            expected_spec = _generation_spec(
                state=state,
                job=job,
                scenario=scenario,
                simulator_runs=simulator_runs,
                input_root=frozen_repository_root(run_root),
                resources=generation_resources,
            )
        expected_key = store.key_for("dataset", expected_spec)
        if artifact.key != expected_key or artifact.manifest.get("spec") != expected_spec:
            raise BenchmarkConfigError(
                f"Generation job {job_id!r} artifact differs from its frozen request"
            )

        manifest_path = unique_direct_manifest(artifact.path / "output", "benchmark-manifest.json")
        recorded_manifest = _artifact_member_from_state(
            artifact,
            raw_benchmark.get("benchmark_manifest"),
            f"Generation job {job_id!r} benchmark manifest",
        )
        if recorded_manifest != manifest_path:
            raise BenchmarkConfigError(
                f"Generation job {job_id!r} records a non-canonical benchmark manifest"
            )
        expected_datasets = datasets_from_benchmark(manifest_path, job_id)
        expected_ids = [item["dataset_id"] for item in expected_datasets]
        if raw_benchmark.get("dataset_ids") != expected_ids:
            raise BenchmarkConfigError(
                f"Generation job {job_id!r} dataset IDs differ from its immutable manifest"
            )
        for expected in expected_datasets:
            dataset_id = expected["dataset_id"]
            if dataset_id in expected_dataset_ids:
                raise BenchmarkConfigError(f"Generated benchmark repeats dataset {dataset_id!r}")
            expected_dataset_ids.add(dataset_id)
            actual = datasets_by_id.get(dataset_id)
            if actual is None:
                raise BenchmarkConfigError(f"Generated benchmark omits dataset {dataset_id!r}")
            expected_record = {
                **expected,
                "dataset_role": job["dataset_role"],
                "split_id": job["split_id"],
                "condition_id": job["condition_id"],
            }
            expected_fields = set(expected_record) | {"dataset_artifact"}
            if set(actual) != expected_fields or any(
                actual.get(field) != value for field, value in expected_record.items()
            ):
                raise BenchmarkConfigError(
                    f"Dataset {dataset_id!r} attribution differs from its frozen manifest"
                )
            if actual.get("dataset_artifact") != raw_benchmark["artifact"]:
                raise BenchmarkConfigError(
                    f"Dataset {dataset_id!r} references a different generation artifact"
                )

    if observed_jobs != set(jobs) or expected_dataset_ids != set(datasets_by_id):
        raise BenchmarkConfigError("Generated benchmark state contains unplanned jobs or datasets")
    cache = state.get("cache")
    if not isinstance(cache, dict) or cache.get("dataset_hits") != dataset_hits:
        raise BenchmarkConfigError("Generated benchmark cache attribution is inconsistent")


def _validate_inference_state_contract(
    *,
    state: dict[str, Any],
    profile_id: str,
    run_root: Path,
) -> dict[tuple[str, str, str], Artifact]:
    """Rebuild every inference record from frozen requests and artifact specifications."""

    frozen_root = frozen_repository_root(run_root)
    profile = frozen_profile(run_root, profile_id)
    store = _artifact_store(run_root)
    _validate_generated_state_contract(
        state=state,
        profile=profile,
        run_root=run_root,
        store=store,
    )
    raw_runs = state.get("inference_runs")
    if not isinstance(raw_runs, list):
        raise BenchmarkConfigError("Inference state has no valid run collection")
    actual_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for actual in raw_runs:
        if not isinstance(actual, dict):
            raise BenchmarkConfigError("Inference state contains a non-object run")
        identity = (
            str(actual.get("dataset_id", "")),
            str(actual.get("batch_id", "")),
            str(actual.get("run_id", "")),
        )
        if not all(identity) or identity in actual_by_identity:
            raise BenchmarkConfigError(
                f"Inference state has invalid duplicate identity {identity!r}"
            )
        actual_by_identity[identity] = actual

    resources = profile["resources"]["infer_network"]
    per_run_resources = {
        key: value for key, value in resources.items() if key != "max_parallel_runs"
    }
    expected_identities: set[tuple[str, str, str]] = set()
    verified: dict[tuple[str, str, str], Artifact] = {}
    validated_artifacts: dict[tuple[str, str], Artifact] = {}
    inference_hits = 0
    for dataset in state["datasets"]:
        for batch in profile["inference_batches"]:
            if not _batch_selects_dataset(batch, dataset):
                continue
            params_payload = load_json_object(
                frozen_path(run_root, batch["tools_params"], "tools params"), "tools params"
            )
            metadata_payload = load_run_metadata(
                frozen_root,
                batch["run_metadata"],
                f"{batch['id']} run metadata",
            )
            metadata_by_run = {item["run_id"]: item for item in metadata_payload["runs"]}
            custom_payload = (
                load_json_object(
                    frozen_path(run_root, batch["custom_tools"], "custom tools"),
                    "custom tools",
                )
                if batch.get("custom_tools")
                else None
            )
            for run in params_payload["runs"]:
                run_id = str(run["run_id"])
                identity = (str(dataset["dataset_id"]), str(batch["id"]), run_id)
                if identity in expected_identities:
                    raise BenchmarkConfigError(
                        f"Frozen profile repeats inference identity {identity!r}"
                    )
                expected_identities.add(identity)
                actual = actual_by_identity.get(identity)
                if actual is None:
                    raise BenchmarkConfigError(f"Inference state omits planned run {identity!r}")
                metadata = metadata_by_run.get(run_id)
                if metadata is None:
                    raise BenchmarkConfigError(
                        f"Frozen metadata omits inference run {run_id!r} in batch {batch['id']!r}"
                    )
                custom_definition = (
                    _single_custom_definition(custom_payload, run_id)
                    if custom_payload is not None
                    else None
                )
                if custom_definition is not None and batch["kind"] == "spathi":
                    custom_definition = {
                        **custom_definition,
                        "docker_image": state["spathi_image"]["id"],
                    }
                tool_image = _runtime_image_contract(
                    state=state,
                    frozen_root=frozen_root,
                    batch_kind=batch["kind"],
                    run=run,
                )
                expected_spec = _inference_spec(
                    state=state,
                    dataset=dataset,
                    batch=batch,
                    run=run,
                    metadata=metadata,
                    custom_definition=custom_definition,
                    tool_image=tool_image,
                    resources=per_run_resources,
                )
                artifact = _validate_state_artifact_reference(
                    store,
                    actual.get("artifact"),
                    expected_kind="inference",
                    label=f"Inference run {identity!r}",
                    validated=validated_artifacts,
                )
                expected_key = store.key_for("inference", expected_spec)
                if artifact.key != expected_key or artifact.manifest.get("spec") != expected_spec:
                    raise BenchmarkConfigError(
                        f"Inference run {identity!r} artifact differs from its frozen request"
                    )
                canonical_report = unique_direct_manifest(
                    artifact.path / "output", "run_report.json"
                )
                recorded_report = _artifact_member_from_state(
                    artifact,
                    actual.get("run_report"),
                    f"Inference run {identity!r} report",
                )
                if recorded_report != canonical_report:
                    raise BenchmarkConfigError(
                        f"Inference run {identity!r} records a non-canonical run report"
                    )
                expected_origin = "custom" if batch["kind"] == "spathi" else "catalog"
                report = _validate_single_run_report(
                    canonical_report,
                    run_id=run_id,
                    dataset_id=dataset["dataset_id"],
                    expected_origin=expected_origin,
                )
                _validate_single_run_preflight(
                    artifact.file("request/preflight.json"),
                    run_id=run_id,
                    dataset_id=str(dataset["dataset_id"]),
                    expected_origin=expected_origin,
                )
                configuration_sha256, execution_sha256 = _parameter_hashes(run, metadata)
                expected_record = {
                    "dataset_id": dataset["dataset_id"],
                    "batch_id": batch["id"],
                    "batch_kind": batch["kind"],
                    "regime": batch["regime"],
                    "run_id": run_id,
                    "tool_id": run["tool_id"],
                    "tool_origin": expected_origin,
                    "configuration_id": metadata["configuration_id"],
                    "selection_role": metadata["selection_role"],
                    "algorithm_seed": metadata["algorithm_seed"],
                    "algorithm_seed_parameter": metadata["algorithm_seed_parameter"],
                    "family": metadata.get("family"),
                    "configuration_sha256": configuration_sha256,
                    "execution_sha256": execution_sha256,
                    "dataset_fingerprint": (report.get("dataset") or {}).get("fingerprint"),
                    "tool_image": tool_image,
                    "run_report": str(canonical_report),
                }
                if set(actual) != set(expected_record) | {"artifact"} or any(
                    actual.get(field) != value for field, value in expected_record.items()
                ):
                    raise BenchmarkConfigError(
                        f"Inference run {identity!r} attribution differs from frozen inputs"
                    )
                if actual["artifact"]["cache_hit"]:
                    inference_hits += 1
                verified[identity] = artifact

    if expected_identities != set(actual_by_identity):
        raise BenchmarkConfigError("Inference state contains unplanned runs")
    cache = state.get("cache")
    if not isinstance(cache, dict) or cache.get("inference_hits") != inference_hits:
        raise BenchmarkConfigError("Inference cache attribution is inconsistent")
    if status_at_least(state, "evaluated"):
        _validate_individual_evaluation_state_contract(
            state=state,
            store=store,
            verified_inference=verified,
        )
    return verified


def _validate_individual_evaluation_state_contract(
    *,
    state: dict[str, Any],
    store: ArtifactStore,
    verified_inference: dict[tuple[str, str, str], Artifact],
) -> None:
    """Require evaluation attribution to be an exact projection of inference state."""

    raw_evaluations = state.get("evaluations")
    if not isinstance(raw_evaluations, list):
        raise BenchmarkConfigError("Evaluated state has no valid individual evaluations")
    actual_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for actual in raw_evaluations:
        if not isinstance(actual, dict):
            raise BenchmarkConfigError("Evaluated state contains a non-object evaluation")
        identity = (
            str(actual.get("dataset_id", "")),
            str(actual.get("batch_id", "")),
            str(actual.get("run_id", "")),
        )
        if not all(identity) or identity in actual_by_identity:
            raise BenchmarkConfigError(
                f"Evaluated state has invalid duplicate identity {identity!r}"
            )
        actual_by_identity[identity] = actual

    datasets = {item["dataset_id"]: item for item in state["datasets"]}
    inference_by_identity = {
        (item["dataset_id"], item["batch_id"], item["run_id"]): item
        for item in state["inference_runs"]
    }
    validated_artifacts: dict[tuple[str, str], Artifact] = {}
    evaluation_hits = 0
    for identity, inference in inference_by_identity.items():
        actual = actual_by_identity.get(identity)
        if actual is None:
            raise BenchmarkConfigError(f"Evaluated state omits individual evaluation {identity!r}")
        if identity not in verified_inference:
            raise BenchmarkConfigError(f"Evaluation references unverified inference {identity!r}")
        dataset = datasets[inference["dataset_id"]]
        expected_spec = _evaluation_spec(state=state, dataset=dataset, inference=inference)
        artifact = _validate_state_artifact_reference(
            store,
            actual.get("artifact"),
            expected_kind="evaluation",
            label=f"Individual evaluation {identity!r}",
            validated=validated_artifacts,
        )
        expected_key = store.key_for("evaluation", expected_spec)
        if artifact.key != expected_key or artifact.manifest.get("spec") != expected_spec:
            raise BenchmarkConfigError(
                f"Individual evaluation {identity!r} artifact differs from its inference"
            )
        canonical_report = unique_direct_manifest(
            artifact.path / "output", "evaluation_report.json"
        )
        recorded_report = _artifact_member_from_state(
            artifact,
            actual.get("evaluation_report"),
            f"Individual evaluation {identity!r} report",
        )
        if recorded_report != canonical_report:
            raise BenchmarkConfigError(
                f"Individual evaluation {identity!r} records a non-canonical report"
            )
        expected_record = {
            **{field: inference[field] for field in _INFERENCE_ATTRIBUTION_FIELDS},
            "run_report": inference["run_report"],
            "evaluation_report": str(canonical_report),
            "inference_artifact": inference["artifact"],
        }
        if set(actual) != set(expected_record) | {"artifact"} or any(
            actual.get(field) != value for field, value in expected_record.items()
        ):
            raise BenchmarkConfigError(
                f"Individual evaluation {identity!r} attribution differs from inference state"
            )
        if actual["artifact"]["cache_hit"]:
            evaluation_hits += 1

    if set(actual_by_identity) != set(inference_by_identity):
        raise BenchmarkConfigError("Evaluated state contains unplanned individual evaluations")
    cache = state.get("cache")
    if not isinstance(cache, dict) or cache.get("evaluation_hits") != evaluation_hits:
        raise BenchmarkConfigError("Evaluation cache attribution is inconsistent")


def validate_inference_attribution_contract(
    *, repo_root: Path, profile_id: str, run_root: Path
) -> None:
    """Fail closed if mutable state no longer describes the immutable inference artifacts."""

    state = _load_existing_state(repo_root, profile_id, run_root)
    if state is None or not status_at_least(state, "inferred"):
        raise BenchmarkConfigError("Inference attribution validation requires completed inference")
    _validate_inference_state_contract(state=state, profile_id=profile_id, run_root=run_root)


def _execute_unique_missing_tasks(
    prepared: list[dict[str, Any]],
    *,
    max_workers: int,
    thread_name_prefix: str,
    execute: Callable[[dict[str, Any]], Artifact],
) -> tuple[int, ...]:
    """Execute one representative per missing artifact key and bind every alias.

    A profile may project one physical inference or evaluation into several
    classificatory regimes.  Those logical records remain distinct downstream,
    while this helper guarantees that their shared immutable artifact is produced
    only once inside the current pipeline invocation.
    """

    groups: dict[ArtifactKey, list[int]] = {}
    for index, task in enumerate(prepared):
        if task["artifact"] is None:
            groups.setdefault(task["key"], []).append(index)
    if not groups:
        return ()

    representatives: list[tuple[ArtifactKey, dict[str, Any]]] = []
    for key, indices in groups.items():
        representative = prepared[indices[0]]
        if any(prepared[index]["spec"] != representative["spec"] for index in indices[1:]):
            raise BenchmarkConfigError(
                f"Tasks sharing artifact key {key} have different execution specifications"
            )
        representatives.append((key, representative))

    artifacts: dict[ArtifactKey, Artifact] = {}
    workers = min(max_workers, len(representatives))
    if workers == 1:
        for key, task in representatives:
            artifacts[key] = execute(task)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=thread_name_prefix) as pool:
            futures = {pool.submit(execute, task): key for key, task in representatives}
            for future in as_completed(futures):
                artifacts[futures[future]] = future.result()

    reused_indices: list[int] = []
    for key, indices in groups.items():
        artifact = artifacts[key]
        if artifact.key != key:
            raise BenchmarkConfigError(
                f"Task executor returned artifact {artifact.key} for requested key {key}"
            )
        for index in indices:
            prepared[index]["artifact"] = artifact
        for index in indices[1:]:
            prepared[index]["cache_hit"] = True
            reused_indices.append(index)
    return tuple(reused_indices)


def run_inference(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    andrea_repo: Path,
) -> Path:
    state = _load_existing_state(repo_root, profile_id, run_root)
    if state is None:
        raise BenchmarkConfigError("Inference requires a generated benchmark")
    _validate_docker_runtime(state, docker_runtime_environment())
    if status_at_least(state, "inferred"):
        _validate_inference_state_contract(
            state=state,
            profile_id=profile_id,
            run_root=run_root,
        )
        return state_path(run_root)
    require_status(state, "generated", "Inference")
    frozen_root = frozen_repository_root(run_root)
    observed_andrea = _validate_andrea_repository(frozen_root, andrea_repo)
    _validate_state_andrea(state, observed_andrea)
    _validate_host_environment(state, andrea_host_environment())
    profile = frozen_profile(run_root, profile_id)
    resources = profile["resources"]["infer_network"]
    per_run_resources = {
        key: value for key, value in resources.items() if key != "max_parallel_runs"
    }
    store = _artifact_store(run_root)
    _validate_generated_state_contract(
        state=state,
        profile=profile,
        run_root=run_root,
        store=store,
    )
    inference_runs: list[dict[str, Any]] = []
    inference_hits = 0
    prepared: list[dict[str, Any]] = []
    ensured_spathi_image = False

    for dataset in state["datasets"]:
        dataset_manifest, _ground_truth_manifest = _validated_dataset_paths(store, dataset)
        for batch in profile["inference_batches"]:
            if not _batch_selects_dataset(batch, dataset):
                continue
            params_path = frozen_path(run_root, batch["tools_params"], "tools params")
            params_payload = load_json_object(params_path, "tools params")
            metadata_payload = load_run_metadata(
                frozen_root,
                batch["run_metadata"],
                f"{batch['id']} run metadata",
            )
            metadata_by_run = {item["run_id"]: item for item in metadata_payload["runs"]}
            custom_payload = (
                load_json_object(
                    frozen_path(run_root, batch["custom_tools"], "custom tools"),
                    "custom tools",
                )
                if batch.get("custom_tools")
                else None
            )

            for run in params_payload["runs"]:
                run_id = str(run["run_id"])
                metadata = metadata_by_run[run_id]
                custom_definition = (
                    _single_custom_definition(custom_payload, run_id)
                    if custom_payload is not None
                    else None
                )
                if custom_definition is not None and batch["kind"] == "spathi":
                    custom_definition = {
                        **custom_definition,
                        "docker_image": state["spathi_image"]["id"],
                    }
                tool_image = _runtime_image_contract(
                    state=state,
                    frozen_root=frozen_root,
                    batch_kind=batch["kind"],
                    run=run,
                )
                spec = _inference_spec(
                    state=state,
                    dataset=dataset,
                    batch=batch,
                    run=run,
                    metadata=metadata,
                    custom_definition=custom_definition,
                    tool_image=tool_image,
                    resources=per_run_resources,
                )
                key = store.key_for("inference", spec)
                artifact = store.lookup(key)
                cache_hit = artifact is not None
                expected_origin = "custom" if batch["kind"] == "spathi" else "catalog"
                if artifact is None and batch["kind"] == "spathi":
                    if not ensured_spathi_image:
                        _ensure_runtime_image(
                            state=state,
                            frozen_root=frozen_root,
                            batch_kind=batch["kind"],
                            run=run,
                        )
                        ensured_spathi_image = True
                else:
                    if artifact is not None:
                        inference_hits += 1
                prepared.append(
                    {
                        "dataset": dataset,
                        "dataset_manifest": dataset_manifest,
                        "batch": batch,
                        "run": run,
                        "metadata": metadata,
                        "custom_definition": custom_definition,
                        "tool_image": tool_image,
                        "spec": spec,
                        "key": key,
                        "artifact": artifact,
                        "cache_hit": cache_hit,
                        "expected_origin": expected_origin,
                    }
                )

    reused_indices = _execute_unique_missing_tasks(
        prepared,
        max_workers=int(resources["max_parallel_runs"]),
        thread_name_prefix="spathi-infer",
        execute=lambda task: _execute_inference_task(
            task,
            store=store,
            repo_root=repo_root,
            andrea_repo=andrea_repo,
            run_root=run_root,
            resources=per_run_resources,
            state=state,
            frozen_root=frozen_root,
        ),
    )
    inference_hits += len(reused_indices)

    for task in prepared:
        dataset = task["dataset"]
        batch = task["batch"]
        run = task["run"]
        metadata = task["metadata"]
        artifact = task["artifact"]
        if not isinstance(artifact, Artifact):
            raise RuntimeError("Inference task completed without an artifact")
        run_id = str(run["run_id"])
        expected_origin = str(task["expected_origin"])
        report_path = unique_direct_manifest(artifact.path / "output", "run_report.json")
        report = _validate_single_run_report(
            report_path,
            run_id=run_id,
            dataset_id=dataset["dataset_id"],
            expected_origin=expected_origin,
        )
        configuration_sha256, execution_sha256 = _parameter_hashes(run, metadata)
        inference_runs.append(
            {
                "dataset_id": dataset["dataset_id"],
                "batch_id": batch["id"],
                "batch_kind": batch["kind"],
                "regime": batch["regime"],
                "run_id": run_id,
                "tool_id": run["tool_id"],
                "tool_origin": expected_origin,
                "configuration_id": metadata["configuration_id"],
                "selection_role": metadata["selection_role"],
                "algorithm_seed": metadata["algorithm_seed"],
                "algorithm_seed_parameter": metadata["algorithm_seed_parameter"],
                "family": metadata.get("family"),
                "configuration_sha256": configuration_sha256,
                "execution_sha256": execution_sha256,
                "dataset_fingerprint": (report.get("dataset") or {}).get("fingerprint"),
                "tool_image": task["tool_image"],
                "run_report": str(report_path),
                "artifact": _artifact_ref(artifact, cache_hit=bool(task["cache_hit"])),
            }
        )

    state["inference_runs"] = inference_runs
    state["cache"]["inference_hits"] = inference_hits
    state["status"] = "inferred"
    state["updated_at"] = _utc_now()
    return write_state(run_root, state)


def validate_handoffs(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
) -> Path:
    state = _load_existing_state(repo_root, profile_id, run_root)
    if state is None:
        raise BenchmarkConfigError("Handoff validation requires inference")
    verified_inference = _validate_inference_state_contract(
        state=state,
        profile_id=profile_id,
        run_root=run_root,
    )
    report_path = run_root / "handoff-validation.json"
    if status_at_least(state, "validated"):
        report = load_json_object(report_path, "handoff validation")
        if report.get("status") != "passed":
            raise BenchmarkConfigError("Recorded handoff validation did not pass")
        return report_path
    require_status(state, "inferred", "Handoff validation")
    store = _artifact_store(run_root)
    datasets = {item["dataset_id"]: item for item in state["datasets"]}
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for inference in state["inference_runs"]:
        identity = (inference["dataset_id"], inference["batch_id"], inference["run_id"])
        artifact = verified_inference[identity]
        dataset = datasets[inference["dataset_id"]]
        dataset_manifest, ground_truth_manifest = _validated_dataset_paths(store, dataset)
        report_path_for_run = _artifact_member_from_state(
            artifact,
            inference.get("run_report"),
            "inference run report",
        )
        report = _validate_single_run_report(
            report_path_for_run,
            run_id=inference["run_id"],
            dataset_id=dataset["dataset_id"],
            expected_origin=inference["tool_origin"],
        )
        truth = load_json_object(ground_truth_manifest, "ground truth")
        local_errors: list[str] = []
        calibration_context_contract: dict[str, Any] | None = None
        report_fingerprint = (report.get("dataset") or {}).get("fingerprint")
        if truth.get("dataset_id") != dataset["dataset_id"]:
            local_errors.append("ground-truth dataset ID differs from benchmark dataset ID")
        if report_fingerprint != truth.get("dataset_fingerprint"):
            local_errors.append("inference and ground-truth dataset fingerprints differ")
        if state["purpose"] in {"calibration", "selection"}:
            expected_role = "validation" if state["purpose"] == "selection" else "development"
            if dataset.get("dataset_role") != expected_role:
                local_errors.append(
                    f"calibration dataset role must be {expected_role!r}, not "
                    f"{dataset.get('dataset_role')!r}"
                )
            if inference.get("batch_kind") != "spathi" or inference.get("regime") != "group_only":
                local_errors.append("calibration inference must be a SPATHI group_only execution")
            try:
                contract = load_dataset_group_contract(
                    dataset_manifest,
                    ground_truth_manifest,
                    expected_dataset_id=dataset["dataset_id"],
                )
                completed_contexts = require_complete_group_contexts(
                    report,
                    inference["run_id"],
                    contract.expected_group_contexts,
                )
                calibration_context_contract = {
                    **contract.to_dict(),
                    "completed_group_contexts": list(completed_contexts),
                }
            except BenchmarkConfigError as exc:
                local_errors.append(str(exc))
        merged_raw = (report.get("outputs") or {}).get("merged_network_raw")
        try:
            merged_path = resolve_manifest_member(
                report_path_for_run.parent,
                merged_raw,
                "merged network",
            )
        except BenchmarkConfigError as exc:
            local_errors.append(str(exc))
            merged_path = None
        result = {
            "dataset_id": dataset["dataset_id"],
            "run_id": inference["run_id"],
            "artifact_key": inference["artifact"]["key"],
            "merged_network_raw": str(merged_path) if merged_path else None,
            "status": "passed" if not local_errors else "failed",
            "errors": local_errors,
        }
        if state["purpose"] in {"calibration", "selection"}:
            result["calibration_context_contract"] = calibration_context_contract
        results.append(result)
        errors.extend(
            f"[{dataset['dataset_id']}/{inference['run_id']}] {error}" for error in local_errors
        )

    atomic_write_json(
        report_path,
        {
            "schema_version": "1.0",
            "created_at": _utc_now(),
            "profile_id": profile_id,
            "status": "passed" if not errors else "failed",
            "results": results,
            "errors": errors,
        },
        "handoff validation report",
        boundary=run_root,
    )
    if errors:
        raise BenchmarkConfigError("Handoff validation failed: " + "; ".join(errors))
    state["handoff_validation"] = str(report_path.resolve())
    state["status"] = "validated"
    state["updated_at"] = _utc_now()
    write_state(run_root, state)
    return report_path


def _integration_spec(
    *,
    state: dict[str, Any],
    dataset: dict[str, Any],
    regime: str,
    inference_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact_keys = sorted(str(run["artifact"]["key"]) for run in inference_runs)
    fingerprints = {canonical_json_sha256(run["dataset_fingerprint"]) for run in inference_runs}
    if len(fingerprints) != 1:
        raise BenchmarkConfigError(
            f"Inference runs for {dataset['dataset_id']!r}/{regime!r} have different "
            "dataset fingerprints"
        )
    return {
        "schema_version": "1.0",
        "integration_contract": "joint-inference-view-v1",
        "spathi_benchmarks_commit": state["benchmark_repository"]["commit"],
        "andrea_commit": state["repositories"]["observed"]["andrea"]["commit"],
        "andrea_environment": state["host_environment"]["fingerprint"],
        "dataset_artifact": dataset["dataset_artifact"]["key"],
        "dataset_id": dataset["dataset_id"],
        "dataset_fingerprint": inference_runs[0]["dataset_fingerprint"],
        "regime": regime,
        "inference_artifacts": artifact_keys,
    }


def _joint_run_id(dataset_id: str, regime: str) -> str:
    return f"joint-{_log_token(dataset_id)}-{_log_token(regime)}"


def _validate_integrated_run_report(
    report_path: Path,
    *,
    dataset_id: str,
    regime: str,
    run_id: str,
    source_run_ids: list[str],
    benchmark_commit: str,
) -> dict[str, Any]:
    report = load_json_object(report_path, "integrated inference run report")
    integration = report.get("integration")
    tools = report.get("tools")
    if report.get("run_id") != run_id or report.get("status") != "executed":
        raise BenchmarkConfigError(
            "Integrated inference run report has the wrong identity or status"
        )
    if (report.get("dataset") or {}).get("id") != dataset_id:
        raise BenchmarkConfigError("Integrated inference run report has the wrong dataset identity")
    if not isinstance(integration, dict) or any(
        (
            integration.get("schema_version") != "1.0",
            integration.get("kind") != "joint-inference-view",
            integration.get("producer") != "spathi-benchmarks",
            integration.get("producer_commit") != benchmark_commit,
            integration.get("regime") != regime,
            integration.get("monolithic_execution") is not False,
        )
    ):
        raise BenchmarkConfigError(
            "Integrated inference run report has an invalid integration contract"
        )
    if not isinstance(tools, dict) or sorted(tools.get("selected") or []) != sorted(source_run_ids):
        raise BenchmarkConfigError("Integrated inference run report has the wrong tool run IDs")
    sources = integration.get("sources")
    observed_source_ids = (
        sorted(str(source.get("source_run_id")) for source in sources if isinstance(source, dict))
        if isinstance(sources, list)
        else []
    )
    if observed_source_ids != sorted(source_run_ids):
        raise BenchmarkConfigError("Integrated inference provenance has the wrong source run IDs")
    return report


def _execute_integration_task(
    task: dict[str, Any],
    *,
    store: ArtifactStore,
    repo_root: Path,
    andrea_repo: Path,
    run_root: Path,
) -> Artifact:
    dataset_id = str(task["dataset"]["dataset_id"])
    regime = str(task["regime"])
    with store.staging_directory(task["key"]) as staging:
        output_root = staging / "output"

        def validate_contract(report_path: Path) -> None:
            andrea_contract_command(
                [
                    "validate-inference",
                    "--andrea-repo",
                    str(andrea_repo),
                    "--run-report",
                    str(report_path),
                ],
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                log_path=run_root
                / "logs"
                / f"35-{_log_token(dataset_id)}-{_log_token(regime)}.log",
            )

        integrate_inference_runs(
            run_report_paths=task["run_report_paths"],
            output_dir=output_root,
            combined_run_id=task["run_id"],
            regime=regime,
            benchmark_commit=task["benchmark_commit"],
            contract_validator=validate_contract,
        )
        return store.publish_from_temp(
            staging,
            kind="joint-inference",
            spec=task["spec"],
        )


def integrate_inference(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    andrea_repo: Path,
) -> Path:
    """Build one ANDREA-compatible joint inference view per dataset and regime."""

    state = _load_existing_state(repo_root, profile_id, run_root)
    if state is None:
        raise BenchmarkConfigError("Inference integration requires validated inference")
    store = _artifact_store(run_root)
    _validate_inference_state_contract(
        state=state,
        profile_id=profile_id,
        run_root=run_root,
    )
    if status_at_least(state, "integrated"):
        for integrated in state["integrated_inference_runs"]:
            artifact = _artifact_from_ref(store, integrated["artifact"])
            report_path = _artifact_member_from_state(
                artifact,
                integrated.get("run_report"),
                "integrated inference run report",
            )
            _validate_integrated_run_report(
                report_path,
                dataset_id=integrated["dataset_id"],
                regime=integrated["regime"],
                run_id=integrated["run_id"],
                source_run_ids=integrated["source_run_ids"],
                benchmark_commit=state["benchmark_repository"]["commit"],
            )
        return state_path(run_root)
    require_status(state, "validated", "Inference integration")
    frozen_root = frozen_repository_root(run_root)
    observed_andrea = _validate_andrea_repository(frozen_root, andrea_repo)
    _validate_state_andrea(state, observed_andrea)
    _validate_host_environment(state, andrea_host_environment())

    datasets = {item["dataset_id"]: item for item in state["datasets"]}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for inference in state["inference_runs"]:
        dataset_id = str(inference["dataset_id"])
        regime = str(inference["regime"])
        if dataset_id not in datasets:
            raise BenchmarkConfigError(
                f"Inference run references an unknown benchmark dataset: {dataset_id!r}"
            )
        grouped[(dataset_id, regime)].append(inference)
    if not grouped:
        raise BenchmarkConfigError("Inference integration found no dataset/regime groups")

    integrated_runs: list[dict[str, Any]] = []
    integration_hits = 0
    for (dataset_id, regime), inference_runs in sorted(grouped.items()):
        ordered_runs = sorted(inference_runs, key=lambda item: str(item["run_id"]))
        source_run_ids = [str(item["run_id"]) for item in ordered_runs]
        if len(set(source_run_ids)) != len(source_run_ids):
            raise BenchmarkConfigError(
                f"Duplicate inference run IDs in {dataset_id!r}/{regime!r}: {source_run_ids}"
            )
        report_paths: list[Path] = []
        for inference in ordered_runs:
            artifact = _artifact_from_ref(store, inference["artifact"])
            report_paths.append(
                _artifact_member_from_state(
                    artifact,
                    inference.get("run_report"),
                    "inference run report",
                )
            )
        dataset = datasets[dataset_id]
        spec = _integration_spec(
            state=state,
            dataset=dataset,
            regime=regime,
            inference_runs=ordered_runs,
        )
        key = store.key_for("joint-inference", spec)
        artifact = store.lookup(key)
        cache_hit = artifact is not None
        run_id = _joint_run_id(dataset_id, regime)
        if artifact is None:
            artifact = _execute_integration_task(
                {
                    "key": key,
                    "spec": spec,
                    "dataset": dataset,
                    "regime": regime,
                    "run_id": run_id,
                    "benchmark_commit": state["benchmark_repository"]["commit"],
                    "run_report_paths": report_paths,
                },
                store=store,
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                run_root=run_root,
            )
        else:
            integration_hits += 1
        report_path = artifact.file("output/run_report.json").resolve()
        report = _validate_integrated_run_report(
            report_path,
            dataset_id=dataset_id,
            regime=regime,
            run_id=run_id,
            source_run_ids=source_run_ids,
            benchmark_commit=state["benchmark_repository"]["commit"],
        )
        integrated_runs.append(
            {
                "dataset_id": dataset_id,
                "regime": regime,
                "run_id": run_id,
                "spathi_benchmarks_commit": state["benchmark_repository"]["commit"],
                "source_run_ids": source_run_ids,
                "source_inference_artifacts": [
                    str(item["artifact"]["key"]) for item in ordered_runs
                ],
                "dataset_fingerprint": (report.get("dataset") or {}).get("fingerprint"),
                "run_report": str(report_path),
                "artifact": _artifact_ref(artifact, cache_hit=cache_hit),
            }
        )

    state["integrated_inference_runs"] = integrated_runs
    state["cache"]["integration_hits"] = integration_hits
    state["status"] = "integrated"
    state["updated_at"] = _utc_now()
    return write_state(run_root, state)


def _evaluation_spec(
    *,
    state: dict[str, Any],
    dataset: dict[str, Any],
    inference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "andrea_commit": state["repositories"]["observed"]["andrea"]["commit"],
        "andrea_environment": state["host_environment"]["fingerprint"],
        "inference_artifact": inference["artifact"]["key"],
        "dataset_artifact": dataset["dataset_artifact"]["key"],
        "dataset_fingerprint": inference["dataset_fingerprint"],
        "evaluation_contract": "evaluate-inference-v1",
    }


def _execute_evaluation_task(
    task: dict[str, Any],
    *,
    store: ArtifactStore,
    repo_root: Path,
    andrea_repo: Path,
    run_root: Path,
) -> Artifact:
    inference = task["inference"]
    dataset = task["dataset"]
    with store.staging_directory(task["key"]) as staging:
        output_root = staging / "output"
        andrea_command(
            [
                "evaluate-inference",
                "--run-report",
                str(task["run_report"]),
                "--ground-truth-manifest",
                str(task["ground_truth_manifest"]),
                "--output-dir",
                str(output_root),
                "--no-view",
            ],
            repo_root=repo_root,
            andrea_repo=andrea_repo,
            log_path=run_root
            / "logs"
            / (
                f"50-{_log_token(dataset['dataset_id'])}-{_log_token(task['scope'])}-"
                f"{_log_token(inference['run_id'])}.log"
            ),
        )
        unique_direct_manifest(output_root, "evaluation_report.json")
        return store.publish_from_temp(
            staging,
            kind="evaluation",
            spec=task["spec"],
        )


def evaluate(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    andrea_repo: Path,
) -> Path:
    state = _load_existing_state(repo_root, profile_id, run_root)
    if state is None:
        raise BenchmarkConfigError("Evaluation requires integrated inference")
    _validate_inference_state_contract(
        state=state,
        profile_id=profile_id,
        run_root=run_root,
    )
    if status_at_least(state, "evaluated"):
        store = _artifact_store(run_root)
        for collection in (state["evaluations"], state["joint_evaluations"]):
            for evaluation in collection:
                artifact = _artifact_from_ref(store, evaluation["artifact"])
                _artifact_member_from_state(
                    artifact,
                    evaluation.get("evaluation_report"),
                    "evaluation report",
                )
        return state_path(run_root)
    require_status(state, "integrated", "Evaluation")
    frozen_root = frozen_repository_root(run_root)
    observed_andrea = _validate_andrea_repository(frozen_root, andrea_repo)
    _validate_state_andrea(state, observed_andrea)
    _validate_host_environment(state, andrea_host_environment())
    store = _artifact_store(run_root)
    profile = frozen_profile(run_root, profile_id)
    max_parallel_runs = int(profile["resources"]["infer_network"]["max_parallel_runs"])
    datasets = {item["dataset_id"]: item for item in state["datasets"]}
    evaluations: list[dict[str, Any]] = []
    joint_evaluations: list[dict[str, Any]] = []
    evaluation_hits = 0
    joint_evaluation_hits = 0
    prepared: list[dict[str, Any]] = []

    for scope, inference_collection in (
        ("individual", state["inference_runs"]),
        ("joint", state["integrated_inference_runs"]),
    ):
        for inference in inference_collection:
            inference_artifact = _artifact_from_ref(store, inference["artifact"])
            dataset = datasets[inference["dataset_id"]]
            _dataset_manifest, ground_truth_manifest = _validated_dataset_paths(store, dataset)
            run_report = _artifact_member_from_state(
                inference_artifact,
                inference.get("run_report"),
                "inference run report",
            )
            spec = _evaluation_spec(state=state, dataset=dataset, inference=inference)
            key = store.key_for("evaluation", spec)
            artifact = store.lookup(key)
            cache_hit = artifact is not None
            if artifact is not None:
                if scope == "individual":
                    evaluation_hits += 1
                else:
                    joint_evaluation_hits += 1
            prepared.append(
                {
                    "scope": scope,
                    "inference": inference,
                    "dataset": dataset,
                    "run_report": run_report,
                    "ground_truth_manifest": ground_truth_manifest,
                    "spec": spec,
                    "key": key,
                    "artifact": artifact,
                    "cache_hit": cache_hit,
                }
            )

    reused_indices = _execute_unique_missing_tasks(
        prepared,
        max_workers=max_parallel_runs,
        thread_name_prefix="spathi-eval",
        execute=lambda task: _execute_evaluation_task(
            task,
            store=store,
            repo_root=repo_root,
            andrea_repo=andrea_repo,
            run_root=run_root,
        ),
    )
    for index in reused_indices:
        if prepared[index]["scope"] == "individual":
            evaluation_hits += 1
        else:
            joint_evaluation_hits += 1

    for task in prepared:
        inference = task["inference"]
        artifact = task["artifact"]
        if not isinstance(artifact, Artifact):
            raise RuntimeError("Evaluation task completed without an artifact")
        evaluation_report = unique_direct_manifest(
            artifact.path / "output",
            "evaluation_report.json",
        )
        common = {
            "run_report": inference["run_report"],
            "evaluation_report": str(evaluation_report),
            "inference_artifact": inference["artifact"],
            "artifact": _artifact_ref(artifact, cache_hit=bool(task["cache_hit"])),
        }
        if task["scope"] == "individual":
            evaluations.append(
                {
                    **{
                        key: inference[key]
                        for key in (
                            "dataset_id",
                            "batch_id",
                            "batch_kind",
                            "regime",
                            "run_id",
                            "tool_id",
                            "tool_origin",
                            "configuration_id",
                            "selection_role",
                            "algorithm_seed",
                            "algorithm_seed_parameter",
                            "family",
                            "configuration_sha256",
                            "execution_sha256",
                            "tool_image",
                        )
                    },
                    **common,
                }
            )
        else:
            joint_evaluations.append(
                {
                    **{
                        key: inference[key]
                        for key in (
                            "dataset_id",
                            "regime",
                            "run_id",
                            "source_run_ids",
                            "source_inference_artifacts",
                            "dataset_fingerprint",
                            "spathi_benchmarks_commit",
                        )
                    },
                    **common,
                }
            )

    state["evaluations"] = evaluations
    state["joint_evaluations"] = joint_evaluations
    state["cache"]["evaluation_hits"] = evaluation_hits
    state["cache"]["joint_evaluation_hits"] = joint_evaluation_hits
    state["status"] = "evaluated"
    state["updated_at"] = _utc_now()
    return write_state(run_root, state)


def _analysis_bundle_spec(
    *,
    state: dict[str, Any],
    source_artifact: dict[str, Any],
    producer_command: str,
    dataset_id: str,
    scope: str,
    regime: str | None,
    run_id: str | None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "dataset_id": dataset_id,
        "scope": scope,
    }
    if regime is not None:
        identity["regime"] = regime
    if run_id is not None:
        identity["run_id"] = run_id
    return {
        "schema_version": "1.0",
        "bundle_contract": "andrea-analysis-v1",
        "spathi_benchmarks_commit": state["benchmark_repository"]["commit"],
        "andrea_commit": state["repositories"]["observed"]["andrea"]["commit"],
        "andrea_environment": state["host_environment"]["fingerprint"],
        "producer_command": producer_command,
        "source_artifact": source_artifact["key"],
        "identity": identity,
    }


def _validate_analysis_bundle_artifact(artifact: Artifact) -> tuple[Path, Path]:
    archive = artifact.file("analysis.zip").resolve()
    manifest_path = artifact.file("analysis.zip.manifest.json").resolve()
    manifest = load_json_object(manifest_path, "analysis bundle manifest")
    archive_entry = manifest.get("archive")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("status") != "complete"
        or not isinstance(archive_entry, dict)
        or archive_entry.get("filename") != "analysis.zip"
        or archive_entry.get("sha256") != sha256_file(archive)
        or archive_entry.get("size") != archive.stat().st_size
    ):
        raise BenchmarkConfigError(f"Invalid analysis bundle artifact: {artifact.key}")
    bundle = manifest.get("bundle")
    if not isinstance(bundle, dict) or bundle.get("id") != "analysis":
        raise BenchmarkConfigError(
            f"Analysis bundle manifest has the wrong bundle ID: {artifact.key}"
        )
    return archive, manifest_path


def _execute_analysis_export_task(
    task: dict[str, Any],
    *,
    store: ArtifactStore,
    repo_root: Path,
    andrea_repo: Path,
    run_root: Path,
) -> Artifact:
    with store.staging_directory(task["key"]) as staging:
        arguments = [
            "export-analysis",
            "--andrea-repo",
            str(andrea_repo),
            "--producer-command",
            task["producer_command"],
            "--root",
            str(task["source_root"]),
            "--dataset-id",
            task["dataset_id"],
            "--output",
            str(staging / "analysis.zip"),
        ]
        andrea_contract_command(
            arguments,
            repo_root=repo_root,
            andrea_repo=andrea_repo,
            log_path=run_root
            / "logs"
            / (
                f"60-{_log_token(task['producer_command'])}-{_log_token(task['dataset_id'])}-"
                f"{_log_token(task['scope'])}-{_log_token(task.get('run_id') or 'dataset')}.log"
            ),
        )
        artifact = store.publish_from_temp(
            staging,
            kind="analysis-bundle",
            spec=task["spec"],
        )
    _validate_analysis_bundle_artifact(artifact)
    return artifact


def _export_component(value: Any, label: str) -> str:
    component = str(value)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", component) is None:
        raise BenchmarkConfigError(f"{label} is not a portable export path component: {value!r}")
    return component


def _bundle_export_directory(bundle: dict[str, Any]) -> Path:
    dataset_id = _export_component(bundle["dataset_id"], "dataset_id")
    producer = bundle["producer_command"]
    if producer == "generate-data":
        if bundle["scope"] != "dataset":
            raise BenchmarkConfigError("generate-data analysis bundles must have dataset scope")
        return Path("datasets") / dataset_id
    category = {
        "infer-network": "inference",
        "evaluate-inference": "evaluation",
    }.get(producer)
    if category is None:
        raise BenchmarkConfigError(f"Unknown analysis bundle producer: {producer!r}")
    regime = _export_component(bundle.get("regime"), "regime")
    scope = bundle.get("scope")
    if scope == "joint":
        return Path(category) / dataset_id / regime / "joint"
    if scope == "individual":
        run_id = _export_component(bundle.get("run_id"), "run_id")
        return Path(category) / dataset_id / regime / "individual" / run_id
    raise BenchmarkConfigError(f"Invalid analysis bundle scope: {scope!r}")


def _tree_file_hashes(root: Path) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise BenchmarkConfigError(f"Export tree is missing or invalid: {root}")
    files: dict[str, dict[str, Any]] = {}
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise BenchmarkConfigError(
                    f"Export tree contains a symbolic link: {directory_path / name}"
                )
        for name in file_names:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise BenchmarkConfigError(f"Export tree contains a non-regular file: {path}")
            relative = path.relative_to(root).as_posix()
            files[relative] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return files


def _materialize_exports(
    *,
    run_root: Path,
    profile_id: str,
    bundles: list[dict[str, Any]],
    store: ArtifactStore,
) -> Path:
    destination = run_root / "exports"
    staging = Path(tempfile.mkdtemp(prefix=".exports-", dir=run_root))
    try:
        manifest_entries: list[dict[str, Any]] = []
        observed_directories: set[str] = set()
        for bundle in bundles:
            source_artifact = _artifact_from_ref(store, bundle["source_artifact"])
            bundle_artifact = _artifact_from_ref(store, bundle["artifact"])
            archive, sidecar = _validate_analysis_bundle_artifact(bundle_artifact)
            relative_directory = _bundle_export_directory(bundle)
            directory_key = relative_directory.as_posix()
            if directory_key in observed_directories:
                raise BenchmarkConfigError(
                    f"Analysis bundle export path collision: {directory_key}"
                )
            observed_directories.add(directory_key)
            output_directory = staging / relative_directory
            output_directory.mkdir(parents=True)
            output_archive = output_directory / "analysis.zip"
            output_sidecar = output_directory / "analysis.zip.manifest.json"
            shutil.copyfile(archive, output_archive)
            shutil.copyfile(sidecar, output_sidecar)
            entry = {
                "producer_command": bundle["producer_command"],
                "scope": bundle["scope"],
                "dataset_id": bundle["dataset_id"],
                "analysis_zip": output_archive.relative_to(staging).as_posix(),
                "analysis_manifest": output_sidecar.relative_to(staging).as_posix(),
                "archive_sha256": sha256_file(output_archive),
                "manifest_sha256": sha256_file(output_sidecar),
                "source_artifact": source_artifact.key.value,
                "bundle_artifact": bundle_artifact.key.value,
            }
            if bundle.get("regime") is not None:
                entry["regime"] = bundle["regime"]
            if bundle.get("run_id") is not None:
                entry["run_id"] = bundle["run_id"]
            manifest_entries.append(entry)
        export_manifest = {
            "schema_version": "1.0",
            "status": "complete",
            "profile_id": profile_id,
            "bundles": manifest_entries,
        }
        atomic_write_json(
            staging / "manifest.json",
            export_manifest,
            "analysis export manifest",
            boundary=staging,
        )
        expected_files = _tree_file_hashes(staging)
        if destination.exists():
            if _tree_file_hashes(destination) != expected_files:
                raise BenchmarkConfigError(
                    f"Existing exports directory differs from the immutable bundle set: {destination}"
                )
        else:
            os.rename(staging, destination)
        for bundle, manifest_entry in zip(bundles, manifest_entries, strict=True):
            bundle["export_zip"] = str((destination / manifest_entry["analysis_zip"]).resolve())
            bundle["export_manifest"] = str(
                (destination / manifest_entry["analysis_manifest"]).resolve()
            )
        return (destination / "manifest.json").resolve()
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def export_analysis(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    andrea_repo: Path,
) -> Path:
    """Export official ANDREA analysis ZIPs for every pipeline handoff."""

    state = _load_existing_state(repo_root, profile_id, run_root)
    if state is None:
        raise BenchmarkConfigError("Analysis export requires completed evaluations")
    store = _artifact_store(run_root)
    _validate_inference_state_contract(
        state=state,
        profile_id=profile_id,
        run_root=run_root,
    )
    if status_at_least(state, "exported"):
        state["exports_manifest"] = str(
            _materialize_exports(
                run_root=run_root,
                profile_id=profile_id,
                bundles=state["analysis_bundles"],
                store=store,
            )
        )
        return state_path(run_root)
    require_status(state, "evaluated", "Analysis export")
    frozen_root = frozen_repository_root(run_root)
    observed_andrea = _validate_andrea_repository(frozen_root, andrea_repo)
    _validate_state_andrea(state, observed_andrea)
    _validate_host_environment(state, andrea_host_environment())

    datasets = {item["dataset_id"]: item for item in state["datasets"]}
    tasks: list[dict[str, Any]] = []

    def add_task(
        *,
        producer_command: str,
        scope: str,
        dataset_id: str,
        source_reference: dict[str, Any],
        source_root: Path,
        regime: str | None = None,
        run_id: str | None = None,
    ) -> None:
        spec = _analysis_bundle_spec(
            state=state,
            source_artifact=source_reference,
            producer_command=producer_command,
            dataset_id=dataset_id,
            scope=scope,
            regime=regime,
            run_id=run_id,
        )
        key = store.key_for("analysis-bundle", spec)
        artifact = store.lookup(key)
        tasks.append(
            {
                "producer_command": producer_command,
                "scope": scope,
                "dataset_id": dataset_id,
                "regime": regime,
                "run_id": run_id,
                "source_artifact": source_reference,
                "source_root": source_root,
                "spec": spec,
                "key": key,
                "artifact": artifact,
                "cache_hit": artifact is not None,
            }
        )

    for dataset_id, dataset in sorted(datasets.items()):
        dataset_manifest, _truth = _validated_dataset_paths(store, dataset)
        add_task(
            producer_command="generate-data",
            scope="dataset",
            dataset_id=dataset_id,
            source_reference=dataset["dataset_artifact"],
            source_root=dataset_manifest.parent.parent.parent,
        )
    for scope, collection in (
        ("individual", state["inference_runs"]),
        ("joint", state["integrated_inference_runs"]),
    ):
        for inference in collection:
            artifact = _artifact_from_ref(store, inference["artifact"])
            report = _artifact_member_from_state(
                artifact,
                inference.get("run_report"),
                "inference run report",
            )
            add_task(
                producer_command="infer-network",
                scope=scope,
                dataset_id=inference["dataset_id"],
                regime=inference["regime"],
                run_id=inference["run_id"] if scope == "individual" else None,
                source_reference=inference["artifact"],
                source_root=report.parent,
            )
    for scope, collection in (
        ("individual", state["evaluations"]),
        ("joint", state["joint_evaluations"]),
    ):
        for evaluation in collection:
            artifact = _artifact_from_ref(store, evaluation["artifact"])
            report = _artifact_member_from_state(
                artifact,
                evaluation.get("evaluation_report"),
                "evaluation report",
            )
            add_task(
                producer_command="evaluate-inference",
                scope=scope,
                dataset_id=evaluation["dataset_id"],
                regime=evaluation["regime"],
                run_id=evaluation["run_id"] if scope == "individual" else None,
                source_reference=evaluation["artifact"],
                source_root=report.parent,
            )

    bundle_hits = 0
    analysis_bundles: list[dict[str, Any]] = []
    for task in tasks:
        artifact = task["artifact"]
        if artifact is None:
            artifact = _execute_analysis_export_task(
                task,
                store=store,
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                run_root=run_root,
            )
        else:
            bundle_hits += 1
            _validate_analysis_bundle_artifact(artifact)
        if not isinstance(artifact, Artifact):
            raise RuntimeError("Analysis export task completed without an artifact")
        archive, manifest = _validate_analysis_bundle_artifact(artifact)
        record = {
            "producer_command": task["producer_command"],
            "scope": task["scope"],
            "dataset_id": task["dataset_id"],
            "source_artifact": task["source_artifact"],
            "analysis_zip": str(archive),
            "analysis_manifest": str(manifest),
            "artifact": _artifact_ref(artifact, cache_hit=bool(task["cache_hit"])),
        }
        if task["regime"] is not None:
            record["regime"] = task["regime"]
        if task["run_id"] is not None:
            record["run_id"] = task["run_id"]
        analysis_bundles.append(record)

    exports_manifest = _materialize_exports(
        run_root=run_root,
        profile_id=profile_id,
        bundles=analysis_bundles,
        store=store,
    )
    state["analysis_bundles"] = analysis_bundles
    state["exports_manifest"] = str(exports_manifest)
    state["cache"]["bundle_hits"] = bundle_hits
    state["status"] = "exported"
    state["updated_at"] = _utc_now()
    return write_state(run_root, state)
