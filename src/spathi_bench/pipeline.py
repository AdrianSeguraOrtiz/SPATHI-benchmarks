"""Incremental, content-addressed SPATHI benchmark pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
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
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    load_repo_lock,
    load_run_metadata,
    validate_all_profiles,
    validate_design_contracts,
    validate_locked_repository,
)
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
    andrea_command,
    andrea_contract_command,
    andrea_host_environment,
    catalog_image_lease,
    catalog_image_leases,
    check_docker,
    docker_image_details,
    stream_command,
    validate_andrea_import,
)
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _artifact_store(run_root: Path) -> ArtifactStore:
    resolved = run_root.resolve()
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
    return ArtifactStore(version_root / "artifacts")


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
    if Path(str(reference.get("path", ""))).resolve() != artifact.path.resolve():
        raise BenchmarkConfigError(f"Artifact reference path differs from its content key: {key}")
    return artifact


def _artifact_member_from_state(artifact: Artifact, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise BenchmarkConfigError(f"{label} is missing from benchmark state")
    candidate = Path(raw_path).resolve()
    root = artifact.path.resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise BenchmarkConfigError(f"{label} escapes or is missing from its artifact: {raw_path}")
    relative = candidate.relative_to(root).as_posix()
    try:
        member = artifact.file(relative)
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} is not declared by its artifact: {raw_path}") from exc
    return member.resolve()


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
) -> Path:
    existing = _load_existing_state(repo_root, profile_id, run_root)
    if existing is not None:
        if not status_at_least(existing, "preflighted"):
            raise BenchmarkConfigError("Existing run has not completed preflight")
        path = run_root / "preflight" / "benchmark-preflight.json"
        if not path.is_file():
            raise BenchmarkConfigError("Preflight state exists but its report is missing")
        return path

    validate_all_profiles(repo_root)
    validate_design_contracts(repo_root, andrea_repo)
    freeze_run_inputs(repo_root, profile_id, run_root)
    frozen_root = frozen_repository_root(run_root)
    profile = frozen_profile(run_root, profile_id)
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

    preflight_dir = run_root / "preflight"
    if preflight_dir.exists():
        report = load_json_object(
            preflight_dir / "benchmark-preflight.json",
            "recoverable benchmark preflight",
        )
        expected = {
            "schema_version": "2.0",
            "profile_id": profile_id,
            "purpose": profile["purpose"],
            "repositories": repositories,
            "catalog_images": catalog_images,
            "host_environment": host_environment,
            "status": "passed",
        }
        if any(report.get(field) != value for field, value in expected.items()):
            raise BenchmarkConfigError(
                f"Preflight output exists without state and cannot be recovered: {preflight_dir}"
            )
        for job in profile["generation_jobs"]:
            for suffix in ("catalog.json", "simulation-plan.json"):
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
        )
        write_state(run_root, state)
        return preflight_dir / "benchmark-preflight.json"
    temporary = Path(tempfile.mkdtemp(prefix=".preflight-", dir=run_root))
    try:
        resources = profile["resources"]["generate_data"]
        for job in profile["generation_jobs"]:
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
            "schema_version": "2.0",
            "created_at": _utc_now(),
            "profile_id": profile_id,
            "purpose": profile["purpose"],
            "repositories": repositories,
            "catalog_images": catalog_images,
            "host_environment": host_environment,
            "status": "passed",
        }
        _write_json(temporary / "benchmark-preflight.json", report)
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
            "schema_version": "2.0",
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
    _write_json(provenance_path, details)
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
        "simulator_runs_sha256": sha256_file(simulator_runs),
        "simulator_images": simulator_images,
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
        scenario = frozen_path(run_root, job["scenario"], "scenario")
        simulator_runs = frozen_path(run_root, job["simulator_runs"], "simulator runs")
        spec = _generation_spec(
            state=state,
            job=job,
            scenario=scenario,
            simulator_runs=simulator_runs,
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
        "dataset_artifact": dataset["dataset_artifact"]["key"],
        "dataset_id": dataset["dataset_id"],
        "andrea_commit": state["repositories"]["observed"]["andrea"]["commit"],
        "andrea_environment": state["host_environment"]["fingerprint"],
        "batch_kind": batch["kind"],
        "regime": batch["regime"],
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
    return canonical_json_sha256(configuration), canonical_json_sha256(params)


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
        _write_json(tools_params, {"runs": [run]})
        custom_tools: Path | None = None
        if task["custom_definition"] is not None:
            custom_tools = request_root / "custom-tools.json"
            _write_json(custom_tools, {"tools": [task["custom_definition"]]})
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

        def execute_and_validate() -> None:
            _ensure_runtime_image(
                state=state,
                frozen_root=frozen_root,
                batch_kind=batch_kind,
                run=run,
            )
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
    if status_at_least(state, "inferred"):
        store = _artifact_store(run_root)
        for inference in state["inference_runs"]:
            _artifact_from_ref(store, inference["artifact"])
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
    inference_runs: list[dict[str, Any]] = []
    inference_hits = 0
    prepared: list[dict[str, Any]] = []
    ensured_spathi_image = False

    for dataset in state["datasets"]:
        dataset_manifest, _ground_truth_manifest = _validated_dataset_paths(store, dataset)
        for batch in profile["inference_batches"]:
            if dataset["dataset_role"] != batch["dataset_role"]:
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

    missing = [(index, task) for index, task in enumerate(prepared) if task["artifact"] is None]
    if missing:
        workers = min(int(resources["max_parallel_runs"]), len(missing))
        if workers == 1:
            for index, task in missing:
                prepared[index]["artifact"] = _execute_inference_task(
                    task,
                    store=store,
                    repo_root=repo_root,
                    andrea_repo=andrea_repo,
                    run_root=run_root,
                    resources=per_run_resources,
                    state=state,
                    frozen_root=frozen_root,
                )
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spathi-infer") as pool:
                futures = {
                    pool.submit(
                        _execute_inference_task,
                        task,
                        store=store,
                        repo_root=repo_root,
                        andrea_repo=andrea_repo,
                        run_root=run_root,
                        resources=per_run_resources,
                        state=state,
                        frozen_root=frozen_root,
                    ): index
                    for index, task in missing
                }
                for future in as_completed(futures):
                    prepared[futures[future]]["artifact"] = future.result()

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
        artifact = _artifact_from_ref(store, inference["artifact"])
        dataset = datasets[inference["dataset_id"]]
        _dataset_manifest, ground_truth_manifest = _validated_dataset_paths(store, dataset)
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
        report_fingerprint = (report.get("dataset") or {}).get("fingerprint")
        if truth.get("dataset_id") != dataset["dataset_id"]:
            local_errors.append("ground-truth dataset ID differs from benchmark dataset ID")
        if report_fingerprint != truth.get("dataset_fingerprint"):
            local_errors.append("inference and ground-truth dataset fingerprints differ")
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
        results.append(
            {
                "dataset_id": dataset["dataset_id"],
                "run_id": inference["run_id"],
                "artifact_key": inference["artifact"]["key"],
                "merged_network_raw": str(merged_path) if merged_path else None,
                "status": "passed" if not local_errors else "failed",
                "errors": local_errors,
            }
        )
        errors.extend(
            f"[{dataset['dataset_id']}/{inference['run_id']}] {error}" for error in local_errors
        )

    _write_json(
        report_path,
        {
            "schema_version": "2.0",
            "created_at": _utc_now(),
            "profile_id": profile_id,
            "status": "passed" if not errors else "failed",
            "results": results,
            "errors": errors,
        },
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
        raise BenchmarkConfigError("Integrated inference run report has the wrong identity or status")
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
        raise BenchmarkConfigError("Integrated inference run report has an invalid integration contract")
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

    missing = [(index, task) for index, task in enumerate(prepared) if task["artifact"] is None]
    if missing:
        workers = min(max_parallel_runs, len(missing))
        if workers == 1:
            for index, task in missing:
                prepared[index]["artifact"] = _execute_evaluation_task(
                    task,
                    store=store,
                    repo_root=repo_root,
                    andrea_repo=andrea_repo,
                    run_root=run_root,
                )
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spathi-eval") as pool:
                futures = {
                    pool.submit(
                        _execute_evaluation_task,
                        task,
                        store=store,
                        repo_root=repo_root,
                        andrea_repo=andrea_repo,
                        run_root=run_root,
                    ): index
                    for index, task in missing
                }
                for future in as_completed(futures):
                    prepared[futures[future]]["artifact"] = future.result()

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
        raise BenchmarkConfigError(f"Analysis bundle manifest has the wrong bundle ID: {artifact.key}")
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
                raise BenchmarkConfigError(f"Export tree contains a symbolic link: {directory_path / name}")
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
                raise BenchmarkConfigError(f"Analysis bundle export path collision: {directory_key}")
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
        _write_json(staging / "manifest.json", export_manifest)
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
