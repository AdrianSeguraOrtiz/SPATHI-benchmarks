"""Concrete benchmark stage implementations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_json_object,
    load_profile,
    load_repo_lock,
    resolve_repo_path,
    validate_all_profiles,
    validate_design_contracts,
    validate_locked_repository,
    validate_profile_references,
)
from .manifests import datasets_from_benchmark, resolve_manifest_member, unique_direct_manifest
from .state import load_state, state_path, write_state


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ensure_absent(path: Path, label: str) -> None:
    if path.exists():
        raise BenchmarkConfigError(f"{label} already exists; refusing to overwrite: {path}")


def _andrea_environment(andrea_repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(andrea_repo.resolve()) + (os.pathsep + existing if existing else "")
    return env


def _stream_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(command)
    print(f"+ {rendered}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}; see {log_path}")


def _andrea_command(
    arguments: list[str],
    *,
    repo_root: Path,
    andrea_repo: Path,
    log_path: Path,
) -> None:
    executable = os.environ.get("ANDREA_BIN", "andrea")
    if shutil.which(executable) is None:
        raise BenchmarkConfigError(
            f"ANDREA executable {executable!r} is unavailable; install ANDREA's CLI first"
        )
    _stream_command(
        [executable, *arguments],
        cwd=repo_root,
        log_path=log_path,
        env=_andrea_environment(andrea_repo),
    )


def _check_docker() -> None:
    if shutil.which("docker") is None:
        raise BenchmarkConfigError("Docker CLI is unavailable")
    completed = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BenchmarkConfigError(
            "Docker daemon is unavailable: " + (completed.stderr or completed.stdout).strip()
        )


def _image_details(tag: str) -> dict[str, Any] | None:
    completed = subprocess.run(
        ["docker", "image", "inspect", tag],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise BenchmarkConfigError(f"Unexpected docker image inspect response for {tag}")
    image = payload[0]
    return {
        "tag": tag,
        "id": image.get("Id"),
        "repo_digests": image.get("RepoDigests", []),
        "created": image.get("Created"),
        "labels": (image.get("Config") or {}).get("Labels") or {},
    }


def _validate_image_contract(repo_root: Path, image_lock: dict[str, Any]) -> None:
    spathi_image = image_lock["images"]["spathi"]
    repo_lock = load_repo_lock(repo_root)
    if spathi_image["source_repository"] != "spathi":
        raise BenchmarkConfigError("SPATHI image source_repository must be 'spathi'")
    if spathi_image["source_commit"] != repo_lock["repositories"]["spathi"]["commit"]:
        raise BenchmarkConfigError("SPATHI image source commit differs from repos.lock.json")
    for relative_path, expected_digest in spathi_image["build_inputs"].items():
        build_input = resolve_repo_path(repo_root, relative_path, "SPATHI image build input")
        observed_digest = hashlib.sha256(build_input.read_bytes()).hexdigest()
        if observed_digest != expected_digest:
            raise BenchmarkConfigError(
                f"SPATHI image build input differs from images.lock.json: {relative_path}"
            )
    constraints: dict[str, str] = {}
    for line in (
        (repo_root / "containers" / "spathi" / "constraints.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if line and not line.startswith("#"):
            name, separator, version = line.partition("==")
            if not separator:
                raise BenchmarkConfigError(f"Unpinned SPATHI dependency constraint: {line}")
            constraints[name] = version
    if constraints != spathi_image["python_dependencies"]:
        raise BenchmarkConfigError(
            "containers/spathi/constraints.txt differs from images.lock.json"
        )


def _validate_andrea_import(andrea_repo: Path) -> str:
    command = [
        sys.executable,
        "-c",
        "import pathlib,andrea; print(pathlib.Path(andrea.__file__).resolve())",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_andrea_environment(andrea_repo),
    )
    if completed.returncode != 0:
        raise BenchmarkConfigError(
            "Cannot import locked ANDREA source: " + (completed.stderr or completed.stdout).strip()
        )
    imported = Path(completed.stdout.strip()).resolve()
    if not imported.is_relative_to(andrea_repo.resolve()):
        raise BenchmarkConfigError(f"ANDREA import resolved outside locked repository: {imported}")
    return str(imported)


def _validate_andrea_repository(repo_root: Path, andrea_repo: Path) -> dict[str, Any]:
    lock = load_repo_lock(repo_root)["repositories"]["andrea"]
    observed = validate_locked_repository(
        name="andrea",
        path=andrea_repo,
        lock_entry=lock,
        require_clean=True,
    )
    observed["import_path"] = _validate_andrea_import(andrea_repo)
    return observed


def _validate_state_andrea(state: dict[str, Any], observed: dict[str, Any]) -> None:
    repositories = state.get("repositories")
    recorded = (
        repositories.get("observed", {}).get("andrea") if isinstance(repositories, dict) else None
    )
    if not isinstance(recorded, dict):
        raise BenchmarkConfigError("Benchmark state has no observed ANDREA provenance")
    for field in ("commit", "origin"):
        if recorded.get(field) != observed.get(field):
            raise BenchmarkConfigError(
                f"ANDREA {field} changed since generation: "
                f"recorded={recorded.get(field)!r}, observed={observed.get(field)!r}"
            )


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


def preflight(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
) -> Path:
    profile = load_profile(repo_root, profile_id, require_runnable=True)
    validate_all_profiles(repo_root)
    validate_design_contracts(repo_root, andrea_repo)
    validate_profile_references(repo_root, profile)
    repo_lock = load_repo_lock(repo_root)
    image_lock = load_image_lock(repo_root)
    _validate_image_contract(repo_root, image_lock)
    repositories = {
        "andrea": validate_locked_repository(
            name="andrea",
            path=andrea_repo,
            lock_entry=repo_lock["repositories"]["andrea"],
            require_clean=True,
        ),
        "spathi": validate_locked_repository(
            name="spathi",
            path=spathi_repo,
            lock_entry=repo_lock["repositories"]["spathi"],
            require_clean=True,
        ),
    }
    _check_docker()
    imported_andrea = _validate_andrea_import(andrea_repo)

    preflight_dir = run_root / "preflight"
    _ensure_absent(preflight_dir, "preflight stage")
    preflight_dir.mkdir(parents=True)
    generation_resources = profile["resources"]["generate_data"]
    for job in profile["generation_jobs"]:
        scenario = resolve_repo_path(repo_root, job["scenario"], "scenario")
        simulator_runs = resolve_repo_path(repo_root, job["simulator_runs"], "simulator runs")
        _andrea_command(
            [
                "generate-data",
                "preflight",
                "--scenario",
                str(scenario),
                "--output-json",
                str(preflight_dir / f"{job['id']}-catalog.json"),
            ],
            repo_root=repo_root,
            andrea_repo=andrea_repo,
            log_path=run_root / "logs" / f"00-{job['id']}-catalog.log",
        )
        _andrea_command(
            [
                "generate-data",
                "plan",
                "--scenario",
                str(scenario),
                "--simulator-runs",
                str(simulator_runs),
                "--out",
                str(preflight_dir / f"{job['id']}-simulation-plan.json"),
                "--max-parallel-tasks",
                str(generation_resources["max_parallel_tasks"]),
                "--max-cores",
                str(generation_resources["max_cores"]),
                "--max-ram-gb",
                str(generation_resources["max_ram_gb"]),
            ],
            repo_root=repo_root,
            andrea_repo=andrea_repo,
            log_path=run_root / "logs" / f"00-{job['id']}-plan.log",
        )

    tag = image_lock["images"]["spathi"]["tag"]
    report = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "profile_id": profile_id,
        "repositories": repositories,
        "andrea_import": imported_andrea,
        "spathi_image": _image_details(tag),
        "spathi_image_required_tag": tag,
        "profiles_validated": [item["id"] for item in validate_all_profiles(repo_root)],
        "status": "passed",
    }
    path = preflight_dir / "benchmark-preflight.json"
    _atomic_json(path, report)
    return path


def build_spathi_image(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    spathi_repo: Path,
) -> Path:
    load_profile(repo_root, profile_id, require_runnable=True)
    _check_docker()
    repo_lock = load_repo_lock(repo_root)
    image_lock = load_image_lock(repo_root)
    _validate_image_contract(repo_root, image_lock)
    validate_locked_repository(
        name="spathi",
        path=spathi_repo,
        lock_entry=repo_lock["repositories"]["spathi"],
        require_clean=True,
    )
    provenance_path = run_root / "provenance" / "spathi-image.json"
    _ensure_absent(provenance_path, "SPATHI image provenance")

    image = image_lock["images"]["spathi"]
    temporary_context = tempfile.TemporaryDirectory(prefix="spathi-build-context-")
    try:
        snapshot_root = Path(temporary_context.name)
        _copy_locked_spathi_source(spathi_repo, snapshot_root)
        build_context = str(snapshot_root)
        source_context = f"git-tree:{spathi_repo.resolve()}@{image['source_commit']}"
        command = [
            "docker",
            "buildx",
            "build",
            "--load",
            "--build-context",
            f"spathi={build_context}",
            "--build-arg",
            f"BASE_IMAGE={image['base_image']}",
            "--build-arg",
            f"SPATHI_COMMIT={image['source_commit']}",
            "--file",
            str(repo_root / "containers" / "spathi" / "Dockerfile"),
            "--tag",
            image["tag"],
            str(repo_root),
        ]
        _stream_command(
            command,
            cwd=repo_root,
            log_path=run_root / "logs" / "05-build-spathi-image.log",
        )
    finally:
        temporary_context.cleanup()
    details = _image_details(image["tag"])
    if details is None:
        raise RuntimeError(f"Built image is unavailable after docker build: {image['tag']}")
    details.update(
        {
            "schema_version": "1.0",
            "built_at": _utc_now(),
            "source_context": source_context,
            "source_commit": image["source_commit"],
            "base_image": image["base_image"],
            "build_inputs": image["build_inputs"],
        }
    )
    _atomic_json(provenance_path, details)
    return provenance_path


def generate_data(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    andrea_repo: Path,
) -> Path:
    profile = load_profile(repo_root, profile_id, require_runnable=True)
    observed_andrea = _validate_andrea_repository(repo_root, andrea_repo)
    _ensure_absent(state_path(run_root), "benchmark state")
    output_parent = run_root / "generation"
    _ensure_absent(output_parent, "generation stage")
    resources = profile["resources"]["generate_data"]

    benchmarks: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    for job in profile["generation_jobs"]:
        output_root = output_parent / job["id"]
        scenario = resolve_repo_path(repo_root, job["scenario"], "scenario")
        simulator_runs = resolve_repo_path(repo_root, job["simulator_runs"], "simulator runs")
        _andrea_command(
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
        manifest_path = unique_direct_manifest(output_root, "benchmark-manifest.json")
        job_datasets = datasets_from_benchmark(manifest_path, job["id"])
        benchmarks.append(
            {
                "generation_job": job["id"],
                "benchmark_manifest": str(manifest_path),
                "dataset_ids": [item["dataset_id"] for item in job_datasets],
            }
        )
        datasets.extend(job_datasets)

    dataset_ids = [item["dataset_id"] for item in datasets]
    if len(set(dataset_ids)) != len(dataset_ids):
        raise BenchmarkConfigError("Generated benchmark contains duplicate dataset IDs")
    state = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "profile_id": profile_id,
        "profile": str((repo_root / "experiments/v1/profiles" / f"{profile_id}.json").resolve()),
        "repositories": {
            "lock": load_repo_lock(repo_root)["repositories"],
            "observed": {"andrea": observed_andrea},
        },
        "benchmarks": benchmarks,
        "datasets": datasets,
        "inference_runs": [],
        "evaluations": [],
        "status": "generated",
    }
    return write_state(run_root, state)


def run_inference(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    andrea_repo: Path,
) -> Path:
    profile = load_profile(repo_root, profile_id, require_runnable=True)
    state = load_state(run_root, profile_id)
    observed_andrea = _validate_andrea_repository(repo_root, andrea_repo)
    _validate_state_andrea(state, observed_andrea)
    if state.get("status") != "generated" or state.get("inference_runs"):
        raise BenchmarkConfigError("Inference requires a generated state with no prior runs")
    output_parent = run_root / "inference"
    _ensure_absent(output_parent, "inference stage")

    image_lock = load_image_lock(repo_root)
    _validate_image_contract(repo_root, image_lock)
    image_tag = image_lock["images"]["spathi"]["tag"]
    image = _image_details(image_tag)
    if image is None:
        raise BenchmarkConfigError(
            f"Required SPATHI image is missing: {image_tag}; run 05_build_spathi_image.sh"
        )
    provenance_path = run_root / "provenance" / "spathi-image.json"
    if not provenance_path.is_file():
        raise BenchmarkConfigError(
            "Missing verified SPATHI image provenance; run 05_build_spathi_image.sh "
            "with this profile and run ID"
        )
    recorded = load_json_object(provenance_path, "SPATHI image provenance")
    if recorded.get("id") != image.get("id"):
        raise BenchmarkConfigError("Local SPATHI image changed after this run built it")
    expected_commit = image_lock["images"]["spathi"]["source_commit"]
    if recorded.get("source_commit") != expected_commit:
        raise BenchmarkConfigError("SPATHI image provenance has the wrong source commit")
    if recorded.get("build_inputs") != image_lock["images"]["spathi"]["build_inputs"]:
        raise BenchmarkConfigError("SPATHI image provenance has different build inputs")
    if image.get("labels", {}).get("org.opencontainers.image.revision") != expected_commit:
        raise BenchmarkConfigError("SPATHI image revision label differs from the locked commit")

    resources = profile["resources"]["infer_network"]
    inference_runs: list[dict[str, Any]] = []
    for dataset in state["datasets"]:
        for batch in profile["inference_batches"]:
            output_root = output_parent / dataset["dataset_id"] / batch["id"]
            tools_params = resolve_repo_path(repo_root, batch["tools_params"], "tools params")
            custom_tools = (
                resolve_repo_path(repo_root, batch["custom_tools"], "custom tools")
                if batch.get("custom_tools")
                else None
            )
            command = [
                "infer-network",
                "execute",
                "--dataset-manifest",
                dataset["dataset_manifest"],
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
            _andrea_command(
                command,
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                log_path=run_root / "logs" / f"20-{dataset['dataset_id']}-{batch['id']}.log",
            )
            report_path = unique_direct_manifest(output_root, "run_report.json")
            requested = load_json_object(tools_params, "tools params")["runs"]
            inference_runs.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "batch_id": batch["id"],
                    "regime": batch["regime"],
                    "run_report": str(report_path),
                    "expected_runs": [item["run_id"] for item in requested],
                }
            )
    state["inference_runs"] = inference_runs
    state["spathi_image"] = {
        "provenance": str(provenance_path.resolve()),
        "tag": image_tag,
        "id": image["id"],
        "source_commit": expected_commit,
    }
    state["updated_at"] = _utc_now()
    state["status"] = "inferred"
    return write_state(run_root, state)


def validate_handoffs(*, profile_id: str, run_root: Path) -> Path:
    state = load_state(run_root, profile_id)
    if state.get("status") != "inferred" or not state.get("inference_runs"):
        raise BenchmarkConfigError("Handoff validation requires completed inference runs")
    datasets = {item["dataset_id"]: item for item in state["datasets"]}
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for item in state["inference_runs"]:
        dataset = datasets[item["dataset_id"]]
        report_path = Path(item["run_report"])
        report = load_json_object(report_path, "run report")
        truth = load_json_object(Path(dataset["ground_truth_manifest"]), "ground truth")
        selected = (report.get("tools") or {}).get("selected", [])
        status_by_tool = (report.get("tools") or {}).get("status_by_tool", {})
        failed = (report.get("tools") or {}).get("failed", {})
        expected = item["expected_runs"]
        local_errors: list[str] = []
        if report.get("status") != "executed":
            local_errors.append(f"run status is {report.get('status')!r}, expected 'executed'")
        if set(selected) != set(expected):
            local_errors.append(f"selected runs {selected!r} differ from expected {expected!r}")
        invalid_status = {
            run_id: status_by_tool.get(run_id)
            for run_id in expected
            if status_by_tool.get(run_id) not in {"completed", "completed_with_warnings"}
        }
        if invalid_status:
            local_errors.append(f"runs did not complete: {invalid_status}")
        if failed:
            local_errors.append(f"run report contains failed tools: {failed}")
        report_dataset = report.get("dataset") or {}
        if report_dataset.get("id") != dataset["dataset_id"]:
            local_errors.append("run-report dataset ID differs from benchmark dataset ID")
        if truth.get("dataset_id") != dataset["dataset_id"]:
            local_errors.append("ground-truth dataset ID differs from benchmark dataset ID")
        if report_dataset.get("fingerprint") != truth.get("dataset_fingerprint"):
            local_errors.append("inference and ground-truth dataset fingerprints differ")
        outputs = report.get("outputs") or {}
        merged_raw = outputs.get("merged_network_raw")
        try:
            merged_path = resolve_manifest_member(report_path.parent, merged_raw, "merged network")
        except BenchmarkConfigError as exc:
            local_errors.append(str(exc))
            merged_path = None
        origins = (report.get("tools") or {}).get("tool_origins", {})
        if "spathi_primary" in expected and origins.get("spathi_primary") != "custom":
            local_errors.append("SPATHI is not recorded as a custom external tool")
        result = {
            "dataset_id": dataset["dataset_id"],
            "batch_id": item["batch_id"],
            "run_report": str(report_path),
            "merged_network_raw": str(merged_path) if merged_path else None,
            "status": "passed" if not local_errors else "failed",
            "errors": local_errors,
        }
        results.append(result)
        errors.extend(
            f"[{dataset['dataset_id']}/{item['batch_id']}] {error}" for error in local_errors
        )

    report_path = run_root / "handoff-validation.json"
    _atomic_json(
        report_path,
        {
            "schema_version": "1.0",
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


def evaluate(
    *,
    repo_root: Path,
    profile_id: str,
    run_root: Path,
    andrea_repo: Path,
) -> Path:
    load_profile(repo_root, profile_id, require_runnable=True)
    state = load_state(run_root, profile_id)
    observed_andrea = _validate_andrea_repository(repo_root, andrea_repo)
    _validate_state_andrea(state, observed_andrea)
    if state.get("status") != "validated" or state.get("evaluations"):
        raise BenchmarkConfigError("Evaluation requires validated, unevaluated inference handoffs")
    output_parent = run_root / "evaluation"
    _ensure_absent(output_parent, "evaluation stage")
    datasets = {item["dataset_id"]: item for item in state["datasets"]}
    evaluations: list[dict[str, Any]] = []
    for inference in state["inference_runs"]:
        dataset = datasets[inference["dataset_id"]]
        output_root = output_parent / inference["dataset_id"] / inference["batch_id"]
        _andrea_command(
            [
                "evaluate-inference",
                "--run-report",
                inference["run_report"],
                "--ground-truth-manifest",
                dataset["ground_truth_manifest"],
                "--output-dir",
                str(output_root),
                "--no-view",
            ],
            repo_root=repo_root,
            andrea_repo=andrea_repo,
            log_path=run_root
            / "logs"
            / f"40-{inference['dataset_id']}-{inference['batch_id']}.log",
        )
        evaluation_report = unique_direct_manifest(output_root, "evaluation_report.json")
        evaluations.append(
            {
                "dataset_id": inference["dataset_id"],
                "batch_id": inference["batch_id"],
                "regime": inference["regime"],
                "run_report": inference["run_report"],
                "evaluation_report": str(evaluation_report),
            }
        )
    state["evaluations"] = evaluations
    state["status"] = "evaluated"
    state["updated_at"] = _utc_now()
    return write_state(run_root, state)
