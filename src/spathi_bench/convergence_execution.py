"""Strict physical execution and ANDREA-compatible fan-out for convergence studies."""

from __future__ import annotations

import copy
import csv
import hashlib
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .andrea_contracts import validate_inference_analysis_contract
from .artifacts import Artifact, ArtifactKey, ArtifactStore, canonical_json_sha256, sha256_file
from .calibration_contexts import load_dataset_group_contract
from .config import BenchmarkConfigError, load_json_object
from .manifests import resolve_manifest_member
from .resource_scheduler import (
    ResourceCapacity,
    ResourceClaim,
    ResourceScheduleError,
    ScheduledTask,
    execute_resource_schedule,
)
from .runtime import andrea_command, stream_command
from .safe_io import atomic_write_json

_ESTIMATOR_COUNTS = (250, 500, 750, 1000)
_NETWORK_COLUMNS = ("source", "target", "score", "sign", "evidence", "context")
_MERGED_COLUMNS = (*_NETWORK_COLUMNS, "tool_id")
_CONTRACT_ID = "spathi_fixed_forest_prefix_fanout_v1"
_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ConvergenceBinding:
    """One logical prefix and the complete physical plan that owns it."""

    plan: dict[str, Any]
    prefix: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _GroupResult:
    artifacts: dict[ArtifactKey, Artifact]
    physical_artifact: Artifact


def index_convergence_plans(payload: Mapping[str, Any]) -> dict[str, ConvergenceBinding]:
    """Index a validated canonical plan by logical run ID, rejecting ambiguity."""

    plans = payload.get("plans")
    if not isinstance(plans, list) or not plans:
        raise BenchmarkConfigError("Convergence plan collection is empty")
    indexed: dict[str, ConvergenceBinding] = {}
    for raw_plan in plans:
        if not isinstance(raw_plan, dict):
            raise BenchmarkConfigError("Convergence plan entries must be objects")
        logical = raw_plan.get("logical_prefixes")
        if not isinstance(logical, list) or len(logical) != 4:
            raise BenchmarkConfigError("Every convergence plan must expose four prefixes")
        counts = tuple(item.get("n_estimators") for item in logical if isinstance(item, dict))
        if counts != _ESTIMATOR_COUNTS:
            raise BenchmarkConfigError(
                f"Convergence prefix schedule must be {_ESTIMATOR_COUNTS}, not {counts}"
            )
        for raw_prefix in logical:
            if not isinstance(raw_prefix, dict):
                raise BenchmarkConfigError("Convergence logical prefixes must be objects")
            run_id = raw_prefix.get("run_id")
            if not isinstance(run_id, str) or not run_id or run_id in indexed:
                raise BenchmarkConfigError(
                    f"Convergence logical run ID is invalid or duplicated: {run_id!r}"
                )
            indexed[run_id] = ConvergenceBinding(
                plan=copy.deepcopy(raw_plan),
                prefix=copy.deepcopy(raw_prefix),
            )
    declared = payload.get("logical_results_per_dataset")
    if declared != len(indexed):
        raise BenchmarkConfigError(
            "Convergence logical result count differs from its canonical plan collection"
        )
    if payload.get("physical_plans_per_dataset") != len(plans):
        raise BenchmarkConfigError(
            "Convergence physical fit count differs from its canonical plan collection"
        )
    return indexed


def logical_execution_contract(binding: ConvergenceBinding) -> dict[str, Any]:
    """Return the immutable strategy fragment included in a logical artifact key."""

    return {
        "kind": _CONTRACT_ID,
        "physical_run_id": binding.plan["physical_run_id"],
        "physical_plan_sha256": canonical_json_sha256(binding.plan),
        "logical_run_id": binding.prefix["run_id"],
        "n_estimators": binding.prefix["n_estimators"],
    }


def _physical_spec(task: Mapping[str, Any]) -> dict[str, Any]:
    binding = task.get("convergence_binding")
    logical_spec = task.get("spec")
    if not isinstance(binding, ConvergenceBinding) or not isinstance(logical_spec, dict):
        raise BenchmarkConfigError("Convergence task has no immutable execution binding")
    return {
        "schema_version": "1.0",
        "execution_contract": _CONTRACT_ID,
        "dataset_artifact": logical_spec["dataset_artifact"],
        "dataset_id": logical_spec["dataset_id"],
        "andrea_commit": logical_spec["andrea_commit"],
        "andrea_environment": logical_spec["andrea_environment"],
        "benchmark_environment": logical_spec["benchmark_environment"],
        "docker_runtime_environment": logical_spec["docker_runtime_environment"],
        "tool_image": logical_spec["tool_image"],
        "resources": logical_spec["resources"],
        "runtime_resources": task["run"]["resources"],
        "execution_plan_fingerprint": logical_spec["execution_plan_fingerprint"],
        "output_profile": logical_spec["output_profile"],
        "plan": binding.plan,
    }


def _group_tasks(tasks: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        binding = task.get("convergence_binding")
        if not isinstance(binding, ConvergenceBinding):
            raise BenchmarkConfigError("Non-convergence task entered convergence execution")
        identity = (
            str(task["dataset"]["dataset_id"]),
            str(task["batch"]["id"]),
            str(binding.plan["physical_run_id"]),
        )
        grouped[identity].append(task)

    ordered: list[list[dict[str, Any]]] = []
    for identity, entries in sorted(grouped.items()):
        entries.sort(key=lambda item: item["convergence_binding"].prefix["n_estimators"])
        binding = entries[0]["convergence_binding"]
        planned_ids = [item["run_id"] for item in binding.plan["logical_prefixes"]]
        observed_ids = [str(item["run"]["run_id"]) for item in entries]
        counts = tuple(int(item["convergence_binding"].prefix["n_estimators"]) for item in entries)
        if observed_ids != planned_ids or counts != _ESTIMATOR_COUNTS:
            raise BenchmarkConfigError(
                f"Convergence group {identity!r} is not the complete canonical fan-out"
            )
        physical_specs = [_physical_spec(item) for item in entries]
        if any(spec != physical_specs[0] for spec in physical_specs[1:]):
            raise BenchmarkConfigError(
                f"Convergence group {identity!r} has inconsistent physical specifications"
            )
        resource_contracts = [item["run"]["resources"] for item in entries]
        if any(value != resource_contracts[0] for value in resource_contracts[1:]):
            raise BenchmarkConfigError(
                f"Convergence group {identity!r} has inconsistent runtime resources"
            )
        ordered.append(entries)
    return ordered


def _dataset_inputs(dataset_manifest: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    manifest = load_json_object(dataset_manifest, "convergence dataset manifest")
    dataset = manifest.get("dataset")
    extras = manifest.get("extras")
    if not isinstance(dataset, dict) or not isinstance(extras, dict):
        raise BenchmarkConfigError("Convergence dataset manifest lacks dataset or extras")
    expression_ref = dataset.get("expression_matrix")
    expression = resolve_manifest_member(
        dataset_manifest.parent,
        expression_ref,
        "convergence expression matrix",
    )
    groups = resolve_manifest_member(
        dataset_manifest.parent,
        extras.get("groups"),
        "convergence groups",
    )
    tf_list = resolve_manifest_member(
        dataset_manifest.parent,
        extras.get("tf_list"),
        "convergence TF list",
    )
    spec = dataset.get("spec")
    if not isinstance(spec, dict):
        raise BenchmarkConfigError("Convergence dataset manifest lacks dataset.spec")
    return expression, groups, tf_list, spec


def _validate_preflight(report_path: Path, *, run_id: str, dataset_id: str) -> dict[str, Any]:
    report = load_json_object(report_path, "convergence logical preflight")
    runs = report.get("runs")
    inputs = report.get("inputs")
    dataset = report.get("dataset")
    if (
        not isinstance(runs, dict)
        or not isinstance(inputs, dict)
        or not isinstance(dataset, dict)
        or inputs.get("dataset_id") != dataset_id
        or dataset.get("dataset_id") != dataset_id
        or runs.get("requested_total") != 1
        or runs.get("selected") != [run_id]
        or runs.get("skipped") != {}
        or runs.get("tool_origins") != {run_id: "custom"}
    ):
        raise BenchmarkConfigError(
            f"Convergence preflight did not accept exactly {run_id!r} on {dataset_id!r}"
        )
    issues = runs.get("issues")
    global_issues = report.get("issues")
    if not isinstance(issues, dict) or not isinstance(global_issues, list):
        raise BenchmarkConfigError("Convergence preflight has no canonical issue inventory")
    blocking = [
        issue
        for issue in [*issues.get(run_id, []), *global_issues]
        if isinstance(issue, dict) and issue.get("severity") == "block"
    ]
    if blocking:
        raise BenchmarkConfigError(f"Convergence preflight blocked {run_id!r}: {blocking!r}")
    execution = runs.get("resolved_execution")
    if not isinstance(execution, dict) or execution.get(run_id) != {"mode": "group_native"}:
        raise BenchmarkConfigError("Convergence preflight did not resolve group_native execution")
    return report


def _write_preflight(
    task: Mapping[str, Any],
    *,
    root: Path,
    repo_root: Path,
    andrea_repo: Path,
    log_path: Path,
) -> dict[str, Any]:
    run = task["run"]
    run_id = str(run["run_id"])
    request = root / run_id
    request.mkdir(parents=True)
    tools_params = request / "tools-params.json"
    custom_tools = request / "custom-tools.json"
    report_path = request / "preflight.json"
    atomic_write_json(tools_params, {"runs": [run]}, "convergence tools request", boundary=root)
    atomic_write_json(
        custom_tools,
        {"tools": [task["custom_definition"]]},
        "convergence custom-tool request",
        boundary=root,
    )
    andrea_command(
        [
            "infer-network",
            "preflight",
            "--dataset-manifest",
            str(task["dataset_manifest"]),
            "--tools-params",
            str(tools_params),
            "--custom-tools",
            str(custom_tools),
            "--output-json",
            str(report_path),
        ],
        repo_root=repo_root,
        andrea_repo=andrea_repo,
        log_path=log_path,
    )
    return _validate_preflight(
        report_path,
        run_id=run_id,
        dataset_id=str(task["dataset"]["dataset_id"]),
    )


def _validate_fanout(
    output: Path, *, plan: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = output / "fanout-manifest.json"
    manifest = load_json_object(manifest_path, "SPATHI convergence fan-out manifest")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("artifact_type") != "spathi_convergence_fanout"
        or manifest.get("status") != "complete"
        or manifest.get("physical_run_id") != plan.get("physical_run_id")
        or manifest.get("physical_fits") != 1
        or manifest.get("logical_results") != 4
        or manifest.get("estimator_counts") != list(_ESTIMATOR_COUNTS)
    ):
        raise BenchmarkConfigError("SPATHI convergence runner returned the wrong fan-out identity")
    outputs = manifest.get("logical_outputs")
    planned = plan.get("logical_prefixes")
    if not isinstance(outputs, list) or not isinstance(planned, list) or len(outputs) != 4:
        raise BenchmarkConfigError("SPATHI convergence runner returned an incomplete fan-out")
    by_run: dict[str, dict[str, Any]] = {}
    for expected, observed in zip(planned, outputs, strict=True):
        if not isinstance(expected, dict) or not isinstance(observed, dict):
            raise BenchmarkConfigError("SPATHI convergence fan-out entries must be objects")
        for field in ("run_id", "configuration_id", "configuration_sha256", "n_estimators"):
            if observed.get(field) != expected.get(field):
                raise BenchmarkConfigError(
                    f"SPATHI convergence fan-out changed logical prefix field {field!r}"
                )
        relative = observed.get("relative_path")
        if relative != f"prefixes/{expected['n_estimators']}":
            raise BenchmarkConfigError("SPATHI convergence fan-out path is not canonical")
        directory = output / relative
        raw_files = observed.get("files")
        if not isinstance(raw_files, list):
            raise BenchmarkConfigError("SPATHI convergence fan-out has no file inventory")
        listed: set[str] = set()
        for record in raw_files:
            if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
                raise BenchmarkConfigError("SPATHI convergence file record is malformed")
            name = record["path"]
            path = directory / name
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or name in listed
                or not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != record["size"]
                or sha256_file(path) != record["sha256"]
            ):
                raise BenchmarkConfigError("SPATHI convergence file inventory differs from disk")
            listed.add(name)
        disk = {
            path.name for path in directory.iterdir() if path.is_file() and not path.is_symlink()
        }
        if disk != listed or "network.csv" not in listed or "run_metadata.json" not in listed:
            raise BenchmarkConfigError("SPATHI convergence prefix files are incomplete")
        metadata = load_json_object(directory / "run_metadata.json", "convergence prefix metadata")
        if any(
            (
                metadata.get("status") != "complete",
                metadata.get("physical_run_id") != plan["physical_run_id"],
                metadata.get("run_id") != expected["run_id"],
                metadata.get("configuration_id") != expected["configuration_id"],
                metadata.get("configuration_sha256") != expected["configuration_sha256"],
                metadata.get("n_estimators") != expected["n_estimators"],
                metadata.get("physical_forest_ceiling") != _ESTIMATOR_COUNTS[-1],
                metadata.get("prefix_schedule") != list(_ESTIMATOR_COUNTS),
            )
        ):
            raise BenchmarkConfigError("SPATHI convergence prefix metadata is inconsistent")
        by_run[str(expected["run_id"])] = observed
    return manifest, by_run


def _write_merged_networks(
    source: Path,
    *,
    raw_path: Path,
    normalized_path: Path,
    run_id: str,
    allowed_contexts: set[str],
) -> int:
    rows: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _NETWORK_COLUMNS:
            raise BenchmarkConfigError("Convergence network.csv has a non-canonical schema")
        for line_number, row in enumerate(reader, start=2):
            values = {field: str(row.get(field, "")).strip() for field in _NETWORK_COLUMNS}
            try:
                score = float(values["score"])
            except ValueError as exc:
                raise BenchmarkConfigError(
                    f"Convergence network score is invalid at line {line_number}"
                ) from exc
            if (
                not math.isfinite(score)
                or score <= 0.0
                or not values["source"]
                or not values["target"]
                or not values["evidence"]
                or values["sign"] != "?"
                or values["context"] not in allowed_contexts
            ):
                raise BenchmarkConfigError(
                    f"Convergence network row {line_number} violates the SPATHI contract"
                )
            rows.append(values)
    scores = [float(row["score"]) for row in rows]
    minimum = min(scores) if scores else 0.0
    maximum = max(scores) if scores else 0.0
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        raw_path.open("x", encoding="utf-8", newline="") as raw_handle,
        normalized_path.open("x", encoding="utf-8", newline="") as normalized_handle,
    ):
        raw_writer = csv.DictWriter(raw_handle, fieldnames=_MERGED_COLUMNS, lineterminator="\n")
        normalized_writer = csv.DictWriter(
            normalized_handle, fieldnames=_MERGED_COLUMNS, lineterminator="\n"
        )
        raw_writer.writeheader()
        normalized_writer.writeheader()
        for row in rows:
            raw_writer.writerow({**row, "tool_id": run_id})
            score = float(row["score"])
            normalized = (score - minimum) / (maximum - minimum) if maximum > minimum else 1.0
            normalized_writer.writerow(
                {**row, "score": format(normalized, ".12g"), "tool_id": run_id}
            )
    return len(rows)


class _DigestTextSink:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, value: str) -> int:
        payload = value.encode("utf-8")
        self.digest.update(payload)
        return len(value)


def _source_network_digest_from_merged(path: Path, *, run_id: str) -> str:
    sink = _DigestTextSink()
    writer = csv.writer(sink, lineterminator="\n")
    writer.writerow(_NETWORK_COLUMNS)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _MERGED_COLUMNS:
            raise BenchmarkConfigError("Convergence merged network has a non-canonical schema")
        for row in reader:
            if row.get("tool_id") != run_id:
                raise BenchmarkConfigError("Convergence merged network has the wrong logical ID")
            writer.writerow(tuple(row[field] for field in _NETWORK_COLUMNS))
    return sink.digest.hexdigest()


def _logical_report(
    task: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    contexts: Sequence[str],
    rows: int,
    physical_artifact: Artifact,
    prefix_record: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    run_id = str(task["run"]["run_id"])
    preflight_runs = preflight["runs"]
    dataset = preflight["dataset"]
    catalog_id = preflight_runs["catalog_tool_ids"][run_id]
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "executed",
        "output_profile": task["spec"]["output_profile"],
        "dataset": {
            "id": task["dataset"]["dataset_id"],
            "fingerprint": dict(fingerprint),
            "column_kind": dataset["column_kind"],
            "expression_profile": dataset["expression_profile"],
            "genes": dataset["genes"],
            "columns": dataset["columns"],
        },
        "tools": {
            "selected": [run_id],
            "catalog_tool_ids": {run_id: catalog_id},
            "tool_origins": {run_id: "custom"},
            "output_capabilities": {
                run_id: {
                    "tool_origin": "custom",
                    "catalog_tool_id": catalog_id,
                    "directed": True,
                    "sign": "none",
                }
            },
            "skipped": {},
            "status_by_tool": {run_id: "completed"},
            "completed": [run_id],
            "completed_contexts": {run_id: list(contexts)},
            "failed": {},
            "results": {
                run_id: {
                    "tool_id": run_id,
                    "status": "completed",
                    "exit_code": 0,
                    "execution": {"mode": "group_native"},
                    "convergence_fanout": {
                        "contract": _CONTRACT_ID,
                        "physical_artifact": physical_artifact.key.value,
                        "physical_run_id": task["convergence_binding"].plan["physical_run_id"],
                        "n_estimators": task["convergence_binding"].prefix["n_estimators"],
                        "source_network_sha256": next(
                            item["sha256"]
                            for item in prefix_record["files"]
                            if item["path"] == "network.csv"
                        ),
                    },
                }
            },
        },
        "outputs": {
            "merged_network_raw": "merged_network_raw.csv",
            "merged_network_normalized": "merged_network_normalized.csv",
            "rows_per_tool": {run_id: rows},
        },
        "issues": [],
        "execution": {
            "contract": _CONTRACT_ID,
            "elapsed_seconds": elapsed_seconds,
            "physical_fits": 1,
            "logical_results": 4,
            "physical_artifact": physical_artifact.key.value,
        },
    }


def _publish_physical_provenance(
    *,
    store: ArtifactStore,
    spec: Mapping[str, Any],
    runtime_root: Path,
) -> Artifact:
    key = store.key_for("convergence-execution", spec)
    existing = store.lookup(key)
    if existing is not None:
        return existing
    with store.staging_directory(key) as staging:
        shutil.copyfile(runtime_root / "request/convergence-plan.json", staging / "plan.json")
        shutil.copyfile(
            runtime_root / "output/fanout-manifest.json", staging / "fanout-manifest.json"
        )
        shutil.copyfile(runtime_root / "physical.log", staging / "physical.log")
        return store.publish_from_temp(staging, kind="convergence-execution", spec=spec)


def _execute_group(
    tasks: Sequence[dict[str, Any]],
    *,
    store: ArtifactStore,
    repo_root: Path,
    andrea_repo: Path,
    run_root: Path,
    image_id: str,
) -> _GroupResult:
    first = tasks[0]
    binding: ConvergenceBinding = first["convergence_binding"]
    physical_spec = _physical_spec(first)
    dataset_manifest = Path(first["dataset_manifest"])
    ground_truth_manifest = Path(first["ground_truth_manifest"])
    expression, groups, tf_list, _dataset_spec = _dataset_inputs(dataset_manifest)
    truth = load_json_object(ground_truth_manifest, "convergence ground truth")
    fingerprint = truth.get("dataset_fingerprint")
    if not isinstance(fingerprint, dict):
        raise BenchmarkConfigError("Convergence ground truth has no dataset fingerprint")
    context_contract = load_dataset_group_contract(
        dataset_manifest,
        ground_truth_manifest,
        expected_dataset_id=str(first["dataset"]["dataset_id"]),
    )
    contexts = context_contract.expected_group_contexts
    runtime_parent = run_root / ".convergence-runtime"
    runtime_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{binding.plan['physical_run_id']}.", dir=runtime_parent
    ) as name:
        runtime_root = Path(name)
        request = runtime_root / "request"
        extra = runtime_root / "extra"
        output = runtime_root / "output"
        preflight_root = runtime_root / "preflight"
        for directory in (request, extra, output, preflight_root):
            directory.mkdir()
        atomic_write_json(
            request / "convergence-plan.json",
            binding.plan,
            "convergence physical plan",
            boundary=runtime_root,
        )
        atomic_write_json(
            request / "execution.json",
            {"mode": "group_native"},
            "convergence physical execution",
            boundary=runtime_root,
        )
        shutil.copyfile(groups, extra / "groups.tsv")
        shutil.copyfile(tf_list, extra / "tf_list.txt")

        preflights = {
            str(task["run"]["run_id"]): _write_preflight(
                task,
                root=preflight_root,
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                log_path=run_root
                / "logs"
                / (f"19-convergence-{task['dataset']['dataset_id']}-{task['run']['run_id']}.log"),
            )
            for task in tasks
        }

        threads = int(first["run"]["resources"]["threads"])
        memory_bytes = round(float(first["run"]["resources"]["ram_gb"]) * _GIB)
        cpuset_cpus = list(first["run"]["resources"]["cpuset_cpus"])
        uid = os.getuid() if hasattr(os, "getuid") else 0
        gid = os.getgid() if hasattr(os, "getgid") else 0
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cpus",
            str(threads),
            "--cpuset-cpus",
            ",".join(str(cpu) for cpu in cpuset_cpus),
            "--memory",
            str(memory_bytes),
            "--memory-swap",
            str(memory_bytes),
            "--pids-limit",
            "4096",
            "--user",
            f"{uid}:{gid}",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=1073741824",
            "--volume",
            f"{runtime_root.resolve()}:/io:rw",
            "--volume",
            f"{expression.resolve()}:/io/input.tsv:ro",
            "--volume",
            f"{request.resolve()}:/io/request:ro",
            "--volume",
            f"{extra.resolve()}:/io/extra:ro",
            "--entrypoint",
            "python",
            image_id,
            "/usr/local/bin/run_spathi_convergence.py",
            "--input",
            "/io/input.tsv",
            "--params",
            "/io/request/convergence-plan.json",
            "--extra",
            "/io/extra",
            "--output-dir",
            "/io/output",
            "--threads",
            str(threads),
        ]
        started = perf_counter()
        stream_command(
            command,
            cwd=repo_root,
            log_path=runtime_root / "physical.log",
        )
        elapsed = perf_counter() - started
        manifest, prefix_by_run = _validate_fanout(output, plan=binding.plan)
        physical_artifact = _publish_physical_provenance(
            store=store,
            spec=physical_spec,
            runtime_root=runtime_root,
        )

        artifacts: dict[ArtifactKey, Artifact] = {}
        for task in tasks:
            existing = task.get("artifact")
            if isinstance(existing, Artifact):
                artifacts[task["key"]] = existing
                continue
            run_id = str(task["run"]["run_id"])
            prefix_record = prefix_by_run[run_id]
            source = output / prefix_record["relative_path"] / "network.csv"
            with store.staging_directory(task["key"]) as staging:
                logical_request = staging / "request"
                logical_request.mkdir()
                preflight_dir = preflight_root / run_id
                for filename in ("tools-params.json", "custom-tools.json", "preflight.json"):
                    shutil.copyfile(preflight_dir / filename, logical_request / filename)
                logical_output = staging / "output" / run_id
                logical_output.mkdir(parents=True)
                rows = _write_merged_networks(
                    source,
                    raw_path=logical_output / "merged_network_raw.csv",
                    normalized_path=logical_output / "merged_network_normalized.csv",
                    run_id=run_id,
                    allowed_contexts=set(contexts),
                )
                provenance = {
                    "schema_version": "1.0",
                    "artifact_type": "spathi_convergence_logical_result",
                    "execution_contract": _CONTRACT_ID,
                    "physical_artifact": physical_artifact.key.value,
                    "physical_run_id": binding.plan["physical_run_id"],
                    "physical_plan_sha256": canonical_json_sha256(binding.plan),
                    "run_id": run_id,
                    "configuration_id": task["convergence_binding"].prefix["configuration_id"],
                    "configuration_sha256": task["convergence_binding"].prefix[
                        "configuration_sha256"
                    ],
                    "n_estimators": task["convergence_binding"].prefix["n_estimators"],
                    "source_network_sha256": next(
                        item["sha256"]
                        for item in prefix_record["files"]
                        if item["path"] == "network.csv"
                    ),
                    "physical_fits": manifest["physical_fits"],
                    "logical_results": manifest["logical_results"],
                }
                atomic_write_json(
                    logical_output / "convergence-provenance.json",
                    provenance,
                    "convergence logical provenance",
                    boundary=staging,
                )
                report = _logical_report(
                    task,
                    preflight=preflights[run_id],
                    fingerprint=fingerprint,
                    contexts=contexts,
                    rows=rows,
                    physical_artifact=physical_artifact,
                    prefix_record=prefix_record,
                    elapsed_seconds=elapsed,
                )
                atomic_write_json(
                    logical_output / "run_report.json",
                    report,
                    "convergence logical run report",
                    boundary=staging,
                )
                validate_inference_analysis_contract(
                    andrea_repo,
                    logical_output / "run_report.json",
                )
                artifact = store.publish_from_temp(
                    staging,
                    kind="inference",
                    spec=task["spec"],
                )
            artifacts[task["key"]] = artifact
    return _GroupResult(artifacts=artifacts, physical_artifact=physical_artifact)


def execute_convergence_tasks(
    prepared: list[dict[str, Any]],
    *,
    capacity: ResourceCapacity,
    exclusive: bool,
    schedule_path: Path,
    store: ArtifactStore,
    repo_root: Path,
    andrea_repo: Path,
    run_root: Path,
    image_id: str,
) -> None:
    """Schedule complete fan-out groups and bind their four logical artifacts."""

    groups = _group_tasks(prepared)
    missing_groups = [group for group in groups if any(task["artifact"] is None for task in group)]
    scheduled = []
    for group in missing_groups:
        spec = _physical_spec(group[0])
        key = ArtifactKey.from_spec("convergence-execution", spec)
        scheduled.append(
            ScheduledTask(
                task_id=key.value,
                payload=group,
                claim=ResourceClaim(
                    cpu_threads=int(group[0]["run"]["resources"]["threads"]),
                    memory_bytes=round(
                        float(group[0]["run"]["resources"]["ram_gb"]) * _GIB
                    ),
                    exclusive=exclusive,
                ),
            )
        )
    try:
        result = execute_resource_schedule(
            scheduled,
            capacity=capacity,
            worker=lambda group: _execute_group(
                group,
                store=store,
                repo_root=repo_root,
                andrea_repo=andrea_repo,
                run_root=run_root,
                image_id=image_id,
            ),
            thread_name_prefix="spathi-convergence",
        )
    except ResourceScheduleError as exc:
        manifest = dict(exc.manifest)
        manifest.update(
            {
                "contract": _CONTRACT_ID,
                "physical_plans_declared": len(groups),
                "logical_results_declared": len(prepared),
                "physical_fits_executed": len(exc.manifest.get("tasks", [])),
            }
        )
        atomic_write_json(
            schedule_path,
            manifest,
            "failed convergence schedule",
            boundary=run_root,
        )
        if exc.__cause__ is None:
            raise
        raise exc.__cause__ from exc

    by_key: dict[ArtifactKey, Artifact] = {}
    for execution in result.tasks:
        group_result = execution.result
        if not isinstance(group_result, _GroupResult):
            raise RuntimeError("Convergence scheduler returned an invalid group result")
        by_key.update(group_result.artifacts)
    for task in prepared:
        if task["artifact"] is None:
            artifact = by_key.get(task["key"])
            if artifact is None or artifact.key != task["key"]:
                raise RuntimeError("Convergence fan-out omitted a logical inference artifact")
            task["artifact"] = artifact

    manifest = result.manifest()
    manifest.update(
        {
            "status": "complete",
            "contract": _CONTRACT_ID,
            "capacity": {
                "cpu_threads": capacity.cpu_threads,
                "memory_bytes": capacity.memory_bytes,
                "max_concurrent_tasks": capacity.max_concurrent_tasks,
            },
            "utilization": result.utilization(capacity),
            "physical_plans_declared": len(groups),
            "logical_results_declared": len(prepared),
            "physical_fits_executed": len(result.tasks),
            "logical_results_materialized": sum(1 for task in prepared if not task["cache_hit"]),
            "logical_cache_hits": sum(1 for task in prepared if task["cache_hit"]),
        }
    )
    atomic_write_json(
        schedule_path,
        manifest,
        "convergence schedule",
        boundary=run_root,
    )


def validate_logical_convergence_artifact(
    task: Mapping[str, Any],
    *,
    artifact: Artifact,
    store: ArtifactStore,
) -> None:
    """Bind a logical handoff back to its immutable physical execution evidence."""

    binding = task.get("convergence_binding")
    if not isinstance(binding, ConvergenceBinding):
        raise BenchmarkConfigError("Convergence validation requires a logical binding")
    output = artifact.path / "output" / str(task["run"]["run_id"])
    provenance = load_json_object(
        output / "convergence-provenance.json",
        "convergence logical provenance",
    )
    physical_spec = _physical_spec(task)
    physical_key = store.key_for("convergence-execution", physical_spec)
    physical = store.lookup(physical_key)
    expected = {
        "schema_version": "1.0",
        "artifact_type": "spathi_convergence_logical_result",
        "execution_contract": _CONTRACT_ID,
        "physical_artifact": physical_key.value,
        "physical_run_id": binding.plan["physical_run_id"],
        "physical_plan_sha256": canonical_json_sha256(binding.plan),
        "run_id": binding.prefix["run_id"],
        "configuration_id": binding.prefix["configuration_id"],
        "configuration_sha256": binding.prefix["configuration_sha256"],
        "n_estimators": binding.prefix["n_estimators"],
        "source_network_sha256": provenance.get("source_network_sha256"),
        "physical_fits": 1,
        "logical_results": 4,
    }
    if provenance != expected:
        raise BenchmarkConfigError("Convergence logical provenance differs from its frozen plan")
    if physical is None or physical.manifest.get("spec") != physical_spec:
        raise BenchmarkConfigError("Convergence physical execution evidence is missing")
    fanout = load_json_object(
        physical.file("fanout-manifest.json"),
        "convergence physical fan-out evidence",
    )
    matches = [
        item
        for item in fanout.get("logical_outputs", [])
        if isinstance(item, dict) and item.get("run_id") == binding.prefix["run_id"]
    ]
    if len(matches) != 1:
        raise BenchmarkConfigError("Convergence physical evidence omits the logical prefix")
    network_records = [
        item
        for item in matches[0].get("files", [])
        if isinstance(item, dict) and item.get("path") == "network.csv"
    ]
    if (
        len(network_records) != 1
        or network_records[0].get("sha256") != provenance["source_network_sha256"]
    ):
        raise BenchmarkConfigError("Convergence logical network differs from physical evidence")
    if (
        _source_network_digest_from_merged(
            output / "merged_network_raw.csv",
            run_id=str(task["run"]["run_id"]),
        )
        != provenance["source_network_sha256"]
    ):
        raise BenchmarkConfigError("Convergence merged network is not the recorded forest prefix")


__all__ = [
    "ConvergenceBinding",
    "execute_convergence_tasks",
    "index_convergence_plans",
    "logical_execution_contract",
    "validate_logical_convergence_artifact",
]
