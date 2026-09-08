from __future__ import annotations

import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.pipeline as pipeline
import spathi_bench.production_policy as production_policy
import spathi_bench.scalability_workflow as scalability
from spathi_bench.artifacts import Artifact, ArtifactStore, canonical_json_sha256, sha256_file
from spathi_bench.config import (
    BenchmarkConfigError,
    load_profile,
    validate_json,
    validate_profile_references,
)
from spathi_bench.runtime import CommandExecutionError
from spathi_bench.state import load_state, write_state

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCKED_SPATHI_COMMIT = json.loads((REPO_ROOT / "repos.lock.json").read_text(encoding="utf-8"))[
    "repositories"
]["spathi"]["commit"]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _execution_resources(*, threads: int = 4, memory_gib: float = 8.0) -> dict[str, Any]:
    return {
        "cpu_threads": threads,
        "memory_bytes": round(memory_gib * 1024**3),
        "memory_gib": memory_gib,
        "cpuset_cpus": list(range(threads)),
        "planner": "heuristic",
        "planner_time_limit_seconds": 60.0,
        "output_profile": "canonical",
        "exclusive": True,
        "execution_plan_fingerprint": "a" * 64,
        "timeout_seconds": 43_200.0,
    }


def _andrea_measurement(
    resources: dict[str, Any],
    *,
    wall_time_seconds: float = 3.5,
    cpu_time_seconds: float | None = 7.0,
    peak_memory_bytes: int | None = 4096,
    telemetry_status: str = "complete",
) -> dict[str, Any]:
    cpuset = list(resources["cpuset_cpus"])
    return {
        "schema_version": "1.0",
        "scope": "physical_task",
        "wall_time_seconds": wall_time_seconds,
        "container_wall_time_seconds": wall_time_seconds - 0.1,
        "cpu_time_seconds": cpu_time_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "output_bytes": 256,
        "io_read_bytes": None,
        "io_write_bytes": None,
        "task_started_at_utc": "2026-01-01T00:00:00Z",
        "task_finished_at_utc": "2026-01-01T00:00:04Z",
        "task_started_monotonic_ns": 1_000_000_000,
        "task_finished_monotonic_ns": 4_500_000_000,
        "container_started_at_utc": "2026-01-01T00:00:00Z",
        "container_finished_at_utc": "2026-01-01T00:00:04Z",
        "assigned_resources": {
            "threads": resources["cpu_threads"],
            "ram_limit_bytes": resources["memory_bytes"],
            "cpuset_cpus_requested": cpuset,
            "cpuset_cpus_configured": cpuset,
            "cpuset_cpus_effective": cpuset,
            "cpu_quota_cores_effective": float(resources["cpu_threads"]),
            "ram_limit_bytes_effective": resources["memory_bytes"],
        },
        "telemetry": {
            "status": telemetry_status,
            "wall_source": "andrea_monotonic_clock",
            "wall_semantics": (
                "physical_task_end_to_end_including_container_launch_and_exit_collection"
            ),
            "container_wall_source": "docker_state_timestamps",
            "cpu_source": "cgroup-v2:cpu.stat",
            "memory_source": "cgroup-v2:memory.peak",
            "memory_semantics": "container_cgroup_peak",
            "io_source": None,
            "io_semantics": "unavailable",
            "throttled_time_seconds": 0.0,
            "oom_events": 0,
            "oom_kill_events": 0,
            "memory_limit_events": 0,
            "oom_killed": False,
        },
    }


def _andrea_execution_measurement(
    resources: dict[str, Any],
    *,
    wall_time_seconds: float = 9.0,
) -> dict[str, Any]:
    cpuset = [list(resources["cpuset_cpus"])]
    return {
        "schema_version": "1.0",
        "scope": "andrea_execution",
        "wall_time_seconds": wall_time_seconds,
        "child_wall_time_sum_seconds": 3.5,
        "postprocess_wall_time_seconds": 0.5,
        "cpu_time_seconds": 7.0,
        "peak_memory_bytes": 4096,
        "output_bytes": 512,
        "io_read_bytes": None,
        "io_write_bytes": None,
        "task_started_at_utc": "2026-01-01T00:00:00Z",
        "task_finished_at_utc": "2026-01-01T00:00:09Z",
        "task_started_monotonic_ns": 1_000_000_000,
        "task_finished_monotonic_ns": 10_000_000_000,
        "assigned_resources": {
            "threads_peak": resources["cpu_threads"],
            "ram_limit_peak_bytes": resources["memory_bytes"],
            "cpuset_cpus_requested_sets": cpuset,
            "cpuset_cpus_effective_sets": cpuset,
        },
        "telemetry": {
            "status": "complete",
            "wall_source": "andrea_monotonic_clock",
            "wall_semantics": (
                "andrea_execution_end_to_end_including_input_verification_preparation_"
                "scheduling_merging_and_requested_output_materialization"
            ),
            "cpu_source": "sum_of_child_container_cpu",
            "cpu_semantics": "sum_of_physical_container_cpu_excludes_andrea_host_process",
            "memory_source": "child_container_cgroup_peaks",
            "memory_semantics": (
                "overlap_upper_bound_from_physical_container_peaks_excludes_andrea_host_process"
            ),
            "io_source": None,
            "io_semantics": "unavailable",
        },
    }


def _andrea_logical_measurement(resources: dict[str, Any]) -> dict[str, Any]:
    cpuset = [list(resources["cpuset_cpus"])]
    return {
        "schema_version": "1.0",
        "scope": "logical_run",
        "wall_time_seconds": 4.5,
        "child_wall_time_sum_seconds": 7.0,
        "postprocess_wall_time_seconds": 0.5,
        "cpu_time_seconds": 6.0,
        "peak_memory_bytes": 4096,
        "output_bytes": 256,
        "io_read_bytes": None,
        "io_write_bytes": None,
        "task_started_at_utc": "2026-01-01T00:00:00Z",
        "task_finished_at_utc": "2026-01-01T00:00:05Z",
        "task_started_monotonic_ns": 1_000_000_000,
        "task_finished_monotonic_ns": 5_500_000_000,
        "assigned_resources": {
            "threads_peak": resources["cpu_threads"],
            "ram_limit_peak_bytes": resources["memory_bytes"],
            "cpuset_cpus_requested_sets": cpuset,
            "cpuset_cpus_effective_sets": cpuset,
        },
        "telemetry": {
            "status": "partial",
            "wall_source": "andrea_monotonic_clock",
            "wall_semantics": (
                "logical_run_from_first_physical_task_start_through_logical_"
                "postprocess; input_preparation_is_excluded"
            ),
            "cpu_source": "sum_of_child_container_cpu",
            "cpu_semantics": "sum_excludes_andrea_host_postprocess",
            "memory_source": "child_container_cgroup_peaks",
            "memory_semantics": (
                "overlap_upper_bound_from_child_peaks_excludes_andrea_host_postprocess"
            ),
            "io_source": None,
            "io_semantics": "unavailable",
        },
    }


def _group_policy(parameters: dict[str, Any]) -> dict[str, Any]:
    multi = dict(parameters)
    multi.pop("random_seed")
    single = {
        **multi,
        "weight_mode": "cell-distance",
        "distance_metric": "euclidean",
        "group_size_correction": "none",
    }
    shared = sorted(set(multi) - {"weight_mode", "distance_metric", "group_size_correction"})
    return {
        "routing_parameter": "group_count",
        "shared_parameters": shared,
        "branches": {
            "single_group": {
                "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                "parameters": single,
            },
            "multi_group": {
                "condition": {"minimum_group_count": 2},
                "parameters": multi,
            },
        },
    }


def _production(parameters: dict[str, Any], *, budget: str = "reduced") -> dict[str, Any]:
    policy = _group_policy(parameters)
    return {
        "schema_version": "1.0",
        "artifact_type": "spathi_production_policy",
        "budget": budget,
        "evidence_role": "descriptive" if budget == "reduced" else "confirmatory",
        "status": "descriptive" if budget == "reduced" else "confirmed",
        "decision": "descriptive_only" if budget == "reduced" else "reference_retained",
        "policy": {
            "policy_id": "spathi_production_v1",
            "policy_sha256": canonical_json_sha256(policy),
            "group_policy": policy,
            "spathi_commit": LOCKED_SPATHI_COMMIT,
        },
        "provenance": {
            name: {"path": f"work/efficiency/{name}.json", "sha256": "9" * 64}
            for name in (
                "calibration_selection",
                "efficiency_plan",
                "efficiency_evidence",
                "efficiency_results",
            )
        },
    }


def _copy_materialization_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for relative in (
        "images.lock.json",
        "schemas/scale-stress-panel.schema.json",
        "schemas/profile.schema.json",
        "schemas/repos-lock.schema.json",
        "schemas/run-metadata.schema.json",
        "schemas/spathi-production-config.schema.json",
        "experiments/v1/seeds/seed-plan.json",
        "repos.lock.json",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    return root


def test_committed_scalability_profiles_are_explicit_scale_stress_contracts() -> None:
    reduced = load_profile(REPO_ROOT, "scalability-reduced", require_runnable=True)
    full = load_profile(REPO_ROOT, "scalability-full", require_runnable=True)

    assert reduced["purpose"] == full["purpose"] == "scalability"
    assert reduced["statistics"]["status"] == full["statistics"]["status"] == "not_applicable"
    assert reduced["inference_batches"] == full["inference_batches"] == []
    assert reduced["generation_jobs"][0]["dataset_role"] == "scalability"
    assert full["generation_jobs"][0]["dataset_role"] == "scalability"
    assert "biological-realism" in reduced["description"]
    assert "public synthetic grid" in full["description"]


def test_materialization_executes_one_shape_with_scalability_seed(
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
) -> None:
    root = _copy_materialization_repository(tmp_path)
    template = load_profile(REPO_ROOT, "scalability-reduced", require_runnable=True)
    panel = json.loads(
        (REPO_ROOT / "experiments/v1/simulations/scale_stress/scale-reduced.json").read_text(
            encoding="utf-8"
        )
    )
    production = _production(frozen_spathi_params)
    production_path = _write_json(root / "work/efficiency/production-config.json", production)

    materialized = scalability.materialize_scalability_dataset(
        repo_root=root,
        budget="reduced",
        template_profile=template,
        production_config_path=production_path,
        production=production,
        panel=panel,
        dataset=panel["datasets"][0],
        stage="scout",
        target_scope={
            "method": "sha256_gene_id",
            "size": 64,
            "salt": "spathi-scalability-scout-v1",
            "dataset_id": panel["datasets"][0]["id"],
        },
        timeout_seconds=43_200.0,
        output_dir=root / "work/scalability/materialized",
    )

    profile = json.loads(Path(materialized["profile_path"]).read_text(encoding="utf-8"))
    request = json.loads(Path(materialized["request_path"]).read_text(encoding="utf-8"))
    metadata = json.loads(Path(materialized["metadata_path"]).read_text(encoding="utf-8"))
    assert profile["production_config"] == "work/efficiency/production-config.json"
    assert len(request["datasets"]) == 1
    assert request["datasets"][0]["id"] == panel["datasets"][0]["id"]
    assert request["scale_grid"] == {
        "provenance": "author_defined_public_synthetic_grid",
        "cells": [sum(group["cells"] for group in panel["datasets"][0]["groups"])],
        "genes": [panel["datasets"][0]["genes"]],
        "transcription_factors": [panel["datasets"][0]["tfs"]],
    }
    assert request["expression_preparation"] == panel["expression_preparation"]
    assert metadata["runs"][0]["seed_derivation"] == materialized["seed_derivation"]
    assert metadata["runs"][0]["seed_derivation"]["namespace"] == "reduced.scalability"
    assert profile["inference_batches"][0]["dataset_role"] == "scalability"
    assert profile["generation_jobs"][0]["id"] == (
        f"generate_{panel['datasets'][0]['id']}"
    )
    assert profile["id"].endswith("_scout")
    tools = json.loads(Path(materialized["tools_path"]).read_text(encoding="utf-8"))
    assert tools["runs"][0]["params"]["target_scope"]["size"] == 64
    assert tools["runs"][0]["resources"]["timeout_seconds"] == 43_200.0

    tools_path = Path(materialized["tools_path"])
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    tools["runs"][0]["params"]["group_policy"]["branches"]["multi_group"]["parameters"][
        "n_estimators"
    ] += 1
    _write_json(tools_path, tools)
    with pytest.raises(BenchmarkConfigError, match="frozen policy"):
        validate_profile_references(root, profile)


def test_materialized_synthetic_control_declares_its_own_scale_grid(
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
) -> None:
    root = _copy_materialization_repository(tmp_path)
    template = load_profile(REPO_ROOT, "scalability-full", require_runnable=True)
    panel = json.loads(
        (REPO_ROOT / "experiments/v1/simulations/scale_stress/scale-full.json").read_text(
            encoding="utf-8"
        )
    )
    production = _production(frozen_spathi_params, budget="full")
    production_path = _write_json(root / "work/efficiency/production-config.json", production)
    control = next(
        dataset
        for dataset in panel["datasets"]
        if dataset["design_role"]["kind"] == "synthetic_control"
    )

    materialized = scalability.materialize_scalability_dataset(
        repo_root=root,
        budget="full",
        template_profile=template,
        production_config_path=production_path,
        production=production,
        panel=panel,
        dataset=control,
        stage="all_targets",
        target_scope={"method": "all"},
        timeout_seconds=86_400.0,
        output_dir=root / "work/scalability/materialized-control",
    )

    request = json.loads(Path(materialized["request_path"]).read_text(encoding="utf-8"))
    assert request["scale_grid"] == {
        "provenance": "author_defined_public_synthetic_grid",
        "cells": [sum(group["cells"] for group in control["groups"])],
        "genes": [control["genes"]],
        "transcription_factors": [control["tfs"]],
    }
    assert request["expression_preparation"] == panel["expression_preparation"]


def test_materialization_rejects_a_scope_inconsistent_with_its_budget(
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
) -> None:
    root = _copy_materialization_repository(tmp_path)
    template = load_profile(REPO_ROOT, "scalability-reduced", require_runnable=True)
    panel = json.loads(
        (REPO_ROOT / "experiments/v1/simulations/scale_stress/scale-reduced.json").read_text(
            encoding="utf-8"
        )
    )
    production = _production(frozen_spathi_params)
    production_path = _write_json(root / "work/efficiency/production-config.json", production)
    dataset = panel["datasets"][0]

    with pytest.raises(BenchmarkConfigError, match="invalid target scope"):
        scalability.materialize_scalability_dataset(
            repo_root=root,
            budget="reduced",
            template_profile=template,
            production_config_path=production_path,
            production=production,
            panel=panel,
            dataset=dataset,
            stage="scout",
            target_scope={
                "method": "sha256_gene_id",
                "size": 128,
                "salt": "spathi-scalability-scout-v1",
                "dataset_id": dataset["id"],
            },
            timeout_seconds=43_200.0,
            output_dir=root / "work/scalability/invalid-scope",
        )


def _publish(
    store: ArtifactStore,
    *,
    kind: str,
    spec: dict[str, Any],
    files: dict[str, str],
) -> Artifact:
    key = store.key_for(kind, spec)
    with store.staging_directory(key) as staging:
        for relative, content in files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return store.publish_from_temp(staging, kind=kind, spec=spec)


def _reference(artifact: Artifact, *, cache_hit: bool = False) -> dict[str, Any]:
    return {
        "kind": artifact.key.kind,
        "digest": artifact.key.digest,
        "key": artifact.key.value,
        "path": str(artifact.path),
        "cache_hit": cache_hit,
    }


def _generation_measurement(*, seconds: float = 1.0) -> str:
    return json.dumps(
        {
            "schema_version": "1.0",
            "scope": "scale_stress_generation",
            "wall_time_seconds": seconds,
            "wall_source": "benchmark_monotonic_clock",
            "wall_semantics": (
                "fresh_artifact_generation_excluding_cache_lookup_and_publication"
            ),
        }
    )


def test_measurement_reconciles_completion_resources_and_disk(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    store = ArtifactStore(root / "work/v1/artifacts")
    dataset_id = "scale_fixture"
    run_id = "spathi_scalability_scale_fixture"
    dataset_spec = {
        "id": dataset_id,
        "seed": 71,
        "genes": 12,
        "tfs": 2,
        "groups": [{"id": "only", "cells": 3}],
        "network_mode": "invariant",
    }
    dataset_artifact = _publish(
        store,
        kind="dataset",
        spec={"dataset": dataset_id},
        files={
            "generation-measurement.json": _generation_measurement(),
            f"output/panel/datasets/{dataset_id}/expression.tsv": (
                "gene\tcell_1\tcell_2\tcell_3\nG1\t1\t2\t3\n"
            ),
        },
    )
    metadata = {
        "status": "complete",
        "input_dimensions": {
            "cells": 3,
            "genes": 12,
            "groups": 1,
            "transcription_factors": 2,
            "targets": 12,
        },
        "effective_parameters": {
            "target_eligibility": {
                "globally_eligible_targets": 10,
                "globally_ineligible_targets": 2,
            }
        },
        "models": {
            "requested": 12,
            "completed": 12,
            "trained": 10,
            "fit_or_importance_failures": 0,
            "fitted_estimators_total": 1000,
        },
        "phase_times_seconds": {"total": 8.5},
        "parallelism": {
            "threads_requested": 4,
            "threads_effective": 4,
            "threads_available": 8,
        },
        "memory_estimate_bytes": {"estimated_peak_heap_with_rough_trees": 2048},
    }
    resources = _execution_resources()
    report = {
        "status": "executed",
        "output_profile": "canonical",
        "execution": {"measurement": _andrea_execution_measurement(resources)},
        "tools": {
            "results": {
                run_id: {
                    "status": "completed",
                    "measurement": _andrea_measurement(resources),
                }
            }
        },
    }
    inference_artifact = _publish(
        store,
        kind="inference",
        spec={"dataset": dataset_id, "run": run_id},
        files={
            f"output/andrea/tools/{run_id}/io/out/run_metadata.json": json.dumps(metadata),
            "output/andrea/run_report.json": json.dumps(report),
            f"output/andrea/tools/{run_id}/io/out/network.csv": (
                "source,target,score,sign,evidence,context\n"
            ),
        },
    )
    state = {
        "status": "inferred",
        "artifact_store": str(store.root),
        "datasets": [{"dataset_id": dataset_id, "dataset_artifact": _reference(dataset_artifact)}],
        "inference_runs": [
            {
                "dataset_id": dataset_id,
                "run_id": run_id,
                "algorithm_seed": 99,
                "artifact": _reference(inference_artifact),
            }
        ],
    }
    derivation = {
        "namespace": "reduced.scalability",
        "stream": "inference",
        "coordinates": {
            "purpose": "scalability",
            "dataset_id": dataset_id,
            "replicate_index": 1,
        },
    }

    measurement = scalability.collect_scalability_measurement(
        repo_root=root,
        runtime_state=state,
        dataset_spec=dataset_spec,
        stage="all_targets",
        target_scope={"method": "all"},
        run_id=run_id,
        algorithm_seed=99,
        seed_derivation=derivation,
        resources=resources,
    )

    assert measurement["policy_branch"] == "single_group"
    assert measurement["dimensions"]["targets"] == 12
    assert measurement["inference"]["targets_eligible"] == 10
    assert measurement["inference"]["andrea_execution_measurement"]["wall_time_seconds"] == 9.0
    assert measurement["inference"]["models_completed"] == 12
    assert measurement["inference"]["trees_fitted"] == 1000
    assert measurement["inference"]["andrea_measurement"]["peak_memory_bytes"] == 4096
    assert measurement["inference"]["threads_effective"] == 4
    assert measurement["inference"]["estimated_peak_heap_with_rough_trees_bytes"] == 2048
    assert measurement["inference"]["spathi_output_bytes"] > 0

    state["inference_runs"][0]["artifact"]["cache_hit"] = True
    with pytest.raises(BenchmarkConfigError, match="cache"):
        scalability.collect_scalability_measurement(
            repo_root=root,
            runtime_state=state,
            dataset_spec=dataset_spec,
            stage="all_targets",
            target_scope={"method": "all"},
            run_id=run_id,
            algorithm_seed=99,
            seed_derivation=derivation,
            resources=resources,
        )


def _measurement_runtime_state(run_root: Path, profile_id: str) -> dict[str, Any]:
    state = {
        "schema_version": "1.0",
        "profile_id": profile_id,
        "status": "inferred",
    }
    write_state(run_root, state)
    return state


def test_published_measurement_is_immediately_bound_to_runtime_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    run_root = root / "work/v1/scalability_fixture/run"
    profile_id = "scalability_fixture"
    state = _measurement_runtime_state(run_root, profile_id)
    measurement_path = root / "work/v1/workflows/scalability/measurements/fixture.json"
    measurement = {"dataset_id": "fixture", "status": "completed", "observed_seconds": 1.25}

    published = scalability._publish_scalability_measurement(
        repo_root=root,
        run_root=run_root,
        profile_id=profile_id,
        runtime_state=state,
        measurement_path=measurement_path,
        measurement=measurement,
        dataset_id="fixture",
    )

    recorded_state = load_state(run_root, profile_id)
    assert recorded_state["scalability_measurement"] == {
        "path": published.relative_to(root).as_posix(),
        "sha256": sha256_file(published),
    }
    assert (
        scalability._load_bound_scalability_measurement(
            repo_root=root,
            runtime_state=recorded_state,
            measurement_path=published,
            dataset_id="fixture",
        )
        == measurement
    )


def test_resume_rejects_orphaned_measurement_after_publication_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    run_root = root / "work/v1/scalability_fixture/run"
    profile_id = "scalability_fixture"
    state = _measurement_runtime_state(run_root, profile_id)
    measurement_path = root / "work/v1/workflows/scalability/measurements/fixture.json"

    def crash_before_state_commit(_run_root: Path, _state: dict[str, Any]) -> Path:
        raise RuntimeError("simulated state write crash")

    monkeypatch.setattr(scalability, "write_state", crash_before_state_commit)
    with pytest.raises(RuntimeError, match="simulated state write crash"):
        scalability._publish_scalability_measurement(
            repo_root=root,
            run_root=run_root,
            profile_id=profile_id,
            runtime_state=state,
            measurement_path=measurement_path,
            measurement={"dataset_id": "fixture", "status": "completed"},
            dataset_id="fixture",
        )
    assert measurement_path.is_file()

    with pytest.raises(BenchmarkConfigError, match="orphaned from its runtime state"):
        scalability._load_bound_scalability_measurement(
            repo_root=root,
            runtime_state=load_state(run_root, profile_id),
            measurement_path=measurement_path,
            dataset_id="fixture",
        )


def test_resume_rejects_tampered_runtime_bound_measurement(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    run_root = root / "work/v1/scalability_fixture/run"
    profile_id = "scalability_fixture"
    state = _measurement_runtime_state(run_root, profile_id)
    measurement_path = root / "work/v1/workflows/scalability/measurements/fixture.json"
    scalability._publish_scalability_measurement(
        repo_root=root,
        run_root=run_root,
        profile_id=profile_id,
        runtime_state=state,
        measurement_path=measurement_path,
        measurement={"dataset_id": "fixture", "status": "completed", "observed_seconds": 1.25},
        dataset_id="fixture",
    )
    measurement_path.write_text(
        '{"dataset_id":"fixture","status":"completed","observed_seconds":999}\n',
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkConfigError, match="measurement changed"):
        scalability._load_bound_scalability_measurement(
            repo_root=root,
            runtime_state=load_state(run_root, profile_id),
            measurement_path=measurement_path,
            dataset_id="fixture",
        )


def test_pipeline_transports_only_an_identity_validated_failed_andrea_report(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "andrea-run"
    progress_path = _write_json(
        run_dir / "tools/spathi/io/out/progress.json",
        {
            "percent": 40,
            "status": "running",
            "phase": "model_inference",
            "message": "Training",
            "warnings": [],
        },
    )
    tool_log = run_dir / "tools/spathi/io/container.log"
    tool_log.parent.mkdir(parents=True, exist_ok=True)
    tool_log.write_text("Out of memory\n", encoding="utf-8")
    report_path = _write_json(
        run_dir / "run_report.json",
        {
            "run_id": "spathi",
            "status": "executed",
            "dataset": {"id": "dataset"},
            "execution": {
                "measurement": _andrea_execution_measurement(_execution_resources())
            },
            "tools": {
                "selected": ["spathi"],
                "status_by_tool": {"spathi": "failed"},
                "tool_origins": {"spathi": "custom"},
                "failed": {"spathi": "Container failed"},
                "results": {
                    "spathi": {
                        "status": "failed",
                        "exit_code": 137,
                        "error": "Container failed: out of memory",
                        "execution": {"mode": "group_native"},
                        "measurement": _andrea_measurement(
                            _execution_resources(), telemetry_status="partial"
                        ),
                        "progress_path": progress_path.relative_to(run_dir).as_posix(),
                        "logs_path": tool_log.relative_to(run_dir).as_posix(),
                    }
                },
            },
        },
    )
    command_log = tmp_path / "command.log"
    command_log.write_text("ANDREA failed\n", encoding="utf-8")
    command_failure = CommandExecutionError(
        command=["python", "-m", "andrea"],
        return_code=1,
        log_path=command_log,
    )

    failure = pipeline._validated_failed_inference(
        report_path,
        run_id="spathi",
        dataset_id="dataset",
        expected_origin="custom",
        command_failure=command_failure,
        batch_id="batch",
    )

    assert failure.tool_exit_code == 137
    assert failure.measurement["scope"] == "physical_task"
    assert failure.progress is not None and failure.progress["percent"] == 40
    assert failure.tool_log_tail == "Out of memory"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["dataset"]["id"] = "different-dataset"
    _write_json(report_path, report)
    with pytest.raises(BenchmarkConfigError, match="wrong dataset identity"):
        pipeline._validated_failed_inference(
            report_path,
            run_id="spathi",
            dataset_id="dataset",
            expected_origin="custom",
            command_failure=command_failure,
            batch_id="batch",
        )

    report["dataset"]["id"] = "dataset"
    _write_json(report_path, report)
    _write_json(progress_path, {"percent": "corrupt"})
    with pytest.raises(BenchmarkConfigError, match="progress has an invalid contract"):
        pipeline._validated_failed_inference(
            report_path,
            run_id="spathi",
            dataset_id="dataset",
            expected_origin="custom",
            command_failure=command_failure,
            batch_id="batch",
        )


def test_pipeline_accepts_grouped_failure_only_with_logical_run_telemetry(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "andrea-run"
    tool_log = run_dir / "tools/genie3/group_aggregation.log"
    tool_log.parent.mkdir(parents=True, exist_ok=True)
    tool_log.write_text("All grouped executions failed\n", encoding="utf-8")
    report_path = _write_json(
        run_dir / "run_report.json",
        {
            "run_id": "genie3",
            "status": "executed",
            "dataset": {"id": "dataset"},
            "execution": {
                "measurement": _andrea_execution_measurement(_execution_resources())
            },
            "tools": {
                "selected": ["genie3"],
                "status_by_tool": {"genie3": "failed"},
                "tool_origins": {"genie3": "catalog"},
                "failed": {"genie3": "All grouped executions failed"},
                "results": {
                    "genie3": {
                        "status": "failed",
                        "exit_code": 1,
                        "error": "All grouped executions failed",
                        "execution": {"mode": "group_emulated"},
                        "measurement": _andrea_logical_measurement(_execution_resources()),
                        "progress_path": None,
                        "logs_path": tool_log.relative_to(run_dir).as_posix(),
                    }
                },
            },
        },
    )
    command_log = tmp_path / "command.log"
    command_log.write_text("ANDREA reported grouped failure\n", encoding="utf-8")
    command_failure = CommandExecutionError(
        command=["python", "-m", "andrea"],
        return_code=1,
        log_path=command_log,
    )

    failure = pipeline._validated_failed_inference(
        report_path,
        run_id="genie3",
        dataset_id="dataset",
        expected_origin="catalog",
        command_failure=command_failure,
        batch_id="group-only",
    )

    assert failure.measurement["scope"] == "logical_run"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["tools"]["results"]["genie3"]["measurement"] = _andrea_measurement(
        _execution_resources()
    )
    _write_json(report_path, report)
    with pytest.raises(BenchmarkConfigError, match="resource telemetry"):
        pipeline._validated_failed_inference(
            report_path,
            run_id="genie3",
            dataset_id="dataset",
            expected_origin="catalog",
            command_failure=command_failure,
            batch_id="group-only",
        )


@pytest.mark.parametrize(
    ("tool_exit_code", "tool_error", "expected_kind"),
    [
        (137, "Container was OOMKilled: out of memory", "oom"),
        (124, "Execution time limit exceeded", "timeout"),
        (2, "Model fitting failed", "tool_error"),
        (127, "Cannot connect to the Docker daemon", "infrastructure_error"),
    ],
)
def test_failed_measurement_preserves_attributable_operational_evidence(
    tmp_path: Path,
    tool_exit_code: int,
    tool_error: str,
    expected_kind: str,
) -> None:
    root = tmp_path / "repo"
    store = ArtifactStore(root / "work/v1/artifacts")
    dataset_id = "scale_failure"
    dataset_spec = {
        "id": dataset_id,
        "seed": 71,
        "genes": 12,
        "tfs": 2,
        "groups": [{"id": "only", "cells": 3}],
        "network_mode": "invariant",
    }
    dataset_artifact = _publish(
        store,
        kind="dataset",
        spec={"dataset": dataset_id},
        files={
            "generation-measurement.json": _generation_measurement(),
            f"output/panel/datasets/{dataset_id}/expression.tsv": (
                "gene\tcell_1\tcell_2\tcell_3\nG1\t1\t2\t3\n"
            )
        },
    )
    state = {
        "status": "generated",
        "artifact_store": str(store.root),
        "datasets": [{"dataset_id": dataset_id, "dataset_artifact": _reference(dataset_artifact)}],
    }
    command_log = root / "work/v1/scale_failure/logs/20-inference.log"
    command_log.parent.mkdir(parents=True, exist_ok=True)
    command_log.write_text(tool_error + "\n", encoding="utf-8")
    resources = _execution_resources()
    failure = scalability.InferenceExecutionFailure(
        dataset_id=dataset_id,
        batch_id="infer_scale_failure",
        run_id="spathi_scalability_scale_failure",
        orchestrator_return_code=1,
        tool_exit_code=tool_exit_code,
        tool_error=tool_error,
        measurement=_andrea_measurement(
            resources,
            cpu_time_seconds=None,
            peak_memory_bytes=None,
            telemetry_status="partial",
        ),
        execution_measurement=_andrea_execution_measurement(resources),
        command_log_path=command_log,
        progress={
            "percent": 62,
            "status": "running",
            "phase": "model_inference",
            "message": "Training models",
            "warnings": [],
        },
        tool_log_tail=tool_error,
    )
    derivation = {
        "namespace": "reduced.scalability",
        "stream": "inference",
        "coordinates": {
            "purpose": "scalability",
            "dataset_id": dataset_id,
            "replicate_index": 1,
        },
    }
    measurement = scalability.collect_failed_scalability_measurement(
        repo_root=root,
        runtime_state=state,
        dataset_spec=dataset_spec,
        stage="all_targets",
        target_scope={"method": "all"},
        run_id="spathi_scalability_scale_failure",
        algorithm_seed=99,
        seed_derivation=derivation,
        resources=resources,
        failure=failure,
    )

    assert measurement["status"] == "failed"
    assert measurement["failure"]["failure_kind"] == expected_kind
    assert measurement["observations"]["andrea_measurement"]["peak_memory_bytes"] is None
    assert measurement["diagnostics"]["progress"]["percent"] == 62
    assert measurement["diagnostics"]["command_log"]["sha256"] == sha256_file(command_log)
    assert (
        scalability.validate_failed_scalability_measurement(
            repo_root=root,
            runtime_state=state,
            dataset_spec=dataset_spec,
            stage="all_targets",
            target_scope={"method": "all"},
            algorithm_seed=99,
            seed_derivation=derivation,
            resources=resources,
            recorded=measurement,
        )
        == measurement
    )


def _fake_measurement(
    dataset: dict[str, Any], *, budget: str, stage: str = "all_targets"
) -> dict[str, Any]:
    dataset_id = dataset["id"]
    groups = len(dataset["groups"])
    genes = dataset["genes"]
    resources = _execution_resources(
        threads=8,
        memory_gib=48.0 if budget == "full" else 32.0,
    )
    resources["timeout_seconds"] = 86_400.0 if budget == "full" else 43_200.0
    scout_size = 128 if budget == "full" else 64
    target_count = scout_size if stage == "scout" else genes
    target_scope = (
        {
            "method": "sha256_gene_id",
            "size": scout_size,
            "salt": "spathi-scalability-scout-v1",
            "dataset_id": dataset_id,
        }
        if stage == "scout"
        else {"method": "all"}
    )
    return {
        "dataset_id": dataset_id,
        "stage": stage,
        "target_scope": target_scope,
        "status": "completed",
        "network_mode": dataset["network_mode"],
        "policy_branch": "single_group" if groups == 1 else "multi_group",
        "generation_seed": dataset["seed"],
        "algorithm_seed": 88,
        "seed_derivation": {
            "namespace": f"{budget}.scalability",
            "stream": "inference",
            "coordinates": {
                "purpose": "scalability",
                "dataset_id": dataset_id,
                "replicate_index": 1,
            },
        },
        "dimensions": {
            "cells": sum(group["cells"] for group in dataset["groups"]),
            "genes": genes,
            "groups": groups,
            "transcription_factors": dataset["tfs"],
            "targets": target_count,
        },
        "execution_limits": resources,
        "generation": {
            "wall_time_seconds": 1.0,
            "cache_hit": False,
            "dataset_artifact_bytes": 100,
            "expression_bytes": 50,
        },
        "inference": {
            "spathi_wall_time_seconds": 1.5,
            "andrea_measurement": _andrea_measurement(
                resources,
                peak_memory_bytes=1024,
            ),
            "andrea_execution_measurement": _andrea_execution_measurement(
                resources,
                wall_time_seconds=2.0,
            ),
            "cache_hit": False,
            "inference_artifact_bytes": 200,
            "spathi_output_bytes": 70,
            "targets_requested": target_count,
            "targets_eligible": target_count,
            "targets_ineligible": 0,
            "models_requested": target_count * groups,
            "models_completed": target_count * groups,
            "models_trained": target_count * groups,
            "models_failed": 0,
            "trees_fitted": target_count * groups * 100,
            "threads_requested": 8,
            "threads_effective": 8,
            "threads_available": 8,
            "estimated_peak_heap_with_rough_trees_bytes": 4096,
        },
        "dataset_artifact": {"kind": "dataset", "digest": "2" * 64},
        "inference_artifact": {"kind": "inference", "digest": "3" * 64},
        "run_metadata": {"path": "work/run_metadata.json", "sha256": "4" * 64},
    }


def _fake_failed_measurement(
    dataset: dict[str, Any], *, budget: str, log_path: Path, root: Path
) -> dict[str, Any]:
    completed = _fake_measurement(dataset, budget=budget)
    for key in ("inference", "inference_artifact", "run_metadata"):
        completed.pop(key)
    completed["status"] = "failed"
    completed["failure"] = {
        "failure_kind": "oom",
        "stage": "inference",
        "message": "Container was OOMKilled",
        "orchestrator_exit_code": 1,
        "tool_exit_code": 137,
    }
    completed["observations"] = {
        "andrea_measurement": _andrea_measurement(
            completed["execution_limits"],
            cpu_time_seconds=None,
            peak_memory_bytes=None,
            telemetry_status="partial",
        ),
        "andrea_execution_measurement": _andrea_execution_measurement(
            completed["execution_limits"],
            wall_time_seconds=2.0,
        ),
    }
    completed["diagnostics"] = {
        "command_log": {
            "path": log_path.relative_to(root).as_posix(),
            "sha256": sha256_file(log_path),
            "size_bytes": log_path.stat().st_size,
        },
        "progress": None,
        "tool_log_tail": "Container was OOMKilled",
    }
    return completed


def _attempt_decision(dataset_id: str, stage: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "lane_id": "fixture_lane",
        "stage": stage,
        "decision": "attempt",
        "reason": "within_preregistered_resource_boundary",
        "evidence_dataset_id": dataset_id if stage == "all_targets" else None,
    }


def _fake_dataset_result(
    dataset: dict[str, Any],
    *,
    budget: str,
    all_targets: dict[str, Any],
) -> dict[str, Any]:
    scout = _fake_measurement(dataset, budget=budget, stage="scout")
    return {
        "dataset_id": dataset["id"],
        "status": all_targets["status"],
        "network_mode": dataset["network_mode"],
        "policy_branch": "single_group" if len(dataset["groups"]) == 1 else "multi_group",
        "generation_seed": dataset["seed"],
        "algorithm_seed": all_targets["algorithm_seed"],
        "seed_derivation": all_targets["seed_derivation"],
        "dimensions": {
            "cells": sum(group["cells"] for group in dataset["groups"]),
            "genes": dataset["genes"],
            "groups": len(dataset["groups"]),
            "transcription_factors": dataset["tfs"],
            "targets": dataset["genes"],
        },
        "lane_ids": ["fixture_lane"],
        "decisions": {
            "scout": _attempt_decision(dataset["id"], "scout"),
            "all_targets": _attempt_decision(dataset["id"], "all_targets"),
        },
        "scout": scout,
        "all_targets": all_targets,
    }


def test_result_document_completes_and_reports_a_mixed_scale_panel(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    profile_path = _write_json(root / "experiments/profile.json", {"id": "scale"})
    production_path = _write_json(root / "work/production.json", {"id": "policy"})
    log_path = root / "work/v1/failed.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("OOMKilled\n", encoding="utf-8")
    datasets = [
        {
            "id": "scale_completed",
            "seed": 10,
            "genes": 12,
            "tfs": 2,
            "groups": [{"id": "g1", "cells": 3}],
            "network_mode": "invariant",
        },
        {
            "id": "scale_failed",
            "seed": 11,
            "genes": 12,
            "tfs": 2,
            "groups": [{"id": "g1", "cells": 3}],
            "network_mode": "invariant",
        },
    ]
    production = {
        "decision": "reference_retained",
        "policy": {
            "policy_id": "spathi_production_v1",
            "policy_sha256": "9" * 64,
            "spathi_commit": "1" * 40,
        },
    }
    dataset_results = [
        _fake_dataset_result(
            datasets[0],
            budget="full",
            all_targets=_fake_measurement(datasets[0], budget="full"),
        ),
        _fake_dataset_result(
            datasets[1],
            budget="full",
            all_targets=_fake_failed_measurement(
                datasets[1], budget="full", log_path=log_path, root=root
            ),
        ),
    ]
    boundary_path = root / "experiments/v1/spathi/scalability-boundary.json"
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "experiments/v1/spathi/scalability-boundary.json", boundary_path)
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))

    results = scalability._result_document(
        repo_root=root,
        budget="full",
        template_path=profile_path,
        production_path=production_path,
        production=production,
        image_id="sha256:image",
        runtime_environment={
            "andrea_commit": "2" * 40,
            "andrea_environment_fingerprint": "3" * 64,
            "benchmark_environment_fingerprint": "4" * 64,
            "docker_runtime_environment_fingerprint": "5" * 64,
        },
        host_resources={
            "system": "Linux",
            "kernel_release": "test",
            "machine": "x86_64",
            "cpu_model": "test CPU",
            "logical_cpus": 8,
            "host_memory_bytes": 64 * 1024**3,
        },
        resources=_execution_resources(threads=8, memory_gib=48.0),
        expression_preparation={
            "input_scale": "nonnegative_synthetic_counts_like",
            "normalization": "library-size-log1p",
            "target_sum": 10000.0,
            "library_size_scope": "all_generated_genes_before_any_filtering",
            "formula": "log1p(target_sum * raw_value / per_cell_library_size)",
        },
        boundary_path=boundary_path,
        boundary=boundary,
        datasets=dataset_results,
    )

    validate_json(
        results,
        REPO_ROOT / "schemas/scalability-results.schema.json",
        "mixed scalability results",
    )
    assert results["status"] == "complete"
    assert results["summary"]["datasets_completed"] == 1
    assert results["summary"]["datasets_failed"] == 1
    assert results["summary"]["all_completed"] is False
    assert results["summary"]["failure_kinds"]["oom"] == 1


def test_infrastructure_failure_cannot_become_scalability_evidence(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "infrastructure.log"
    log_path.write_text("Cannot connect to the Docker daemon\n", encoding="utf-8")
    dataset = {
        "id": "scale_infrastructure",
        "seed": 10,
        "genes": 500,
        "tfs": 20,
        "groups": [{"id": "g1", "cells": 100}],
        "network_mode": "invariant",
    }
    failed = _fake_failed_measurement(
        dataset,
        budget="reduced",
        log_path=log_path,
        root=tmp_path,
    )
    failed["failure"]["failure_kind"] = "infrastructure_error"

    with pytest.raises(BenchmarkConfigError, match="Infrastructure failure"):
        scalability._attempt_resource_result(
            failed,
            full_target_count=500,
            scout_target_count=64,
        )


def test_high_level_workflow_freezes_publication_scale_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_spathi_params: dict[str, Any],
) -> None:
    root = tmp_path / "repo"
    for relative in {
        *scalability._IDENTITY_INPUTS,
        "experiments/v1/profiles/scalability-full.json",
        "experiments/v1/simulations/scale_stress/scale-full.json",
        "experiments/v1/simulations/scale_stress/scale-reduced.json",
        "schemas/profile.schema.json",
    }:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    work = root / "work"
    production = _production(frozen_spathi_params, budget="full")
    production_path = _write_json(work / "efficiency/production-config.json", production)

    monkeypatch.setattr(
        scalability,
        "completed_production_config",
        lambda **_kwargs: (production_path, production),
    )
    monkeypatch.setattr(
        scalability,
        "host_execution_lease",
        lambda **_kwargs: nullcontext(root / "execution.lock"),
    )
    attempted: list[tuple[str, str, int]] = []

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        attempted.append(
            (
                kwargs["dataset"]["id"],
                kwargs["stage"],
                kwargs["target_scope"].get("size", kwargs["dataset"]["genes"]),
            )
        )
        return _fake_measurement(kwargs["dataset"], budget="full", stage=kwargs["stage"])

    monkeypatch.setattr(scalability, "_run_one_dataset", fake_run)
    monkeypatch.setattr(
        scalability,
        "_host_resources",
        lambda: {
            "system": "Linux",
            "kernel_release": "test",
            "machine": "x86_64",
            "cpu_model": "test cpu",
            "logical_cpus": 8,
            "host_memory_bytes": 64 * 1024**3,
        },
    )
    monkeypatch.setattr(
        scalability,
        "observe_benchmark_repository",
        lambda *_args, **_kwargs: {
            "path": str(root),
            "commit": "1" * 40,
            "origin": "https://example.test/spathi-benchmarks.git",
            "clean": True,
        },
    )
    monkeypatch.setattr(scalability, "andrea_host_environment", lambda: {"fingerprint": "a" * 64})
    monkeypatch.setattr(
        scalability, "benchmark_host_environment", lambda: {"fingerprint": "b" * 64}
    )
    monkeypatch.setattr(
        scalability, "docker_runtime_environment", lambda: {"fingerprint": "c" * 64}
    )
    monkeypatch.setattr(
        scalability,
        "load_state",
        lambda *_args, **_kwargs: {
            "spathi_image": {"id": "sha256:image"},
            "repositories": {"observed": {"andrea": {"commit": "5" * 40}}},
            "host_environment": {"fingerprint": "6" * 64},
            "benchmark_environment": {"fingerprint": "7" * 64},
            "docker_runtime_environment": {"fingerprint": "8" * 64},
        },
    )
    rendered: dict[str, Path] = {}

    def fake_report(**kwargs: Path) -> dict[str, dict[str, str]]:
        rendered.update(kwargs)
        publication = json.loads(
            (kwargs["workflow_root"] / "workflow.json").read_text(encoding="utf-8")
        )
        assert publication["status"] == "publishing"
        assert "reports" not in publication
        relative = "work/v1/reports/full/" + kwargs["workflow_root"].name + "/scalability.html"
        report_path = kwargs["repo_root"] / relative
        manifest_path = report_path.parent / "scalability-report.manifest.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("<!doctype html>\n", encoding="utf-8")
        manifest_path.write_text("{}\n", encoding="utf-8")
        return {
            "schema_version": "1.0",
            "html": {"path": relative, "sha256": sha256_file(report_path)},
            "manifest": {
                "path": manifest_path.relative_to(kwargs["repo_root"]).as_posix(),
                "sha256": sha256_file(manifest_path),
            },
        }

    monkeypatch.setattr(scalability, "render_scalability_report", fake_report)

    state = scalability.run_scalability_workflow(
        repo_root=root,
        andrea_repo=tmp_path / "ANDREA",
        spathi_repo=tmp_path / "SPATHI",
        work_root=work,
        profile="full",
    )
    results = json.loads((root / state["results"]).read_text(encoding="utf-8"))
    validate_json(
        results,
        root / "schemas/scalability-results.schema.json",
        "test scalability results",
    )
    assert state["workflow_type"] == "scalability"
    assert results["evidence_role"] == "publication"
    assert results["claim_scope"] == "computational_scale_stress_only"
    assert results["summary"]["datasets_completed"] == len(results["datasets"])
    assert results["summary"]["scouts_attempted"] == len(results["datasets"])
    assert results["summary"]["all_targets_attempted"] == len(results["datasets"])
    assert [stage for _dataset, stage, _targets in attempted] == [
        stage for _dataset in results["datasets"] for stage in ("scout", "all_targets")
    ]
    assert all(targets == 128 for _dataset, stage, targets in attempted if stage == "scout")
    assert all(
        row["scout"]["algorithm_seed"] == row["all_targets"]["algorithm_seed"]
        for row in results["datasets"]
    )
    assert results["summary"]["all_completed"] is True
    assert state["interactive_report"].endswith("/scalability.html")
    assert state["reports"]["html"]["path"] == state["interactive_report"]
    assert (
        json.loads(
            (
                root / "work/v1/workflows/scalability/full" / state["flow_id"] / "workflow.json"
            ).read_text(encoding="utf-8")
        )["reports"]
        == state["reports"]
    )
    assert rendered["workflow_root"].name == state["flow_id"]


def test_scalability_rejects_missing_efficiency_policy(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkConfigError, match="justify-efficiency"):
        production_policy.completed_production_config(
            repo_root=tmp_path,
            work_root=tmp_path / "work",
            budget="reduced",
            efficiency_run_id=None,
        )


def _completed_efficiency_reference(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "repo"
    work = root / "work"
    workflow_root = work / "v1/workflows/efficiency/reduced/efficiency-current"
    production_path = _write_json(workflow_root / "production-config.json", {})
    workflow_path = _write_json(
        workflow_root / "workflow.json",
        {
            "workflow_type": "efficiency",
            "profile": "reduced",
            "status": "complete",
            "production_config": production_path.relative_to(root).as_posix(),
            "production_config_sha256": sha256_file(production_path),
        },
    )
    return root, work, workflow_root, workflow_path


def test_completed_production_policy_must_belong_to_the_requested_efficiency_run(
    tmp_path: Path,
) -> None:
    root, work, workflow_root, workflow_path = _completed_efficiency_reference(tmp_path)
    production_path = workflow_root / "production-config.json"
    foreign_path = _write_json(workflow_root.parent / "foreign-production-config.json", {})
    state = json.loads(workflow_path.read_text(encoding="utf-8"))
    state["production_config"] = foreign_path.relative_to(root).as_posix()
    state["production_config_sha256"] = sha256_file(foreign_path)
    _write_json(workflow_path, state)
    production_path.unlink()

    with pytest.raises(BenchmarkConfigError, match="policy is missing or changed"):
        production_policy.completed_production_config(
            repo_root=root,
            work_root=work,
            budget="reduced",
            efficiency_run_id="efficiency-current",
        )


@pytest.mark.parametrize("symlink_kind", ["workflow", "production", "directory"])
def test_completed_production_policy_rejects_symlinked_workflow_material(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    root, work, workflow_root, workflow_path = _completed_efficiency_reference(tmp_path)
    if symlink_kind == "workflow":
        target = workflow_root / "workflow-target.json"
        workflow_path.rename(target)
        workflow_path.symlink_to(target.name)
    elif symlink_kind == "production":
        production_path = workflow_root / "production-config.json"
        target = workflow_root / "production-target.json"
        production_path.rename(target)
        production_path.symlink_to(target.name)
    else:
        target = workflow_root.with_name("efficiency-target")
        workflow_root.rename(target)
        workflow_root.symlink_to(target.name, target_is_directory=True)

    with pytest.raises(BenchmarkConfigError, match="must not be a symlink|missing or changed"):
        production_policy.completed_production_config(
            repo_root=root,
            work_root=work,
            budget="reduced",
            efficiency_run_id="efficiency-current",
        )
