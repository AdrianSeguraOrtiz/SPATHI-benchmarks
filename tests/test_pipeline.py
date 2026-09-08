from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.pipeline as pipeline
from spathi_bench.artifacts import Artifact, ArtifactKey
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.pipeline import (
    _analysis_bundle_spec,
    _artifact_store,
    _batch_selects_dataset,
    _counterbalanced_efficiency_development_tasks,
    _dataset_execution_run,
    _evaluation_spec,
    _execute_unique_missing_tasks,
    _generation_spec,
    _inference_spec,
    _integration_spec,
    _spathi_build_fingerprint,
    _validate_inference_state_contract,
    build_spathi_image,
)
from spathi_bench.resource_scheduler import ResourceCapacity, ResourceClaim

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKER_RUNTIME = {"fingerprint": "d" * 64}
INFERENCE_RESOURCES = {
    "cpu_threads": 1,
    "memory_bytes": 4 * 1024**3,
    "memory_gib": 4.0,
    "planner": "heuristic",
    "planner_time_limit_seconds": 30,
}
EXECUTION_PLAN = {
    "fingerprint": "x" * 64,
    "output_profile": "canonical",
    "exclusive_inference": False,
    "capacity": {
        "cpu_threads": 1,
        "memory_bytes": 4 * 1024**3,
        "memory_gib": 4.0,
    },
    "detected": {"allowed_cpus": [0], "cpu_allocation_order": [0]},
    "operations": {"inference": deepcopy(INFERENCE_RESOURCES)},
}


def test_efficiency_development_tasks_are_counterbalanced_by_dataset_and_seed() -> None:
    labels = ("adaptive", "combined", "eligibility", "reference")
    tasks = [
        {
            "dataset": {"dataset_id": dataset_id},
            "metadata": {
                "algorithm_seed": seed,
                "configuration_id": label,
                "efficiency_development_step": "joint_confirmation",
                "execution_order_seed": 99,
                "family": "spathi_efficiency_development",
            },
        }
        for dataset_id in ("dataset_a", "dataset_b", "dataset_c", "dataset_d", "dataset_e")
        for seed in (11, 12)
        for label in labels
    ]

    ordered = _counterbalanced_efficiency_development_tasks(list(reversed(tasks)))
    blocks = [ordered[index : index + len(labels)] for index in range(0, len(ordered), 4)]
    position_counts = {
        label: [
            sum(block[position]["metadata"]["configuration_id"] == label for block in blocks)
            for position in range(len(labels))
        ]
        for label in labels
    }

    assert all(
        len({(task["dataset"]["dataset_id"], task["metadata"]["algorithm_seed"]) for task in block})
        == 1
        for block in blocks
    )
    assert all(max(counts) - min(counts) <= 1 for counts in position_counts.values())
    for algorithm_seed in (11, 12):
        seed_blocks = [
            block for block in blocks if block[0]["metadata"]["algorithm_seed"] == algorithm_seed
        ]
        seed_position_counts = {
            label: [
                sum(
                    block[position]["metadata"]["configuration_id"] == label
                    for block in seed_blocks
                )
                for position in range(len(labels))
            ]
            for label in labels
        }
        assert all(
            max(counts) - min(counts) <= 1 for counts in seed_position_counts.values()
        )


def _physical_measurement() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scope": "physical_task",
        "wall_time_seconds": 12.5,
        "cpu_time_seconds": 10.0,
        "peak_memory_bytes": 1024,
        "output_bytes": 128,
        "io_read_bytes": None,
        "io_write_bytes": None,
        "task_started_at_utc": "2026-01-01T00:00:00Z",
        "task_finished_at_utc": "2026-01-01T00:00:13Z",
        "task_started_monotonic_ns": 1,
        "task_finished_monotonic_ns": 12_500_000_001,
        "container_started_at_utc": "2026-01-01T00:00:00Z",
        "container_finished_at_utc": "2026-01-01T00:00:13Z",
        "assigned_resources": {
            "threads": 1,
            "ram_limit_bytes": 4 * 1024**3,
            "cpuset_cpus_requested": None,
            "cpuset_cpus_configured": None,
            "cpuset_cpus_effective": None,
            "cpu_quota_cores_effective": 1.0,
            "ram_limit_bytes_effective": 4 * 1024**3,
        },
        "telemetry": {
            "status": "complete",
            "cpu_source": "test",
            "memory_source": "test",
            "memory_semantics": "container_peak",
            "io_source": None,
            "io_semantics": "unavailable",
            "throttled_time_seconds": 0.0,
            "oom_events": 0,
            "oom_killed": False,
        },
    }


def _execution_measurement() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scope": "andrea_execution",
        "wall_time_seconds": 13.0,
        "assigned_resources": {},
        "telemetry": {
            "wall_source": "andrea_monotonic_clock",
            "wall_semantics": (
                "andrea_execution_end_to_end_including_input_verification_preparation_"
                "scheduling_merging_and_requested_output_materialization"
            ),
        },
    }


@pytest.fixture(autouse=True)
def _stable_docker_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "docker_runtime_environment",
        lambda: deepcopy(DOCKER_RUNTIME),
    )


def _state() -> dict[str, Any]:
    return {
        "benchmark_repository": {"commit": "b" * 40},
        "repositories": {"observed": {"andrea": {"commit": "a" * 40, "path": str(REPO_ROOT)}}},
        "host_environment": {"fingerprint": "h" * 64},
        "benchmark_environment": {"fingerprint": "j" * 64},
        "docker_runtime_environment": deepcopy(DOCKER_RUNTIME),
        "execution_plan": deepcopy(EXECUTION_PLAN),
        "catalog_images": {"simulator:dyngen": {"locked_repo_digest": "dyngen@sha256:" + "d" * 64}},
    }


def test_preflight_revalidates_seed_contracts_before_resuming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resumed_state_was_read = False

    def stale_design(_repo_root: Path, _andrea_repo: Path) -> None:
        raise BenchmarkConfigError("Reproducibility audit failed: stale seed fixture")

    def existing_state(*_args: Any) -> dict[str, Any]:
        nonlocal resumed_state_was_read
        resumed_state_was_read = True
        return {"status": "preflighted"}

    monkeypatch.setattr(pipeline, "validate_all_profiles", lambda _root: None)
    monkeypatch.setattr(pipeline, "validate_environment_lock", lambda _root: None)
    monkeypatch.setattr(pipeline, "validate_design_contracts", stale_design)
    monkeypatch.setattr(pipeline, "_load_existing_state", existing_state)

    with pytest.raises(BenchmarkConfigError, match="stale seed fixture"):
        pipeline.preflight(
            repo_root=tmp_path,
            profile_id="smoke",
            run_root=tmp_path / "run",
            andrea_repo=tmp_path / "andrea",
            spathi_repo=tmp_path / "spathi",
        )
    assert not resumed_state_was_read


def test_docker_runtime_change_invalidates_a_preflighted_run() -> None:
    state = {"docker_runtime_environment": {"fingerprint": "a" * 64}}

    with pytest.raises(BenchmarkConfigError, match="Docker runtime.*changed"):
        pipeline._validate_docker_runtime(state, {"fingerprint": "b" * 64})


def test_preflight_writes_scale_stress_generation_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "work/v1/scalability-reduced/test-run"
    run_root.mkdir(parents=True)
    profile = {
        "id": "scalability-reduced",
        "purpose": "scalability",
        "generation_jobs": [
            {
                "id": "scale_stress_scalability_reduced",
                "generator_request": ("experiments/v1/simulations/scale_stress/scale-reduced.json"),
            }
        ],
        "execution_class": "scale_canonical",
    }
    monkeypatch.setattr(pipeline, "validate_all_profiles", lambda _root: None)
    monkeypatch.setattr(pipeline, "validate_design_contracts", lambda _root, _andrea: None)
    monkeypatch.setattr(pipeline, "_load_existing_state", lambda *_args: None)
    monkeypatch.setattr(pipeline, "freeze_run_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "frozen_repository_root", lambda _run: REPO_ROOT)
    monkeypatch.setattr(
        pipeline,
        "frozen_path",
        lambda _run, raw_path, _label: REPO_ROOT / raw_path,
    )
    monkeypatch.setattr(pipeline, "frozen_profile", lambda *_args: profile)
    monkeypatch.setattr(pipeline, "validate_profile_references", lambda *_args: None)
    monkeypatch.setattr(
        pipeline,
        "load_repo_lock",
        lambda _root: {
            "repositories": {
                "andrea": {"commit": "a" * 40},
                "spathi": {"commit": "b" * 40},
            }
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_andrea_repository",
        lambda *_args: {"commit": "a" * 40},
    )
    monkeypatch.setattr(
        pipeline,
        "validate_locked_repository",
        lambda **_kwargs: {"commit": "b" * 40},
    )
    monkeypatch.setattr(pipeline, "check_docker", lambda: None)
    monkeypatch.setattr(pipeline, "_validate_spathi_image_contract", lambda _root: {})
    monkeypatch.setattr(pipeline, "_pin_profile_catalog_images", lambda **_kwargs: {})
    monkeypatch.setattr(pipeline, "andrea_host_environment", lambda: {"host": "test"})
    monkeypatch.setattr(
        pipeline,
        "benchmark_host_environment",
        lambda: {"fingerprint": "f" * 64},
    )
    monkeypatch.setattr(
        pipeline,
        "_initial_state",
        lambda **_kwargs: {"schema_version": "1.0", "status": "preflighted"},
    )
    monkeypatch.setattr(pipeline, "write_state", lambda *_args: None)
    runtime_plan = {"fingerprint": "r" * 64}
    execution_plan = {
        "operations": {
            "generate_data": {
                "max_parallel_tasks": 1,
                "cpu_threads": 1,
                "memory_gib": 1,
            }
        }
    }
    monkeypatch.setattr(pipeline, "_prepare_runtime_plan", lambda **_kwargs: runtime_plan)
    monkeypatch.setattr(
        pipeline,
        "resolve_execution_class",
        lambda **_kwargs: execution_plan,
    )

    report = pipeline.preflight(
        repo_root=REPO_ROOT,
        profile_id="scalability-reduced",
        run_root=run_root,
        andrea_repo=tmp_path / "andrea",
        spathi_repo=tmp_path / "spathi",
    )

    assert report.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))["docker_runtime_environment"] == (
        DOCKER_RUNTIME
    )
    plan = json.loads(
        (run_root / "preflight/scale_stress_scalability_reduced-generation-plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["generator"] == "spathi_scale_stress"
    assert plan["expression_preparation"] == {
        "input_scale": "nonnegative_synthetic_counts_like",
        "normalization": "library-size-log1p",
        "target_sum": 10000.0,
        "library_size_scope": "all_generated_genes_before_any_filtering",
        "formula": "log1p(target_sum * raw_value / per_cell_library_size)",
    }
    assert [dataset["dataset_id"] for dataset in plan["datasets"]] == [
        "scale_stress_reduced_baseline",
        "scale_stress_reduced_moderate",
    ]
    assert plan["datasets"][0] == {
        "dataset_id": "scale_stress_reduced_baseline",
        "design_role": {
            "kind": "synthetic_scale_case",
            "case_id": "reduced_baseline",
            "purpose": "Exercise a rounded multi-group baseline with modest development cost.",
        },
        "seed": 887098426,
        "cells": 1000,
        "genes": 5000,
        "tfs": 250,
        "groups": 3,
    }


def test_inference_batch_without_generation_jobs_selects_entire_role() -> None:
    batch = {"dataset_role": "development"}

    assert _batch_selects_dataset(
        batch, {"dataset_role": "development", "generation_job": "dyngen"}
    )
    assert _batch_selects_dataset(
        batch, {"dataset_role": "development", "generation_job": "scmultisim"}
    )
    assert not _batch_selects_dataset(
        batch, {"dataset_role": "validation", "generation_job": "dyngen"}
    )


def test_inference_batch_generation_jobs_filter_datasets_within_role() -> None:
    batch = {"dataset_role": "development", "generation_jobs": ["dyngen"]}

    assert _batch_selects_dataset(
        batch, {"dataset_role": "development", "generation_job": "dyngen"}
    )
    assert not _batch_selects_dataset(
        batch, {"dataset_role": "development", "generation_job": "scmultisim"}
    )
    assert not _batch_selects_dataset(
        batch, {"dataset_role": "validation", "generation_job": "dyngen"}
    )


def test_inference_batch_replicate_indices_are_one_based_and_job_scoped() -> None:
    batch = {
        "dataset_role": "development",
        "generation_jobs": ["dyngen", "scmultisim"],
        "replicate_indices": {"dyngen": [1, 3], "scmultisim": [2]},
    }

    assert _batch_selects_dataset(
        batch,
        {
            "dataset_role": "development",
            "generation_job": "dyngen",
            "replicate_index": 1,
        },
    )
    assert not _batch_selects_dataset(
        batch,
        {
            "dataset_role": "development",
            "generation_job": "dyngen",
            "replicate_index": 2,
        },
    )
    assert _batch_selects_dataset(
        batch,
        {
            "dataset_role": "development",
            "generation_job": "scmultisim",
            "replicate_index": 2,
        },
    )


def test_hash_target_scope_is_bound_to_dataset_without_mutating_logical_run() -> None:
    run = {
        "run_id": "high_tf",
        "params": {
            "random_seed": 17,
            "target_scope": {
                "method": "sha256_gene_id",
                "size": 256,
                "salt": "spathi-calibration-high-tf-v1",
            },
        },
    }

    resolved = _dataset_execution_run(run, {"dataset_id": "dataset_a"})

    assert resolved["params"]["target_scope"]["dataset_id"] == "dataset_a"
    assert "dataset_id" not in run["params"]["target_scope"]


def test_parameter_hashes_preserve_declared_logical_calibration_identity() -> None:
    run = {
        "run_id": "geometry__base__seed_17",
        "params": {"group_policy": {"branches": {}}, "random_seed": 17},
    }
    metadata = {
        "algorithm_seed": 17,
        "algorithm_seed_parameter": "random_seed",
        "configuration_sha256": "a" * 64,
    }

    configuration_sha256, execution_sha256 = pipeline._parameter_hashes(run, metadata)

    assert configuration_sha256 == "a" * 64
    assert execution_sha256 == pipeline.canonical_json_sha256(run["params"])


def _inference_attribution_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path]:
    run_root = tmp_path / "work/v1/smoke/run"
    params_path = tmp_path / "tools-params.json"
    run = {
        "run_id": "genie3_seed_17",
        "tool_id": "genie3",
        "params": {"random_seed": 17},
        "execution": {"mode": "group_emulated"},
        "resources": {"threads": 1, "ram_gb": 4.0, "cpuset_cpus": [0]},
    }
    params_path.write_text(json.dumps({"runs": [run]}), encoding="utf-8")
    metadata = {
        "run_id": run["run_id"],
        "configuration_id": "genie3_default",
        "selection_role": "baseline",
        "algorithm_seed": 17,
        "algorithm_seed_parameter": "random_seed",
        "family": "genie3",
    }
    batch = {
        "id": "group_only_baseline",
        "kind": "baseline",
        "dataset_role": "smoke",
        "regime": "group_only",
        "tools_params": "tools-params.json",
        "run_metadata": "run-metadata.json",
    }
    profile = {
        "generation_jobs": [],
        "inference_batches": [batch],
        "execution_class": "bulk_canonical",
    }
    dataset = {
        "dataset_id": "dataset_a",
        "dataset_role": "smoke",
        "generation_job": "dyngen_smoke",
        "dataset_artifact": {
            "kind": "dataset",
            "digest": "d" * 64,
            "key": "sha256:" + "d" * 64,
            "path": str(tmp_path / "unused-dataset"),
            "cache_hit": False,
        },
    }
    state = {
        **_state(),
        "status": "inferred",
        "datasets": [dataset],
        "cache": {"dataset_hits": 0, "inference_hits": 0},
        "inference_runs": [],
    }
    tool_image = {"kind": "catalog", "repo_digest": "genie3@sha256:" + "e" * 64}
    resources = deepcopy(INFERENCE_RESOURCES)
    spec = _inference_spec(
        state=state,
        dataset=dataset,
        batch=batch,
        run=run,
        metadata=metadata,
        custom_definition=None,
        tool_image=tool_image,
        resources=resources,
    )
    store = _artifact_store(run_root)
    key = store.key_for("inference", spec)
    fingerprint = {"algorithm": "sha256", "value": "f" * 64}
    with store.staging_directory(key) as staging:
        preflight_path = staging / "request/preflight.json"
        preflight_path.parent.mkdir(parents=True)
        preflight_path.write_text(
            json.dumps(
                {
                    "inputs": {"dataset_id": dataset["dataset_id"]},
                    "dataset": {"dataset_id": dataset["dataset_id"]},
                    "runs": {
                        "requested_total": 1,
                        "selected": [run["run_id"]],
                        "skipped": {},
                        "tool_origins": {run["run_id"]: "catalog"},
                        "issues": {run["run_id"]: []},
                    },
                    "issues": [],
                }
            ),
            encoding="utf-8",
        )
        report_path = staging / "output/execution/run_report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(
            json.dumps(
                {
                    "run_id": run["run_id"],
                    "status": "executed",
                    "dataset": {"id": dataset["dataset_id"], "fingerprint": fingerprint},
                    "tools": {
                        "selected": [run["run_id"]],
                        "status_by_tool": {run["run_id"]: "completed"},
                        "failed": [],
                        "tool_origins": {run["run_id"]: "catalog"},
                    },
                }
            ),
            encoding="utf-8",
        )
        artifact = store.publish_from_temp(staging, kind="inference", spec=spec)
    published_report = artifact.file("output/execution/run_report.json").resolve()
    configuration_sha256, execution_sha256 = pipeline._parameter_hashes(run, metadata)
    state["inference_runs"] = [
        {
            "dataset_id": dataset["dataset_id"],
            "batch_id": batch["id"],
            "batch_kind": batch["kind"],
            "regime": batch["regime"],
            "run_id": run["run_id"],
            "tool_id": run["tool_id"],
            "tool_origin": "catalog",
            "configuration_id": metadata["configuration_id"],
            "selection_role": metadata["selection_role"],
            "algorithm_seed": metadata["algorithm_seed"],
            "algorithm_seed_parameter": metadata["algorithm_seed_parameter"],
            "family": metadata["family"],
            "configuration_sha256": configuration_sha256,
            "execution_sha256": execution_sha256,
            "dataset_fingerprint": fingerprint,
            "tool_image": tool_image,
            "resources": run["resources"],
            "run_report": str(published_report),
            "artifact": pipeline._artifact_ref(artifact, cache_hit=False),
        }
    ]
    monkeypatch.setattr(pipeline, "frozen_repository_root", lambda _run_root: tmp_path)
    monkeypatch.setattr(pipeline, "frozen_profile", lambda _run_root, _profile_id: profile)
    monkeypatch.setattr(pipeline, "frozen_path", lambda _run_root, _raw, _label: params_path)
    monkeypatch.setattr(
        pipeline,
        "load_run_metadata",
        lambda _root, _path, _label: {"schema_version": "1.0", "runs": [metadata]},
    )
    monkeypatch.setattr(pipeline, "_runtime_image_contract", lambda **_kwargs: tool_image)
    monkeypatch.setattr(
        pipeline,
        "_inference_toolspec",
        lambda **_kwargs: {
            "id": "genie3",
            "runtime_resources": {"threading": {"supported": True, "max_threads": None}},
        },
    )
    monkeypatch.setattr(pipeline, "_validate_generated_state_contract", lambda **_kwargs: None)
    return state, run_root


def _single_run_preflight_payload() -> dict[str, Any]:
    return {
        "inputs": {"dataset_id": "dataset_a"},
        "dataset": {"dataset_id": "dataset_a"},
        "runs": {
            "requested_total": 1,
            "selected": ["genie3_seed_17"],
            "skipped": {},
            "tool_origins": {"genie3_seed_17": "catalog"},
            "issues": {"genie3_seed_17": []},
        },
        "issues": [],
    }


def test_single_run_preflight_requires_exact_dataset_and_run(tmp_path: Path) -> None:
    report_path = tmp_path / "preflight.json"
    report_path.write_text(json.dumps(_single_run_preflight_payload()), encoding="utf-8")

    report = pipeline._validate_single_run_preflight(
        report_path,
        run_id="genie3_seed_17",
        dataset_id="dataset_a",
        expected_origin="catalog",
    )

    assert report["runs"]["selected"] == ["genie3_seed_17"]


def test_failed_inference_diagnostics_are_published_as_separate_immutable_evidence(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "work/v1/calibration-reduced/run"
    run_root.mkdir(parents=True)
    command_log = run_root / "logs/20-inference.log"
    command_log.parent.mkdir()
    command_log.write_text("ANDREA command failed\n", encoding="utf-8")

    attempt = run_root / "attempt/output/dataset_run"
    tool_root = attempt / "tools/spathi_run"
    adapter_root = tool_root / "io/out/failure_diagnostics"
    adapter_root.mkdir(parents=True)
    container_log = tool_root / "container.log"
    container_log.write_text("SPATHI adapter failed\n", encoding="utf-8")
    progress = tool_root / "io/out/progress.json"
    progress.write_text(
        json.dumps(
            {
                "percent": 100,
                "status": "failed",
                "phase": "failed",
                "message": "invalid feature importances",
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    diagnostics = adapter_root / "model_diagnostics.tsv.gz"
    diagnostics.write_bytes(b"diagnostic evidence")
    adapter_manifest = {
        "schema_version": "1.0",
        "artifact_type": "spathi_adapter_failure_diagnostics",
        "status": "failed",
        "error": {"type": "RuntimeError", "message": "79 models failed"},
        "files": [
            {
                "path": diagnostics.name,
                "sha256": pipeline.sha256_file(diagnostics),
                "size": diagnostics.stat().st_size,
            }
        ],
        "model_diagnostics": {"status_counts": {"invalid_feature_importances": 79}},
    }
    (adapter_root / "failure.json").write_text(
        json.dumps(adapter_manifest),
        encoding="utf-8",
    )
    report_path = attempt / "run_report.json"
    report_path.write_text(
        json.dumps(
            {
                "tools": {
                    "results": {
                        "spathi_run": {
                            "logs_path": str(container_log.resolve()),
                            "progress_path": str(progress.resolve()),
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    inference_spec = {"dataset": "dataset_a", "run": "spathi_run"}
    task = {
        "key": ArtifactKey.from_spec("inference", inference_spec),
        "spec": inference_spec,
    }
    failure = pipeline.InferenceExecutionFailure(
        dataset_id="dataset_a",
        batch_id="spathi",
        run_id="spathi_run",
        orchestrator_return_code=1,
        tool_exit_code=2,
        tool_error="79 models failed",
        measurement=_physical_measurement(),
        execution_measurement=_execution_measurement(),
        command_log_path=command_log,
        progress={
            "percent": 100,
            "status": "failed",
            "phase": "failed",
            "message": "invalid feature importances",
            "warnings": [],
        },
        tool_log_tail="SPATHI adapter failed",
    )

    artifact = pipeline._publish_inference_failure_evidence(
        store=_artifact_store(run_root),
        task=task,
        run_root=run_root,
        report_path=report_path,
        failure=failure,
    )

    assert artifact.key.kind == "inference-failure"
    assert artifact.manifest["status"] == "complete"
    failure_manifest = json.loads(artifact.file("failure.json").read_text(encoding="utf-8"))
    assert failure_manifest["inference_status"] == "failed"
    assert failure_manifest["attribution"] == {
        "dataset_id": "dataset_a",
        "batch_id": "spathi",
        "run_id": "spathi_run",
    }
    assert (
        artifact.file("tool/io/out/failure_diagnostics/model_diagnostics.tsv.gz").read_bytes()
        == b"diagnostic evidence"
    )
    assert not any(path.name == "network.csv" for path in artifact.path.rglob("*"))
    bound = failure.with_evidence(artifact)
    assert bound.evidence_artifact == artifact
    assert str(artifact.file("failure.json").resolve()) in str(bound)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dataset", "wrong dataset identity"),
        ("skipped", "skipped requested run"),
        ("origin", "wrong origin"),
        ("run_block", "blocked 'genie3_seed_17'"),
        ("global_block", "blocking dataset issues"),
    ],
)
def test_single_run_preflight_rejects_ineligible_dataset_routes(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    payload = _single_run_preflight_payload()
    if mutation == "dataset":
        payload["dataset"]["dataset_id"] = "dataset_b"
    elif mutation == "skipped":
        payload["runs"]["skipped"] = {"genie3_seed_17": "incompatible"}
    elif mutation == "origin":
        payload["runs"]["tool_origins"]["genie3_seed_17"] = "custom"
    elif mutation == "run_block":
        payload["runs"]["issues"]["genie3_seed_17"] = [
            {"severity": "block", "code": "compatibility", "message": "incompatible"}
        ]
    else:
        payload["issues"] = [{"severity": "block", "code": "dataset", "message": "invalid dataset"}]
    report_path = tmp_path / "preflight.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match=message):
        pipeline._validate_single_run_preflight(
            report_path,
            run_id="genie3_seed_17",
            dataset_id="dataset_a",
            expected_origin="catalog",
        )


def test_inference_state_reconciles_frozen_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, run_root = _inference_attribution_case(tmp_path, monkeypatch)

    verified = _validate_inference_state_contract(
        state=state,
        profile_id="smoke",
        run_root=run_root,
    )

    assert set(verified) == {("dataset_a", "group_only_baseline", "genie3_seed_17")}


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("configuration_id", "another_configuration"),
        ("configuration_sha256", "0" * 64),
        ("algorithm_seed", 18),
        ("selection_role", "candidate"),
    ],
)
def test_inference_state_rejects_mutated_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: Any,
) -> None:
    state, run_root = _inference_attribution_case(tmp_path, monkeypatch)
    state["inference_runs"][0][field] = replacement

    with pytest.raises(BenchmarkConfigError, match="attribution differs from frozen inputs"):
        _validate_inference_state_contract(
            state=state,
            profile_id="smoke",
            run_root=run_root,
        )


def test_inference_state_rejects_mutated_artifact_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, run_root = _inference_attribution_case(tmp_path, monkeypatch)
    state["inference_runs"][0]["artifact"]["digest"] = "0" * 64

    with pytest.raises(BenchmarkConfigError, match="differs from its digest"):
        _validate_inference_state_contract(
            state=state,
            profile_id="smoke",
            run_root=run_root,
        )


def test_inference_state_rejects_mutated_evaluation_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state, run_root = _inference_attribution_case(tmp_path, monkeypatch)
    inference = state["inference_runs"][0]
    dataset = state["datasets"][0]
    spec = _evaluation_spec(state=state, dataset=dataset, inference=inference)
    store = _artifact_store(run_root)
    key = store.key_for("evaluation", spec)
    with store.staging_directory(key) as staging:
        report_path = staging / "output/evaluation/evaluation_report.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text('{"status":"complete"}\n', encoding="utf-8")
        artifact = store.publish_from_temp(staging, kind="evaluation", spec=spec)
    state["status"] = "evaluated"
    state["cache"]["evaluation_hits"] = 0
    state["evaluations"] = [
        {
            **{field: inference[field] for field in pipeline._INFERENCE_ATTRIBUTION_FIELDS},
            "run_report": inference["run_report"],
            "evaluation_report": str(
                artifact.file("output/evaluation/evaluation_report.json").resolve()
            ),
            "inference_artifact": inference["artifact"],
            "artifact": pipeline._artifact_ref(artifact, cache_hit=False),
        }
    ]
    state["evaluations"][0]["configuration_id"] = "another_configuration"

    with pytest.raises(BenchmarkConfigError, match="attribution differs from inference state"):
        _validate_inference_state_contract(
            state=state,
            profile_id="smoke",
            run_root=run_root,
        )


def test_inference_key_ignores_classification_regime() -> None:
    state = _state()
    dataset = {
        "dataset_id": "dataset_a",
        "dataset_artifact": {"key": "sha256:" + "d" * 64},
    }
    run = {
        "run_id": "spathi_selected__seed_1729",
        "tool_id": "custom_spathi_selected__seed_1729",
        "params": {"random_seed": 1729},
        "execution": {"mode": "group_native"},
    }
    metadata = {
        "run_id": run["run_id"],
        "configuration_id": "spathi_selected",
        "selection_role": "selected",
        "algorithm_seed": 1729,
        "algorithm_seed_parameter": "random_seed",
    }
    custom_definition = {
        "run_id": run["run_id"],
        "docker_image": "sha256:" + "e" * 64,
    }
    tool_image = {"kind": "spathi", "image_id": "sha256:" + "e" * 64}
    resources = {
        "max_cores": 2,
        "max_ram_gb": 8,
        "planner": "heuristic",
        "planner_time_limit_seconds": 30,
    }

    group_only = _inference_spec(
        state=state,
        dataset=dataset,
        batch={"kind": "spathi", "regime": "group_only"},
        run=run,
        metadata=metadata,
        custom_definition=custom_definition,
        tool_image=tool_image,
        resources=resources,
    )
    order_aware = _inference_spec(
        state=state,
        dataset=dataset,
        batch={"kind": "spathi", "regime": "order_aware"},
        run=run,
        metadata=metadata,
        custom_definition=custom_definition,
        tool_image=tool_image,
        resources=resources,
    )

    assert group_only == order_aware
    assert ArtifactKey.from_spec("inference", group_only) == ArtifactKey.from_spec(
        "inference", order_aware
    )


def test_inference_spec_and_cache_key_bind_explicit_run_ram() -> None:
    state = _state()
    dataset = {
        "dataset_id": "dataset_a",
        "dataset_artifact": {"key": "sha256:" + "d" * 64},
    }
    metadata = {
        "run_id": "genie3_seed_17",
        "configuration_id": "genie3_default",
        "selection_role": "baseline",
        "algorithm_seed": 17,
        "algorithm_seed_parameter": "random_seed",
    }
    base_run = {
        "run_id": metadata["run_id"],
        "tool_id": "genie3",
        "params": {"random_seed": 17},
        "execution": {"mode": "group_emulated"},
        "resources": {"threads": 1, "ram_gb": 4.0},
    }
    common = {
        "state": state,
        "dataset": dataset,
        "batch": {"kind": "baseline", "regime": "group_only"},
        "metadata": metadata,
        "custom_definition": None,
        "tool_image": {"kind": "catalog", "repo_digest": "genie3@sha256:" + "e" * 64},
        "resources": deepcopy(INFERENCE_RESOURCES),
    }
    low_ram = _inference_spec(run=base_run, **common)
    high_ram_run = deepcopy(base_run)
    high_ram_run["resources"]["ram_gb"] = 8.0
    high_ram = _inference_spec(run=high_ram_run, **common)

    assert low_ram["run"]["resources"]["ram_gb"] == 4.0
    assert high_ram["run"]["resources"]["ram_gb"] == 8.0
    assert low_ram != high_ram
    assert ArtifactKey.from_spec("inference", low_ram) != ArtifactKey.from_spec(
        "inference", high_ram
    )


def test_missing_tasks_execute_once_per_key_and_reuse_the_artifact(tmp_path: Path) -> None:
    first_key = ArtifactKey.from_spec("inference", {"request": "first"})
    second_key = ArtifactKey.from_spec("inference", {"request": "second"})
    first_spec = {"request": "first"}
    prepared = [
        {"key": first_key, "spec": first_spec, "artifact": None, "cache_hit": False},
        {
            "key": first_key,
            "spec": deepcopy(first_spec),
            "artifact": None,
            "cache_hit": False,
        },
        {
            "key": second_key,
            "spec": {"request": "second"},
            "artifact": None,
            "cache_hit": False,
        },
    ]
    artifacts = {
        key: Artifact(key=key, path=tmp_path / key.digest, manifest={})
        for key in (first_key, second_key)
    }
    calls: list[ArtifactKey] = []

    def execute(task: dict[str, Any]) -> Artifact:
        calls.append(task["key"])
        return artifacts[task["key"]]

    reused = _execute_unique_missing_tasks(
        prepared,
        capacity=ResourceCapacity(cpu_threads=2, memory_bytes=2),
        claim=lambda _task: ResourceClaim(cpu_threads=1, memory_bytes=1),
        thread_name_prefix="test-deduplication",
        schedule_path=tmp_path / "schedule.json",
        run_root=tmp_path,
        execute=execute,
    )

    assert sorted(calls, key=lambda key: key.value) == sorted(
        (first_key, second_key), key=lambda key: key.value
    )
    assert reused == (1,)
    assert prepared[0]["artifact"] is artifacts[first_key]
    assert prepared[1]["artifact"] is artifacts[first_key]
    assert prepared[2]["artifact"] is artifacts[second_key]
    assert [task["cache_hit"] for task in prepared] == [False, True, False]


def test_missing_tasks_reject_same_key_with_different_specs(tmp_path: Path) -> None:
    key = ArtifactKey.from_spec("inference", {"request": "first"})
    prepared = [
        {"key": key, "spec": {"request": "first"}, "artifact": None},
        {"key": key, "spec": {"request": "different"}, "artifact": None},
    ]

    with pytest.raises(BenchmarkConfigError, match="different execution specifications"):
        _execute_unique_missing_tasks(
            prepared,
            capacity=ResourceCapacity(cpu_threads=1, memory_bytes=1),
            claim=lambda _task: ResourceClaim(cpu_threads=1, memory_bytes=1),
            thread_name_prefix="test-deduplication",
            schedule_path=tmp_path / "schedule.json",
            run_root=tmp_path,
            execute=lambda _task: pytest.fail("inconsistent tasks must not execute"),
        )


def test_missing_tasks_reject_executor_artifact_with_another_key(tmp_path: Path) -> None:
    requested = ArtifactKey.from_spec("inference", {"request": "expected"})
    returned = ArtifactKey.from_spec("inference", {"request": "wrong"})
    prepared = [
        {
            "key": requested,
            "spec": {"request": "expected"},
            "artifact": None,
            "cache_hit": False,
        }
    ]

    with pytest.raises(BenchmarkConfigError, match="for requested key"):
        _execute_unique_missing_tasks(
            prepared,
            capacity=ResourceCapacity(cpu_threads=1, memory_bytes=1),
            claim=lambda _task: ResourceClaim(cpu_threads=1, memory_bytes=1),
            thread_name_prefix="test-deduplication",
            schedule_path=tmp_path / "schedule.json",
            run_root=tmp_path,
            execute=lambda _task: Artifact(
                key=returned,
                path=tmp_path / returned.digest,
                manifest={},
            ),
        )


def test_generation_key_does_not_depend_on_spathi(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario.json"
    runs = tmp_path / "runs.json"
    scenario.write_text('{"schema_version":"1.0","id":"scenario"}\n', encoding="utf-8")
    runs.write_text('{"runs":[{"simulator_id":"dyngen"}]}\n', encoding="utf-8")
    state = _state()
    job = {
        "id": "development",
        "dataset_role": "development",
        "split_id": "split_a",
        "condition_id": "condition_a",
    }
    resources = {"max_parallel_tasks": 1, "max_cores": 2, "max_ram_gb": 8}

    before = _generation_spec(
        state=state,
        job=job,
        scenario=scenario,
        simulator_runs=runs,
        input_root=tmp_path,
        resources=resources,
    )
    state["spathi_image"] = {"source_commit": "1" * 40}
    after = _generation_spec(
        state=state,
        job=job,
        scenario=scenario,
        simulator_runs=runs,
        input_root=tmp_path,
        resources=resources,
    )

    assert before == after


def test_generation_key_binds_every_external_scenario_input(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    scenario_dir = root / "experiments/scenario"
    scenario_dir.mkdir(parents=True)
    network = scenario_dir / "network.tsv"
    tree = scenario_dir / "tree.nwk"
    network.write_text("target\tregulator\nG2\tG1\n", encoding="utf-8")
    tree.write_text("(A,B);\n", encoding="utf-8")
    scenario = scenario_dir / "scenario.json"
    scenario.write_text(
        json.dumps(
            {
                "inputs": {
                    "regulatory_network": {"path": "network.tsv"},
                    "tree_newick": {"path": "tree.nwk"},
                }
            }
        ),
        encoding="utf-8",
    )
    runs = scenario_dir / "runs.json"
    runs.write_text('{"runs":[{"simulator_id":"dyngen"}]}\n', encoding="utf-8")
    arguments = {
        "state": _state(),
        "job": {
            "id": "development",
            "dataset_role": "development",
            "split_id": "split_a",
            "condition_id": "condition_a",
        },
        "scenario": scenario,
        "simulator_runs": runs,
        "input_root": root,
        "resources": {"max_parallel_tasks": 1, "max_cores": 2, "max_ram_gb": 8},
    }

    before = _generation_spec(**arguments)
    before_key = ArtifactKey.from_spec("dataset", before)
    network.write_text("target\tregulator\nG3\tG1\n", encoding="utf-8")
    after = _generation_spec(**arguments)

    assert before["scenario_sha256"] == after["scenario_sha256"]
    assert before["scenario_external_inputs"] != after["scenario_external_inputs"]
    assert ArtifactKey.from_spec("dataset", after) != before_key


def test_scale_stress_generation_key_binds_seed_plan_and_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spathi_bench import scale_stress

    root = tmp_path / "repository"
    request = root / "request.json"
    seed_plan = root / "seed-plan.json"
    root.mkdir()
    request.write_text('{"request":"fixed"}\n', encoding="utf-8")
    seed_plan.write_text('{"master_seed":1}\n', encoding="utf-8")
    monkeypatch.setattr(
        scale_stress,
        "validate_scale_stress_request",
        lambda _root, _request: {
            "seed_plan": "seed-plan.json",
            "scale_grid": {
                "provenance": "author_defined_public_synthetic_grid",
                "cells": [1000],
                "genes": [5000],
                "transcription_factors": [250],
            },
        },
    )
    arguments = {
        "state": _state(),
        "job": {
            "id": "scale_stress",
            "dataset_role": "scalability",
            "split_id": "scalability_v1",
            "condition_id": "scale_grid",
        },
        "request": request,
        "input_root": root,
        "resources": {"max_parallel_tasks": 1, "max_cores": 2, "max_ram_gb": 8},
    }

    before = pipeline._scale_stress_generation_spec(**arguments)
    before_key = ArtifactKey.from_spec("dataset", before)
    seed_plan.write_text('{"master_seed":2}\n', encoding="utf-8")
    after_seed_change = pipeline._scale_stress_generation_spec(**arguments)
    request.write_text('{"request":"changed"}\n', encoding="utf-8")
    after_request_change = pipeline._scale_stress_generation_spec(**arguments)

    assert before["request_sha256"] == after_seed_change["request_sha256"]
    assert before["request_dependencies"] != after_seed_change["request_dependencies"]
    assert ArtifactKey.from_spec("dataset", after_seed_change) != before_key
    assert after_seed_change["request_sha256"] != after_request_change["request_sha256"]
    assert ArtifactKey.from_spec("dataset", after_request_change) != ArtifactKey.from_spec(
        "dataset", after_seed_change
    )


@pytest.mark.parametrize("unsafe_kind", ("symlink", "escape"))
def test_generation_spec_rejects_unsafe_external_scenario_input(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    root = tmp_path / "repository"
    scenario_dir = root / "experiments/scenario"
    scenario_dir.mkdir(parents=True)
    outside = tmp_path / "outside.tsv"
    outside.write_text("target\tregulator\nG2\tG1\n", encoding="utf-8")
    if unsafe_kind == "symlink":
        (scenario_dir / "network.tsv").symlink_to(outside)
        declared_path = "network.tsv"
        error = "symlink"
    else:
        declared_path = "../../../outside.tsv"
        error = "escapes"
    scenario = scenario_dir / "scenario.json"
    scenario.write_text(
        json.dumps({"inputs": {"regulatory_network": {"path": declared_path}}}),
        encoding="utf-8",
    )
    runs = scenario_dir / "runs.json"
    runs.write_text('{"runs":[{"simulator_id":"dyngen"}]}\n', encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match=error):
        _generation_spec(
            state=_state(),
            job={
                "id": "development",
                "dataset_role": "development",
                "split_id": "split_a",
                "condition_id": "condition_a",
            },
            scenario=scenario,
            simulator_runs=runs,
            input_root=root,
            resources={"max_parallel_tasks": 1, "max_cores": 2, "max_ram_gb": 8},
        )


def test_spathi_change_invalidates_only_spathi_inference_and_evaluation() -> None:
    state = _state()
    dataset = {
        "dataset_id": "dataset_a",
        "dataset_artifact": {"key": "sha256:" + "d" * 64},
    }
    resources = {
        "max_cores": 2,
        "max_ram_gb": 8,
        "planner": "heuristic",
        "planner_time_limit_seconds": 30,
    }
    baseline_batch = {"kind": "baseline", "regime": "group_only"}
    baseline_run = {
        "run_id": "genie3",
        "tool_id": "genie3",
        "params": {"seed": 1},
        "execution": {"mode": "group_emulated"},
    }
    baseline_metadata = {
        "run_id": "genie3",
        "configuration_id": "genie3",
        "selection_role": "baseline",
        "algorithm_seed": 1,
        "algorithm_seed_parameter": "seed",
    }
    baseline_image = {"kind": "catalog", "repo_digest": "genie3@sha256:" + "e" * 64}
    baseline_spec = _inference_spec(
        state=state,
        dataset=dataset,
        batch=baseline_batch,
        run=baseline_run,
        metadata=baseline_metadata,
        custom_definition=None,
        tool_image=baseline_image,
        resources=resources,
    )

    spathi_batch = {"kind": "spathi", "regime": "group_only"}
    spathi_run = {
        "run_id": "spathi_candidate",
        "tool_id": "custom_spathi_candidate",
        "params": {"n_estimators": 100, "random_seed": 1},
        "execution": {"mode": "group_native"},
    }
    spathi_metadata = {
        "run_id": "spathi_candidate",
        "configuration_id": "spathi_candidate",
        "selection_role": "candidate",
        "algorithm_seed": 1,
        "algorithm_seed_parameter": "random_seed",
    }
    custom_definition = {
        "run_id": "spathi_candidate",
        "docker_image": "spathi:locked",
    }
    old_spathi_image = {"kind": "spathi", "source_commit": "1" * 40}
    new_spathi_image = {"kind": "spathi", "source_commit": "2" * 40}
    old_spathi_spec = _inference_spec(
        state=state,
        dataset=dataset,
        batch=spathi_batch,
        run=spathi_run,
        metadata=spathi_metadata,
        custom_definition=custom_definition,
        tool_image=old_spathi_image,
        resources=resources,
    )
    new_spathi_spec = _inference_spec(
        state=state,
        dataset=dataset,
        batch=spathi_batch,
        run=spathi_run,
        metadata=spathi_metadata,
        custom_definition=custom_definition,
        tool_image=new_spathi_image,
        resources=resources,
    )

    baseline_key = ArtifactKey.from_spec("inference", baseline_spec)
    assert baseline_key == ArtifactKey.from_spec("inference", deepcopy(baseline_spec))
    old_spathi_key = ArtifactKey.from_spec("inference", old_spathi_spec)
    new_spathi_key = ArtifactKey.from_spec("inference", new_spathi_spec)
    assert old_spathi_key != new_spathi_key
    changed_environment = deepcopy(state)
    changed_environment["benchmark_environment"]["fingerprint"] = "k" * 64
    changed_environment_spec = _inference_spec(
        state=changed_environment,
        dataset=dataset,
        batch=baseline_batch,
        run=baseline_run,
        metadata=baseline_metadata,
        custom_definition=None,
        tool_image=baseline_image,
        resources=resources,
    )
    assert baseline_key != ArtifactKey.from_spec("inference", changed_environment_spec)

    changed_docker = deepcopy(state)
    changed_docker["docker_runtime_environment"]["fingerprint"] = "e" * 64
    changed_docker_spec = _inference_spec(
        state=changed_docker,
        dataset=dataset,
        batch=baseline_batch,
        run=baseline_run,
        metadata=baseline_metadata,
        custom_definition=None,
        tool_image=baseline_image,
        resources=resources,
    )
    assert baseline_key != ArtifactKey.from_spec("inference", changed_docker_spec)

    baseline_inference = {
        "artifact": {"key": baseline_key.value},
        "dataset_fingerprint": {"algorithm": "sha256", "value": "f" * 64},
    }
    old_spathi_inference = {
        "artifact": {"key": old_spathi_key.value},
        "dataset_fingerprint": baseline_inference["dataset_fingerprint"],
    }
    new_spathi_inference = {
        "artifact": {"key": new_spathi_key.value},
        "dataset_fingerprint": baseline_inference["dataset_fingerprint"],
    }
    baseline_evaluation = ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=baseline_inference),
    )
    assert baseline_evaluation == ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=deepcopy(baseline_inference)),
    )
    assert ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=old_spathi_inference),
    ) != ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=new_spathi_inference),
    )

    baseline_for_joint = {
        **baseline_inference,
        "run_id": "genie3",
    }
    old_spathi_for_joint = {
        **old_spathi_inference,
        "run_id": "spathi_candidate",
    }
    new_spathi_for_joint = {
        **new_spathi_inference,
        "run_id": "spathi_candidate",
    }
    old_joint_spec = _integration_spec(
        state=state,
        dataset=dataset,
        regime="group_only",
        inference_runs=[baseline_for_joint, old_spathi_for_joint],
    )
    new_joint_spec = _integration_spec(
        state=state,
        dataset=dataset,
        regime="group_only",
        inference_runs=[baseline_for_joint, new_spathi_for_joint],
    )
    old_joint_key = ArtifactKey.from_spec("joint-inference", old_joint_spec)
    new_joint_key = ArtifactKey.from_spec("joint-inference", new_joint_spec)
    assert old_joint_key != new_joint_key

    old_joint = {
        "artifact": {"key": old_joint_key.value},
        "dataset_fingerprint": baseline_inference["dataset_fingerprint"],
    }
    new_joint = {
        "artifact": {"key": new_joint_key.value},
        "dataset_fingerprint": baseline_inference["dataset_fingerprint"],
    }
    assert ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=old_joint),
    ) != ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=new_joint),
    )

    baseline_bundle = ArtifactKey.from_spec(
        "analysis-bundle",
        _analysis_bundle_spec(
            state=state,
            source_artifact=baseline_inference["artifact"],
            producer_command="infer-network",
            dataset_id="dataset_a",
            scope="individual",
            regime="group_only",
            run_id="genie3",
        ),
    )
    assert baseline_bundle == ArtifactKey.from_spec(
        "analysis-bundle",
        _analysis_bundle_spec(
            state=state,
            source_artifact=deepcopy(baseline_inference["artifact"]),
            producer_command="infer-network",
            dataset_id="dataset_a",
            scope="individual",
            regime="group_only",
            run_id="genie3",
        ),
    )
    assert ArtifactKey.from_spec(
        "analysis-bundle",
        _analysis_bundle_spec(
            state=state,
            source_artifact=old_joint["artifact"],
            producer_command="infer-network",
            dataset_id="dataset_a",
            scope="joint",
            regime="group_only",
            run_id=None,
        ),
    ) != ArtifactKey.from_spec(
        "analysis-bundle",
        _analysis_bundle_spec(
            state=state,
            source_artifact=new_joint["artifact"],
            producer_command="infer-network",
            dataset_id="dataset_a",
            scope="joint",
            regime="group_only",
            run_id=None,
        ),
    )

    changed_orchestrator = deepcopy(state)
    changed_orchestrator["benchmark_repository"]["commit"] = "c" * 40
    assert (
        ArtifactKey.from_spec(
            "joint-inference",
            _integration_spec(
                state=changed_orchestrator,
                dataset=dataset,
                regime="group_only",
                inference_runs=[baseline_for_joint, new_spathi_for_joint],
            ),
        )
        != new_joint_key
    )
    assert (
        ArtifactKey.from_spec(
            "analysis-bundle",
            _analysis_bundle_spec(
                state=changed_orchestrator,
                source_artifact=baseline_inference["artifact"],
                producer_command="infer-network",
                dataset_id="dataset_a",
                scope="individual",
                regime="group_only",
                run_id="genie3",
            ),
        )
        != baseline_bundle
    )


def test_spathi_build_fingerprint_covers_every_locked_input() -> None:
    image = {
        "source_commit": "1" * 40,
        "base_image": "python@sha256:" + "2" * 64,
        "build_inputs": {"adapter.py": "3" * 64},
        "python_dependencies": {"numpy": "1.0"},
    }
    original = _spathi_build_fingerprint(image)

    for field, replacement in (
        ("source_commit", "4" * 40),
        ("base_image", "python@sha256:" + "5" * 64),
        ("build_inputs", {"adapter.py": "6" * 64}),
        ("python_dependencies", {"numpy": "2.0"}),
    ):
        changed = deepcopy(image)
        changed[field] = replacement
        assert _spathi_build_fingerprint(changed) != original


def test_spathi_image_provenance_records_the_requested_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = {
        "tag": "spathi-benchmark:test",
        "source_commit": "1" * 40,
        "base_image": "python@sha256:" + "2" * 64,
        "build_inputs": {"adapter.py": "3" * 64},
        "python_dependencies": {"numpy": "1.0"},
    }
    fingerprint = _spathi_build_fingerprint(image)
    details = {
        "reference": image["tag"],
        "id": "sha256:" + "4" * 64,
        "repo_digests": [],
        "created": "2026-08-28T00:00:00Z",
        "labels": {
            "org.opencontainers.image.revision": image["source_commit"],
            "dev.spathi.benchmark.build-fingerprint": fingerprint,
        },
    }
    run_root = tmp_path / "work" / "v1" / "smoke" / "run"
    frozen_root = tmp_path / "frozen"
    monkeypatch.setattr(
        pipeline,
        "_load_existing_state",
        lambda *_args, **_kwargs: {
            "status": "preflighted",
            "docker_runtime_environment": deepcopy(DOCKER_RUNTIME),
        },
    )
    monkeypatch.setattr(pipeline, "frozen_repository_root", lambda _run_root: frozen_root)
    monkeypatch.setattr(
        pipeline,
        "load_repo_lock",
        lambda _root: {"repositories": {"spathi": {}}},
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_spathi_image_contract",
        lambda _root: {"images": {"spathi": image}},
    )
    monkeypatch.setattr(pipeline, "validate_locked_repository", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "docker_image_details", lambda _reference: deepcopy(details))

    provenance_path = build_spathi_image(
        repo_root=tmp_path,
        profile_id="smoke",
        run_root=run_root,
        spathi_repo=tmp_path / "SPATHI",
    )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert provenance["tag"] == image["tag"]
    assert state["spathi_image"]["tag"] == image["tag"]


def test_artifact_store_rejects_a_filesystem_root_target(tmp_path: Path) -> None:
    shallow = Path(tmp_path.anchor) / "profile" / "run"
    with pytest.raises(BenchmarkConfigError, match="refusing an artifact store"):
        _artifact_store(shallow)


def test_artifact_store_is_shared_by_runs_in_one_version(tmp_path: Path) -> None:
    first = _artifact_store(tmp_path / "v1" / "smoke" / "run_a")
    second = _artifact_store(tmp_path / "v1" / "smoke" / "run_b")

    assert first.root == second.root == (tmp_path / "v1" / "artifacts").resolve()
