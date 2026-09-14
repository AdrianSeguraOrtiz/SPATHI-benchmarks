from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from spathi_bench.artifacts import canonical_json_sha256, sha256_file
from spathi_bench.config import BenchmarkConfigError, load_profile, validate_json
from spathi_bench.performance import (
    _catalog_scientific_requests,
    build_performance_tasks,
    load_performance_design,
    materialize_performance_protocol,
    resolve_performance_assignment,
    validate_performance_catalog_threading,
)
from spathi_bench.performance_workflow import (
    _canonical_network_sha256,
    _measurement_record,
    _validate_performance_host_identity,
    summarize_performance_measurements,
)
from spathi_bench.provenance import (
    freeze_run_inputs,
    frozen_profile,
    frozen_repository_root,
    validate_run_inputs,
)
from spathi_bench.reporting.performance import render_performance_report
from spathi_bench.resource_plan import resolve_runtime_plan

REPO_ROOT = Path(__file__).resolve().parents[1]


def _runtime_plan(threads: int = 60) -> dict[str, object]:
    return {
        "capacity": {"cpu_threads": threads},
        "detected": {
            "allowed_cpus": list(range(threads)),
            "cpu_allocation_order": list(range(threads)),
        },
    }


def _row(
    *,
    threads: int,
    repetition: int,
    wall: float,
    network: str | None = "a" * 64,
    status: str = "completed",
    phase: str = "measurement",
    oom: bool = False,
) -> dict[str, object]:
    return {
        "phase": phase,
        "dataset": {"dataset_id": "dataset-a", "size_class": "medium"},
        "configuration_id": "spathi_production",
        "tool_id": "spathi",
        "threads": threads,
        "system_repetition": repetition,
        "status": status,
        "measurement": {
            "wall_time_seconds": wall,
            "cpu_time_seconds": wall * threads,
            "peak_memory_bytes": 1000 * threads,
            "output_bytes": 2000,
        },
        "oom_observed": oom,
        "network_sha256": network,
    }


def test_frozen_performance_design_has_exact_panels_and_counts() -> None:
    design = load_performance_design(REPO_ROOT)

    assert design["claim_scope"] == "computational_performance_only"
    assert design["protocol"]["inference_cache"] == "disabled"
    assert design["protocol"]["output_profile"] == "full"
    reduced = design["budgets"]["reduced"]
    full = design["budgets"]["full"]
    assert reduced["thread_counts"] == [1, 8]
    assert reduced["expected_measured_runs"] == 8
    assert full["thread_counts"] == [1, 2, 4, 8]
    assert full["expected_measured_runs"] == 144
    assert full["expected_warmup_runs"] == 48
    assert [item["configuration_id"] for item in full["tools"]] == [
        "spathi_production",
        "aracne3_default",
        "genie3_default",
        "grnboost2_default",
        "infercsn_none",
        "tigress_default",
    ]


def test_catalog_requests_share_one_budget_specific_reproducible_seed() -> None:
    design = load_performance_design(REPO_ROOT)
    seeds: dict[str, int] = {}
    for budget in ("reduced", "full"):
        requests, seed = _catalog_scientific_requests(
            repo_root=REPO_ROOT,
            budget_design=design["budgets"][budget],
        )
        seeds[budget] = seed
        assert {item["algorithm_seed"] for item in requests} == {seed}
        assert {item["metadata"]["seed_derivation"]["namespace"] for item in requests} == {
            f"{budget}.performance"
        }
        assert all(item["request"]["params"]["seed"] == seed for item in requests)
    assert seeds["reduced"] != seeds["full"]


def test_primary_catalog_tools_must_support_every_thread_endpoint(
    tmp_path: Path,
) -> None:
    design = load_performance_design(REPO_ROOT)
    tool_ids = {
        item["tool_id"]
        for budget in design["budgets"].values()
        for item in budget["tools"]
        if item["origin"] == "catalog"
    }
    for tool_id in tool_ids:
        spec_path = tmp_path / "andrea/catalog_inference_tools/tools" / tool_id / "toolspec.json"
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text(
            json.dumps(
                {
                    "id": tool_id,
                    "runtime_resources": {
                        "threading": {
                            "supported": True,
                            "default_threads": 1,
                            "max_threads": 8,
                            "upstream_mapping": "environment:OMP_NUM_THREADS",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    validate_performance_catalog_threading(tmp_path, design)
    genie3 = tmp_path / "andrea/catalog_inference_tools/tools/genie3/toolspec.json"
    invalid = json.loads(genie3.read_text(encoding="utf-8"))
    invalid["runtime_resources"]["threading"]["max_threads"] = 4
    genie3.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="genie3.*1–8 thread contract"):
        validate_performance_catalog_threading(tmp_path, design)


@pytest.mark.parametrize("budget", ["reduced", "full"])
def test_materialization_freezes_runnable_requests_resources_and_schedule(
    tmp_path: Path,
    budget: str,
) -> None:
    # Copy only tracked public configuration/build inputs, never workspace data.
    repo = tmp_path / "repo"
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "ls-files",
            "-z",
            "--",
            "experiments",
            "schemas",
            "containers/spathi",
            "repos.lock.json",
            "images.lock.json",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    for relative in filter(None, tracked):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    (repo / ".gitignore").write_text("work/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "add",
            "origin",
            "https://example.invalid/performance-test.git",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Performance Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Freeze public test inputs",
        ],
        check=True,
    )
    group_policy = {"shared": {"n_estimators": 250}}
    test_root = repo / "work/performance-test"
    test_root.mkdir(parents=True)
    provenance = {}
    for name in (
        "calibration_selection",
        "efficiency_plan",
        "efficiency_evidence",
        "efficiency_results",
    ):
        evidence = test_root / f"{name}.json"
        evidence.write_text("{}\n", encoding="utf-8")
        provenance[name] = {
            "path": evidence.relative_to(repo).as_posix(),
            "sha256": sha256_file(evidence),
        }
    production = {
        "schema_version": "1.0",
        "artifact_type": "spathi_production_policy",
        "budget": budget,
        "evidence_role": "descriptive" if budget == "reduced" else "confirmatory",
        "status": "descriptive" if budget == "reduced" else "confirmed",
        "decision": "descriptive_only" if budget == "reduced" else "reference_retained",
        "policy": {
            "policy_id": "test_policy",
            "policy_sha256": canonical_json_sha256(group_policy),
            "group_policy": group_policy,
            "spathi_commit": "1" * 40,
        },
        "provenance": provenance,
    }
    detected = {
        "effective_cpu_threads": 8,
        "effective_memory_bytes": 64 * 1024**3,
        "allowed_cpus": list(range(8)),
        "cpu_allocation_order": list(range(8)),
    }
    runtime_plan = resolve_runtime_plan(detected)
    template_path = repo / f"experiments/v1/profiles/performance-{budget}.json"
    template = load_profile(repo, f"performance-{budget}", require_runnable=True)
    production_path = test_root / "production.json"
    runtime_plan_path = test_root / "runtime-plan.json"
    production_path.write_text(json.dumps(production), encoding="utf-8")
    runtime_plan_path.write_text(json.dumps(runtime_plan), encoding="utf-8")
    materialized = materialize_performance_protocol(
        repo_root=repo,
        budget=budget,
        template_path=template_path,
        template_profile=template,
        production_path=production_path,
        production=production,
        runtime_plan_path=runtime_plan_path,
        runtime_plan=runtime_plan,
        output_dir=test_root / "materialized",
    )

    manifest = materialized["manifest"]
    validate_json(
        manifest,
        repo / "schemas/performance-materialization.schema.json",
        "test performance materialization",
    )
    assert len(manifest["tasks"]) == (16 if budget == "reduced" else 192)
    assert manifest["execution"]["inference_cache"] == "disabled"
    assert manifest["execution"]["output_profile"] == "full"
    assert manifest["runtime_plan"]["detected"] == detected
    assert all(task["cpuset_cpus"] == list(range(task["threads"])) for task in manifest["tasks"])
    assert {task["ram_gb"] for task in manifest["tasks"]} == {64.0}
    spathi = materialized["scientific_requests"]["spathi_production"]
    assert spathi["custom_tool"]["runtime_resources"] == {
        "threading": {
            "supported": True,
            "default_threads": 1,
            "max_threads": None,
            "upstream_mapping": "cli:--threads",
        },
    }

    run_root = test_root / "run"
    profile_id = materialized["profile_id"]
    snapshot = freeze_run_inputs(
        repo, profile_id, run_root, profile_file=materialized["profile_path"]
    )
    frozen = frozen_repository_root(run_root)
    entries = {entry["path"]: entry for entry in snapshot["files"]}
    runtime_profile = materialized["runtime_profile"]
    original_bytes = {}
    for batch in runtime_profile["inference_batches"]:
        for field in ("tools_params", "run_metadata", "custom_tools"):
            relative = batch.get(field)
            if relative is None:
                continue
            original_bytes[relative] = (repo / relative).read_bytes()
            assert entries[relative]["kind"] == "runtime-derived"
            assert entries[relative]["sha256"] == sha256_file(repo / relative)
            assert (frozen / relative).read_bytes() == original_bytes[relative]
        metadata = json.loads((frozen / batch["run_metadata"]).read_text())
        assert {row["algorithm_seed"] for row in metadata["runs"]} == {spathi["algorithm_seed"]}
    assert frozen_profile(run_root, profile_id) == runtime_profile
    schedule_before = sha256_file(materialized["manifest_path"])
    assert manifest["schedule_sha256"] == canonical_json_sha256(manifest["tasks"])

    # A resumed run must read its frozen files, not mutable source materializations.
    catalog = next(
        batch for batch in runtime_profile["inference_batches"] if batch["kind"] == "baseline"
    )
    (repo / catalog["tools_params"]).write_text("{}\n", encoding="utf-8")
    (repo / catalog["run_metadata"]).unlink()
    assert validate_run_inputs(repo, profile_id, run_root) == snapshot
    assert (
        freeze_run_inputs(repo, profile_id, run_root, profile_file=materialized["profile_path"])
        == snapshot
    )
    assert frozen_profile(run_root, profile_id) == runtime_profile
    assert sha256_file(materialized["manifest_path"]) == schedule_before
    assert {
        relative: (frozen / relative).read_bytes() for relative in original_bytes
    } == original_bytes


@pytest.mark.parametrize(
    ("budget", "warmups", "measurements", "repetitions"),
    [("reduced", 8, 8, 1), ("full", 48, 144, 3)],
)
def test_task_schedule_is_deterministic_complete_and_counterbalanced(
    budget: str,
    warmups: int,
    measurements: int,
    repetitions: int,
) -> None:
    design = load_performance_design(REPO_ROOT)
    selected = design["budgets"][budget]
    first = build_performance_tasks(
        budget_design=selected,
        runtime_plan=_runtime_plan(),
        order_salt=design["protocol"]["order_salt"],
        ram_gb=48,
    )
    second = build_performance_tasks(
        budget_design=selected,
        runtime_plan=_runtime_plan(),
        order_salt=design["protocol"]["order_salt"],
        ram_gb=48,
    )

    assert first == second
    assert [item["sequence_index"] for item in first] == list(range(1, len(first) + 1))
    assert len({item["task_id"] for item in first}) == len(first)
    assert sum(item["phase"] == "warmup" for item in first) == warmups
    assert sum(item["phase"] == "measurement" for item in first) == measurements
    condition_counts = Counter(
        (item["generation_job_id"], item["configuration_id"], item["threads"])
        for item in first
        if item["phase"] == "measurement"
    )
    assert set(condition_counts.values()) == {repetitions}
    assert all(item["cpuset_cpus"] == list(range(item["threads"])) for item in first)
    assert {item["ram_gb"] for item in first} == {48.0}


@pytest.mark.parametrize("ram_gb", [0, -1, True, float("nan")])
def test_performance_tasks_reject_invalid_ram_allocations(ram_gb: object) -> None:
    design = load_performance_design(REPO_ROOT)
    with pytest.raises(BenchmarkConfigError, match="RAM allocation must be positive"):
        build_performance_tasks(
            budget_design=design["budgets"]["reduced"],
            runtime_plan=_runtime_plan(),
            order_salt=design["protocol"]["order_salt"],
            ram_gb=ram_gb,  # type: ignore[arg-type]
        )


def test_performance_assignment_uses_exact_prefix_and_rejects_oversubscription() -> None:
    plan = {
        "capacity": {"cpu_threads": 4},
        "detected": {
            "allowed_cpus": [2, 4, 6, 8],
            "cpu_allocation_order": [2, 6, 4, 8],
        },
    }

    assert resolve_performance_assignment(plan, 2) == {
        "threads": 2,
        "cpuset_cpus": [2, 6],
    }
    assert resolve_performance_assignment(plan, 4) == {
        "threads": 4,
        "cpuset_cpus": [2, 4, 6, 8],
    }
    with pytest.raises(BenchmarkConfigError, match="only 4 are schedulable"):
        resolve_performance_assignment(plan, 8)
    with pytest.raises(BenchmarkConfigError, match="CPU contract is invalid"):
        resolve_performance_assignment(
            {
                "capacity": {"cpu_threads": 4},
                "detected": {
                    "allowed_cpus": [2, 2],
                    "cpu_allocation_order": [2, 2],
                },
            },
            2,
        )


def test_resume_accepts_volatile_capacity_changes_but_rejects_another_host() -> None:
    stable = {
        "logical_cpus_host": 16,
        "affinity_cpus": list(range(8)),
        "allowed_cpus": list(range(8)),
        "cpu_allocation_order": list(range(8)),
        "physical_cores_allowed": 4,
        "cpu_packages_allowed": 1,
        "cgroup_cpu_quota_cores": 8.0,
        "host_memory_total_bytes": 64 * 1024**3,
        "cgroup_memory_limit_bytes": None,
        "host_memory_available_bytes": 48 * 1024**3,
        "workspace_storage_free_bytes": 1000,
    }
    current = {
        **stable,
        "host_memory_available_bytes": 32 * 1024**3,
        "workspace_storage_free_bytes": 500,
    }

    _validate_performance_host_identity({"detected": stable}, current)
    current["allowed_cpus"] = list(range(1, 9))
    with pytest.raises(BenchmarkConfigError, match="allowed_cpus"):
        _validate_performance_host_identity({"detected": stable}, current)


def test_summary_excludes_warmups_and_computes_scaling_from_system_repetitions() -> None:
    rows = [
        _row(threads=1, repetition=0, wall=999, phase="warmup"),
        _row(threads=1, repetition=1, wall=10),
        _row(threads=1, repetition=2, wall=12),
        _row(threads=1, repetition=3, wall=14),
        _row(threads=8, repetition=1, wall=2),
        _row(threads=8, repetition=2, wall=3),
        _row(threads=8, repetition=3, wall=4),
    ]

    aggregates, summary = summarize_performance_measurements(rows)

    by_threads = {item["threads"]: item for item in aggregates}
    assert by_threads[1]["wall_time_median_seconds"] == 12
    assert by_threads[1]["speedup_vs_one_thread"] == 1
    assert by_threads[8]["wall_time_median_seconds"] == 3
    assert by_threads[8]["speedup_vs_one_thread"] == 4
    assert by_threads[8]["parallel_efficiency"] == 0.5
    assert by_threads[8]["container_peak_overlap_upper_bound_max_bytes"] == 8000
    assert by_threads[8]["scaling_metrics_status"] == "published"
    assert summary["measured_runs_requested"] == 6
    assert summary["all_measured_runs_completed"] is True
    assert summary["equivalence_confirmed"] == 1


@pytest.mark.parametrize(
    "rows",
    [
        [
            _row(threads=1, repetition=1, wall=10, network="a" * 64),
            _row(threads=8, repetition=1, wall=2, network="b" * 64),
        ],
        [
            _row(threads=1, repetition=1, wall=10, network="a" * 64),
            _row(threads=1, repetition=2, wall=11, network="b" * 64),
            _row(threads=8, repetition=1, wall=2, network="a" * 64),
            _row(threads=8, repetition=2, wall=3, network="a" * 64),
        ],
    ],
)
def test_semantic_drift_withholds_every_scaling_metric(
    rows: list[dict[str, object]],
) -> None:
    aggregates, summary = summarize_performance_measurements(rows)

    assert summary["equivalence_failed"] == 1
    assert all(item["speedup_vs_one_thread"] is None for item in aggregates)
    assert all(item["parallel_efficiency"] is None for item in aggregates)
    assert {
        item["scaling_metrics_status"] for item in aggregates
    } == {"withheld_semantic_drift"}


def test_incomplete_condition_withholds_speedup_and_equivalence_and_retains_oom() -> None:
    rows = [
        _row(threads=1, repetition=1, wall=10),
        _row(threads=8, repetition=1, wall=3, network=None, status="failed", oom=True),
    ]

    aggregates, summary = summarize_performance_measurements(rows)

    eight = next(item for item in aggregates if item["threads"] == 8)
    assert eight["repetitions_completed"] == 0
    assert eight["speedup_vs_one_thread"] is None
    assert eight["parallel_efficiency"] is None
    assert eight["scaling_metrics_status"] == "withheld_incomplete_execution"
    assert eight["network_equivalent_within_condition"] is None
    assert eight["oom_count"] == 1
    assert summary["all_measured_runs_completed"] is False
    assert summary["oom_runs"] == 1
    assert summary["equivalence_unavailable"] == 1


def test_semantic_network_hash_ignores_csv_row_order(tmp_path: Path) -> None:
    header = "source,target,score,sign,evidence,context,tool_id\n"
    rows = ["TF1,G2,0.2,,inferred,g1,spathi\n", "TF1,G1,0.8,,inferred,g1,spathi\n"]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(header + "".join(rows), encoding="utf-8")
    second.write_text(header + "".join(reversed(rows)), encoding="utf-8")

    assert _canonical_network_sha256(first) == _canonical_network_sha256(second)


def test_measurement_record_requires_exact_affinity_and_preserves_telemetry(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "runs" / "measure_medium_spathi_production_t8_r1"
    output = task_root / "output" / "dataset-a_20260101T000000Z"
    output.mkdir(parents=True)
    network = output / "merged_network_normalized.csv"
    network.write_text(
        "source,target,score,sign,evidence,context,tool_id\nTF1,G1,0.7,,inferred,g1,spathi\n",
        encoding="utf-8",
    )
    measurement = {
        "schema_version": "1.0",
        "scope": "andrea_execution",
        "wall_time_seconds": 4.5,
        "cpu_time_seconds": 25.0,
        "peak_memory_bytes": 4096,
        "output_bytes": 2048,
        "assigned_resources": {
            "threads_peak": 8,
            "ram_limit_peak_bytes": 64 * 1024**3,
            "cpuset_cpus_requested_sets": [list(range(8))],
            "cpuset_cpus_effective_sets": [list(range(8))],
        },
        "telemetry": {
            "status": "complete",
            "wall_source": "andrea_monotonic_clock",
            "wall_semantics": (
                "andrea_execution_end_to_end_including_input_verification_preparation_"
                "scheduling_merging_and_requested_output_materialization"
            ),
            "memory_source": "child_container_cgroup_peaks",
            "memory_semantics": (
                "overlap_upper_bound_from_physical_container_peaks_"
                "excludes_andrea_host_process"
            ),
            "oom_events": 0,
            "oom_killed": False,
        },
    }
    report = {
        "run_id": "spathi_policy__seed_11",
        "status": "executed",
        "output_profile": "full",
        "dataset": {"id": "dataset-a", "genes": 400, "columns": 600},
        "execution": {"measurement": measurement},
        "tools": {
            "selected": ["spathi_policy__seed_11"],
            "tool_origins": {"spathi_policy__seed_11": "custom"},
            "status_by_tool": {"spathi_policy__seed_11": "completed"},
            "failed": {},
            "results": {
                "spathi_policy__seed_11": {
                    "status": "completed",
                    "measurement": {"scope": "logical_run", "telemetry": {"status": "complete"}},
                }
            },
        },
        "outputs": {"merged_network_normalized": "merged_network_normalized.csv"},
    }
    report_path = output / "run_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    task = {
        "task_id": "measure_medium_spathi_production_t8_r1",
        "sequence_index": 9,
        "phase": "measurement",
        "generation_job_id": "scmultisim_performance_medium",
        "replicate_index": 1,
        "size_class": "medium",
        "configuration_id": "spathi_production",
        "tool_id": "spathi",
        "origin": "spathi_production_policy",
        "threads": 8,
        "ram_gb": 64.0,
        "cpuset_cpus": list(range(8)),
        "system_repetition": 1,
    }
    dataset = {"dataset_id": "dataset-a", "seed": 42}
    request = {
        "algorithm_seed": 11,
        "request": {"run_id": "spathi_policy__seed_11"},
    }

    record = _measurement_record(
        repo_root=tmp_path,
        task=task,
        dataset=dataset,
        request=request,
        report_path=report_path,
        task_root=task_root,
        command_return_code=0,
    )

    assert record["measurement"] == measurement
    assert record["dataset"]["cells"] == 600
    assert record["network_sha256"] == _canonical_network_sha256(network)
    assert record["io"]["status"] == "unavailable"
    tampered = deepcopy(report)
    tampered["execution"]["measurement"]["assigned_resources"]["cpuset_cpus_effective_sets"] = [
        list(range(1, 9))
    ]
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="resources differ"):
        _measurement_record(
            repo_root=tmp_path,
            task=task,
            dataset=dataset,
            request=request,
            report_path=report_path,
            task_root=task_root,
            command_return_code=0,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("scope", "physical_task"), "canonical end-to-end telemetry"),
        (("schema_version", "2.0"), "canonical end-to-end telemetry"),
        (("telemetry.status", "partial"), "complete end-to-end telemetry"),
        (("assigned_resources.ram_limit_peak_bytes", 1), "resources differ"),
        (("assigned_resources.threads_peak", 2), "resources differ"),
        (("wall_time_seconds", 0), "no measurable execution"),
        (("cpu_time_seconds", None), "CPU time must be"),
        (("peak_memory_bytes", None), "container-memory upper bound must be"),
        (
            ("telemetry.memory_source", "host_rss"),
            "container-memory overlap upper bound",
        ),
        (
            ("telemetry.memory_semantics", "whole_process_peak_ram"),
            "container-memory overlap upper bound",
        ),
    ],
)
def test_measurement_record_rejects_noncanonical_performance_telemetry(
    tmp_path: Path,
    mutation: tuple[str, object],
    message: str,
) -> None:
    task_root = tmp_path / "runs" / "measure_medium_spathi_production_t1_r1"
    output = task_root / "output" / "dataset-a_20260101T000000Z"
    output.mkdir(parents=True)
    (output / "merged_network_normalized.csv").write_text(
        "source,target,score,sign,evidence,context,tool_id\nTF1,G1,0.7,,inferred,g1,spathi\n",
        encoding="utf-8",
    )
    measurement: dict[str, object] = {
        "schema_version": "1.0",
        "scope": "andrea_execution",
        "wall_time_seconds": 1.0,
        "cpu_time_seconds": 0.8,
        "peak_memory_bytes": 4096,
        "output_bytes": 2048,
        "assigned_resources": {
            "threads_peak": 1,
            "ram_limit_peak_bytes": 8 * 1024**3,
            "cpuset_cpus_requested_sets": [[0]],
            "cpuset_cpus_effective_sets": [[0]],
        },
        "telemetry": {
            "status": "complete",
            "wall_source": "andrea_monotonic_clock",
            "wall_semantics": (
                "andrea_execution_end_to_end_including_input_verification_preparation_"
                "scheduling_merging_and_requested_output_materialization"
            ),
            "memory_source": "child_container_cgroup_peaks",
            "memory_semantics": (
                "overlap_upper_bound_from_physical_container_peaks_"
                "excludes_andrea_host_process"
            ),
            "oom_events": 0,
            "oom_killed": False,
        },
    }
    key, value = mutation
    target: dict[str, object] = measurement
    path = key.split(".")
    for part in path[:-1]:
        nested = target[part]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    run_id = "spathi_policy__seed_11"
    report = {
        "run_id": run_id,
        "status": "executed",
        "output_profile": "full",
        "dataset": {"id": "dataset-a", "genes": 400, "columns": 600},
        "execution": {"measurement": measurement},
        "tools": {
            "selected": [run_id],
            "tool_origins": {run_id: "custom"},
            "status_by_tool": {run_id: "completed"},
            "failed": {},
            "results": {
                run_id: {
                    "status": "completed",
                    "measurement": {
                        "scope": "logical_run",
                        "telemetry": {"status": "complete"},
                    },
                }
            },
        },
        "outputs": {"merged_network_normalized": "merged_network_normalized.csv"},
    }
    report_path = output / "run_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match=message):
        _measurement_record(
            repo_root=tmp_path,
            task={
                "task_id": "measure_medium_spathi_production_t1_r1",
                "sequence_index": 1,
                "phase": "measurement",
                "generation_job_id": "scmultisim_performance_medium",
                "replicate_index": 1,
                "size_class": "medium",
                "configuration_id": "spathi_production",
                "tool_id": "spathi",
                "origin": "spathi_production_policy",
                "threads": 1,
                "ram_gb": 8.0,
                "cpuset_cpus": [0],
                "system_repetition": 1,
            },
            dataset={"dataset_id": "dataset-a", "seed": 42},
            request={"algorithm_seed": 11, "request": {"run_id": run_id}},
            report_path=report_path,
            task_root=task_root,
            command_return_code=0,
        )


def test_interactive_report_renders_complete_reduced_evidence(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "schemas", root / "schemas")
    flow_id = "a" * 20
    version = root / "work/v1"
    workflow_root = version / f"workflows/performance/reduced/{flow_id}"
    workflow_root.mkdir(parents=True)
    detected = {
        "effective_cpu_threads": 8,
        "effective_memory_bytes": 64 * 1024**3,
        "allowed_cpus": list(range(8)),
        "cpu_allocation_order": list(range(8)),
        "logical_cpus_host": 16,
        "host_memory_total_bytes": 128 * 1024**3,
    }
    runtime_plan = resolve_runtime_plan(detected)
    runtime_plan_path = workflow_root / "runtime-plan.json"
    runtime_plan_path.write_text(json.dumps(runtime_plan), encoding="utf-8")
    runtime_reference = {
        "path": runtime_plan_path.relative_to(root).as_posix(),
        "sha256": sha256_file(runtime_plan_path),
        "fingerprint": runtime_plan["fingerprint"],
        "execution_class": "performance_full",
        "request": runtime_plan["request"],
        "detected": runtime_plan["detected"],
        "capacity": {
            "cpu_threads": runtime_plan["capacity"]["cpu_threads"],
            "memory_bytes": runtime_plan["capacity"]["memory_bytes"],
        },
    }
    materialization = {
        "schema_version": "1.0",
        "artifact_type": "spathi_performance_materialization",
        "budget": "reduced",
        "design": {"path": "design.json", "sha256": "1" * 64},
        "template_profile": {"path": "profile.json", "sha256": "2" * 64},
        "production_config": {
            "path": "production.json",
            "sha256": "3" * 64,
            "policy_id": "policy",
            "policy_sha256": "4" * 64,
        },
        "runtime_plan": runtime_reference,
        "execution": {
            "output_profile": "full",
            "inference_cache": "disabled",
            "concurrency": "one_logical_execution_at_a_time",
            "order_algorithm": "sha256_rank_rotate_reverse_v1",
            "order_salt": "spathi-performance-counterbalance-v1",
            "max_ram_gb": 64,
            "planner": "heuristic",
            "planner_time_limit_seconds": 300,
        },
        "datasets": [
            {
                "generation_job_id": "scmultisim_performance_medium",
                "replicate_index": 1,
                "size_class": "medium",
            }
        ],
        "scientific_requests": [
            {
                "configuration_id": "spathi_production",
                "tool_id": "spathi",
                "origin": "spathi_production_policy",
                "algorithm_seed": 11,
                "request": {"run_id": "spathi"},
                "custom_tool": {},
            }
        ],
        "tasks": [
            {
                "task_id": "warmup_medium_spathi_production_t1_r0",
                "sequence_index": 1,
                "phase": "warmup",
                "generation_job_id": "scmultisim_performance_medium",
                "replicate_index": 1,
                "size_class": "medium",
                "configuration_id": "spathi_production",
                "tool_id": "spathi",
                "origin": "spathi_production_policy",
                "threads": 1,
                "ram_gb": 64.0,
                "cpuset_cpus": [0],
                "system_repetition": 0,
            }
        ],
        "schedule_sha256": "5" * 64,
    }
    materialization_path = workflow_root / "performance-materialization.json"
    materialization_path.write_text(json.dumps(materialization), encoding="utf-8")
    configurations = (
        ("spathi_production", "spathi", "spathi_production_policy"),
        ("genie3_default", "genie3", "catalog"),
        ("grnboost2_default", "grnboost2", "catalog"),
        ("infercsn_none", "infercsn", "catalog"),
    )
    records: list[dict[str, object]] = []
    sequence = 0
    for phase in ("warmup", "measurement"):
        for configuration, tool, origin in configurations:
            for threads in (1, 8):
                sequence += 1
                records.append(
                    {
                        "schema_version": "1.0",
                        "task_id": f"{phase}_{configuration}_t{threads}",
                        "sequence_index": sequence,
                        "phase": phase,
                        "dataset": {
                            "dataset_id": "dataset-medium",
                            "generation_job_id": "scmultisim_performance_medium",
                            "replicate_index": 1,
                            "size_class": "medium",
                            "seed": 17,
                            "cells": 600,
                            "genes": 400,
                        },
                        "configuration_id": configuration,
                        "tool_id": tool,
                        "origin": origin,
                        "threads": threads,
                        "ram_gb": 64.0,
                        "cpuset_cpus": list(range(threads)),
                        "system_repetition": 0 if phase == "warmup" else 1,
                        "algorithm_seed": 11,
                        "status": "completed",
                        "tool_status": "completed",
                        "command_return_code": 0,
                        "measurement": {
                            "schema_version": "1.0",
                            "scope": "andrea_execution",
                            "wall_time_seconds": 8 / threads,
                            "cpu_time_seconds": 7.5,
                            "peak_memory_bytes": 1024**3,
                            "output_bytes": 4096,
                            "assigned_resources": {
                                "threads_peak": threads,
                                "ram_limit_peak_bytes": 64 * 1024**3,
                                "cpuset_cpus_requested_sets": [list(range(threads))],
                                "cpuset_cpus_effective_sets": [list(range(threads))],
                            },
                            "telemetry": {
                                "status": "complete",
                                "memory_source": "child_container_cgroup_peaks",
                                "memory_semantics": (
                                    "overlap_upper_bound_from_physical_container_peaks_"
                                    "excludes_andrea_host_process"
                                ),
                            },
                        },
                        "io": {
                            "status": "unavailable",
                            "read_bytes": None,
                            "write_bytes": None,
                            "reason": "Not available.",
                        },
                        "oom_observed": False,
                        "network_sha256": "6" * 64,
                        "run_report": {"path": "run.json", "sha256": "7" * 64},
                        "output": {"path": "output", "tree_bytes": 8192},
                    }
                )
    aggregates, summary = summarize_performance_measurements(records)
    results = {
        "schema_version": "1.0",
        "artifact_type": "spathi_performance_results",
        "budget": "reduced",
        "evidence_role": "operational",
        "claim_scope": "computational_performance_only",
        "status": "complete",
        "design": {"path": "design.json", "sha256": "1" * 64},
        "materialization": {
            "path": materialization_path.relative_to(root).as_posix(),
            "sha256": sha256_file(materialization_path),
            "schedule_sha256": "5" * 64,
        },
        "production_config": materialization["production_config"],
        "execution_environment": {
            "runtime_plan": runtime_reference,
            "andrea": {"system": "Linux", "machine": "x86_64"},
            "benchmark": {},
            "docker": {},
            "andrea_commit": "8" * 40,
            "spathi_image_id": "sha256:image",
        },
        "protocol": {
            "output_profile": "full",
            "inference_cache": "disabled",
            "concurrency": "one_logical_execution_at_a_time",
            "repetition_semantics": "same_algorithm_seed_independent_system_repetitions",
            "order_algorithm": "sha256_rank_rotate_reverse_v1",
            "order_salt": "spathi-performance-counterbalance-v1",
            "measurement_source": "run_report.execution.measurement",
            "measurement_scope": "andrea_execution",
            "wall_time_semantics": (
                "andrea_execution_end_to_end_including_input_verification_preparation_"
                "scheduling_merging_and_requested_output_materialization"
            ),
            "memory_measurement_source": (
                "run_report.execution.measurement.peak_memory_bytes"
            ),
            "memory_measurement_semantics": (
                "overlap_upper_bound_from_physical_container_peaks_"
                "excludes_andrea_host_process"
            ),
            "io_measurement": "unavailable_in_andrea_v1_output_bytes_only",
            "thread_counts": [1, 8],
            "system_repetitions": 1,
            "warmups_per_condition": 1,
        },
        "summary": summary,
        "aggregates": aggregates,
        "measurements": [row for row in records if row["phase"] == "measurement"],
        "warmups": [row for row in records if row["phase"] == "warmup"],
        "notes": ["Computational evidence only."],
    }
    results_path = workflow_root / "performance-results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    measurements_csv = workflow_root / "performance-measurements.csv"
    summary_csv = workflow_root / "performance-summary.csv"
    measurements_csv.write_text("task_id\nfixture\n", encoding="utf-8")
    summary_csv.write_text("configuration_id\nspathi_production\n", encoding="utf-8")
    report_path = version / f"reports/reduced/{flow_id}/performance.html"
    workflow = {
        "schema_version": "1.0",
        "workflow_type": "performance",
        "profile": "reduced",
        "flow_id": flow_id,
        "identity": {"fingerprint": "9" * 64},
        "status": "publishing",
        "runtime_plan": runtime_plan_path.relative_to(root).as_posix(),
        "runtime_plan_sha256": sha256_file(runtime_plan_path),
        "results": results_path.relative_to(root).as_posix(),
        "results_sha256": sha256_file(results_path),
        "materialization": materialization_path.relative_to(root).as_posix(),
        "materialization_sha256": sha256_file(materialization_path),
        "measurements_csv": measurements_csv.relative_to(root).as_posix(),
        "measurements_csv_sha256": sha256_file(measurements_csv),
        "summary_csv": summary_csv.relative_to(root).as_posix(),
        "summary_csv_sha256": sha256_file(summary_csv),
        "interactive_report": report_path.relative_to(root).as_posix(),
    }
    (workflow_root / "workflow.json").write_text(json.dumps(workflow), encoding="utf-8")

    bundle = render_performance_report(
        repo_root=root,
        version_root=version,
        workflow_root=workflow_root,
    )

    html = report_path.read_text(encoding="utf-8")
    assert bundle["html"]["path"] == report_path.relative_to(root).as_posix()
    assert "How do runtime, memory, and parallel scaling compare?" in html
    assert "Container-memory overlap upper bound" in html
    assert "whole-process peak RAM" in html
    assert "performance-measurements.csv" in html
    assert "plotly.js" in html.lower()
