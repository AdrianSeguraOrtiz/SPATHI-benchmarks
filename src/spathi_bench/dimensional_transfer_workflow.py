"""Post-freeze, truth-bearing dimensional accuracy-transfer workflow."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .benchmark_materialization import build_selected_spathi_requests
from .config import (
    BenchmarkConfigError,
    load_image_lock,
    load_profile,
    validate_json,
    validate_profile_references,
)
from .dimensional_transfer import (
    PANEL_PATH,
    analyze_dimensional_transfer,
    condition_ids_for_budget,
    transfer_algorithm_seed,
    validate_transfer_panel,
)
from .provenance import observe_benchmark_repository
from .report_publication import publish_workflow_report
from .reporting import render_dimensional_transfer_report
from .resource_plan import detect_host_capacity, load_or_create_runtime_plan
from .resource_scheduler import host_execution_lease
from .runtime import (
    andrea_host_environment,
    benchmark_host_environment,
    docker_runtime_environment,
)
from .safe_io import path_below_no_symlinks, write_idempotent_json
from .selected_policy import completed_selected_config
from .state import run_lock
from .study_context import bind_phase_identity, phase_flow_id
from .workflows import run_profile_pipeline

_IDENTITY_INPUTS = (
    "src/spathi_bench/spathi_diagnostics.py",
    "experiments/v1/execution/policy.json",
    "experiments/v1/seeds/seed-plan.json",
    "images.lock.json",
    "repos.lock.json",
    "schemas/dimensional-transfer-panel.schema.json",
    "schemas/execution-policy.schema.json",
    "schemas/dimensional-transfer-results.schema.json",
    "schemas/profile.schema.json",
    "schemas/spathi-selected-config.schema.json",
    "src/spathi_bench/dimensional_transfer.py",
    "src/spathi_bench/dimensional_transfer_workflow.py",
    "src/spathi_bench/selected_policy.py",
    "src/spathi_bench/report_publication.py",
)


def _relative(repo_root: Path, path: Path, label: str) -> str:
    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    absolute = path_below_no_symlinks(path, root, label)
    return absolute.relative_to(root).as_posix()


def _panel_dependency_hashes(repo_root: Path, panel: Mapping[str, Any]) -> dict[str, str]:
    paths = {PANEL_PATH}
    for condition in panel["conditions"]:
        scenario_path = path_below_no_symlinks(
            repo_root / condition["scenario"],
            repo_root,
            "dimensional-transfer scenario",
            require_file=True,
        )
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        paths.add(str(condition["scenario"]))
        paths.update(str(path) for path in condition["runs"].values())
        for input_spec in scenario["inputs"].values():
            paths.add(
                _relative(
                    repo_root,
                    scenario_path.parent / input_spec["path"],
                    "dimensional-transfer scenario input",
                )
            )
    return {
        path: sha256_file(
            path_below_no_symlinks(
                repo_root / path,
                repo_root,
                f"dimensional-transfer dependency {path}",
                require_file=True,
            )
        )
        for path in sorted(paths)
    }


def materialize_dimensional_transfer_profile(
    *,
    repo_root: Path,
    budget: str,
    template_profile: Mapping[str, Any],
    selected_config_path: Path,
    production: Mapping[str, Any],
    panel: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Bind the declared panel to exactly one immutable production policy."""

    root = repo_root.resolve()
    expected_id = f"dimensional-transfer-{budget}"
    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError("Dimensional-transfer budget must be 'reduced' or 'full'")
    if (
        template_profile.get("id") != expected_id
        or template_profile.get("purpose") != "dimensional_transfer"
        or template_profile.get("status") != "runnable"
    ):
        raise BenchmarkConfigError("Dimensional-transfer template does not match its budget")
    if production.get("calibration_budget") != budget:
        raise BenchmarkConfigError("Production policy and transfer profile use different budgets")
    policy = production.get("selected_policy")
    if not isinstance(policy, Mapping):
        raise BenchmarkConfigError("Dimensional-transfer production policy is missing")

    active_condition_ids = condition_ids_for_budget(budget)
    expected_jobs = {
        str(condition["id"]): str(condition["runs"][budget])
        for condition in panel["conditions"]
        if str(condition["id"]) in active_condition_ids
    }
    jobs = template_profile.get("generation_jobs")
    if not isinstance(jobs, list) or len(jobs) != len(expected_jobs):
        raise BenchmarkConfigError("Dimensional-transfer profile does not cover its budget panel")
    observed_jobs = {str(job.get("condition_id")): str(job.get("simulator_runs")) for job in jobs}
    if observed_jobs != expected_jobs:
        raise BenchmarkConfigError("Dimensional-transfer profile differs from the frozen panel")

    runtime_dir = path_below_no_symlinks(
        output_dir,
        root / "work",
        "Dimensional-transfer requests",
    )
    algorithm_seed, _derivation = transfer_algorithm_seed(root, budget=budget)
    tools, custom_tools, metadata = build_selected_spathi_requests(
        policy,
        [algorithm_seed],
        docker_image=load_image_lock(root)["images"]["spathi"]["tag"],
        seed_namespace=f"{budget}.dimensional_transfer",
    )
    tools_path = write_idempotent_json(
        runtime_dir / "tools-params.json", tools, "tools request", boundary=root
    )
    custom_path = write_idempotent_json(
        runtime_dir / "custom-tools.json",
        custom_tools,
        "custom-tools request",
        boundary=root,
    )
    metadata_path = write_idempotent_json(
        runtime_dir / "run-metadata.json",
        metadata,
        "run-metadata request",
        boundary=root,
    )
    inference_batches = [
        {
            "id": f"infer_dimensional_transfer_{budget}",
            "kind": "spathi",
            "dataset_role": "dimensional_transfer",
            "generation_jobs": [str(job["id"]) for job in jobs],
            "regime": "group_only",
            "tools_params": _relative(root, tools_path, "tools request"),
            "custom_tools": _relative(root, custom_path, "custom-tools request"),
            "run_metadata": _relative(root, metadata_path, "run-metadata request"),
        }
    ]
    runtime_profile = copy.deepcopy(dict(template_profile))
    profile_id = f"dimensional-transfer-{budget}-runtime"
    runtime_profile.update(
        {
            "id": profile_id,
            "description": (
                "Runtime dimensional-transfer profile executing exactly one frozen SPATHI "
                f"policy over the {budget} truth-bearing condition panel."
            ),
            "selected_config": _relative(root, selected_config_path, "production policy"),
            "inference_batches": inference_batches,
        }
    )
    profile_path = write_idempotent_json(
        runtime_dir / "profile.json",
        runtime_profile,
        "dimensional-transfer runtime profile",
        boundary=root,
    )
    validate_json(
        runtime_profile,
        root / "schemas/profile.schema.json",
        "dimensional-transfer runtime profile",
    )
    validate_profile_references(root, runtime_profile)
    return {
        "profile_id": profile_id,
        "profile_path": profile_path,
        "tools_path": tools_path,
        "custom_tools_path": custom_path,
        "metadata_path": metadata_path,
        "algorithm_seed": algorithm_seed,
    }


def run_dimensional_transfer_workflow(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    profile: str,
    calibration_run_id: str | None = None,
    max_cores: int | None = None,
    max_ram_gb: float | None = None,
    reserve_cores: int = 0,
    reserve_ram_gb: float = 0.0,
) -> dict[str, Any]:
    """Run or resume one post-freeze dimensional accuracy-transfer study."""

    if profile not in {"reduced", "full"}:
        raise BenchmarkConfigError("Dimensional-transfer profile must be 'reduced' or 'full'")
    root = repo_root.resolve()
    work = path_below_no_symlinks(
        work_root,
        root / "work",
        "Dimensional-transfer work root",
    )

    production_path, production = completed_selected_config(
        repo_root=root,
        work_root=work,
        budget=profile,
        calibration_run_id=calibration_run_id,
    )
    template_id = f"dimensional-transfer-{profile}"
    template_path = root / f"experiments/v1/profiles/{template_id}.json"
    template = load_profile(root, template_id, require_runnable=True)
    panel_path = root / PANEL_PATH
    panel = validate_transfer_panel(root, panel_path)
    benchmark = observe_benchmark_repository(root, require_clean=True)
    identity = canonical_json_sha256(
        {
            "template_profile_sha256": sha256_file(template_path),
            "panel_dependencies": _panel_dependency_hashes(root, panel),
            "selected_config_sha256": sha256_file(production_path),
            "benchmark_repository": {
                "commit": benchmark["commit"],
                "origin": benchmark["origin"],
            },
            "andrea_environment": andrea_host_environment()["fingerprint"],
            "benchmark_environment": benchmark_host_environment()["fingerprint"],
            "docker_runtime": docker_runtime_environment()["fingerprint"],
            "implementation": {path: sha256_file(root / path) for path in _IDENTITY_INPUTS},
        }
    )
    flow_id = phase_flow_id(identity[:20])
    workflow_root = work / "v1/workflows/dimensional_transfer" / profile / flow_id
    identity = bind_phase_identity(workflow_root, {"fingerprint": identity})["fingerprint"]
    workflow_path = workflow_root / "workflow.json"
    runtime_plan = load_or_create_runtime_plan(
        workflow_root / "runtime-plan.json",
        boundary=workflow_root,
        max_cores=max_cores,
        max_ram_gb=max_ram_gb,
        reserve_cores=reserve_cores,
        reserve_ram_gb=reserve_ram_gb,
        detected=detect_host_capacity(workspace_path=work),
    )
    with host_execution_lease(workload=f"dimensional-transfer:{flow_id}"), run_lock(workflow_root):
        materialized = materialize_dimensional_transfer_profile(
            repo_root=root,
            budget=profile,
            template_profile=template,
            selected_config_path=production_path,
            production=production,
            panel=panel,
            output_dir=workflow_root / "materialized",
        )
        profile_id = str(materialized["profile_id"])
        runtime_profile_path = Path(materialized["profile_path"])
        run_root = work / "v1" / profile_id / flow_id
        pipeline = run_profile_pipeline(
            repo_root=root,
            profile_id=profile_id,
            profile_file=runtime_profile_path,
            run_root=run_root,
            andrea_repo=andrea_repo,
            spathi_repo=spathi_repo,
            statistics=False,
            runtime_plan=runtime_plan,
        )
        results_path = analyze_dimensional_transfer(
            repo_root=root,
            budget=profile,
            panel_path=panel_path,
            profile_path=runtime_profile_path,
            selected_config_path=production_path,
            run_root=run_root,
            output_path=workflow_root / "dimensional-transfer-results.json",
        )
        report_path = work / "v1/reports" / profile / flow_id / "dimensional_transfer.html"
        publishing_state = {
            "schema_version": "1.0",
            "workflow_type": "dimensional_transfer",
            "profile": profile,
            "flow_id": flow_id,
            "identity": {"fingerprint": identity},
            "status": "publishing",
            "runtime_plan": runtime_plan,
            "selected_config": _relative(root, production_path, "production policy"),
            "selected_config_sha256": sha256_file(production_path),
            "template_profile": _relative(root, template_path, "template profile"),
            "template_profile_sha256": sha256_file(template_path),
            "runtime_profile": _relative(root, runtime_profile_path, "runtime profile"),
            "runtime_profile_sha256": sha256_file(runtime_profile_path),
            "panel": _relative(root, panel_path, "dimensional-transfer panel"),
            "panel_sha256": sha256_file(panel_path),
            "results": _relative(root, results_path, "dimensional-transfer results"),
            "results_sha256": sha256_file(results_path),
            "run_root": _relative(root, run_root, "dimensional-transfer pipeline"),
            "interactive_report": _relative(root, report_path, "dimensional-transfer report"),
            "pipeline": pipeline,
        }
        state, _report = publish_workflow_report(
            repo_root=root,
            workflow_path=workflow_path,
            publishing_state=publishing_state,
            html_path=report_path,
            manifest_path=report_path.parent / "dimensional-transfer-report.manifest.json",
            render=lambda: render_dimensional_transfer_report(
                repo_root=root,
                version_root=work / "v1",
                workflow_root=workflow_root,
            ),
            label="dimensional transfer",
        )
    return state


__all__ = [
    "materialize_dimensional_transfer_profile",
    "run_dimensional_transfer_workflow",
]
