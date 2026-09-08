from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.cli as cli
import spathi_bench.efficiency_workflow as workflow
from spathi_bench.artifacts import canonical_json_sha256, sha256_file
from spathi_bench.calibration import locked_spathi_identity
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.efficiency_development import load_efficiency_development_contract
from spathi_bench.efficiency_order import counterbalanced_execution_orders
from spathi_bench.state import state_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_efficiency_development_joint_confirmation_reruns_every_arm() -> None:
    contract = load_efficiency_development_contract(REPO_ROOT)

    for budget_id, expected_fits in (("reduced", 75), ("full", 285)):
        budget = contract["budgets"][budget_id]
        confirmation = budget["steps"]["joint_confirmation"]
        assert confirmation["algorithm_seed_indices"] == list(
            range(1, len(budget["algorithm_seeds"]) + 1)
        )
        assert confirmation["expected_physical_fits"] == (
            budget["expected_independent_datasets"]
            * confirmation["expected_candidate_count"]
            * len(budget["algorithm_seeds"])
        )
        assert budget["expected_physical_fits"] == expected_fits


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _repository(
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
    *,
    budget: str = "reduced",
) -> tuple[Path, Path, Path, Path, str]:
    root = tmp_path / "repo"
    for relative in {
        *workflow._IDENTITY_INPUTS,
        "experiments/v1/profiles/efficiency-reduced.json",
        "experiments/v1/profiles/efficiency-full.json",
        "experiments/v1/profiles/efficiency-development-reduced.json",
        "experiments/v1/profiles/efficiency-development-full.json",
        "experiments/v1/profiles/efficiency-safety-full.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/linear-baseline-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/linear-noisy-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/bifurcating-baseline-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/bifurcating-noisy-reduced-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_validation/dynamic-reduced-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/linear-baseline-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/linear-noisy-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/bifurcating-baseline-runs.json",
        "experiments/v1/simulations/dyngen/efficiency_validation/bifurcating-noisy-runs.json",
        "experiments/v1/simulations/scmultisim/efficiency_validation/dynamic-runs.json",
        "experiments/v1/seeds/seed-plan.json",
        "schemas/noninferiority-panel.schema.json",
        "schemas/noninferiority-plan.schema.json",
        "schemas/noninferiority-study.schema.json",
        "schemas/profile.schema.json",
        "schemas/seed-plan.schema.json",
        "schemas/spathi-production-config.schema.json",
        "schemas/spathi-selected-config.schema.json",
    }:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)

    parameters = dict(frozen_spathi_params)
    parameters.pop("random_seed")
    parameters["n_estimators"] = 250
    parameters["bootstrap"] = False
    single_group = {
        **parameters,
        "weight_mode": "cell-distance",
        "distance_metric": "euclidean",
        "group_size_correction": "none",
    }
    shared_parameters = sorted(
        set(parameters) - {"weight_mode", "distance_metric", "group_size_correction"}
    )
    group_policy = {
        "routing_parameter": "group_count",
        "shared_parameters": shared_parameters,
        "branches": {
            "single_group": {
                "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                "parameters": single_group,
            },
            "multi_group": {
                "condition": {"minimum_group_count": 2},
                "parameters": parameters,
            },
        },
    }
    policy_sha256 = canonical_json_sha256(group_policy)
    identity = locked_spathi_identity(root)
    ranking_path = _write_json(
        root
        / f"work/v1/workflows/calibration/{budget}/calibration-current/stages/validation/ranking.json",
        {
            "stage_id": "validation",
            "dataset_role": "validation",
            "benchmark_commit": "1" * 40,
            "calibration_rows": [
                {"dataset_id": f"selection_dataset_{index}"} for index in range(5)
            ],
        },
    )
    image_id = "sha256:" + "7" * 64
    selected = {
        "schema_version": "1.0",
        "id": "spathi_selected_v1",
        "search_space": "experiments/v1/spathi/search-space.json",
        "calibration_budget": budget,
        "status": "selected",
        "phase": "frozen",
        "selected_policy": {
            "policy_id": "spathi_conditional_v1",
            "source_candidate_id": "selected_spathi",
            "spathi_commit": identity["spathi_commit"],
            "source_configuration_sha256": canonical_json_sha256(parameters),
            "policy_sha256": policy_sha256,
            "group_policy": group_policy,
        },
        "selection_evidence": {
            "validation_ranking": ranking_path.relative_to(root).as_posix(),
            "calibration_budget": budget,
            "validation_ranking_sha256": sha256_file(ranking_path),
            "search_space_sha256": "3" * 64,
            "validation_manifest_sha256": "4" * 64,
            "benchmark_summary_sha256": "5" * 64,
            "spathi_build_fingerprint": identity["spathi_build_fingerprint"],
            "spathi_image_id": image_id,
            "primary_metric": "aupr_directed",
            "hierarchical_score": 0.2,
            "winner_candidate_id": "selected_spathi",
            "winner_configuration_sha256": canonical_json_sha256(parameters),
            "winner_policy_sha256": policy_sha256,
            "tie_breakers_applied": ["stability", "simplicity", "canonical_configuration_id"],
            "frozen_at": "2026-09-04T00:00:00Z",
        },
        "test_accessed_before_freeze": False,
        "notes": ["Test selection."],
    }
    calibration_root = root / f"work/v1/workflows/calibration/{budget}/calibration-current"
    selected_path = _write_json(calibration_root / "selected-config.json", selected)
    _write_json(
        calibration_root / "workflow.json",
        {
            "workflow_type": "calibration",
            "calibration_budget": budget,
            "status": "complete",
            "selected_config": selected_path.relative_to(root).as_posix(),
            "selected_config_sha256": sha256_file(selected_path),
        },
    )
    andrea = tmp_path / "ANDREA"
    spathi = tmp_path / "SPATHI"
    andrea.mkdir()
    spathi.mkdir()
    return root, root / "work", andrea, spathi, image_id


def _independent_panel(root: Path, selected: dict[str, Any]) -> dict[str, Any]:
    ranking_path = root / selected["selection_evidence"]["validation_ranking"]
    return {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_predeclared_panel",
        "selection_validation_manifest_sha256": selected["selection_evidence"][
            "validation_manifest_sha256"
        ],
        "selection_validation_ranking": {
            "path": ranking_path.relative_to(root).as_posix(),
            "sha256": sha256_file(ranking_path),
        },
        "budget": "reduced",
        "dataset_role": "efficiency_validation",
        "split_id": "efficiency_validation_v1",
        "datasets": [
            {
                "dataset_id": f"efficiency_dataset_{index}",
                "simulator_id": simulator,
                "condition_id": condition,
                "generation_seed": 41001 + index,
                "contexts": ["group:a", "group:b"],
                "dataset_artifact": {
                    "kind": "dataset",
                    "digest": canonical_json_sha256({"dataset": index}),
                },
            }
            for index, (simulator, condition) in enumerate(
                (
                    ("dyngen", "linear_baseline"),
                    ("dyngen", "linear_noisy"),
                    ("dyngen", "bifurcating_baseline"),
                    ("dyngen", "bifurcating_noisy"),
                    ("scmultisim", "scmultisim_dynamic"),
                )
            )
        ],
    }


def _fake_development_paths(**kwargs: Any) -> tuple[Path, Path, Path]:
    root = kwargs["workflow_root"] / "efficiency-development"
    run = (
        kwargs["work_root"]
        / "v1"
        / f"efficiency-development-{kwargs['budget']}"
        / kwargs["flow_id"]
    )
    profile = _write_json(root / "profile.json", {"materialized": True})
    selection = _write_json(root / "selection.json", {"selected": True})
    run.mkdir(parents=True, exist_ok=True)
    return run, profile, selection


def test_efficiency_development_executes_the_frozen_two_step_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    work = root / "work"
    workflow_root = work / "v1/workflows/efficiency/reduced/flow"
    selected_path = _write_json(root / "selected.json", {"status": "selected"})
    andrea = root / "ANDREA"
    spathi = root / "SPATHI"
    events: list[tuple[str, str]] = []

    def fake_materialize(
        *,
        repo_root: Path,
        selected_config_path: Path,
        budget: str,
        step: str,
        component_selection_path: Path | None,
        output_dir: Path,
        profile_output: Path,
    ) -> tuple[Path, Path]:
        assert (repo_root, selected_config_path, budget) == (root, selected_path, "reduced")
        if step == "component_scout":
            assert component_selection_path is None
        else:
            assert step == "joint_confirmation"
            assert component_selection_path == workflow_root / (
                "efficiency-development/component-selection.json"
            )
            assert component_selection_path.is_file()
        events.append(("materialize", step))
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(profile_output, {"id": step})
        return output_dir, profile_output

    def fake_run_profile_pipeline(
        *,
        repo_root: Path,
        profile_id: str,
        profile_file: Path,
        run_root: Path,
        andrea_repo: Path,
        spathi_repo: Path,
        statistics: bool,
        runtime_plan: dict[str, Any],
    ) -> dict[str, Any]:
        assert (repo_root, andrea_repo, spathi_repo) == (root, andrea, spathi)
        assert statistics is False
        assert runtime_plan == {"frozen": True}
        assert profile_file.is_file()
        step = "component_scout" if profile_id.endswith("component-scout") else "joint_confirmation"
        events.append(("run", step))
        reports = run_root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "metrics.csv").write_text("metric\n", encoding="utf-8")
        return {"status": "summarized"}

    def fake_select_components(
        *,
        repo_root: Path,
        selected_config_path: Path,
        budget: str,
        scout_metrics_path: Path,
        output_path: Path,
    ) -> Path:
        assert (repo_root, selected_config_path, budget) == (root, selected_path, "reduced")
        assert "component-scout" in scout_metrics_path.as_posix()
        assert scout_metrics_path.is_file()
        events.append(("select", "component_scout"))
        return _write_json(output_path, {"component_nominees": ["eligibility", "adaptive"]})

    def fake_select_development(
        *,
        repo_root: Path,
        selected_config_path: Path,
        budget: str,
        component_selection_path: Path,
        confirmation_metrics_path: Path,
        output_path: Path,
    ) -> Path:
        assert (repo_root, selected_config_path, budget) == (root, selected_path, "reduced")
        assert component_selection_path.is_file()
        assert "joint-confirmation" in confirmation_metrics_path.as_posix()
        assert confirmation_metrics_path.is_file()
        events.append(("select", "joint_confirmation"))
        return _write_json(output_path, {"nominee": "combined"})

    monkeypatch.setattr(workflow, "materialize_efficiency_development", fake_materialize)
    monkeypatch.setattr(workflow, "select_efficiency_components", fake_select_components)
    monkeypatch.setattr(workflow, "select_efficiency_development", fake_select_development)
    monkeypatch.setattr(
        "spathi_bench.workflows.run_profile_pipeline",
        fake_run_profile_pipeline,
    )

    run_path, profile_path, selection_path = workflow._run_efficiency_development(
        repo_root=root,
        andrea_repo=andrea,
        spathi_repo=spathi,
        work_root=work,
        workflow_root=workflow_root,
        flow_id="flow",
        budget="reduced",
        selected_path=selected_path,
        runtime_plan={"frozen": True},
    )

    assert events == [
        ("materialize", "component_scout"),
        ("run", "component_scout"),
        ("select", "component_scout"),
        ("materialize", "joint_confirmation"),
        ("run", "joint_confirmation"),
        ("select", "joint_confirmation"),
    ]
    assert run_path == work / "v1/efficiency-development-reduced-joint-confirmation/flow"
    assert profile_path == workflow_root / (
        "efficiency-development/joint-confirmation/profile.json"
    )
    assert selection_path == workflow_root / "efficiency-development/selection.json"


@pytest.mark.parametrize(
    ("candidate_selected", "expected_generation_profiles"),
    (
        (False, ["efficiency-full"]),
        (True, ["efficiency-full", "efficiency-safety-full"]),
    ),
)
def test_full_safety_generation_is_content_bound_and_strictly_conditional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_spathi_params: dict[str, Any],
    candidate_selected: bool,
    expected_generation_profiles: list[str],
) -> None:
    root, work, andrea, spathi, image_id = _repository(
        tmp_path,
        frozen_spathi_params,
        budget="full",
    )
    generation_profiles: list[str] = []
    safety_job_ids: list[str] = []

    def fake_generate_profile(**kwargs: Any) -> dict[str, Any]:
        generation_profiles.append(kwargs["profile_id"])
        assert kwargs["expected_spathi_image_id"] == image_id
        return {
            "status": "generated",
            "profile_id": kwargs["profile_id"],
            "spathi_image": {"id": image_id},
        }

    def fake_materialize_plan(**kwargs: Any) -> Path:
        selected = json.loads(kwargs["selected_config_path"].read_text(encoding="utf-8"))
        selected_policy = selected["selected_policy"]
        candidate = {
            "id": "candidate_policy",
            "policy_sha256": "6" * 64,
            "group_policy": {"mode": "optimized"},
        }
        return _write_json(
            kwargs["output_path"],
            {
                "budget": {"id": "full", "analysis_role": "confirmatory"},
                "analysis": {
                    "adoption_rule": "lowest_geometric_mean_runtime_ratio_then_candidate_id_among_adoptable"
                },
                "reference": {
                    "policy_id": selected_policy["policy_id"],
                    "policy_sha256": selected_policy["policy_sha256"],
                    "group_policy": selected_policy["group_policy"],
                    "spathi_commit": selected_policy["spathi_commit"],
                },
                "candidates": [candidate],
            },
        )

    def fake_run(**kwargs: Any) -> Path:
        return _write_json(kwargs["artifact_index_path"], {"verified": True})

    def fake_evidence(**kwargs: Any) -> Path:
        return _write_json(kwargs["output_path"], {"verified": True})

    def fake_analyze(**kwargs: Any) -> tuple[Path, Path, Path]:
        plan = json.loads(kwargs["plan_path"].read_text(encoding="utf-8"))
        selected = plan["candidates"][0] if candidate_selected else plan["reference"]
        adoption = {
            "decision": "candidate_adopted" if candidate_selected else "reference_retained",
            "selected_policy_id": (selected["id"] if candidate_selected else selected["policy_id"]),
            "selected_policy_sha256": selected["policy_sha256"],
            "selected_group_policy": selected["group_policy"],
            "selected_candidate_id": selected["id"] if candidate_selected else None,
        }
        output = kwargs["output_dir"]
        return (
            _write_json(
                output / "results.json",
                {"analysis_role": "confirmatory", "adoption": adoption},
            ),
            _write_json(output / "comparisons.csv", {"verified": True}),
            _write_json(output / "manifest.json", {"verified": True}),
        )

    def fake_provisional(*_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        return (
            {
                "id": "candidate_policy",
                "policy_sha256": "6" * 64,
                "group_policy": {"mode": "optimized"},
            }
            if candidate_selected
            else None
        )

    def fake_safety_panel(**kwargs: Any) -> dict[str, Any]:
        jobs = kwargs["profile_payload"]["generation_jobs"]
        safety_job_ids.extend(job["id"] for job in jobs)
        assert len(jobs) == 6
        assert {job["dataset_role"] for job in jobs} == {"efficiency_safety"}
        return {"artifact_type": "spathi_efficiency_safety_panel"}

    def fake_safety_plan(**kwargs: Any) -> Path:
        return _write_json(kwargs["output_path"], {"safety": True})

    def fake_safety_analysis(**kwargs: Any) -> Path:
        return _write_json(
            kwargs["output_path"],
            {
                "decision": {
                    "status": "candidate_authorized",
                    "selected_policy_id": "candidate_policy",
                    "selected_policy_sha256": "6" * 64,
                    "selected_group_policy": {"mode": "optimized"},
                }
            },
        )

    def fake_production(**kwargs: Any) -> dict[str, Any]:
        selected = json.loads(kwargs["selected_path"].read_text(encoding="utf-8"))
        policy = selected["selected_policy"]
        return {
            "schema_version": "1.0",
            "artifact_type": "spathi_production_policy",
            "budget": "full",
            "evidence_role": "confirmatory",
            "status": "confirmed",
            "decision": "candidate_adopted" if candidate_selected else "reference_retained",
            "policy": {
                "policy_id": policy["policy_id"],
                "policy_sha256": policy["policy_sha256"],
                "group_policy": policy["group_policy"],
                "spathi_commit": policy["spathi_commit"],
            },
            "provenance": {},
        }

    def fake_publish(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        state = {
            **kwargs["publishing_state"],
            "status": "complete",
            "reports": {"html": {"path": "report.html", "sha256": "7" * 64}},
        }
        _write_json(kwargs["workflow_path"], state)
        return state, {}

    monkeypatch.setattr(workflow, "_generate_profile", fake_generate_profile)
    monkeypatch.setattr(workflow, "_run_efficiency_development", _fake_development_paths)
    monkeypatch.setattr(
        workflow,
        "_panel_from_generation",
        lambda **kwargs: {
            **_independent_panel(kwargs["repo_root"], kwargs["selected"]),
            "budget": "full",
        },
    )
    monkeypatch.setattr(workflow, "materialize_noninferiority_plan", fake_materialize_plan)
    monkeypatch.setattr(workflow, "run_noninferiority_artifacts", fake_run)
    monkeypatch.setattr(workflow, "write_noninferiority_evidence", fake_evidence)
    monkeypatch.setattr(workflow, "analyze_noninferiority", fake_analyze)
    monkeypatch.setattr(workflow, "provisional_candidate", fake_provisional)
    monkeypatch.setattr(workflow, "_safety_panel_from_generation", fake_safety_panel)
    monkeypatch.setattr(workflow, "materialize_safety_plan", fake_safety_plan)
    monkeypatch.setattr(workflow, "analyze_safety", fake_safety_analysis)
    monkeypatch.setattr(workflow, "_production_config", fake_production)
    monkeypatch.setattr(workflow, "validate_json", lambda *_args: None)
    monkeypatch.setattr(workflow, "validate_calibration_contracts", lambda *_args: ({}, {}))
    monkeypatch.setattr(
        workflow,
        "observe_benchmark_repository",
        lambda *_args, **_kwargs: {
            "path": str(root),
            "commit": "1" * 40,
            "origin": "https://example.test/spathi-benchmarks.git",
            "clean": True,
        },
    )
    monkeypatch.setattr(workflow, "andrea_host_environment", lambda: {"fingerprint": "8" * 64})
    monkeypatch.setattr(workflow, "benchmark_host_environment", lambda: {"fingerprint": "9" * 64})
    monkeypatch.setattr(workflow, "docker_runtime_environment", lambda: {"fingerprint": "a" * 64})
    monkeypatch.setattr(workflow, "publish_workflow_report", fake_publish)

    state = workflow.run_efficiency_workflow(
        repo_root=root,
        andrea_repo=andrea,
        spathi_repo=spathi,
        work_root=work,
        profile="full",
        calibration_run_id="calibration-current",
    )

    assert generation_profiles == expected_generation_profiles
    assert set(state["identity"]["profile_sha256s"]) == {
        "efficiency",
        "efficiency_development",
        "efficiency_safety",
    }
    if candidate_selected:
        assert len(safety_job_ids) == 6
        assert state["efficiency_safety"]["status"] == "completed"
        assert state["efficiency_safety"]["generation_run"].endswith(
            f"efficiency-safety-full/{state['flow_id']}"
        )
    else:
        assert safety_job_ids == []
        assert state["efficiency_safety"] == {
            "status": "not_required",
            "reason": "exact_reference_retained_by_confirmatory_study",
        }


def test_completed_selection_must_belong_to_the_requested_calibration_run(
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
) -> None:
    root, work, _andrea, _spathi, _image_id = _repository(tmp_path, frozen_spathi_params)
    calibration_root = work / "v1/workflows/calibration/reduced/calibration-current"
    selected_path = calibration_root / "selected-config.json"
    foreign_path = work / "v1/workflows/calibration/reduced/foreign-selected-config.json"
    shutil.copyfile(selected_path, foreign_path)
    workflow_path = calibration_root / "workflow.json"
    state = json.loads(workflow_path.read_text(encoding="utf-8"))
    state["selected_config"] = foreign_path.relative_to(root).as_posix()
    state["selected_config_sha256"] = sha256_file(foreign_path)
    _write_json(workflow_path, state)

    with pytest.raises(BenchmarkConfigError, match="selection is missing or changed"):
        workflow._completed_selection(
            work=work,
            repo_root=root,
            budget="reduced",
            calibration_run_id="calibration-current",
        )


@pytest.mark.parametrize("symlink_kind", ["workflow", "selection", "directory"])
def test_completed_selection_rejects_symlinked_workflow_material(
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
    symlink_kind: str,
) -> None:
    root, work, _andrea, _spathi, _image_id = _repository(tmp_path, frozen_spathi_params)
    calibration_root = work / "v1/workflows/calibration/reduced/calibration-current"
    if symlink_kind == "workflow":
        workflow_path = calibration_root / "workflow.json"
        target = calibration_root / "workflow-target.json"
        workflow_path.rename(target)
        workflow_path.symlink_to(target.name)
    elif symlink_kind == "selection":
        selected_path = calibration_root / "selected-config.json"
        target = calibration_root / "selection-target.json"
        selected_path.rename(target)
        selected_path.symlink_to(target.name)
    else:
        target = calibration_root.with_name("calibration-target")
        calibration_root.rename(target)
        calibration_root.symlink_to(target.name, target_is_directory=True)

    with pytest.raises(BenchmarkConfigError, match="must not be a symlink|missing or changed"):
        workflow._completed_selection(
            work=work,
            repo_root=root,
            budget="reduced",
            calibration_run_id="calibration-current",
        )


def test_high_level_efficiency_workflow_is_complete_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_spathi_params: dict[str, Any],
) -> None:
    root, work, andrea, spathi, image_id = _repository(tmp_path, frozen_spathi_params)
    calls: list[str] = []

    def fake_preflight(**_kwargs: Any) -> Path:
        calls.append("preflight")
        return root / "preflight.json"

    def fake_build(**_kwargs: Any) -> Path:
        calls.append("build")
        return root / "image.json"

    def fake_generate(**kwargs: Any) -> Path:
        calls.append("generate")
        return _write_json(
            state_path(kwargs["run_root"]),
            {
                "status": "generated",
                "profile_id": kwargs["profile_id"],
                "spathi_image": {"id": image_id},
            },
        )

    def fake_panel(**kwargs: Any) -> dict[str, Any]:
        calls.append("panel")
        return _independent_panel(kwargs["repo_root"], kwargs["selected"])

    def fake_plan(**kwargs: Any) -> Path:
        selected = json.loads(kwargs["selected_config_path"].read_text(encoding="utf-8"))
        policy = selected["selected_policy"]
        return _write_json(
            kwargs["output_path"],
            {
                "budget": {"id": "reduced", "analysis_role": "descriptive"},
                "analysis": {
                    "adoption_rule": "lowest_geometric_mean_runtime_ratio_then_candidate_id_among_adoptable"
                },
                "reference": {
                    **policy,
                    "spathi_image_id": image_id,
                },
                "candidates": [
                    {"id": component, "component": component}
                    for component in ("target_eligibility", "adaptive_trees", "combined")
                ],
            },
        )

    def fake_run(**kwargs: Any) -> Path:
        calls.append("run")
        execution_plan = kwargs["execution_plan"]
        assert execution_plan["class_id"] == "bulk_full"
        assert execution_plan["output_profile"] == "full"
        assert execution_plan["operations"]["inference"]["cpu_threads"] >= 1
        plan = json.loads(kwargs["plan_path"].read_text(encoding="utf-8"))
        assert plan["reference"]["spathi_image_id"] == image_id
        assert {candidate["component"] for candidate in plan["candidates"]} == {
            "target_eligibility",
            "adaptive_trees",
            "combined",
        }
        return _write_json(kwargs["artifact_index_path"], {"verified": True})

    def fake_evidence(**kwargs: Any) -> Path:
        calls.append("evidence")
        assert kwargs["artifact_index_path"].is_file()
        return _write_json(kwargs["output_path"], {"verified": True})

    def fake_analyze(**kwargs: Any) -> tuple[Path, Path, Path]:
        calls.append("analyze")
        assert kwargs["evidence_path"].is_file()
        output = kwargs["output_dir"]
        plan = json.loads(kwargs["plan_path"].read_text(encoding="utf-8"))
        results = _write_json(
            output / "results.json",
            {
                "analysis_role": "descriptive",
                "adoption": {
                    "decision": "descriptive_only",
                    "selected_policy_id": plan["reference"]["policy_id"],
                    "selected_policy_sha256": plan["reference"]["policy_sha256"],
                    "selected_group_policy": plan["reference"]["group_policy"],
                    "selected_candidate_id": None,
                },
            },
        )
        return (
            results,
            _write_json(output / "comparisons.csv", {"verified": True}),
            _write_json(output / "manifest.json", {"verified": True}),
        )

    def fake_report(**kwargs: Any) -> dict[str, Any]:
        calls.append("report")
        publication = json.loads(
            (kwargs["workflow_root"] / "workflow.json").read_text(encoding="utf-8")
        )
        assert publication["status"] in {"publishing", "complete"}
        assert ("reports" in publication) is (publication["status"] == "complete")
        report_path = (
            kwargs["version_root"]
            / "reports/reduced"
            / kwargs["workflow_root"].name
            / "efficiency.html"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("<!doctype html>\n", encoding="utf-8")
        manifest_path = report_path.parent / "efficiency-report.manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        return {
            "schema_version": "1.0",
            "html": {
                "path": report_path.relative_to(kwargs["repo_root"]).as_posix(),
                "sha256": sha256_file(report_path),
            },
            "manifest": {
                "path": manifest_path.relative_to(kwargs["repo_root"]).as_posix(),
                "sha256": sha256_file(manifest_path),
            },
        }

    monkeypatch.setattr(workflow, "preflight", fake_preflight)
    monkeypatch.setattr(workflow, "build_spathi_image", fake_build)
    monkeypatch.setattr(workflow, "generate_data", fake_generate)
    monkeypatch.setattr(workflow, "_run_efficiency_development", _fake_development_paths)
    monkeypatch.setattr(workflow, "_panel_from_generation", fake_panel)
    monkeypatch.setattr(workflow, "materialize_noninferiority_plan", fake_plan)
    monkeypatch.setattr(workflow, "run_noninferiority_artifacts", fake_run)
    monkeypatch.setattr(workflow, "write_noninferiority_evidence", fake_evidence)
    monkeypatch.setattr(workflow, "analyze_noninferiority", fake_analyze)
    monkeypatch.setattr(workflow, "render_efficiency_report", fake_report)
    monkeypatch.setattr(workflow, "validate_calibration_contracts", lambda *_args: ({}, {}))
    monkeypatch.setattr(
        workflow,
        "observe_benchmark_repository",
        lambda *_args, **_kwargs: {
            "path": str(root),
            "commit": "1" * 40,
            "origin": "https://example.test/spathi-benchmarks.git",
            "clean": True,
        },
    )
    monkeypatch.setattr(workflow, "andrea_host_environment", lambda: {"fingerprint": "8" * 64})
    monkeypatch.setattr(workflow, "benchmark_host_environment", lambda: {"fingerprint": "9" * 64})
    monkeypatch.setattr(workflow, "docker_runtime_environment", lambda: {"fingerprint": "a" * 64})

    arguments = {
        "repo_root": root,
        "andrea_repo": andrea,
        "spathi_repo": spathi,
        "work_root": work,
        "profile": "reduced",
        "calibration_run_id": "calibration-current",
    }
    first = workflow.run_efficiency_workflow(**arguments)
    second = workflow.run_efficiency_workflow(**arguments)

    assert first == second
    assert second["status"] == "complete"
    assert second["reports"]["html"]["path"] == second["interactive_report"]
    assert (
        json.loads(
            (
                root / "work/v1/workflows/efficiency/reduced" / second["flow_id"] / "workflow.json"
            ).read_text(encoding="utf-8")
        )["reports"]
        == second["reports"]
    )
    assert Path(root / second["panel"]).is_file()
    assert Path(root / second["plan"]).is_file()
    assert Path(root / second["artifact_index"]).is_file()
    assert Path(root / second["evidence"]).is_file()
    assert (
        calls
        == [
            "preflight",
            "build",
            "generate",
            "panel",
            "run",
            "evidence",
            "analyze",
            "report",
        ]
        * 2
    )


def test_natural_cli_dispatches_the_complete_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_workflow(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "run_efficiency_workflow", fake_workflow)
    assert (
        cli.main(
            [
                "justify-efficiency",
                "--repo-root",
                str(tmp_path / "repo"),
                "--andrea-repo",
                str(tmp_path / "ANDREA"),
                "--spathi-repo",
                str(tmp_path / "SPATHI"),
                "--profile",
                "reduced",
                "--calibration-run-id",
                "calibration-current",
            ]
        )
        == 0
    )
    assert captured["profile"] == "reduced"
    assert captured["calibration_run_id"] == "calibration-current"
    assert captured["work_root"] == (tmp_path / "repo/work").resolve()


def test_efficiency_workflow_rejects_path_like_calibration_ids(
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
) -> None:
    root, work, andrea, spathi, _image_id = _repository(tmp_path, frozen_spathi_params)
    with pytest.raises(BenchmarkConfigError, match="Invalid calibration run ID"):
        workflow.run_efficiency_workflow(
            repo_root=root,
            andrea_repo=andrea,
            spathi_repo=spathi,
            work_root=work,
            profile="reduced",
            calibration_run_id="../escape",
        )


def test_efficiency_rejects_calibration_from_another_benchmark_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_spathi_params: dict[str, Any],
) -> None:
    root, work, andrea, spathi, _image_id = _repository(tmp_path, frozen_spathi_params)
    monkeypatch.setattr(workflow, "validate_calibration_contracts", lambda *_args: ({}, {}))
    monkeypatch.setattr(
        workflow,
        "observe_benchmark_repository",
        lambda *_args, **_kwargs: {
            "path": str(root),
            "commit": "2" * 40,
            "origin": "https://example.test/spathi-benchmarks.git",
            "clean": True,
        },
    )

    with pytest.raises(BenchmarkConfigError, match="another benchmark revision"):
        workflow.run_efficiency_workflow(
            repo_root=root,
            andrea_repo=andrea,
            spathi_repo=spathi,
            work_root=work,
            profile="reduced",
            calibration_run_id="calibration-current",
        )


def test_execution_order_is_deterministic_and_position_balanced() -> None:
    labels = ["exact", "eligibility", "adaptive", "combined"]
    orders = counterbalanced_execution_orders(
        (f"dataset_{index}" for index in range(5)),
        (1729, 31415),
        labels,
        execution_order_seed=99,
    )
    assert orders == counterbalanced_execution_orders(
        reversed([f"dataset_{index}" for index in range(5)]),
        reversed([1729, 31415]),
        labels,
        execution_order_seed=99,
    )
    position_counts = {
        label: [sum(order[position] == label for order in orders.values()) for position in range(4)]
        for label in labels
    }
    assert all(max(counts) - min(counts) <= 1 for counts in position_counts.values())
    for algorithm_seed in (1729, 31415):
        seed_orders = [
            order for (_dataset_id, seed), order in orders.items() if seed == algorithm_seed
        ]
        seed_position_counts = {
            label: [
                sum(order[position] == label for order in seed_orders) for position in range(4)
            ]
            for label in labels
        }
        assert all(
            max(counts) - min(counts) <= 1 for counts in seed_position_counts.values()
        )
    assert orders != counterbalanced_execution_orders(
        (f"dataset_{index}" for index in range(5)),
        (1729, 31415),
        labels,
        execution_order_seed=100,
    )


def test_reference_only_execution_order_is_valid_and_empty_labels_are_not() -> None:
    assert counterbalanced_execution_orders(
        ("dataset_a", "dataset_b"),
        (1729,),
        ["reference"],
        execution_order_seed=99,
    ) == {
        ("dataset_a", 1729): ["reference"],
        ("dataset_b", 1729): ["reference"],
    }
    with pytest.raises(ValueError, match="at least one unique label"):
        counterbalanced_execution_orders(
            ("dataset_a",),
            (1729,),
            [],
            execution_order_seed=99,
        )
