from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.workflows as workflows
from spathi_bench.artifacts import sha256_file
from spathi_bench.config import BenchmarkConfigError

IDENTITY = {
    "benchmark_commit": "1" * 40,
    "benchmark_origin": "https://example.test/spathi-benchmarks",
    "spathi_commit": "2" * 40,
    "andrea_commit": "3" * 40,
    "search_space_sha256": "4" * 64,
    "fingerprint": "5" * 64,
}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _fake_report_bundle(
    *, repo: Path, version_root: Path, scope: str, flow_id: str, kind: str
) -> dict[str, Any]:
    html_path = version_root / f"reports/{scope}/{flow_id}/{kind}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(f"{kind} report for {flow_id}\n", encoding="utf-8")
    manifest_path = _write_json(
        html_path.parent / f"{kind}-report.manifest.json",
        {"flow_id": flow_id, "report_type": kind},
    )
    return {
        "schema_version": "1.0",
        "html": {
            "path": html_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(html_path),
        },
        "manifest": {
            "path": manifest_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
    }


def _workflow_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    work = repo / "work"
    andrea = tmp_path / "ANDREA"
    spathi = tmp_path / "SPATHI"
    for path in (repo, work, andrea, spathi):
        path.mkdir(parents=True, exist_ok=True)
    return repo, work, andrea, spathi


def _write_efficiency_workflow(
    repo: Path,
    work: Path,
    *,
    budget: str,
    flow_id: str,
    status: str = "complete",
) -> tuple[Path, Path]:
    workflow_root = work / "v1/workflows/efficiency" / budget / flow_id
    production = _write_json(
        workflow_root / "production-config.json",
        {"artifact_type": "spathi_production_policy", "budget": budget},
    )
    state = {
        "schema_version": "1.0",
        "workflow_type": "efficiency",
        "flow_id": flow_id,
        "identity": IDENTITY,
        "profile": budget,
        "status": status,
        "stages": {},
    }
    if status == "complete":
        state.update(
            {
                "production_config": str(production.relative_to(repo)),
                "production_config_sha256": sha256_file(production),
            }
        )
    _write_json(workflow_root / "workflow.json", state)
    return workflow_root, production


def test_default_calibration_flow_identity_is_budget_specific() -> None:
    reduced_identity = workflows._calibration_identity(IDENTITY, "reduced")
    full_identity = workflows._calibration_identity(IDENTITY, "full")

    assert reduced_identity["fingerprint"] != full_identity["fingerprint"]
    assert workflows._flow_id(reduced_identity, None) != workflows._flow_id(full_identity, None)


def test_workflow_identity_includes_the_docker_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    search = tmp_path / "experiments/v1/spathi/search-space.json"
    _write_json(search, {"schema_version": "1.0"})
    monkeypatch.setattr(
        workflows,
        "observe_benchmark_repository",
        lambda _root, require_clean: {
            "commit": "1" * 40,
            "origin": "https://example.test/benchmarks",
        },
    )
    monkeypatch.setattr(
        workflows,
        "load_repo_lock",
        lambda _root: {
            "repositories": {
                "spathi": {"commit": "2" * 40},
                "andrea": {"commit": "3" * 40},
            }
        },
    )
    monkeypatch.setattr(workflows, "andrea_host_environment", lambda: {"fingerprint": "4" * 64})
    monkeypatch.setattr(workflows, "benchmark_host_environment", lambda: {"fingerprint": "5" * 64})
    runtime = {"fingerprint": "6" * 64}
    monkeypatch.setattr(workflows, "docker_runtime_environment", lambda: runtime)

    first = workflows._workflow_identity(tmp_path)
    runtime["fingerprint"] = "7" * 64
    second = workflows._workflow_identity(tmp_path)

    assert first["docker_runtime_environment"] == "6" * 64
    assert second["docker_runtime_environment"] == "7" * 64
    assert first["fingerprint"] != second["fingerprint"]


@pytest.mark.parametrize("fail_first_publication", [False, True])
def test_calibrate_chains_ten_phases_freezes_and_resumes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_first_publication: bool,
) -> None:
    repo, work, andrea, spathi = _workflow_paths(tmp_path)
    monkeypatch.setattr(workflows, "_workflow_identity", lambda _root: IDENTITY)
    materialized: list[tuple[tuple[str, str], Path | None]] = []
    pipelines: list[tuple[str, bool]] = []
    ranked: list[str] = []
    frozen: list[Path] = []

    def fake_materialize(
        *,
        repo_root: Path,
        stage_id: str,
        calibration_budget: str,
        fidelity: str,
        parent_ranking_path: Path | None,
        output_dir: Path,
        profile_output: Path,
    ) -> tuple[Path, Path]:
        assert repo_root == repo
        assert calibration_budget == "reduced"
        materialized.append(((stage_id, fidelity), parent_ranking_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "calibration-manifest.json",
            {"stage_id": stage_id, "fidelity": fidelity},
        )
        if not profile_output.exists():
            _write_json(
                profile_output,
                {"id": workflows.stage_profile_id(stage_id, calibration_budget, fidelity)},
            )
        return output_dir, profile_output

    def fake_pipeline(**kwargs: Any) -> dict[str, Any]:
        profile_id = str(kwargs["profile_id"])
        statistics = bool(kwargs["statistics"])
        pipelines.append((profile_id, statistics))
        run_root = Path(kwargs["run_root"])
        summary = _write_json(
            run_root / "reports/summary.json",
            {"profile_id": profile_id},
        )
        return {
            "run_root": str(run_root.resolve()),
            "summary": str(summary.resolve()),
            "reports": [str(summary.resolve())],
            "statistics": [],
        }

    def fake_rank(
        *,
        repo_root: Path,
        stage_id: str,
        calibration_budget: str,
        fidelity: str,
        manifest_path: Path,
        summary_path: Path,
        output_path: Path,
    ) -> Path:
        assert repo_root == repo
        assert calibration_budget == "reduced"
        assert manifest_path.is_file()
        assert summary_path.is_file()
        ranked.append(f"{stage_id}__{fidelity}")
        if not output_path.exists():
            _write_json(output_path, {"stage_id": stage_id, "fidelity": fidelity})
        return output_path

    def fake_freeze(*, repo_root: Path, ranking_path: Path, output_path: Path) -> Path:
        assert repo_root == repo
        assert json.loads(ranking_path.read_text(encoding="utf-8"))["stage_id"] == "validation"
        frozen.append(ranking_path)
        if not output_path.exists():
            _write_json(output_path, {"status": "selected", "phase": "frozen"})
        return output_path

    monkeypatch.setattr(workflows, "materialize_calibration_stage", fake_materialize)
    monkeypatch.setattr(workflows, "run_profile_pipeline", fake_pipeline)
    monkeypatch.setattr(workflows, "rank_calibration_stage", fake_rank)
    monkeypatch.setattr(workflows, "freeze_calibration_selection", fake_freeze)
    rendered: list[Path] = []

    def fake_report(*, repo_root: Path, version_root: Path, workflow_root: Path) -> dict[str, Any]:
        assert repo_root == repo
        assert version_root == work / "v1"
        assert (workflow_root / "selected-config.json").is_file()
        state = json.loads((workflow_root / "workflow.json").read_text(encoding="utf-8"))
        assert state["status"] in {"publishing", "complete"}
        rendered.append(workflow_root)
        if fail_first_publication and len(rendered) == 1:
            raise BenchmarkConfigError("Simulated calibration report failure")
        return _fake_report_bundle(
            repo=repo,
            version_root=version_root,
            scope="reduced",
            flow_id="calibration-test",
            kind="calibration",
        )

    monkeypatch.setattr(workflows, "render_calibration_report", fake_report)

    workflow_args = {
        "repo_root": repo,
        "andrea_repo": andrea,
        "spathi_repo": spathi,
        "work_root": work,
        "calibration_budget": "reduced",
        "run_id": "calibration-test",
    }
    if fail_first_publication:
        with pytest.raises(BenchmarkConfigError, match="Simulated calibration report failure"):
            workflows.run_calibration_workflow(**workflow_args)
        state = json.loads((rendered[0] / "workflow.json").read_text(encoding="utf-8"))
        assert state["status"] == "publishing"
        assert state["selected_config_sha256"] == sha256_file(repo / state["selected_config"])

    first = workflows.run_calibration_workflow(**workflow_args)
    second = workflows.run_calibration_workflow(**workflow_args)

    expected_phases = list(workflows.CALIBRATION_PHASES)
    phase_ids = [f"{stage}__{fidelity}" for stage, fidelity in expected_phases]
    stage_count = len(expected_phases)
    assert [phase for phase, _parent in materialized[:stage_count]] == expected_phases
    assert materialized[0][1] is None
    for index, (_stage, parent) in enumerate(materialized[1:stage_count], start=1):
        assert parent is not None
        assert parent.parent.name == phase_ids[index - 1]
        assert parent.name == "ranking.json"
    assert pipelines[:stage_count] == [
        (workflows.stage_profile_id(stage, "reduced", fidelity), False)
        for stage, fidelity in expected_phases
    ]
    assert ranked[:stage_count] == phase_ids
    assert len(frozen) == 1
    assert first == second
    assert second["status"] == "complete"
    assert second["calibration_budget"] == "reduced"
    assert set(second["stages"]) == set(phase_ids)
    selected = repo / second["selected_config"]
    assert selected.is_file()
    assert second["selected_config_sha256"] == sha256_file(selected)
    assert len(materialized) == len(pipelines) == len(ranked) == stage_count
    assert len(rendered) == 2 + int(fail_first_publication)
    assert second["reports"]["html"]["path"] == (
        "work/v1/reports/reduced/calibration-test/calibration.html"
    )
    assert second["reports"]["manifest"]["path"] == (
        "work/v1/reports/reduced/calibration-test/calibration-report.manifest.json"
    )
    assert second["interactive_report"] == second["reports"]["html"]["path"]


def test_compare_requires_a_completed_matching_efficiency_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, work, andrea, spathi = _workflow_paths(tmp_path)
    monkeypatch.setattr(workflows, "_workflow_identity", lambda _root: IDENTITY)
    _write_efficiency_workflow(
        repo, work, budget="full", flow_id="incomplete-efficiency", status="running"
    )

    with pytest.raises(BenchmarkConfigError, match="No completed full efficiency study"):
        workflows.run_comparison_workflow(
            repo_root=repo,
            andrea_repo=andrea,
            spathi_repo=spathi,
            work_root=work,
            efficiency_run_id="incomplete-efficiency",
        )


def test_compare_explains_when_efficiency_study_has_not_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, work, andrea, spathi = _workflow_paths(tmp_path)
    monkeypatch.setattr(workflows, "_workflow_identity", lambda _root: IDENTITY)

    with pytest.raises(
        BenchmarkConfigError,
        match=r"run spathi-bench run full to complete the study's efficiency phase first",
    ):
        workflows.run_comparison_workflow(
            repo_root=repo,
            andrea_repo=andrea,
            spathi_repo=spathi,
            work_root=work,
        )


def test_compare_never_reuses_a_different_efficiency_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, work, andrea, spathi = _workflow_paths(tmp_path)
    monkeypatch.setattr(workflows, "_workflow_identity", lambda _root: IDENTITY)
    efficiency_id = "shared-name"
    _write_efficiency_workflow(repo, work, budget="reduced", flow_id=efficiency_id)

    with pytest.raises(BenchmarkConfigError, match="No completed full efficiency study"):
        workflows.run_comparison_workflow(
            repo_root=repo,
            andrea_repo=andrea,
            spathi_repo=spathi,
            work_root=work,
            profile="full",
            efficiency_run_id=efficiency_id,
        )


def test_compare_defaults_to_full_materializes_and_runs_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, work, andrea, spathi = _workflow_paths(tmp_path)
    monkeypatch.setattr(workflows, "_workflow_identity", lambda _root: IDENTITY)
    efficiency_id = "complete-efficiency"
    _efficiency_root, production = _write_efficiency_workflow(
        repo,
        work,
        budget="full",
        flow_id=efficiency_id,
    )
    monkeypatch.setattr(
        workflows,
        "completed_production_config",
        lambda **_kwargs: (production, {"budget": "full"}),
    )
    materializations: list[dict[str, Any]] = []

    def fake_benchmark_materialization(**kwargs: Any) -> tuple[Path, Path]:
        materializations.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        profile_output = Path(kwargs["profile_output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "benchmark-manifest.json", {"profile_id": "full-v1"})
        if not profile_output.exists():
            _write_json(profile_output, {"id": "full-v1"})
        return output_dir, profile_output

    monkeypatch.setattr(workflows, "materialize_benchmark_plan", fake_benchmark_materialization)
    pipeline_calls: list[str] = []

    def stage(name: str):
        def fake_stage(**_kwargs: Any) -> Path:
            pipeline_calls.append(name)
            return tmp_path / name

        return fake_stage

    for name in (
        "preflight",
        "build_spathi_image",
        "generate_data",
        "run_inference",
        "validate_handoffs",
        "integrate_inference",
        "evaluate",
        "export_analysis",
    ):
        monkeypatch.setattr(workflows, name, stage(name))

    def fake_summarize(**kwargs: Any) -> tuple[Path, Path]:
        pipeline_calls.append("summarize")
        reports = Path(kwargs["run_root"]) / "reports"
        return (
            _write_json(reports / "metrics.json", {"status": "complete"}),
            _write_json(reports / "summary.json", {"status": "complete"}),
        )

    def fake_statistics(**kwargs: Any) -> tuple[Path]:
        pipeline_calls.append("statistics")
        return (
            _write_json(
                Path(kwargs["run_root"]) / "statistics.json",
                {
                    "contract": {"status": "inferential"},
                    "analyses": [{"analysis_role": "confirmatory", "status": "tested"}],
                },
            ),
        )

    monkeypatch.setattr(workflows, "summarize", fake_summarize)
    monkeypatch.setattr(workflows, "analyze_statistics", fake_statistics)
    rendered: list[Path] = []

    def fake_report(*, repo_root: Path, version_root: Path, workflow_root: Path) -> dict[str, Any]:
        assert repo_root == repo
        assert version_root == work / "v1"
        state = json.loads((workflow_root / "workflow.json").read_text(encoding="utf-8"))
        assert state["status"] == "publishing"
        rendered.append(workflow_root)
        return _fake_report_bundle(
            repo=repo,
            version_root=version_root,
            scope="full",
            flow_id=workflow_root.name,
            kind="comparison",
        )

    monkeypatch.setattr(workflows, "render_comparison_report", fake_report)

    result = workflows.run_comparison_workflow(
        repo_root=repo,
        andrea_repo=andrea,
        spathi_repo=spathi,
        work_root=work,
        efficiency_run_id=efficiency_id,
    )

    assert result["status"] == "complete"
    assert result["profile"] == "full"
    assert result["efficiency_budget"] == "full"
    assert result["efficiency_flow_id"] == efficiency_id
    assert len(materializations) == 1
    request = materializations[0]
    assert request["template_profile_id"] == "full"
    assert request["baseline_plan_path"] == (repo / "experiments/v1/inference/plans/full-v1.json")
    assert request["production_config_path"] == production
    assert Path(request["profile_output"]).name == "full-v1.json"
    assert pipeline_calls == [
        "preflight",
        "build_spathi_image",
        "generate_data",
        "run_inference",
        "validate_handoffs",
        "integrate_inference",
        "evaluate",
        "export_analysis",
        "summarize",
        "statistics",
    ]
    assert len(rendered) == 1
    assert result["reports"]["html"]["path"] == (
        "work/v1/reports/full/full-complete-efficiency-5555555555/comparison.html"
    )
    assert result["reports"]["manifest"]["path"].endswith("/comparison-report.manifest.json")
    assert result["interactive_report"] == result["reports"]["html"]["path"]
    assert "interactive_report" not in result["stages"]["comparison"]["pipeline"]


def test_profile_pipeline_rejects_an_incomplete_confirmatory_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "preflight",
        "build_spathi_image",
        "generate_data",
        "run_inference",
        "validate_handoffs",
        "integrate_inference",
        "evaluate",
        "export_analysis",
    ):
        monkeypatch.setattr(workflows, name, lambda **_kwargs: tmp_path / "stage")
    monkeypatch.setattr(workflows, "summarize", lambda **_kwargs: ())
    report = _write_json(
        tmp_path / "statistics.json",
        {
            "contract": {"status": "inferential"},
            "analyses": [{"analysis_role": "confirmatory", "status": "incomplete_panel"}],
        },
    )
    monkeypatch.setattr(workflows, "analyze_statistics", lambda **_kwargs: (report,))

    with pytest.raises(BenchmarkConfigError, match="confirmatory primary analysis"):
        workflows.run_profile_pipeline(
            repo_root=tmp_path,
            profile_id="full-v1",
            profile_file=tmp_path / "full-v1.json",
            run_root=tmp_path / "work/v1/full-v1/run",
            andrea_repo=tmp_path / "ANDREA",
            spathi_repo=tmp_path / "SPATHI",
            statistics=True,
        )


def test_profile_pipeline_accepts_a_complete_descriptive_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "preflight",
        "build_spathi_image",
        "generate_data",
        "run_inference",
        "validate_handoffs",
        "integrate_inference",
        "evaluate",
        "export_analysis",
    ):
        monkeypatch.setattr(workflows, name, lambda **_kwargs: tmp_path / "stage")
    monkeypatch.setattr(workflows, "summarize", lambda **_kwargs: ())
    report = _write_json(
        tmp_path / "descriptive-analysis.json",
        {
            "contract": {"status": "descriptive"},
            "status": "descriptive_only",
            "analyses": [{"analysis_role": "descriptive", "status": "descriptive_only"}],
        },
    )
    monkeypatch.setattr(workflows, "analyze_statistics", lambda **_kwargs: (report,))

    result = workflows.run_profile_pipeline(
        repo_root=tmp_path,
        profile_id="reduced-v1",
        profile_file=tmp_path / "reduced-v1.json",
        run_root=tmp_path / "work/v1/reduced-v1/run",
        andrea_repo=tmp_path / "ANDREA",
        spathi_repo=tmp_path / "SPATHI",
        statistics=True,
    )

    assert result["statistics"] == [str(report.resolve())]
