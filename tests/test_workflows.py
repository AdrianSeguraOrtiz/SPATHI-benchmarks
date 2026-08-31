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
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _workflow_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    work = repo / "work"
    andrea = tmp_path / "ANDREA"
    spathi = tmp_path / "SPATHI"
    for path in (repo, work, andrea, spathi):
        path.mkdir(parents=True, exist_ok=True)
    return repo, work, andrea, spathi


def test_default_calibration_flow_identity_is_budget_specific() -> None:
    reduced_identity = workflows._calibration_identity(IDENTITY, "reduced")
    full_identity = workflows._calibration_identity(IDENTITY, "full")

    assert reduced_identity["fingerprint"] != full_identity["fingerprint"]
    assert workflows._flow_id(reduced_identity, None) != workflows._flow_id(full_identity, None)


def test_calibrate_chains_four_stages_freezes_and_resumes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, work, andrea, spathi = _workflow_paths(tmp_path)
    monkeypatch.setattr(workflows, "_workflow_identity", lambda _root: IDENTITY)
    materialized: list[tuple[str, Path | None]] = []
    pipelines: list[tuple[str, bool]] = []
    ranked: list[str] = []
    frozen: list[Path] = []

    def fake_materialize(
        *,
        repo_root: Path,
        stage_id: str,
        calibration_budget: str,
        parent_ranking_path: Path | None,
        output_dir: Path,
        profile_output: Path,
    ) -> tuple[Path, Path]:
        assert repo_root == repo
        assert calibration_budget == "reduced"
        materialized.append((stage_id, parent_ranking_path))
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "calibration-manifest.json",
            {"stage_id": stage_id},
        )
        if not profile_output.exists():
            _write_json(
                profile_output,
                {"id": workflows.stage_profile_id(stage_id, calibration_budget)},
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
        manifest_path: Path,
        summary_path: Path,
        output_path: Path,
    ) -> Path:
        assert repo_root == repo
        assert calibration_budget == "reduced"
        assert manifest_path.is_file()
        assert summary_path.is_file()
        ranked.append(stage_id)
        if not output_path.exists():
            _write_json(output_path, {"stage_id": stage_id})
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

    first = workflows.run_calibration_workflow(
        repo_root=repo,
        andrea_repo=andrea,
        spathi_repo=spathi,
        work_root=work,
        calibration_budget="reduced",
        run_id="calibration-test",
    )
    second = workflows.run_calibration_workflow(
        repo_root=repo,
        andrea_repo=andrea,
        spathi_repo=spathi,
        work_root=work,
        calibration_budget="reduced",
        run_id="calibration-test",
    )

    expected_stages = list(workflows.CALIBRATION_STAGES)
    assert [stage for stage, _parent in materialized[:4]] == expected_stages
    assert materialized[0][1] is None
    for index, (_stage, parent) in enumerate(materialized[1:4], start=1):
        assert parent is not None
        assert parent.parent.name == expected_stages[index - 1]
        assert parent.name == "ranking.json"
    assert pipelines[:4] == [
        (workflows.stage_profile_id(stage, "reduced"), False) for stage in expected_stages
    ]
    assert ranked[:4] == expected_stages
    assert len(frozen) == 2
    assert first == second
    assert second["status"] == "complete"
    assert second["calibration_budget"] == "reduced"
    assert set(second["stages"]) == set(expected_stages)
    selected = repo / second["selected_config"]
    assert selected.is_file()
    assert second["selected_config_sha256"] == sha256_file(selected)
    assert len(materialized) == len(pipelines) == len(ranked) == 8


def test_compare_requires_a_completed_matching_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, work, andrea, spathi = _workflow_paths(tmp_path)
    monkeypatch.setattr(workflows, "_workflow_identity", lambda _root: IDENTITY)
    calibration_root = work / "v1/workflows/calibration/full/incomplete-calibration"
    calibration_identity = workflows._calibration_identity(IDENTITY, "full")
    _write_json(
        calibration_root / "workflow.json",
        {
            "schema_version": "1.0",
            "workflow_type": "calibration",
            "flow_id": "incomplete-calibration",
            "identity": calibration_identity,
            "calibration_budget": "full",
            "status": "running",
            "stages": {},
        },
    )

    with pytest.raises(BenchmarkConfigError, match="No completed full calibration matches"):
        workflows.run_comparison_workflow(
            repo_root=repo,
            andrea_repo=andrea,
            spathi_repo=spathi,
            work_root=work,
            calibration_run_id="incomplete-calibration",
        )


def test_compare_explains_when_calibration_has_not_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, work, andrea, spathi = _workflow_paths(tmp_path)
    monkeypatch.setattr(workflows, "_workflow_identity", lambda _root: IDENTITY)

    with pytest.raises(
        BenchmarkConfigError,
        match=r"run scripts/calibrate\.sh --profile full first",
    ):
        workflows.run_comparison_workflow(
            repo_root=repo,
            andrea_repo=andrea,
            spathi_repo=spathi,
            work_root=work,
        )


def test_compare_never_reuses_a_different_calibration_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, work, andrea, spathi = _workflow_paths(tmp_path)
    monkeypatch.setattr(workflows, "_workflow_identity", lambda _root: IDENTITY)
    calibration_id = "shared-name"
    calibration_root = work / "v1/workflows/calibration/reduced" / calibration_id
    selected = _write_json(calibration_root / "selected-config.json", {"status": "selected"})
    _write_json(
        calibration_root / "workflow.json",
        {
            "schema_version": "1.0",
            "workflow_type": "calibration",
            "flow_id": calibration_id,
            "identity": workflows._calibration_identity(IDENTITY, "reduced"),
            "calibration_budget": "reduced",
            "status": "complete",
            "stages": {},
            "selected_config": str(selected.relative_to(repo)),
            "selected_config_sha256": sha256_file(selected),
        },
    )

    with pytest.raises(BenchmarkConfigError, match="No completed full calibration matches"):
        workflows.run_comparison_workflow(
            repo_root=repo,
            andrea_repo=andrea,
            spathi_repo=spathi,
            work_root=work,
            profile="full",
            calibration_run_id=calibration_id,
        )


def test_compare_defaults_to_full_materializes_and_runs_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, work, andrea, spathi = _workflow_paths(tmp_path)
    monkeypatch.setattr(workflows, "_workflow_identity", lambda _root: IDENTITY)
    calibration_id = "complete-calibration"
    calibration_root = work / "v1/workflows/calibration/full" / calibration_id
    calibration_identity = workflows._calibration_identity(IDENTITY, "full")
    selected = _write_json(
        calibration_root / "selected-config.json",
        {"status": "selected", "phase": "frozen"},
    )
    _write_json(
        calibration_root / "workflow.json",
        {
            "schema_version": "1.0",
            "workflow_type": "calibration",
            "flow_id": calibration_id,
            "identity": calibration_identity,
            "calibration_budget": "full",
            "status": "complete",
            "stages": {},
            "selected_config": str(selected.relative_to(repo)),
            "selected_config_sha256": sha256_file(selected),
        },
    )
    monkeypatch.setattr(
        workflows,
        "validate_calibration_contracts",
        lambda repo_root, selected_path: (
            {},
            {"status": "selected", "calibration_budget": "full"},
        ),
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
        return (_write_json(Path(kwargs["run_root"]) / "statistics.json", {}),)

    monkeypatch.setattr(workflows, "summarize", fake_summarize)
    monkeypatch.setattr(workflows, "analyze_statistics", fake_statistics)

    result = workflows.run_comparison_workflow(
        repo_root=repo,
        andrea_repo=andrea,
        spathi_repo=spathi,
        work_root=work,
        calibration_run_id=calibration_id,
    )

    assert result["status"] == "complete"
    assert result["profile"] == "full"
    assert result["calibration_budget"] == "full"
    assert len(materializations) == 1
    request = materializations[0]
    assert request["template_profile_id"] == "full"
    assert request["baseline_plan_path"] == (repo / "experiments/v1/inference/plans/full-v1.json")
    assert request["selected_config_path"] == selected
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
