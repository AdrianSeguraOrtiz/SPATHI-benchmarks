from __future__ import annotations

from pathlib import Path
from typing import Any

import spathi_bench.study_workflow as study


def test_complete_study_carries_exact_upstream_ids_without_discovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    def workflow(name: str, flow_id: str):
        def run(**kwargs: Any) -> dict[str, str]:
            events.append((name, kwargs))
            return {"flow_id": flow_id, "status": "complete"}

        return run

    monkeypatch.setattr(study, "run_calibration_workflow", workflow("calibration", "cal-123"))
    monkeypatch.setattr(study, "run_efficiency_workflow", workflow("efficiency", "eff-456"))
    monkeypatch.setattr(
        study,
        "run_dimensional_transfer_workflow",
        workflow("dimensional_transfer", "dim-789"),
    )
    monkeypatch.setattr(study, "run_comparison_workflow", workflow("comparison", "cmp-789"))
    monkeypatch.setattr(study, "run_scalability_workflow", workflow("scalability", "scale-789"))
    monkeypatch.setattr(study, "run_performance_workflow", workflow("performance", "perf-789"))

    result = study.run_study_workflow(
        repo_root=tmp_path / "repo",
        andrea_repo=tmp_path / "ANDREA",
        spathi_repo=tmp_path / "SPATHI",
        work_root=tmp_path / "repo/work",
        profile="full",
        max_cores=60,
        max_ram_gb=900.0,
        reserve_cores=4,
        reserve_ram_gb=100.0,
    )

    assert [name for name, _kwargs in events] == [
        "calibration",
        "efficiency",
        "dimensional_transfer",
        "comparison",
        "scalability",
        "performance",
    ]
    calibration = events[0][1]
    assert calibration["calibration_budget"] == "full"
    assert calibration["run_id"] is None
    efficiency = events[1][1]
    assert efficiency["calibration_run_id"] == "cal-123"
    for _name, arguments in events[2:]:
        assert arguments["efficiency_run_id"] == "eff-456"
    for _name, arguments in events:
        assert arguments["max_cores"] == 60
        assert arguments["max_ram_gb"] == 900.0
        assert arguments["reserve_cores"] == 4
        assert arguments["reserve_ram_gb"] == 100.0
    assert result == {
        "schema_version": "1.0",
        "workflow_type": "study",
        "profile": "full",
        "status": "complete",
        "calibration_run_id": "cal-123",
        "efficiency_run_id": "eff-456",
        "phases": {
            "calibration": {"flow_id": "cal-123", "status": "complete"},
            "efficiency": {"flow_id": "eff-456", "status": "complete"},
            "dimensional_transfer": {"flow_id": "dim-789", "status": "complete"},
            "comparison": {"flow_id": "cmp-789", "status": "complete"},
            "scalability": {"flow_id": "scale-789", "status": "complete"},
            "performance": {"flow_id": "perf-789", "status": "complete"},
        },
    }
