from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.state import load_state, write_state
from spathi_bench.summarize import summarize


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _evaluated_run(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    evaluation_report = run_root / "evaluation" / "evaluation_report.json"
    run_report = run_root / "inference" / "run_report.json"
    _write_json(
        evaluation_report,
        {
            "metrics": [
                {
                    "tool_id": "genie3_group_emulated",
                    "level": "group",
                    "context": "group:a",
                    "status": "ok",
                    "auroc": 0.8,
                    "aupr": 0.4,
                    "f1_at_truth_count": 0.5,
                    "epr_at_truth_count": 1.2,
                },
                {
                    "tool_id": "genie3_group_emulated",
                    "level": "group",
                    "context": "group:b",
                    "status": "partial",
                    "auroc": 0.6,
                    "aupr": None,
                    "f1_at_truth_count": 0.3,
                    "epr_at_truth_count": 0.8,
                    "diagnostics": {"reason": "small truth"},
                },
                {
                    "tool_id": "custom_spathi_primary",
                    "level": "group",
                    "context": "group:a",
                    "status": "not_evaluable",
                    "auroc": None,
                    "aupr": None,
                    "f1_at_truth_count": None,
                    "epr_at_truth_count": None,
                },
            ]
        },
    )
    _write_json(
        run_report,
        {
            "tools": {
                "tool_origins": {
                    "genie3_group_emulated": "catalog",
                    "custom_spathi_primary": "custom",
                }
            }
        },
    )
    write_state(
        run_root,
        {
            "schema_version": "1.0",
            "profile_id": "smoke",
            "status": "evaluated",
            "datasets": [
                {
                    "dataset_id": "dyngen-1729",
                    "generation_job": "dyngen_smoke",
                    "simulator_id": "dyngen",
                    "seed": 1729,
                }
            ],
            "evaluations": [
                {
                    "dataset_id": "dyngen-1729",
                    "batch_id": "group_only",
                    "regime": "group_only",
                    "evaluation_report": str(evaluation_report),
                    "run_report": str(run_report),
                }
            ],
        },
    )
    return run_root


def test_summarize_preserves_rows_and_computes_descriptive_means(tmp_path: Path) -> None:
    run_root = _evaluated_run(tmp_path)

    metrics_path, summary_path = summarize(profile_id="smoke", run_root=run_root)

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["profile_id"] == "smoke"
    assert rows[0]["generation_job"] == "dyngen_smoke"
    assert rows[0]["regime"] == "group_only"
    assert rows[0]["tool_origin"] == "catalog"
    assert rows[1]["aupr"] == ""
    assert rows[1]["diagnostics"] == '{"reason":"small truth"}'

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metric_rows"] == 3
    aggregates = {
        (item["tool_id"], item["level"], item["regime"]): item for item in summary["aggregates"]
    }
    genie3 = aggregates[("genie3_group_emulated", "group", "group_only")]
    assert genie3["n_datasets"] == 1
    assert genie3["n_metric_rows"] == 2
    assert genie3["mean_auroc"] == pytest.approx(0.7)
    assert genie3["mean_aupr"] == pytest.approx(0.4)
    assert genie3["mean_f1_at_truth_count"] == pytest.approx(0.4)
    assert genie3["mean_epr_at_truth_count"] == pytest.approx(1.0)
    assert ("custom_spathi_primary", "group", "group_only") not in aggregates

    state = load_state(run_root, "smoke")
    assert state["status"] == "completed"
    assert state["reports"] == {
        "metrics_csv": str(metrics_path),
        "summary_json": str(summary_path),
    }


def test_summarize_refuses_incomplete_or_repeated_stage(tmp_path: Path) -> None:
    run_root = tmp_path / "incomplete"
    write_state(
        run_root,
        {
            "schema_version": "1.0",
            "profile_id": "smoke",
            "status": "inferred",
            "evaluations": [],
        },
    )
    with pytest.raises(BenchmarkConfigError, match="requires completed evaluations"):
        summarize(profile_id="smoke", run_root=run_root)

    evaluated = _evaluated_run(tmp_path / "second")
    summarize(profile_id="smoke", run_root=evaluated)
    completed_state = load_state(evaluated, "smoke")
    completed_state["status"] = "evaluated"
    write_state(evaluated, completed_state)
    with pytest.raises(BenchmarkConfigError, match="Reports stage already exists"):
        summarize(profile_id="smoke", run_root=evaluated)


def test_summarize_gives_datasets_equal_weight(tmp_path: Path) -> None:
    run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    second_evaluation = run_root / "evaluation-2" / "evaluation_report.json"
    second_run = run_root / "inference-2" / "run_report.json"
    _write_json(
        second_evaluation,
        {
            "metrics": [
                {
                    "tool_id": "genie3_group_emulated",
                    "level": "group",
                    "context": "group:c",
                    "status": "ok",
                    "auroc": 0.1,
                    "aupr": 0.2,
                    "f1_at_truth_count": 0.1,
                    "epr_at_truth_count": 0.2,
                }
            ]
        },
    )
    _write_json(
        second_run,
        {"tools": {"tool_origins": {"genie3_group_emulated": "catalog"}}},
    )
    state["datasets"].append(
        {
            "dataset_id": "dyngen-1730",
            "generation_job": "dyngen_smoke",
            "simulator_id": "dyngen",
            "seed": 1730,
        }
    )
    state["evaluations"].append(
        {
            "dataset_id": "dyngen-1730",
            "batch_id": "group_only",
            "regime": "group_only",
            "evaluation_report": str(second_evaluation),
            "run_report": str(second_run),
        }
    )
    write_state(run_root, state)

    _metrics_path, summary_path = summarize(profile_id="smoke", run_root=run_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = next(
        item
        for item in summary["aggregates"]
        if item["tool_id"] == "genie3_group_emulated" and item["level"] == "group"
    )
    assert aggregate["n_datasets"] == 2
    assert aggregate["mean_auroc"] == pytest.approx(0.4)


def test_summarize_rejects_metric_provenance_conflicts(tmp_path: Path) -> None:
    run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    report_path = Path(state["evaluations"][0]["evaluation_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["metrics"][0]["tool_origin"] = "custom"
    _write_json(report_path, report)

    with pytest.raises(BenchmarkConfigError, match="conflicts with frozen provenance"):
        summarize(profile_id="smoke", run_root=run_root)


@pytest.mark.parametrize("metrics", [None, {}, ["not-an-object"]])
def test_summarize_rejects_invalid_metric_collections(
    tmp_path: Path,
    metrics: object,
) -> None:
    run_root = _evaluated_run(tmp_path)
    state = load_state(run_root, "smoke")
    report_path = Path(state["evaluations"][0]["evaluation_report"])
    _write_json(report_path, {"metrics": metrics})

    with pytest.raises(BenchmarkConfigError, match="metrics must be an array of objects"):
        summarize(profile_id="smoke", run_root=run_root)
