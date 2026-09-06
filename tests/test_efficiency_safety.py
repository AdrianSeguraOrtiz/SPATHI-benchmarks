from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from spathi_bench.config import BenchmarkConfigError, load_json_object
from spathi_bench.efficiency_safety import (
    SAFETY_CONDITION_SCALES,
    SAFETY_CONDITIONS,
    SAFETY_DATASETS_PER_CONDITION,
    SAFETY_EXPECTED_DATASETS,
    SAFETY_THRESHOLDS,
    _validate_panel_dataset_identities,
    analyze_safety,
    evaluate_safety_pairs,
)
from spathi_bench.scenario_inputs import scenario_external_input_records
from spathi_bench.seeds import generation_seed_for_consumer, load_seed_plan, plan_seed_record

REPO_ROOT = Path(__file__).resolve().parents[1]

RUN_FILES = {
    "g3000_tf100_c300": "g3000-tf100-c300-runs.json",
    "g3000_tf1500_c300": "g3000-tf1500-c300-runs.json",
    "g15000_tf100_c300": "g15000-tf100-c300-runs.json",
    "g100_tf10_c500_single_group": "g100-tf10-c500-single-group-runs.json",
    "g100_tf10_c300_groups_10_40_250": "g100-tf10-c300-groups-10-40-250-runs.json",
    "g100_tf10_c10000": "g100-tf10-c10000-runs.json",
}


def _thresholds() -> dict[str, object]:
    return {
        **SAFETY_THRESHOLDS,
        "minimum_each_dataset_target_fraction": 0.5,
        "minimum_each_dataset_truth_edge_fraction": 0.9,
        "basis": "test_guardrail",
    }


def _pairs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition_index, condition_id in enumerate(SAFETY_CONDITIONS):
        for replicate, (difference, ratio) in enumerate(
            ((-0.001, 0.8), (0.0, 0.9), (0.001, 1.1)), start=1
        ):
            rows.append(
                {
                    "dataset_id": f"safety_{condition_index}_{replicate}",
                    "condition_id": condition_id,
                    "aupr_difference": difference,
                    "target_fraction": 0.8,
                    "truth_edge_fraction": 0.95,
                    "reference_complete": True,
                    "candidate_complete": True,
                    "wall_time_ratio": ratio,
                }
            )
    return rows


def _schema_condition_scales() -> dict[str, dict[str, object]]:
    return {
        condition_id: {
            "generation_job": f"safety_{condition_id}",
            "simulator": "scmultisim",
            **dimensions,
            "scenario": {"path": f"inputs/{condition_id}/scenario.json", "sha256": "1" * 64},
            "simulator_runs": {
                "path": f"inputs/{condition_id}/runs.json",
                "sha256": "2" * 64,
            },
            "scenario_inputs": {
                "regulatory_network": {
                    "declared_path": "network.tsv",
                    "kind": "file",
                    "files": [
                        {
                            "path": f"inputs/{condition_id}/network.tsv",
                            "sha256": "3" * 64,
                        }
                    ],
                },
                "tree_newick": {
                    "declared_path": "tree.nwk",
                    "kind": "file",
                    "files": [
                        {
                            "path": f"inputs/{condition_id}/tree.nwk",
                            "sha256": "4" * 64,
                        }
                    ],
                },
            },
        }
        for condition_id, dimensions in SAFETY_CONDITION_SCALES.items()
    }


def test_safety_veto_is_descriptive_strict_and_reports_global_and_condition_cost() -> None:
    summary, gates, passed = evaluate_safety_pairs(_pairs(), _thresholds())

    assert passed
    assert all(gates.values())
    assert summary["global"]["completed_dataset_pairs"] == SAFETY_EXPECTED_DATASETS
    assert summary["global"]["median_wall_time_ratio"] == pytest.approx(0.9)
    assert summary["global"]["geometric_mean_wall_time_ratio"] == pytest.approx(
        (0.8 * 0.9 * 1.1) ** (1 / 3)
    )
    assert set(summary["by_condition"]) == set(SAFETY_CONDITIONS)
    assert all(
        item["dataset_pairs"] == SAFETY_DATASETS_PER_CONDITION
        for item in summary["by_condition"].values()
    )
    assert "confidence" not in json.dumps(summary)
    assert "p_value" not in json.dumps(summary)

    no_speedup = _pairs()
    for row in no_speedup:
        row["wall_time_ratio"] = 1.0
    _summary, strict_gates, strict_passed = evaluate_safety_pairs(no_speedup, _thresholds())
    assert strict_gates["global_median_wall_time_ratio"] is False
    assert strict_passed is False


@pytest.mark.parametrize(
    ("mutation", "failed_gate"),
    (
        (
            lambda rows: rows[0].update(candidate_complete=False),
            "completion_18_of_18",
        ),
        (
            lambda rows: rows[0].update(candidate_complete=False),
            "completion_each_condition_3_of_3",
        ),
        (
            lambda rows: rows[1].update(target_fraction=0.49),
            "target_coverage_each_dataset",
        ),
        (
            lambda rows: rows[1].update(truth_edge_fraction=0.89),
            "truth_edge_coverage_each_dataset",
        ),
        (
            lambda rows: [row.update(aupr_difference=-0.003) for row in rows],
            "global_mean_aupr_difference",
        ),
        (
            lambda rows: [row.update(aupr_difference=-0.003) for row in rows],
            "global_median_aupr_difference",
        ),
        (
            lambda rows: [
                row.update(aupr_difference=-0.003)
                for row in rows
                if row["condition_id"] == SAFETY_CONDITIONS[0]
            ],
            "condition_mean_aupr_difference",
        ),
        (
            lambda rows: [
                row.update(aupr_difference=-0.003)
                for row in rows
                if row["condition_id"] == SAFETY_CONDITIONS[0]
            ],
            "condition_median_aupr_difference",
        ),
        (
            lambda rows: rows[0].update(aupr_difference=-0.011),
            "each_dataset_aupr_difference",
        ),
    ),
)
def test_every_efficiency_safety_accuracy_or_completion_guardrail_can_veto(
    mutation: Any, failed_gate: str
) -> None:
    rows = _pairs()
    mutation(rows)

    _summary, gates, passed = evaluate_safety_pairs(rows, _thresholds())

    assert gates[failed_gate] is False
    assert passed is False


def test_safety_veto_rejects_incomplete_condition_product_and_non_finite_pairs() -> None:
    duplicate = _pairs()
    duplicate[-1]["dataset_id"] = duplicate[-2]["dataset_id"]
    with pytest.raises(BenchmarkConfigError, match="distinct dataset pairs"):
        evaluate_safety_pairs(duplicate, _thresholds())

    wrong_condition = _pairs()
    wrong_condition[-1]["condition_id"] = SAFETY_CONDITIONS[0]
    with pytest.raises(BenchmarkConfigError, match="condition-by-replicate"):
        evaluate_safety_pairs(wrong_condition, _thresholds())

    non_finite = _pairs()
    non_finite[0]["wall_time_ratio"] = float("nan")
    with pytest.raises(BenchmarkConfigError, match="non-finite"):
        evaluate_safety_pairs(non_finite, _thresholds())


@pytest.mark.parametrize(
    ("field", "message"),
    (("dataset_id", "dataset ID"), ("generation_seed", "generation seed")),
)
def test_safety_plan_requires_separately_unique_dataset_ids_and_generation_seeds(
    field: str,
    message: str,
) -> None:
    datasets = [
        {
            "dataset_id": row["dataset_id"],
            "generation_seed": index + 1,
        }
        for index, row in enumerate(_pairs())
    ]
    datasets[-1][field] = datasets[0][field]

    with pytest.raises(BenchmarkConfigError, match=message):
        _validate_panel_dataset_identities({"datasets": datasets})


def test_committed_safety_design_has_exact_seed_and_six_condition_contract() -> None:
    for name in (
        "efficiency-safety-panel.schema.json",
        "efficiency-safety-plan.schema.json",
        "efficiency-safety-results.schema.json",
    ):
        Draft202012Validator.check_schema(load_json_object(REPO_ROOT / "schemas" / name))

    seed_plan = load_seed_plan(REPO_ROOT / "experiments/v1/seeds/seed-plan.json")
    profile = load_json_object(REPO_ROOT / "experiments/v1/profiles/efficiency-safety-full.json")
    jobs = {job["condition_id"]: job for job in profile["generation_jobs"]}
    assert set(jobs) == set(SAFETY_CONDITIONS)
    scenario_inputs = {
        condition_id: scenario_external_input_records(
            REPO_ROOT / job["scenario"],
            REPO_ROOT,
        )
        for condition_id, job in jobs.items()
    }
    assert all(
        set(inputs) == {"regulatory_network", "tree_newick"} for inputs in scenario_inputs.values()
    )
    assert (
        len(
            {
                inputs["regulatory_network"]["files"][0]["path"]
                for inputs in scenario_inputs.values()
            }
        )
        == 4
    )
    assert (
        len({inputs["tree_newick"]["files"][0]["path"] for inputs in scenario_inputs.values()}) == 2
    )
    observed_scales: dict[str, dict[str, object]] = {}
    for condition_id, filename in RUN_FILES.items():
        relative = f"experiments/v1/simulations/scmultisim/efficiency_safety/{filename}"
        runs = load_json_object(REPO_ROOT / relative)["runs"]
        assert len(runs) == 1
        run = runs[0]
        assert run["run_id"] == f"scmultisim_efficiency_safety_{condition_id}"
        assert run["replicates"] == SAFETY_DATASETS_PER_CONDITION
        assert run["base_seed"] == generation_seed_for_consumer(
            seed_plan,
            path=relative,
            run_id=run["run_id"],
        )
        params = run["params"]
        observed_scales[condition_id] = {
            "genes": params["num_genes"],
            "transcription_factors": SAFETY_CONDITION_SCALES[condition_id]["transcription_factors"],
            "cells": params["num_cells"],
            "group_sizes": params["discrete_population"]["pop_size"],
            "target_list": None,
        }
    assert observed_scales == SAFETY_CONDITION_SCALES
    algorithm = plan_seed_record(
        seed_plan,
        namespace="full.efficiency_safety",
        stream="inference",
        coordinates={"purpose": "common_algorithm_seed"},
    )
    assert algorithm["seed"] == 2065706725


def test_safety_threshold_input_is_not_mutated() -> None:
    thresholds = _thresholds()
    before = deepcopy(thresholds)
    evaluate_safety_pairs(_pairs(), thresholds)
    assert thresholds == before


def test_bound_execution_failure_is_an_auditable_fail_safe_veto(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    schema_dir = root / "schemas"
    schema_dir.mkdir(parents=True)
    (schema_dir / "efficiency-safety-results.schema.json").write_text(
        (REPO_ROOT / "schemas/efficiency-safety-results.schema.json").read_text(),
        encoding="utf-8",
    )
    plan = {
        "budget": {"algorithm_seeds": [11]},
        "analysis": {
            "thresholds": _thresholds(),
            "condition_scales": _schema_condition_scales(),
        },
        "reference": {
            "policy_id": "exact",
            "policy_sha256": "b" * 64,
            "group_policy": {"mode": "exact"},
        },
        "candidates": [
            {
                "id": "provisional",
                "policy_sha256": "a" * 64,
                "group_policy": {"mode": "optimized"},
            }
        ],
        "interpretation_limit": "Descriptive safety veto only.",
    }
    plan_path = root / "work/plan.json"
    evidence_path = root / "work/evidence.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    evidence_path.write_text("{}\n", encoding="utf-8")
    failure = {
        "dataset_id": "safety_0_1",
        "dataset_manifest_sha256": "c" * 64,
        "generation_seed": 101,
        "algorithm_seed": 11,
        "configuration_id": "provisional",
        "selection_role": "candidate",
        "inference_artifact": "d" * 64,
        "failure_marker_sha256": "e" * 64,
        "run_report_sha256": "f" * 64,
        "tool_status": "failed",
        "failure_message": "model fit failed",
    }
    monkeypatch.setattr("spathi_bench.efficiency_safety.validate_safety_plan", lambda *_: plan)
    monkeypatch.setattr(
        "spathi_bench.efficiency_safety.verify_noninferiority_evidence",
        lambda **_: {"rows": [], "execution_failures": [failure]},
    )
    monkeypatch.setattr(
        "spathi_bench.efficiency_safety._analysis_implementation_sha256",
        lambda _root: "1" * 64,
    )

    output = root / "work/results.json"
    analyze_safety(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
        output_path=output,
    )
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["execution_failures"] == [failure]
    assert result["paired_datasets"] == []
    assert result["summary"]["global"]["completed_dataset_pairs"] == 0
    assert all(
        item["completed_dataset_pairs"] == 0 for item in result["summary"]["by_condition"].values()
    )
    assert not any(result["gates"].values())
    assert result["decision"]["status"] == "reference_retained_safety_veto"
    assert result["decision"]["selected_policy_id"] == "exact"
