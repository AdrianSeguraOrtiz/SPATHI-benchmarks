from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.reporting.deliverables import (
    build_study_deliverables,
    normalize_snapshot,
    rows_to_csv,
)


def _snapshot() -> dict[str, Any]:
    return {
        "study": {
            "id": "article-full-1",
            "profile": "full",
            "status": "complete",
            "resources": {
                "cpu_threads": 120,
                "memory_bytes": 1_099_511_627_776,
            },
        },
        "phases": {
            "calibration": {
                "summary": {
                    "primary_metric": "aupr_directed",
                    "datasets": [
                        {
                            "dataset_id": "development-1",
                            "simulator": "dyngen",
                            "dataset_role": "development",
                            "cells": 300,
                            "genes": 100,
                            "tf_count": 10,
                            "groups": 3,
                        }
                    ],
                },
                "rows": [
                    {
                        "dataset_id": "development-1",
                        "configuration_id": "weight-anchored__depth-4",
                        "metric": "aupr_directed",
                        "value": 0.41,
                        "duration_seconds": 11.4,
                        "parameters": {"max_depth": 4, "n_estimators": 50},
                    },
                    {
                        "dataset_id": "development-1",
                        "configuration_id": "weight-cell__depth-8",
                        "metric": "aupr_directed",
                        "value": 0.38,
                        "duration_seconds": 13.2,
                        "parameters": {"max_depth": 8, "n_estimators": 50},
                    },
                ],
                "decisions": [
                    {
                        "decision": "selected_configuration",
                        "selected": "weight-anchored__depth-4",
                        "reason": "highest admissible accuracy at lower cost",
                    }
                ],
                "failures": [],
                "provenance": {"plan_sha256": "a" * 64},
            },
            "comparison": {
                "summary": {
                    "primary_metric": "aupr_directed",
                    "datasets": [
                        {
                            "dataset_id": "test-1",
                            "simulator": "scmultisim",
                            "dataset_role": "test",
                            "cells": 1_000,
                            "genes": 1_000,
                            "tf_count": 100,
                            "groups": 3,
                        }
                    ],
                    "stage_summaries": [
                        {
                            "families": [
                                {
                                    "id": "a-information-matched",
                                    "label": "A · Information-matched directed TF-to-target inference",
                                    "rationale": "Every method receives the same expression, groups and TF list.",
                                    "statistical_role": "primary_friedman_holm",
                                    "route_ids": ["spathi", "genie3"],
                                },
                                {
                                    "id": "b-order-aware",
                                    "label": "B · Ordered-state directed inference",
                                    "rationale": "Order-aware methods are assessed separately.",
                                    "statistical_role": "separate_family_analysis",
                                    "route_ids": ["spathi", "simic-ordered"],
                                },
                                {
                                    "id": "c-lineage-aware",
                                    "label": "C · Lineage-aware directed inference",
                                    "rationale": "Lineage-aware methods receive additional lineage information.",
                                    "statistical_role": "separate_family_analysis",
                                    "route_ids": ["spathi", "scmtni-lineage"],
                                },
                                {
                                    "id": "d-directed-all-gene",
                                    "label": "D · Directed unrestricted-regulator fitting",
                                    "rationale": "All-gene fitting has a different fitting search space.",
                                    "statistical_role": "separate_family_analysis",
                                    "route_ids": ["spathi", "metasem", "kscreni"],
                                },
                                {
                                    "id": "e-undirected-topology",
                                    "label": "E · Undirected topology recovery",
                                    "rationale": "Direction-collapsed outputs are scored as topology only.",
                                    "statistical_role": "separate_family_analysis",
                                    "route_ids": ["spathi", "pidc", "clr"],
                                },
                            ],
                            "estimand": {
                                "scenario_weighting": "equal_replicates_per_scenario",
                                "sampling_unit": "independent_seeded_dataset",
                                "missing_data_rule": ("complete_blocks_within_family_and_metric"),
                            },
                            "statistics": {
                                "schema_version": "1.0",
                                "design": "complete_paired_blocks",
                                "post_hoc": "all_pairs_wilcoxon_holm",
                                "alpha": 0.05,
                                "block_fields": ["dataset_id"],
                                "analyses": [
                                    {
                                        "analysis_id": "group_only_directed_tf::aupr_directed",
                                        "family": "group_only_directed_tf",
                                        "metric": "aupr_directed",
                                        "status": "ok",
                                        "reason": None,
                                        "higher_is_better": True,
                                        "alpha": 0.05,
                                        "block_fields": ["dataset_id"],
                                        "methods": ["spathi", "genie3"],
                                        "observed_methods": ["genie3", "spathi"],
                                        "missing_methods": [],
                                        "n_candidate_blocks": 4,
                                        "n_complete_blocks": 3,
                                        "complete_blocks": [
                                            {"dataset_id": "test-1"},
                                            {"dataset_id": "test-2"},
                                            {"dataset_id": "test-3"},
                                        ],
                                        "excluded_blocks": [
                                            {
                                                "block": {"dataset_id": "test-4"},
                                                "missing_methods": ["genie3"],
                                                "duplicate_methods": [],
                                                "non_finite_methods": [],
                                            }
                                        ],
                                        "rankings": [
                                            {
                                                "method_id": "spathi",
                                                "n_observed": 4,
                                                "n_complete_blocks": 3,
                                                "mean_score_observed": 0.47,
                                                "median_score_observed": 0.47,
                                                "mean_score_complete": 0.46,
                                                "median_score_complete": 0.47,
                                                "average_rank": 1.2,
                                            },
                                            {
                                                "method_id": "genie3",
                                                "n_observed": 3,
                                                "n_complete_blocks": 3,
                                                "mean_score_observed": 0.39,
                                                "median_score_observed": 0.39,
                                                "mean_score_complete": 0.39,
                                                "median_score_complete": 0.39,
                                                "average_rank": 1.8,
                                            },
                                        ],
                                        "omnibus": {
                                            "status": "ok",
                                            "test": "friedman",
                                            "statistic": 3.0,
                                            "degrees_of_freedom": 1,
                                            "p_value": 0.0832645,
                                            "n_blocks": 3,
                                            "n_methods": 2,
                                        },
                                        "pairwise": [
                                            {
                                                "pair_id": "spathi__vs__genie3",
                                                "method_a": "spathi",
                                                "method_b": "genie3",
                                                "status": "ok",
                                                "n_blocks": 3,
                                                "wins_a": 3,
                                                "ties": 0,
                                                "wins_b": 0,
                                                "median_difference_a_minus_b": 0.08,
                                                "p_value": 0.25,
                                                "p_holm": 0.25,
                                                "reject_holm": False,
                                                "method": "exact_sign_permutation",
                                            }
                                        ],
                                    }
                                ],
                            },
                        }
                    ],
                },
                "rows": [
                    {
                        "dataset_id": "test-1",
                        "method_id": "spathi",
                        "family": "group_only_directed_tf",
                        "metric": "aupr_directed",
                        "value": 0.47,
                        "status": "completed",
                    },
                    {
                        "dataset_id": "test-1",
                        "method_id": "genie3",
                        "family": "group_only_directed_tf",
                        "metric": "aupr_directed",
                        "value": 0.39,
                        "status": "completed",
                    },
                    {
                        "dataset_id": "test-1",
                        "method_id": "simic",
                        "family": "order_aware_directed_tf",
                        "metric": "aupr_directed",
                        "value": 0.43,
                        "status": "completed",
                    },
                ],
                "decisions": [{"decision": "primary_family", "selected": "group_only_directed_tf"}],
                "failures": [],
                "provenance": {"plan_sha256": "b" * 64},
            },
            "scalability": {
                "summary": {
                    "datasets": [
                        {
                            "dataset_id": "scale-medium",
                            "condition_id": "P-M",
                            "dataset_role": "computational",
                            "cells": 1_000,
                            "genes": 1_000,
                            "tf_count": 100,
                            "groups": 3,
                        }
                    ]
                },
                "rows": [
                    {
                        "dataset_id": "scale-medium",
                        "scale_id": "P-M",
                        "method_id": "spathi",
                        "status": "completed",
                        "cells": 1_000,
                        "genes": 1_000,
                        "tf_count": 100,
                        "measurement": "size_ladder",
                        "requested_threads": 8,
                        "effective_threads": 8,
                        "threads": 8,
                        "duration_seconds": 312.5,
                        "peak_memory_bytes": 3_221_225_472,
                    },
                    {
                        "dataset_id": "scale-large",
                        "scale_id": "P-L",
                        "method_id": "simic",
                        "status": "estimated_infeasible_time",
                        "cells": 3_000,
                        "genes": 5_000,
                        "tf_count": 500,
                        "measurement": "size_ladder",
                        "requested_threads": 120,
                        "effective_threads": 8,
                        "threads": 8,
                    },
                ],
                "decisions": [{"decision": "advance", "method_id": "spathi", "scale_id": "P-L"}],
                "failures": [
                    {
                        "method_id": "simic",
                        "scale_id": "P-L",
                        "status": "estimated_infeasible_time",
                        "reason": "projected runtime exceeds the frozen budget",
                    }
                ],
                "provenance": {"plan_sha256": "c" * 64},
            },
        },
    }


def _calibration_stage_snapshot() -> dict[str, Any]:
    snapshot = _snapshot()
    calibration = snapshot["phases"]["calibration"]
    shared_parameters = {
        "weight_mode": "cell-distance-group-anchored",
        "distance_space": "expression",
        "n_components": 100,
        "distance_standardization": "standard",
        "distance_metric": "cosine",
        "kernel": "gaussian",
        "bandwidth_scale": 0.5,
        "group_size_correction": "none",
        "tree_method": "extra-trees",
        "max_features": 0.25,
        "min_samples_leaf": 5,
        "max_depth": 8,
        "min_weight_fraction_leaf": 0.1,
    }
    calibration["summary"]["stage_summaries"] = [
        {"fits": 27},
        {"candidate_count": 27, "finalist_count": 4},
        {
            "physical_fits": 8,
            "logical_results": 32,
            "prefixes": [25, 50, 100, 250],
            "prefix_semantics": "one shared deterministic forest per candidate and seed",
        },
        {"primary_metric": "aupr", "evaluated_configurations": 16},
    ]
    calibration["rows"] = [
        {
            "dataset_id": "development-1",
            "dataset_role": "development",
            "configuration_id": "candidate-01",
            "metric": "aupr",
            "value": 0.41,
            "duration_seconds": 11.4,
            "n_estimators": 50,
            "parameters": {**shared_parameters, "n_estimators": 50, "random_seed": 71},
        },
        {
            "dataset_id": "development-1",
            "dataset_role": "development",
            "configuration_id": "candidate-02",
            "metric": "aupr",
            "value": 0.38,
            "duration_seconds": 13.2,
            "n_estimators": 50,
            "parameters": {
                **shared_parameters,
                "max_depth": 16,
                "n_estimators": 50,
                "random_seed": 71,
            },
        },
        {
            "configuration_id": "candidate-01",
            "mean_aupr": 0.41,
            "stability_sd": 0.01,
            "mean_duration_seconds": 11.4,
            "parameters": shared_parameters,
        },
        {
            "configuration_id": "candidate-02",
            "mean_aupr": 0.38,
            "stability_sd": 0.02,
            "mean_duration_seconds": 13.2,
            "parameters": {**shared_parameters, "max_depth": 16},
        },
        {
            "dataset_id": "validation-1",
            "dataset_role": "validation",
            "configuration_id": "candidate-01",
            "metric": "aupr",
            "value": 0.43,
            "duration_seconds": None,
            "n_estimators": 25,
            "parameters": {**shared_parameters, "n_estimators": 25, "random_seed": 83},
        },
        {
            "dataset_id": "validation-1",
            "dataset_role": "validation",
            "configuration_id": "candidate-01",
            "metric": "aupr",
            "value": 0.45,
            "duration_seconds": None,
            "n_estimators": 50,
            "parameters": {**shared_parameters, "n_estimators": 50, "random_seed": 83},
        },
        {
            "configuration_id": "candidate-01",
            "n_estimators": 25,
            "mean_aupr": 0.43,
            "stability_sd": 0.012,
            "parameters": shared_parameters,
        },
        {
            "configuration_id": "candidate-01",
            "n_estimators": 50,
            "mean_aupr": 0.45,
            "stability_sd": 0.009,
            "parameters": shared_parameters,
        },
    ]
    calibration["decisions"] = [
        {
            "decision": "advance_equal_candidates",
            "selected": ["candidate-01", "candidate-02"],
        },
        {
            "decision": "selected_configuration",
            "selected": "candidate-01",
            "n_estimators": 50,
            "practical_equivalence_threshold": 0.441,
        },
    ]
    return snapshot


def test_builds_one_complete_portable_directory(tmp_path: Path) -> None:
    bundle = build_study_deliverables(_snapshot(), tmp_path / "deliverables")
    output = tmp_path / "deliverables/article-full-1"
    expected = {
        "article-report.html",
        "calibration.html",
        "comparison.html",
        "scalability.html",
        "provenance.json",
        "results-calibration.csv",
        "results-comparison.csv",
        "results-outcomes.csv",
        "results-scalability.csv",
        "SHA256SUMS",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert set(bundle["files"]) == expected

    for filename in (
        "article-report.html",
        "calibration.html",
        "comparison.html",
        "scalability.html",
    ):
        content = (output / filename).read_text(encoding="utf-8")
        assert "plotly.js" in content.lower()
        assert 'id="report-data"' in content
        assert "Self-contained SPATHI benchmark report" in content
        assert "<script src=" not in content
        assert "<link " not in content
        assert "../" not in content

    # Standalone phase pages embed only their own result rows.
    assert "weight-anchored__depth-4" in (output / "calibration.html").read_text()
    assert "genie3" not in (output / "calibration.html").read_text()
    assert "genie3" in (output / "comparison.html").read_text()
    assert "estimated_infeasible_time" in (output / "scalability.html").read_text()


def test_csv_exports_are_complete_flat_and_not_html_dependencies(tmp_path: Path) -> None:
    build_study_deliverables(_snapshot(), tmp_path)
    output = tmp_path / "article-full-1"
    with (output / "results-calibration.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["parameters.max_depth"] == "4"
    assert rows[0]["parameters.n_estimators"] == "50"
    html = (output / "calibration.html").read_text(encoding="utf-8")
    assert "results-calibration.csv" not in html

    with (output / "results-outcomes.csv").open(newline="", encoding="utf-8") as handle:
        outcomes = list(csv.DictReader(handle))
    assert {row["phase"] for row in outcomes} == {"comparison", "scalability"}
    assert {row["record_type"] for row in outcomes} == {
        "execution_outcome",
        "failure",
    }
    failure = next(row for row in outcomes if row["record_type"] == "failure")
    assert failure["method_id"] == "simic"
    assert failure["status"] == "estimated_infeasible_time"


def test_calibration_report_separates_screening_validation_and_selection(
    tmp_path: Path,
) -> None:
    build_study_deliverables(_calibration_stage_snapshot(), tmp_path)
    html = (tmp_path / "article-full-1/calibration.html").read_text(encoding="utf-8")

    assert "Equal-budget screening" in html
    assert "All declared configurations are evaluated on the development blocks" in html
    assert "Screening accuracy–runtime" in html
    assert "Validation prefixes are deliberately excluded" in html
    assert "Finalist advancement" in html
    assert "Exact-prefix validation" in html
    assert "runtime is not assigned to individual prefix points" in html
    assert 'title: "Exact tree prefix", type: "category"' in html
    assert "Final selection and practical-equivalence band" in html
    assert "not a statistical non-inferiority test" in html
    assert "Complete candidate parameter vectors" in html
    assert "distance_standardization" in html
    assert '"dataset_role":"development"' in html
    assert '"dataset_role":"validation"' in html
    assert '"practical_equivalence_threshold":0.441' in html
    assert "Candidate accuracy and cost" not in html


def test_portable_report_javascript_passes_node_syntax_check() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = Path("src/spathi_bench/reporting/assets/deliverable.js")
    completed = subprocess.run(
        [node, "--check", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_comparison_report_renders_complete_symmetric_statistics(tmp_path: Path) -> None:
    build_study_deliverables(_snapshot(), tmp_path)
    html = (tmp_path / "article-full-1/comparison.html").read_text(encoding="utf-8")

    assert "Friedman rankings and Wilcoxon–Holm comparisons" in html
    assert "No method is designated as a control" in html
    assert "Prospectively declared estimand" in html
    assert "equal_replicates_per_scenario" in html
    assert "complete_blocks_within_family_and_metric" in html
    assert "All-pairs Wilcoxon tests with Holm correction" in html
    assert "Complete blocks used" in html
    assert "Excluded blocks and reasons" in html
    assert "Analysis coverage" in html
    assert '"n_complete_blocks":3' in html
    assert '"p_holm":0.25' in html
    assert '"missing_methods":["genie3"]' in html
    assert "controlSelect" not in html


def test_comparison_accuracy_views_separate_descriptive_available_cases_from_inference(
    tmp_path: Path,
) -> None:
    build_study_deliverables(_snapshot(), tmp_path)
    html = (tmp_path / "article-full-1/comparison.html").read_text(encoding="utf-8")

    assert "Available-case accuracy distributions (descriptive)" in html
    assert "Unequal sample sizes are retained for transparency" in html
    assert "this plot is descriptive and is not used for paired inference" in html
    assert "Complete-block accuracy distributions (descriptive)" in html
    assert "Friedman and Wilcoxon–Holm inference is reported separately below" in html
    assert "function completeBlockRows(family, metric, rows)" in html


def test_comparison_report_reserves_resource_comparisons_for_scalability(
    tmp_path: Path,
) -> None:
    build_study_deliverables(_snapshot(), tmp_path)
    html = (tmp_path / "article-full-1/comparison.html").read_text(encoding="utf-8")

    assert "Runtime and memory comparisons are reported only in Scalability" in html
    assert "joint comparison telemetry can include scheduler waiting" in html
    assert "Runtime telemetry remains in the matching CSV for audit" in html
    assert 'if (name !== "comparison") auditColumns.push("duration_seconds", "peak_memory_bytes")' in html


def test_each_phase_places_dataset_context_before_result_plots(tmp_path: Path) -> None:
    build_study_deliverables(_snapshot(), tmp_path)
    html = (tmp_path / "article-full-1/article-report.html").read_text(encoding="utf-8")

    assert "Synthetic evidence represented in this phase" in html
    assert "shown before any result plot" in html
    context_call = html.index("renderDatasetContext(wrapper, phase);")
    assert context_call < html.index("calibrationPlots(wrapper, phase);")
    assert context_call < html.index("comparisonPlots(wrapper, phase);")
    assert context_call < html.index("scalabilityPlots(wrapper, phase);")


def test_comparison_report_exposes_formal_competitor_family_contracts(
    tmp_path: Path,
) -> None:
    build_study_deliverables(_snapshot(), tmp_path)
    html = (tmp_path / "article-full-1/comparison.html").read_text(encoding="utf-8")

    assert "Predeclared competitor families" in html
    assert "information contract, output semantics and statistical role" in html
    for label in (
        "A · Information-matched directed TF-to-target inference",
        "B · Ordered-state directed inference",
        "C · Lineage-aware directed inference",
        "D · Directed unrestricted-regulator fitting",
        "E · Undirected topology recovery",
    ):
        assert label in html
    assert '"route_ids":["spathi","genie3"]' in html
    assert '"statistical_role":"primary_friedman_holm"' in html
    assert '"statistical_role":"separate_family_analysis"' in html
    assert 'family.route_ids.join(", ")' in html
    assert "Every method receives the same expression, groups and TF list." in html
    assert "Direction-collapsed outputs are scored as topology only." in html


def test_comparison_coverage_uses_explicit_execution_records_and_deduplicates_runs(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot()
    snapshot["phases"]["comparison"]["summary"]["execution_coverage"] = [
        {
            "dataset_id": "test-1",
            "method_id": "spathi",
            "family": "group_only_directed_tf",
            "status": "completed",
        },
        {
            "dataset_id": "test-1",
            "method_id": "spathi",
            "family": "group_only_directed_tf",
            "status": "completed",
        },
        {
            "dataset_id": "test-1",
            "method_id": "genie3",
            "family": "group_only_directed_tf",
            "status": "runtime_failure",
        },
    ]

    build_study_deliverables(snapshot, tmp_path)
    html = (tmp_path / "article-full-1/comparison.html").read_text(encoding="utf-8")
    assert "Unique dataset–method executions" in html
    assert "Multiple metric rows from one execution are counted once." in html
    assert "comparisonCoverageRows(phase)" in html
    assert (
        "JSON.stringify([normalized.dataset_id, normalized.method_id, normalized.family])" in html
    )

    with (tmp_path / "article-full-1/results-outcomes.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        outcomes = list(csv.DictReader(handle))
    coverage = [row for row in outcomes if row["record_type"] == "execution_coverage"]
    assert len(coverage) == 3
    assert {row["status"] for row in coverage} == {"completed", "runtime_failure"}


def test_scalability_report_uses_declared_scale_and_explicit_thread_budgets(
    tmp_path: Path,
) -> None:
    build_study_deliverables(_snapshot(), tmp_path)
    html = (tmp_path / "article-full-1/scalability.html").read_text(encoding="utf-8")

    assert 'title: "Synthetic scale", type: "category"' in html
    assert "Cells in synthetic dataset" not in html
    assert "Cells: %{customdata[0]}" in html
    assert "Genes: %{customdata[1]}" in html
    assert "TFs: %{customdata[2]}" in html
    assert "Requested threads:" in html
    assert "Effective threads:" in html
    assert '"requested_threads":120' in html
    assert '"effective_threads":8' in html


def test_scalability_report_distinguishes_not_attempted_from_failed(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot()
    snapshot["phases"]["scalability"]["rows"].append(
        {
            "dataset_id": "scale-xlarge",
            "scale_id": "P-XL",
            "method_id": "simic",
            "status": "not_attempted",
            "reason": "desisted_after_andrea_resource_outcome",
            "measurement": "size_ladder",
        }
    )
    build_study_deliverables(snapshot, tmp_path)
    html = (tmp_path / "article-full-1/scalability.html").read_text(encoding="utf-8")

    assert '"status":"not_attempted"' in html
    assert "not attempted means no process was launched" in html
    assert "Not attempted — a prior ANDREA resource outcome stopped higher scales" in html
    assert "status--not-attempted" in html
    assert "not_attempted: -.3" in html


def test_provenance_and_checksum_inventory_are_verifiable(tmp_path: Path) -> None:
    from hashlib import sha256

    build_study_deliverables(_snapshot(), tmp_path)
    output = tmp_path / "article-full-1"
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["study_id"] == "article-full-1"
    assert provenance["phases"]["comparison"]["plan_sha256"] == "b" * 64
    assert provenance["renderer"]["fingerprint"]
    entries = {}
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        entries[filename] = digest
    assert "SHA256SUMS" not in entries
    assert set(entries) == {path.name for path in output.iterdir()} - {"SHA256SUMS"}
    for filename, digest in entries.items():
        assert sha256((output / filename).read_bytes()).hexdigest() == digest


def test_rebuild_is_byte_deterministic(tmp_path: Path) -> None:
    first = build_study_deliverables(_snapshot(), tmp_path)
    second = build_study_deliverables(_snapshot(), tmp_path)
    assert {name: record["sha256"] for name, record in first["files"].items()} == {
        name: record["sha256"] for name, record in second["files"].items()
    }


def test_rebuild_replaces_the_complete_bundle_and_removes_stale_entries(
    tmp_path: Path,
) -> None:
    build_study_deliverables(_snapshot(), tmp_path)
    output = tmp_path / "article-full-1"
    (output / "stale-report.html").write_text("obsolete", encoding="utf-8")

    snapshot = _snapshot()
    snapshot["study"]["status"] = "running"
    build_study_deliverables(snapshot, tmp_path)

    assert not (output / "stale-report.html").exists()
    assert '"status":"running"' in (output / "article-report.html").read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".article-full-1-*-*"))


def test_failed_directory_swap_restores_the_previous_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spathi_bench.reporting import deliverables

    build_study_deliverables(_snapshot(), tmp_path)
    output = tmp_path / "article-full-1"
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    real_replace = deliverables.os.replace

    def fail_new_bundle(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name.startswith(".article-full-1-staging-") and destination_path == output:
            raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(deliverables.os, "replace", fail_new_bundle)
    snapshot = _snapshot()
    snapshot["study"]["status"] = "running"
    with pytest.raises(OSError, match="injected publication failure"):
        build_study_deliverables(snapshot, tmp_path)

    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert after == before
    assert not list(tmp_path.glob(".article-full-1-*-*"))


def test_script_terminators_are_escaped_in_embedded_data(tmp_path: Path) -> None:
    snapshot = _snapshot()
    snapshot["phases"]["comparison"]["failures"] = [
        {"reason": "</script><script>window.compromised=true</script>"}
    ]
    build_study_deliverables(snapshot, tmp_path)
    content = (tmp_path / "article-full-1/article-report.html").read_text(encoding="utf-8")
    assert "</script><script>window.compromised" not in content
    assert "\\u003c/script\\u003e" in content


def test_snapshot_contract_rejects_incomplete_or_unknown_phase_shapes() -> None:
    snapshot = _snapshot()
    del snapshot["phases"]["scalability"]
    snapshot["phases"]["unknown"] = {}
    with pytest.raises(BenchmarkConfigError, match="missing scalability; unexpected unknown"):
        normalize_snapshot(snapshot)


def test_snapshot_rejects_nonfinite_values_before_writing(tmp_path: Path) -> None:
    snapshot = _snapshot()
    snapshot["phases"]["calibration"]["rows"][0]["value"] = float("nan")
    with pytest.raises(BenchmarkConfigError, match="strict JSON"):
        build_study_deliverables(snapshot, tmp_path)
    assert not list(tmp_path.iterdir())


def test_rows_to_csv_has_stable_preferred_columns() -> None:
    content = rows_to_csv(
        [
            {
                "value": 0.3,
                "dataset_id": "d1",
                "method_id": "spathi",
                "parameters": {"max_depth": 4},
            }
        ]
    )
    assert content.splitlines()[0] == "dataset_id,method_id,value,parameters.max_depth"
