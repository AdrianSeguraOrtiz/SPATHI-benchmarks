from __future__ import annotations

import csv
import json
import re
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
from plotly.offline import get_plotlyjs

import spathi_bench.reporting.calibration as calibration_reporting
import spathi_bench.reporting.common as reporting_common
import spathi_bench.reporting.comparison as comparison_reporting
from spathi_bench.artifacts import ArtifactStore, canonical_json_sha256, sha256_file
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.reporting.common import json_for_script, render_html, write_report_bundle


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _runtime_plan(*, cpu_threads: int = 8, memory_gib: float = 32.0) -> dict[str, Any]:
    unsigned = {
        "schema_version": "1.0",
        "detected": {
            "effective_cpu_threads": cpu_threads,
            "effective_memory_bytes": round(memory_gib * 1024**3),
            "source": "report-test",
        },
        "request": {
            "max_cores": cpu_threads,
            "max_ram_gb": memory_gib,
            "reserve_cores": 0,
            "reserve_ram_gb": 0.0,
        },
        "capacity": {
            "cpu_threads": cpu_threads,
            "memory_bytes": round(memory_gib * 1024**3),
            "memory_gib": memory_gib,
        },
    }
    return {**unsigned, "fingerprint": canonical_json_sha256(unsigned)}


def _dataset_fixture(
    repo_root: Path,
    *,
    version_root: Path | None = None,
    artifact_id: str,
    dataset_id: str,
    dataset_role: str,
    simulator_id: str,
    condition_id: str,
    seed: int,
    genes: int = 80,
    cells: int = 300,
) -> dict[str, Any]:
    version = version_root or repo_root / "work/v1"
    store = ArtifactStore(version / "artifacts")
    artifact_spec = {
        "fixture": "interactive-reporting",
        "fixture_id": artifact_id,
        "dataset_id": dataset_id,
        "dataset_role": dataset_role,
        "simulator_id": simulator_id,
        "condition_id": condition_id,
        "seed": seed,
        "genes": genes,
        "cells": cells,
    }
    key = store.key_for("dataset", artifact_spec)
    with store.staging_directory(key) as staging:
        dataset_root = staging / "output" / dataset_id
        groups_path = dataset_root / "extras/groups.tsv"
        tf_path = dataset_root / "extras/tf_list.txt"
        expression_path = dataset_root / "expression.tsv"
        tf_path.parent.mkdir(parents=True, exist_ok=True)
        groups_path.write_text(
            "column\tcluster\ncell1\ta\ncell2\tb\ncell3\tc\ncell4\td\n",
            encoding="utf-8",
        )
        tf_path.write_text("TF1\nTF2\nTF3\n", encoding="utf-8")
        expression_path.write_text("gene\tcell1\nG1\t1\n", encoding="utf-8")
        _write_json(
            dataset_root / "dataset-manifest.json",
            {
                "schema_version": "1.0",
                "dataset": {
                    "spec": {
                        "schema_version": "1.0",
                        "id": dataset_id,
                        "name": dataset_id,
                        "expression": {
                            "genes": genes,
                            "columns": cells,
                            "column_kind": "cells",
                            "expression_profile": "scrna",
                        },
                    },
                    "expression_matrix": "expression.tsv",
                },
                "extras": {
                    "groups": "extras/groups.tsv",
                    "tf_list": "extras/tf_list.txt",
                },
            },
        )
        artifact = store.publish_from_temp(
            staging,
            kind="dataset",
            spec=artifact_spec,
        )
    manifest_path = artifact.path / "output" / dataset_id / "dataset-manifest.json"
    return {
        "dataset_id": dataset_id,
        "dataset_role": dataset_role,
        "simulator_id": simulator_id,
        "condition_id": condition_id,
        "seed": seed,
        "split_id": f"{dataset_role}_v1",
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_artifact": {
            "kind": artifact.key.kind,
            "digest": artifact.key.digest,
            "key": artifact.key.value,
            "path": str(artifact.path.resolve()),
            "cache_hit": False,
        },
    }


def _calibration_fixture(
    tmp_path: Path, *, isolated: bool = False
) -> tuple[Path, Path, dict[str, Any]]:
    repo_root = tmp_path / "repo"
    work_root = repo_root / "work" / "isolated" if isolated else repo_root / "work"
    version_root = work_root / "v1"
    workflow_root = version_root / "workflows/calibration/reduced/calibration-fixture"
    workflow_root.mkdir(parents=True)
    spathi_commit = "2" * 40
    search_path = repo_root / "experiments/v1/spathi/search-space.json"
    stage_ids = tuple(
        dict.fromkeys(stage for stage, _fidelity in calibration_reporting.CALIBRATION_PHASE_ORDER)
    )
    development_stages = tuple(stage for stage in stage_ids if stage != "validation")
    phase_designs: dict[str, Any] = {}
    for stage_id, fidelity in calibration_reporting.CALIBRATION_PHASE_ORDER:
        tier = {
            "candidate_source": "all_expanded" if fidelity == "scout" else "shortlist",
            "advance_limit": 1,
            "dataset_selector": {
                "condition_ids": ["linear_baseline"],
                "replicate_indices": [1],
            },
            "algorithm_seed_indices": [1],
            "target_scope": {"method": "all"},
            "expected_physical_fits": 2,
        }
        if stage_id == "convergence":
            tier["expected_logical_results"] = 8
        if fidelity == "final":
            phase_designs[stage_id] = tier
        else:
            phase_designs.setdefault(stage_id, {})[fidelity] = tier
    search_space = {
        "objective": {"primary_metric": "aupr_directed", "direction": "maximize"},
        "dataset_partition": {"assignment_unit": "dataset_instance"},
        "selection_protocol": {},
        "stages": [
            {
                "id": stage_id,
                "order": index,
                "purpose": f"Purpose for {stage_id}",
                "strategy": "one_factor_at_a_time",
                "advance_limit": 1,
                "candidates": [
                    {"id": "kept", "rationale": f"Keep the {stage_id} candidate."},
                    {"id": "runner", "rationale": f"Compare the {stage_id} runner."},
                ],
            }
            for index, stage_id in enumerate(development_stages, start=1)
        ],
        "calibration_budgets": {"reduced": {"multifidelity": {"stages": phase_designs}}},
    }
    _write_json(search_path, search_space)
    search_sha256 = sha256_file(search_path)

    stage_records: dict[str, Any] = {}
    ranking_paths: list[Path] = []
    winning_candidate: dict[str, Any] | None = None
    development_dataset = _dataset_fixture(
        repo_root,
        version_root=version_root,
        artifact_id="development",
        dataset_id="dataset-a",
        dataset_role="development",
        simulator_id="dyngen",
        condition_id="linear_baseline",
        seed=11001,
    )
    validation_dataset = _dataset_fixture(
        repo_root,
        version_root=version_root,
        artifact_id="validation",
        dataset_id="dataset-a",
        dataset_role="validation",
        simulator_id="dyngen",
        condition_id="linear_baseline",
        seed=31001,
    )
    for stage_id, fidelity in calibration_reporting.CALIBRATION_PHASE_ORDER:
        phase_id = f"{stage_id}__{fidelity}"
        dataset_role = "validation" if stage_id == "validation" else "development"
        kept = {
            "configuration_id": f"{stage_id}__kept",
            "configuration_sha256": (stage_id[0] * 64),
            "variant_id": "kept",
            "parent_configuration_id": None,
            "parameters": {
                "n_estimators": 100,
                "max_depth": None,
                "distance_metric": "cosine",
            },
        }
        runner = {
            **kept,
            "configuration_id": f"{stage_id}__runner",
            "configuration_sha256": (stage_id[-1] * 64),
            "variant_id": "runner",
            "parameters": {**kept["parameters"], "n_estimators": 250},
        }
        ranking_path = workflow_root / "stages" / phase_id / "ranking.json"
        parent = None
        if ranking_paths:
            parent = {
                "path": ranking_paths[-1].relative_to(repo_root).as_posix(),
                "sha256": sha256_file(ranking_paths[-1]),
                "stage_id": calibration_reporting.CALIBRATION_PHASE_ORDER[len(ranking_paths) - 1][
                    0
                ],
                "fidelity": calibration_reporting.CALIBRATION_PHASE_ORDER[len(ranking_paths) - 1][
                    1
                ],
            }
        winner = None
        if stage_id == "validation":
            winner = {
                "candidate_id": kept["configuration_id"],
                "configuration_sha256": kept["configuration_sha256"],
                "hierarchical_score": 0.2,
            }
            winning_candidate = kept
        ranking = {
            "schema_version": "1.0",
            "stage_id": stage_id,
            "fidelity": fidelity,
            "calibration_budget": "reduced",
            "profile_id": f"calibration-reduced-{stage_id}-{fidelity}",
            "search_space_sha256": search_sha256,
            "spathi_commit": spathi_commit,
            "parent_ranking": parent,
            "test_data_accessed": False,
            "dataset_role": dataset_role,
            "primary_metric": "aupr_directed",
            # Deliberately not sorted by score: reporting must preserve the frozen ranking.
            "ranking": [
                {
                    "candidate": kept,
                    "hierarchical_aupr": 0.2,
                    "stability": 0.01,
                    "median_wall_time_seconds": 2.0,
                    "median_peak_memory_mb": None,
                    "simplicity": 1,
                    "dataset_count": 2,
                    "algorithm_seed_count": 2,
                },
                {
                    "candidate": runner,
                    "hierarchical_aupr": 0.9,
                    "stability": 0.02,
                    "median_wall_time_seconds": 5.0,
                    "median_peak_memory_mb": None,
                    "simplicity": 2,
                    "dataset_count": 2,
                    "algorithm_seed_count": 2,
                },
            ],
            "selected_candidates": [kept],
            "calibration_rows": [
                {
                    "configuration_id": kept["configuration_id"],
                    "dataset_id": "dataset-a",
                    "dataset_role": dataset_role,
                    "n_group_contexts": 4,
                    "algorithm_seed": 1729,
                    "aupr_directed": 0.2,
                    "wall_time_seconds": 2.0,
                    "peak_memory_mb": None,
                }
            ],
            "winner": winner,
        }
        _write_json(ranking_path, ranking)
        ranking_paths.append(ranking_path)
        stage_records[phase_id] = {
            "status": "ranked",
            "profile_id": ranking["profile_id"],
            "ranking": ranking_path.relative_to(repo_root).as_posix(),
            "ranking_sha256": sha256_file(ranking_path),
            "run_root": (version_root / f"calibration-{phase_id}")
            .relative_to(repo_root)
            .as_posix(),
        }
        _write_json(
            version_root / f"calibration-{phase_id}" / "state.json",
            {
                "schema_version": "1.0",
                "status": "completed",
                "profile_id": ranking["profile_id"],
                "artifact_store": str((version_root / "artifacts").resolve()),
                "datasets": [
                    {
                        **(validation_dataset if stage_id == "validation" else development_dataset),
                        "dataset_artifact": {
                            **(validation_dataset if stage_id == "validation" else development_dataset)[
                                "dataset_artifact"
                            ],
                            "cache_hit": stage_id != "validation" and phase_id != "geometry__scout",
                        },
                    }
                ],
            },
        )

    assert winning_candidate is not None
    group_policy = {
        "routing_parameter": "group_count",
        "branches": {
            "single_group": {
                "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                "parameters": winning_candidate["parameters"],
            },
            "multi_group": {
                "condition": {"minimum_group_count": 2},
                "parameters": winning_candidate["parameters"],
            },
        },
    }
    selected = {
        "schema_version": "1.0",
        "status": "selected",
        "phase": "frozen",
        "calibration_budget": "reduced",
        "test_accessed_before_freeze": False,
        "selected_policy": {
            "policy_id": "spathi_conditional_v1",
            "source_candidate_id": winning_candidate["configuration_id"],
            "source_configuration_sha256": winning_candidate["configuration_sha256"],
            "spathi_commit": spathi_commit,
            "policy_sha256": canonical_json_sha256(group_policy),
            "group_policy": group_policy,
        },
    }
    selected_path = _write_json(workflow_root / "selected-config.json", selected)
    workflow = {
        "schema_version": "1.0",
        "workflow_type": "calibration",
        "flow_id": "calibration-fixture",
        "status": "publishing",
        "calibration_budget": "reduced",
        "identity": {
            "spathi_commit": spathi_commit,
            "fingerprint": "5" * 64,
        },
        "stages": stage_records,
        "selected_config": selected_path.relative_to(repo_root).as_posix(),
        "selected_config_sha256": sha256_file(selected_path),
        "interactive_report": (
            version_root / "reports/reduced/calibration-fixture/calibration.html"
        )
        .relative_to(repo_root)
        .as_posix(),
    }
    _write_json(workflow_root / "workflow.json", workflow)
    return repo_root, workflow_root, selected


def _comparison_fixture(
    tmp_path: Path, *, isolated: bool = False
) -> tuple[Path, Path, Path, dict[str, Any]]:
    repo_root = tmp_path / "repo"
    work_root = repo_root / "work" / "isolated" if isolated else repo_root / "work"
    version_root = work_root / "v1"
    workflow_root = version_root / "workflows/comparison/reduced/comparison-fixture"
    run_root = version_root / "reduced-v1/comparison-fixture"
    reports_root = run_root / "reports"
    statistics_root = reports_root / "statistics"
    statistics_root.mkdir(parents=True)

    profile = _write_json(workflow_root / "reduced-v1.json", {"id": "reduced-v1"})
    metrics_path = reports_root / "metrics.csv"
    fields = [
        "inference_artifact",
        "configuration_id",
        "dataset_id",
        "simulator_id",
        "condition_id",
        "dataset_role",
        "regime",
        "run_id",
        "catalog_tool_id",
        "family",
        "algorithm_seed",
        "wall_time_seconds",
        "peak_memory_mb",
    ]
    runtime_row = {
        "inference_artifact": "sha256:" + "a" * 64,
        "configuration_id": "genie3_default",
        "dataset_id": "dataset-a",
        "simulator_id": "dyngen",
        "condition_id": "baseline",
        "dataset_role": "development",
        "regime": "group_only",
        "run_id": "genie3_default_seed_1729",
        "catalog_tool_id": "genie3",
        "family": "catalog_tf_restricted",
        "algorithm_seed": "1729",
        "wall_time_seconds": "12.5",
        "peak_memory_mb": "",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        # The same inference contributes multiple metric/context rows in real reports.
        writer.writerow(runtime_row)
        writer.writerow(runtime_row)

    summary = {
        "schema_version": "1.0",
        "profile_id": "reduced-v1",
        "purpose": "regression",
        "aggregates": [{"marker": "published-summary"}],
        "scheduling": {},
    }
    summary_path = _write_json(reports_root / "summary.json", summary)
    statistics = {
        "schema_version": "1.0",
        "status": "complete",
        "profile_id": "reduced-v1",
        "purpose": "regression",
        "inputs": {
            "benchmark_environment": {
                "fingerprint": "f" * 64,
                "machine": "x86_64",
                "python_base_prefix": "/private/python",
                "python_executable": "/private/venv/bin/python",
                "python_executable_sha256": "e" * 64,
                "python_implementation": "CPython",
                "python_prefix": "/private/venv",
                "python_version": "3.13.5",
                "system": "Linux",
            },
            "metrics_csv": {"sha256": sha256_file(metrics_path)},
            "summary_json": {"sha256": sha256_file(summary_path)},
        },
        "analyses": [
            {
                "analysis_id": "primary-aupr",
                "analysis_role": "exploratory",
                "comparison_role": "primary",
                "metric": "aupr",
                "stratum": {
                    "dataset_role": "development",
                    "level": "directed",
                    "regime": "group_only",
                },
                "blocks": [
                    {
                        "dataset_id": "dataset-a",
                        "simulator_id": "dyngen",
                        "n_contexts": 4,
                        "contexts": ["group:a", "group:b", "group:c", "group:d"],
                    }
                ],
                "marker": "published-statistics",
            }
        ],
    }
    statistics_path = _write_json(statistics_root / "statistics.json", statistics)
    rankings_path = statistics_root / "rankings.csv"
    rankings_path.write_text("analysis_id,configuration_id\n", encoding="utf-8")
    pairwise_path = statistics_root / "pairwise.csv"
    pairwise_path.write_text("analysis_id,control\n", encoding="utf-8")
    markdown_path = statistics_root / "report.md"
    markdown_path.write_text("# Statistical report\n", encoding="utf-8")
    statistic_files = {
        path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in (statistics_path, rankings_path, pairwise_path, markdown_path)
    }

    exports_root = run_root / "exports"
    bundles = []
    for producer, scope, relative in (
        ("generate-data", "dataset", "datasets/dataset-a/analysis.zip"),
        (
            "infer-network",
            "joint",
            "inference/dataset-a/group_only/joint/analysis.zip",
        ),
        (
            "evaluate-inference",
            "joint",
            "evaluation/dataset-a/group_only/joint/analysis.zip",
        ),
    ):
        archive = exports_root / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"fixture zip")
        bundle: dict[str, Any] = {
            "dataset_id": "dataset-a",
            "producer_command": producer,
            "scope": scope,
            "analysis_zip": relative,
            "archive_sha256": sha256_file(archive),
        }
        if scope == "joint":
            bundle["regime"] = "group_only"
        bundles.append(bundle)
    exports_path = _write_json(
        exports_root / "manifest.json",
        {
            "schema_version": "1.0",
            "status": "complete",
            "profile_id": "reduced-v1",
            "bundles": bundles,
        },
    )

    dataset = _dataset_fixture(
        repo_root,
        version_root=version_root,
        artifact_id="comparison",
        dataset_id="dataset-a",
        dataset_role="development",
        simulator_id="dyngen",
        condition_id="baseline",
        seed=11001,
    )
    state = {
        "schema_version": "1.0",
        "status": "completed",
        "profile_id": "reduced-v1",
        "purpose": "regression",
        "artifact_store": str((version_root / "artifacts").resolve()),
        "datasets": [dataset],
        "reports": {
            "metrics_csv": str(metrics_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
        "statistics": {
            "schema_version": "1.0",
            "directory": str(statistics_root.resolve()),
            "files": statistic_files,
            "input_metrics_sha256": sha256_file(metrics_path),
            "input_summary_sha256": sha256_file(summary_path),
        },
        "exports_manifest": str(exports_path.resolve()),
    }
    _write_json(run_root / "state.json", state)
    production = _write_json(
        version_root / "workflows/efficiency/reduced/efficiency-fixture/production-config.json",
        {"artifact_type": "spathi_production_policy", "budget": "reduced"},
    )
    workflow = {
        "schema_version": "1.0",
        "workflow_type": "comparison",
        "flow_id": "comparison-fixture",
        "status": "publishing",
        "profile": "reduced",
        "profile_file": profile.relative_to(repo_root).as_posix(),
        "profile_sha256": sha256_file(profile),
        "efficiency_flow_id": "efficiency-fixture",
        "production_config": production.relative_to(repo_root).as_posix(),
        "production_config_sha256": sha256_file(production),
        "identity": {"spathi_commit": "2" * 40, "fingerprint": "5" * 64},
        "runtime_plan": _runtime_plan(),
        "interactive_report": (version_root / "reports/reduced/comparison-fixture/comparison.html")
        .relative_to(repo_root)
        .as_posix(),
        "stages": {
            "comparison": {
                "status": "complete",
                "pipeline": {"run_root": str(run_root.resolve())},
            }
        },
    }
    _write_json(workflow_root / "workflow.json", workflow)
    return repo_root, workflow_root, run_root, statistics


def test_json_for_script_escapes_script_terminators() -> None:
    hostile = "</script><script>alert('&')</script>\u2028next\u2029"

    rendered = json_for_script({"hostile": hostile})

    assert "</script" not in rendered.lower()
    assert "<" not in rendered
    assert "&" not in rendered
    assert "\\u003c/script\\u003e" in rendered
    assert json.loads(rendered) == {"hostile": hostile}


def test_rendered_html_is_self_contained_without_script_src_or_cdn() -> None:
    html = render_html(
        kind="calibration",
        payload={"workflow": {}, "selected": {}, "stages": [], "links": []},
    )

    assert "Plotly" in html
    assert 'id="report-data"' in html
    assert not re.search(r"<script\b[^>]*\bsrc\s*=", html, flags=re.IGNORECASE)
    assert not re.search(
        r"<(?:script|link)\b[^>]*(?:src|href)\s*=\s*['\"]https?://",
        html,
        flags=re.IGNORECASE,
    )


def test_report_bundle_generation_is_byte_deterministic_and_idempotent(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    input_path = repo_root / "work/input.json"
    _write_json(input_path, {"stable": True})
    output_path = repo_root / "work/reports/calibration.html"
    payload = {
        "workflow": {"flow_id": "flow-a", "identity": {"fingerprint": "a" * 64}},
        "selected": {},
        "stages": [],
        "links": [],
    }

    first = write_report_bundle(
        repo_root=repo_root,
        kind="calibration",
        output_path=output_path,
        payload=payload,
        input_paths=[input_path],
        manifest_name="calibration-report.manifest.json",
    )
    first_html = output_path.read_bytes()
    manifest_path = output_path.parent / "calibration-report.manifest.json"
    first_manifest = manifest_path.read_bytes()
    second = write_report_bundle(
        repo_root=repo_root,
        kind="calibration",
        output_path=output_path,
        payload=payload,
        input_paths=[input_path],
        manifest_name="calibration-report.manifest.json",
    )

    assert second == first
    assert output_path.read_bytes() == first_html
    assert manifest_path.read_bytes() == first_manifest
    assert second["html"]["sha256"] == sha256_file(output_path)


def test_renderer_resources_have_separate_deterministic_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    input_path = repo_root / "work/input.json"
    _write_json(input_path, {"stable": True})
    output_path = repo_root / "work/reports/calibration.html"
    manifest_path = output_path.parent / "calibration-report.manifest.json"
    payload = {
        "workflow": {"flow_id": "flow-a", "identity": {"fingerprint": "a" * 64}},
        "selected": {},
        "stages": [],
        "links": [],
    }

    write_report_bundle(
        repo_root=repo_root,
        kind="calibration",
        output_path=output_path,
        payload=payload,
        input_paths=[input_path],
        manifest_name=manifest_path.name,
    )
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_output_sha256 = original_manifest["output"]["sha256"]
    renderer = original_manifest["renderer"]

    assert original_manifest["fingerprint_scope"] == "scientific_inputs"
    assert renderer["schema_version"] == reporting_common.RENDERER_SCHEMA_VERSION
    assert renderer["dependencies"] == {
        "jinja2": metadata.version("Jinja2"),
        "plotly": metadata.version("plotly"),
    }
    assert (
        renderer["resources"]["assets/report.css"]["sha256"]
        == sha256(reporting_common._resource_text("assets/report.css").encode("utf-8")).hexdigest()
    )
    assert (
        renderer["resources"]["vendor/plotly.js"]["sha256"]
        == sha256(get_plotlyjs().encode("utf-8")).hexdigest()
    )
    for relative in (
        "common.py",
        "calibration.py",
        "templates/base.html.j2",
        "templates/calibration.html.j2",
        "assets/report.css",
        "assets/common.js",
        "assets/calibration.js",
        "vendor/plotly.js",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", renderer["resources"][relative]["sha256"])
        assert renderer["resources"][relative]["size_bytes"] > 0

    original_resource_text = reporting_common._resource_text

    def changed_resource(relative: str) -> str:
        content = original_resource_text(relative)
        if relative == "assets/report.css":
            return content + "\n/* renderer fingerprint test */\n"
        return content

    monkeypatch.setattr(reporting_common, "_resource_text", changed_resource)
    write_report_bundle(
        repo_root=repo_root,
        kind="calibration",
        output_path=output_path,
        payload=payload,
        input_paths=[input_path],
        manifest_name=manifest_path.name,
    )
    changed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert changed_manifest["fingerprint"] == original_manifest["fingerprint"]
    assert (
        changed_manifest["renderer"]["fingerprint"] != original_manifest["renderer"]["fingerprint"]
    )
    assert (
        changed_manifest["renderer"]["resources"]["assets/report.css"]
        != (original_manifest["renderer"]["resources"]["assets/report.css"])
    )
    assert changed_manifest["output"]["sha256"] != original_output_sha256

    html = output_path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert embedded["report"]["fingerprint"] == changed_manifest["fingerprint"]
    assert embedded["report"]["fingerprint_scope"] == "scientific_inputs"
    assert embedded["report"]["renderer"] == changed_manifest["renderer"]


@pytest.mark.parametrize("isolated", [False, True], ids=["default-work", "isolated-work"])
def test_calibration_report_preserves_frozen_ranking_selection_and_null_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated: bool,
) -> None:
    repo_root, workflow_root, selected = _calibration_fixture(tmp_path, isolated=isolated)
    version_root = workflow_root.parents[3]
    captured: dict[str, Any] = {}

    def capture_bundle(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"captured": True}

    monkeypatch.setattr(calibration_reporting, "write_report_bundle", capture_bundle)

    result = calibration_reporting.render_calibration_report(
        repo_root=repo_root,
        version_root=version_root,
        workflow_root=workflow_root,
    )

    assert result == {"captured": True}
    assert captured["output_path"] == (
        version_root / "reports/reduced/calibration-fixture/calibration.html"
    )
    assert captured["output_path"] != workflow_root / "reports/calibration.html"
    assert captured["manifest_name"] == "calibration-report.manifest.json"
    assert captured["immutable"] is True
    payload = captured["payload"]
    assert payload["workflow"]["status"] == "complete"
    assert payload["selected"] == selected
    assert [stage["id"] for stage in payload["stages"]] == [
        f"{stage_id}__{fidelity}"
        for stage_id, fidelity in calibration_reporting.CALIBRATION_PHASE_ORDER
    ]
    assert payload["datasets"]["summary"] == {
        "dataset_count": 2,
        "total_cells": 600,
        "simulators": [{"id": "dyngen", "count": 2}],
        "roles": [
            {"id": "development", "count": 1},
            {"id": "validation", "count": 1},
        ],
        "genes_min": 80,
        "genes_max": 80,
        "cells_min": 300,
        "cells_max": 300,
        "groups_min": 4,
        "groups_max": 4,
        "transcription_factors_min": 3,
        "transcription_factors_max": 3,
    }
    for stage in payload["stages"]:
        assert [candidate["configuration_id"] for candidate in stage["candidates"]] == [
            f"{stage['stage_id']}__kept",
            f"{stage['stage_id']}__runner",
        ]
        assert [candidate["score"] for candidate in stage["candidates"]] == [0.2, 0.9]
        assert [candidate["selected"] for candidate in stage["candidates"]] == [True, False]
        assert stage["memory_available"] is False
        assert stage["candidates"][0]["median_peak_memory_mb"] is None
        assert stage["candidates"][0]["parameters"]["max_depth"] is None
        assert stage["compared_parameters"] == [{"name": "n_estimators", "values": ["100", "250"]}]


@pytest.mark.parametrize("isolated", [False, True], ids=["default-work", "isolated-work"])
def test_comparison_report_uses_published_inputs_joint_links_and_deduplicated_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated: bool,
) -> None:
    repo_root, workflow_root, run_root, published_statistics = _comparison_fixture(
        tmp_path, isolated=isolated
    )
    version_root = workflow_root.parents[3]
    captured: dict[str, Any] = {}

    def capture_bundle(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"captured": True}

    monkeypatch.setattr(comparison_reporting, "write_report_bundle", capture_bundle)

    result = comparison_reporting.render_comparison_report(
        repo_root=repo_root,
        version_root=version_root,
        workflow_root=workflow_root,
        run_root=run_root,
    )

    assert result == {"captured": True}
    report_path = version_root / "reports/reduced/comparison-fixture/comparison.html"
    assert captured["output_path"] == report_path
    assert captured["output_path"] != run_root / "reports/comparison.html"
    assert captured["manifest_name"] == "comparison-report.manifest.json"
    assert captured["immutable"] is True
    payload = captured["payload"]
    assert payload["workflow"]["status"] == "complete"
    expected_statistics = json.loads(json.dumps(published_statistics))
    expected_statistics["inputs"]["benchmark_environment"] = {
        "fingerprint": "f" * 64,
        "machine": "x86_64",
        "python_executable_sha256": "e" * 64,
        "python_implementation": "CPython",
        "python_version": "3.13.5",
        "system": "Linux",
    }
    assert payload["statistics"] == expected_statistics
    assert published_statistics["inputs"]["benchmark_environment"]["python_executable"] == (
        "/private/venv/bin/python"
    )
    assert payload["summary"]["aggregates"] == [{"marker": "published-summary"}]
    assert payload["workflow"]["analysis_roles"] == ["exploratory"]
    assert payload["workflow"]["comparison_roles"] == ["primary"]
    assert payload["datasets"]["summary"] == {
        "dataset_count": 1,
        "total_cells": 300,
        "simulators": [{"id": "dyngen", "count": 1}],
        "roles": [{"id": "development", "count": 1}],
        "genes_min": 80,
        "genes_max": 80,
        "cells_min": 300,
        "cells_max": 300,
        "groups_min": 4,
        "groups_max": 4,
        "transcription_factors_min": 3,
        "transcription_factors_max": 3,
    }
    assert payload["runtimes"] == {
        "rows": [
            {
                "inference_artifact": "sha256:" + "a" * 64,
                "configuration_id": "genie3_default",
                "dataset_id": "dataset-a",
                "simulator_id": "dyngen",
                "condition_id": "baseline",
                "dataset_role": "development",
                "regime": "group_only",
                "run_id": "genie3_default_seed_1729",
                "tool_id": "genie3",
                "family": "catalog_tf_restricted",
                "algorithm_seed": 1729,
                "wall_time_seconds": 12.5,
                "peak_memory_mb": None,
            }
        ],
        "wall_time_available": True,
        "memory_available": False,
        "deduplication_unit": "inference_artifact",
    }
    assert payload["exports"]["datasets"] == [
        {
            "dataset_id": "dataset-a",
            "truth_zip": (
                "../../../reduced-v1/comparison-fixture/exports/datasets/dataset-a/analysis.zip"
            ),
            "regimes": [
                {
                    "regime": "group_only",
                    "inference_zip": (
                        "../../../reduced-v1/comparison-fixture/exports/"
                        "inference/dataset-a/group_only/joint/analysis.zip"
                    ),
                    "evaluation_zip": (
                        "../../../reduced-v1/comparison-fixture/exports/"
                        "evaluation/dataset-a/group_only/joint/analysis.zip"
                    ),
                }
            ],
        }
    ]
    dataset_links = payload["exports"]["datasets"][0]
    assert (report_path.parent / dataset_links["truth_zip"]).resolve() == (
        run_root / "exports/datasets/dataset-a/analysis.zip"
    ).resolve()
    regime_links = dataset_links["regimes"][0]
    assert (report_path.parent / regime_links["inference_zip"]).resolve() == (
        run_root / "exports/inference/dataset-a/group_only/joint/analysis.zip"
    ).resolve()
    assert (report_path.parent / regime_links["evaluation_zip"]).resolve() == (
        run_root / "exports/evaluation/dataset-a/group_only/joint/analysis.zip"
    ).resolve()


@pytest.mark.parametrize("workflow_kind", ["calibration", "comparison"])
def test_report_bundle_is_identical_while_publishing_and_after_completion(
    tmp_path: Path,
    workflow_kind: str,
) -> None:
    if workflow_kind == "calibration":
        repo_root, workflow_root, _selected = _calibration_fixture(tmp_path)
        renderer = calibration_reporting.render_calibration_report
        kwargs: dict[str, Any] = {"version_root": workflow_root.parents[3]}
    else:
        repo_root, workflow_root, run_root, _statistics = _comparison_fixture(tmp_path)
        renderer = comparison_reporting.render_comparison_report
        kwargs = {"version_root": workflow_root.parents[3], "run_root": run_root}

    publishing_bundle = renderer(
        repo_root=repo_root,
        workflow_root=workflow_root,
        **kwargs,
    )
    workflow_path = workflow_root / "workflow.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    assert workflow["status"] == "publishing"
    workflow["status"] = "complete"
    workflow["reports"] = publishing_bundle
    _write_json(workflow_path, workflow)

    completed_bundle = renderer(
        repo_root=repo_root,
        workflow_root=workflow_root,
        **kwargs,
    )

    assert completed_bundle == publishing_bundle
    for member in ("html", "manifest"):
        assert (
            sha256_file(repo_root / completed_bundle[member]["path"])
            == (publishing_bundle[member]["sha256"])
        )


@pytest.mark.parametrize("workflow_kind", ["calibration", "comparison"])
def test_reporting_rejects_a_flow_id_that_disagrees_with_its_workflow_directory(
    tmp_path: Path,
    workflow_kind: str,
) -> None:
    if workflow_kind == "calibration":
        repo_root, workflow_root, _selected = _calibration_fixture(tmp_path)
        renderer = calibration_reporting.render_calibration_report
        kwargs: dict[str, Any] = {"version_root": workflow_root.parents[3]}
    else:
        repo_root, workflow_root, run_root, _statistics = _comparison_fixture(tmp_path)
        renderer = comparison_reporting.render_comparison_report
        kwargs = {"version_root": workflow_root.parents[3], "run_root": run_root}
    workflow_path = workflow_root / "workflow.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["flow_id"] = "different-flow"
    _write_json(workflow_path, workflow)

    with pytest.raises(BenchmarkConfigError, match="path disagrees with its flow ID"):
        renderer(repo_root=repo_root, workflow_root=workflow_root, **kwargs)


@pytest.mark.parametrize("first_cache_hit", [False, True], ids=["miss-then-hit", "hit-then-miss"])
def test_calibration_report_reconciles_cache_access_without_changing_phase_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_cache_hit: bool,
) -> None:
    repo_root, workflow_root, _selected = _calibration_fixture(tmp_path)
    version_root = workflow_root.parents[3]
    recorded_states = {}
    for index, (stage_id, fidelity) in enumerate(calibration_reporting.CALIBRATION_PHASE_ORDER):
        path = version_root / f"calibration-{stage_id}__{fidelity}" / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["datasets"][0]["dataset_artifact"]["cache_hit"] = (
            first_cache_hit if index == 0 else not first_cache_hit
        )
        _write_json(path, state)
        recorded_states[path] = path.read_bytes()
    captured: dict[str, Any] = {}

    def capture_bundle(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"captured": True}

    monkeypatch.setattr(calibration_reporting, "write_report_bundle", capture_bundle)
    calibration_reporting.render_calibration_report(
        repo_root=repo_root, version_root=version_root, workflow_root=workflow_root,
    )
    assert captured["payload"]["datasets"]["summary"]["dataset_count"] == 2
    assert {path: path.read_bytes() for path in recorded_states} == recorded_states


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("seed", 999),
        ("split_id", "different-split"),
        ("condition_id", "different-condition"),
        ("dataset_manifest", "/different/manifest.json"),
        ("dataset_artifact.digest", "f" * 64),
        ("dataset_artifact.path", "/different/artifact"),
    ],
)
def test_calibration_report_still_rejects_real_dataset_disagreements(
    tmp_path: Path, field: str, replacement: Any,
) -> None:
    repo_root, workflow_root, _selected = _calibration_fixture(tmp_path)
    version_root = workflow_root.parents[3]
    state_path = version_root / "calibration-geometry__confirmation/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    target = state["datasets"][0]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement
    _write_json(state_path, state)

    with pytest.raises(BenchmarkConfigError, match="disagree on dataset.*geometry__confirmation"):
        calibration_reporting.render_calibration_report(
            repo_root=repo_root, version_root=version_root, workflow_root=workflow_root,
        )


@pytest.mark.parametrize("cache_hit", [None, 0, 1, "false"])
def test_calibration_report_rejects_invalid_cache_telemetry(tmp_path: Path, cache_hit: Any) -> None:
    repo_root, workflow_root, _selected = _calibration_fixture(tmp_path)
    version_root = workflow_root.parents[3]
    state_path = version_root / "calibration-geometry__confirmation/state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["datasets"][0]["dataset_artifact"]["cache_hit"] = cache_hit
    _write_json(state_path, state)

    with pytest.raises(BenchmarkConfigError, match="cache_hit must be boolean"):
        calibration_reporting.render_calibration_report(
            repo_root=repo_root, version_root=version_root, workflow_root=workflow_root,
        )


def test_calibration_report_rejects_a_ranking_changed_after_workflow_completion(
    tmp_path: Path,
) -> None:
    repo_root, workflow_root, _selected = _calibration_fixture(tmp_path)
    ranking_path = workflow_root / "stages/convergence__final/ranking.json"
    ranking_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="changed after the workflow was recorded"):
        calibration_reporting.render_calibration_report(
            repo_root=repo_root,
            version_root=workflow_root.parents[3],
            workflow_root=workflow_root,
        )


@pytest.mark.parametrize("workflow_kind", ["calibration", "comparison"])
def test_reporting_rejects_incomplete_workflows(
    tmp_path: Path,
    workflow_kind: str,
) -> None:
    if workflow_kind == "calibration":
        repo_root, workflow_root, _selected = _calibration_fixture(tmp_path)
        renderer = calibration_reporting.render_calibration_report
        kwargs: dict[str, Any] = {"version_root": workflow_root.parents[3]}
    else:
        repo_root, workflow_root, run_root, _statistics = _comparison_fixture(tmp_path)
        renderer = comparison_reporting.render_comparison_report
        kwargs = {"version_root": workflow_root.parents[3], "run_root": run_root}
    workflow_path = workflow_root / "workflow.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow["status"] = "running"
    _write_json(workflow_path, workflow)

    with pytest.raises(BenchmarkConfigError, match="require a publishable workflow"):
        renderer(repo_root=repo_root, workflow_root=workflow_root, **kwargs)


def test_comparison_report_rejects_a_tampered_dataset_artifact_member(
    tmp_path: Path,
) -> None:
    repo_root, workflow_root, run_root, _statistics = _comparison_fixture(tmp_path)
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    manifest_path = Path(state["datasets"][0]["dataset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expression_path = manifest_path.parent / manifest["dataset"]["expression_matrix"]
    original = expression_path.read_text(encoding="utf-8")
    expression_path.write_text(original.replace("G1\t1", "G1\t2"), encoding="utf-8")

    with pytest.raises(
        BenchmarkConfigError,
        match="Invalid content-addressed dataset artifact.*hash differs",
    ):
        comparison_reporting.render_comparison_report(
            repo_root=repo_root,
            version_root=workflow_root.parents[3],
            workflow_root=workflow_root,
            run_root=run_root,
        )
