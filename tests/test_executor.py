from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from spathi_bench.artifacts import sha256_file
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.executor import (
    AndreaExecutor,
    _candidate_space_evidence,
    _candidate_space_identity,
    _dataset_materialization_contract,
    _fingerprint_dataset,
    _materialize_generated_inputs,
    _neutralize_lineage_information,
    _newick,
    _planning_resource_decisions,
)
from spathi_bench.study_store import TaskRecord


def _executor(tmp_path: Path) -> AndreaExecutor:
    return AndreaExecutor(
        repo_root=tmp_path,
        andrea_repo=tmp_path / "andrea",
        spathi_repo=tmp_path / "spathi",
        study_root=tmp_path / "study",
    )


def _task(kind: str, spec: dict[str, Any]) -> TaskRecord:
    spec = {"catalog_definitions": {}, **spec}
    return TaskRecord(
        task_id=f"task-{kind}",
        phase="calibration",
        stage="test",
        kind=kind,
        scientific_digest="a" * 64,
        executor_contract="test",
        spec=spec,
        dependencies=(),
        status="running",
        attempt=1,
        result=None,
        error=None,
    )


def _candidate_hash(seed: str = "a") -> str:
    return seed * 64


def _materialized_record(**values: Any) -> dict[str, Any]:
    dimensions = values.pop(
        "dimensions",
        {"cells": 3, "genes": 3, "transcription_factors": 2, "groups": 2},
    )
    return {
        "dataset_manifest": "/unused/dataset-manifest.json",
        "ground_truth_manifest": "/unused/ground-truth-manifest.json",
        "materialization": {
            "policy": "exact",
            "dimension_policy": {field: "exact" for field in dimensions},
            "status": "matched",
        },
        "requested_dimensions": dict(dimensions),
        "observed_dimensions": dict(dimensions),
        "dimension_deviations": {},
        **values,
    }


def _physical_resource_decision(
    *,
    status: str = "eligible",
    eta_source: str = "fallback_no_cost",
    evidence: str = "conservative_fallback",
) -> dict[str, Any]:
    return {
        "task_id": "run-1",
        "schema_version": "1.0",
        "status": status,
        "reason": "test",
        "estimated_seconds": 7200.0,
        "timeout_seconds": 60.0,
        "eta_source": eta_source,
        "evidence": evidence,
        "advisory": True,
    }


def _logical_resource_decision(physical: dict[str, Any]) -> dict[str, Any]:
    infeasible = [physical["task_id"]] if physical["status"] == "estimated_infeasible_time" else []
    return {
        "schema_version": "1.0",
        "status": "estimated_infeasible_time" if infeasible else "eligible",
        "reason": "test",
        "infeasible_physical_tasks": infeasible,
        "physical_tasks": [physical],
        "advisory": True,
    }


@pytest.mark.parametrize("eta_source", ["fallback_no_cost", "fallback_invalid_cost"])
def test_planning_fallback_without_usable_cost_profile_remains_eligible(
    eta_source: str,
) -> None:
    physical = _physical_resource_decision(eta_source=eta_source)
    plan = {
        "runs": [
            {
                "run_id": "run-1",
                "resource_decision": _logical_resource_decision(physical),
            }
        ]
    }

    decisions = _planning_resource_decisions(plan, ["run-1"])

    assert decisions["run-1"]["status"] == "eligible"
    assert decisions["run-1"]["physical_tasks"][0]["eta_source"] == eta_source


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda run: run.pop("resource_decision"), "omits its resource_decision"),
        (
            lambda run: run["resource_decision"].update(advisory=False),
            "unsupported resource decision",
        ),
        (
            lambda run: run["resource_decision"]["physical_tasks"][0].update(
                status="estimated_infeasible_time"
            ),
            "misuses fallback evidence",
        ),
        (
            lambda run: run["resource_decision"].update(infeasible_physical_tasks=[None]),
            "inconsistent infeasible-task evidence",
        ),
    ],
)
def test_planning_rejects_incomplete_or_non_empirical_infeasibility(
    mutation: Any, message: str
) -> None:
    run = {
        "run_id": "run-1",
        "resource_decision": _logical_resource_decision(_physical_resource_decision()),
    }
    mutation(run)
    with pytest.raises(BenchmarkConfigError, match=message):
        _planning_resource_decisions({"runs": [run]}, ["run-1"])


def _write_truth_candidate_space(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "tf_list.txt").write_text("TF2\nTF1\n", encoding="utf-8")
    (root / "gene_universe.txt").write_text("TF1\nTF2\nG1\n", encoding="utf-8")
    manifest = root / "ground-truth-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate_space": {
                    "sources": "tf_list.txt",
                    "targets": "gene_universe.txt",
                    "allow_self_edges": False,
                }
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_materialized_dataset(
    root: Path, *, simulator_id: str = "dyngen", seed: int = 17
) -> Path:
    dataset = root / "datasets/actual"
    extras = dataset / "extras"
    truth = dataset / "truth"
    extras.mkdir(parents=True)
    truth.mkdir()
    (dataset / "expression.tsv").write_text(
        "gene\tc1\tc2\tc3\nTF1\t1\t2\t3\nTF2\t2\t3\t4\nG1\t3\t4\t5\n",
        encoding="utf-8",
    )
    (extras / "tf_list.txt").write_text("TF1\nTF2\n", encoding="utf-8")
    (extras / "groups.tsv").write_text("column\tcluster\nc1\ta\nc2\ta\nc3\tb\n", encoding="utf-8")
    (truth / "gene_universe.txt").write_text("TF1\nTF2\nG1\n", encoding="utf-8")
    (dataset / "dataset-manifest.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "spec": {
                        "expression": {
                            "genes": 3,
                            "columns": 3,
                            "column_kind": "cells",
                        }
                    },
                    "expression_matrix": "expression.tsv",
                },
                "extras": {
                    "groups": "extras/groups.tsv",
                    "tf_list": "extras/tf_list.txt",
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset / "ground-truth-manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "actual",
                "simulator_id": simulator_id,
                "candidate_space": {
                    "sources": "extras/tf_list.txt",
                    "targets": "truth/gene_universe.txt",
                    "allow_self_edges": False,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = root / "benchmark-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "dataset_id": "actual",
                        "simulator_id": simulator_id,
                        "seed": seed,
                        "dataset_manifest": "datasets/actual/dataset-manifest.json",
                        "ground_truth_manifest": ("datasets/actual/ground-truth-manifest.json"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_dyngen_records_only_valid_backbone_dimension_expansion(tmp_path: Path) -> None:
    contract = _dataset_materialization_contract(
        _write_materialized_dataset(tmp_path),
        {
            "generator": "dyngen",
            "seed": 17,
            "dimensions": {
                "cells": 3,
                "genes": 2,
                "transcription_factors": 1,
                "groups": 2,
            },
            "simulator_parameters": {"num_tfs": 1, "num_targets": 1, "num_hks": 0},
            "capabilities": ["group_labels", "tf_list"],
        },
    )

    assert contract["policy"] == "dyngen_native_materialization"
    assert contract["dimension_policy"] == {
        "cells": "exact",
        "genes": "dyngen_backbone_augmented",
        "transcription_factors": "dyngen_backbone_augmented",
        "groups": "dyngen_observed_milestone_occupancy",
    }
    assert contract["status"] == "accepted_declared_deviation"
    assert contract["observed_dimensions"] == {
        "cells": 3,
        "genes": 3,
        "transcription_factors": 2,
        "groups": 2,
    }
    assert contract["dimension_deviations"] == {
        "genes": {"requested": 2, "observed": 3, "delta": 1},
        "transcription_factors": {"requested": 1, "observed": 2, "delta": 1},
    }


@pytest.mark.parametrize("generator", ["scale-stress", "scmultisim", "sergio", "boolode"])
def test_exact_simulators_require_exact_materialized_dimensions(
    tmp_path: Path, generator: str
) -> None:
    with pytest.raises(BenchmarkConfigError, match="violated its exact dimensions"):
        _dataset_materialization_contract(
            _write_materialized_dataset(
                tmp_path,
                simulator_id=("spathi_scale_stress" if generator == "scale-stress" else generator),
            ),
            {
                "generator": generator,
                "seed": 17,
                "dimensions": {
                    "cells": 3,
                    "genes": 2,
                    "transcription_factors": 1,
                    "groups": 2,
                },
                "capabilities": ["group_labels", "tf_list"],
            },
        )


def test_exact_simulator_records_matched_policy(tmp_path: Path) -> None:
    contract = _dataset_materialization_contract(
        _write_materialized_dataset(tmp_path, simulator_id="scmultisim"),
        {
            "generator": "scmultisim",
            "seed": 17,
            "dimensions": {
                "cells": 3,
                "genes": 3,
                "transcription_factors": 2,
                "groups": 2,
            },
            "capabilities": ["group_labels", "tf_list"],
        },
    )

    assert contract["policy"] == "exact"
    assert contract["status"] == "matched"
    assert set(contract["dimension_policy"].values()) == {"exact"}


def test_dyngen_requires_exact_cells(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkConfigError, match="violated its exact dimensions"):
        _dataset_materialization_contract(
            _write_materialized_dataset(tmp_path),
            {
                "generator": "dyngen",
                "seed": 17,
                "dimensions": {
                    "cells": 2,
                    "genes": 2,
                    "transcription_factors": 1,
                    "groups": 2,
                },
                "simulator_parameters": {"num_tfs": 1, "num_targets": 1, "num_hks": 0},
                "capabilities": ["group_labels", "tf_list"],
            },
        )


def test_dyngen_accepts_and_records_unoccupied_declared_milestones(tmp_path: Path) -> None:
    contract = _dataset_materialization_contract(
        _write_materialized_dataset(tmp_path),
        {
            "generator": "dyngen",
            "seed": 17,
            "dimensions": {
                "cells": 3,
                "genes": 2,
                "transcription_factors": 1,
                "groups": 3,
            },
            "simulator_parameters": {"num_tfs": 1, "num_targets": 1, "num_hks": 0},
            "capabilities": ["group_labels", "tf_list"],
        },
    )

    assert contract["status"] == "accepted_declared_deviation"
    assert contract["observed_dimensions"]["groups"] == 2
    assert contract["dimension_deviations"]["groups"] == {
        "requested": 3,
        "observed": 2,
        "delta": -1,
    }


def test_dyngen_rejects_more_groups_than_declared_milestones(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkConfigError, match="milestone-occupancy contract"):
        _dataset_materialization_contract(
            _write_materialized_dataset(tmp_path),
            {
                "generator": "dyngen",
                "seed": 17,
                "dimensions": {
                    "cells": 3,
                    "genes": 2,
                    "transcription_factors": 1,
                    "groups": 1,
                },
                "simulator_parameters": {"num_tfs": 1, "num_targets": 1, "num_hks": 0},
                "capabilities": ["group_labels", "tf_list"],
            },
        )


def test_dyngen_rejects_single_observed_group(tmp_path: Path) -> None:
    manifest = _write_materialized_dataset(tmp_path)
    (tmp_path / "datasets/actual/extras/groups.tsv").write_text(
        "column\tcluster\nc1\ta\nc2\ta\nc3\ta\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkConfigError, match="milestone-occupancy contract"):
        _dataset_materialization_contract(
            manifest,
            {
                "generator": "dyngen",
                "seed": 17,
                "dimensions": {
                    "cells": 3,
                    "genes": 2,
                    "transcription_factors": 1,
                    "groups": 2,
                },
                "simulator_parameters": {"num_tfs": 1, "num_targets": 1, "num_hks": 0},
                "capabilities": ["group_labels", "tf_list"],
            },
        )


def test_dyngen_rejects_unexplained_gene_expansion(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkConfigError, match="backbone-augmentation contract"):
        _dataset_materialization_contract(
            _write_materialized_dataset(tmp_path),
            {
                "generator": "dyngen",
                "seed": 17,
                "dimensions": {
                    "cells": 3,
                    "genes": 3,
                    "transcription_factors": 1,
                    "groups": 2,
                },
                "simulator_parameters": {"num_tfs": 1, "num_targets": 2, "num_hks": 0},
                "capabilities": ["group_labels", "tf_list"],
            },
        )


def test_materialization_rejects_missing_promised_capability(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkConfigError, match="omits promised capabilities: lineage_tree"):
        _dataset_materialization_contract(
            _write_materialized_dataset(tmp_path),
            {
                "generator": "dyngen",
                "seed": 17,
                "dimensions": {
                    "cells": 3,
                    "genes": 3,
                    "transcription_factors": 2,
                    "groups": 2,
                },
                "capabilities": ["group_labels", "tf_list", "lineage_tree"],
            },
        )


def test_generated_scmultisim_inputs_are_seeded_and_dimension_exact(tmp_path: Path) -> None:
    recipes = {
        "regulatory_network": {
            "kind": "directed_scale_free_v1",
            "path": "inputs/network.tsv",
            "genes": 20,
            "transcription_factors": 4,
            "edges_per_target": 2,
            "seed": 71,
        },
        "tree_newick": {
            "kind": "balanced_population_tree_v1",
            "path": "inputs/tree.nwk",
            "groups": 3,
        },
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _materialize_generated_inputs(first, recipes)
    _materialize_generated_inputs(second, recipes)

    assert (first / "inputs/network.tsv").read_bytes() == (
        second / "inputs/network.tsv"
    ).read_bytes()
    lines = (first / "inputs/network.tsv").read_text().splitlines()
    assert len(lines) == 1 + (20 - 4 - 1) * 2
    genes = {value for line in lines[1:] for value in line.split("\t")[:2]}
    assert len(genes) == 19
    assert {f"TF{index:02d}" for index in range(1, 5)} <= genes
    assert (first / "inputs/tree.nwk").read_text() == ("((pop_1:1,pop_2:1):1,pop_3:1);\n")
    assert _newick(1) == "(pop_1:1);\n"


def test_lineage_rates_are_neutralized_and_truth_fingerprint_is_refreshed(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    dataset = package / "datasets" / "d1"
    extras = dataset / "extras"
    truth = dataset / "truth"
    extras.mkdir(parents=True)
    truth.mkdir()
    (dataset / "expression.tsv").write_text("gene\tc1\nTF1\t1\nG1\t2\n")
    (extras / "groups.tsv").write_text("cell\tcluster\nc1\ta\n")
    (extras / "lineage_tree.tsv").write_text(
        "child\tparent\tgain_rate\tloss_rate\na\t__root__\t0\t0\nb\ta\t0.91\t0.17\n"
    )
    dataset_manifest = dataset / "dataset-manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "dataset": {"expression_matrix": "expression.tsv"},
                "extras": {
                    "groups": "extras/groups.tsv",
                    "lineage_tree": "extras/lineage_tree.tsv",
                },
            }
        )
    )
    truth_manifest = dataset / "ground-truth-manifest.json"
    truth_manifest.write_text(json.dumps({"dataset_id": "d1", "dataset_fingerprint": {}}))
    benchmark = package / "benchmark-manifest.json"
    benchmark.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "dataset_id": "d1",
                        "dataset_manifest": "datasets/d1/dataset-manifest.json",
                        "ground_truth_manifest": "datasets/d1/ground-truth-manifest.json",
                    }
                ]
            }
        )
    )

    evidence = _neutralize_lineage_information(benchmark)

    assert len(evidence) == 1
    lines = (extras / "lineage_tree.tsv").read_text().splitlines()
    assert lines[1].endswith("\t0\t0")
    assert lines[2].endswith("\t0.5\t0.5")
    refreshed = json.loads(truth_manifest.read_text())["dataset_fingerprint"]
    assert refreshed == _fingerprint_dataset(dataset_manifest)


def test_candidate_space_evidence_is_semantic_portable_and_records_exclusions(
    tmp_path: Path,
) -> None:
    manifest = _write_truth_candidate_space(tmp_path / "truth")
    identity = _candidate_space_identity(manifest)
    evidence = _candidate_space_evidence(
        identity,
        {
            "ground_truth": {
                "candidate_space": {
                    "mode": "explicit",
                    "sources": "tf_list.txt",
                    "targets": "gene_universe.txt",
                    "allow_self_edges": False,
                    "n_sources": 2,
                    "n_targets": 3,
                    "n_source_target_overlap": 2,
                    "truth_rows_total": 9,
                    "n_truth_rows_outside_candidate_space_by_level": {
                        "topology": 1,
                        "directed": 2,
                        "signed": 2,
                    },
                    "n_truth_edges_outside_candidate_space_by_level": {
                        "topology": 1,
                        "directed": 2,
                        "signed": 2,
                    },
                    "truth_filtering_by_context": [{"context": "group:a"}],
                    "n_candidates_by_level": {
                        "topology": 3,
                        "directed": 4,
                        "signed": 8,
                    },
                }
            }
        },
    )

    assert identity["n_sources"] == 2
    assert identity["n_targets"] == 3
    assert len(identity["candidate_space_sha256"]) == 64
    assert not any(str(tmp_path) in str(value) for value in evidence["identity"].values())
    assert evidence["evaluation"]["n_truth_rows_outside_candidate_space_by_level"]["directed"] == 2


def test_docker_image_identity_is_thread_safe_and_durably_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)
    calls: list[list[str]] = []

    def inspect(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "Id": "sha256:" + "a" * 64,
                        "RepoDigests": ["example/tool@sha256:" + "b" * 64],
                    }
                ]
            ),
        )

    monkeypatch.setattr("spathi_bench.executor.subprocess.run", inspect)
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(executor._docker_image_identity, ["example/tool:1"] * 8))

    assert len(calls) == 8
    assert len({json.dumps(value, sort_keys=True) for value in identities}) == 1
    lock = json.loads((tmp_path / "study/resolved-images.json").read_text())
    assert lock["images"]["example/tool:1"] == {
        "image_id": "sha256:" + "a" * 64,
        "repo_digest": "example/tool@sha256:" + "b" * 64,
    }


def test_resume_rejects_a_mutated_docker_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)

    def result(image_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{"Id": image_id, "RepoDigests": []}]),
        )

    monkeypatch.setattr(
        "spathi_bench.executor.subprocess.run",
        lambda *_args, **_kwargs: result("sha256:" + "a" * 64),
    )
    executor._docker_image_identity("example/tool:1")

    resumed = _executor(tmp_path)
    monkeypatch.setattr(
        "spathi_bench.executor.subprocess.run",
        lambda *_args, **_kwargs: result("sha256:" + "c" * 64),
    )
    with pytest.raises(BenchmarkConfigError, match="changed within this study"):
        resumed._docker_image_identity("example/tool:1")


def test_inference_image_provenance_covers_catalog_and_custom_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)
    toolspec = tmp_path / "andrea/andrea/catalog_inference_tools/tools/genie3/toolspec.json"
    toolspec.parent.mkdir(parents=True)
    toolspec.write_text(
        json.dumps({"id": "genie3", "docker_image": "catalog/genie3:1"}),
        encoding="utf-8",
    )
    calls: list[tuple[str, bool]] = []

    def identity(image: str, *, freeze_mutable_tag: bool = True) -> dict[str, Any]:
        calls.append((image, freeze_mutable_tag))
        return {
            "requested_image": image,
            "image_id": "sha256:" + "a" * 64,
            "repo_digest": None,
        }

    monkeypatch.setattr(executor, "_docker_image_identity", identity)
    identities = executor._inference_image_identities(
        {
            "runs": [
                {"run_id": "g1", "tool_id": "genie3"},
                {"run_id": "g2", "tool_id": "genie3"},
                {"run_id": "s1", "tool_id": "custom_s1"},
            ]
        },
        {
            "tools": [
                {
                    "run_id": "s1",
                    "docker_image": "local/spathi:content-hash",
                }
            ]
        },
    )

    assert calls == [
        ("catalog/genie3:1", True),
        ("local/spathi:content-hash", False),
    ]
    assert identities["genie3"]["origin"] == "andrea_catalog"
    assert identities["genie3"]["definition_sha256"] == sha256_file(toolspec)
    assert identities["custom_s1"]["origin"] == "custom"


def test_inference_artifact_identity_includes_resource_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)
    captured: dict[str, Any] = {}

    class CapturedSpec(RuntimeError):
        pass

    def capture(kind: str, spec: dict[str, Any]) -> Any:
        captured["kind"] = kind
        captured["spec"] = spec
        raise CapturedSpec

    monkeypatch.setattr(executor, "_inference_image_identities", lambda *_args: {})
    monkeypatch.setattr(executor.store, "key_for", capture)

    with pytest.raises(CapturedSpec):
        executor.infer(
            dataset_manifest="/unused/dataset-manifest.json",
            tools_params={"runs": []},
            custom_tools=None,
            resources={"cpu_threads": 8, "memory_bytes": 16 * 1024**3},
            purpose="comparison",
        )

    assert captured["kind"] == "inference"
    assert captured["spec"]["resource_contract"] == "machine-relative-memory-slots"


def test_generation_identity_covers_catalog_simulator_and_host_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)
    simulatorspec = (
        tmp_path
        / "andrea/andrea/catalog_simulation_data_tools/simulators/dyngen/simulatorspec.json"
    )
    simulatorspec.parent.mkdir(parents=True)
    simulatorspec.write_text(
        json.dumps({"id": "dyngen", "docker_image": "catalog/dyngen:1"}),
        encoding="utf-8",
    )
    calls: list[tuple[str, bool]] = []

    def identity(image: str, *, freeze_mutable_tag: bool = True) -> dict[str, Any]:
        calls.append((image, freeze_mutable_tag))
        return {
            "requested_image": image,
            "image_id": "sha256:" + "d" * 64,
            "repo_digest": "catalog/dyngen@sha256:" + "e" * 64,
        }

    monkeypatch.setattr(executor, "_docker_image_identity", identity)
    catalog = executor._generation_identity({"generator": "dyngen"})
    host = executor._generation_identity({"generator": "scale-stress"})

    assert calls == [("catalog/dyngen:1", True)]
    assert catalog == {
        "requested_image": "catalog/dyngen:1",
        "image_id": "sha256:" + "d" * 64,
        "repo_digest": "catalog/dyngen@sha256:" + "e" * 64,
        "origin": "andrea_simulator_catalog",
        "contract": "andrea-catalog-definition-v1",
        "catalog_kind": "simulator",
        "catalog_id": "dyngen",
        "definition_sha256": sha256_file(simulatorspec),
    }
    assert host["origin"] == "benchmark_host_generator"
    assert host["generator_id"] == "spathi_scale_stress"
    assert isinstance(host["generator_contract"], str)


def test_equal_candidate_selector_has_no_reference_or_privilege(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    dependencies: dict[str, dict[str, Any]] = {
        "screen": {
            "kind": "calibration_screen",
            "rows": [
                {
                    "configuration_id": "candidate-01",
                    "dataset_id": "development-1",
                    "metric": "aupr",
                    "status": "completed",
                    "candidate_space_sha256": _candidate_hash(),
                    "value": 0.4,
                    "duration_seconds": 10,
                    "parameters": {"max_depth": 2},
                },
                {
                    "configuration_id": "candidate-02",
                    "dataset_id": "development-1",
                    "metric": "aupr",
                    "status": "completed",
                    "candidate_space_sha256": _candidate_hash(),
                    "value": 0.5,
                    "duration_seconds": 12,
                    "parameters": {"max_depth": 4},
                },
            ],
        }
    }
    result = executor.execute_task(
        _task(
            "select_calibration_finalists",
            {
                "candidate_ids": ["candidate-01", "candidate-02"],
                "protocol": {"finalists": 1},
            },
        ),
        dependencies,
    )
    assert result["selected_candidates"] == [{"id": "candidate-02", "parameters": {"max_depth": 4}}]
    assert result["decision"]["decision"] == "advance_equal_candidates"


def test_winner_stability_is_measured_between_datasets_after_seed_averaging(
    tmp_path: Path,
) -> None:
    executor = _executor(tmp_path)
    rows = []
    # Candidate A has seed variation within each dataset but identical dataset
    # means. Candidate B has no within-dataset variation but transfers less
    # consistently between datasets. Their global mean is exactly equal.
    for dataset_id, a_values, b_values in (
        ("validation-1", (0.7, 0.9), (0.7, 0.7)),
        ("validation-2", (0.75, 0.85), (0.9, 0.9)),
    ):
        for configuration_id, values in (("candidate-a", a_values), ("candidate-b", b_values)):
            for seed, value in enumerate(values, start=1):
                rows.append(
                    {
                        "configuration_id": configuration_id,
                        "dataset_id": dataset_id,
                        "algorithm_seed": seed,
                        "n_estimators": 50,
                        "metric": "aupr",
                        "status": "completed",
                        "candidate_space_sha256": _candidate_hash(dataset_id[-1]),
                        "value": value,
                        "parameters": {"max_depth": 4},
                    }
                )

    result = executor.execute_task(
        _task(
            "select_calibration_winner",
            {
                "protocol": {"practical_equivalence_fraction": 0.0},
            },
        ),
        {"validation": {"kind": "calibration_validate", "rows": rows}},
    )

    assert result["winner"]["configuration_id"] == "candidate-a"
    ranking = {row["configuration_id"]: row for row in result["rows"]}
    assert ranking["candidate-a"]["stability_sd"] == pytest.approx(0.0)
    assert ranking["candidate-b"]["stability_sd"] > 0.0
    assert ranking["candidate-a"]["stability_unit"] == ("validation_dataset_after_seed_averaging")


def test_validation_uses_one_physical_forest_per_finalist_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)
    captured: dict[str, Any] = {}

    def fake_panel(**kwargs: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        captured.update(kwargs)
        return [], [{"key": "forest"}]

    monkeypatch.setattr(executor, "_run_convergence_panel", fake_panel)
    dependencies = {
        "finalists": {
            "kind": "select_calibration_finalists",
            "selected_candidates": [
                {"id": "c1", "parameters": {}},
                {"id": "c2", "parameters": {}},
            ],
        }
    }
    result = executor.execute_task(
        _task(
            "calibration_validate",
            {
                "algorithm_seeds": [1, 2],
                "exact_prefixes": [25, 50, 100, 250],
            },
        ),
        dependencies,
    )
    assert captured["candidates"] == dependencies["finalists"]["selected_candidates"]
    assert result["summary"]["physical_fits"] == 4
    assert result["summary"]["logical_results"] == 16


def test_comparison_request_applies_same_timeout_to_every_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)
    monkeypatch.setattr(
        executor,
        "_spathi_image",
        lambda _spec: {"tag": "local/spathi:test"},
    )
    spec = {
        "dataset": {"seed": 4},
        "resources": {"cpu_threads": 32},
        "timeout_seconds": 7200,
        "routes": [
            {
                "id": "spathi",
                "tool_id": "spathi",
                "configuration_id": "winner",
                "delivery": "group_native",
                "parameters": {},
                "seed_parameter": "random_seed",
                "output_semantics": "directed_tf_to_target",
                "max_threads": None,
            },
            {
                "id": "genie3",
                "tool_id": "genie3",
                "configuration_id": "default",
                "delivery": "group_emulated",
                "parameters": {},
                "seed_parameter": "seed",
                "output_semantics": "directed_tf_to_target",
                "max_threads": 8,
            },
            {
                "id": "clr",
                "tool_id": "clr",
                "configuration_id": "default",
                "delivery": "group_emulated",
                "parameters": {},
                "seed_parameter": None,
                "output_semantics": "undirected_topology",
                "max_threads": 1,
            },
        ],
    }
    tools, custom, _metadata = executor._comparison_requests(
        spec,
        {"parameters": {"n_estimators": 50}},
    )
    assert {run["resources"]["timeout_seconds"] for run in tools["runs"]} == {7200}
    assert {run["run_id"]: run["resources"]["threads"] for run in tools["runs"]} == {
        "spathi": 8,
        "genie3": 8,
        "clr": 1,
    }
    assert custom is not None and len(custom["tools"]) == 1


def test_convergence_artifact_identity_includes_exact_container_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)
    captured: dict[str, Any] = {}

    class CapturedSpec(RuntimeError):
        pass

    def capture(kind: str, spec: dict[str, Any]) -> Any:
        captured["kind"] = kind
        captured["spec"] = spec
        raise CapturedSpec

    monkeypatch.setattr(executor.store, "key_for", capture)
    image = {
        "tag": "local/spathi:content",
        "id": "sha256:" + "a" * 64,
        "repo_digest": "local/spathi@sha256:" + "b" * 64,
        "fingerprint": "c" * 64,
    }
    with pytest.raises(CapturedSpec):
        executor._run_convergence_forest(
            record={
                "dataset_manifest": "/unused/dataset-manifest.json",
                "ground_truth_manifest": "/unused/ground-truth-manifest.json",
            },
            configuration_id="candidate-01",
            parameters={"max_depth": 4},
            seed=71,
            prefixes=(25, 50, 100, 250),
            image=image,
            threads=4,
            memory_bytes=2 * 1024**3,
            purpose="validation",
        )

    assert captured["kind"] == "convergence"
    assert captured["spec"]["image"] == {
        "requested_image": image["tag"],
        "image_id": image["id"],
        "repo_digest": image["repo_digest"],
        "source_fingerprint": image["fingerprint"],
    }


def test_comparison_with_no_completed_runs_preserves_outcomes_without_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)
    decision = tmp_path / "decision.json"
    report = tmp_path / "run_report.json"
    decision.write_text(
        json.dumps(
            {
                "status": "completed_with_failures",
                "infeasible_runs": [],
                "preflight": {"runs": {"skipped": {}}},
                "image_identities": {"genie3": {"image_id": "sha256:" + "a" * 64}},
            }
        ),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "tools": {
                    "completed": [],
                    "resource_outcomes": {
                        "genie3": {
                            "status": "timeout",
                            "reason": "wall_time_limit_exceeded",
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        executor,
        "_comparison_requests",
        lambda *_args, **_kwargs: (
            {"runs": [{"run_id": "genie3"}]},
            None,
            {"genie3": {"method_id": "genie3"}},
        ),
    )
    monkeypatch.setattr(
        executor,
        "infer",
        lambda **_kwargs: {
            "members": {"decision": "decision.json", "run_report": "run_report.json"}
        },
    )
    monkeypatch.setattr(
        "spathi_bench.executor.resolve_artifact_reference",
        lambda _store, _reference, member: {"decision": decision, "run_report": report}[member],
    )

    def unexpected_evaluation(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("an empty completed inventory must not be evaluated")

    monkeypatch.setattr(executor, "evaluate", unexpected_evaluation)
    result = executor._execute_comparison(
        {
            "dataset": {"id": "d1"},
            "resources": {"cpu_threads": 8, "memory_bytes": 8 * 1024**3},
            "routes": [],
            "families": [],
            "timeout_seconds": 60,
        },
        {
            "generated": {
                "kind": "generate_dataset",
                "record": _materialized_record(),
            },
            "winner": {
                "kind": "select_calibration_winner",
                "winner": {"parameters": {"n_estimators": 50}},
            },
        },
    )

    assert result["rows"] == []
    assert result["failures"] == [
        {
            "dataset_id": "d1",
            "method_id": "genie3",
            "status": "timeout",
            "reason": "wall_time_limit_exceeded",
        }
    ]
    assert result["decisions"][0]["decision"] == "execution_did_not_complete"
    assert result["summary"]["resource_outcomes"] == {
        "genie3": {"status": "timeout", "reason": "wall_time_limit_exceeded"}
    }


def test_comparison_coverage_counts_declared_failures_per_applicable_family(
    tmp_path: Path,
) -> None:
    executor = _executor(tmp_path)
    coverage = executor._comparison_coverage(
        spec={
            "dataset": {"id": "d1", "condition_id": "t1", "replicate": 1},
            "families": [
                {"id": "family-a", "route_ids": ["spathi", "genie3"]},
                {"id": "conditional-absent", "route_ids": ["spathi", "simic"]},
            ],
        },
        metadata={
            "spathi": {
                "method_id": "spathi",
                "configuration_id": "selected",
                "requested_threads": 8,
                "effective_threads": 8,
            },
            "genie3": {
                "method_id": "genie3",
                "configuration_id": "upstream-default",
                "requested_threads": 8,
                "effective_threads": 8,
            },
        },
        statuses={
            "spathi": {"status": "completed", "reason": "execution_completed"},
            "genie3": {"status": "timeout", "reason": "wall_time_limit_exceeded"},
        },
    )

    assert [(row["family"], row["method_id"], row["status"]) for row in coverage] == [
        ("family-a", "genie3", "timeout"),
        ("family-a", "spathi", "completed"),
    ]


def test_failed_family_member_does_not_hide_completed_member_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)
    decision = tmp_path / "decision.json"
    report = tmp_path / "run_report.json"
    evidence = tmp_path / "candidate-space-evidence.json"
    decision.write_text(
        json.dumps(
            {
                "status": "completed_with_failures",
                "infeasible_runs": [],
                "preflight": {"runs": {"skipped": {}}},
                "image_identities": {},
            }
        ),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "tools": {
                    "completed": ["spathi"],
                    "resource_outcomes": {
                        "spathi": {"status": "completed", "reason": "execution_completed"},
                        "genie3": {"status": "timeout", "reason": "wall_time_limit_exceeded"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    evidence.write_text(json.dumps({"identity": {}, "evaluation": {}}), encoding="utf-8")
    metadata = {
        "spathi": {"method_id": "spathi", "output_semantics": "directed_tf_to_target"},
        "genie3": {"method_id": "genie3", "output_semantics": "directed_tf_to_target"},
    }
    monkeypatch.setattr(
        executor,
        "_comparison_requests",
        lambda *_args, **_kwargs: ({"runs": []}, None, metadata),
    )
    monkeypatch.setattr(
        executor,
        "infer",
        lambda **_kwargs: {
            "members": {"decision": "decision.json", "run_report": "run_report.json"}
        },
    )

    def resolve(_store: Any, _reference: Any, member: str) -> Path:
        return {
            "decision": decision,
            "run_report": report,
            "candidate_space_evidence": evidence,
        }[member]

    monkeypatch.setattr("spathi_bench.executor.resolve_artifact_reference", resolve)
    monkeypatch.setattr(executor, "evaluate", lambda **_kwargs: {"members": {}})
    captured: dict[str, Any] = {}

    def metric_rows(_evaluation: Any, **kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return [
            {
                "dataset_id": "d1",
                "family": "family-a",
                "method_id": "spathi",
                "metric": "aupr",
                "value": 0.5,
                "status": "completed",
            }
        ]

    monkeypatch.setattr(executor, "_metric_rows", metric_rows)
    result = executor._execute_comparison(
        {
            "dataset": {"id": "d1", "condition_id": "t1", "replicate": 1},
            "resources": {"cpu_threads": 8, "memory_bytes": 8 * 1024**3},
            "routes": [],
            "families": [{"id": "family-a", "route_ids": ["spathi", "genie3"]}],
            "timeout_seconds": 60,
        },
        {
            "generated": {
                "kind": "generate_dataset",
                "record": _materialized_record(),
            },
            "winner": {
                "kind": "select_calibration_winner",
                "winner": {"parameters": {"n_estimators": 50}},
            },
        },
    )

    assert set(captured["runs"]) == {"spathi"}
    assert result["rows"][0]["method_id"] == "spathi"
    assert result["failures"][0]["method_id"] == "genie3"
    assert {row["method_id"] for row in result["execution_coverage"]} == {
        "spathi",
        "genie3",
    }


def test_prefix_run_report_is_accepted_by_andrea_evaluation_contract(tmp_path: Path) -> None:
    from andrea.core.commands.evaluate_inference.evaluation import (
        validate_inference_analysis_inputs,
    )

    source = tmp_path / "network.csv"
    source.write_text(
        "source,target,score,sign,evidence,context\nTF1,G1,0.8,,model,group:a\n",
        encoding="utf-8",
    )
    output = tmp_path / "logical"
    rows, contexts = AndreaExecutor._merge_prefix_network(source, output, "candidate_trees_25")
    report = AndreaExecutor._prefix_run_report(
        output=output,
        run_id="candidate_trees_25",
        dataset_id="synthetic-1",
        dataset={
            "spec": {
                "expression": {
                    "column_kind": "cells",
                    "expression_profile": "scrna",
                    "genes": 2,
                    "columns": 4,
                }
            }
        },
        fingerprint={"algorithm": "sha256", "value": "a" * 64},
        contexts=contexts,
        rows=rows,
        duration_seconds=None,
        n_estimators=25,
        physical_digest="b" * 64,
    )
    report_path = output / "run_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    validate_inference_analysis_inputs(run_report_path=report_path)


def test_failed_scale_prevents_only_larger_scale_execution(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    blocked = executor._blocked_by_predecessor(
        {
            "previous": {
                "kind": "scalability_measurement",
                "rows": [{"status": "timeout"}],
            }
        },
        "previous",
    )
    assert blocked == {"status": "timeout"}
    assert executor._blocked_by_predecessor({}, None) is None


def test_unlaunched_scale_is_not_relabelled_as_a_timeout(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    result = executor._execute_scalability(
        {
            "dataset": {
                "id": "p-medium-r02",
                "condition_id": "p-medium",
                "dimensions": {"cells": 1000, "genes": 1000},
            },
            "route": {"id": "genie3", "max_threads": 8},
            "measurement": "size_ladder",
            "requested_threads": 8,
            "effective_threads": 8,
            "advance_dependency": "previous",
            "resources": {"cpu_threads": 8, "memory_bytes": 16 * 1024**3},
        },
        {
            "generated": {
                "kind": "generate_dataset",
                "record": _materialized_record(
                    dimensions={
                        "cells": 1000,
                        "genes": 1000,
                        "transcription_factors": 100,
                        "groups": 3,
                    }
                ),
            },
            "previous": {
                "kind": "scalability_measurement",
                "rows": [
                    {
                        "status": "not_attempted",
                        "reason": "desisted_after_andrea_resource_outcome",
                        "predecessor_status": "timeout",
                        "predecessor_reason": "wall_time_limit_exceeded",
                    }
                ],
            },
        },
    )

    [row] = result["rows"]
    assert row["status"] == "not_attempted"
    assert row["reason"] == "desisted_after_andrea_resource_outcome"
    assert row["predecessor_status"] == "timeout"
    assert row["predecessor_reason"] == "wall_time_limit_exceeded"


def test_scalability_gives_each_isolated_method_the_frozen_ram_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = _executor(tmp_path)
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "status": "estimated_infeasible_time",
                "image_identities": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        executor,
        "_comparison_requests",
        lambda *_args, **_kwargs: (
            {"runs": [{"run_id": "spathi", "resources": {}}]},
            None,
            {"spathi": {"method_id": "spathi"}},
        ),
    )
    captured: dict[str, Any] = {}

    def infer(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"members": {"decision": "decision.json"}}

    monkeypatch.setattr(executor, "infer", infer)
    monkeypatch.setattr(
        "spathi_bench.executor.resolve_artifact_reference",
        lambda _store, _reference, _member: decision,
    )
    memory_bytes = 1024 * 1024**3
    result = executor._execute_scalability(
        {
            "dataset": {
                "id": "p-xlarge-r01",
                "condition_id": "p-xlarge",
                "dimensions": {"cells": 10_000, "genes": 20_000},
            },
            "route": {"id": "spathi", "max_threads": None},
            "measurement": "size_ladder",
            "requested_threads": 120,
            "effective_threads": 120,
            "advance_dependency": None,
            "timeout_seconds": 14_400,
            "resources": {"cpu_threads": 120, "memory_bytes": memory_bytes},
        },
        {
            "generated": {
                "kind": "generate_dataset",
                "record": _materialized_record(
                    dimensions={
                        "cells": 10_000,
                        "genes": 20_000,
                        "transcription_factors": 1_500,
                        "groups": 3,
                    }
                ),
            },
            "winner": {
                "kind": "select_calibration_winner",
                "winner": {"parameters": {"n_estimators": 250}},
            },
        },
    )

    assert captured["tools_params"]["runs"][0]["resources"] == {
        "threads": 120,
        "ram_gb": 1024.0,
        "timeout_seconds": 14_400,
    }
    assert result["rows"][0]["status"] == "estimated_infeasible_time"
    assert result["rows"][0]["ram_gb"] == 1024.0


def test_statistics_keep_repeated_method_ids_separate_between_families(
    tmp_path: Path,
) -> None:
    executor = _executor(tmp_path)
    rows: list[dict[str, Any]] = []
    for index in range(3):
        dataset_id = f"dataset-{index}"
        for family, competitor in (("family-a", "method-a"), ("family-b", "method-b")):
            rows.extend(
                [
                    {
                        "dataset_id": dataset_id,
                        "family": family,
                        "method_id": "spathi",
                        "metric": "aupr",
                        "status": "completed",
                        "candidate_space_sha256": _candidate_hash(str(index + 1)),
                        "value": 0.8 - index * 0.01,
                    },
                    {
                        "dataset_id": dataset_id,
                        "family": family,
                        "method_id": competitor,
                        "metric": "aupr",
                        "status": "completed",
                        "candidate_space_sha256": _candidate_hash(str(index + 1)),
                        "value": 0.5 + index * 0.01,
                    },
                ]
            )

    result = executor.execute_task(
        _task(
            "comparison_statistics",
            {
                "families": [
                    {"id": "family-a", "route_ids": ["spathi", "method-a"]},
                    {"id": "family-b", "route_ids": ["spathi", "method-b"]},
                ],
                "statistics": {
                    "metrics": ["aupr"],
                    "alpha": 0.05,
                    "primary_family": "family-a",
                    "estimand": {
                        "scenario_weighting": "equal_replicates_per_scenario",
                        "sampling_unit": "independent_seeded_dataset",
                        "missing_data_rule": "complete_blocks_within_family_and_metric",
                    },
                },
            },
        ),
        {"comparison": {"kind": "comparison_dataset", "rows": rows}},
    )

    analyses = result["statistics"]["analyses"]
    assert [(analysis["family"], analysis["status"]) for analysis in analyses] == [
        ("family-a", "ok"),
        ("family-b", "ok"),
    ]
    assert all(analysis["n_complete_blocks"] == 3 for analysis in analyses)
    assert result["summary"]["families"] == [
        {"id": "family-a", "route_ids": ["spathi", "method-a"]},
        {"id": "family-b", "route_ids": ["spathi", "method-b"]},
    ]


def test_comparable_block_rejects_mixed_candidate_spaces(tmp_path: Path) -> None:
    executor = _executor(tmp_path)
    with pytest.raises(BenchmarkConfigError, match="different candidate spaces"):
        executor._require_comparable_candidate_spaces(
            [
                {
                    "dataset_id": "d1",
                    "method_id": "spathi",
                    "metric": "aupr",
                    "status": "completed",
                    "candidate_space_sha256": _candidate_hash("a"),
                },
                {
                    "dataset_id": "d1",
                    "method_id": "genie3",
                    "metric": "aupr",
                    "status": "completed",
                    "candidate_space_sha256": _candidate_hash("b"),
                },
            ]
        )
