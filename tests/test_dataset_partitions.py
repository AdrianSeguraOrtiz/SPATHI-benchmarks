from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.dataset_partitions import (
    build_dataset_partition_contract,
    validate_dataset_partition_exclusivity,
)


def _scenario(
    *,
    scenario_id: str,
    seed: int | None,
    input_path: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "id": scenario_id,
        "data_axes": {
            "measurement": "rna_expression",
            "resolution": "single_cell",
            "column_kind": "cells",
            "experimental_design": "trajectory",
        },
        "truth_requirements": {"contexts": ["global", "group"]},
        "organism": {"taxonomic_group": "synthetic", "ncbi_taxon_id": None},
        "requested_extras": ["groups", "tf_list"],
    }
    if seed is not None:
        payload["base_seed"] = seed
    if input_path is not None:
        payload["inputs"] = {"regulatory_network": {"path": input_path, "format": "tsv"}}
    return payload


def _runs(*, run_id: str, seed: int | None, replicates: int = 1) -> dict[str, Any]:
    run: dict[str, Any] = {
        "run_id": run_id,
        "simulator_id": "dyngen",
        "replicates": replicates,
        "native_outputs": [],
        "params": {"num_cells": 30},
    }
    if seed is not None:
        run["base_seed"] = seed
    return {
        "schema_version": "1.0",
        "runs": [run],
    }


def _stable_seed(request_id: str, semantic_key: str, simulator_id: str) -> int:
    digest = hashlib.sha256(f"{request_id}|{semantic_key}|{simulator_id}".encode()).hexdigest()
    return (int(digest[:8], 16) % 2_147_483_646) + 1


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path.as_posix()


def _job(
    repo_root: Path,
    *,
    directory: str,
    role: str,
    scenario: dict[str, Any],
    runs: dict[str, Any],
) -> dict[str, Any]:
    scenario_path = repo_root / directory / "scenario.json"
    runs_path = repo_root / directory / "runs.json"
    _write_json(scenario_path, scenario)
    _write_json(runs_path, runs)
    return {
        "id": f"{directory}_job",
        "scenario": scenario_path.relative_to(repo_root).as_posix(),
        "simulator_runs": runs_path.relative_to(repo_root).as_posix(),
        "dataset_role": role,
        "split_id": directory,
        "condition_id": "baseline",
    }


def _profile(profile_id: str, job: dict[str, Any]) -> dict[str, Any]:
    return {"id": profile_id, "generation_jobs": [job]}


def test_identical_content_at_different_paths_is_rejected(tmp_path: Path) -> None:
    scenario = _scenario(scenario_id="shared", seed=101)
    runs = _runs(run_id="shared", seed=201)
    development = _job(
        tmp_path,
        directory="development",
        role="development",
        scenario=scenario,
        runs=runs,
    )
    validation = _job(
        tmp_path,
        directory="validation",
        role="validation",
        scenario=scenario,
        runs=runs,
    )

    with pytest.raises(BenchmarkConfigError, match="Dataset partition leakage") as error:
        validate_dataset_partition_exclusivity(
            tmp_path,
            [_profile("calibration", development), _profile("selection", validation)],
        )

    assert "generation_task sha256:" in str(error.value)


def test_equal_filenames_with_different_content_do_not_conflict(tmp_path: Path) -> None:
    development = _job(
        tmp_path,
        directory="development",
        role="development",
        scenario=_scenario(scenario_id="development", seed=101),
        runs=_runs(run_id="development", seed=201),
    )
    validation = _job(
        tmp_path,
        directory="validation",
        role="validation",
        scenario=_scenario(scenario_id="validation", seed=102),
        runs=_runs(run_id="validation", seed=202),
    )

    contract = validate_dataset_partition_exclusivity(
        tmp_path,
        [_profile("calibration", development), _profile("selection", validation)],
    )

    assert contract["identity_contract_version"] == "1.0"
    assert contract["protected_roles"] == [
        "development",
        "dimensional_transfer",
        "efficiency_safety",
        "efficiency_validation",
        "test",
        "validation",
    ]
    assert contract["identity_types"] == ["generation_task", "related_input"]
    assert len(contract["records"]) == 2


def test_referenced_input_identity_uses_bytes_instead_of_relative_path(tmp_path: Path) -> None:
    for directory, content in (("development", "TF1\tG1\n"), ("validation", "TF2\tG2\n")):
        input_path = tmp_path / directory / "network.tsv"
        input_path.parent.mkdir(parents=True)
        input_path.write_text(content, encoding="utf-8")
    development = _job(
        tmp_path,
        directory="development",
        role="development",
        scenario=_scenario(scenario_id="development", seed=101, input_path="network.tsv"),
        runs=_runs(run_id="development", seed=201),
    )
    validation = _job(
        tmp_path,
        directory="validation",
        role="validation",
        scenario=_scenario(scenario_id="validation", seed=102, input_path="network.tsv"),
        runs=_runs(run_id="validation", seed=202),
    )

    validate_dataset_partition_exclusivity(
        tmp_path,
        [_profile("calibration", development), _profile("selection", validation)],
    )


def test_same_external_input_content_cannot_cross_dataset_roles(tmp_path: Path) -> None:
    for directory in ("development", "validation"):
        input_path = tmp_path / directory / "network.tsv"
        input_path.parent.mkdir(parents=True)
        input_path.write_text("TF1\tG1\n", encoding="utf-8")
    development = _job(
        tmp_path,
        directory="development",
        role="development",
        scenario=_scenario(scenario_id="development", seed=101, input_path="network.tsv"),
        runs=_runs(run_id="development", seed=201),
    )
    validation = _job(
        tmp_path,
        directory="validation",
        role="validation",
        scenario=_scenario(scenario_id="validation", seed=102, input_path="network.tsv"),
        runs=_runs(run_id="validation", seed=202),
    )

    with pytest.raises(BenchmarkConfigError, match="related_input sha256:") as error:
        validate_dataset_partition_exclusivity(
            tmp_path,
            [_profile("calibration", development), _profile("selection", validation)],
        )

    assert "generation_task sha256:" not in str(error.value)


def test_generation_identity_catches_overlap_hidden_by_request_differences(
    tmp_path: Path,
) -> None:
    development = _job(
        tmp_path,
        directory="development",
        role="development",
        scenario=_scenario(scenario_id="development", seed=101),
        runs=_runs(run_id="development", seed=500, replicates=1),
    )
    validation = _job(
        tmp_path,
        directory="validation",
        role="validation",
        scenario=_scenario(scenario_id="validation", seed=102),
        runs=_runs(run_id="renamed", seed=500, replicates=2),
    )

    with pytest.raises(BenchmarkConfigError, match="generation_task sha256:"):
        validate_dataset_partition_exclusivity(
            tmp_path,
            [_profile("calibration", development), _profile("selection", validation)],
        )


def test_same_simulator_request_with_disjoint_scenario_seeds_is_allowed(
    tmp_path: Path,
) -> None:
    runs = _runs(run_id="same", seed=None)
    development = _job(
        tmp_path,
        directory="development",
        role="development",
        scenario=_scenario(scenario_id="development", seed=101),
        runs=runs,
    )
    validation = _job(
        tmp_path,
        directory="validation",
        role="validation",
        scenario=_scenario(scenario_id="validation", seed=202),
        runs=runs,
    )

    validate_dataset_partition_exclusivity(
        tmp_path,
        [_profile("calibration", development), _profile("selection", validation)],
    )


def test_same_scenario_with_disjoint_explicit_run_seeds_is_allowed(tmp_path: Path) -> None:
    scenario = _scenario(scenario_id="shared", seed=101)
    development = _job(
        tmp_path,
        directory="development",
        role="development",
        scenario=scenario,
        runs=_runs(run_id="shared", seed=301),
    )
    validation = _job(
        tmp_path,
        directory="validation",
        role="validation",
        scenario=scenario,
        runs=_runs(run_id="shared", seed=302),
    )

    validate_dataset_partition_exclusivity(
        tmp_path,
        [_profile("calibration", development), _profile("selection", validation)],
    )


def test_seedless_scenario_derives_each_run_seed_from_its_identity(tmp_path: Path) -> None:
    runs = _runs(run_id="first", seed=None, replicates=2)
    runs["runs"].append(
        {
            "run_id": "second",
            "simulator_id": "dyngen",
            "replicates": 1,
            "native_outputs": [],
            "params": {"num_cells": 30},
        }
    )
    job = _job(
        tmp_path,
        directory="development",
        role="development",
        scenario=_scenario(scenario_id="seedless", seed=None),
        runs=runs,
    )

    contract = build_dataset_partition_contract(
        tmp_path,
        [_profile("calibration", job)],
    )

    semantic_key = "rna_expression|single_cell|cells|trajectory|global,group"
    first_seed = _stable_seed("seedless", semantic_key, "first|dyngen")
    second_seed = _stable_seed("seedless", semantic_key, "second|dyngen")
    observed = {
        (record["run_id"], record["replicate_index"]): record["seed"]
        for record in contract["records"]
    }
    assert observed == {
        ("first", 1): first_seed,
        ("first", 2): ((first_seed - 1 + 1) % 2_147_483_646) + 1,
        ("second", 1): second_seed,
    }


def test_explicit_scenario_seed_uses_offsets_across_runs(tmp_path: Path) -> None:
    runs = _runs(run_id="first", seed=None, replicates=2)
    runs["runs"].append(
        {
            "run_id": "second",
            "simulator_id": "dyngen",
            "replicates": 1,
            "native_outputs": [],
            "params": {"num_cells": 30},
        }
    )
    job = _job(
        tmp_path,
        directory="development",
        role="development",
        scenario=_scenario(scenario_id="seeded", seed=100),
        runs=runs,
    )

    contract = build_dataset_partition_contract(
        tmp_path,
        [_profile("calibration", job)],
    )

    assert {
        (record["run_id"], record["replicate_index"]): record["seed"]
        for record in contract["records"]
    } == {("first", 1): 100, ("first", 2): 101, ("second", 1): 102}


def test_reuse_inside_one_role_is_allowed_and_smoke_is_outside_the_partition(
    tmp_path: Path,
) -> None:
    scenario = _scenario(scenario_id="shared", seed=101)
    runs = _runs(run_id="shared", seed=201)
    first = _job(
        tmp_path,
        directory="first",
        role="development",
        scenario=scenario,
        runs=runs,
    )
    second = _job(
        tmp_path,
        directory="second",
        role="development",
        scenario=scenario,
        runs=runs,
    )
    smoke = _job(
        tmp_path,
        directory="smoke",
        role="smoke",
        scenario=scenario,
        runs=runs,
    )

    contract = build_dataset_partition_contract(
        tmp_path,
        [
            _profile("calibration", first),
            _profile("reduced", second),
            _profile("smoke", smoke),
        ],
    )
    validate_dataset_partition_exclusivity(
        tmp_path,
        [
            _profile("calibration", first),
            _profile("reduced", second),
            _profile("smoke", smoke),
        ],
    )

    assert {record["dataset_role"] for record in contract["records"]} == {"development"}
