from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from spathi_bench.artifacts import sha256_file
from spathi_bench.calibration import (
    build_calibration_ranking,
    build_frozen_selected_config,
    expand_stage_candidates,
    load_search_space,
    locked_spathi_identity,
)
from spathi_bench.cli import main
from spathi_bench.config import BenchmarkConfigError, validate_calibration_contracts

REPO_ROOT = Path(__file__).resolve().parents[1]


def _validation_ranking() -> dict[str, Any]:
    search = load_search_space(REPO_ROOT)
    spathi_identity = locked_spathi_identity(REPO_ROOT)
    convergence = expand_stage_candidates(search, "convergence")
    structures = expand_stage_candidates(search, "structure", parents=(convergence[0],))
    estimators = expand_stage_candidates(search, "estimator", parents=(structures[0],))
    finalists = estimators[:3]
    seeds = (1729, 31415, 104729)
    rows = [
        {
            "configuration_id": candidate.configuration_id,
            "configuration_sha256": candidate.configuration_sha256,
            "dataset_role": "validation",
            "dataset_id": dataset_id,
            "algorithm_seed": seed,
            "aupr_directed": 0.7 if candidate == finalists[1] else 0.5,
            "wall_time_seconds": 10.0,
            "peak_memory_mb": 100.0,
            "n_group_contexts": 2,
            "group_contexts": ["group:a", "group:b"],
            "spathi_commit": spathi_identity["spathi_commit"],
            "spathi_build_fingerprint": spathi_identity["spathi_build_fingerprint"],
            "spathi_image_id": "sha256:" + "1" * 64,
        }
        for candidate in finalists
        for dataset_id in ("dataset_a", "dataset_b")
        for seed in seeds
    ]
    return build_calibration_ranking(
        search,
        "validation",
        finalists,
        rows,
        search_space_sha256=sha256_file(REPO_ROOT / "experiments/v1/spathi/search-space.json"),
        calibration_manifest_sha256="a" * 64,
        benchmark_summary_sha256="b" * 64,
        **spathi_identity,
    )


def _repository_with_ranking(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    for relative in (
        "repos.lock.json",
        "images.lock.json",
        "schemas/spathi-search-space.schema.json",
        "schemas/spathi-selected-config.schema.json",
        "schemas/calibration-ranking.schema.json",
        "schemas/repos-lock.schema.json",
        "experiments/v1/spathi/search-space.json",
        "experiments/v1/spathi/selected-config.json",
    ):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    ranking_path = repo / "experiments/v1/spathi/evidence/validation-ranking.json"
    ranking_path.parent.mkdir(parents=True)
    ranking_path.write_text(
        json.dumps(_validation_ranking(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_path = repo / "experiments/v1/spathi/selected-config.json"
    return repo, ranking_path, selected_path


def _freeze(repo: Path, ranking_path: Path, selected_path: Path) -> dict[str, Any]:
    selected = build_frozen_selected_config(
        repo,
        ranking_path,
        frozen_at="2026-08-28T12:00:00Z",
    )
    selected_path.write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selected


def test_validation_ranking_is_complete_ordered_and_does_not_mutate_selection() -> None:
    selected_path = REPO_ROOT / "experiments/v1/spathi/selected-config.json"
    before = selected_path.read_bytes()

    ranking = _validation_ranking()

    schema = json.loads(
        (REPO_ROOT / "schemas/calibration-ranking.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(ranking)
    assert len(ranking["ranking"]) == 3
    assert ranking["selected_candidates"] == [ranking["ranking"][0]["candidate"]]
    assert (
        ranking["winner"]["candidate_id"] == ranking["ranking"][0]["candidate"]["configuration_id"]
    )
    assert selected_path.read_bytes() == before


def test_freeze_cli_generates_a_reconciled_selected_config(tmp_path: Path) -> None:
    repo, ranking_path, selected_path = _repository_with_ranking(tmp_path)

    assert (
        main(
            [
                "calibration-freeze",
                "--repo-root",
                str(repo),
                "--ranking",
                str(ranking_path),
                "--output",
                str(selected_path),
            ]
        )
        == 0
    )
    _search, selected = validate_calibration_contracts(repo)
    assert selected["status"] == "selected"
    assert selected["selection_evidence"]["validation_ranking_sha256"] == sha256_file(ranking_path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda selected: selected["selection_evidence"].__setitem__("hierarchical_score", 0.1),
        lambda selected: selected["selection_evidence"].__setitem__(
            "winner_candidate_id", "manual_winner"
        ),
        lambda selected: selected["selected_candidate"].__setitem__(
            "candidate_id", "manual_winner"
        ),
        lambda selected: selected["selected_candidate"].__setitem__(
            "configuration_sha256", "0" * 64
        ),
    ],
)
def test_test_gate_rejects_manually_unreconciled_selection(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    repo, ranking_path, selected_path = _repository_with_ranking(tmp_path)
    selected = _freeze(repo, ranking_path, selected_path)
    contaminated = copy.deepcopy(selected)
    mutate(contaminated)
    selected_path.write_text(json.dumps(contaminated), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="not reconciled"):
        validate_calibration_contracts(repo)


def test_test_gate_rejects_ranking_changed_after_freeze(tmp_path: Path) -> None:
    repo, ranking_path, selected_path = _repository_with_ranking(tmp_path)
    _freeze(repo, ranking_path, selected_path)
    ranking_path.write_text(ranking_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="ranking hash"):
        validate_calibration_contracts(repo)


def test_test_gate_rejects_missing_ranking_evidence(tmp_path: Path) -> None:
    repo, ranking_path, selected_path = _repository_with_ranking(tmp_path)
    selected = _freeze(repo, ranking_path, selected_path)
    selected["selection_evidence"]["validation_ranking"] = (
        "experiments/v1/spathi/evidence/missing-ranking.json"
    )
    selected_path.write_text(json.dumps(selected), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="does not exist"):
        validate_calibration_contracts(repo)


def test_freeze_recomputes_scores_from_embedded_validation_rows(tmp_path: Path) -> None:
    repo, ranking_path, _selected_path = _repository_with_ranking(tmp_path)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking["ranking"][0]["hierarchical_aupr"] = 0.1
    ranking_path.write_text(json.dumps(ranking), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="scores are not reconciled"):
        build_frozen_selected_config(repo, ranking_path)


def test_freeze_rejects_evidence_from_another_spathi_build(tmp_path: Path) -> None:
    repo, ranking_path, _selected_path = _repository_with_ranking(tmp_path)
    image_lock_path = repo / "images.lock.json"
    image_lock = json.loads(image_lock_path.read_text(encoding="utf-8"))
    image_lock["images"]["spathi"]["source_commit"] = "0" * 40
    image_lock_path.write_text(json.dumps(image_lock), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="different SPATHI build"):
        build_frozen_selected_config(repo, ranking_path)
