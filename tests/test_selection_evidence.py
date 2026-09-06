from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from spathi_bench.artifacts import canonical_json_sha256, sha256_file
from spathi_bench.calibration import (
    build_calibration_ranking,
    build_frozen_selected_config,
    candidate_from_dict,
    expand_stage_candidates,
    load_search_space,
    locked_spathi_identity,
    validate_ranking_chain,
)
from spathi_bench.calibration_materialization import freeze_calibration_selection
from spathi_bench.config import BenchmarkConfigError, validate_calibration_contracts

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ranking_rows(
    candidates: tuple[Any, ...],
    *,
    role: str,
    seeds: tuple[int, ...],
    spathi_identity: dict[str, str],
    preferred: Any | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "configuration_id": candidate.configuration_id,
            "configuration_sha256": candidate.configuration_sha256,
            "dataset_role": role,
            "dataset_id": dataset_id,
            "algorithm_seed": seed,
            "run_id": (
                f"validation__{candidate.configuration_id}__seed_{seed}"
                if role == "validation"
                else f"{candidate.configuration_id}__seed_{seed}"
            ),
            "execution_sha256": canonical_json_sha256(
                {"configuration_id": candidate.configuration_id, "random_seed": seed}
            ),
            "aupr_directed": 0.7 if candidate == preferred else 0.5,
            "wall_time_seconds": 10.0,
            "peak_memory_mb": 100.0,
            "n_group_contexts": 2,
            "group_contexts": ["group:a", "group:b"],
            "spathi_commit": spathi_identity["spathi_commit"],
            "spathi_build_fingerprint": spathi_identity["spathi_build_fingerprint"],
            "spathi_image_id": "sha256:" + "1" * 64,
        }
        for candidate in candidates
        for dataset_id in ("dataset_a", "dataset_b")
        for seed in seeds
    ]


def _calibration_panel(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    dataset_ids = sorted({row["dataset_id"] for row in rows})
    run_ids = sorted({row["run_id"] for row in rows})
    return {
        "schema_version": "1.0",
        "dataset_role": role,
        "dataset_ids": dataset_ids,
        "run_ids": run_ids,
        "n_expected_executions": len(dataset_ids) * len(run_ids),
        "handoff_validation_sha256": "a" * 64,
        "datasets": [
            {
                "dataset_id": dataset_id,
                "expected_group_contexts": ["group:a", "group:b"],
                "n_group_contexts": 2,
                "dataset_manifest_sha256": "b" * 64,
                "group_assignments_sha256": "c" * 64,
                "ground_truth_manifest_sha256": "d" * 64,
                "truth_network_sha256": "e" * 64,
            }
            for dataset_id in dataset_ids
        ],
    }


def _validation_ranking(repo: Path) -> dict[str, Any]:
    search = load_search_space(REPO_ROOT)
    spathi_identity = locked_spathi_identity(REPO_ROOT)
    search_hash = sha256_file(REPO_ROOT / "experiments/v1/spathi/search-space.json")
    evidence = repo / "experiments/v1/spathi/evidence"
    evidence.mkdir(parents=True, exist_ok=True)

    parents: tuple[Any, ...] = ()
    parent_link: dict[str, Any] | None = None
    for index, stage_id in enumerate(
        ("geometry", "estimator", "interaction", "estimator_scale_transfer", "convergence"),
        start=1,
    ):
        candidates = expand_stage_candidates(search, stage_id, parents=parents)
        seeds = tuple(search["calibration_budgets"]["full"]["algorithmic_seeds"][stage_id])
        rows = _ranking_rows(
            candidates,
            role="development",
            seeds=seeds,
            spathi_identity=spathi_identity,
        )
        ranking = build_calibration_ranking(
            search,
            stage_id,
            candidates,
            rows,
            calibration_panel=_calibration_panel(rows, "development"),
            calibration_budget="full",
            profile_id=f"calibration-full-{stage_id}",
            parent_ranking=parent_link,
            benchmark_commit="1" * 40,
            search_space_sha256=search_hash,
            calibration_manifest_sha256=f"{index:x}" * 64,
            benchmark_summary_sha256=f"{index + 4:x}" * 64,
            **spathi_identity,
        )
        ranking_path = evidence / f"{stage_id}-ranking.json"
        ranking_path.write_text(json.dumps(ranking), encoding="utf-8")
        parent_link = {
            "path": ranking_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(ranking_path),
            "stage_id": stage_id,
            "calibration_budget": "full",
        }
        parents = tuple(candidate_from_dict(item) for item in ranking["selected_candidates"])

    finalists = parents
    validation_seeds = tuple(
        search["calibration_budgets"]["full"]["algorithmic_seeds"]["validation"]
    )
    assert parent_link is not None
    rows = _ranking_rows(
        finalists,
        role="validation",
        seeds=validation_seeds,
        spathi_identity=spathi_identity,
        preferred=finalists[1],
    )
    return build_calibration_ranking(
        search,
        "validation",
        finalists,
        rows,
        calibration_panel=_calibration_panel(rows, "validation"),
        calibration_budget="full",
        profile_id="selection-full-validation",
        parent_ranking=parent_link,
        benchmark_commit="1" * 40,
        search_space_sha256=search_hash,
        calibration_manifest_sha256="0" * 64,
        benchmark_summary_sha256="9" * 64,
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
        "experiments/v1/seeds/seed-plan.json",
        "experiments/v1/spathi/search-space.json",
        "experiments/v1/spathi/selected-config.json",
    ):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    ranking_path = repo / "experiments/v1/spathi/evidence/validation-ranking.json"
    ranking_path.parent.mkdir(parents=True)
    ranking_path.write_text(
        json.dumps(_validation_ranking(repo), indent=2, sort_keys=True) + "\n",
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


def test_validation_ranking_is_complete_ordered_and_does_not_mutate_selection(
    tmp_path: Path,
) -> None:
    selected_path = REPO_ROOT / "experiments/v1/spathi/selected-config.json"
    before = selected_path.read_bytes()

    ranking = _validation_ranking(tmp_path / "repo")

    schema = json.loads(
        (REPO_ROOT / "schemas/calibration-ranking.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(ranking)
    assert len(ranking["ranking"]) == 2
    assert ranking["selected_candidates"] == [ranking["ranking"][0]["candidate"]]
    assert (
        ranking["winner"]["candidate_id"] == ranking["ranking"][0]["candidate"]["configuration_id"]
    )
    assert selected_path.read_bytes() == before


def test_ranking_chain_rejects_finalists_outside_parent_shortlist(tmp_path: Path) -> None:
    repo, ranking_path, _selected_path = _repository_with_ranking(tmp_path)
    valid = json.loads(ranking_path.read_text(encoding="utf-8"))
    search = load_search_space(REPO_ROOT)
    identity = locked_spathi_identity(REPO_ROOT)
    convergence_path = repo / valid["parent_ranking"]["path"]
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    incompatible = tuple(
        candidate_from_dict(item["candidate"]) for item in convergence["ranking"][-3:]
    )
    rows = _ranking_rows(
        incompatible,
        role="validation",
        seeds=tuple(search["calibration_budgets"]["full"]["algorithmic_seeds"]["validation"]),
        spathi_identity=identity,
    )
    ranking = build_calibration_ranking(
        search,
        "validation",
        incompatible,
        rows,
        calibration_panel=_calibration_panel(rows, "validation"),
        calibration_budget="full",
        profile_id="selection-full-validation",
        parent_ranking=valid["parent_ranking"],
        benchmark_commit="1" * 40,
        search_space_sha256=sha256_file(REPO_ROOT / "experiments/v1/spathi/search-space.json"),
        calibration_manifest_sha256="6" * 64,
        benchmark_summary_sha256="7" * 64,
        **identity,
    )

    with pytest.raises(BenchmarkConfigError, match="convergence shortlist"):
        validate_ranking_chain(
            repo,
            search,
            ranking,
            expected_search_space_sha256=sha256_file(
                REPO_ROOT / "experiments/v1/spathi/search-space.json"
            ),
        )


def test_ranking_chain_rejects_mixed_benchmark_commits(tmp_path: Path) -> None:
    repo, ranking_path, _selected_path = _repository_with_ranking(tmp_path)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    convergence_path = repo / ranking["parent_ranking"]["path"]
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    convergence["benchmark_commit"] = "2" * 40
    convergence_path.write_text(json.dumps(convergence), encoding="utf-8")
    ranking["parent_ranking"]["sha256"] = sha256_file(convergence_path)

    with pytest.raises(BenchmarkConfigError, match="mixes benchmark commits"):
        validate_ranking_chain(
            repo,
            load_search_space(REPO_ROOT),
            ranking,
            expected_search_space_sha256=sha256_file(
                REPO_ROOT / "experiments/v1/spathi/search-space.json"
            ),
        )


def test_ranking_chain_rejects_mixed_calibration_budgets(tmp_path: Path) -> None:
    repo, ranking_path, _selected_path = _repository_with_ranking(tmp_path)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking["parent_ranking"]["calibration_budget"] = "reduced"

    with pytest.raises(BenchmarkConfigError, match="different calibration budget"):
        validate_ranking_chain(
            repo,
            load_search_space(REPO_ROOT),
            ranking,
            expected_search_space_sha256=sha256_file(
                REPO_ROOT / "experiments/v1/spathi/search-space.json"
            ),
            expected_calibration_budget="full",
        )


def test_freeze_materialization_generates_a_reconciled_selected_config(
    tmp_path: Path,
) -> None:
    repo, ranking_path, selected_path = _repository_with_ranking(tmp_path)

    assert (
        freeze_calibration_selection(
            repo_root=repo,
            ranking_path=ranking_path,
            output_path=selected_path,
        )
        == selected_path
    )
    _search, selected = validate_calibration_contracts(repo)
    assert selected["status"] == "selected"
    assert selected["calibration_budget"] == "full"
    assert selected["selection_evidence"]["calibration_budget"] == "full"
    assert selected["selection_evidence"]["validation_ranking_sha256"] == sha256_file(ranking_path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda selected: selected["selection_evidence"].__setitem__("hierarchical_score", 0.1),
        lambda selected: selected["selection_evidence"].__setitem__(
            "winner_candidate_id", "manual_winner"
        ),
        lambda selected: selected["selected_policy"].__setitem__(
            "source_candidate_id", "manual_winner"
        ),
        lambda selected: selected["selected_policy"].__setitem__("policy_sha256", "0" * 64),
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
