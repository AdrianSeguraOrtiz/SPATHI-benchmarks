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
from spathi_bench.benchmark_materialization import load_frozen_selected_policy
from spathi_bench.calibration import (
    build_calibration_ranking,
    build_frozen_selected_config,
    candidate_from_dict,
    expand_stage_candidates,
    load_parent_ranking,
    load_search_space,
    locked_spathi_identity,
    stage_profile_id,
    validate_ranking_chain,
)
from spathi_bench.calibration_design import calibration_phase
from spathi_bench.calibration_materialization import (
    freeze_calibration_selection,
    materialize_calibration_stage,
)
from spathi_bench.config import BenchmarkConfigError, validate_calibration_contracts
from spathi_bench.study_context import study_execution

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ranking_rows(
    candidates: tuple[Any, ...],
    *,
    role: str,
    seeds: tuple[int, ...],
    fidelity: str,
    dataset_count: int,
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
                f"validation__{candidate.configuration_id}__final__seed_{seed}"
                if role == "validation"
                else f"{candidate.configuration_id}__{fidelity}__seed_{seed}"
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
        for dataset_id in (f"dataset_{index:03d}" for index in range(1, dataset_count + 1))
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


def _validation_ranking(repo: Path, *, resumed: bool = False) -> dict[str, Any]:
    search = load_search_space(REPO_ROOT)
    spathi_identity = locked_spathi_identity(REPO_ROOT)
    search_hash = sha256_file(REPO_ROOT / "experiments/v1/spathi/search-space.json")
    evidence = repo / "experiments/v1/spathi/evidence"
    evidence.mkdir(parents=True, exist_ok=True)

    selected: tuple[Any, ...] = ()
    parent_link: dict[str, Any] | None = None
    phases = (
        ("geometry", "scout"),
        ("geometry", "confirmation"),
        ("estimator", "scout"),
        ("estimator", "confirmation"),
        ("interaction", "scout"),
        ("interaction", "confirmation"),
        ("estimator_scale_transfer", "scout"),
        ("estimator_scale_transfer", "confirmation"),
        ("convergence", "final"),
    )
    for index, (stage_id, fidelity) in enumerate(
        phases,
        start=1,
    ):
        candidates = (
            expand_stage_candidates(search, stage_id, parents=selected)
            if fidelity == "scout" or stage_id == "convergence"
            else selected
        )
        phase = calibration_phase(search, "full", stage_id, fidelity)
        all_seeds = search["calibration_budgets"]["full"]["algorithmic_seeds"][stage_id]
        seeds = tuple(all_seeds[position - 1] for position in phase.algorithm_seed_indices)
        rows = _ranking_rows(
            candidates,
            role="development",
            seeds=seeds,
            fidelity=fidelity,
            dataset_count=phase.dataset_count,
            spathi_identity=spathi_identity,
        )
        ranking = build_calibration_ranking(
            search,
            stage_id,
            fidelity,
            candidates,
            rows,
            calibration_panel=_calibration_panel(rows, "development"),
            calibration_budget="full",
            profile_id=stage_profile_id(stage_id, "full", fidelity),
            parent_ranking=parent_link,
            benchmark_commit=("2" if resumed and index > 1 else "1") * 40,
            search_space_sha256=search_hash,
            calibration_manifest_sha256=f"{index:x}" * 64,
            benchmark_summary_sha256=f"{index + 4:x}" * 64,
            **spathi_identity,
        )
        ranking_path = evidence / f"{stage_id}-{fidelity}-ranking.json"
        ranking_path.write_text(json.dumps(ranking), encoding="utf-8")
        parent_link = {
            "path": ranking_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(ranking_path),
            "stage_id": stage_id,
            "fidelity": fidelity,
            "calibration_budget": "full",
        }
        selected = tuple(candidate_from_dict(item) for item in ranking["selected_candidates"])

    finalists = selected
    validation_seeds = tuple(
        search["calibration_budgets"]["full"]["algorithmic_seeds"]["validation"]
    )
    assert parent_link is not None
    rows = _ranking_rows(
        finalists,
        role="validation",
        seeds=validation_seeds,
        fidelity="final",
        dataset_count=calibration_phase(search, "full", "validation", "final").dataset_count,
        spathi_identity=spathi_identity,
        preferred=finalists[1],
    )
    return build_calibration_ranking(
        search,
        "validation",
        "final",
        finalists,
        rows,
        calibration_panel=_calibration_panel(rows, "validation"),
        calibration_budget="full",
        profile_id=stage_profile_id("validation", "full", "final"),
        parent_ranking=parent_link,
        benchmark_commit=("2" if resumed else "1") * 40,
        search_space_sha256=search_hash,
        calibration_manifest_sha256="0" * 64,
        benchmark_summary_sha256="9" * 64,
        **spathi_identity,
    )


def _repository_with_ranking(
    tmp_path: Path, *, resumed: bool = False,
) -> tuple[Path, Path, Path]:
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
        json.dumps(_validation_ranking(repo, resumed=resumed), indent=2, sort_keys=True) + "\n",
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
    selected_ids = {item["configuration_id"] for item in convergence["selected_candidates"]}
    incompatible = tuple(
        candidate_from_dict(item["candidate"])
        for item in convergence["ranking"]
        if item["candidate"]["configuration_id"] not in selected_ids
    )[:2]
    assert len(incompatible) == 2
    rows = _ranking_rows(
        incompatible,
        role="validation",
        seeds=tuple(search["calibration_budgets"]["full"]["algorithmic_seeds"]["validation"]),
        fidelity="final",
        dataset_count=calibration_phase(search, "full", "validation", "final").dataset_count,
        spathi_identity=identity,
    )
    ranking = build_calibration_ranking(
        search,
        "validation",
        "final",
        incompatible,
        rows,
        calibration_panel=_calibration_panel(rows, "validation"),
        calibration_budget="full",
        profile_id=stage_profile_id("validation", "full", "final"),
        parent_ranking=valid["parent_ranking"],
        benchmark_commit="1" * 40,
        search_space_sha256=sha256_file(REPO_ROOT / "experiments/v1/spathi/search-space.json"),
        calibration_manifest_sha256="6" * 64,
        benchmark_summary_sha256="7" * 64,
        **identity,
    )

    with pytest.raises(BenchmarkConfigError, match="parent shortlist"):
        validate_ranking_chain(
            repo,
            search,
            ranking,
            expected_search_space_sha256=sha256_file(
                REPO_ROOT / "experiments/v1/spathi/search-space.json"
            ),
        )


def test_ranking_chain_accepts_recorded_benchmark_revisions_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    repo, ranking_path, _selected_path = _repository_with_ranking(tmp_path)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    convergence_path = repo / ranking["parent_ranking"]["path"]
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    convergence["benchmark_commit"] = "2" * 40
    convergence_path.write_text(json.dumps(convergence), encoding="utf-8")
    ranking["parent_ranking"]["sha256"] = sha256_file(convergence_path)

    before = convergence_path.read_bytes()
    scores = validate_ranking_chain(
        repo,
        load_search_space(REPO_ROOT),
        ranking,
        expected_search_space_sha256=sha256_file(
            REPO_ROOT / "experiments/v1/spathi/search-space.json"
        ),
    )
    assert len(scores) == 2
    assert "spans benchmark revisions" in caplog.text
    assert "1" * 40 in caplog.text and "2" * 40 in caplog.text
    assert convergence_path.read_bytes() == before


def test_resumed_full_chain_can_advance_freeze_and_load_portable_selection(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    repo, ranking_path, _ = _repository_with_ranking(tmp_path, resumed=True)
    evidence = ranking_path.parent
    original_bytes = {path.name: path.read_bytes() for path in evidence.glob("*.json")}
    # This is the server's failing transition: old geometry scout, new confirmation,
    # then load the confirmation shortlist to materialize the estimator scout.
    parents, link = load_parent_ranking(
        repo, load_search_space(repo), "estimator", "full", "scout",
        evidence / "geometry-confirmation-ranking.json",
    )
    assert parents and link is not None
    assert expand_stage_candidates(load_search_space(repo), "estimator", parents=parents)
    for directory in ("schemas", "experiments/v1/profiles"):
        shutil.copytree(REPO_ROOT / directory, repo / directory, dirs_exist_ok=True)
    plan = dict(
        repo_root=repo, stage_id="estimator", calibration_budget="full", fidelity="scout",
        parent_ranking_path=evidence / "geometry-confirmation-ranking.json",
        output_dir=repo / "work/estimator/requests",
        profile_output=repo / "work/estimator/calibration-full-estimator-scout.json",
    )
    with study_execution("full-1", revisions={"benchmark": {"commit": "3" * 40}}, resume=True):
        materialized = materialize_calibration_stage(**plan)
    request_bytes = {path.name: path.read_bytes() for path in materialized[0].glob("*.json")}
    assert materialize_calibration_stage(**plan) == materialized
    assert {path.name: path.read_bytes() for path in materialized[0].glob("*.json")} == request_bytes
    selected_path = repo / "work/candidate/selected-config.json"
    selected_path.parent.mkdir(parents=True)
    selection = _freeze(repo, ranking_path, selected_path)
    policy, seeds = load_frozen_selected_policy(repo, selected_path, budget="full")
    assert policy == selection["selected_policy"]
    assert len(seeds) == 3
    # Future phases read copied evidence without any live resume context or journal.
    frozen = tmp_path / "frozen-repository"
    shutil.copytree(repo, frozen)
    assert load_frozen_selected_policy(
        frozen, frozen / selected_path.relative_to(repo), budget="full",
    ) == (policy, seeds)
    assert {path.name: path.read_bytes() for path in evidence.glob("*.json")} == original_bytes
    assert "geometry.scout → geometry.confirmation" in caplog.text
    assert "Scientific evidence verified" in caplog.text


@pytest.mark.parametrize(
    "mutation",
    ["parent_hash", "search_hash", "seed", "score", "missing_row", "shortlist", "build"],
)
def test_resumed_chain_still_rejects_changed_scientific_evidence(
    tmp_path: Path, mutation: str,
) -> None:
    repo, ranking_path, _ = _repository_with_ranking(tmp_path, resumed=True)
    ranking = json.loads(ranking_path.read_text())
    if mutation == "parent_hash":
        ranking["parent_ranking"]["sha256"] = "0" * 64
    elif mutation == "search_hash":
        ranking["search_space_sha256"] = "0" * 64
    elif mutation == "seed":
        ranking["calibration_rows"][0]["algorithm_seed"] += 1
    elif mutation == "score":
        ranking["ranking"][0]["hierarchical_aupr"] = 0.1
    elif mutation == "missing_row":
        ranking["calibration_rows"].pop()
    elif mutation == "shortlist":
        ranking["selected_candidates"] = [ranking["ranking"][1]["candidate"]]
    else:
        image_path = repo / "images.lock.json"
        image = json.loads(image_path.read_text())
        image["images"]["spathi"]["build_inputs"]["containers/spathi/run_tool.py"] = "0" * 64
        image_path.write_text(json.dumps(image))
    ranking_path.write_text(json.dumps(ranking))
    with pytest.raises(BenchmarkConfigError):
        build_frozen_selected_config(repo, ranking_path)


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
