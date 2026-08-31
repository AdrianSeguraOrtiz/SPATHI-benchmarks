from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.pipeline as pipeline
from spathi_bench.artifacts import ArtifactKey
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.pipeline import (
    _analysis_bundle_spec,
    _artifact_store,
    _evaluation_spec,
    _generation_spec,
    _inference_spec,
    _integration_spec,
    _spathi_build_fingerprint,
    build_spathi_image,
)


def _state() -> dict[str, Any]:
    return {
        "repositories": {"observed": {"andrea": {"commit": "a" * 40}}},
        "host_environment": {"fingerprint": "h" * 64},
        "catalog_images": {"simulator:dyngen": {"locked_repo_digest": "dyngen@sha256:" + "d" * 64}},
    }


def test_generation_key_does_not_depend_on_spathi(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario.json"
    runs = tmp_path / "runs.json"
    scenario.write_text('{"schema_version":"1.0","id":"scenario"}\n', encoding="utf-8")
    runs.write_text('{"runs":[{"simulator_id":"dyngen"}]}\n', encoding="utf-8")
    state = _state()
    job = {
        "id": "development",
        "dataset_role": "development",
        "split_id": "split_a",
        "condition_id": "condition_a",
    }
    resources = {"max_parallel_tasks": 1, "max_cores": 2, "max_ram_gb": 8}

    before = _generation_spec(
        state=state,
        job=job,
        scenario=scenario,
        simulator_runs=runs,
        resources=resources,
    )
    state["spathi_image"] = {"source_commit": "1" * 40}
    after = _generation_spec(
        state=state,
        job=job,
        scenario=scenario,
        simulator_runs=runs,
        resources=resources,
    )

    assert before == after


def test_spathi_change_invalidates_only_spathi_inference_and_evaluation() -> None:
    state = _state()
    dataset = {
        "dataset_id": "dataset_a",
        "dataset_artifact": {"key": "sha256:" + "d" * 64},
    }
    resources = {
        "max_cores": 2,
        "max_ram_gb": 8,
        "planner": "heuristic",
        "planner_time_limit_seconds": 30,
    }
    baseline_batch = {"kind": "baseline", "regime": "group_only"}
    baseline_run = {
        "run_id": "genie3",
        "tool_id": "genie3",
        "params": {"seed": 1},
        "execution": {"mode": "group_emulated"},
    }
    baseline_metadata = {
        "run_id": "genie3",
        "configuration_id": "genie3",
        "selection_role": "baseline",
        "algorithm_seed": 1,
        "algorithm_seed_parameter": "seed",
    }
    baseline_image = {"kind": "catalog", "repo_digest": "genie3@sha256:" + "e" * 64}
    baseline_spec = _inference_spec(
        state=state,
        dataset=dataset,
        batch=baseline_batch,
        run=baseline_run,
        metadata=baseline_metadata,
        custom_definition=None,
        tool_image=baseline_image,
        resources=resources,
    )

    spathi_batch = {"kind": "spathi", "regime": "group_only"}
    spathi_run = {
        "run_id": "spathi_candidate",
        "tool_id": "custom_spathi_candidate",
        "params": {"n_estimators": 100, "random_seed": 1},
        "execution": {"mode": "group_native"},
    }
    spathi_metadata = {
        "run_id": "spathi_candidate",
        "configuration_id": "spathi_candidate",
        "selection_role": "candidate",
        "algorithm_seed": 1,
        "algorithm_seed_parameter": "random_seed",
    }
    custom_definition = {
        "run_id": "spathi_candidate",
        "docker_image": "spathi:locked",
    }
    old_spathi_image = {"kind": "spathi", "source_commit": "1" * 40}
    new_spathi_image = {"kind": "spathi", "source_commit": "2" * 40}
    old_spathi_spec = _inference_spec(
        state=state,
        dataset=dataset,
        batch=spathi_batch,
        run=spathi_run,
        metadata=spathi_metadata,
        custom_definition=custom_definition,
        tool_image=old_spathi_image,
        resources=resources,
    )
    new_spathi_spec = _inference_spec(
        state=state,
        dataset=dataset,
        batch=spathi_batch,
        run=spathi_run,
        metadata=spathi_metadata,
        custom_definition=custom_definition,
        tool_image=new_spathi_image,
        resources=resources,
    )

    baseline_key = ArtifactKey.from_spec("inference", baseline_spec)
    assert baseline_key == ArtifactKey.from_spec("inference", deepcopy(baseline_spec))
    old_spathi_key = ArtifactKey.from_spec("inference", old_spathi_spec)
    new_spathi_key = ArtifactKey.from_spec("inference", new_spathi_spec)
    assert old_spathi_key != new_spathi_key

    baseline_inference = {
        "artifact": {"key": baseline_key.value},
        "dataset_fingerprint": {"algorithm": "sha256", "value": "f" * 64},
    }
    old_spathi_inference = {
        "artifact": {"key": old_spathi_key.value},
        "dataset_fingerprint": baseline_inference["dataset_fingerprint"],
    }
    new_spathi_inference = {
        "artifact": {"key": new_spathi_key.value},
        "dataset_fingerprint": baseline_inference["dataset_fingerprint"],
    }
    baseline_evaluation = ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=baseline_inference),
    )
    assert baseline_evaluation == ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=deepcopy(baseline_inference)),
    )
    assert ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=old_spathi_inference),
    ) != ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=new_spathi_inference),
    )

    baseline_for_joint = {
        **baseline_inference,
        "run_id": "genie3",
    }
    old_spathi_for_joint = {
        **old_spathi_inference,
        "run_id": "spathi_candidate",
    }
    new_spathi_for_joint = {
        **new_spathi_inference,
        "run_id": "spathi_candidate",
    }
    old_joint_spec = _integration_spec(
        state=state,
        dataset=dataset,
        regime="group_only",
        inference_runs=[baseline_for_joint, old_spathi_for_joint],
    )
    new_joint_spec = _integration_spec(
        state=state,
        dataset=dataset,
        regime="group_only",
        inference_runs=[baseline_for_joint, new_spathi_for_joint],
    )
    old_joint_key = ArtifactKey.from_spec("joint-inference", old_joint_spec)
    new_joint_key = ArtifactKey.from_spec("joint-inference", new_joint_spec)
    assert old_joint_key != new_joint_key

    old_joint = {
        "artifact": {"key": old_joint_key.value},
        "dataset_fingerprint": baseline_inference["dataset_fingerprint"],
    }
    new_joint = {
        "artifact": {"key": new_joint_key.value},
        "dataset_fingerprint": baseline_inference["dataset_fingerprint"],
    }
    assert ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=old_joint),
    ) != ArtifactKey.from_spec(
        "evaluation",
        _evaluation_spec(state=state, dataset=dataset, inference=new_joint),
    )

    baseline_bundle = ArtifactKey.from_spec(
        "analysis-bundle",
        _analysis_bundle_spec(
            state=state,
            source_artifact=baseline_inference["artifact"],
            producer_command="infer-network",
            dataset_id="dataset_a",
            scope="individual",
            regime="group_only",
            run_id="genie3",
        ),
    )
    assert baseline_bundle == ArtifactKey.from_spec(
        "analysis-bundle",
        _analysis_bundle_spec(
            state=state,
            source_artifact=deepcopy(baseline_inference["artifact"]),
            producer_command="infer-network",
            dataset_id="dataset_a",
            scope="individual",
            regime="group_only",
            run_id="genie3",
        ),
    )
    assert ArtifactKey.from_spec(
        "analysis-bundle",
        _analysis_bundle_spec(
            state=state,
            source_artifact=old_joint["artifact"],
            producer_command="infer-network",
            dataset_id="dataset_a",
            scope="joint",
            regime="group_only",
            run_id=None,
        ),
    ) != ArtifactKey.from_spec(
        "analysis-bundle",
        _analysis_bundle_spec(
            state=state,
            source_artifact=new_joint["artifact"],
            producer_command="infer-network",
            dataset_id="dataset_a",
            scope="joint",
            regime="group_only",
            run_id=None,
        ),
    )


def test_spathi_build_fingerprint_covers_every_locked_input() -> None:
    image = {
        "source_commit": "1" * 40,
        "base_image": "python@sha256:" + "2" * 64,
        "build_inputs": {"adapter.py": "3" * 64},
        "python_dependencies": {"numpy": "1.0"},
    }
    original = _spathi_build_fingerprint(image)

    for field, replacement in (
        ("source_commit", "4" * 40),
        ("base_image", "python@sha256:" + "5" * 64),
        ("build_inputs", {"adapter.py": "6" * 64}),
        ("python_dependencies", {"numpy": "2.0"}),
    ):
        changed = deepcopy(image)
        changed[field] = replacement
        assert _spathi_build_fingerprint(changed) != original


def test_spathi_image_provenance_records_the_requested_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = {
        "tag": "spathi-benchmark:test",
        "source_commit": "1" * 40,
        "base_image": "python@sha256:" + "2" * 64,
        "build_inputs": {"adapter.py": "3" * 64},
        "python_dependencies": {"numpy": "1.0"},
    }
    fingerprint = _spathi_build_fingerprint(image)
    details = {
        "reference": image["tag"],
        "id": "sha256:" + "4" * 64,
        "repo_digests": [],
        "created": "2026-08-28T00:00:00Z",
        "labels": {
            "org.opencontainers.image.revision": image["source_commit"],
            "dev.spathi.benchmark.build-fingerprint": fingerprint,
        },
    }
    run_root = tmp_path / "work" / "v1" / "smoke" / "run"
    frozen_root = tmp_path / "frozen"
    monkeypatch.setattr(
        pipeline,
        "_load_existing_state",
        lambda *_args, **_kwargs: {"status": "preflighted"},
    )
    monkeypatch.setattr(pipeline, "frozen_repository_root", lambda _run_root: frozen_root)
    monkeypatch.setattr(
        pipeline,
        "load_repo_lock",
        lambda _root: {"repositories": {"spathi": {}}},
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_spathi_image_contract",
        lambda _root: {"images": {"spathi": image}},
    )
    monkeypatch.setattr(pipeline, "validate_locked_repository", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline, "docker_image_details", lambda _reference: deepcopy(details))

    provenance_path = build_spathi_image(
        repo_root=tmp_path,
        profile_id="smoke",
        run_root=run_root,
        spathi_repo=tmp_path / "SPATHI",
    )

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    state = json.loads((run_root / "state.json").read_text(encoding="utf-8"))
    assert provenance["tag"] == image["tag"]
    assert state["spathi_image"]["tag"] == image["tag"]


def test_artifact_store_rejects_a_filesystem_root_target(tmp_path: Path) -> None:
    shallow = Path(tmp_path.anchor) / "profile" / "run"
    with pytest.raises(BenchmarkConfigError, match="refusing an artifact store"):
        _artifact_store(shallow)


def test_artifact_store_is_shared_by_runs_in_one_version(tmp_path: Path) -> None:
    first = _artifact_store(tmp_path / "v1" / "smoke" / "run_a")
    second = _artifact_store(tmp_path / "v1" / "smoke" / "run_b")

    assert first.root == second.root == (tmp_path / "v1" / "artifacts").resolve()
