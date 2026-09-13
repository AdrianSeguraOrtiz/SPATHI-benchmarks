from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.pipeline as pipeline
from spathi_bench.artifacts import ArtifactKey, ArtifactStore
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.runtime import CommandExecutionError


@pytest.fixture
def generation_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    run_root = tmp_path / "work/v1/profile/run"
    run_root.mkdir(parents=True)
    job = {
        "id": "boolode_test",
        "scenario": "scenario.json",
        "simulator_runs": "runs.json",
        "dataset_role": "test",
        "split_id": "test",
        "condition_id": "invariant",
    }
    state = {
        "status": "image_ready",
        "cache": {},
        "execution_plan": {
            "operations": {
                "generate_data": {"max_parallel_tasks": 1, "cpu_threads": 1, "memory_gib": 1}
            }
        },
    }
    spec = {"job": job, "seed": 1729}
    monkeypatch.setattr(pipeline, "_load_existing_state", lambda *_args: state)
    monkeypatch.setattr(pipeline, "docker_runtime_environment", lambda: {})
    monkeypatch.setattr(pipeline, "_validate_docker_runtime", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_validate_andrea_repository", lambda *_args: {})
    monkeypatch.setattr(pipeline, "_validate_state_andrea", lambda *_args: None)
    monkeypatch.setattr(pipeline, "andrea_host_environment", lambda: {})
    monkeypatch.setattr(pipeline, "_validate_host_environment", lambda *_args: None)
    monkeypatch.setattr(pipeline, "frozen_repository_root", lambda *_args: tmp_path)
    monkeypatch.setattr(pipeline, "frozen_profile", lambda *_args: {"generation_jobs": [job]})
    monkeypatch.setattr(pipeline, "frozen_path", lambda _root, path, _label: tmp_path / path)
    monkeypatch.setattr(pipeline, "_generation_spec", lambda **_kwargs: spec)
    monkeypatch.setattr(pipeline, "_generation_image_requirements", lambda **_kwargs: [])
    monkeypatch.setattr(pipeline, "catalog_image_leases", lambda *_args: nullcontext())
    monkeypatch.setattr(pipeline, "_validate_generation_images", lambda **_kwargs: None)
    monkeypatch.setattr(
        pipeline, "datasets_from_benchmark", lambda *_args: [{"dataset_id": "dataset"}]
    )
    store = pipeline._artifact_store(run_root)
    return {
        "state": state,
        "spec": spec,
        "store": store,
        "key": store.key_for("dataset", spec),
        "arguments": {
            "repo_root": tmp_path,
            "profile_id": "profile",
            "run_root": run_root,
            "andrea_repo": tmp_path / "andrea",
        },
    }


def _failure_artifacts(store: ArtifactStore) -> list[Path]:
    return sorted((store.root / "dataset-failure").glob("*/failure.json"))


def test_failed_generation_preserves_diagnostics_and_retry_is_not_a_cache_hit(
    generation_attempt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = generation_attempt["store"]
    attempts: list[Path] = []

    def fake_andrea(arguments: list[str], *, log_path: Path, **_kwargs: Any) -> None:
        output = Path(arguments[arguments.index("--output-dir") + 1])
        attempts.append(output)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"attempt {len(attempts)}\n", encoding="utf-8")
        raw = output / "benchmark/failed_runs/dataset/provenance/raw"
        raw.mkdir(parents=True)
        (raw / "wrapper_error.log").write_text("KeyError: 'E0_708'\n", encoding="utf-8")
        (output / "benchmark/failed_runs/dataset/progress.json").write_text(
            '{"status": "failed"}', encoding="utf-8"
        )
        raise CommandExecutionError(command=arguments, return_code=1, log_path=log_path)

    monkeypatch.setattr(pipeline, "andrea_command", fake_andrea)
    for attempt in range(1, 3):
        with pytest.raises(BenchmarkConfigError, match="durable generation diagnostics") as caught:
            pipeline.generate_data(**generation_attempt["arguments"])

        evidence = _failure_artifacts(store)
        assert len(evidence) == attempt
        assert any(str(path) in str(caught.value) for path in evidence)
        assert store.lookup(generation_attempt["key"]) is None
        assert generation_attempt["state"]["status"] == "image_ready"
        assert list(store.staging_root.iterdir()) == []

    assert len(attempts) == 2
    assert all(not path.exists() for path in attempts)
    recorded_logs = set()
    for manifest_path in _failure_artifacts(store):
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert record["generation_status"] == "failed"
        assert record["source_dataset"]["spec"] == generation_attempt["spec"]
        assert record["error_type"] == "CommandExecutionError"
        evidence_key = ArtifactKey("dataset-failure", manifest_path.parent.name)
        artifact = store.validate(evidence_key)
        diagnostic = artifact.file(
            "output/benchmark/failed_runs/dataset/provenance/raw/wrapper_error.log"
        )
        assert diagnostic.read_text(encoding="utf-8") == "KeyError: 'E0_708'\n"
        recorded_logs.add(artifact.file("command.log").read_text(encoding="utf-8"))
        assert artifact.file("output/benchmark/failed_runs/dataset/progress.json").is_file()
    assert recorded_logs == {"attempt 1\n", "attempt 2\n"}

    def succeeding_andrea(arguments: list[str], **_kwargs: Any) -> None:
        output = Path(arguments[arguments.index("--output-dir") + 1])
        benchmark = output / "benchmark"
        benchmark.mkdir(parents=True)
        (benchmark / "benchmark-manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pipeline, "andrea_command", succeeding_andrea)
    pipeline.generate_data(**generation_attempt["arguments"])
    assert store.lookup(generation_attempt["key"]) is not None
    assert generation_attempt["state"]["status"] == "generated"
    assert generation_attempt["state"]["cache"]["dataset_hits"] == 0
    assert len(_failure_artifacts(store)) == 2


def test_invalid_generated_output_is_retained_for_diagnosis(
    generation_attempt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_manifest(arguments: list[str], **_kwargs: Any) -> None:
        output = Path(arguments[arguments.index("--output-dir") + 1])
        output.mkdir()
        (output / "incomplete.tsv").write_text("gene\tcell\nTF1\t1\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "andrea_command", missing_manifest)
    with pytest.raises(BenchmarkConfigError, match="Expected exactly one benchmark-manifest.json"):
        pipeline.generate_data(**generation_attempt["arguments"])

    store = generation_attempt["store"]
    manifest, = _failure_artifacts(store)
    assert (manifest.parent / "output/incomplete.tsv").is_file()
    assert store.lookup(generation_attempt["key"]) is None
    assert list(store.staging_root.iterdir()) == []


def test_generation_startup_error_preserves_evidence_without_command_log(
    generation_attempt: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_python(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError("Missing ANDREA Python executable")

    monkeypatch.setattr(pipeline, "andrea_command", unavailable_python)
    with pytest.raises(BenchmarkConfigError, match="Missing ANDREA Python executable"):
        pipeline.generate_data(**generation_attempt["arguments"])

    store = generation_attempt["store"]
    manifest, = _failure_artifacts(store)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["error_type"] == "FileNotFoundError"
    assert record["message"] == "Missing ANDREA Python executable"
    assert store.lookup(generation_attempt["key"]) is None
    assert list(store.staging_root.iterdir()) == []
