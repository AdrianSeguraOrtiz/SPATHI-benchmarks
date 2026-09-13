from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

import spathi_bench.study_amendments as amendments
from spathi_bench.artifacts import sha256_file
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.pipeline import _artifact_store, _generation_spec
from spathi_bench.provenance import observe_benchmark_repository
from spathi_bench.study_state import PHASES, observe_revisions, scientific_inputs

REQUEST = "experiments/v1/simulations/boolode/runs.json"
SCENARIO = "experiments/v1/simulations/boolode/scenario.json"
PROFILE = "experiments/v1/profiles/reduced.json"


def _git(root, *arguments):
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


@pytest.fixture
def stopped_study(tmp_path, monkeypatch):
    repos = {}
    for name in ("repo", "ANDREA", "SPATHI"):
        repo = tmp_path / name
        repo.mkdir()
        _git(repo, "init", "-q")
        _git(repo, "config", "user.name", "Test")
        _git(repo, "config", "user.email", "test@example.invalid")
        _git(repo, "remote", "add", "origin", f"https://example.invalid/{name}")
        (repo / ".gitignore").write_text("work/\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "Initialize")
        repos[name] = repo
    repo = repos["repo"]
    job = {
        "id": "boolode", "scenario": SCENARIO, "simulator_runs": REQUEST,
        "dataset_role": "test", "split_id": "test", "condition_id": "clean",
    }
    profile = {"id": "reduced-v1", "generation_jobs": [job], "inference_batches": []}
    _write(repo / PROFILE, profile)
    _write(repo / SCENARIO, {})
    _write(repo / REQUEST, {"runs": [{"run_id": "clean", "simulator_id": "boolode", "base_seed": 42, "params": {"integration_step_size": 0.1}}]})
    _write(repo / "repos.lock.json", {"repositories": {"andrea": {"commit": _git(repos["ANDREA"], "rev-parse", "HEAD")}}})
    _write(repo / amendments._MATRIX, {"andrea_commit": _git(repos["ANDREA"], "rev-parse", "HEAD")})
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Original design")
    root = repo / "work/studies/pilot"
    run_root = root / "v1/reduced-v1/pilot"
    frozen = run_root / "input_snapshot/repository"
    files = []
    for relative in (PROFILE, SCENARIO, REQUEST, "repos.lock.json"):
        source = repo / relative
        _write(frozen / relative, json.loads(source.read_text()))
        files.append({"path": relative, "sha256": sha256_file(source), "size": source.stat().st_size, "kind": "versioned"})
    source = next(entry for entry in files if entry["path"] == PROFILE)
    _write(run_root / "input_snapshot/manifest.json", {
        "schema_version": "1.0", "status": "complete", "profile_id": "reduced-v1",
        "profile_source": {key: source[key] for key in ("path", "sha256", "kind")},
        "files": files, "benchmark_repository": observe_benchmark_repository(repo),
    })
    _write(run_root / "preflight/boolode-simulation-plan.json", {"integration_step_size": 0.1})
    resources = {"max_parallel_tasks": 1, "cpu_threads": 8, "memory_gib": 32.0}
    pipeline_state = {
        "schema_version": "1.0", "profile_id": "reduced-v1", "status": "image_ready",
        "execution_plan": {"operations": {"generate_data": resources}},
        "repositories": {"observed": {"andrea": {"commit": _git(repos["ANDREA"], "rev-parse", "HEAD")}}},
        "host_environment": {"fingerprint": "old-environment"},
        "catalog_images": {"simulator:boolode": {"locked_repo_digest": "immutable-image"}},
    }
    _write(run_root / "state.json", pipeline_state)
    evidence = root / "v1/workflows/calibration/reduced/pilot/workflow.json"
    _write(evidence, {"status": "complete"})
    phases = {name: {"status": "pending"} for name in PHASES}
    phases["calibration"] = {"status": "complete", "evidence": {evidence.relative_to(root).as_posix(): sha256_file(evidence)}}
    phases["comparison"] = {"status": "failed"}
    common = {"repo_root": repo, "work_root": repo / "work", "andrea_repo": repos["ANDREA"], "spathi_repo": repos["SPATHI"]}
    revisions = observe_revisions(repo, repos["ANDREA"], repos["SPATHI"])
    _write(root / "study.json", {
        "schema_version": "1.0", "workflow_type": "study", "run_id": "pilot", "profile": "reduced",
        "status": "failed", "phases": phases, "warnings": [], "scientific_inputs": scientific_inputs(repo),
        "invocations": [{"revisions": revisions}],
    })
    original_state = (run_root / "state.json").read_bytes()
    original_evidence = evidence.read_bytes()
    request = json.loads((repo / REQUEST).read_text())
    request["runs"][0]["params"]["integration_step_size"] = 0.01
    _write(repo / REQUEST, request)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Correct simulator request")
    monkeypatch.setattr(amendments, "validate_all_profiles", lambda _: [profile])
    monkeypatch.setattr(amendments, "validate_design_contracts", lambda _: None)
    monkeypatch.setattr(amendments, "frozen_profile", lambda *_: copy.deepcopy(profile))
    plans = []

    def plan(arguments, **kwargs):
        plans.append(arguments)
        _write(Path(arguments[arguments.index("--out") + 1]), {"integration_step_size": 0.01})

    monkeypatch.setattr(amendments, "andrea_command", plan)
    return {
        "common": common, "root": root, "run_root": run_root, "frozen": frozen,
        "job": job, "resources": resources, "state": pipeline_state, "plans": plans,
        "original_state": original_state, "evidence": evidence, "original_evidence": original_evidence,
    }


def _amend(setup):
    return amendments.amend_study_inputs(**setup["common"], run_id="pilot", reason="Correct BoolODE integration step")


def test_amendment_preserves_results_identities_and_archives_old_inputs(stopped_study):
    setup = stopped_study
    old_lock = (setup["frozen"] / "repos.lock.json").read_bytes()
    result = _amend(setup)
    assert result["status"] == "amended"
    assert json.loads((setup["frozen"] / REQUEST).read_text())["runs"][0]["params"]["integration_step_size"] == 0.01
    assert (setup["run_root"] / "state.json").read_bytes() == setup["original_state"]
    assert setup["evidence"].read_bytes() == setup["original_evidence"]
    assert (setup["frozen"] / "repos.lock.json").read_bytes() == old_lock
    journal = json.loads((setup["root"] / "study.json").read_text())
    assert journal["scientific_inputs"] == scientific_inputs(setup["common"]["repo_root"])
    assert len(journal["input_amendments"]) == 1
    archived = Path(result["amendment"]) / "before" / (setup["frozen"] / REQUEST).relative_to(setup["root"])
    assert json.loads(archived.read_text())["runs"][0]["params"]["integration_step_size"] == 0.1
    assert len(setup["plans"]) == 1
    assert _amend(setup)["status"] == "unchanged"


def test_published_dataset_is_protected_even_before_generation_checkpoint(stopped_study):
    setup = stopped_study
    store = _artifact_store(setup["run_root"])
    spec = _generation_spec(state=setup["state"], job=setup["job"], scenario=setup["frozen"] / SCENARIO,
                            simulator_runs=setup["frozen"] / REQUEST, input_root=setup["frozen"], resources=setup["resources"])
    with store.staging_directory(store.key_for("dataset", spec)) as temporary:
        (temporary / "evidence.txt").write_text("Published dataset")
        store.publish_from_temp(temporary, kind="dataset", spec=spec)
    with pytest.raises(BenchmarkConfigError, match="published dataset"):
        _amend(setup)
    assert not list((setup["root"] / "amendments").glob("*/amendment.json"))


@pytest.mark.parametrize("status", ["generated", "inferred", "completed"])
def test_generation_completion_prevents_input_amendment(stopped_study, status):
    setup = stopped_study
    setup["state"]["status"] = status
    _write(setup["run_root"] / "state.json", setup["state"])
    with pytest.raises(BenchmarkConfigError, match="after generation completed"):
        _amend(setup)


def test_corrupt_completed_evidence_prevents_amendment(stopped_study):
    setup = stopped_study
    setup["evidence"].write_text("tampered")
    with pytest.raises(BenchmarkConfigError, match="missing or changed"):
        _amend(setup)


def test_amendment_recovers_after_partial_snapshot_write(stopped_study, monkeypatch):
    setup = stopped_study
    real_write = amendments.atomic_write_text
    count = 0

    def interrupted_write(path, text, label, **kwargs):
        nonlocal count
        if label == "Amended study input":
            count += 1
            if count == 2:
                raise KeyboardInterrupt("Injected interruption")
        return real_write(path, text, label, **kwargs)

    monkeypatch.setattr(amendments, "atomic_write_text", interrupted_write)
    with pytest.raises(KeyboardInterrupt):
        _amend(setup)
    monkeypatch.setattr(amendments, "atomic_write_text", real_write)
    assert _amend(setup)["status"] == "amended"
    assert len(setup["plans"]) == 1
    assert (setup["run_root"] / "state.json").read_bytes() == setup["original_state"]


def test_normal_resume_waits_for_amendment_completion_marker(stopped_study, monkeypatch):
    import spathi_bench.study_workflow as study

    setup = stopped_study
    real_write = amendments.atomic_write_json

    def interrupt_marker(path, payload, label, **kwargs):
        if label == "Applied input amendment":
            raise KeyboardInterrupt("Before completion marker")
        return real_write(path, payload, label, **kwargs)

    monkeypatch.setattr(amendments, "atomic_write_json", interrupt_marker)
    with pytest.raises(KeyboardInterrupt):
        _amend(setup)
    journal = (setup["root"] / "study.json").read_bytes()
    with pytest.raises(BenchmarkConfigError, match="interrupted input amendment"):
        study.run_study_workflow(**setup["common"], run_id="pilot")
    assert (setup["root"] / "study.json").read_bytes() == journal
    monkeypatch.setattr(amendments, "atomic_write_json", real_write)
    assert _amend(setup)["status"] == "amended"
    amendments.require_no_pending_amendments(setup["root"])


def test_recovery_rejects_corrupt_original_backup(stopped_study, monkeypatch):
    setup = stopped_study
    real_apply = amendments._apply_prepared
    monkeypatch.setattr(amendments, "_apply_prepared", lambda *_: {"status": "prepared"})
    assert _amend(setup)["status"] == "prepared"
    transaction = next((setup["root"] / "amendments").glob("*/amendment.json")).parent
    (transaction / "before/study.json").write_text("tampered backup")
    monkeypatch.setattr(amendments, "_apply_prepared", real_apply)
    with pytest.raises(BenchmarkConfigError, match="Study files changed"):
        _amend(setup)


def test_unrelated_scientific_change_is_rejected(stopped_study):
    setup = stopped_study
    repo = setup["common"]["repo_root"]
    _write(repo / SCENARIO, {"new_design": True})
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Unrelated design change")
    with pytest.raises(BenchmarkConfigError, match="Only pending simulator requests"):
        _amend(setup)


def test_seed_changes_are_not_parameter_repairs(stopped_study):
    setup = stopped_study
    repo = setup["common"]["repo_root"]
    request = json.loads((repo / REQUEST).read_text())
    request["runs"][0]["base_seed"] = 99
    _write(repo / REQUEST, request)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Change seed")
    with pytest.raises(BenchmarkConfigError, match="not seeds"):
        _amend(setup)


def test_andrea_metadata_can_advance_without_rewriting_frozen_locks(stopped_study):
    setup = stopped_study
    repo, andrea = setup["common"]["repo_root"], setup["common"]["andrea_repo"]
    (andrea / "README.md").write_text("Improve diagnostics")
    _git(andrea, "add", ".")
    _git(andrea, "commit", "-qm", "Diagnostics")
    head = _git(andrea, "rev-parse", "HEAD")
    _write(repo / "repos.lock.json", {"repositories": {"andrea": {"commit": head}}})
    _write(repo / amendments._MATRIX, {"andrea_commit": head})
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Update ANDREA identity")
    old_lock = (setup["frozen"] / "repos.lock.json").read_bytes()
    assert _amend(setup)["status"] == "amended"
    assert (setup["frozen"] / "repos.lock.json").read_bytes() == old_lock


def test_changed_lock_fields_other_than_andrea_commit_are_rejected(stopped_study):
    setup = stopped_study
    repo = setup["common"]["repo_root"]
    lock = json.loads((repo / "repos.lock.json").read_text())
    lock["repositories"]["andrea"]["url"] = "https://invalid.example/changed"
    _write(repo / "repos.lock.json", lock)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Change repository origin")
    with pytest.raises(BenchmarkConfigError, match="Only the observed ANDREA commit"):
        _amend(setup)
