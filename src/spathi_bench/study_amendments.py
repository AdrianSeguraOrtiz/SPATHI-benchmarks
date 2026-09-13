"""Explicit, journaled corrections to generation requests that have produced no data."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .config import (
    BenchmarkConfigError,
    load_json_object,
    validate_all_profiles,
    validate_design_contracts,
)
from .provenance import frozen_profile, frozen_repository_root, validate_run_inputs
from .runtime import andrea_command
from .safe_io import atomic_write_json, atomic_write_text, path_below_no_symlinks
from .state import load_state, run_lock, status_at_least
from .study_context import study_execution
from .study_state import (
    load_study,
    now,
    observe_revisions,
    scientific_inputs,
    study_directory,
    validate_receipt,
)

_MATRIX = "experiments/v1/inference/catalog-matrix.json"


def _pending_amendments(study_root: Path) -> list[Path]:
    directory = path_below_no_symlinks(study_root / "amendments", study_root, "Study amendments")
    return [
        path_below_no_symlinks(path, directory, "Prepared input amendment", require_file=True)
        for path in sorted(directory.glob("[0-9a-f]*/amendment.json"))
        if not (path.parent / "applied.json").is_file()
    ]


def require_no_pending_amendments(study_root: Path) -> None:
    """Do not let a normal run change the journal while an amendment is being recovered."""
    for path in _pending_amendments(study_root):
        reason = shlex.quote(load_json_object(path, "Prepared input amendment")["reason"])
        raise BenchmarkConfigError(
            f"Study {study_root.name} has an interrupted input amendment. Repeat "
            f"spathi-bench amend-inputs {study_root.name} --reason {reason} before resuming."
        )


def _recorded_json(repo_root: Path, state: dict[str, Any], relative: str) -> dict[str, Any]:
    """Recover a committed old input by its recorded checksum, not an assumed revision."""
    records = [*state["invocations"], *state.get("input_amendments", [])]
    for record in reversed(records):
        commit = record["revisions"]["benchmark"]["commit"]
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
            capture_output=True, check=False,
        )
        if result.returncode == 0:
            if hashlib.sha256(result.stdout).hexdigest() == state["scientific_inputs"][relative]:
                return json.loads(result.stdout)
    raise BenchmarkConfigError(f"Cannot recover the recorded committed input: {relative}")


def _validate_changes(
    repo_root: Path, state: dict[str, Any], inputs: dict[str, str], revisions: dict[str, Any],
) -> list[str]:
    previous = state["scientific_inputs"]
    changed = sorted(key for key in set(previous) | set(inputs) if previous.get(key) != inputs.get(key))
    if any(key not in previous or key not in inputs for key in changed):
        raise BenchmarkConfigError("Input amendment cannot add or remove experimental files")
    profiles = validate_all_profiles(repo_root)
    generation_requests = {
        job["simulator_runs"]
        for profile in profiles
        for job in profile.get("generation_jobs", [])
        if "simulator_runs" in job
    }
    for relative in changed:
        if relative in generation_requests:
            old = _recorded_json(repo_root, state, relative)
            new = load_json_object(repo_root / relative, "Amended simulator request")
            for payload in (old, new):
                for run in payload.get("runs", []):
                    run.pop("params", None)
            if old != new:
                raise BenchmarkConfigError(f"Only simulator parameters may change, not seeds or run identities: {relative}")
            continue
        if relative not in {"repos.lock.json", _MATRIX}:
            raise BenchmarkConfigError(f"Only pending simulator requests can be amended: {relative}")
        old = _recorded_json(repo_root, state, relative)
        new = load_json_object(repo_root / relative, "amended repository metadata")
        if relative == "repos.lock.json":
            old["repositories"]["andrea"]["commit"] = revisions["andrea"]["commit"]
        else:
            old["andrea_commit"] = revisions["andrea"]["commit"]
        if old != new:
            raise BenchmarkConfigError(f"Only the observed ANDREA commit may change in {relative}")
    validate_design_contracts(repo_root)
    return changed


def _stage_replacement(
    transaction: Path, study_root: Path, target: Path, text: str, replacements: list[dict[str, Any]],
) -> None:
    target = path_below_no_symlinks(target, study_root, "Amended study file", require_file=True)
    relative = target.relative_to(study_root).as_posix()
    old_path, new_path = transaction / "before" / relative, transaction / "after" / relative
    old_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(target, old_path)
    atomic_write_text(new_path, text, "Prepared input amendment", boundary=transaction)
    replacements.append({"path": relative, "before_sha256": sha256_file(old_path), "after_sha256": sha256_file(new_path)})


def _prepare_snapshot(
    *, repo_root: Path, study_root: Path, run_root: Path, transaction: Path,
    changed: list[str], andrea_repo: Path, replacements: list[dict[str, Any]],
) -> None:
    # These helpers define the actual cache identity; do not approximate it using dataset names.
    from .pipeline import _artifact_store, _generation_spec

    raw = load_json_object(run_root / "input_snapshot/manifest.json", "Frozen input manifest")
    profile_id = raw["profile_id"]
    manifest = validate_run_inputs(repo_root, profile_id, run_root)
    profile = frozen_profile(run_root, profile_id)
    frozen_root = frozen_repository_root(run_root)
    affected = [job for job in profile["generation_jobs"] if job.get("simulator_runs") in changed]
    if not affected:
        return
    state = load_state(run_root, profile_id)
    if status_at_least(state, "generated"):
        raise BenchmarkConfigError(f"Cannot amend inputs after generation completed: {run_root}")
    store = _artifact_store(run_root)
    resources = state["execution_plan"]["operations"]["generate_data"]
    for job in affected:
        spec = _generation_spec(
            state=state, job=job, scenario=frozen_root / job["scenario"],
            simulator_runs=frozen_root / job["simulator_runs"], input_root=frozen_root,
            resources=resources,
        )
        if store.lookup(store.key_for("dataset", spec)) is not None:
            raise BenchmarkConfigError(f"Cannot amend a request with a published dataset: {job['id']}")
        plan = run_root / "preflight" / f"{job['id']}-simulation-plan.json"
        staged_plan = transaction / "plans" / run_root.relative_to(study_root) / plan.name
        staged_plan.parent.mkdir(parents=True, exist_ok=True)
        # Planning is cheap, does not start Docker, and rejects unsupported simulator parameters.
        andrea_command(
            ["generate-data", "plan", "--scenario", str(frozen_root / job["scenario"]),
             "--simulator-runs", str(repo_root / job["simulator_runs"]), "--out", str(staged_plan),
             "--max-parallel-tasks", str(resources["max_parallel_tasks"]),
             "--max-cores", str(resources["cpu_threads"]), "--max-ram-gb", str(resources["memory_gib"])],
            repo_root=repo_root, andrea_repo=andrea_repo,
            log_path=staged_plan.with_suffix(".log"),
        )
        _stage_replacement(transaction, study_root, plan, staged_plan.read_text(), replacements)
    request_paths = {job["simulator_runs"] for job in affected}
    updated = copy.deepcopy(manifest)
    for entry in updated["files"]:
        relative = entry["path"]
        if relative in request_paths:
            source = repo_root / relative
            _stage_replacement(transaction, study_root, frozen_root / relative, source.read_text(), replacements)
            entry.update(sha256=sha256_file(source), size=source.stat().st_size)
    _stage_replacement(
        transaction, study_root, run_root / "input_snapshot/manifest.json",
        json.dumps(updated, indent=2, sort_keys=True) + "\n", replacements,
    )


def _apply_prepared(transaction: Path, study_root: Path, inputs: dict[str, str]) -> dict[str, Any]:
    manifest = load_json_object(transaction / "amendment.json", "Prepared input amendment")
    if manifest["scientific_inputs"] != inputs:
        raise BenchmarkConfigError("Checkout inputs changed during a prepared amendment; restore them before retrying")
    # Validate the whole transaction before writing, including recovery after a partial application.
    for record in manifest["replacements"]:
        target = path_below_no_symlinks(study_root / record["path"], study_root, "Amended study file", require_file=True)
        original = path_below_no_symlinks(transaction / "before" / record["path"], transaction, "Original amendment evidence", require_file=True)
        replacement = path_below_no_symlinks(transaction / "after" / record["path"], transaction, "Prepared replacement", require_file=True)
        if (
            sha256_file(target) not in {record["before_sha256"], record["after_sha256"]}
            or sha256_file(original) != record["before_sha256"]
            or sha256_file(replacement) != record["after_sha256"]
        ):
            raise BenchmarkConfigError(f"Study files changed during the amendment: {record['path']}")
    for record in manifest["replacements"]:
        target = study_root / record["path"]
        if sha256_file(target) != record["after_sha256"]:
            atomic_write_text(target, (transaction / "after" / record["path"]).read_text(), "Amended study input", boundary=study_root)
    atomic_write_json(transaction / "applied.json", {"applied_at": now()}, "Applied input amendment", boundary=transaction)
    return {"run_id": study_root.name, "status": "amended", "amendment": str(transaction), "resume_command": f"spathi-bench run --id {study_root.name}"}


def amend_study_inputs(
    *, repo_root: Path, andrea_repo: Path, spathi_repo: Path, work_root: Path, run_id: str, reason: str,
) -> dict[str, Any]:
    """Correct only unproduced generation inputs; never alter completed results or cache identities."""
    if not reason.strip():
        raise BenchmarkConfigError("An input amendment requires a non-empty reason")
    root = study_directory(repo_root, work_root, run_id)
    with run_lock(root):
        state = load_study(root)
        if state["status"] not in {"failed", "interrupted", "created"}:
            raise BenchmarkConfigError("Only a stopped, unfinished study can amend pending inputs")
        revisions = observe_revisions(repo_root, andrea_repo, spathi_repo)
        for name, observed in revisions.items():
            if state["invocations"][-1]["revisions"][name]["origin"] != observed["origin"]:
                raise BenchmarkConfigError(f"{name} repository origin changed within the study")
        inputs = scientific_inputs(repo_root)
        for record in state["phases"].values():
            if record["status"] == "complete":
                validate_receipt(root, record)
        for transaction in _pending_amendments(root):
            return _apply_prepared(transaction.parent, root, inputs)
        changed = _validate_changes(repo_root, state, inputs, revisions)
        if not changed:
            return {"run_id": run_id, "status": "unchanged", "resume_command": f"spathi-bench run --id {run_id}"}
        amendment_id = canonical_json_sha256({"before": state["scientific_inputs"], "after": inputs, "reason": reason})[:20]
        amendments_root = path_below_no_symlinks(root / "amendments", root, "Study amendments")
        amendments_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".preparing-", dir=amendments_root) as directory:
            temporary = Path(directory)
            replacements: list[dict[str, Any]] = []
            with study_execution(run_id, revisions=revisions, resume=True):
                for path in sorted((root / "v1").glob("*/*/input_snapshot/manifest.json")):
                    _prepare_snapshot(
                        repo_root=repo_root, study_root=root, run_root=path.parent.parent,
                        transaction=temporary, changed=changed, andrea_repo=andrea_repo,
                        replacements=replacements,
                    )
            record = {
                "id": amendment_id, "reason": reason.strip(), "created_at": now(), "revisions": revisions,
                "changed_inputs": {path: {"before": state["scientific_inputs"][path], "after": inputs[path]} for path in changed},
                "evidence": f"amendments/{amendment_id}/amendment.json",
            }
            state["scientific_inputs"] = inputs
            state.setdefault("input_amendments", []).append(record)
            state["warnings"].append(
                f"Explicit pending-input amendment {amendment_id}: {reason.strip()}. "
                "Original requests, plans and journal are preserved; completed results retain their original provenance."
            )
            state["updated_at"] = now()
            # The journal is applied last, so ordinary resume cannot use a half-updated snapshot.
            _stage_replacement(temporary, root, root / "study.json", json.dumps(state, indent=2, sort_keys=True) + "\n", replacements)
            atomic_write_json(
                temporary / "amendment.json",
                {"schema_version": "1.0", **record, "scientific_inputs": inputs, "replacements": replacements},
                "Prepared input amendment", boundary=temporary,
            )
            transaction = amendments_root / amendment_id
            os.rename(temporary, transaction)
        return _apply_prepared(transaction, root, inputs)


__all__ = ["amend_study_inputs", "require_no_pending_amendments"]
