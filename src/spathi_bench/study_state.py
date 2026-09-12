"""Durable identity, scientific-input ledger and phase receipts for a study."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import BenchmarkConfigError, load_json_object
from .provenance import observe_benchmark_repository
from .safe_io import atomic_write_json, path_below_no_symlinks

logger = logging.getLogger(__name__)
PHASES = (
    "calibration", "efficiency", "dimensional_transfer", "comparison", "scalability", "performance",
)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def study_directory(repo_root: Path, work_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", run_id):
        raise BenchmarkConfigError("Study ID must contain 1–96 lowercase letters, digits, hyphens or underscores")
    work = path_below_no_symlinks(work_root, repo_root / "work", "Study work directory")
    return path_below_no_symlinks(work / "studies" / run_id, work, "Study directory")


def new_id(profile: str) -> str:
    return f"{profile}-{datetime.now(UTC):%Y%m%dt%H%M%Sz}-{uuid.uuid4().hex[:6]}"


def observe_revisions(repo_root: Path, andrea_repo: Path, spathi_repo: Path) -> dict[str, Any]:
    # A clean revision is always attributable. A resume may change revisions,
    # but it may not silently execute unrecorded edits in a dirty checkout.
    return {
        name: observe_benchmark_repository(path)
        for name, path in (("benchmark", repo_root), ("andrea", andrea_repo), ("spathi", spathi_repo))
    }


def scientific_inputs(repo_root: Path) -> dict[str, str]:
    """Freeze experimental choices, not implementation/doc revision identities."""
    paths = [
        path for path in (repo_root / "experiments").rglob("*")
        if path.is_file() and path.suffix != ".md"
    ]
    paths.extend(
        path for name in ("repos.lock.json", "images.lock.json", "constraints/benchmark.txt")
        if (path := repo_root / name).is_file()
    )
    return {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in sorted(paths)
    }


def load_study(study_root: Path) -> dict[str, Any]:
    state = load_json_object(study_root / "study.json", "study journal")
    if (
        state.get("schema_version") != "1.0"
        or state.get("workflow_type") != "study"
        or state.get("run_id") != study_root.name
        or state.get("profile") not in {"smoke", "reduced", "full"}
        or not isinstance(state.get("phases"), dict)
    ):
        raise BenchmarkConfigError("Invalid study journal")
    phases = ("smoke",) if state["profile"] == "smoke" else PHASES
    if set(state["phases"]) != set(phases) or any(
        not isinstance(record, dict) or record.get("status") not in {"pending", "running", "complete", "failed", "interrupted"}
        for record in state["phases"].values()
    ):
        raise BenchmarkConfigError("Invalid study phase coverage or status")
    state["phases"] = {name: state["phases"][name] for name in phases}
    return state


def save_study(study_root: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    atomic_write_json(study_root / "study.json", state, "study journal", boundary=study_root)


def record_invocation(
    state: dict[str, Any], revisions: dict[str, Any], inputs: dict[str, str],
) -> None:
    if state["scientific_inputs"] != inputs:
        changed = sorted(
            key for key in set(inputs) | set(state["scientific_inputs"])
            if inputs.get(key) != state["scientific_inputs"].get(key)
        )
        raise BenchmarkConfigError(
            "Study scientific inputs changed (" + ", ".join(changed[:5])
            + "). Start a new study ID; seeds, datasets, search choices and execution locks cannot change on resume."
        )
    previous = state["invocations"][-1]["revisions"] if state["invocations"] else revisions
    for name, observed in revisions.items():
        if previous[name]["origin"] != observed["origin"]:
            raise BenchmarkConfigError(f"{name} repository origin changed within the study")
        if previous[name]["commit"] != observed["commit"]:
            warning = (
                f"{name} revision changed from {previous[name]['commit'][:12]} to "
                f"{observed['commit'][:12]}. Resuming preserved results; review mixed-revision "
                "provenance before publication."
            )
            logger.warning(warning)
            state["warnings"].append(warning)
    state["invocations"].append({"started_at": now(), "revisions": revisions})


def phase_receipt(
    repo_root: Path, study_root: Path, workflow_root: Path, result: dict[str, Any],
) -> dict[str, Any]:
    """Seal small phase evidence; large artifacts retain their own integrity manifests."""
    references: dict[str, str] = {}
    for path in sorted(workflow_root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".csv", ".tsv"}:
            safe = path_below_no_symlinks(path, study_root, "Phase evidence")
            references[safe.relative_to(study_root).as_posix()] = sha256_file(safe)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                path = path_below_no_symlinks(repo_root / value["path"], study_root, "Phase output")
                if not path.is_file() or sha256_file(path) != value["sha256"]:
                    raise BenchmarkConfigError(f"Phase output checksum mismatch: {path}")
                references[path.relative_to(study_root).as_posix()] = value["sha256"]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    # Only published reports have this universal reference contract. Other
    # provenance can reference committed inputs outside the runtime directory.
    visit(result.get("reports", {}))

    def collect_references(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect_references(child)
        elif isinstance(value, list):
            for child in value:
                collect_references(child)
        elif isinstance(value, str) and Path(value).suffix in {".json", ".csv", ".tsv"}:
            path = repo_root / value
            if path.is_relative_to(study_root) and path.is_file():
                safe = path_below_no_symlinks(path, study_root, "Referenced phase evidence")
                references[safe.relative_to(study_root).as_posix()] = sha256_file(safe)

    collect_references(result)
    if not references:
        raise BenchmarkConfigError("Completed phase has no durable evidence")
    receipt: dict[str, Any] = {
        "status": "complete", "flow_id": result["flow_id"], "completed_at": now(),
        "evidence": references,
        "workflow_root": workflow_root.relative_to(study_root).as_posix(),
    }
    html = result.get("reports", {}).get("html") if isinstance(result.get("reports"), dict) else None
    if isinstance(html, dict):
        receipt["report"] = (repo_root / html["path"]).relative_to(study_root).as_posix()
    return receipt


def validate_receipt(study_root: Path, receipt: dict[str, Any]) -> None:
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise BenchmarkConfigError("Completed study phase has no evidence receipt")
    for relative, digest in evidence.items():
        path = path_below_no_symlinks(study_root / relative, study_root, "Study phase evidence")
        if not path.is_file() or sha256_file(path) != digest:
            raise BenchmarkConfigError(f"Completed study evidence is missing or changed: {relative}")
