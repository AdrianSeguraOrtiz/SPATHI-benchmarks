"""Crash-safe publication of immutable interactive workflow reports."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import BenchmarkConfigError, load_json_object
from .safe_io import atomic_write_text, path_below_no_symlinks

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPORT_MEMBERS = ("html", "manifest")


def _encoded_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _canonical_bundle_paths(
    *, repo_root: Path, html_path: Path, manifest_path: Path
) -> dict[str, str]:
    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    work_root = root / "work"
    expected: dict[str, str] = {}
    for member, path in (("html", html_path), ("manifest", manifest_path)):
        absolute = path_below_no_symlinks(
            path,
            work_root,
            f"Canonical report {member}",
        )
        expected[member] = absolute.relative_to(root).as_posix()
    return expected


def _validate_bundle(
    *,
    repo_root: Path,
    bundle: Any,
    expected_paths: Mapping[str, str],
    allow_missing: bool,
) -> dict[str, Any]:
    if not isinstance(bundle, dict) or set(bundle) != {
        "schema_version",
        *_REPORT_MEMBERS,
    }:
        raise BenchmarkConfigError("Workflow report bundle has an invalid structure")
    if bundle.get("schema_version") != "1.0":
        raise BenchmarkConfigError("Workflow report bundle has an unsupported schema")

    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    validated: dict[str, Any] = {"schema_version": "1.0"}
    for member in _REPORT_MEMBERS:
        reference = bundle.get(member)
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise BenchmarkConfigError(f"Workflow report {member} reference is invalid")
        relative = reference.get("path")
        digest = reference.get("sha256")
        if relative != expected_paths[member] or not isinstance(digest, str):
            raise BenchmarkConfigError(f"Workflow report {member} reference is not canonical")
        if not _SHA256.fullmatch(digest):
            raise BenchmarkConfigError(f"Workflow report {member} reference has no valid SHA-256")
        path = path_below_no_symlinks(
            root / relative,
            root / "work",
            f"Workflow report {member}",
        )
        if path.exists():
            if not path.is_file():
                raise BenchmarkConfigError(f"Workflow report {member} is not a file")
            observed = sha256_file(path)
            if observed != digest:
                raise BenchmarkConfigError(
                    f"Workflow report {member} changed after publication: "
                    f"expected {digest}, observed {observed}"
                )
        elif not allow_missing:
            raise BenchmarkConfigError(f"Workflow report {member} is missing: {path}")
        validated[member] = {"path": relative, "sha256": digest}
    return validated


def publish_workflow_report(
    *,
    repo_root: Path,
    workflow_path: Path,
    publishing_state: dict[str, Any],
    html_path: Path,
    manifest_path: Path,
    render: Callable[[], dict[str, Any]],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish a deterministic report, then atomically mark its workflow complete.

    A missing member of an already published bundle may be reconstructed. Existing
    bytes are never silently accepted or replaced: the renderer must reproduce the
    exact bundle recorded in the final workflow state.
    """

    if publishing_state.get("status") != "publishing" or "reports" in publishing_state:
        raise BenchmarkConfigError(f"{label} publication must start from a clean state")
    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    path = path_below_no_symlinks(
        workflow_path,
        root / "work",
        f"{label} workflow state",
    )
    expected_paths = _canonical_bundle_paths(
        repo_root=repo_root,
        html_path=html_path,
        manifest_path=manifest_path,
    )

    completed_state: dict[str, Any] | None = None
    recorded_bundle: dict[str, Any] | None = None
    if path.exists():
        if not path.is_file():
            raise BenchmarkConfigError(f"{label} workflow state is not a file")
        existing = load_json_object(path, f"{label} workflow state")
        status = existing.get("status")
        comparable = dict(existing)
        if status == "complete":
            recorded_bundle = _validate_bundle(
                repo_root=repo_root,
                bundle=comparable.pop("reports", None),
                expected_paths=expected_paths,
                allow_missing=True,
            )
            comparable["status"] = "publishing"
            completed_state = existing
        elif status != "publishing":
            raise BenchmarkConfigError(
                f"Existing {label} workflow has an invalid publication status"
            )
        if comparable != publishing_state:
            raise BenchmarkConfigError(
                f"Existing {label} workflow differs from the current publication"
            )
    else:
        atomic_write_text(path, _encoded_json(publishing_state), f"{label} workflow state")

    # Rendering is also the deterministic verification step on every completed
    # resume. Immutable report writers reject altered existing bytes, while absent
    # members are recreated from the frozen scientific inputs.
    rendered_bundle = _validate_bundle(
        repo_root=repo_root,
        bundle=render(),
        expected_paths=expected_paths,
        allow_missing=False,
    )
    if recorded_bundle is not None:
        if rendered_bundle != recorded_bundle:
            raise BenchmarkConfigError(
                f"Reconstructed {label} report differs from its published bundle"
            )
        if completed_state is None:  # pragma: no cover - defensive invariant
            raise AssertionError("Completed report publication lost its workflow state")
        return completed_state, recorded_bundle

    final_state = {
        **publishing_state,
        "status": "complete",
        "reports": rendered_bundle,
    }
    current = path.read_text(encoding="utf-8")
    if current != _encoded_json(publishing_state):
        raise BenchmarkConfigError(f"{label} workflow changed during report publication")
    atomic_write_text(path, _encoded_json(final_state), f"{label} workflow state")
    return final_state, rendered_bundle


__all__ = ["publish_workflow_report"]
