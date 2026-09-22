"""Build portable, self-contained reports from a normalized study snapshot.

The reporting layer deliberately knows nothing about the execution database.  The
study core supplies one immutable snapshot and this module renders a disposable
view of it.  A rendering failure therefore cannot change a scientific task's
status, and the reports can always be rebuilt without repeating an inference.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from functools import lru_cache
from hashlib import sha256
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, select_autoescape
from plotly.offline import get_plotlyjs

from ..config import BenchmarkConfigError
from ..safe_io import atomic_write_text, path_below_no_symlinks

DELIVERABLE_SCHEMA_VERSION = "1.0"
_PHASES = ("calibration", "comparison", "scalability")
_STUDY_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_FILENAMES = {
    "article": "article-report.html",
    "calibration": "calibration.html",
    "comparison": "comparison.html",
    "scalability": "scalability.html",
}
_PREFERRED_COLUMNS = (
    "phase",
    "record_type",
    "dataset_id",
    "simulator",
    "dataset_role",
    "comparison_panel",
    "family",
    "configuration_id",
    "method_id",
    "scale_id",
    "metric",
    "metric_scope",
    "context",
    "group_size",
    "group_fraction",
    "group_size_class",
    "value",
    "status",
    "seed",
    "measurement",
    "requested_threads",
    "effective_threads",
    "threads",
    "ram_gb",
    "cells",
    "genes",
    "tf_count",
    "groups",
    "duration_seconds",
    "peak_memory_bytes",
)


def _json_text(value: Any, *, pretty: bool = False) -> str:
    """Serialize strict JSON and make it safe inside an HTML script element."""

    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError(f"Study snapshot is not strict JSON: {exc}") from exc
    return (
        rendered.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _resource_text(relative: str) -> str:
    node = resources.files("spathi_bench.reporting")
    for part in relative.split("/"):
        node = node.joinpath(part)
    return node.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _renderer() -> dict[str, Any]:
    template = _resource_text("templates/deliverable.html.j2")
    css = _resource_text("assets/deliverable.css")
    javascript = _resource_text("assets/deliverable.js")
    plotly = get_plotlyjs()
    material = {
        "template": template,
        "css": css,
        "javascript": javascript,
        "plotly": plotly,
    }
    resource_hashes = {
        name: sha256(content.encode("utf-8")).hexdigest() for name, content in material.items()
    }
    identity = {
        "name": "spathi-bench-portable-reports",
        "schema_version": DELIVERABLE_SCHEMA_VERSION,
        "dependencies": {
            "jinja2": metadata.version("Jinja2"),
            "plotly": metadata.version("plotly"),
        },
        "resources": resource_hashes,
    }
    identity["fingerprint"] = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**material, "identity": identity}


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkConfigError(f"{label} must be an object")
    return dict(value)


def _require_rows(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BenchmarkConfigError(f"{label} must be an array")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        rows.append(_require_mapping(item, f"{label}.{index}"))
    return rows


def normalize_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach the renderer input from mutable caller objects.

    Required shape::

        {"study": {"id", "profile", "status", "resources"},
         "phases": {phase: {"summary", "rows", "decisions", "failures", "provenance"}}}

    Phase collections may be empty while a study is running.  All three phase
    records must nevertheless be present so ``reduced`` exercises exactly the
    same reporting paths as ``full``.
    """

    root = _require_mapping(snapshot, "study snapshot")
    study = _require_mapping(root.get("study"), "study snapshot.study")
    study_id = study.get("id")
    if not isinstance(study_id, str) or not _STUDY_ID.fullmatch(study_id):
        raise BenchmarkConfigError("study snapshot.study.id must match [a-z0-9][a-z0-9._-]{0,127}")
    if study.get("profile") not in {"smoke", "reduced", "full"}:
        raise BenchmarkConfigError("study snapshot.study.profile must be smoke, reduced or full")
    if not isinstance(study.get("status"), str) or not study["status"]:
        raise BenchmarkConfigError("study snapshot.study.status must be a non-empty string")
    study["resources"] = _require_mapping(
        study.get("resources", {}), "study snapshot.study.resources"
    )

    raw_phases = _require_mapping(root.get("phases"), "study snapshot.phases")
    unexpected = sorted(set(raw_phases) - set(_PHASES))
    missing = sorted(set(_PHASES) - set(raw_phases))
    if unexpected or missing:
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if unexpected:
            detail.append(f"unexpected {', '.join(unexpected)}")
        raise BenchmarkConfigError(f"Study snapshot phases are invalid: {'; '.join(detail)}")

    phases: dict[str, Any] = {}
    for name in _PHASES:
        phase = _require_mapping(raw_phases[name], f"study snapshot.phases.{name}")
        phases[name] = {
            "summary": _require_mapping(
                phase.get("summary", {}), f"study snapshot.phases.{name}.summary"
            ),
            "rows": _require_rows(phase.get("rows", []), f"study snapshot.phases.{name}.rows"),
            "decisions": _require_rows(
                phase.get("decisions", []), f"study snapshot.phases.{name}.decisions"
            ),
            "failures": _require_rows(
                phase.get("failures", []), f"study snapshot.phases.{name}.failures"
            ),
            "provenance": _require_mapping(
                phase.get("provenance", {}), f"study snapshot.phases.{name}.provenance"
            ),
        }
    normalized = {
        "schema_version": DELIVERABLE_SCHEMA_VERSION,
        "study": study,
        "phases": phases,
    }
    # This also rejects NaN, infinities and non-JSON objects before any file is written.
    return json.loads(_json_text(normalized))


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten(item, label))
        elif isinstance(item, (list, tuple)):
            flattened[label] = json.dumps(
                item, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
        elif item is None:
            flattened[label] = ""
        elif isinstance(item, bool):
            flattened[label] = "true" if item else "false"
        else:
            flattened[label] = item
    return flattened


def rows_to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return deterministic RFC 4180-compatible text for arbitrary result rows."""

    flattened = [_flatten(row) for row in rows]
    if not flattened:
        return ""
    available = {key for row in flattened for key in row}
    columns = [key for key in _PREFERRED_COLUMNS if key in available]
    columns.extend(sorted(available - set(columns)))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flattened)
    return buffer.getvalue()


def _outcome_rows(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect the execution audit trail without conflating it with metrics.

    Metric rows, explicit comparison coverage and failures are separate records:
    retaining their origin makes the flat export complete without pretending that
    one metric is one execution.  The report itself uses explicit coverage when
    present; this CSV also preserves the underlying records for independent audit.
    """

    outcomes: list[dict[str, Any]] = []
    phases = snapshot["phases"]
    for phase_name in _PHASES:
        phase = phases[phase_name]
        for row in phase["rows"]:
            if row.get("status") not in (None, ""):
                outcomes.append({**row, "phase": phase_name, "record_type": "execution_outcome"})
        coverage = phase["summary"].get("execution_coverage", [])
        if isinstance(coverage, list):
            for row in coverage:
                if isinstance(row, Mapping):
                    outcomes.append(
                        {
                            **row,
                            "phase": phase_name,
                            "record_type": "execution_coverage",
                        }
                    )
        for row in phase["failures"]:
            outcomes.append({**row, "phase": phase_name, "record_type": "failure"})
    return outcomes


def _report_payload(snapshot: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "article":
        phases = snapshot["phases"]
    else:
        phases = {kind: snapshot["phases"][kind]}
    return {
        "schema_version": DELIVERABLE_SCHEMA_VERSION,
        "report_kind": kind,
        "study": snapshot["study"],
        "phases": phases,
    }


def _render_html(snapshot: dict[str, Any], kind: str) -> str:
    renderer = _renderer()
    environment = Environment(
        autoescape=select_autoescape(("html", "xml")),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = environment.from_string(renderer["template"])
    titles = {
        "article": "SPATHI benchmark · complete study",
        "calibration": "SPATHI benchmark · configuration selection",
        "comparison": "SPATHI benchmark · accuracy comparison",
        "scalability": "SPATHI benchmark · computational scalability",
    }
    html = template.render(
        title=titles[kind],
        report_kind=kind,
        report_data=_json_text(_report_payload(snapshot, kind)),
        report_css=renderer["css"],
        report_js=renderer["javascript"],
        plotly_js=renderer["plotly"],
    )
    return html if html.endswith("\n") else html + "\n"


def _sha256_text(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _directory_inventory(path: Path) -> set[str]:
    """Return the flat regular-file inventory required for one portable bundle."""

    inventory: set[str] = set()
    for member in path.iterdir():
        if member.is_symlink() or not member.is_file():
            raise BenchmarkConfigError(
                f"Study deliverable bundle contains a non-regular entry: {member}"
            )
        inventory.add(member.name)
    return inventory


def _publish_bundle(*, staging: Path, output: Path, root: Path, expected: set[str]) -> None:
    """Replace a complete bundle, restoring the previous directory on failure.

    Files are fully rendered below a unique sibling directory before publication.
    The old bundle is moved to a private rollback directory only for the duration
    of the final rename, so readers never observe a directory containing files
    from two report generations.
    """

    observed = _directory_inventory(staging)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise BenchmarkConfigError(
            f"Staged study bundle inventory is invalid: missing={missing}, unexpected={unexpected}"
        )

    rollback_root: Path | None = None
    previous: Path | None = None
    published = False
    try:
        if output.exists():
            output = path_below_no_symlinks(
                output, root, "Existing study deliverables", require_directory=True
            )
            rollback_root = Path(tempfile.mkdtemp(prefix=f".{output.name}-rollback-", dir=root))
            previous = rollback_root / "bundle"
            os.replace(output, previous)
        try:
            os.replace(staging, output)
            published = True
        except BaseException:
            if previous is not None and previous.exists() and not output.exists():
                os.replace(previous, output)
            raise
        if rollback_root is not None:
            shutil.rmtree(rollback_root)
            rollback_root = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if rollback_root is not None and rollback_root.exists():
            if previous is not None and previous.exists() and not output.exists():
                os.replace(previous, output)
            if rollback_root.exists():
                shutil.rmtree(rollback_root)

    if not published:
        raise RuntimeError("Study deliverable bundle was not published")
    final_inventory = _directory_inventory(output)
    if final_inventory != expected:
        raise BenchmarkConfigError("Published study bundle inventory changed unexpectedly")


def build_study_deliverables(
    snapshot: Mapping[str, Any], deliverables_root: Path
) -> dict[str, Any]:
    """Write one portable article bundle under ``deliverables/<study-id>``.

    Every HTML embeds its CSS, JavaScript, Plotly runtime and data.  The CSV files
    are convenient exports, not runtime dependencies of the reports.  Rebuilding
    publishes a complete staged directory and never mutates execution state or
    scientific results.
    """

    normalized = normalize_snapshot(snapshot)
    root = path_below_no_symlinks(deliverables_root, deliverables_root, "Deliverables root")
    output = path_below_no_symlinks(root / normalized["study"]["id"], root, "Study deliverables")
    root.mkdir(parents=True, exist_ok=True)
    root = path_below_no_symlinks(root, root, "Deliverables root", require_directory=True)

    rendered: dict[str, str] = {
        filename: _render_html(normalized, kind) for kind, filename in _FILENAMES.items()
    }
    for phase in _PHASES:
        rendered[f"results-{phase}.csv"] = rows_to_csv(normalized["phases"][phase]["rows"])
    rendered["results-outcomes.csv"] = rows_to_csv(_outcome_rows(normalized))

    snapshot_bytes = json.dumps(
        normalized, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    provenance = {
        "schema_version": DELIVERABLE_SCHEMA_VERSION,
        "study_id": normalized["study"]["id"],
        "profile": normalized["study"]["profile"],
        "status": normalized["study"]["status"],
        "snapshot_sha256": sha256(snapshot_bytes).hexdigest(),
        "study": normalized["study"],
        "phases": {name: normalized["phases"][name]["provenance"] for name in _PHASES},
        "renderer": _renderer()["identity"],
        "outputs": {
            name: {"sha256": _sha256_text(content), "size_bytes": len(content.encode("utf-8"))}
            for name, content in sorted(rendered.items())
        },
    }
    provenance_text = _json_text(provenance, pretty=True) + "\n"
    rendered["provenance.json"] = provenance_text
    checksums = "".join(
        f"{_sha256_text(content)}  {name}\n" for name, content in sorted(rendered.items())
    )
    rendered["SHA256SUMS"] = checksums

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{normalized['study']['id']}-staging-",
            dir=root,
        )
    )
    try:
        for filename, content in rendered.items():
            atomic_write_text(
                staging / filename,
                content,
                f"Study deliverable {filename}",
                boundary=root,
            )
        _publish_bundle(
            staging=staging,
            output=output,
            root=root,
            expected=set(rendered),
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return {
        "schema_version": DELIVERABLE_SCHEMA_VERSION,
        "study_id": normalized["study"]["id"],
        "directory": str(output),
        "files": {
            filename: {
                "path": str(output / filename),
                "sha256": _sha256_text(content),
                "size_bytes": len(content.encode("utf-8")),
            }
            for filename, content in sorted(rendered.items())
        },
    }


__all__ = [
    "DELIVERABLE_SCHEMA_VERSION",
    "build_study_deliverables",
    "normalize_snapshot",
    "rows_to_csv",
]
