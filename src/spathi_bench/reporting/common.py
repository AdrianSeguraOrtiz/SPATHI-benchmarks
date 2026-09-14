"""Shared validation and rendering utilities for interactive reports."""

from __future__ import annotations

import json
import os
import re
from hashlib import sha256
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from jinja2 import DictLoader, Environment, select_autoescape
from plotly.offline import get_plotlyjs

from ..artifacts import canonical_json_sha256, sha256_file
from ..config import BenchmarkConfigError, load_json_object
from ..safe_io import atomic_write_text, path_below_no_symlinks, write_idempotent_text

REPORT_SCHEMA_VERSION = "1.0"
RENDERER_SCHEMA_VERSION = "1.0"
_REPORT_FLOW_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
_REPORT_SCOPES = {
    "calibration": {"reduced", "full"},
    "comparison": {"reduced", "full"},
    "dimensional_transfer": {"reduced", "full"},
    "performance": {"reduced", "full"},
    "scalability": {"reduced", "full"},
}


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkConfigError(f"{label} must be an object")
    return value


def require_array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkConfigError(f"{label} must be an array")
    return value


def require_reportable_workflow(path: Path, workflow_type: str) -> dict[str, Any]:
    """Load a workflow whose immutable results are ready for report publication.

    ``publishing`` is deliberately distinct from ``complete``: report-producing
    workflows enter it before writing their HTML bundle and become complete only
    after the bundle references and hashes have been committed to ``workflow.json``.
    """

    workflow = load_json_object(path, f"{workflow_type} workflow")
    if workflow.get("schema_version") != "1.0":
        raise BenchmarkConfigError(f"Unsupported {workflow_type} workflow schema")
    if workflow.get("workflow_type") != workflow_type:
        raise BenchmarkConfigError(f"Expected a {workflow_type} workflow: {path}")
    status = workflow.get("status")
    if status not in {"publishing", "complete"}:
        raise BenchmarkConfigError(
            f"Interactive {workflow_type} reports require a publishable workflow: {path}"
        )
    if status == "publishing" and "reports" in workflow:
        raise BenchmarkConfigError(
            f"Publishing {workflow_type} workflow already records a report bundle"
        )
    if status == "complete":
        require_mapping(workflow.get("reports"), f"completed {workflow_type} report bundle")
    if not isinstance(workflow.get("flow_id"), str) or not workflow["flow_id"]:
        raise BenchmarkConfigError(f"Publishable {workflow_type} workflow has no flow_id")
    require_mapping(workflow.get("identity"), f"{workflow_type} workflow identity")
    return workflow


def reconcile_report_bundle(
    workflow: dict[str, Any], bundle: dict[str, Any], workflow_type: str
) -> dict[str, Any]:
    """Require a completed workflow to retain the exact deterministic bundle."""

    if workflow["status"] == "complete" and workflow.get("reports") != bundle:
        raise BenchmarkConfigError(
            f"Rendered {workflow_type} report differs from its completed workflow record"
        )
    return bundle


def path_below(path: Path, root: Path, label: str, *, file: bool = True) -> Path:
    return path_below_no_symlinks(
        path,
        root,
        label,
        require_file=file,
        require_directory=not file,
    )


def repo_file(repo_root: Path, raw_path: Any, label: str, *, below: Path | None = None) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise BenchmarkConfigError(f"{label} path is missing")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return path_below(candidate, below or repo_root, label)


def verify_sha256(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise BenchmarkConfigError(f"{label} has no valid recorded SHA-256")
    observed = sha256_file(path)
    if observed != expected:
        raise BenchmarkConfigError(
            f"{label} changed after the workflow was recorded: expected {expected}, "
            f"observed {observed}"
        )


def safe_relative_link(report_path: Path, target: Path, allowed_root: Path) -> str:
    safe_target = path_below(target, allowed_root, "report link")
    return Path(os.path.relpath(safe_target, report_path.parent.resolve())).as_posix()


def validated_export_link(exports_root: Path, raw_path: Any, report_path: Path) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise BenchmarkConfigError("Export bundle has no analysis_zip path")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise BenchmarkConfigError(f"Unsafe export bundle path: {raw_path!r}")
    return safe_relative_link(report_path, exports_root / relative, exports_root)


def central_report_path(
    *,
    version_root: Path,
    report_type: str,
    scope: str,
    flow_id: str,
) -> Path:
    """Return the single canonical HTML location for one completed workflow."""

    allowed_scopes = _REPORT_SCOPES.get(report_type)
    if allowed_scopes is None:
        raise BenchmarkConfigError(f"Unsupported interactive report type: {report_type!r}")
    if scope not in allowed_scopes:
        raise BenchmarkConfigError(
            f"Invalid {report_type} report scope {scope!r}; expected reduced or full"
        )
    if not _REPORT_FLOW_ID.fullmatch(flow_id):
        raise BenchmarkConfigError(f"Invalid interactive report flow ID: {flow_id!r}")
    reports_root = version_root.resolve() / "reports"
    output = (reports_root / scope / flow_id / f"{report_type}.html").resolve()
    if not output.is_relative_to(reports_root):
        raise BenchmarkConfigError("Interactive report path escapes the central report root")
    return output


def json_for_script(payload: dict[str, Any]) -> str:
    """Serialize JSON without allowing a value to terminate its script element."""

    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
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


def _text_sha256(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _renderer_material(kind: str) -> dict[str, Any]:
    """Load and fingerprint the exact renderer resources used for one report kind."""

    if kind not in {
        "calibration",
        "comparison",
        "dimensional_transfer",
        "performance",
        "scalability",
    }:
        raise ValueError(f"Unsupported report kind: {kind}")
    resource_names = (
        "common.py",
        f"{kind}.py",
        "templates/base.html.j2",
        f"templates/{kind}.html.j2",
        "assets/report.css",
        "assets/common.js",
        f"assets/{kind}.js",
    )
    texts = {name: _resource_text(name) for name in resource_names}
    plotly_js = get_plotlyjs()
    resources_provenance = {
        name: {
            "sha256": _text_sha256(content),
            "size_bytes": len(content.encode("utf-8")),
        }
        for name, content in texts.items()
    }
    resources_provenance["vendor/plotly.js"] = {
        "sha256": _text_sha256(plotly_js),
        "size_bytes": len(plotly_js.encode("utf-8")),
    }
    renderer_identity = {
        "schema_version": RENDERER_SCHEMA_VERSION,
        "report_type": kind,
        "dependencies": {
            "jinja2": metadata.version("Jinja2"),
            "plotly": metadata.version("plotly"),
        },
        "resources": resources_provenance,
    }
    return {
        "texts": texts,
        "plotly_js": plotly_js,
        "provenance": {
            **renderer_identity,
            "fingerprint": canonical_json_sha256(renderer_identity),
        },
    }


def render_html(
    *,
    kind: str,
    payload: dict[str, Any],
    renderer: dict[str, Any] | None = None,
) -> str:
    """Render a report, reusing preloaded renderer material when available."""

    if renderer is None:
        renderer = _renderer_material(kind)
    texts = renderer["texts"]
    environment = Environment(
        loader=DictLoader(
            {
                "base.html.j2": texts["templates/base.html.j2"],
                f"{kind}.html.j2": texts[f"templates/{kind}.html.j2"],
            }
        ),
        autoescape=select_autoescape(("html", "xml")),
        keep_trailing_newline=True,
    )
    template = environment.get_template(f"{kind}.html.j2")
    titles = {
        "calibration": "SPATHI calibration report",
        "comparison": "SPATHI benchmark comparison",
        "dimensional_transfer": "SPATHI dimensional accuracy transfer",
        "performance": "SPATHI computational performance",
        "scalability": "SPATHI computational scalability",
    }
    return template.render(
        title=titles[kind],
        report_kind=kind,
        report_data_json=json_for_script(payload),
        plotly_js=renderer["plotly_js"],
        report_css=texts["assets/report.css"],
        common_js=texts["assets/common.js"],
        report_js=texts[f"assets/{kind}.js"],
    )


def _publish_immutable_text(path: Path, content: str, label: str) -> None:
    """Publish deterministic report material without overwriting existing bytes."""

    write_idempotent_text(path, content, label)


def write_report_bundle(
    *,
    repo_root: Path,
    kind: str,
    output_path: Path,
    payload: dict[str, Any],
    input_paths: list[Path],
    manifest_name: str,
    immutable: bool = False,
) -> dict[str, Any]:
    """Write one deterministic, self-contained HTML document and its provenance manifest."""

    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    output = path_below_no_symlinks(
        output_path,
        root / "work",
        "Interactive report output",
    )
    inputs = []
    for candidate in sorted({Path(os.path.abspath(path)) for path in input_paths}, key=str):
        source = path_below(candidate, root, "interactive report input")
        inputs.append(
            {
                "path": source.relative_to(root).as_posix(),
                "sha256": sha256_file(source),
            }
        )
    report_identity = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": kind,
        "inputs": inputs,
        "workflow_identity": payload.get("workflow", {}).get("identity"),
        "flow_id": payload.get("workflow", {}).get("flow_id"),
    }
    scientific_fingerprint = canonical_json_sha256(report_identity)
    renderer = _renderer_material(kind)
    payload = {
        **payload,
        "report": {
            "schema_version": REPORT_SCHEMA_VERSION,
            "fingerprint": scientific_fingerprint,
            "fingerprint_scope": "scientific_inputs",
            "renderer": renderer["provenance"],
            "self_contained": True,
            "statistics_recomputed": False,
        },
    }
    html = render_html(kind=kind, payload=payload, renderer=renderer)
    if not html.endswith("\n"):
        html += "\n"
    manifest_path = output.parent / manifest_name
    output_sha256 = _text_sha256(html)
    manifest = {
        **report_identity,
        "fingerprint": scientific_fingerprint,
        "fingerprint_scope": "scientific_inputs",
        "renderer": renderer["provenance"],
        "output": {
            "path": output.relative_to(root).as_posix(),
            "sha256": output_sha256,
            "self_contained": True,
        },
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable:
        _publish_immutable_text(output, html, f"{kind} report HTML")
        _publish_immutable_text(
            manifest_path,
            manifest_text,
            f"{kind} report manifest",
        )
    else:
        atomic_write_text(output, html, f"{kind} report HTML")
        atomic_write_text(manifest_path, manifest_text, f"{kind} report manifest")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "html": {
            "path": output.relative_to(root).as_posix(),
            "sha256": output_sha256,
        },
        "manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
    }
