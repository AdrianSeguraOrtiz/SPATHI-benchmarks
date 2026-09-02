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

REPORT_SCHEMA_VERSION = "1.0"
RENDERER_SCHEMA_VERSION = "1.0"
_REPORT_FLOW_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
_REPORT_SCOPES = {
    "calibration": {"reduced", "full"},
    "comparison": {"reduced", "full"},
}


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkConfigError(f"{label} must be an object")
    return value


def require_array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkConfigError(f"{label} must be an array")
    return value


def require_completed_workflow(path: Path, workflow_type: str) -> dict[str, Any]:
    workflow = load_json_object(path, f"{workflow_type} workflow")
    if workflow.get("schema_version") != "1.0":
        raise BenchmarkConfigError(f"Unsupported {workflow_type} workflow schema")
    if workflow.get("workflow_type") != workflow_type:
        raise BenchmarkConfigError(f"Expected a {workflow_type} workflow: {path}")
    if workflow.get("status") != "complete":
        raise BenchmarkConfigError(
            f"Interactive {workflow_type} reports require a completed workflow: {path}"
        )
    if not isinstance(workflow.get("flow_id"), str) or not workflow["flow_id"]:
        raise BenchmarkConfigError(f"Completed {workflow_type} workflow has no flow_id")
    require_mapping(workflow.get("identity"), f"{workflow_type} workflow identity")
    return workflow


def path_below(path: Path, root: Path, label: str, *, file: bool = True) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise BenchmarkConfigError(f"{label} escapes {resolved_root}")
    if file and not resolved.is_file():
        raise BenchmarkConfigError(f"{label} is missing: {resolved}")
    if not file and not resolved.is_dir():
        raise BenchmarkConfigError(f"{label} directory is missing: {resolved}")
    return resolved


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

    if kind not in {"calibration", "comparison"}:
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


def _render_html(*, kind: str, payload: dict[str, Any], renderer: dict[str, Any]) -> str:
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
    return template.render(
        title=(
            "SPATHI calibration report" if kind == "calibration" else "SPATHI benchmark comparison"
        ),
        report_kind=kind,
        report_data_json=json_for_script(payload),
        plotly_js=renderer["plotly_js"],
        report_css=texts["assets/report.css"],
        common_js=texts["assets/common.js"],
        report_js=texts[f"assets/{kind}.js"],
    )


def render_html(*, kind: str, payload: dict[str, Any]) -> str:
    return _render_html(kind=kind, payload=payload, renderer=_renderer_material(kind))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_report_bundle(
    *,
    repo_root: Path,
    kind: str,
    output_path: Path,
    payload: dict[str, Any],
    input_paths: list[Path],
    manifest_name: str,
) -> dict[str, Any]:
    """Write one deterministic, self-contained HTML document and its provenance manifest."""

    root = repo_root.resolve()
    output = output_path.resolve()
    if not output.is_relative_to(root / "work"):
        raise BenchmarkConfigError("Interactive reports must be written below repository work/")
    inputs = []
    for source in sorted({path.resolve() for path in input_paths}, key=str):
        path_below(source, root, "interactive report input")
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
    html = _render_html(kind=kind, payload=payload, renderer=renderer)
    if not html.endswith("\n"):
        html += "\n"
    _atomic_write(output, html)
    manifest_path = output.parent / manifest_name
    manifest = {
        **report_identity,
        "fingerprint": scientific_fingerprint,
        "fingerprint_scope": "scientific_inputs",
        "renderer": renderer["provenance"],
        "output": {
            "path": output.relative_to(root).as_posix(),
            "sha256": sha256_file(output),
            "self_contained": True,
        },
    }
    _atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "html": {
            "path": output.relative_to(root).as_posix(),
            "sha256": sha256_file(output),
        },
        "manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
    }
