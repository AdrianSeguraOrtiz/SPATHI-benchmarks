"""Portable study entry page linking the canonical reports of its scientific phases."""

from __future__ import annotations

import html
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .config import BenchmarkConfigError
from .reporting import (
    render_calibration_report,
    render_comparison_report,
    render_dimensional_transfer_report,
    render_efficiency_report,
    render_performance_report,
    render_scalability_report,
)
from .safe_io import atomic_write_text, path_below_no_symlinks
from .study_state import validate_receipt

_PHASES = {
    "calibration": (
        "Calibration", "Select the exact SPATHI policy by directed AUPR.",
        "Independent dyngen and scMultiSim development, high-TF and validation panels.",
    ),
    "efficiency": (
        "Efficiency", "Compare target eligibility and adaptive trees with exact SPATHI.",
        "Separate development and validation panels; full adds a conditional safety veto.",
    ),
    "dimensional_transfer": (
        "Dimensional transfer", "Inspect the frozen policy across dimensions and routing controls.",
        "Truth-bearing scMultiSim; reduced routing checks and full descriptive scale contrasts.",
    ),
    "comparison": (
        "Held-out comparison", "Compare frozen SPATHI with catalog defaults.",
        "Dynamic dyngen/scMultiSim and invariant SERGIO, BoolODE and GeneSPIDER2 controls.",
    ),
    "scalability": (
        "Scalability", "Measure SPATHI feasibility and declared resource boundaries.",
        "Public synthetic expression stress grid; computational evidence only.",
    ),
    "performance": (
        "Performance", "Measure isolated tool cost and parallel scaling.",
        "Independent fixed-size scMultiSim datasets, controlled threads and system repetitions.",
    ),
    "smoke": (
        "Docker smoke", "Exercise simulation, inference, evaluation and analysis archives.",
        "One tiny dyngen dataset, GENIE3 and SPATHI. No calibration or scientific selection.",
    ),
}

_RENDERERS = {
    "calibration": render_calibration_report,
    "efficiency": render_efficiency_report,
    "dimensional_transfer": render_dimensional_transfer_report,
    "comparison": render_comparison_report,
    "scalability": render_scalability_report,
    "performance": render_performance_report,
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _link(study_root: Path, relative: str, label: str) -> str:
    target = path_below_no_symlinks(study_root / relative, study_root, "Study report link")
    if not target.is_file():
        return f'<span class="muted">{_escape(label)} not yet published</span>'
    href = quote(os.path.relpath(target, study_root / "reports"), safe="/")
    return f'<a href="{href}">{_escape(label)}</a>'


def _phase_card(study_root: Path, name: str, record: Mapping[str, Any]) -> str:
    title, purpose, scope = _PHASES.get(name, (name.replace("_", " "), "", ""))
    status = str(record.get("status", "pending"))
    links: list[str] = []
    if isinstance(record.get("report"), str):
        links.append(_link(study_root, record["report"], "Open phase report"))
    workflow_root = record.get("workflow_root")
    if isinstance(workflow_root, str):
        candidates = [("workflow.json", "Phase evidence")]
        if name == "smoke":
            candidates = [
                ("reports/summary.json", "Metric summary"),
                ("exports/manifest.json", "ANDREA analysis archives"),
                ("state.json", "Pipeline evidence"),
            ]
        for filename, label in candidates:
            relative = (Path(workflow_root) / filename).as_posix()
            safe = path_below_no_symlinks(study_root / relative, study_root, "Phase link")
            if safe.is_file():
                links.append(_link(study_root, relative, label))
    failure = record.get("error", record.get("failure"))
    error = f'<p class="error">{_escape(failure)}</p>' if failure else ""
    return (
        '<article class="phase"><div class="phase-heading">'
        f'<h3>{_escape(title)}</h3><span class="status">{_escape(status)}</span></div>'
        f'<p>{_escape(purpose)}</p><p class="muted">{_escape(scope)}</p>'
        f'{error}<div class="links">{" · ".join(links)}</div></article>'
    )


def render_study_index(repo_root: Path, study_root: Path, state: dict[str, Any]) -> Path:
    """Publish a navigable snapshot without recomputing scientific results."""
    root = path_below_no_symlinks(study_root, repo_root / "work", "Study report directory")
    run_id = state.get("run_id", root.name)
    profile = str(state.get("profile", "unknown"))
    phases = state.get("phases", {})
    if not isinstance(phases, Mapping):
        raise BenchmarkConfigError("Study phases must be an object")
    names = ("smoke",) if profile == "smoke" else tuple(k for k in _PHASES if k != "smoke")
    cards = "".join(_phase_card(root, name, phases.get(name, {})) for name in names)
    warnings = "".join(f"<li>{_escape(value)}</li>" for value in state.get("warnings", []))
    warning_section = (
        f'<section class="warning"><h2>Continuation warnings</h2><ul>{warnings}</ul></section>'
        if warnings else ""
    )
    qualification_section = ""
    qualification = state.get("qualification")
    if isinstance(qualification, Mapping):
        link = ""
        if isinstance(qualification.get("report"), str):
            qualification_report = path_below_no_symlinks(
                Path(qualification["report"]), repo_root / "work", "Qualification report link",
                require_file=True,
            )
            href = quote(os.path.relpath(qualification_report, root / "reports"), safe="/")
            link = f'<p><a href="{href}">Open contract checks and metric oracle report</a></p>'
        qualification_section = (
            '<section><h2>Scientific contract qualification</h2>'
            f'<p>Status: {_escape(qualification.get("status", "unknown"))}. '
            'Real local SPATHI and ANDREA checks on a tiny known fixture. '
            'This does not certify all Docker workflows or confer statistical validity on a study.</p>'
            f'{link}</section>'
        )
    invocations = state.get("invocations", [])
    revisions: list[str] = []
    for invocation in invocations:
        observed = invocation.get("revisions", {})
        commits = []
        for repository in ("benchmark", "andrea", "spathi"):
            revision = observed.get(repository, {})
            commit = revision.get("commit", "unavailable") if isinstance(revision, Mapping) else revision
            commits.append(f"<td><code>{_escape(commit)}</code></td>")
        revisions.append(f'<tr><td>{_escape(invocation.get("started_at", ""))}</td>{"".join(commits)}</tr>')
    revision_rows = "".join(revisions) or '<tr><td colspan="4">No invocation recorded.</td></tr>'
    resources = state.get("resource_plan", state.get("resources", {}))
    resource_text = _escape(json.dumps(resources, indent=2, sort_keys=True, default=str))
    claim = (
        "Operational integration check. No ranking, calibration or statistical claim is authorized."
        if profile == "smoke" else
        "Descriptive development study. Point estimates and ranks do not establish a statistical winner."
        if profile == "reduced" else
        "Publication design. Confirmatory claims are restricted to the declared completed analyses; "
        "exploratory and computational results retain their separate scope."
    )
    html_text = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Study {_escape(run_id)} · SPATHI Benchmarks</title>
<style>
:root{{font-family:system-ui,sans-serif;color:#183045;background:#f3f6fa;line-height:1.55}}
body{{margin:0}}main{{max-width:1120px;margin:auto;padding:2rem 1.5rem 4rem}}
header{{background:#103b5a;color:white;padding:2rem 1.5rem}}header>div{{max-width:1120px;margin:auto}}
h1{{font-size:2.1rem;margin:.35rem 0}}h2{{font-size:1.25rem}}h3{{font-size:1.08rem;margin:0}}
.eyebrow{{font-size:.8rem;letter-spacing:.16em}}.meta{{display:flex;gap:1rem;flex-wrap:wrap}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:1rem}}
.phase,section{{background:white;border:1px solid #d9e3ec;border-radius:12px;padding:1.25rem}}
section{{margin:1.25rem 0}}.phase-heading{{display:flex;gap:1rem;justify-content:space-between;align-items:center}}
.status{{font-size:.8rem;border-radius:1rem;background:#e8f1f8;padding:.2rem .65rem}}
.muted{{color:#586b7b;font-size:.92rem}}a{{color:#126da8;text-underline-offset:3px}}
.links{{font-size:.92rem}}.warning{{border-color:#e3b14f;background:#fff8e7}}.error{{color:#a02323}}
table{{width:100%;border-collapse:collapse;text-align:left;font-size:.85rem}}th,td{{padding:.6rem;border-bottom:1px solid #e1e7ed}}
code{{overflow-wrap:anywhere}}pre{{white-space:pre-wrap;font-size:.85rem}}.scroll{{overflow-x:auto}}
footer{{font-size:.9rem;color:#586b7b;margin-top:1.5rem}}
</style></head><body><header><div><div class="eyebrow">SPATHI · BENCHMARKS</div>
<h1>Study {_escape(run_id)}</h1><div class="meta"><span>Profile: {_escape(profile)}</span>
<span>Status: {_escape(state.get("status", "unknown"))}</span>
<span>Updated: {_escape(state.get("updated_at", state.get("created_at", "")))}</span></div></div></header>
<main><section><h2>How to read this study</h2><p>{_escape(claim)}</p>
<p>Each phase report begins with the datasets actually used, their dimensions and roles.
Open a phase to inspect configurations, uncertainty, failures and canonical evidence.
Pending phases contribute no result to the study.</p>
<p>{_link(root, "study.json", "Study journal")} · Resume: <code>spathi-bench run --id {_escape(run_id)}</code></p></section>
{warning_section}{qualification_section}<h2>Phases and evidence</h2><div class="grid">{cards}</div>
<section><h2>Resources</h2><p>Saved machine capacity and operator ceilings. Individual task quotas and
measured usage are recorded in the phase reports; machine thread count is not observed CPU usage.</p>
<details><summary>Inspect resource plan</summary><pre>{resource_text}</pre></details></section>
<section><h2>Checkout revisions per invocation</h2><p>Earlier computations retain their original
provenance when a study continues after a revision change. A newer SPATHI checkout does not change
the study's locked container: each inference artifact records its actual tool image identity.</p><div class="scroll"><table>
<thead><tr><th>Invocation (UTC)</th><th>Benchmark commit</th><th>ANDREA commit</th><th>SPATHI commit</th></tr></thead>
<tbody>{revision_rows}</tbody></table></div></section>
<footer>This page links saved evidence and does not recalculate accuracy, rankings or decisions.
Keep the study directory together to preserve relative links.</footer></main></body></html>
'''
    return atomic_write_text(root / "reports/index.html", html_text, "Study entry report", boundary=root)


def rebuild_phase_reports(
    repo_root: Path, study_root: Path, state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Reconcile completed phase bundles using their existing integrity contracts."""
    bundles: dict[str, dict[str, Any]] = {}
    for name, record in state.get("phases", {}).items():
        if record.get("status") != "complete":
            continue
        scientific_receipt = {
            **record,
            "evidence": {
                relative: digest
                for relative, digest in record.get("evidence", {}).items()
                if not (
                    Path(relative).parts[:2] == ("v1", "reports")
                    and not (study_root / relative).exists()
                )
            },
        }
        validate_receipt(study_root, scientific_receipt)
        renderer = _RENDERERS.get(name)
        if renderer is None:
            if name == "smoke":
                continue
            raise BenchmarkConfigError(f"No phase report renderer for {name!r}")
        workflow_root = path_below_no_symlinks(
            study_root / record["workflow_root"], study_root, "Study phase workflow",
            require_directory=True,
        )
        bundles[name] = renderer(
            repo_root=repo_root, version_root=study_root / "v1", workflow_root=workflow_root,
        )
        validate_receipt(study_root, record)
    return bundles
