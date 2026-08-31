"""Build a joint ANDREA analysis view from independent inference runs.

The benchmark executes tools independently so their results can be cached.  This
module projects those immutable results back into the three-file ``analysis``
handoff expected by ANDREA without pretending that a monolithic inference run
took place.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import BenchmarkConfigError

MERGED_NETWORK_COLUMNS = (
    "source",
    "target",
    "score",
    "sign",
    "evidence",
    "context",
    "tool_id",
)
_OUTPUT_CAPABILITY_KEYS = {"tool_origin", "catalog_tool_id", "directed", "sign"}
_OUTPUT_SIGNS = {"none", "signed", "mixed"}
_TOOL_ORIGINS = {"catalog", "custom"}
_EXECUTION_MODES = {
    "global",
    "group_native",
    "group_emulated",
    "column_native",
    "group_aggregated",
}
_PORTABLE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")

ContractValidator = Callable[[Path], None]


class IntegrationError(BenchmarkConfigError):
    """Raised when independent inference results cannot form one joint view."""


@dataclass(frozen=True, slots=True)
class _NetworkFile:
    path: Path
    content: bytes
    header: tuple[str, ...]
    header_bytes: bytes
    body_bytes: bytes
    rows_per_tool: dict[str, int]
    contexts_per_tool: dict[str, set[str]]
    signs_per_tool: dict[str, set[str]]
    sha256: str


@dataclass(frozen=True, slots=True)
class _SourceRun:
    report_path: Path
    report: dict[str, Any]
    source_run_id: str
    selected: tuple[str, ...]
    raw: _NetworkFile
    normalized: _NetworkFile
    report_sha256: str


def integrate_inference_runs(
    *,
    run_report_paths: Sequence[Path],
    output_dir: Path,
    combined_run_id: str,
    regime: str,
    contract_validator: ContractValidator | None = None,
) -> Path:
    """Create a deterministic joint infer-network ``analysis`` directory.

    ``contract_validator`` can be the lazily imported
    ``andrea.core.commands.evaluate_inference.validate_inference_analysis_inputs``.
    It is optional because the benchmark's orchestration environment and the
    locked ANDREA runtime may use different Python interpreters.

    The destination must not exist.  Publication is atomic, so neither source
    validation failures nor a failing external contract validator leave a
    partial analysis directory behind.
    """

    _validate_identifier(combined_run_id, "combined_run_id")
    _validate_identifier(regime, "regime")
    if isinstance(run_report_paths, (str, bytes)) or not run_report_paths:
        raise IntegrationError("run_report_paths must be a non-empty sequence")

    sources = [_load_source(Path(path)) for path in run_report_paths]
    _validate_sources(sources)
    sources.sort(key=lambda source: source.source_run_id)

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise IntegrationError(f"Joint analysis destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        raw_path = staging / "merged_network_raw.csv"
        normalized_path = staging / "merged_network_normalized.csv"
        _write_merged_network(raw_path, [source.raw for source in sources])
        _write_merged_network(normalized_path, [source.normalized for source in sources])

        report = _build_joint_report(
            sources=sources,
            combined_run_id=combined_run_id,
            regime=regime,
        )
        report_path = staging / "run_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        if contract_validator is not None:
            try:
                contract_validator(report_path)
            except Exception as exc:
                raise IntegrationError(
                    f"ANDREA rejected the joint inference analysis: {exc}"
                ) from exc
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "run_report.json"


def lazy_andrea_contract_validator(run_report_path: Path) -> None:
    """Validate with ANDREA when its locked Python package is importable.

    Importing is deliberately deferred so importing :mod:`spathi_bench.integrate`
    never couples the benchmark's own virtual environment to ANDREA's runtime.
    """

    try:
        from andrea.core.commands.evaluate_inference import (  # type: ignore[import-not-found]
            validate_inference_analysis_inputs,
        )
    except ImportError as exc:
        raise IntegrationError(
            "ANDREA is not importable in this Python interpreter; use its locked "
            "runtime or pass another contract_validator"
        ) from exc
    validate_inference_analysis_inputs(run_report_path=run_report_path)


def _load_source(report_path: Path) -> _SourceRun:
    resolved = report_path.resolve()
    try:
        content = resolved.read_bytes()
        report = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrationError(f"Cannot read inference run report {resolved}: {exc}") from exc
    if not isinstance(report, dict):
        raise IntegrationError(f"Inference run report must be an object: {resolved}")

    source_run_id = _validate_identifier(report.get("run_id"), f"{resolved}: run_id")
    if report.get("status") != "executed":
        raise IntegrationError(f"{resolved}: status must be exactly 'executed'")
    if not isinstance(report.get("execution"), dict):
        raise IntegrationError(f"{resolved}: execution must be an object")
    if not isinstance(report.get("issues"), list):
        raise IntegrationError(f"{resolved}: issues must be an array")

    selected = _validate_tools(report, resolved)
    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise IntegrationError(f"{resolved}: outputs must be an object")
    raw_path = _resolve_report_member(
        resolved.parent,
        outputs.get("merged_network_raw"),
        f"{resolved}: outputs.merged_network_raw",
    )
    normalized_path = _resolve_report_member(
        resolved.parent,
        outputs.get("merged_network_normalized"),
        f"{resolved}: outputs.merged_network_normalized",
    )
    raw = _load_network_file(raw_path, normalized=False)
    normalized = _load_network_file(normalized_path, normalized=True)
    if raw.header != normalized.header:
        raise IntegrationError(f"{resolved}: raw and normalized merged-network schemas differ")

    expected_counts = outputs.get("rows_per_tool")
    selected_set = set(selected)
    if not isinstance(expected_counts, dict) or set(expected_counts) != selected_set:
        raise IntegrationError(
            f"{resolved}: outputs.rows_per_tool must have exactly the selected run IDs"
        )
    for run_id, count in expected_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise IntegrationError(
                f"{resolved}: outputs.rows_per_tool[{run_id!r}] must be a non-negative integer"
            )
    observed_raw = {run_id: raw.rows_per_tool.get(run_id, 0) for run_id in selected}
    observed_normalized = {run_id: normalized.rows_per_tool.get(run_id, 0) for run_id in selected}
    unexpected = (set(raw.rows_per_tool) | set(normalized.rows_per_tool)) - selected_set
    if unexpected:
        raise IntegrationError(
            f"{resolved}: merged networks contain undeclared run IDs: {sorted(unexpected)}"
        )
    if observed_raw != expected_counts:
        raise IntegrationError(
            f"{resolved}: raw merged-network row counts do not match outputs.rows_per_tool"
        )
    if observed_normalized != expected_counts:
        raise IntegrationError(
            f"{resolved}: normalized merged-network row counts do not match outputs.rows_per_tool"
        )

    _validate_contexts_and_signs(
        report=report,
        report_path=resolved,
        selected=selected,
        raw=raw,
        normalized=normalized,
    )
    return _SourceRun(
        report_path=resolved,
        report=report,
        source_run_id=source_run_id,
        selected=selected,
        raw=raw,
        normalized=normalized,
        report_sha256=hashlib.sha256(content).hexdigest(),
    )


def _validate_tools(report: dict[str, Any], report_path: Path) -> tuple[str, ...]:
    tools = report.get("tools")
    if not isinstance(tools, dict):
        raise IntegrationError(f"{report_path}: tools must be an object")
    selected_value = tools.get("selected")
    if not isinstance(selected_value, list) or not selected_value:
        raise IntegrationError(f"{report_path}: tools.selected must be a non-empty array")
    selected = tuple(
        _validate_identifier(value, f"{report_path}: tools.selected") for value in selected_value
    )
    if len(set(selected)) != len(selected):
        raise IntegrationError(f"{report_path}: tools.selected contains duplicate run IDs")
    selected_set = set(selected)

    catalog_ids = tools.get("catalog_tool_ids")
    origins = tools.get("tool_origins")
    capabilities = tools.get("output_capabilities")
    for field, mapping in (
        ("catalog_tool_ids", catalog_ids),
        ("tool_origins", origins),
        ("output_capabilities", capabilities),
    ):
        if not isinstance(mapping, dict) or set(mapping) != selected_set:
            raise IntegrationError(
                f"{report_path}: tools.{field} must have exactly the selected run IDs"
            )
    assert isinstance(catalog_ids, dict)
    assert isinstance(origins, dict)
    assert isinstance(capabilities, dict)
    for run_id in selected:
        catalog_id = _validate_identifier(
            catalog_ids[run_id],
            f"{report_path}: tools.catalog_tool_ids[{run_id!r}]",
        )
        origin = origins[run_id]
        if origin not in _TOOL_ORIGINS:
            raise IntegrationError(f"{report_path}: tools.tool_origins[{run_id!r}] is invalid")
        if origin == "custom" and catalog_id != f"custom_{run_id}":
            raise IntegrationError(
                f"{report_path}: custom run {run_id!r} must use catalog ID 'custom_{run_id}'"
            )
        capability = capabilities[run_id]
        if not isinstance(capability, dict) or set(capability) != _OUTPUT_CAPABILITY_KEYS:
            raise IntegrationError(
                f"{report_path}: output capability for {run_id!r} has the wrong schema"
            )
        if capability["tool_origin"] != origin or capability["catalog_tool_id"] != catalog_id:
            raise IntegrationError(
                f"{report_path}: output capability identity for {run_id!r} is inconsistent"
            )
        if not isinstance(capability["directed"], bool):
            raise IntegrationError(
                f"{report_path}: output capability directed for {run_id!r} must be boolean"
            )
        if capability["sign"] not in _OUTPUT_SIGNS:
            raise IntegrationError(
                f"{report_path}: output capability sign for {run_id!r} is invalid"
            )

    completed = tools.get("completed")
    if completed != list(selected):
        raise IntegrationError(
            f"{report_path}: every selected run must be completed, in selected order"
        )
    failed = tools.get("failed")
    if failed not in ({}, None):
        raise IntegrationError(f"{report_path}: failed tools cannot enter a joint view")
    status_by_tool = tools.get("status_by_tool")
    if not isinstance(status_by_tool, dict) or set(status_by_tool) != selected_set:
        raise IntegrationError(
            f"{report_path}: tools.status_by_tool must have exactly the selected run IDs"
        )
    for run_id in selected:
        if status_by_tool[run_id] not in {"completed", "completed_with_warnings"}:
            raise IntegrationError(f"{report_path}: run {run_id!r} did not complete")
    return selected


def _validate_contexts_and_signs(
    *,
    report: dict[str, Any],
    report_path: Path,
    selected: tuple[str, ...],
    raw: _NetworkFile,
    normalized: _NetworkFile,
) -> None:
    tools = report["tools"]
    contexts = tools.get("completed_contexts")
    results = tools.get("results")
    selected_set = set(selected)
    if not isinstance(contexts, dict) or set(contexts) != selected_set:
        raise IntegrationError(
            f"{report_path}: tools.completed_contexts must have exactly the selected run IDs"
        )
    if not isinstance(results, dict):
        raise IntegrationError(f"{report_path}: tools.results must be an object")
    capabilities = tools["output_capabilities"]
    for run_id in selected:
        result = results.get(run_id)
        execution = result.get("execution") if isinstance(result, dict) else None
        mode = execution.get("mode") if isinstance(execution, dict) else None
        if mode not in _EXECUTION_MODES:
            raise IntegrationError(
                f"{report_path}: completed run {run_id!r} has no valid execution mode"
            )
        inventory = contexts[run_id]
        if (
            not isinstance(inventory, list)
            or not inventory
            or not all(
                isinstance(context, str) and context and context == context.strip()
                for context in inventory
            )
            or len(set(inventory)) != len(inventory)
        ):
            raise IntegrationError(f"{report_path}: completed contexts for {run_id!r} are invalid")
        for context in inventory:
            if not _context_matches_mode(context, mode):
                raise IntegrationError(
                    f"{report_path}: context {context!r} contradicts mode {mode!r}"
                )
        observed_contexts = raw.contexts_per_tool.get(
            run_id, set()
        ) | normalized.contexts_per_tool.get(run_id, set())
        unexpected_contexts = observed_contexts - set(inventory)
        if unexpected_contexts:
            raise IntegrationError(
                f"{report_path}: run {run_id!r} emitted undeclared contexts "
                f"{sorted(unexpected_contexts)}"
            )
        declared_sign = capabilities[run_id]["sign"]
        observed_signs = raw.signs_per_tool.get(run_id, set()) | normalized.signs_per_tool.get(
            run_id, set()
        )
        if declared_sign == "none" and observed_signs - {"?"}:
            raise IntegrationError(f"{report_path}: unsigned run {run_id!r} emitted signed edges")
        if declared_sign == "signed" and observed_signs - {"+", "-"}:
            raise IntegrationError(f"{report_path}: signed run {run_id!r} emitted unsigned edges")


def _load_network_file(path: Path, *, normalized: bool) -> _NetworkFile:
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IntegrationError(f"Cannot read merged network {path}: {exc}") from exc
    newline_index = content.find(b"\n")
    if newline_index < 0:
        raise IntegrationError(f"Merged network must contain a newline-terminated header: {path}")
    header_bytes = content[: newline_index + 1]
    body_bytes = content[newline_index + 1 :]
    if body_bytes and not content.endswith((b"\n", b"\r")):
        raise IntegrationError(f"Merged network rows must be newline terminated: {path}")

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = tuple(next(reader))
    except (StopIteration, csv.Error) as exc:
        raise IntegrationError(f"Merged network has no valid header: {path}") from exc
    if header != MERGED_NETWORK_COLUMNS:
        raise IntegrationError(
            f"Merged network schema must be exactly {list(MERGED_NETWORK_COLUMNS)}: {path}"
        )

    counts: Counter[str] = Counter()
    contexts: dict[str, set[str]] = {}
    signs: dict[str, set[str]] = {}
    try:
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(MERGED_NETWORK_COLUMNS):
                raise IntegrationError(
                    f"Merged network row {line_number} has {len(row)} columns, "
                    f"expected {len(MERGED_NETWORK_COLUMNS)}: {path}"
                )
            values = dict(zip(MERGED_NETWORK_COLUMNS, row, strict=True))
            source = values["source"].strip()
            target = values["target"].strip()
            evidence = values["evidence"].strip()
            context = values["context"].strip()
            tool_id = values["tool_id"].strip()
            sign = values["sign"].strip()
            if not source or not target or not evidence or not context or not tool_id:
                raise IntegrationError(
                    f"Merged network row {line_number} has an empty required value: {path}"
                )
            if sign not in {"+", "-", "?"}:
                raise IntegrationError(
                    f"Merged network row {line_number} has invalid sign {sign!r}: {path}"
                )
            try:
                score = float(values["score"])
            except (TypeError, ValueError) as exc:
                raise IntegrationError(
                    f"Merged network row {line_number} has invalid score: {path}"
                ) from exc
            valid_score = (
                math.isfinite(score) and 0.0 <= score <= 1.0
                if normalized
                else math.isfinite(score) and score > 0.0
            )
            if not valid_score:
                expectation = "in [0, 1]" if normalized else "a positive finite magnitude"
                raise IntegrationError(
                    f"Merged network row {line_number} score must be {expectation}: {path}"
                )
            counts[tool_id] += 1
            contexts.setdefault(tool_id, set()).add(context)
            signs.setdefault(tool_id, set()).add(sign)
    except csv.Error as exc:
        raise IntegrationError(f"Cannot parse merged network {path}: {exc}") from exc
    return _NetworkFile(
        path=path,
        content=content,
        header=header,
        header_bytes=header_bytes,
        body_bytes=body_bytes,
        rows_per_tool=dict(counts),
        contexts_per_tool=contexts,
        signs_per_tool=signs,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _validate_sources(sources: list[_SourceRun]) -> None:
    source_ids = [source.source_run_id for source in sources]
    if len(set(source_ids)) != len(source_ids):
        raise IntegrationError("Source run_report.json files must have unique run_id values")
    all_selected = [run_id for source in sources for run_id in source.selected]
    duplicates = sorted(run_id for run_id, count in Counter(all_selected).items() if count > 1)
    if duplicates:
        raise IntegrationError(f"Tool run IDs must be unique across sources: {duplicates}")

    first_dataset = sources[0].report.get("dataset")
    if not isinstance(first_dataset, dict):
        raise IntegrationError(f"{sources[0].report_path}: dataset must be an object")
    dataset_id = first_dataset.get("id")
    fingerprint = _validate_fingerprint(
        first_dataset.get("fingerprint"),
        f"{sources[0].report_path}: dataset.fingerprint",
    )
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise IntegrationError(f"{sources[0].report_path}: dataset.id must be non-empty")
    for source in sources[1:]:
        dataset = source.report.get("dataset")
        if not isinstance(dataset, dict):
            raise IntegrationError(f"{source.report_path}: dataset must be an object")
        other_fingerprint = _validate_fingerprint(
            dataset.get("fingerprint"),
            f"{source.report_path}: dataset.fingerprint",
        )
        if dataset.get("id") != dataset_id or other_fingerprint != fingerprint:
            raise IntegrationError(
                "All joint inference sources must have the same dataset ID and fingerprint"
            )

    expected_header = sources[0].raw.header
    for source in sources:
        if source.raw.header != expected_header or source.normalized.header != expected_header:
            raise IntegrationError("All merged-network files must have identical schemas")


def _build_joint_report(
    *,
    sources: list[_SourceRun],
    combined_run_id: str,
    regime: str,
) -> dict[str, Any]:
    selected = sorted(run_id for source in sources for run_id in source.selected)
    tools: dict[str, Any] = {
        "selected": selected,
        "catalog_tool_ids": {},
        "tool_origins": {},
        "output_capabilities": {},
        "skipped": {},
        "status_by_tool": {},
        "completed": selected,
        "completed_contexts": {},
        "failed": {},
        "results": {},
    }
    rows_per_tool: dict[str, int] = {}
    integration_sources: list[dict[str, Any]] = []
    combined_issues: list[dict[str, Any]] = []

    by_run_id = {run_id: source for source in sources for run_id in source.selected}
    for run_id in selected:
        source = by_run_id[run_id]
        source_tools = source.report["tools"]
        tools["catalog_tool_ids"][run_id] = source_tools["catalog_tool_ids"][run_id]
        tools["tool_origins"][run_id] = source_tools["tool_origins"][run_id]
        tools["output_capabilities"][run_id] = deepcopy(source_tools["output_capabilities"][run_id])
        tools["status_by_tool"][run_id] = source_tools["status_by_tool"][run_id]
        tools["completed_contexts"][run_id] = deepcopy(source_tools["completed_contexts"][run_id])
        source_result = source_tools["results"][run_id]
        tools["results"][run_id] = {
            "tool_id": run_id,
            "status": source_tools["status_by_tool"][run_id],
            "execution": deepcopy(source_result["execution"]),
            "integration_source": {
                "source_run_id": source.source_run_id,
                "run_report_sha256": source.report_sha256,
            },
        }
        rows_per_tool[run_id] = source.raw.rows_per_tool.get(run_id, 0)

    for source in sources:
        integration_sources.append(
            {
                "source_run_id": source.source_run_id,
                "tool_run_ids": list(source.selected),
                "dataset_id": source.report["dataset"]["id"],
                "files": {
                    "run_report.json": {"sha256": source.report_sha256},
                    "merged_network_raw.csv": {"sha256": source.raw.sha256},
                    "merged_network_normalized.csv": {"sha256": source.normalized.sha256},
                },
            }
        )
        for issue in source.report["issues"]:
            if isinstance(issue, dict):
                integrated_issue = deepcopy(issue)
                integrated_issue["integration_source_run_id"] = source.source_run_id
                combined_issues.append(integrated_issue)

    return {
        "run_id": combined_run_id,
        "status": "executed",
        "dataset": deepcopy(sources[0].report["dataset"]),
        "tools": tools,
        "outputs": {
            "merged_network_raw": "merged_network_raw.csv",
            "merged_network_normalized": "merged_network_normalized.csv",
            "rows_per_tool": rows_per_tool,
        },
        "issues": combined_issues,
        "execution": {
            "kind": "integrated_analysis_view",
            "producer": "spathi-benchmarks",
            "regime": regime,
            "source_reports_total": len(sources),
            "tools_selected": len(selected),
            "tools_completed": len(selected),
            "tools_failed": 0,
        },
        "integration": {
            "schema_version": "1.0",
            "kind": "joint-inference-view",
            "producer": "spathi-benchmarks",
            "regime": regime,
            "sources": integration_sources,
            "ordering": {
                "sources": "source_run_id ascending",
                "rows": "source order; source row bytes preserved",
            },
            "monolithic_execution": False,
        },
        "notes": [
            "This is a SPATHI-benchmarks integration of independently executed inference runs.",
            "Source identities and content hashes are recorded under integration.sources.",
        ],
    }


def _write_merged_network(destination: Path, files: list[_NetworkFile]) -> None:
    with destination.open("wb") as handle:
        handle.write(files[0].header_bytes)
        for source in files:
            handle.write(source.body_bytes)


def _resolve_report_member(root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip():
        raise IntegrationError(f"{label} must be a non-empty relative POSIX path")
    if "\\" in raw_path or "//" in raw_path:
        raise IntegrationError(f"{label} must be a canonical relative POSIX path")
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or str(relative) != raw_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise IntegrationError(f"{label} must be a canonical relative POSIX path")
    candidate = root.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise IntegrationError(f"{label} does not resolve to a file inside its run directory")
    return candidate


def _validate_identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _PORTABLE_IDENTIFIER.fullmatch(value) is None
    ):
        raise IntegrationError(f"{label} must match [A-Za-z0-9][A-Za-z0-9._-]* exactly")
    return value


def _validate_fingerprint(value: Any, label: str) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or set(value) != {"algorithm", "value"}
        or value.get("algorithm") != "sha256"
        or not isinstance(value.get("value"), str)
        or _SHA256.fullmatch(value["value"]) is None
    ):
        raise IntegrationError(
            f"{label} must contain exactly sha256 and a lowercase 64-character digest"
        )
    return {"algorithm": "sha256", "value": value["value"]}


def _context_matches_mode(context: str, mode: str) -> bool:
    if mode == "global":
        return context == "global"
    if mode in {"group_native", "group_emulated", "group_aggregated"}:
        return context.startswith("group:") and context != "group:"
    if mode == "column_native":
        return context.startswith("column:") and context != "column:"
    return False


__all__ = [
    "ContractValidator",
    "IntegrationError",
    "integrate_inference_runs",
    "lazy_andrea_contract_validator",
]
