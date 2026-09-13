"""Project validated logical batch records onto unique scientific executions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .artifacts import ArtifactError, ArtifactKey, canonical_json_bytes
from .config import BenchmarkConfigError


def _execution_content(record: dict[str, Any]) -> dict[str, Any]:
    content = {key: value for key, value in record.items() if key != "batch_id"}
    for field in ("artifact", "inference_artifact"):
        if field == "inference_artifact" and field not in record:
            continue
        reference = record.get(field)
        if not isinstance(reference, dict) or any(
            not isinstance(reference.get(key), str) or not reference[key].strip()
            for key in ("kind", "key", "digest", "path")
        ):
            raise BenchmarkConfigError(f"Execution record has no valid {field} identity")
        try:
            artifact_key = ArtifactKey(kind=reference["kind"], digest=reference["digest"])
        except ArtifactError as exc:
            raise BenchmarkConfigError(
                f"Execution record has invalid {field} identity: {exc}"
            ) from exc
        if reference["key"] != artifact_key.value:
            raise BenchmarkConfigError(f"Execution record {field} key differs from its digest")
        if "cache_hit" in reference and not isinstance(reference["cache_hit"], bool):
            raise BenchmarkConfigError(f"Execution record {field}.cache_hit must be boolean")
        content[field] = {key: value for key, value in reference.items() if key != "cache_hit"}
    return content


def unique_execution_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain one record per dataset, regime and run, without changing input records.

    Callers first validate their logical batch records against the frozen study
    contracts. Repeated references are aliases only when their artifact identities
    and every scientific/attribution field agree. Batch membership and artifact
    cache observations do not distinguish a scientific execution. Unknown fields
    remain part of the comparison, so this projection cannot hide new conflicts.
    """

    unique: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        identity = (record["dataset_id"], record["regime"], record["run_id"])
        content = _execution_content(record)
        previous_content = seen.get(identity)
        if previous_content is None:
            seen[identity] = content
            unique.append(record)
            continue
        if canonical_json_bytes(previous_content) != canonical_json_bytes(content):
            differing = sorted(
                key
                for key in previous_content.keys() | content.keys()
                if key not in previous_content
                or key not in content
                or canonical_json_bytes(previous_content[key]) != canonical_json_bytes(content[key])
            )
            raise BenchmarkConfigError(
                f"Conflicting execution records for {identity!r}; differing fields: {differing}"
            )
    return unique
