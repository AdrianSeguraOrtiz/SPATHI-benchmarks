"""Canonical ANDREA custom-tool contract for the benchmarked SPATHI build."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def spathi_runtime_resources() -> dict[str, Any]:
    """Return a fresh SPATHI standardized runtime-resource declaration."""

    return {
        "threading": {
            "supported": True,
            "default_threads": 1,
            "max_threads": None,
            "upstream_mapping": "cli:--threads",
        }
    }


def oom_observed(result: Mapping[str, Any]) -> bool:
    """Return whether a physical result or any nested child reports an OOM event."""

    measurement = result.get("measurement")
    telemetry = measurement.get("telemetry") if isinstance(measurement, Mapping) else None
    if isinstance(telemetry, Mapping) and (
        telemetry.get("oom_killed") is True
        or (
            isinstance(telemetry.get("oom_events"), int)
            and not isinstance(telemetry.get("oom_events"), bool)
            and int(telemetry["oom_events"]) > 0
        )
    ):
        return True
    children = result.get("child_results")
    return isinstance(children, Mapping) and any(
        isinstance(child, Mapping) and oom_observed(child) for child in children.values()
    )


__all__ = ["oom_observed", "spathi_runtime_resources"]
