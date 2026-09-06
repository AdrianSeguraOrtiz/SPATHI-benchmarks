"""Per-configuration inference regulator-scope contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config import BenchmarkConfigError

INFERENCE_REGULATOR_SCOPES = frozenset({"tf_restricted", "all_gene"})


def inference_regulator_scopes(
    comparison: Mapping[str, Any],
) -> dict[str, str]:
    """Return the complete regulator scope exposed to each inference configuration."""

    comparison_id = str(comparison.get("id", "<unknown>"))
    raw_configurations = comparison.get("configuration_ids")
    if (
        not isinstance(raw_configurations, Sequence)
        or isinstance(raw_configurations, (str, bytes))
        or not raw_configurations
        or not all(isinstance(item, str) and item for item in raw_configurations)
    ):
        raise BenchmarkConfigError(f"Comparison {comparison_id!r} has invalid configuration_ids")
    configurations = tuple(raw_configurations)
    if len(set(configurations)) != len(configurations):
        raise BenchmarkConfigError(f"Comparison {comparison_id!r} has duplicate configuration_ids")

    raw_contract = comparison.get("inference_regulator_scope_by_configuration")
    if not isinstance(raw_contract, Mapping):
        raise BenchmarkConfigError(
            f"Comparison {comparison_id!r} has no per-configuration inference regulator "
            "scope contract"
        )
    observed_keys = set(raw_contract)
    expected_keys = set(configurations)
    if observed_keys != expected_keys:
        raise BenchmarkConfigError(
            f"Comparison {comparison_id!r} inference-scope keys differ from "
            f"configuration_ids: missing={sorted(expected_keys - observed_keys)}, "
            f"unknown={sorted(observed_keys - expected_keys)}"
        )

    normalized: dict[str, str] = {}
    for configuration_id in configurations:
        scope = raw_contract[configuration_id]
        if not isinstance(scope, str) or scope not in INFERENCE_REGULATOR_SCOPES:
            raise BenchmarkConfigError(
                f"Comparison {comparison_id!r} configuration {configuration_id!r} has "
                f"invalid inference regulator scope {scope!r}"
            )
        normalized[configuration_id] = scope
    return normalized
