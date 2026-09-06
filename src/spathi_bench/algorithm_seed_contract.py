"""Exact per-configuration algorithm-seed contracts for benchmark comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config import BenchmarkConfigError

MAX_ALGORITHM_SEED = 2_147_483_646


def algorithm_seeds_by_configuration(
    comparison: Mapping[str, Any],
) -> dict[str, tuple[int | None, ...]]:
    """Return a validated, complete seed panel for every compared configuration."""

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

    raw_contract = comparison.get("algorithm_seeds_by_configuration")
    if not isinstance(raw_contract, Mapping):
        raise BenchmarkConfigError(
            f"Comparison {comparison_id!r} has no per-configuration algorithm-seed contract"
        )
    observed_keys = set(raw_contract)
    expected_keys = set(configurations)
    if observed_keys != expected_keys:
        raise BenchmarkConfigError(
            f"Comparison {comparison_id!r} algorithm-seed keys differ from configuration_ids: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"unknown={sorted(observed_keys - expected_keys)}"
        )

    normalized: dict[str, tuple[int | None, ...]] = {}
    for configuration_id in configurations:
        raw_seeds = raw_contract[configuration_id]
        if (
            not isinstance(raw_seeds, Sequence)
            or isinstance(raw_seeds, (str, bytes))
            or not raw_seeds
        ):
            raise BenchmarkConfigError(
                f"Comparison {comparison_id!r} configuration {configuration_id!r} "
                "requires a non-empty algorithm-seed panel"
            )
        seeds = tuple(raw_seeds)
        if seeds == (None,):
            normalized[configuration_id] = seeds
            continue
        if any(seed is None for seed in seeds):
            raise BenchmarkConfigError(
                f"Comparison {comparison_id!r} configuration {configuration_id!r} cannot "
                "mix null with caller-controlled algorithm seeds"
            )
        if any(
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed < 1
            or seed > MAX_ALGORITHM_SEED
            for seed in seeds
        ):
            raise BenchmarkConfigError(
                f"Comparison {comparison_id!r} configuration {configuration_id!r} has an "
                "invalid algorithm seed"
            )
        if len(set(seeds)) != len(seeds):
            raise BenchmarkConfigError(
                f"Comparison {comparison_id!r} configuration {configuration_id!r} has "
                "duplicate algorithm seeds"
            )
        normalized[configuration_id] = seeds
    return normalized
