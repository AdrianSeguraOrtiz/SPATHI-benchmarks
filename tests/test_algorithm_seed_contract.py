from __future__ import annotations

import pytest

from spathi_bench.algorithm_seed_contract import algorithm_seeds_by_configuration
from spathi_bench.config import BenchmarkConfigError


def _comparison() -> dict[str, object]:
    return {
        "id": "all-gene-topology",
        "configuration_ids": ["stochastic", "deterministic"],
        "algorithm_seeds_by_configuration": {
            "stochastic": [11, 13, 17],
            "deterministic": [None],
        },
    }


def test_seed_contract_accepts_seeded_and_single_deterministic_executions() -> None:
    assert algorithm_seeds_by_configuration(_comparison()) == {
        "stochastic": (11, 13, 17),
        "deterministic": (None,),
    }


def test_seed_contract_requires_exact_configuration_identity() -> None:
    comparison = _comparison()
    comparison["algorithm_seeds_by_configuration"] = {"stochastic": [11, 13, 17]}

    with pytest.raises(BenchmarkConfigError, match="keys differ from configuration_ids"):
        algorithm_seeds_by_configuration(comparison)


@pytest.mark.parametrize("seeds", [[None, 11], [None, None], [0], [True]])
def test_seed_contract_rejects_ambiguous_or_invalid_panels(seeds: list[object]) -> None:
    comparison = _comparison()
    comparison["algorithm_seeds_by_configuration"] = {
        "stochastic": seeds,
        "deterministic": [None],
    }

    with pytest.raises(BenchmarkConfigError):
        algorithm_seeds_by_configuration(comparison)
