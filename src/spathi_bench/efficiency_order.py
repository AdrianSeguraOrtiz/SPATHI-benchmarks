"""Deterministic counterbalancing for paired efficiency measurements."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from .artifacts import canonical_json_bytes
from .seeds import MAX_SEED


def counterbalanced_execution_orders(
    dataset_ids: Iterable[str],
    algorithm_seeds: Iterable[int],
    labels: Sequence[str],
    *,
    execution_order_seed: int,
) -> dict[tuple[str, int], list[str]]:
    """Assign reproducibly seeded rotations with near-equal positional balance."""

    ordered_labels = list(labels)
    if not ordered_labels or len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError("Counterbalancing requires at least one unique label")
    datasets = tuple(dataset_ids)
    seeds = tuple(algorithm_seeds)
    blocks = [(dataset_id, seed) for dataset_id in datasets for seed in seeds]
    if not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("Counterbalancing requires unique dataset-by-seed blocks")
    if (
        not isinstance(execution_order_seed, int)
        or isinstance(execution_order_seed, bool)
        or not 1 <= execution_order_seed <= MAX_SEED
    ):
        raise ValueError(f"execution_order_seed must be between 1 and {MAX_SEED}")
    ranked = sorted(
        blocks,
        key=lambda block: hashlib.sha256(
            canonical_json_bytes(
                {
                    "algorithm_seed": block[1],
                    "dataset_id": block[0],
                    "execution_order_seed": execution_order_seed,
                }
            )
        ).digest(),
    )
    orders: dict[tuple[str, int], list[str]] = {}
    for index, block in enumerate(ranked):
        offset = index % len(ordered_labels)
        orders[block] = ordered_labels[offset:] + ordered_labels[:offset]
    return orders


def runner_implementation_sha256() -> str:
    """Fingerprint the efficiency runner and counterbalancing implementation."""

    digest = hashlib.sha256()
    for path in (Path(__file__).with_name("noninferiority_run.py"), Path(__file__)):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["counterbalanced_execution_orders", "runner_implementation_sha256"]
