"""Verify the SPATHI model-sidecar contract independently of its installation.

Seeds use SPATHI's public scientific task identity: BLAKE2s-32 over canonical
JSON ``[run_seed, group, target]``, personalized with ``SPATHI-1``. Diagnostics
cover every requested group/target, including models skipped for numerical safety.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BenchmarkConfigError

MODEL_COLUMNS = (
    "target_group",
    "target",
    "status",
    "random_seed",
    "n_samples",
    "n_positive_weight_samples",
    "weight_sum",
    "n_predictors_input",
    "n_predictors_used",
    "discarded_predictors_json",
    "constant_predictors_json",
    "n_edges",
    "importance_sum",
    "fit_seconds",
    "n_estimators_fitted",
    "message",
)
TRAINED_STATUSES = frozenset({"trained", "trained_no_positive_importance"})
SKIPPED_STATUSES = frozenset(
    {
        "insufficient_positive_weight_samples",
        "constant_target",
        "no_predictors_after_self_exclusion",
        "no_variable_predictors",
    }
)


def model_seed(run_seed: int, group: str, target: str) -> int:
    """Reproduce SPATHI's deterministic, scheduling-independent model seed."""
    payload = json.dumps([run_seed, group, target], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=4, person=b"SPATHI-1")
    return int.from_bytes(digest.digest(), byteorder="little", signed=False)


def _integer(raw: str, label: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError(f"Model diagnostic {label} must be an integer") from exc
    if value < 0:
        raise BenchmarkConfigError(f"Model diagnostic {label} must be non-negative")
    return value


def _number(raw: str, label: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError(f"Model diagnostic {label} must be numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise BenchmarkConfigError(f"Model diagnostic {label} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class ModelAccounting:
    status_counts: dict[str, int]
    models_requested: int
    trees_fitted: int

    @property
    def trained(self) -> int:
        return sum(self.status_counts.get(status, 0) for status in TRAINED_STATUSES)

    def validate_metadata(self, models: Mapping[str, Any]) -> None:
        skipped = self.models_requested - self.trained
        expected = {
            "requested": self.models_requested,
            "completed": self.models_requested,
            "trained": self.trained,
            "preflight_skipped": skipped,
            "fit_or_importance_failures": 0,
            "skipped_target_records": skipped,
            "fitted_estimators_total": self.trees_fitted,
        }
        if any(
            type(models.get(key)) is not int or models[key] != value
            for key, value in expected.items()
        ):
            raise BenchmarkConfigError(
                "SPATHI accounting counters differ from complete diagnostics"
            )


def read_model_accounting(
    path: Path,
    *,
    contexts: Sequence[str],
    targets: Sequence[str],
    parameters: Mapping[str, Any],
    algorithm_seed: int,
) -> ModelAccounting:
    """Check exact model coverage, deterministic seeds and fixed tree counts."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != MODEL_COLUMNS:
                raise BenchmarkConfigError(f"Model diagnostics have an unexpected header: {path}")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise BenchmarkConfigError(f"Cannot read model diagnostics {path}: {exc}") from exc
    expected_keys = {
        (context.removeprefix("group:"), target) for context in contexts for target in targets
    }
    observed_keys = {(row["target_group"], row["target"]) for row in rows}
    if observed_keys != expected_keys or len(rows) != len(expected_keys):
        raise BenchmarkConfigError("Model diagnostics do not cover the exact group-target product")
    counts: Counter[str] = Counter()
    trees = 0
    for row in rows:
        group, target, status = row["target_group"], row["target"], row["status"]
        label = f"{group}/{target}"
        if status not in TRAINED_STATUSES | SKIPPED_STATUSES:
            raise BenchmarkConfigError(
                f"Model diagnostic reports invalid/failed status for {label}"
            )
        if _integer(row["random_seed"], f"seed[{label}]") != model_seed(
            algorithm_seed, group, target
        ):
            raise BenchmarkConfigError(f"Model diagnostic seed differs for {label}")
        fitted = _integer(row["n_estimators_fitted"], f"trees[{label}]")
        required = parameters["n_estimators"]
        if type(required) is not int or required < 1:
            raise BenchmarkConfigError("Requested tree count must be a positive integer")
        if status in TRAINED_STATUSES:
            if fitted != required:
                raise BenchmarkConfigError(
                    f"Fixed-tree model has an invalid tree budget for {label}"
                )
        elif fitted:
            raise BenchmarkConfigError(f"Untrained model reports fitted trees for {label}")
        _number(row["fit_seconds"], f"fit_seconds[{label}]")
        counts[status] += 1
        trees += fitted
    return ModelAccounting(
        status_counts=dict(counts),
        models_requested=len(rows),
        trees_fitted=trees,
    )
