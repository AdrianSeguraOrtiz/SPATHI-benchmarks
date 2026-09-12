"""Verify the SPATHI model-sidecar contract independently of its installation.

Seeds use SPATHI's public scientific task identity: BLAKE2s-32 over canonical
JSON ``[run_seed, group, target]``, personalized with ``SPATHI-1``. Diagnostics
cover every requested group/target, including globally excluded targets.
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
    "target_detected_cells",
    "target_detected_fraction",
    "target_weighted_detected_fraction",
    "target_weighted_detected_ess",
    "n_estimators_fitted",
    "adaptive_converged",
    "convergence_delta",
    "convergence_checks",
    "message",
)
TRAINED_STATUSES = frozenset({"trained", "trained_no_positive_importance"})
SKIPPED_STATUSES = frozenset(
    {
        "insufficient_positive_weight_samples",
        "constant_target",
        "no_predictors_after_self_exclusion",
        "no_variable_predictors",
        "target_not_estimable",
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
    eligible_targets: dict[str, tuple[str, ...]]
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
            "target_not_estimable": self.status_counts.get("target_not_estimable", 0),
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
    globally_eligible: Sequence[str],
    parameters: Mapping[str, Any],
    algorithm_seed: int,
) -> ModelAccounting:
    """Check completion, seeds, eligibility precedence and actual tree budgets."""
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
    automatic = parameters["target_eligibility"] == "automatic"
    global_targets = set(globally_eligible)
    if not global_targets <= set(targets) or (not automatic and global_targets != set(targets)):
        raise BenchmarkConfigError(
            "Global target eligibility contradicts the requested target space"
        )
    eligible: dict[str, list[str]] = {context: [] for context in contexts}
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
        maximum = int(parameters["n_estimators"])
        adaptive = parameters["adaptive_trees"]
        raw_converged = row["adaptive_converged"].lower()
        if raw_converged not in {"true", "false"}:
            raise BenchmarkConfigError(f"Model adaptive_converged is not a boolean for {label}")
        converged = raw_converged == "true"
        if status in TRAINED_STATUSES:
            if not 0 < fitted <= maximum:
                raise BenchmarkConfigError(f"Trained model has an invalid tree budget for {label}")
            if adaptive:
                step = int(parameters["adaptive_tree_step"])
                minimum = int(parameters["adaptive_min_estimators"])
                if fitted != maximum and (fitted % step or fitted < minimum or not converged):
                    raise BenchmarkConfigError(f"Adaptive tree count is off schedule for {label}")
                if converged:
                    delta = _number(row["convergence_delta"], f"convergence_delta[{label}]")
                    if (
                        fitted < minimum
                        or delta > float(parameters["adaptive_tolerance"])
                        or fitted <= step * int(parameters["adaptive_patience"])
                    ):
                        raise BenchmarkConfigError(
                            f"Adaptive convergence is inconsistent for {label}"
                        )
            elif fitted != maximum or converged:
                raise BenchmarkConfigError(
                    f"Fixed-tree model has an invalid tree budget for {label}"
                )
        elif fitted or converged:
            raise BenchmarkConfigError(
                f"Untrained model reports fitted trees/convergence for {label}"
            )

        estimable = target in global_targets
        if automatic and estimable:
            # SPATHI checks sample support before group-specific detection.
            positive = _integer(row["n_positive_weight_samples"], f"positive_samples[{label}]")
            if positive < 2:
                if status != "insufficient_positive_weight_samples":
                    raise BenchmarkConfigError(
                        f"Model status contradicts sample support for {label}"
                    )
            else:
                fraction = _number(row["target_weighted_detected_fraction"], f"fraction[{label}]")
                ess = _number(row["target_weighted_detected_ess"], f"ESS[{label}]")
                if fraction > 1 + 1e-12 or ess > positive + 1e-9:
                    raise BenchmarkConfigError(
                        f"Weighted detection is outside its range for {label}"
                    )
                estimable = fraction >= float(
                    parameters["min_target_weighted_detected_fraction"]
                ) and ess >= float(parameters["min_target_weighted_detected_ess"])
        if (status == "target_not_estimable") != (not estimable):
            raise BenchmarkConfigError(f"Model status contradicts target eligibility for {label}")
        if estimable:
            eligible[f"group:{group}"].append(target)
        counts[status] += 1
        trees += fitted
    return ModelAccounting(
        eligible_targets={context: tuple(sorted(values)) for context, values in eligible.items()},
        status_counts=dict(counts),
        models_requested=len(rows),
        trees_fitted=trees,
    )
