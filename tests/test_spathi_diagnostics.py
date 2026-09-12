from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.spathi_diagnostics import MODEL_COLUMNS, model_seed, read_model_accounting


@pytest.mark.parametrize(
    ("seed", "group", "target", "expected"),
    [
        (7, "a", "G1", 3811930675),
        (7, "a", "G2", 978580909),
        (7, "b", "G1", 3082443443),
        (123, "A", "TF1", 977727371),
        (123, "B", "G2", 1402291036),
        (0, "α", "β", 2360801408),
    ],
)
def test_model_seed_matches_spathi_task_identity(
    seed: int, group: str, target: str, expected: int
) -> None:
    # Golden values emitted by SPATHI.parallel.stable_task_seed, independently
    # recorded rather than derived from the verifier under test.
    assert model_seed(seed, group, target) == expected


def _diagnostics(path: Path, updates: dict[str, object] | None = None) -> Path:
    records = [
        {
            "target_group": "a",
            "target": "G1",
            "status": "trained",
            "random_seed": 3811930675,
            "n_positive_weight_samples": 25,
            "n_estimators_fitted": 150,
            "adaptive_converged": True,
            "convergence_delta": 0.001,
            "convergence_checks": 1,
            "target_weighted_detected_fraction": 1.0,
            "target_weighted_detected_ess": 25.0,
        },
        {
            "target_group": "a",
            "target": "G2",
            "status": "target_not_estimable",
            "random_seed": 978580909,
            "n_positive_weight_samples": 25,
            "n_estimators_fitted": 0,
            "adaptive_converged": False,
            # Globally excluded targets have no group detection statistics.
        },
    ]
    records[0].update(updates or {})
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MODEL_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    return path


def _read(path: Path, *, globally_eligible: tuple[str, ...] = ("G1",)):
    return read_model_accounting(
        path,
        contexts=("group:a",),
        targets=("G1", "G2"),
        globally_eligible=globally_eligible,
        algorithm_seed=7,
        parameters={
            "target_eligibility": "automatic",
            "adaptive_trees": True,
            "n_estimators": 250,
            "adaptive_min_estimators": 125,
            "adaptive_tree_step": 50,
            "adaptive_patience": 2,
            "adaptive_tolerance": 0.01,
            "min_target_weighted_detected_fraction": 0.01,
            "min_target_weighted_detected_ess": 10,
        },
    )


def test_full_requested_product_includes_globally_excluded_models(tmp_path: Path) -> None:
    accounting = _read(_diagnostics(tmp_path / "models.tsv.gz"))
    assert accounting.eligible_targets == {"group:a": ("G1",)}
    assert accounting.models_requested == 2
    assert accounting.trained == 1
    assert accounting.trees_fitted == 150
    assert accounting.status_counts == {"trained": 1, "target_not_estimable": 1}


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"random_seed": 7}, "seed differs"),
        ({"target": "G9"}, "exact group-target product"),
        ({"status": "invalid_feature_importances"}, "invalid/failed status"),
        ({"n_estimators_fitted": 175}, "off schedule"),
        ({"adaptive_converged": False}, "off schedule"),
        ({"convergence_delta": 0.2}, "convergence is inconsistent"),
        ({"target_weighted_detected_fraction": 0.005}, "contradicts target eligibility"),
        ({"target_weighted_detected_ess": "nan"}, "finite and non-negative"),
    ],
)
def test_corrupted_diagnostics_fail_before_evidence_is_published(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    with pytest.raises(BenchmarkConfigError, match=message):
        _read(_diagnostics(tmp_path / "models.tsv.gz", updates))


def test_group_specific_exclusion_preserves_global_eligibility(tmp_path: Path) -> None:
    path = _diagnostics(
        tmp_path / "models.tsv.gz",
        {
            "status": "target_not_estimable",
            "n_estimators_fitted": 0,
            "adaptive_converged": False,
            "target_weighted_detected_fraction": 0.005,
        },
    )
    accounting = _read(path)
    assert accounting.eligible_targets == {"group:a": ()}
    assert accounting.models_requested == 2
    assert accounting.status_counts == {"target_not_estimable": 2}


def test_global_exclusion_takes_precedence_over_group_detection(tmp_path: Path) -> None:
    path = _diagnostics(tmp_path / "models.tsv.gz")
    with pytest.raises(BenchmarkConfigError, match="contradicts target eligibility"):
        _read(path, globally_eligible=())
