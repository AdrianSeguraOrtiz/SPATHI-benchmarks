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
            "n_estimators_fitted": 250,
            "fit_seconds": 0.1,
        },
        {
            "target_group": "a",
            "target": "G2",
            "status": "constant_target",
            "random_seed": 978580909,
            "n_positive_weight_samples": 25,
            "n_estimators_fitted": 0,
            "fit_seconds": 0.0,
        },
    ]
    records[0].update(updates or {})
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MODEL_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    return path


def _read(path: Path):
    return read_model_accounting(
        path,
        contexts=("group:a",),
        targets=("G1", "G2"),
        algorithm_seed=7,
        parameters={
            "n_estimators": 250,
        },
    )


def test_full_requested_product_includes_constant_targets(tmp_path: Path) -> None:
    accounting = _read(_diagnostics(tmp_path / "models.tsv.gz"))
    assert accounting.models_requested == 2
    assert accounting.trained == 1
    assert accounting.trees_fitted == 250
    assert accounting.status_counts == {"trained": 1, "constant_target": 1}


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"random_seed": 7}, "seed differs"),
        ({"target": "G9"}, "exact group-target product"),
        ({"status": "invalid_feature_importances"}, "invalid/failed status"),
        ({"n_estimators_fitted": 175}, "invalid tree budget"),
        ({"status": "constant_target"}, "Untrained model reports fitted trees"),
        ({"fit_seconds": "nan"}, "finite and non-negative"),
    ],
)
def test_corrupted_diagnostics_fail_before_evidence_is_published(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    with pytest.raises(BenchmarkConfigError, match=message):
        _read(_diagnostics(tmp_path / "models.tsv.gz", updates))


def test_metadata_counters_match_diagnostics(tmp_path: Path) -> None:
    accounting = _read(_diagnostics(tmp_path / "models.tsv.gz"))
    counters = {
        "requested": 2,
        "completed": 2,
        "trained": 1,
        "preflight_skipped": 1,
        "fit_or_importance_failures": 0,
        "skipped_target_records": 1,
        "fitted_estimators_total": 250,
    }
    accounting.validate_metadata(counters)
    with pytest.raises(BenchmarkConfigError, match="counters differ"):
        accounting.validate_metadata({**counters, "fitted_estimators_total": 150})
