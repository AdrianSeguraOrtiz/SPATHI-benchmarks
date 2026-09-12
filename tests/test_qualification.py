from __future__ import annotations

from pathlib import Path

import pytest

from spathi_bench.artifacts import sha256_file
from spathi_bench.config import BenchmarkConfigError, load_json_object
from spathi_bench.qualification import _completed_receipt, dense_metrics, run_qualification

ROOT = Path(__file__).resolve().parents[1]


def test_dense_metric_oracle_ties_and_missing_predictions() -> None:
    # Candidates: TF->A (positive), TF->B (negative), TF->C (positive).
    # At score .5 both A and B are retrieved; C is an implicit zero.
    metrics = dense_metrics(
        {("TF", "A"): 0.5, ("TF", "B"): 0.5},
        sources=("TF",),
        targets=("TF", "A", "B", "C"),
        truth={("TF", "A"), ("TF", "C"), ("outside", "A")},
    )
    assert metrics["aupr"] == pytest.approx(0.5 / 2 + (2 / 3) / 2)
    assert metrics["auroc"] == 0.25
    assert dense_metrics(
        {},
        sources=("TF",),
        targets=("A", "B", "C"),
        truth={("TF", "A"), ("TF", "C")},
    ) == {"aupr": 2 / 3, "auroc": 0.5}


def test_dense_metric_oracle_rejects_nonfinite_or_single_class_inputs() -> None:
    with pytest.raises(BenchmarkConfigError, match="finite scores and both classes"):
        dense_metrics(
            {("TF", "A"): float("nan")}, sources=("TF",), targets=("A", "B"), truth={("TF", "A")}
        )
    with pytest.raises(BenchmarkConfigError, match="finite scores and both classes"):
        dense_metrics({}, sources=("TF",), targets=("A",), truth={("TF", "A")})


@pytest.mark.integration
def test_real_spathi_and_andrea_scientific_contracts(tmp_path: Path) -> None:
    andrea = ROOT.parent / "ANDREA"
    spathi = ROOT.parent / "SPATHI"
    if not all((repo / ".venv/bin/python").is_file() for repo in (andrea, spathi)):
        pytest.skip("Physical qualification needs installed sibling ANDREA and SPATHI checkouts")
    receipt = run_qualification(
        repo_root=ROOT,
        andrea_repo=andrea,
        spathi_repo=spathi,
        work_root=tmp_path,
        run_id="real-scientific-contracts",
        threads=2,
    )
    assert receipt["status"] == "complete"
    assert receipt["threads_tested"] == [1, 2]
    assert receipt["models"]["automatic_adaptive"]["status_counts"]["target_not_estimable"] == 4
    assert len(receipt["metrics"]) == 8
    assert all(row["n_truth_rows_outside_candidate_space"] == 1 for row in receipt["metrics"])
    assert Path(receipt["report"]).is_file()
    assert Path(receipt["andrea_report"]).is_file()
    assert load_json_object(Path(receipt["receipt_path"])) == receipt
    for relative, digest in receipt["files"].items():
        assert sha256_file(Path(receipt["attempt"]) / relative) == digest
    resumed = run_qualification(
        repo_root=ROOT,
        andrea_repo=andrea,
        spathi_repo=spathi,
        work_root=tmp_path,
        run_id="real-scientific-contracts",
        threads=2,
    )
    assert resumed == receipt
    # Changed code requires a new physical attempt; a completed receipt cannot
    # attest to code it never executed.
    receipt_root = Path(receipt["receipt_path"]).parent
    assert (
        _completed_receipt(receipt_root, software={"new_revision": True}, resources={"threads": 2})
        is None
    )
    Path(receipt["report"]).write_text("changed", encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="qualification output changed"):
        _completed_receipt(receipt_root, software=receipt["software"], resources={"threads": 2})
