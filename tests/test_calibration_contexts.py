from __future__ import annotations

import json
from pathlib import Path

import pytest

from spathi_bench.calibration_contexts import (
    completed_group_contexts,
    load_dataset_group_contract,
    require_complete_group_contexts,
)
from spathi_bench.config import BenchmarkConfigError


def _write_dataset(tmp_path: Path, *, truth_contexts: tuple[str, ...]) -> tuple[Path, Path]:
    dataset_id = "dataset_a"
    extras = tmp_path / "extras"
    truth = tmp_path / "truth"
    extras.mkdir()
    truth.mkdir()
    (extras / "groups.tsv").write_text(
        "column\tcluster\ncell_1\ta\ncell_2\tb\n",
        encoding="utf-8",
    )
    (truth / "networks.csv").write_text(
        "source,target,score,sign,evidence,context\n"
        + "".join(f"TF1,G1,1,+,simulated_truth,{context}\n" for context in truth_contexts),
        encoding="utf-8",
    )
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {
                "dataset": {"spec": {"id": dataset_id}},
                "extras": {"groups": "extras/groups.tsv"},
            }
        ),
        encoding="utf-8",
    )
    truth_manifest = tmp_path / "ground-truth-manifest.json"
    truth_manifest.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "truth_requirements": {"contexts": ["global", "group"]},
                "outputs": {"networks": "truth/networks.csv"},
            }
        ),
        encoding="utf-8",
    )
    return dataset_manifest, truth_manifest


def test_dataset_group_contract_reconciles_assignments_and_truth(tmp_path: Path) -> None:
    dataset_manifest, truth_manifest = _write_dataset(
        tmp_path,
        truth_contexts=("global", "group:b", "group:a"),
    )

    contract = load_dataset_group_contract(
        dataset_manifest,
        truth_manifest,
        expected_dataset_id="dataset_a",
    )

    assert contract.expected_group_contexts == ("group:a", "group:b")
    assert contract.to_dict()["n_group_contexts"] == 2


def test_dataset_group_contract_rejects_truth_missing_an_assigned_group(
    tmp_path: Path,
) -> None:
    dataset_manifest, truth_manifest = _write_dataset(
        tmp_path,
        truth_contexts=("global", "group:a"),
    )

    with pytest.raises(BenchmarkConfigError, match="assignments and truth contexts differ"):
        load_dataset_group_contract(
            dataset_manifest,
            truth_manifest,
            expected_dataset_id="dataset_a",
        )


def test_completed_group_contexts_rejects_missing_and_duplicate_inventory() -> None:
    complete = {"tools": {"completed_contexts": {"run_a": ["group:b", "group:a"]}}}
    assert completed_group_contexts(complete, "run_a") == ("group:a", "group:b")

    with pytest.raises(BenchmarkConfigError, match="differ from the dataset contract"):
        require_complete_group_contexts(
            complete,
            "run_a",
            ("group:a", "group:b", "group:c"),
        )

    with pytest.raises(BenchmarkConfigError, match="complete, unique"):
        completed_group_contexts(
            {"tools": {"completed_contexts": {"run_a": ["group:a", "group:a"]}}},
            "run_a",
        )
