from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.design_sensitivity import (
    evaluate_design_sensitivity,
    load_design_sensitivity_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_contract_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    shutil.copytree(REPO_ROOT / "experiments", root / "experiments")
    (root / "schemas").mkdir()
    for name in (
        "efficiency-design-sensitivity.schema.json",
        "profile.schema.json",
    ):
        shutil.copyfile(REPO_ROOT / "schemas" / name, root / "schemas" / name)
    return root


def test_committed_efficiency_design_sensitivity_is_valid_and_deterministic() -> None:
    schema = json.loads(
        (REPO_ROOT / "schemas/efficiency-design-sensitivity.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(schema)
    contract, first = load_design_sensitivity_contract(REPO_ROOT)
    second = evaluate_design_sensitivity(contract)

    assert contract["method"]["random_sampling"] is False
    assert contract["method"]["confirmatory_candidate_count"] == 1
    assert contract["method"]["per_candidate_one_sided_alpha"] == 0.05
    assert contract["target_design"]["expected_independent_datasets"] == 25
    assert contract["target_design"]["expected_effective_sample_size"] == 16.0
    assert {item["replicates"] for item in contract["target_design"]["strata"]} == {5}
    assert first == second
    assert len(first["accuracy"]) == 25
    assert len(first["runtime"]) == 30
    assert all(0.0 <= row["approximate_gate_probability"] <= 1.0 for row in first["accuracy"])
    assert all(0.0 <= row["approximate_gate_probability"] <= 1.0 for row in first["runtime"])


@pytest.mark.parametrize(
    "mutation",
    (
        "dataset_count",
        "stratum_replication",
        "multiplicity",
        "accuracy_margin",
        "runtime_threshold",
    ),
)
def test_design_sensitivity_rejects_drift_from_the_frozen_study(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _copy_contract_repository(tmp_path)
    path = root / "experiments/v1/spathi/efficiency-design-sensitivity.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    changed = deepcopy(contract)
    if mutation == "dataset_count":
        changed["target_design"]["expected_independent_datasets"] -= 1
    elif mutation == "stratum_replication":
        changed["target_design"]["strata"][0]["replicates"] -= 1
    elif mutation == "multiplicity":
        changed["method"]["per_candidate_one_sided_alpha"] = 0.04
    elif mutation == "accuracy_margin":
        changed["scenario_grid"]["accuracy"]["noninferiority_margin"] = 0.003
    else:
        changed["scenario_grid"]["runtime"]["maximum_ratio"] = 0.95
    path.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(BenchmarkConfigError):
        load_design_sensitivity_contract(root)
