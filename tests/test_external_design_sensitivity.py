from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.external_design_sensitivity import (
    evaluate_external_design_sensitivity,
    load_external_design_sensitivity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_external_design_sensitivity_is_bound_and_deterministic() -> None:
    schema = json.loads(
        (REPO_ROOT / "schemas/external-comparison-design-sensitivity.schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    contract, first = load_external_design_sensitivity(REPO_ROOT)
    assert first == evaluate_external_design_sensitivity(contract)
    assert first["independent_blocks"] == 40
    assert contract["target_design"]["simulator_blocks"] == {
        "dyngen": 20,
        "scmultisim": 20,
    }
    assert all(0.0 <= row["approximate_power"] <= 1.0 for row in first["power"])
    assert [row["approximate_power"] for row in first["power"]] == sorted(
        row["approximate_power"] for row in first["power"]
    )
    assert first["precision_two_sided_alpha"] == pytest.approx(0.0125)
    assert first["precision"][0]["familywise_95_percent_half_width"] == pytest.approx(
        first["critical_value"] * 0.01 / math.sqrt(40)
    )


def test_external_design_sensitivity_rejects_primary_rank_double_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spathi_bench import external_design_sensitivity as module

    real_loader = module.load_json_object

    def changed_loader(path: Path, label: str):
        payload = real_loader(path, label)
        if label == "external comparison plan":
            payload = deepcopy(payload)
            primary = next(
                item
                for item in payload["statistics"]["comparisons"]
                if item["id"] == "full-dynamic-directed-accuracy"
            )
            primary["configuration_ids"].append("pyscenic_default")
        return payload

    monkeypatch.setattr(module, "load_json_object", changed_loader)
    with pytest.raises(BenchmarkConfigError, match="Implementation-sensitivity"):
        load_external_design_sensitivity(REPO_ROOT)
