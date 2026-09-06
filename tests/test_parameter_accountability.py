from __future__ import annotations

import json
from pathlib import Path

from spathi_bench.config import validate_json

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCOUNTABILITY_PATH = REPO_ROOT / "experiments/v1/spathi/parameter-accountability.json"
SCHEMA_PATH = REPO_ROOT / "schemas/spathi-parameter-accountability.schema.json"
SEARCH_PATH = REPO_ROOT / "experiments/v1/spathi/search-space.json"


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_parameter_accountability_is_schema_valid_and_complete() -> None:
    accountability = _read(ACCOUNTABILITY_PATH)
    validate_json(accountability, SCHEMA_PATH, "SPATHI parameter accountability")

    entries = accountability["inference_parameters"]
    assert isinstance(entries, list)
    identifiers = [entry["id"] for entry in entries]
    assert len(identifiers) == len(set(identifiers)) == 27

    search = _read(SEARCH_PATH)
    base = search["base_configuration"]
    assert isinstance(base, dict)
    assert set(identifiers) == {*base, "random_seed"}


def test_optional_scientific_inputs_are_explicitly_outside_benchmarking() -> None:
    accountability = _read(ACCOUNTABILITY_PATH)
    optional = {entry["id"]: entry["disposition"] for entry in accountability["optional_inputs"]}
    assert optional == {
        "target_list": "excluded_unavailable_input",
        "centroid_weights": "excluded_annotation_assisted_input",
    }
