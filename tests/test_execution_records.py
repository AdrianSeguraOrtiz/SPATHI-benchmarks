from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.execution_records import unique_execution_records


def _reference(kind: str, *, cache_hit: bool = False) -> dict[str, Any]:
    digest = "a" * 64
    return {
        "kind": kind,
        "key": f"sha256:{digest}",
        "digest": digest,
        "path": f"/artifacts/{kind}/{digest}",
        "cache_hit": cache_hit,
    }


def _record(*, evaluation: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "dataset_id": "dynamic",
        "regime": "group_only",
        "run_id": "spathi_policy__seed_42",
        "batch_id": "main",
        "batch_kind": "spathi",
        "configuration_id": "spathi_policy",
        "configuration_sha256": "b" * 64,
        "execution_sha256": "c" * 64,
        "algorithm_seed": 42,
        "selection_role": "primary",
        "resources": {"cpu_threads": 8, "memory_bytes": 2**30},
        "run_report": "/artifacts/inference/report.json",
        "artifact": _reference("evaluation" if evaluation else "inference"),
    }
    if evaluation:
        record["inference_artifact"] = _reference("inference")
        record["evaluation_report"] = "/artifacts/evaluation/report.json"
    return record


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("first_hit", [False, True])
def test_batch_aliases_keep_first_record_without_mutating_inputs(
    evaluation: bool, first_hit: bool
) -> None:
    first = _record(evaluation=evaluation)
    second = deepcopy(first)
    second["batch_id"] = "all_genes"
    for field in ("artifact", "inference_artifact"):
        if field in first:
            first[field]["cache_hit"] = first_hit
            second[field]["cache_hit"] = not first_hit
    inputs = [first, second]
    before = deepcopy(inputs)

    output = unique_execution_records(iter(inputs))

    assert output == [first]
    assert output[0] is first
    assert inputs == before


def test_empty_input_and_stable_order_for_distinct_execution_contexts() -> None:
    first = _record()
    records = [first]
    for field, value in (
        ("dataset_id", "invariant"),
        ("regime", "order_aware"),
        ("run_id", "spathi_policy__seed_43"),
    ):
        record = deepcopy(first)
        record[field] = value
        records.append(record)

    assert unique_execution_records([]) == []
    result = unique_execution_records(records)
    assert all(actual is original for actual, original in zip(result, records, strict=True))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("configuration_sha256", "d" * 64),
        ("execution_sha256", "d" * 64),
        ("algorithm_seed", 43),
        ("selection_role", "sensitivity"),
        ("resources", {"cpu_threads": 1, "memory_bytes": 2**30}),
        ("run_report", "/another/report.json"),
        ("new_metadata", {"cache_hit": True}),
        ("cache_hit", True),
        ("batch_kind", "andrea"),
    ],
)
def test_same_run_rejects_scientific_and_unknown_field_differences(field: str, value: Any) -> None:
    first = _record()
    second = deepcopy(first)
    second["batch_id"] = "all_genes"
    second[field] = value

    with pytest.raises(BenchmarkConfigError, match=f"Conflicting execution records.*{field}"):
        unique_execution_records([first, second])


@pytest.mark.parametrize("reference_field", ["artifact", "inference_artifact"])
@pytest.mark.parametrize("field", ["path", "kind", "key_and_digest", "new_metadata"])
def test_same_run_rejects_different_artifact_identity_or_metadata(
    reference_field: str, field: str
) -> None:
    first = _record(evaluation=True)
    second = deepcopy(first)
    reference = second[reference_field]
    if field == "key_and_digest":
        reference["digest"] = "d" * 64
        reference["key"] = f"sha256:{reference['digest']}"
    else:
        reference[field] = "different"

    with pytest.raises(
        BenchmarkConfigError, match=f"Conflicting execution records.*{reference_field}"
    ):
        unique_execution_records([first, second])


@pytest.mark.parametrize("reference_field", ["artifact", "inference_artifact"])
@pytest.mark.parametrize("field", ["kind", "key", "digest", "path"])
def test_aliases_require_nonempty_artifact_identity(reference_field: str, field: str) -> None:
    first = _record(evaluation=True)
    first[reference_field][field] = ""

    with pytest.raises(BenchmarkConfigError, match="no valid .* identity"):
        unique_execution_records([first, deepcopy(first)])


def test_singleton_record_also_requires_artifact_identity() -> None:
    record = _record()
    del record["artifact"]

    with pytest.raises(BenchmarkConfigError, match="no valid artifact identity"):
        unique_execution_records([record])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "bad/kind", "invalid artifact identity"),
        ("digest", "incorrect", "invalid artifact identity"),
        ("key", "sha256:" + "d" * 64, "key differs from its digest"),
        ("cache_hit", "false", "cache_hit must be boolean"),
    ],
)
def test_aliases_reject_invalid_artifact_identity(field: str, value: Any, message: str) -> None:
    first = _record()
    first["artifact"][field] = value

    with pytest.raises(BenchmarkConfigError, match=message):
        unique_execution_records([first, deepcopy(first)])


def test_aliases_do_not_confuse_json_booleans_and_numbers() -> None:
    first = _record()
    first["new_metadata"] = False
    second = deepcopy(first)
    second["new_metadata"] = 0

    with pytest.raises(BenchmarkConfigError, match="new_metadata"):
        unique_execution_records([first, second])
