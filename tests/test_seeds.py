from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from spathi_bench.seeds import (
    MAX_SEED,
    SEED_STREAMS,
    SeedPlanError,
    derive_plan_seed,
    derive_seed,
    generation_seed_for_consumer,
    load_seed_plan,
    plan_seed_record,
    recalculate_seed_plan,
    validate_plan_seed_record,
    validate_seed_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_PLAN_PATH = REPO_ROOT / "experiments/v1/seeds/seed-plan.json"


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_seed_plan_matches_schema_and_recalculates_exactly() -> None:
    raw = _read_json(SEED_PLAN_PATH)
    schema = _read_json(REPO_ROOT / "schemas/seed-plan.schema.json")

    Draft202012Validator(schema).validate(raw)
    assert load_seed_plan(SEED_PLAN_PATH) == raw
    assert recalculate_seed_plan(raw) == raw


def test_seed_derivation_is_coordinate_order_independent_and_domain_separated() -> None:
    plan = load_seed_plan(SEED_PLAN_PATH)
    common = {
        "master_seed": plan["master_seed"],
        "namespace": "full.test",
        "coordinates": {"condition": "linear", "replicate": 1},
    }
    generation = derive_seed(stream="generation", **common)
    reordered = derive_seed(
        master_seed=plan["master_seed"],
        namespace="full.test",
        stream="generation",
        coordinates={"replicate": 1, "condition": "linear"},
    )
    inference = derive_seed(stream="inference", **common)
    reduced = derive_seed(
        master_seed=plan["master_seed"],
        namespace="reduced.test",
        stream="generation",
        coordinates=common["coordinates"],
    )

    assert generation == reordered
    assert generation == 1_725_932_867
    assert len({generation, inference, reduced}) == 3
    assert all(1 <= seed <= MAX_SEED for seed in (generation, inference, reduced))

    assert (
        derive_plan_seed(
            plan,
            namespace="full.test",
            stream="generation",
            coordinates={"condition": "linear", "replicate": 1},
        )
        == generation
    )


def test_seed_plan_reserves_only_consumed_domains_without_seed_overlap() -> None:
    plan = load_seed_plan(SEED_PLAN_PATH)
    namespaces = {entry["id"]: entry["dataset_role"] for entry in plan["namespaces"]}
    observed_pairs = {(entry["namespace"], entry["stream"]) for entry in plan["allocations"]}
    expected_pairs = {
        (namespace, stream)
        for namespace, role in namespaces.items()
        for stream in (
            {"inference"}
            | ({"generation"} if role != "scalability" else {"augmentation"})
            | (
                {"statistical"}
                if namespace == "full.test"
                else set()
            )
        )
    }

    assert observed_pairs == expected_pairs
    assert {stream for _, stream in observed_pairs} == set(SEED_STREAMS)
    occupied: set[int] = set()
    for allocation in plan["allocations"]:
        block = set(range(allocation["seed"], allocation["seed"] + allocation["span"]))
        assert occupied.isdisjoint(block)
        occupied.update(block)



def test_every_simulator_manifest_is_registered_and_matches_its_seed_block() -> None:
    plan = load_seed_plan(SEED_PLAN_PATH)
    registered_paths: set[str] = set()
    namespace_seeds: dict[str, set[int]] = {}
    for allocation in plan["allocations"]:
        if allocation["stream"] != "generation":
            continue
        namespace = allocation["namespace"]
        namespace_seeds.setdefault(namespace, set())
        for consumer in allocation.get("consumers", []):
            registered_paths.add(consumer["path"])
            payload = _read_json(REPO_ROOT / consumer["path"])
            assert payload["schema_version"] == "1.0"
            assert isinstance(payload["runs"], list) and len(payload["runs"]) == 1
            run = payload["runs"][0]
            expected_seed = allocation["seed"] + consumer["offset"]
            assert run["run_id"] == consumer["run_id"]
            assert run["replicates"] == consumer["span"]
            assert run["base_seed"] == expected_seed
            assert (
                generation_seed_for_consumer(
                    plan,
                    path=consumer["path"],
                    run_id=consumer["run_id"],
                )
                == expected_seed
            )
            assert (
                generation_seed_for_consumer(
                    plan,
                    path=consumer["path"],
                    run_id=consumer["run_id"],
                    replicate_index=consumer["span"],
                )
                == expected_seed + consumer["span"] - 1
            )
            namespace_seeds[namespace].update(
                range(expected_seed, expected_seed + consumer["span"])
            )

    simulator_paths = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "experiments/v1/simulations").rglob("*runs.json")
    }
    assert registered_paths == simulator_paths
    assert namespace_seeds["reduced.development"].isdisjoint(namespace_seeds["full.development"])
    assert namespace_seeds["reduced.validation"].isdisjoint(namespace_seeds["full.validation"])


def test_self_describing_seed_records_round_trip_and_reject_drift() -> None:
    plan = load_seed_plan(SEED_PLAN_PATH)
    record = plan_seed_record(
        plan,
        namespace="full.scalability",
        stream="augmentation",
        coordinates={
            "phase": "scalability",
            "dataset_id": "fixture",
            "component": "dropout",
        },
    )

    assert (
        validate_plan_seed_record(
            plan,
            seed=record["seed"],
            seed_derivation=record["seed_derivation"],
            required_namespace="full.scalability",
            required_stream="augmentation",
            required_coordinates={
                "component": "dropout",
                "dataset_id": "fixture",
                "phase": "scalability",
            },
        )
        == record
    )
    with pytest.raises(SeedPlanError, match="not recorded seed"):
        validate_plan_seed_record(
            plan,
            seed=record["seed"] + 1,
            seed_derivation=record["seed_derivation"],
        )


def test_stale_seed_and_overlapping_consumer_blocks_are_rejected() -> None:
    plan = load_seed_plan(SEED_PLAN_PATH)
    stale = deepcopy(plan)
    stale["allocations"][0]["seed"] += 1
    with pytest.raises(SeedPlanError, match="is stale"):
        validate_seed_plan(stale)

    overlap = deepcopy(plan)
    generation = next(
        allocation
        for allocation in overlap["allocations"]
        if allocation["namespace"] == "full.test" and allocation["stream"] == "generation"
    )
    generation["consumers"][1]["offset"] = generation["consumers"][0]["offset"]
    with pytest.raises(SeedPlanError, match="overlapping seed blocks"):
        validate_seed_plan(overlap)


def test_recalculation_does_not_mutate_the_supplied_plan() -> None:
    plan = _read_json(SEED_PLAN_PATH)
    stale = deepcopy(plan)
    stale["allocations"][0]["seed"] = 1
    before = deepcopy(stale)

    recalculated = recalculate_seed_plan(stale)

    assert stale == before
    assert recalculated["allocations"][0]["seed"] == plan["allocations"][0]["seed"]
