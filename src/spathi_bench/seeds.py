"""Canonical, domain-separated random seeds for the v1 benchmark design.

The seed ledger is deliberately independent from execution order.  Every
allocation is derived from semantic coordinates, so adding an unrelated run or
reordering a manifest cannot silently change an existing experiment.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import canonical_json_bytes

SEED_ALGORITHM = "sha256-domain-separated-uint32-v1"
SEED_DOMAIN_SEPARATOR = "org.spathi.benchmarks.seed-plan.v1"
SEED_ENCODING = "canonical-json-utf8"
SEED_PROJECTION = "first-64-bits-big-endian-mod-2147483646-plus-1"
MAX_SEED = 2_147_483_646
SEED_STREAMS = (
    "generation",
    "augmentation",
    "inference",
    "statistical",
)

_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_COORDINATE_KEY_RE = re.compile(r"[a-z][a-z0-9_]*")
_BUDGETS = frozenset({"smoke", "reduced", "full"})
_ROLES = frozenset(
    {
        "smoke",
        "development",
        "validation",
        "test",
        "scalability",
        "dimensional_transfer",
        "performance",
    }
)


class SeedPlanError(ValueError):
    """Raised when a seed plan is malformed or no longer reproducible."""


def _require_int(value: Any, label: str, *, maximum: int = MAX_SEED) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise SeedPlanError(f"{label} must be an integer between 1 and {maximum}")
    return value


def _require_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise SeedPlanError(f"{label} must be a canonical lowercase identifier")
    return value


def _coordinates(value: Any, label: str) -> dict[str, str | int | bool]:
    if not isinstance(value, Mapping) or not value:
        raise SeedPlanError(f"{label} must be a non-empty object")
    normalized: dict[str, str | int | bool] = {}
    for key, member in value.items():
        if not isinstance(key, str) or _COORDINATE_KEY_RE.fullmatch(key) is None:
            raise SeedPlanError(f"{label} contains an invalid coordinate name: {key!r}")
        if isinstance(member, bool):
            normalized[key] = member
        elif isinstance(member, int):
            normalized[key] = member
        elif isinstance(member, str) and member:
            normalized[key] = member
        else:
            raise SeedPlanError(f"{label}.{key} must be a non-empty string, integer, or boolean")
    return normalized


def derive_seed(
    *,
    master_seed: int,
    namespace: str,
    stream: str,
    coordinates: Mapping[str, str | int | bool],
    domain_separator: str = SEED_DOMAIN_SEPARATOR,
) -> int:
    """Derive one portable positive 31-bit seed from semantic coordinates."""

    master_seed = _require_int(master_seed, "master_seed")
    namespace = _require_id(namespace, "namespace")
    if stream not in SEED_STREAMS:
        raise SeedPlanError(f"Unknown seed stream: {stream!r}")
    if domain_separator != SEED_DOMAIN_SEPARATOR:
        raise SeedPlanError(f"Unsupported seed domain separator: {domain_separator!r}")
    normalized_coordinates = _coordinates(coordinates, "coordinates")
    payload = {
        "coordinates": normalized_coordinates,
        "domain_separator": domain_separator,
        "master_seed": master_seed,
        "namespace": namespace,
        "stream": stream,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return (int.from_bytes(digest[:8], byteorder="big", signed=False) % MAX_SEED) + 1


def derive_plan_seed(
    plan: Mapping[str, Any],
    *,
    namespace: str,
    stream: str,
    coordinates: Mapping[str, str | int | bool],
) -> int:
    """Derive a concrete task seed from a validated ledger.

    Non-generation allocations reserve each namespace/stream domain.  Concrete
    tasks derive directly from the immutable master seed with their own semantic
    coordinates and must record both coordinates and result in their artifact.
    """

    validated = validate_seed_plan(plan)
    declared_namespaces = {entry["id"] for entry in validated["namespaces"]}
    if namespace not in declared_namespaces:
        raise SeedPlanError(f"Seed namespace is not declared: {namespace!r}")
    declared_domains = {(entry["namespace"], entry["stream"]) for entry in validated["allocations"]}
    if (namespace, stream) not in declared_domains:
        raise SeedPlanError(f"Seed stream {stream!r} is not allocated in namespace {namespace!r}")
    return derive_seed(
        master_seed=validated["master_seed"],
        namespace=namespace,
        stream=stream,
        coordinates=coordinates,
        domain_separator=validated["derivation"]["domain_separator"],
    )


def plan_seed_record(
    plan: Mapping[str, Any],
    *,
    namespace: str,
    stream: str,
    coordinates: Mapping[str, str | int | bool],
) -> dict[str, Any]:
    """Return a self-describing task seed derived from the canonical ledger."""

    normalized_coordinates = _coordinates(coordinates, "coordinates")
    seed = derive_plan_seed(
        plan,
        namespace=namespace,
        stream=stream,
        coordinates=normalized_coordinates,
    )
    return {
        "seed": seed,
        "seed_derivation": {
            "namespace": namespace,
            "stream": stream,
            "coordinates": normalized_coordinates,
        },
    }


def validate_plan_seed_record(
    plan: Mapping[str, Any],
    *,
    seed: Any,
    seed_derivation: Any,
    required_namespace: str | None = None,
    required_stream: str | None = None,
    required_coordinates: Mapping[str, str | int | bool] | None = None,
) -> dict[str, Any]:
    """Validate and normalize a persisted seed plus its complete derivation."""

    recorded_seed = _require_int(seed, "seed")
    if not isinstance(seed_derivation, Mapping) or set(seed_derivation) != {
        "namespace",
        "stream",
        "coordinates",
    }:
        raise SeedPlanError(
            "seed_derivation must contain exactly namespace, stream, and coordinates"
        )
    namespace = _require_id(seed_derivation["namespace"], "seed_derivation.namespace")
    stream = seed_derivation["stream"]
    if not isinstance(stream, str) or stream not in SEED_STREAMS:
        raise SeedPlanError(f"seed_derivation.stream is invalid: {stream!r}")
    coordinates = _coordinates(seed_derivation["coordinates"], "seed_derivation.coordinates")
    if required_namespace is not None and namespace != required_namespace:
        raise SeedPlanError(
            f"seed_derivation.namespace must be {required_namespace!r}, got {namespace!r}"
        )
    if required_stream is not None and stream != required_stream:
        raise SeedPlanError(f"seed_derivation.stream must be {required_stream!r}, got {stream!r}")
    if required_coordinates is not None:
        expected_coordinates = _coordinates(required_coordinates, "required_coordinates")
        if coordinates != expected_coordinates:
            raise SeedPlanError(
                "seed_derivation.coordinates differ from the required semantic coordinates"
            )
    expected = plan_seed_record(
        plan,
        namespace=namespace,
        stream=stream,
        coordinates=coordinates,
    )
    if recorded_seed != expected["seed"]:
        raise SeedPlanError(
            f"seed_derivation resolves to {expected['seed']}, not recorded seed {recorded_seed}"
        )
    return expected


def generation_seed_for_consumer(
    plan: Mapping[str, Any],
    *,
    path: str,
    run_id: str,
    replicate_index: int = 1,
) -> int:
    """Resolve the exact ANDREA seed registered for one simulator replicate."""

    validated = validate_seed_plan(plan)
    if not isinstance(replicate_index, int) or isinstance(replicate_index, bool):
        raise SeedPlanError("replicate_index must be an integer >= 1")
    for allocation in validated["allocations"]:
        if allocation["stream"] != "generation":
            continue
        for consumer in allocation["consumers"]:
            if consumer["path"] != path or consumer["run_id"] != run_id:
                continue
            if not 1 <= replicate_index <= consumer["span"]:
                raise SeedPlanError(
                    f"replicate_index exceeds registered span {consumer['span']} for "
                    f"{path}:{run_id}"
                )
            return allocation["seed"] + consumer["offset"] + replicate_index - 1
    raise SeedPlanError(f"No generation seed consumer is registered for {path}:{run_id}")


def _validate_derivation(value: Any) -> None:
    expected = {
        "algorithm": SEED_ALGORITHM,
        "domain_separator": SEED_DOMAIN_SEPARATOR,
        "encoding": SEED_ENCODING,
        "projection": SEED_PROJECTION,
    }
    if value != expected:
        raise SeedPlanError(f"derivation must equal the v1 contract: {expected}")


def _validate_namespace(raw: Any, index: int) -> tuple[str, str, str]:
    label = f"namespaces[{index}]"
    if not isinstance(raw, Mapping) or set(raw) != {"id", "budget", "dataset_role"}:
        raise SeedPlanError(f"{label} must contain exactly id, budget, and dataset_role")
    namespace_id = _require_id(raw["id"], f"{label}.id")
    budget = raw["budget"]
    role = raw["dataset_role"]
    if budget not in _BUDGETS:
        raise SeedPlanError(f"{label}.budget is invalid")
    if role not in _ROLES:
        raise SeedPlanError(f"{label}.dataset_role is invalid")
    if (budget == "smoke") != (role == "smoke"):
        raise SeedPlanError("The smoke budget and smoke dataset role may only be paired together")
    if namespace_id != f"{budget}.{role}":
        raise SeedPlanError(f"{label}.id must be {budget}.{role}")
    return namespace_id, str(budget), str(role)


def _validate_consumer(value: Any, label: str) -> tuple[str, str, int, int]:
    if not isinstance(value, Mapping) or set(value) != {"path", "run_id", "offset", "span"}:
        raise SeedPlanError(f"{label} must contain exactly path, run_id, offset, and span")
    path = value["path"]
    run_id = value["run_id"]
    if not isinstance(path, str) or not path or "\\" in path:
        raise SeedPlanError(f"{label}.path must be a canonical relative POSIX path")
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or str(parsed) != path
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise SeedPlanError(f"{label}.path must be a canonical relative POSIX path")
    offset = value["offset"]
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise SeedPlanError(f"{label}.offset must be an integer >= 0")
    span = _require_int(value["span"], f"{label}.span", maximum=1_000_000)
    return path, _require_id(run_id, f"{label}.run_id"), offset, span


def recalculate_seed_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy with every recorded allocation seed recalculated."""

    if not isinstance(plan, Mapping):
        raise SeedPlanError("seed plan must be an object")
    recalculated = deepcopy(dict(plan))
    master_seed = _require_int(recalculated.get("master_seed"), "master_seed")
    _validate_derivation(recalculated.get("derivation"))
    allocations = recalculated.get("allocations")
    if not isinstance(allocations, list):
        raise SeedPlanError("allocations must be an array")
    for index, allocation in enumerate(allocations):
        if not isinstance(allocation, dict):
            raise SeedPlanError(f"allocations[{index}] must be an object")
        allocation["seed"] = derive_seed(
            master_seed=master_seed,
            namespace=allocation.get("namespace"),
            stream=allocation.get("stream"),
            coordinates=_coordinates(
                allocation.get("coordinates"), f"allocations[{index}].coordinates"
            ),
            domain_separator=recalculated["derivation"]["domain_separator"],
        )
    return recalculated


def validate_seed_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate structure, derivations, namespace separation, and seed-block uniqueness."""

    if not isinstance(plan, Mapping):
        raise SeedPlanError("seed plan must be an object")
    if plan.get("schema_version") != "1.0":
        raise SeedPlanError("schema_version must be '1.0'")
    allowed_top_level = {
        "schema_version",
        "description",
        "master_seed",
        "derivation",
        "streams",
        "namespaces",
        "allocations",
    }
    if set(plan) != allowed_top_level:
        raise SeedPlanError("seed plan contains missing or unknown top-level fields")
    if not isinstance(plan.get("description"), str) or not plan["description"]:
        raise SeedPlanError("description must be a non-empty string")
    _require_int(plan.get("master_seed"), "master_seed")
    _validate_derivation(plan.get("derivation"))
    if plan.get("streams") != list(SEED_STREAMS):
        raise SeedPlanError(f"streams must equal the ordered v1 streams: {list(SEED_STREAMS)}")

    raw_namespaces = plan.get("namespaces")
    if not isinstance(raw_namespaces, list) or not raw_namespaces:
        raise SeedPlanError("namespaces must be a non-empty array")
    namespaces: dict[str, tuple[str, str]] = {}
    for index, raw_namespace in enumerate(raw_namespaces):
        namespace_id, budget, role = _validate_namespace(raw_namespace, index)
        if namespace_id in namespaces:
            raise SeedPlanError(f"Duplicate seed namespace: {namespace_id}")
        namespaces[namespace_id] = (budget, role)
    expected_namespaces = (
        {"smoke.smoke"}
        | {
            f"{budget}.{role}"
            for budget in ("reduced", "full")
            for role in (
                "development",
                "validation",
                "test",
                "scalability",
                "dimensional_transfer",
                "performance",
            )
        }
    )
    if set(namespaces) != expected_namespaces:
        raise SeedPlanError(
            "namespaces must define smoke plus disjoint reduced/full development, "
            "validation, test, scalability, "
            "dimensional-transfer, and performance namespaces"
        )

    raw_allocations = plan.get("allocations")
    if not isinstance(raw_allocations, list) or not raw_allocations:
        raise SeedPlanError("allocations must be a non-empty array")
    recalculated = recalculate_seed_plan(plan)
    recalculated_allocations = recalculated["allocations"]
    allocation_ids: set[str] = set()
    allocation_pairs: set[tuple[str, str]] = set()
    consumers: set[tuple[str, str]] = set()
    occupied_seeds: dict[int, str] = {}
    for index, (allocation, expected) in enumerate(
        zip(raw_allocations, recalculated_allocations, strict=True)
    ):
        label = f"allocations[{index}]"
        if not isinstance(allocation, Mapping):
            raise SeedPlanError(f"{label} must be an object")
        allowed = {"id", "namespace", "stream", "coordinates", "span", "seed", "consumers"}
        if not set(allocation) <= allowed or not {
            "id",
            "namespace",
            "stream",
            "coordinates",
            "span",
            "seed",
        } <= set(allocation):
            raise SeedPlanError(f"{label} contains missing or unknown fields")
        allocation_id = _require_id(allocation["id"], f"{label}.id")
        if allocation_id in allocation_ids:
            raise SeedPlanError(f"Duplicate seed allocation: {allocation_id}")
        allocation_ids.add(allocation_id)
        namespace = _require_id(allocation["namespace"], f"{label}.namespace")
        if namespace not in namespaces:
            raise SeedPlanError(f"{label}.namespace is not declared: {namespace!r}")
        stream = allocation["stream"]
        if not isinstance(stream, str) or stream not in SEED_STREAMS:
            raise SeedPlanError(f"{label}.stream is invalid: {stream!r}")
        allocation_pair = (namespace, stream)
        if allocation_pair in allocation_pairs:
            raise SeedPlanError(
                f"Duplicate allocation for namespace {namespace!r} and stream {stream!r}"
            )
        allocation_pairs.add(allocation_pair)
        _coordinates(allocation["coordinates"], f"{label}.coordinates")
        span = _require_int(allocation["span"], f"{label}.span", maximum=1_000_000)
        seed = _require_int(allocation["seed"], f"{label}.seed")
        if seed != expected["seed"]:
            raise SeedPlanError(
                f"{label}.seed is stale: recorded {seed}, recalculated {expected['seed']}"
            )
        if seed + span - 1 > MAX_SEED:
            raise SeedPlanError(f"{label} seed block exceeds the supported seed range")
        raw_consumers = allocation.get("consumers")
        if stream == "generation" and raw_consumers is not None:
            if not isinstance(raw_consumers, list) or not raw_consumers:
                raise SeedPlanError(f"{label}.consumers must be a non-empty array")
            covered_offsets: set[int] = set()
            for consumer_index, raw_consumer in enumerate(raw_consumers):
                path, run_id, offset, consumer_span = _validate_consumer(
                    raw_consumer, f"{label}.consumers[{consumer_index}]"
                )
                consumer_key = (path, run_id)
                if consumer_key in consumers:
                    raise SeedPlanError(f"Duplicate generation consumer: {path}:{run_id}")
                consumers.add(consumer_key)
                if offset + consumer_span > span:
                    raise SeedPlanError(
                        f"{label}.consumers[{consumer_index}] exceeds its allocation"
                    )
                for consumer_offset in range(offset, offset + consumer_span):
                    if consumer_offset in covered_offsets:
                        raise SeedPlanError(f"{label}.consumers contain overlapping seed blocks")
                    covered_offsets.add(consumer_offset)
            if covered_offsets != set(range(span)):
                raise SeedPlanError(f"{label}.consumers must cover its complete seed block")
        elif stream == "generation":
            if span != 1 or allocation["coordinates"] != {"purpose": "phase_root"}:
                raise SeedPlanError(
                    f"{label} without simulator consumers must reserve one phase-root seed"
                )
        elif raw_consumers is not None:
            raise SeedPlanError(f"{label}.consumers is only valid for generation allocations")
        for allocated_seed in range(seed, seed + span):
            previous = occupied_seeds.get(allocated_seed)
            if previous is not None:
                raise SeedPlanError(
                    f"Seed {allocated_seed} is shared by allocations {previous!r} and "
                    f"{allocation_id!r}"
                )
            occupied_seeds[allocated_seed] = allocation_id
    expected_pairs = {
        (namespace, stream)
        for namespace, (_budget, role) in namespaces.items()
        for stream in (
            {"inference"}
            | ({"generation"} if role not in {"scalability"} else {"augmentation"})
            | (
                {"statistical"}
                if namespace == "full.test"
                else set()
            )
        )
    }
    if allocation_pairs != expected_pairs:
        missing = sorted(expected_pairs - allocation_pairs)
        unused = sorted(allocation_pairs - expected_pairs)
        raise SeedPlanError(
            "seed allocations must equal the stochastic domains consumed by the v1 design; "
            f"missing={missing}, unused={unused}"
        )
    return deepcopy(dict(plan))


def load_seed_plan(path: Path) -> dict[str, Any]:
    """Read and validate a canonical seed ledger from JSON."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedPlanError(f"Cannot read seed plan {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SeedPlanError(f"Seed plan must contain a JSON object: {path}")
    return validate_seed_plan(payload)
