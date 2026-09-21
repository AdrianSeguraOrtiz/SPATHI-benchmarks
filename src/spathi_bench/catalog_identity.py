"""Exact identities for the ANDREA catalog definitions used by a study.

Docker image digests identify executable bytes, but they do not identify the
ANDREA contract that selects inputs, delivery mode and default parameters for
those bytes. A task therefore binds both identities before it can be reused.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .config import BenchmarkConfigError

CATALOG_IDENTITY_CONTRACT = "andrea-catalog-definition-v1"
_CATALOG_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
_LOCATIONS = {
    "tool": ("andrea/catalog_inference_tools/tools", "toolspec.json"),
    "simulator": (
        "andrea/catalog_simulation_data_tools/simulators",
        "simulatorspec.json",
    ),
}


def catalog_definition_identity(
    andrea_repo: Path, *, catalog_kind: str, catalog_id: str
) -> dict[str, str]:
    """Return the effective catalog contract identity for one selected entry."""

    try:
        directory, filename = _LOCATIONS[catalog_kind]
    except KeyError as exc:
        raise BenchmarkConfigError(f"Unknown ANDREA catalog kind: {catalog_kind!r}") from exc
    if not _CATALOG_ID.fullmatch(catalog_id):
        raise BenchmarkConfigError(f"Invalid ANDREA catalog id: {catalog_id!r}")
    path = andrea_repo.resolve() / directory / catalog_id / filename
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkConfigError(f"ANDREA catalog definition is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkConfigError(f"Cannot read ANDREA catalog definition {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise BenchmarkConfigError(f"ANDREA catalog definition must be an object: {path}")
    if payload.get("id") != catalog_id:
        raise BenchmarkConfigError(
            f"ANDREA catalog definition id is {payload.get('id')!r}, expected {catalog_id!r}"
        )
    image = payload.get("docker_image")
    if not isinstance(image, str) or not image:
        raise BenchmarkConfigError(f"ANDREA catalog definition {catalog_id!r} has no Docker image")
    return {
        "contract": CATALOG_IDENTITY_CONTRACT,
        "catalog_kind": catalog_kind,
        "catalog_id": catalog_id,
        "definition_sha256": sha256_file(path),
        "requested_image": image,
    }


def verify_catalog_definitions(andrea_repo: Path, expected: Mapping[str, Any]) -> None:
    """Reject source drift between planning and the instant a task starts."""

    for key, raw in sorted(expected.items()):
        if not isinstance(raw, Mapping):
            raise BenchmarkConfigError(f"Catalog identity {key!r} must be an object")
        catalog_kind = raw.get("catalog_kind")
        catalog_id = raw.get("catalog_id")
        if not isinstance(catalog_kind, str) or not isinstance(catalog_id, str):
            raise BenchmarkConfigError(f"Catalog identity {key!r} is incomplete")
        canonical_key = f"{catalog_kind}:{catalog_id}"
        if key != canonical_key:
            raise BenchmarkConfigError(
                f"Catalog identity key is {key!r}, expected {canonical_key!r}"
            )
        observed = catalog_definition_identity(
            andrea_repo, catalog_kind=catalog_kind, catalog_id=catalog_id
        )
        if dict(raw) != observed:
            raise BenchmarkConfigError(
                f"ANDREA catalog definition changed after planning for {key!r}; "
                "resume the same study so the changed task and its descendants are replanned"
            )


__all__ = [
    "CATALOG_IDENTITY_CONTRACT",
    "catalog_definition_identity",
    "verify_catalog_definitions",
]
