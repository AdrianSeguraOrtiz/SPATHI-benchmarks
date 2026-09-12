"""Execution context for a named study and its explicitly permitted revision drift.

Frozen scientific inputs and artifact checksums remain authoritative. Source revisions
may advance during an operator-requested resume; every invocation records its actual
revisions in the study journal and every newly published artifact records its producer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .config import BenchmarkConfigError, load_json_object
from .safe_io import write_idempotent_json

_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar("spathi_study", default=None)
logger = logging.getLogger(__name__)


@contextmanager
def study_execution(
    study_id: str, *, revisions: Mapping[str, Any], resume: bool
) -> Iterator[None]:
    token = _ACTIVE.set({"study_id": study_id, "revisions": dict(revisions), "resume": resume})
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def execution_provenance() -> dict[str, Any] | None:
    context = _ACTIVE.get()
    if context is None:
        return None
    return {"study_id": context["study_id"], "revisions": context["revisions"]}


def phase_flow_id(content_id: str) -> str:
    context = _ACTIVE.get()
    return context["study_id"] if context is not None else content_id


def permit_revision_change(label: str, recorded: Any, observed: Any) -> bool:
    context = _ACTIVE.get()
    if context is None or not context["resume"]:
        return False
    # A warning is emitted once per distinct transition within this invocation.
    marker = (label, str(recorded), str(observed))
    seen = context.setdefault("warnings", set())
    if marker not in seen:
        logger.warning(
            "%s changed (%s → %s). Resuming study %s with preserved results; "
            "the study journal records both revisions.",
            label, recorded, observed, context["study_id"],
        )
        seen.add(marker)
    return True


def bind_phase_identity(workflow_root: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the initial phase identity while recording the actual resumed producer.

    Only named-study resumes permit a change. Files, seeds, image contracts and
    materialized scientific requests continue through their normal content validators.
    """
    path = workflow_root / "identity.json"
    if path.is_file():
        recorded = load_json_object(path, "phase identity")
        if recorded != identity and not permit_revision_change(
            "Phase implementation identity", recorded, dict(identity)
        ):
            raise BenchmarkConfigError("Phase identity changed outside a named study resume")
        return recorded
    write_idempotent_json(path, dict(identity), "phase identity", boundary=workflow_root)
    return dict(identity)
