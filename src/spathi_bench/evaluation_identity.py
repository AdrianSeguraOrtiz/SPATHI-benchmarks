"""Semantic identities for the candidate and truth spaces used by ANDREA.

ANDREA validates that inference and evaluation use the same dataset fingerprint,
but its metric rows expose only candidate-space cardinalities.  Benchmark pairing
needs a stronger, portable identity: equal counts do not imply equal genes or
edges.  This module derives content identities from the immutable ground-truth
artifact that was actually evaluated.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import Artifact, canonical_json_sha256
from .config import BenchmarkConfigError, load_json_object

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LEVELS = ("topology", "directed", "signed")
_VALID_SIGNS = {"+", "-"}


@dataclass(frozen=True, slots=True)
class EvaluationSpaceIdentity:
    """Lossless semantic identity of one evaluated ground-truth package."""

    dataset_fingerprint_sha256: str
    source_universe_sha256: str
    target_universe_sha256: str
    candidate_space_sha256_by_level: dict[str, str]
    truth_edges_sha256_by_context_level: dict[tuple[str, str], str]
    candidate_count_by_level: dict[str, int]
    truth_edge_count_by_context_level: dict[tuple[str, str], int]
    genes: frozenset[str]
    sources: frozenset[str]
    targets: frozenset[str]
    allow_self_edges: bool

    def metric_provenance(self, *, context: str, level: str) -> dict[str, str]:
        """Return the identities that must agree within a paired metric context."""

        if level not in self.candidate_space_sha256_by_level:
            raise BenchmarkConfigError(f"Unsupported ANDREA evaluation level: {level!r}")
        truth_digest = self.truth_edges_sha256_by_context_level.get((context, level))
        if truth_digest is None:
            raise BenchmarkConfigError(
                f"Ground truth has no semantic edge identity for {context!r}/{level!r}"
            )
        return {
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "candidate_source_universe_sha256": self.source_universe_sha256,
            "candidate_target_universe_sha256": self.target_universe_sha256,
            "candidate_edge_universe_sha256": self.candidate_space_sha256_by_level[level],
            "truth_edge_universe_sha256": truth_digest,
        }

    def validate_evaluation_report(self, report: dict[str, Any], *, dataset_id: str) -> None:
        """Reject an evaluation report detached from this immutable truth package."""

        expected_fingerprint = {
            "algorithm": "sha256",
            "value": self.dataset_fingerprint_sha256,
        }
        inputs = report.get("inputs")
        ground_truth = report.get("ground_truth")
        if not isinstance(inputs, dict) or not isinstance(ground_truth, dict):
            raise BenchmarkConfigError("Evaluation report lacks inputs or ground_truth provenance")
        if (
            inputs.get("inference_dataset_id") != dataset_id
            or inputs.get("ground_truth_dataset_id") != dataset_id
            or ground_truth.get("dataset_id") != dataset_id
        ):
            raise BenchmarkConfigError("Evaluation report has a different dataset identity")
        if any(
            value != expected_fingerprint
            for value in (
                inputs.get("inference_dataset_fingerprint"),
                inputs.get("ground_truth_dataset_fingerprint"),
                ground_truth.get("dataset_fingerprint"),
            )
        ):
            raise BenchmarkConfigError(
                "Evaluation report fingerprints differ from the immutable ground-truth artifact"
            )
        candidate = ground_truth.get("candidate_space")
        if not isinstance(candidate, dict):
            raise BenchmarkConfigError("Evaluation report lacks candidate-space provenance")
        expected_counts = {
            "n_sources": len(self.sources),
            "n_targets": len(self.targets),
            "n_source_target_overlap": len(self.sources & self.targets),
        }
        for name, expected in expected_counts.items():
            if candidate.get(name) != expected:
                raise BenchmarkConfigError(
                    f"Evaluation report {name} differs from the immutable candidate space"
                )
        if ground_truth.get("gene_universe_size") != len(self.genes):
            raise BenchmarkConfigError(
                "Evaluation report gene-universe size differs from the immutable gene universe"
            )
        if candidate.get("allow_self_edges") is not self.allow_self_edges:
            raise BenchmarkConfigError(
                "Evaluation report self-edge policy differs from the immutable candidate space"
            )
        if candidate.get("n_candidates_by_level") != self.candidate_count_by_level:
            raise BenchmarkConfigError(
                "Evaluation report candidate counts differ from the immutable candidate space"
            )

    def validate_metric(self, metric: dict[str, Any]) -> None:
        """Check ANDREA's row counts against the semantic spaces being hashed."""

        context = metric.get("context")
        level = metric.get("level")
        if not isinstance(context, str) or not isinstance(level, str):
            raise BenchmarkConfigError("Evaluation metric lacks context or level identity")
        expected_candidates = self.candidate_count_by_level.get(level)
        expected_truth = self.truth_edge_count_by_context_level.get((context, level))
        if expected_candidates is None or expected_truth is None:
            raise BenchmarkConfigError(
                f"Evaluation metric refers to unknown semantic space {context!r}/{level!r}"
            )
        if metric.get("n_candidates") != expected_candidates:
            raise BenchmarkConfigError(
                f"Evaluation metric candidate count differs from semantic identity for "
                f"{context!r}/{level!r}"
            )
        if metric.get("n_candidate_genes") != len(self.genes):
            raise BenchmarkConfigError(
                "Evaluation metric gene-universe count differs from semantic identity"
            )
        if metric.get("n_candidate_sources") != len(self.sources):
            raise BenchmarkConfigError(
                "Evaluation metric source count differs from semantic identity"
            )
        if metric.get("n_candidate_targets") != len(self.targets):
            raise BenchmarkConfigError(
                "Evaluation metric target count differs from semantic identity"
            )
        if metric.get("n_truth_edges") != expected_truth:
            raise BenchmarkConfigError(
                f"Evaluation metric truth-edge count differs from semantic identity for "
                f"{context!r}/{level!r}"
            )


def _artifact_relative_member(
    artifact: Artifact,
    *,
    manifest_path: Path,
    raw_reference: Any,
    label: str,
    suffix: str,
) -> Path:
    if not isinstance(raw_reference, str) or not raw_reference:
        raise BenchmarkConfigError(f"{label} must be a non-empty relative path")
    reference = PurePosixPath(raw_reference)
    if (
        reference.is_absolute()
        or str(reference) != raw_reference
        or "\\" in raw_reference
        or "//" in raw_reference
        or any(part in {".", ".."} for part in reference.parts)
        or not raw_reference.endswith(suffix)
    ):
        raise BenchmarkConfigError(f"{label} is not a canonical {suffix} artifact path")
    candidate = manifest_path.parent.joinpath(*reference.parts).resolve()
    root = artifact.path.resolve()
    if not candidate.is_relative_to(root):
        raise BenchmarkConfigError(f"{label} escapes its immutable dataset artifact")
    try:
        return artifact.file(candidate.relative_to(root).as_posix()).resolve()
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} is not declared by its dataset artifact") from exc


def _gene_set(path: Path, *, label: str) -> frozenset[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkConfigError(f"Cannot read {label} {path}: {exc}") from exc
    genes: list[str] = []
    seen: set[str] = set()
    for line_number, gene in enumerate(lines, start=1):
        if not gene or gene != gene.strip():
            raise BenchmarkConfigError(
                f"{label} contains a non-canonical gene at line {line_number}: {path}"
            )
        if gene in seen:
            raise BenchmarkConfigError(
                f"{label} contains duplicate gene {gene!r} at line {line_number}: {path}"
            )
        seen.add(gene)
        genes.append(gene)
    if not genes:
        raise BenchmarkConfigError(f"{label} contains no genes: {path}")
    return frozenset(genes)


def _fingerprint_sha256(manifest: dict[str, Any]) -> str:
    fingerprint = manifest.get("dataset_fingerprint")
    if not isinstance(fingerprint, dict) or set(fingerprint) != {"algorithm", "value"}:
        raise BenchmarkConfigError(
            "Ground-truth dataset_fingerprint must contain exactly algorithm and value"
        )
    digest = fingerprint.get("value")
    if fingerprint.get("algorithm") != "sha256" or not isinstance(digest, str):
        raise BenchmarkConfigError("Ground-truth dataset_fingerprint must use SHA-256")
    if _SHA256_RE.fullmatch(digest) is None:
        raise BenchmarkConfigError("Ground-truth dataset_fingerprint has an invalid SHA-256 value")
    return digest


def _candidate_count(
    *, sources: frozenset[str], targets: frozenset[str], allow_self_edges: bool, level: str
) -> int:
    overlap = len(sources & targets)
    directed = len(sources) * len(targets)
    if not allow_self_edges:
        directed -= overlap
    if level == "topology":
        return directed - (overlap * (overlap - 1) // 2)
    if level == "directed":
        return directed
    if level == "signed":
        return directed * len(_VALID_SIGNS)
    raise BenchmarkConfigError(f"Unsupported ANDREA evaluation level: {level!r}")


def _edge_key(source: str, target: str, sign: str, *, level: str) -> tuple[str, ...]:
    if level == "topology":
        return tuple(sorted((source, target)))
    if level == "directed":
        return (source, target)
    if level == "signed":
        return (source, target, sign)
    raise BenchmarkConfigError(f"Unsupported ANDREA evaluation level: {level!r}")


def _is_candidate(
    source: str,
    target: str,
    *,
    sources: frozenset[str],
    targets: frozenset[str],
    allow_self_edges: bool,
    level: str,
) -> bool:
    direct = source in sources and target in targets and (allow_self_edges or source != target)
    if level == "topology":
        reverse = target in sources and source in targets and (allow_self_edges or source != target)
        return direct or reverse
    return direct


def _truth_edge_identities(
    path: Path,
    *,
    sources: frozenset[str],
    targets: frozenset[str],
    allow_self_edges: bool,
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], int]]:
    keys: dict[tuple[str, str], set[tuple[str, ...]]] = {}
    contexts: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"source", "target", "score", "sign", "context"}
            if not required.issubset(reader.fieldnames or ()):
                raise BenchmarkConfigError(
                    f"Ground-truth network lacks semantic identity columns: {path}"
                )
            for line_number, row in enumerate(reader, start=2):
                source = str(row.get("source", "")).strip()
                target = str(row.get("target", "")).strip()
                context = str(row.get("context", "")).strip()
                sign = str(row.get("sign", "")).strip()
                try:
                    score = float(row.get("score", ""))
                except (TypeError, ValueError) as exc:
                    raise BenchmarkConfigError(
                        f"Ground-truth network has an invalid score at line {line_number}: {path}"
                    ) from exc
                if (
                    not source
                    or not target
                    or not context
                    or not math.isfinite(score)
                    or score <= 0
                ):
                    raise BenchmarkConfigError(
                        f"Ground-truth network has an invalid row at line {line_number}: {path}"
                    )
                contexts.add(context)
                for level in _LEVELS:
                    if level == "signed" and sign not in _VALID_SIGNS:
                        continue
                    if _is_candidate(
                        source,
                        target,
                        sources=sources,
                        targets=targets,
                        allow_self_edges=allow_self_edges,
                        level=level,
                    ):
                        keys.setdefault((context, level), set()).add(
                            _edge_key(source, target, sign, level=level)
                        )
    except OSError as exc:
        raise BenchmarkConfigError(f"Cannot read ground-truth network {path}: {exc}") from exc
    if not contexts:
        raise BenchmarkConfigError(f"Ground-truth network contains no truth rows: {path}")
    for context in contexts:
        for level in _LEVELS:
            keys.setdefault((context, level), set())
    identities = {
        key: canonical_json_sha256(
            {
                "semantic_type": "andrea_truth_edge_universe",
                "level": key[1],
                "context": key[0],
                "edges": [list(edge) for edge in sorted(edges)],
            }
        )
        for key, edges in keys.items()
    }
    return identities, {key: len(edges) for key, edges in keys.items()}


def load_evaluation_space_identity(
    *,
    dataset_artifact: Artifact,
    ground_truth_manifest_path: Path,
    expected_dataset_id: str,
) -> EvaluationSpaceIdentity:
    """Derive exact comparison identities from one immutable dataset artifact."""

    root = dataset_artifact.path.resolve()
    manifest_path = ground_truth_manifest_path.resolve()
    if not manifest_path.is_relative_to(root):
        raise BenchmarkConfigError("Ground-truth manifest escapes its dataset artifact")
    try:
        manifest_path = dataset_artifact.file(manifest_path.relative_to(root).as_posix()).resolve()
    except ValueError as exc:
        raise BenchmarkConfigError(
            "Ground-truth manifest is not declared by its dataset artifact"
        ) from exc
    manifest = load_json_object(manifest_path, "ground-truth manifest")
    if manifest.get("dataset_id") != expected_dataset_id:
        raise BenchmarkConfigError(
            "Ground-truth manifest dataset_id differs from benchmark dataset identity"
        )
    outputs = manifest.get("outputs")
    candidate = manifest.get("candidate_space")
    if not isinstance(outputs, dict) or not isinstance(candidate, dict):
        raise BenchmarkConfigError("Ground-truth manifest lacks outputs or candidate_space")
    if set(candidate) != {"sources", "targets", "allow_self_edges"}:
        raise BenchmarkConfigError(
            "Ground-truth candidate_space must contain exactly sources, targets, and allow_self_edges"
        )
    allow_self_edges = candidate.get("allow_self_edges")
    if allow_self_edges is not False:
        raise BenchmarkConfigError("Ground-truth candidate_space.allow_self_edges must be false")

    gene_universe_path = _artifact_relative_member(
        dataset_artifact,
        manifest_path=manifest_path,
        raw_reference=outputs.get("gene_universe"),
        label="Ground-truth outputs.gene_universe",
        suffix=".txt",
    )
    source_path = _artifact_relative_member(
        dataset_artifact,
        manifest_path=manifest_path,
        raw_reference=candidate.get("sources"),
        label="Ground-truth candidate_space.sources",
        suffix=".txt",
    )
    target_path = _artifact_relative_member(
        dataset_artifact,
        manifest_path=manifest_path,
        raw_reference=candidate.get("targets"),
        label="Ground-truth candidate_space.targets",
        suffix=".txt",
    )
    network_path = _artifact_relative_member(
        dataset_artifact,
        manifest_path=manifest_path,
        raw_reference=outputs.get("networks"),
        label="Ground-truth outputs.networks",
        suffix=".csv",
    )
    genes = _gene_set(gene_universe_path, label="Ground-truth gene universe")
    sources = _gene_set(source_path, label="Ground-truth candidate source universe")
    targets = _gene_set(target_path, label="Ground-truth candidate target universe")
    for label, universe in (("source", sources), ("target", targets)):
        outside = sorted(universe - genes)
        if outside:
            raise BenchmarkConfigError(
                f"Candidate {label} universe contains genes outside the gene universe: "
                + ", ".join(outside[:8])
            )

    source_hash = canonical_json_sha256(sorted(sources))
    target_hash = canonical_json_sha256(sorted(targets))
    candidate_hashes = {
        level: canonical_json_sha256(
            {
                "semantic_type": "andrea_candidate_edge_universe",
                "level": level,
                "source_genes": sorted(sources),
                "target_genes": sorted(targets),
                "allow_self_edges": allow_self_edges,
                "n_candidates": _candidate_count(
                    sources=sources,
                    targets=targets,
                    allow_self_edges=allow_self_edges,
                    level=level,
                ),
            }
        )
        for level in _LEVELS
    }
    truth_hashes, truth_counts = _truth_edge_identities(
        network_path,
        sources=sources,
        targets=targets,
        allow_self_edges=allow_self_edges,
    )
    candidate_counts = {
        level: _candidate_count(
            sources=sources,
            targets=targets,
            allow_self_edges=allow_self_edges,
            level=level,
        )
        for level in _LEVELS
    }
    return EvaluationSpaceIdentity(
        dataset_fingerprint_sha256=_fingerprint_sha256(manifest),
        source_universe_sha256=source_hash,
        target_universe_sha256=target_hash,
        candidate_space_sha256_by_level=candidate_hashes,
        truth_edges_sha256_by_context_level=truth_hashes,
        candidate_count_by_level=candidate_counts,
        truth_edge_count_by_context_level=truth_counts,
        genes=genes,
        sources=sources,
        targets=targets,
        allow_self_edges=allow_self_edges,
    )


__all__ = [
    "EvaluationSpaceIdentity",
    "load_evaluation_space_identity",
]
