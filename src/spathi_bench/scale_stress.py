"""Deterministic, memory-bounded synthetic scale-stress generator.

The generator is deliberately not presented as a biological simulator.  It creates
ANDREA-compatible single-cell datasets with known group-specific regulatory truth so
that completion, resource use and target coverage can be audited on an author-defined,
public and reproducible grid.  Only the already frozen production policy runs here;
configuration selection and scientific accuracy remain confined to independent catalog
simulator panels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import BenchmarkConfigError, load_json_object, validate_json
from .safe_io import (
    atomic_write_json,
    atomic_write_text,
    path_below_no_symlinks,
    reject_symlink_components,
    unique_temporary_file,
)
from .seeds import load_seed_plan, plan_seed_record, validate_plan_seed_record

GENERATOR_ID = "spathi_scale_stress"
_BLOCK_SIZE = 64
_PREPARATION_NORMALIZATION = "library-size-log1p"
_PREPARATION_TARGET_SUM = 10_000.0
_PREPARATION_LIBRARY_SIZE_SCOPE = "all_generated_genes_before_any_filtering"
_PREPARATION_FORMULA = "log1p(target_sum * raw_value / per_cell_library_size)"
_PREPARATION_INPUT_SCALE = "nonnegative_synthetic_counts_like"


@dataclass(frozen=True, slots=True)
class GeneratedDataset:
    """Stable identity and handoff paths for one generated dataset."""

    dataset_id: str
    seed: int
    root: Path
    dataset_manifest: Path
    ground_truth_manifest: Path


def expression_preparation_contract() -> dict[str, str | float]:
    """Return the exact expression transformation consumed by SPATHI inference."""

    return {
        "input_scale": _PREPARATION_INPUT_SCALE,
        "normalization": _PREPARATION_NORMALIZATION,
        "target_sum": _PREPARATION_TARGET_SUM,
        "library_size_scope": _PREPARATION_LIBRARY_SIZE_SCOPE,
        "formula": _PREPARATION_FORMULA,
    }


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_scale_grid(request: Mapping[str, Any]) -> None:
    """Bind every case to the complete author-defined grid declared in its request."""

    requested_preparation = dict(request["expression_preparation"])
    expected_preparation = expression_preparation_contract()
    if requested_preparation != expected_preparation:
        raise BenchmarkConfigError(
            "scale-stress expression preparation differs from the generator contract"
        )

    grid = request["scale_grid"]
    axis_values = {
        "cells": sorted(
            {sum(int(group["cells"]) for group in item["groups"]) for item in request["datasets"]}
        ),
        "genes": sorted({int(item["genes"]) for item in request["datasets"]}),
        "transcription_factors": sorted({int(item["tfs"]) for item in request["datasets"]}),
    }
    for axis, values in axis_values.items():
        declared = [int(value) for value in grid[axis]]
        if declared != sorted(set(declared)):
            raise BenchmarkConfigError(
                f"scale-stress grid axis {axis!r} must be strictly increasing and unique"
            )
        if declared != values:
            raise BenchmarkConfigError(
                f"scale-stress grid axis {axis!r} must exactly enumerate its dataset values"
            )

    declared_roles: list[tuple[str, str]] = []
    for dataset in request["datasets"]:
        role = dataset["design_role"]
        kind = str(role["kind"])
        if kind == "synthetic_scale_case":
            declared_roles.append((kind, str(role["case_id"])))
        elif kind == "synthetic_control":
            declared_roles.append((kind, str(role["control_id"])))
        else:  # The JSON schema is authoritative; this guards direct callers.
            raise BenchmarkConfigError(
                f"scale-stress dataset {dataset['id']} has unknown design role {kind!r}"
            )

    if len(set(declared_roles)) != len(declared_roles):
        raise BenchmarkConfigError("scale-stress panel contains duplicate design-role identities")


def validate_scale_stress_request(repo_root: Path, request_path: Path) -> dict[str, Any]:
    """Load and validate a self-contained synthetic scale-stress request."""

    root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    safe_request = reject_symlink_components(request_path, "scale-stress panel request")
    if not safe_request.is_file():
        raise BenchmarkConfigError(f"scale-stress panel request does not exist: {safe_request}")
    request = load_json_object(safe_request, "scale-stress panel request")
    validate_json(
        request,
        root / "schemas/scale-stress-panel.schema.json",
        "scale-stress panel request",
    )
    dataset_ids = [str(item["id"]) for item in request["datasets"]]
    if len(set(dataset_ids)) != len(dataset_ids):
        raise BenchmarkConfigError("scale-stress panel contains duplicate dataset IDs")
    _validate_scale_grid(request)
    seed_plan_path = path_below_no_symlinks(
        root / str(request["seed_plan"]),
        root,
        "scale-stress seed plan",
        require_file=True,
    )
    seed_plan = load_seed_plan(seed_plan_path)
    for item in request["datasets"]:
        derivation = item["seed_derivation"]
        expected_coordinates = {
            "phase": "scalability",
            "dataset_id": item["id"],
            "augmentation_contract": "scale_stress_v1",
        }
        try:
            validate_plan_seed_record(
                seed_plan,
                seed=item["seed"],
                seed_derivation=derivation,
                required_namespace=f"{derivation['namespace'].split('.', 1)[0]}.scalability",
                required_stream="augmentation",
                required_coordinates=expected_coordinates,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkConfigError(
                f"scale-stress dataset {item['id']} has an invalid seed derivation: {exc}"
            ) from exc
        genes = int(item["genes"])
        tfs = int(item["tfs"])
        active_tfs = max(1, math.floor(tfs * float(item["active_tf_fraction"])))
        if tfs >= genes:
            raise BenchmarkConfigError(f"scale-stress dataset {item['id']} must have tfs < genes")
        if int(item["regulators_per_target"]) > active_tfs:
            raise BenchmarkConfigError(
                f"scale-stress dataset {item['id']} requests more regulators per target than active TFs"
            )
        group_ids = [str(group["id"]) for group in item["groups"]]
        if len(set(group_ids)) != len(group_ids):
            raise BenchmarkConfigError(
                f"scale-stress dataset {item['id']} contains duplicate group IDs"
            )
        if item["network_mode"] == "invariant" and float(item["rewiring_fraction"]) != 0:
            raise BenchmarkConfigError(
                f"Invariant scale-stress dataset {item['id']} must set rewiring_fraction=0"
            )
        if item["network_mode"] == "dynamic" and len(item["groups"]) < 2:
            raise BenchmarkConfigError(
                f"Dynamic scale-stress dataset {item['id']} requires at least two groups"
            )
        if float(item["low_detection_dropout_probability"]) < float(
            item["base_dropout_probability"]
        ):
            raise BenchmarkConfigError(
                f"scale-stress dataset {item['id']} low-detection dropout must be at least base dropout"
            )
    return request


def _dataset_fingerprint(expression: Path, extras: Mapping[str, Path]) -> dict[str, str]:
    components = {"expression": _sha256_file(expression)}
    for name, path in sorted(extras.items()):
        components[f"extra:{name}"] = _sha256_file(path)
    return {
        "algorithm": "sha256",
        "value": hashlib.sha256(_canonical_json(components)).hexdigest(),
    }


def _group_assignments(
    groups: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> tuple[list[str], np.ndarray, tuple[str, ...]]:
    group_ids = tuple(str(group["id"]) for group in groups)
    indexes = np.concatenate(
        [np.full(int(group["cells"]), index, dtype=np.int16) for index, group in enumerate(groups)]
    )
    rng.shuffle(indexes)
    cells = [f"cell_{index + 1:05d}" for index in range(indexes.size)]
    return cells, indexes, group_ids


def _write_expression_header(handle: Any, cells: Sequence[str]) -> None:
    handle.write("gene\t")
    handle.write("\t".join(cells))
    handle.write("\n")


def _write_expression_rows(
    handle: Any,
    names: Sequence[str],
    values: np.ndarray,
) -> None:
    if values.shape != (len(names), values.shape[1]):
        raise AssertionError("expression block shape differs from its gene names")
    for name, row in zip(names, values, strict=True):
        encoded = "\t".join(repr(float(value)) for value in row)
        handle.write(f"{name}\t{encoded}\n")


def _raw_library_sizes(raw_expression: np.memmap) -> np.ndarray:
    """Sum every generated gene in float64 without loading the matrix into RAM."""

    library_sizes = np.zeros(raw_expression.shape[1], dtype=np.float64)
    for start in range(0, raw_expression.shape[0], _BLOCK_SIZE):
        stop = min(start + _BLOCK_SIZE, raw_expression.shape[0])
        block = np.asarray(raw_expression[start:stop], dtype=np.float64)
        if not np.isfinite(block).all() or np.any(block < 0):
            raise BenchmarkConfigError(
                "scale-stress raw expression contains a negative or non-finite value"
            )
        library_sizes += np.sum(block, axis=0, dtype=np.float64)
    invalid = np.flatnonzero(~np.isfinite(library_sizes) | (library_sizes <= 0))
    if invalid.size:
        shown = ", ".join(str(int(index)) for index in invalid[:8])
        raise BenchmarkConfigError(
            "scale-stress expression has cells without a positive finite library size; "
            f"zero-based column indexes: {shown}"
        )
    return library_sizes


def _write_normalized_expression(
    path: Path,
    *,
    raw_expression: np.memmap,
    gene_names: Sequence[str],
    cells: Sequence[str],
    target_sum: float,
) -> tuple[np.ndarray, float]:
    """Write library-size/log1p expression in bounded row blocks.

    The temporary matrix contains the counts-like values used by the synthetic
    regulatory model.  Library sizes are calculated across every row before any
    output filtering, exactly matching the numerical contract of ``spathi
    prepare``.  The returned error audits the pre-log linear-scale sums.
    """

    if raw_expression.shape != (len(gene_names), len(cells)):
        raise AssertionError("raw expression axes differ from their identifiers")
    library_sizes = _raw_library_sizes(raw_expression)
    factors = np.asarray(target_sum / library_sizes, dtype=np.float64)
    normalized_linear_sums = np.zeros(len(cells), dtype=np.float64)
    with path.open("x", encoding="utf-8", newline="") as handle:
        _write_expression_header(handle, cells)
        for start in range(0, len(gene_names), _BLOCK_SIZE):
            stop = min(start + _BLOCK_SIZE, len(gene_names))
            normalized = np.asarray(raw_expression[start:stop], dtype=np.float64)
            normalized *= factors[np.newaxis, :]
            normalized_linear_sums += np.sum(normalized, axis=0, dtype=np.float64)
            np.log1p(normalized, out=normalized)
            if not np.isfinite(normalized).all():
                raise BenchmarkConfigError("scale-stress normalization produced non-finite values")
            _write_expression_rows(handle, gene_names[start:stop], normalized)
    maximum_error = float(np.max(np.abs(normalized_linear_sums - target_sum)))
    return library_sizes, maximum_error


def _edge_rows(
    *,
    target: str,
    group_ids: Sequence[str],
    regulators: np.ndarray,
    coefficients: np.ndarray,
    tf_names: Sequence[str],
) -> Iterable[tuple[str, str, float, str, str, str]]:
    global_edges: dict[int, list[float]] = {}
    for group_index, group_id in enumerate(group_ids):
        for regulator, coefficient in zip(
            regulators[group_index], coefficients[group_index], strict=True
        ):
            regulator_index = int(regulator)
            value = float(coefficient)
            global_edges.setdefault(regulator_index, []).append(value)
            yield (
                tf_names[regulator_index],
                target,
                abs(value),
                "+" if value > 0 else "-",
                "scale_stress_known_model",
                f"group:{group_id}",
            )
    for regulator_index in sorted(global_edges):
        values = global_edges[regulator_index]
        mean = float(np.mean(values))
        yield (
            tf_names[regulator_index],
            target,
            float(np.mean(np.abs(values))),
            "+" if mean >= 0 else "-",
            "scale_stress_known_model",
            "global",
        )


def _generate_one(
    destination: Path,
    spec: Mapping[str, Any],
    seed_plan: Mapping[str, Any],
) -> GeneratedDataset:
    dataset_id = str(spec["id"])
    dataset_root = destination / "datasets" / dataset_id
    if dataset_root.exists():
        raise FileExistsError(f"scale-stress dataset output already exists: {dataset_root}")
    extras_root = dataset_root / "extras"
    truth_root = dataset_root / "truth"
    extras_root.mkdir(parents=True)
    truth_root.mkdir(parents=True)

    root_seed = int(spec["seed"])
    root_derivation = spec["seed_derivation"]
    component_seed_records = {
        component: plan_seed_record(
            seed_plan,
            namespace=root_derivation["namespace"],
            stream="augmentation",
            coordinates={
                **dict(root_derivation["coordinates"]),
                "component": component,
            },
        )
        for component in ("groups", "regulatory_model", "expression", "dropout")
    }
    group_rng = np.random.default_rng(component_seed_records["groups"]["seed"])
    model_rng = np.random.default_rng(component_seed_records["regulatory_model"]["seed"])
    expression_rng = np.random.default_rng(component_seed_records["expression"]["seed"])
    dropout_rng = np.random.default_rng(component_seed_records["dropout"]["seed"])

    cells, group_index, group_ids = _group_assignments(spec["groups"], group_rng)
    n_cells = len(cells)
    n_genes = int(spec["genes"])
    n_tfs = int(spec["tfs"])
    n_targets = n_genes - n_tfs
    active_tf_count = max(1, math.floor(n_tfs * float(spec["active_tf_fraction"])))
    regulated_count = max(1, math.floor(n_targets * float(spec["regulated_target_fraction"])))
    low_detection_count = math.floor(regulated_count * float(spec["low_detection_target_fraction"]))
    regulators_per_target = int(spec["regulators_per_target"])
    group_count = len(group_ids)

    tf_names = [f"TF_{index + 1:04d}" for index in range(n_tfs)]
    target_names = [f"GENE_{index + 1:05d}" for index in range(n_targets)]
    gene_names = [*tf_names, *target_names]

    latent_count = int(spec["latent_factors"])
    latent = expression_rng.normal(0.0, 1.0, size=(n_cells, latent_count)).astype(np.float32)
    group_tf_means = expression_rng.normal(0.2, 0.45, size=(group_count, n_tfs))
    tf_loadings = expression_rng.normal(0.0, 0.25, size=(latent_count, n_tfs))
    library_size = expression_rng.lognormal(
        mean=0.0,
        sigma=float(spec["library_size_log_sd"]),
        size=n_cells,
    ).astype(np.float32)
    log_tf = (
        group_tf_means[group_index]
        + latent @ tf_loadings
        + expression_rng.normal(0.0, float(spec["noise_sd"]), size=(n_cells, n_tfs))
    )
    tf_values = np.expm1(np.clip(log_tf + 1.2, 0.0, 5.0))
    tf_values *= library_size[:, np.newaxis]
    tf_dropout = dropout_rng.random(tf_values.shape) < float(spec["base_dropout_probability"])
    tf_values[tf_dropout] = 0.0
    tf_values = tf_values.astype(np.float32, copy=False)
    standardized_tf = np.log1p(tf_values)
    means = standardized_tf.mean(axis=0)
    scales = standardized_tf.std(axis=0)
    scales[scales == 0] = 1.0
    standardized_tf = ((standardized_tf - means) / scales).astype(np.float32, copy=False)

    expression_path = dataset_root / "expression.tsv"
    network_path = truth_root / "networks.csv"
    preparation = expression_preparation_contract()
    raw_expression: np.memmap | None = None
    with unique_temporary_file(
        dataset_root,
        "scale-stress raw expression staging file",
        boundary=destination,
        prefix=".scale-stress-raw-expression-",
        suffix=".float32",
    ) as raw_path:
        try:
            raw_expression = np.memmap(
                raw_path,
                dtype=np.float32,
                mode="w+",
                shape=(n_genes, n_cells),
                order="C",
            )
            raw_expression[:n_tfs] = tf_values.T
            del tf_values
            with network_path.open("x", encoding="utf-8", newline="") as network_handle:
                network_writer = csv.writer(network_handle, lineterminator="\n")
                network_writer.writerow(
                    ("source", "target", "score", "sign", "evidence", "context")
                )

                for block_start in range(0, n_targets, _BLOCK_SIZE):
                    block_stop = min(block_start + _BLOCK_SIZE, n_targets)
                    block_names = target_names[block_start:block_stop]
                    block = np.empty((len(block_names), n_cells), dtype=np.float32)
                    for local_index, target_name in enumerate(block_names):
                        target_index = block_start + local_index
                        regulated = target_index < regulated_count
                        low_detection = target_index < low_detection_count
                        baseline = -0.8 if low_detection else expression_rng.normal(0.5, 0.25)
                        values = np.full(n_cells, baseline, dtype=np.float32)
                        values += latent @ expression_rng.normal(
                            0.0, 0.16, size=latent_count
                        ).astype(np.float32)
                        if regulated:
                            base_regulators = model_rng.choice(
                                active_tf_count,
                                size=regulators_per_target,
                                replace=False,
                            )
                            base_coefficients = model_rng.uniform(
                                0.2, 1.0, size=regulators_per_target
                            )
                            signs = model_rng.choice(
                                np.array([-1.0, 1.0]), size=regulators_per_target
                            )
                            base_coefficients *= signs
                            regulators = np.tile(base_regulators, (group_count, 1))
                            coefficients = np.tile(base_coefficients, (group_count, 1))
                            if spec["network_mode"] == "dynamic":
                                changes = max(
                                    1,
                                    math.floor(
                                        regulators_per_target * float(spec["rewiring_fraction"])
                                    ),
                                )
                                for current_group in range(1, group_count):
                                    positions = model_rng.choice(
                                        regulators_per_target, size=changes, replace=False
                                    )
                                    available = np.setdiff1d(
                                        np.arange(active_tf_count),
                                        regulators[current_group],
                                        assume_unique=False,
                                    )
                                    if available.size:
                                        replacements = model_rng.choice(
                                            available,
                                            size=min(changes, available.size),
                                            replace=False,
                                        )
                                        regulators[
                                            current_group, positions[: replacements.size]
                                        ] = replacements
                                    coefficients[current_group, positions] = model_rng.uniform(
                                        0.2, 1.0, size=changes
                                    ) * model_rng.choice(np.array([-1.0, 1.0]), size=changes)
                            for current_group in range(group_count):
                                mask = group_index == current_group
                                values[mask] += standardized_tf[mask][
                                    :, regulators[current_group]
                                ] @ coefficients[current_group].astype(np.float32)
                            for edge in _edge_rows(
                                target=target_name,
                                group_ids=group_ids,
                                regulators=regulators,
                                coefficients=coefficients,
                                tf_names=tf_names,
                            ):
                                network_writer.writerow(edge)
                        values += expression_rng.normal(
                            0.0, float(spec["noise_sd"]), size=n_cells
                        ).astype(np.float32)
                        counts = np.expm1(np.clip(values + 1.0, 0.0, 6.0)) * library_size
                        dropout_probability = (
                            float(spec["low_detection_dropout_probability"])
                            if low_detection
                            else float(spec["base_dropout_probability"])
                        )
                        counts[dropout_rng.random(n_cells) < dropout_probability] = 0.0
                        block[local_index] = counts.astype(np.float32, copy=False)
                    raw_expression[n_tfs + block_start : n_tfs + block_stop] = block
            raw_expression.flush()
            library_sizes, normalized_linear_sum_maximum_error = _write_normalized_expression(
                expression_path,
                raw_expression=raw_expression,
                gene_names=gene_names,
                cells=cells,
                target_sum=float(preparation["target_sum"]),
            )
        finally:
            if raw_expression is not None:
                try:
                    raw_expression.flush()
                finally:
                    raw_expression._mmap.close()  # noqa: SLF001 - owned memmap resource
                del raw_expression

    tf_path = extras_root / "tf_list.txt"
    atomic_write_text(
        tf_path,
        "".join(f"{name}\n" for name in tf_names),
        "scale-stress TF list",
        boundary=destination,
    )
    groups_path = extras_root / "groups.tsv"
    with groups_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("cell", "cluster"))
        writer.writerows(
            (cell, group_ids[int(index)]) for cell, index in zip(cells, group_index, strict=True)
        )
    universe_path = truth_root / "gene_universe.txt"
    atomic_write_text(
        universe_path,
        "".join(f"{name}\n" for name in gene_names),
        "scale-stress gene universe",
        boundary=destination,
    )

    fingerprint = _dataset_fingerprint(
        expression_path,
        {"groups": groups_path, "tf_list": tf_path},
    )
    dataset_manifest_path = dataset_root / "dataset-manifest.json"
    atomic_write_json(
        dataset_manifest_path,
        {
            "schema_version": "1.0",
            "id": f"{dataset_id}_manifest",
            "dataset": {
                "spec": {
                    "schema_version": "1.0",
                    "id": dataset_id,
                    "name": f"Deterministic scale-stress scale stress: {dataset_id}",
                    "expression": {
                        "genes": n_genes,
                        "columns": n_cells,
                        "column_kind": "cells",
                        "expression_profile": "scrna",
                        "gene_id_kind": "symbol",
                    },
                    "organism": {"taxonomic_group": "synthetic", "ncbi_taxon_id": None},
                    "context": {"study_system": "author-defined synthetic scale stress"},
                },
                "expression_matrix": "expression.tsv",
            },
            "extras": {"groups": "extras/groups.tsv", "tf_list": "extras/tf_list.txt"},
            "notes": (
                "Benchmark-owned deterministic scale-stress model; not a biological simulator. "
                "No target list is supplied. Expression is prepared as "
                "log1p(10000 * raw_value / per_cell_library_size), where each library size "
                "uses every generated gene before any filtering."
            ),
        },
        "scale-stress dataset manifest",
        boundary=destination,
    )
    ground_truth_path = dataset_root / "ground-truth-manifest.json"
    atomic_write_json(
        ground_truth_path,
        {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "dataset_fingerprint": fingerprint,
            "simulator_id": GENERATOR_ID,
            "data_axes": {
                "measurement": "rna_expression",
                "resolution": "single_cell",
                "column_kind": "cells",
                "experimental_design": "differentiation",
            },
            "truth_requirements": {"contexts": ["global", "group"]},
            "outputs": {
                "gene_universe": "truth/gene_universe.txt",
                "networks": "truth/networks.csv",
            },
            "candidate_space": {
                "sources": "extras/tf_list.txt",
                "targets": "truth/gene_universe.txt",
                "allow_self_edges": False,
            },
            "notes": (
                "Known linear regulatory model with explicit regulated low-detection targets; "
                "all expression genes remain in the requested target universe. The expression "
                "matrix is library-size normalized to 10000 and log1p transformed; regulatory "
                "truth and its coefficients describe the pre-normalization synthetic model."
            ),
        },
        "scale-stress ground-truth manifest",
        boundary=destination,
    )
    atomic_write_json(
        dataset_root / "generator-provenance.json",
        {
            "schema_version": "1.0",
            "generator_id": GENERATOR_ID,
            "model_scope": "computational scale stress only",
            "dataset_id": dataset_id,
            "root_seed": root_seed,
            "seed_derivation": dict(spec["seed_derivation"]),
            "effective_component_seeds": component_seed_records,
            "request": dict(spec),
            "expression_preparation": {
                **preparation,
                "library_size_before_normalization": {
                    "minimum": float(np.min(library_sizes)),
                    "median": float(np.median(library_sizes)),
                    "maximum": float(np.max(library_sizes)),
                },
                "validation": {
                    "positive_finite_library_sizes": True,
                    "normalized_linear_scale_sum_target": float(preparation["target_sum"]),
                    "maximum_absolute_linear_scale_sum_error": (
                        normalized_linear_sum_maximum_error
                    ),
                },
                "working_storage": {
                    "kind": "temporary_disk_backed_float32",
                    "retained": False,
                },
            },
            "derived_counts": {
                "cells": n_cells,
                "genes": n_genes,
                "tfs": n_tfs,
                "active_tfs": active_tf_count,
                "regulated_targets": regulated_count,
                "regulated_low_detection_targets": low_detection_count,
                "null_targets": n_targets - regulated_count,
            },
            "content_sha256": {
                "expression": _sha256_file(expression_path),
                "groups": _sha256_file(groups_path),
                "tf_list": _sha256_file(tf_path),
                "gene_universe": _sha256_file(universe_path),
                "networks": _sha256_file(network_path),
            },
        },
        "scale-stress generator provenance",
        boundary=destination,
    )
    return GeneratedDataset(
        dataset_id=dataset_id,
        seed=root_seed,
        root=dataset_root,
        dataset_manifest=dataset_manifest_path,
        ground_truth_manifest=ground_truth_path,
    )


def generate_scale_stress_panel(*, repo_root: Path, request_path: Path, output_root: Path) -> Path:
    """Generate one strict ANDREA benchmark package from a committed request."""

    repo_root = path_below_no_symlinks(repo_root, repo_root, "Benchmark repository")
    request = validate_scale_stress_request(repo_root, request_path)
    seed_plan = load_seed_plan(repo_root / str(request["seed_plan"]))
    safe_output_root = reject_symlink_components(output_root, "scale-stress output root")
    package_root = path_below_no_symlinks(
        safe_output_root / str(request["id"]),
        safe_output_root,
        "scale-stress benchmark package",
    )
    if package_root.exists():
        raise FileExistsError(f"scale-stress benchmark output already exists: {package_root}")
    package_root.mkdir(parents=True)
    generated: list[GeneratedDataset] = []
    try:
        for spec in request["datasets"]:
            generated.append(_generate_one(package_root, spec, seed_plan))

        def relative(path: Path) -> str:
            return path.relative_to(package_root).as_posix()

        datasets = [
            {
                "dataset_id": item.dataset_id,
                "run_id": item.dataset_id,
                "simulator_id": GENERATOR_ID,
                "seed": item.seed,
                "path": relative(item.root),
                "dataset_manifest": relative(item.dataset_manifest),
                "ground_truth_manifest": relative(item.ground_truth_manifest),
            }
            for item in generated
        ]
        artifacts = []
        for item in generated:
            root = item.root
            artifacts.append(
                {
                    "dataset_id": item.dataset_id,
                    "run_id": item.dataset_id,
                    "simulator_id": GENERATOR_ID,
                    "expression_matrix": relative(root / "expression.tsv"),
                    "gene_universe": relative(root / "truth/gene_universe.txt"),
                    "groups": relative(root / "extras/groups.tsv"),
                    "tf_list": relative(root / "extras/tf_list.txt"),
                    "networks": relative(root / "truth/networks.csv"),
                    "native_outputs": {},
                    "column_descriptors": None,
                    "column_phenotypes": None,
                    "cluster_identities": None,
                    "cell_cell_interactions": None,
                    "chromatin_accessibility": None,
                    "chromatin_regions": None,
                    "enrichment_background": None,
                    "interventions": None,
                    "lineage_tree": None,
                    "perturbation_design": None,
                    "pseudotime": None,
                    "prior_grn": None,
                    "prior_grn_by_group": None,
                    "replicates": None,
                    "spatial_coordinates": None,
                    "timepoints": None,
                }
            )
        runs = []
        tasks = []
        for index, spec in enumerate(request["datasets"], start=1):
            cell_count = sum(int(group["cells"]) for group in spec["groups"])
            memory_gb = max(0.1, cell_count * int(spec["tfs"]) * 8 / 1_000_000_000)
            provenance = {
                "kind": "benchmark_owned_deterministic_generator",
                "prediction_role": "not_applicable",
            }
            runs.append(
                {
                    "run_id": spec["id"],
                    "simulator_id": GENERATOR_ID,
                    "replicates": 1,
                    "base_seed": spec["seed"],
                    "replicate_seeds": [spec["seed"]],
                    "resolved_simulator_params": dict(spec),
                    "runtime_resources": {"threads": 1},
                    "ram_gb": memory_gb,
                    "eta_seconds": 0,
                    "eta_source": "not_predicted",
                    "eta_start_seconds": 0,
                    "eta_end_seconds": 0,
                    "eta_provenance": provenance,
                }
            )
            tasks.append(
                {
                    "task_id": f"task_{index:03d}",
                    "run_id": spec["id"],
                    "simulator_id": GENERATOR_ID,
                    "replicate_index": 1,
                    "seed": spec["seed"],
                    "dataset_id": spec["id"],
                    "runtime_resources": {"threads": 1},
                    "ram_gb": memory_gb,
                    "eta_seconds": 0,
                    "eta_source": "not_predicted",
                    "eta_start_seconds": 0,
                    "eta_end_seconds": 0,
                    "eta_wave": index,
                    "eta_provenance": provenance,
                }
            )
        atomic_write_json(
            package_root / "benchmark-manifest.json",
            {
                "schema_version": "1.0",
                "id": request["id"],
                "organism": {"taxonomic_group": "synthetic", "ncbi_taxon_id": None},
                "requested_extras": ["groups", "tf_list"],
                "effective_extras": ["groups", "tf_list"],
                "base_seed": None,
                "inputs": {
                    "generator": GENERATOR_ID,
                    "request_sha256": _sha256_file(request_path),
                    "scale_grid": dict(request["scale_grid"]),
                    "expression_preparation": dict(request["expression_preparation"]),
                    "description": request["description"],
                },
                "runs": runs,
                "tasks": tasks,
                "execution": {
                    "max_parallel_tasks": 1,
                    "max_cores": 1,
                    "max_ram_gb": max(float(run["ram_gb"]) for run in runs),
                    "eta_total_seconds": 0,
                    "waves": [
                        {
                            "index": index,
                            "threads_used": 1,
                            "ram_gb_used": float(run["ram_gb"]),
                            "eta_seconds": 0,
                            "eta_start_seconds": 0,
                            "eta_end_seconds": 0,
                            "tasks": [
                                {
                                    "task_id": f"task_{index:03d}",
                                    "run_id": run["run_id"],
                                    "simulator_id": GENERATOR_ID,
                                    "threads": 1,
                                    "ram_gb": float(run["ram_gb"]),
                                    "eta_seconds": 0,
                                    "eta_source": "not_predicted",
                                    "eta_start_seconds": 0,
                                    "eta_end_seconds": 0,
                                }
                            ],
                        }
                        for index, run in enumerate(runs, start=1)
                    ],
                    "warnings": [
                        "This panel characterizes computational scale and is not a substitute "
                        "for catalog-simulator accuracy evidence."
                    ],
                },
                "datasets": datasets,
                "artifacts": artifacts,
                "data_axes": {
                    "measurement": "rna_expression",
                    "resolution": "single_cell",
                    "column_kind": "cells",
                    "experimental_design": "differentiation",
                },
                "truth_requirements": {"contexts": ["global", "group"]},
                "notes": request["description"],
            },
            "scale-stress benchmark manifest",
            boundary=package_root,
        )
    except BaseException:
        # The caller normally writes into an artifact-store staging directory.  Leave
        # no apparently complete package when generation fails part-way through.
        import shutil

        shutil.rmtree(package_root, ignore_errors=True)
        raise
    return (package_root / "benchmark-manifest.json").resolve()


__all__ = [
    "GENERATOR_ID",
    "GeneratedDataset",
    "expression_preparation_contract",
    "generate_scale_stress_panel",
    "validate_scale_stress_request",
]
