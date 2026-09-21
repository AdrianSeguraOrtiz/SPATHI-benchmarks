"""Memory-bounded synthetic datasets for computational scalability.

This generator is deliberately a computational stress model, not a biological
simulator.  Catalog simulators provide the accuracy evidence.  Here a known
group-varying linear model provides auditable truth while allowing the matrix
dimensions to reach the intended real-world scale without embedding any real
data in the repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import BenchmarkConfigError
from .safe_io import atomic_write_json, atomic_write_text, unique_temporary_file

GENERATOR_ID = "spathi_scale_stress"
GENERATOR_CONTRACT = "scale-stress-v1"
_BLOCK_ROWS = 64
_TARGET_SUM = 10_000.0


def _component_seed(seed: int, component: str) -> int:
    digest = hashlib.sha256(f"{GENERATOR_CONTRACT}\0{seed}\0{component}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") % 2_147_483_646) + 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_fingerprint(expression: Path, extras: Mapping[str, Path]) -> dict[str, str]:
    components = {"expression": _sha256(expression)}
    components.update({f"extra:{name}": _sha256(path) for name, path in sorted(extras.items())})
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":")).encode("ascii")
    return {"algorithm": "sha256", "value": hashlib.sha256(canonical).hexdigest()}


def _validate(dataset_id: str, seed: int, parameters: Mapping[str, Any]) -> dict[str, Any]:
    if not dataset_id or any(character in dataset_id for character in "/\\\r\n"):
        raise BenchmarkConfigError("Synthetic dataset_id must be a simple non-empty identifier")
    if type(seed) is not int or seed < 1:
        raise BenchmarkConfigError("Synthetic seed must be a positive integer")
    required = {
        "groups",
        "genes",
        "transcription_factors",
        "network_mode",
        "active_tf_fraction",
        "regulated_target_fraction",
        "low_detection_target_fraction",
        "regulators_per_target",
        "rewiring_fraction",
        "latent_factors",
        "library_size_log_sd",
        "noise_sd",
        "base_dropout_probability",
        "low_detection_dropout_probability",
    }
    if set(parameters) != required:
        raise BenchmarkConfigError(
            "Synthetic scale parameters differ from the contract: "
            f"missing={sorted(required - set(parameters))}, "
            f"extra={sorted(set(parameters) - required)}"
        )
    normalized = dict(parameters)
    groups = normalized["groups"]
    if (
        not isinstance(groups, list)
        or not groups
        or any(type(value) is not int or value < 1 for value in groups)
    ):
        raise BenchmarkConfigError("Synthetic groups must be a non-empty list of cell counts")
    genes = normalized["genes"]
    tfs = normalized["transcription_factors"]
    if type(genes) is not int or type(tfs) is not int or not 1 <= tfs < genes:
        raise BenchmarkConfigError("Synthetic dimensions require 1 <= TFs < genes")
    if normalized["network_mode"] not in {"dynamic", "invariant"}:
        raise BenchmarkConfigError("Synthetic network_mode must be dynamic or invariant")
    regulators = normalized["regulators_per_target"]
    if type(regulators) is not int or not 1 <= regulators <= tfs:
        raise BenchmarkConfigError("regulators_per_target must be between one and TF count")
    fractions = (
        "active_tf_fraction",
        "regulated_target_fraction",
        "low_detection_target_fraction",
        "rewiring_fraction",
        "base_dropout_probability",
        "low_detection_dropout_probability",
    )
    for name in fractions:
        value = normalized[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise BenchmarkConfigError(f"{name} must be a fraction in [0, 1]")
    if normalized["low_detection_dropout_probability"] < normalized["base_dropout_probability"]:
        raise BenchmarkConfigError("Low-detection dropout must not be below baseline dropout")
    if type(normalized["latent_factors"]) is not int or normalized["latent_factors"] < 1:
        raise BenchmarkConfigError("latent_factors must be a positive integer")
    for name in ("library_size_log_sd", "noise_sd"):
        value = normalized[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise BenchmarkConfigError(f"{name} must be a non-negative number")
    return normalized


def _groups(
    group_sizes: Sequence[int], rng: np.random.Generator
) -> tuple[list[str], np.ndarray, tuple[str, ...]]:
    group_ids = tuple(f"group_{index + 1}" for index in range(len(group_sizes)))
    assignments = np.concatenate(
        [np.full(size, index, dtype=np.int16) for index, size in enumerate(group_sizes)]
    )
    rng.shuffle(assignments)
    cells = [f"cell_{index + 1:06d}" for index in range(assignments.size)]
    return cells, assignments, group_ids


def _write_expression(
    path: Path,
    raw: np.memmap,
    genes: Sequence[str],
    cells: Sequence[str],
) -> dict[str, float]:
    library_sizes = np.zeros(raw.shape[1], dtype=np.float64)
    for start in range(0, raw.shape[0], _BLOCK_ROWS):
        block = np.asarray(raw[start : start + _BLOCK_ROWS], dtype=np.float64)
        if not np.isfinite(block).all() or np.any(block < 0):
            raise BenchmarkConfigError("Synthetic expression contains invalid raw values")
        library_sizes += block.sum(axis=0, dtype=np.float64)
    if np.any(~np.isfinite(library_sizes)) or np.any(library_sizes <= 0):
        raise BenchmarkConfigError("Synthetic expression contains an empty cell library")
    factors = _TARGET_SUM / library_sizes
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write("gene\t" + "\t".join(cells) + "\n")
        for start in range(0, len(genes), _BLOCK_ROWS):
            stop = min(start + _BLOCK_ROWS, len(genes))
            block = np.asarray(raw[start:stop], dtype=np.float64)
            block *= factors[np.newaxis, :]
            np.log1p(block, out=block)
            for name, row in zip(genes[start:stop], block, strict=True):
                handle.write(name)
                handle.write("\t")
                handle.write("\t".join(format(float(value), ".8g") for value in row))
                handle.write("\n")
    return {
        "minimum": float(library_sizes.min()),
        "median": float(np.median(library_sizes)),
        "maximum": float(library_sizes.max()),
    }


def _write_group_extras(
    extras: Path,
    cells: Sequence[str],
    assignments: np.ndarray,
    group_ids: Sequence[str],
) -> dict[str, Path]:
    groups = extras / "groups.tsv"
    phenotypes = extras / "column_phenotypes.tsv"
    with (
        groups.open("x", encoding="utf-8", newline="") as groups_handle,
        phenotypes.open("x", encoding="utf-8", newline="") as phenotype_handle,
    ):
        group_writer = csv.writer(groups_handle, delimiter="\t", lineterminator="\n")
        phenotype_writer = csv.writer(phenotype_handle, delimiter="\t", lineterminator="\n")
        group_writer.writerow(("cell", "cluster"))
        phenotype_writer.writerow(("column", "phenotype", "order"))
        for cell, group_index in zip(cells, assignments, strict=True):
            group = group_ids[int(group_index)]
            group_writer.writerow((cell, group))
            phenotype_writer.writerow((cell, group, int(group_index)))
    lineage = extras / "lineage_tree.tsv"
    with lineage.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("child", "parent", "gain_rate", "loss_rate"))
        for index, group in enumerate(group_ids):
            parent = "__root__" if index == 0 else group_ids[index - 1]
            rate = 0.0 if index == 0 else 0.5
            writer.writerow((group, parent, rate, rate))
    return {"groups": groups, "column_phenotypes": phenotypes, "lineage_tree": lineage}


def generate_scale_dataset(
    *,
    output_root: Path,
    dataset_id: str,
    seed: int,
    parameters: Mapping[str, Any],
) -> Path:
    """Generate one complete ANDREA benchmark package and return its manifest."""

    spec = _validate(dataset_id, seed, parameters)
    package = output_root / dataset_id
    dataset = package / "datasets" / dataset_id
    if package.exists():
        raise FileExistsError(f"Synthetic output already exists: {package}")
    extras = dataset / "extras"
    truth = dataset / "truth"
    extras.mkdir(parents=True)
    truth.mkdir()

    component_seeds = {
        name: _component_seed(seed, name) for name in ("groups", "model", "expression", "dropout")
    }
    group_rng = np.random.default_rng(component_seeds["groups"])
    model_rng = np.random.default_rng(component_seeds["model"])
    expression_rng = np.random.default_rng(component_seeds["expression"])
    dropout_rng = np.random.default_rng(component_seeds["dropout"])

    cells, assignments, group_ids = _groups(spec["groups"], group_rng)
    n_cells = len(cells)
    n_genes = int(spec["genes"])
    n_tfs = int(spec["transcription_factors"])
    n_targets = n_genes - n_tfs
    active_tfs = max(1, math.floor(n_tfs * float(spec["active_tf_fraction"])))
    regulated_targets = max(1, math.floor(n_targets * float(spec["regulated_target_fraction"])))
    low_detection_targets = math.floor(
        regulated_targets * float(spec["low_detection_target_fraction"])
    )
    regulators_per_target = int(spec["regulators_per_target"])
    tf_names = [f"TF_{index + 1:05d}" for index in range(n_tfs)]
    target_names = [f"GENE_{index + 1:05d}" for index in range(n_targets)]
    gene_names = [*tf_names, *target_names]

    latent_count = int(spec["latent_factors"])
    latent = expression_rng.normal(size=(n_cells, latent_count)).astype(np.float32)
    group_means = expression_rng.normal(0.2, 0.45, size=(len(group_ids), n_tfs))
    loadings = expression_rng.normal(0.0, 0.25, size=(latent_count, n_tfs))
    library_scale = expression_rng.lognormal(
        mean=0.0, sigma=float(spec["library_size_log_sd"]), size=n_cells
    ).astype(np.float32)
    tf_values = np.expm1(
        np.clip(
            group_means[assignments]
            + latent @ loadings
            + expression_rng.normal(0.0, float(spec["noise_sd"]), (n_cells, n_tfs))
            + 1.2,
            0.0,
            5.0,
        )
    )
    tf_values *= library_scale[:, np.newaxis]
    tf_values[dropout_rng.random(tf_values.shape) < float(spec["base_dropout_probability"])] = 0.0
    tf_values = tf_values.astype(np.float32, copy=False)
    standardized_tf = np.log1p(tf_values)
    tf_sd = standardized_tf.std(axis=0)
    tf_sd[tf_sd == 0] = 1.0
    standardized_tf = ((standardized_tf - standardized_tf.mean(axis=0)) / tf_sd).astype(
        np.float32, copy=False
    )

    network = truth / "networks.csv"
    raw: np.memmap | None = None
    with unique_temporary_file(
        dataset,
        "synthetic raw expression",
        boundary=package,
        prefix=".raw-expression-",
        suffix=".float32",
    ) as raw_path:
        try:
            raw = np.memmap(
                raw_path, dtype=np.float32, mode="w+", shape=(n_genes, n_cells), order="C"
            )
            raw[:n_tfs] = tf_values.T
            del tf_values
            with network.open("x", encoding="utf-8", newline="") as network_handle:
                writer = csv.writer(network_handle, lineterminator="\n")
                writer.writerow(("source", "target", "score", "sign", "evidence", "context"))
                for block_start in range(0, n_targets, _BLOCK_ROWS):
                    block_stop = min(block_start + _BLOCK_ROWS, n_targets)
                    block = np.empty((block_stop - block_start, n_cells), dtype=np.float32)
                    for local_index, target_index in enumerate(range(block_start, block_stop)):
                        regulated = target_index < regulated_targets
                        low_detection = target_index < low_detection_targets
                        baseline = -0.8 if low_detection else expression_rng.normal(0.5, 0.25)
                        values = np.full(n_cells, baseline, dtype=np.float32)
                        values += latent @ expression_rng.normal(0.0, 0.16, latent_count).astype(
                            np.float32
                        )
                        if regulated:
                            base_regulators = model_rng.choice(
                                active_tfs, size=regulators_per_target, replace=False
                            )
                            base_coefficients = model_rng.uniform(
                                0.2, 1.0, regulators_per_target
                            ) * model_rng.choice((-1.0, 1.0), regulators_per_target)
                            regulators = np.tile(base_regulators, (len(group_ids), 1))
                            coefficients = np.tile(base_coefficients, (len(group_ids), 1))
                            if spec["network_mode"] == "dynamic":
                                changes = max(
                                    1,
                                    math.floor(
                                        regulators_per_target * float(spec["rewiring_fraction"])
                                    ),
                                )
                                for group_index in range(1, len(group_ids)):
                                    positions = model_rng.choice(
                                        regulators_per_target, size=changes, replace=False
                                    )
                                    available = np.setdiff1d(
                                        np.arange(active_tfs), regulators[group_index]
                                    )
                                    if available.size:
                                        replacements = model_rng.choice(
                                            available,
                                            size=min(changes, available.size),
                                            replace=False,
                                        )
                                        regulators[group_index, positions[: replacements.size]] = (
                                            replacements
                                        )
                                    coefficients[group_index, positions] = model_rng.uniform(
                                        0.2, 1.0, changes
                                    ) * model_rng.choice((-1.0, 1.0), changes)
                            global_edges: dict[int, list[float]] = {}
                            for group_index, group in enumerate(group_ids):
                                mask = assignments == group_index
                                values[mask] += standardized_tf[mask][
                                    :, regulators[group_index]
                                ] @ coefficients[group_index].astype(np.float32)
                                for regulator, coefficient in zip(
                                    regulators[group_index], coefficients[group_index], strict=True
                                ):
                                    source = int(regulator)
                                    value = float(coefficient)
                                    global_edges.setdefault(source, []).append(value)
                                    writer.writerow(
                                        (
                                            tf_names[source],
                                            target_names[target_index],
                                            abs(value),
                                            "+" if value > 0 else "-",
                                            GENERATOR_ID,
                                            f"group:{group}",
                                        )
                                    )
                            for source, coefficients_by_group in sorted(global_edges.items()):
                                writer.writerow(
                                    (
                                        tf_names[source],
                                        target_names[target_index],
                                        float(np.mean(np.abs(coefficients_by_group))),
                                        "+" if np.mean(coefficients_by_group) >= 0 else "-",
                                        GENERATOR_ID,
                                        "global",
                                    )
                                )
                        values += expression_rng.normal(
                            0.0, float(spec["noise_sd"]), n_cells
                        ).astype(np.float32)
                        counts = np.expm1(np.clip(values + 1.0, 0.0, 6.0)) * library_scale
                        dropout = (
                            float(spec["low_detection_dropout_probability"])
                            if low_detection
                            else float(spec["base_dropout_probability"])
                        )
                        counts[dropout_rng.random(n_cells) < dropout] = 0.0
                        block[local_index] = counts.astype(np.float32, copy=False)
                    raw[n_tfs + block_start : n_tfs + block_stop] = block
            raw.flush()
            library_summary = _write_expression(dataset / "expression.tsv", raw, gene_names, cells)
        finally:
            if raw is not None:
                raw.flush()
                raw._mmap.close()  # noqa: SLF001 - this function owns the memmap
                del raw

    extra_paths = _write_group_extras(extras, cells, assignments, group_ids)
    tf_list = extras / "tf_list.txt"
    atomic_write_text(
        tf_list,
        "".join(f"{name}\n" for name in tf_names),
        "Synthetic TF list",
        boundary=package,
    )
    extra_paths["tf_list"] = tf_list
    universe = truth / "gene_universe.txt"
    atomic_write_text(
        universe,
        "".join(f"{name}\n" for name in gene_names),
        "Synthetic gene universe",
        boundary=package,
    )
    fingerprint = _dataset_fingerprint(dataset / "expression.tsv", extra_paths)
    dataset_manifest = dataset / "dataset-manifest.json"
    atomic_write_json(
        dataset_manifest,
        {
            "schema_version": "1.0",
            "id": f"{dataset_id}_manifest",
            "dataset": {
                "spec": {
                    "schema_version": "1.0",
                    "id": dataset_id,
                    "name": f"Synthetic computational scale case {dataset_id}",
                    "expression": {
                        "genes": n_genes,
                        "columns": n_cells,
                        "column_kind": "cells",
                        "expression_profile": "scrna",
                        "gene_id_kind": "symbol",
                    },
                    "organism": {"taxonomic_group": "synthetic", "ncbi_taxon_id": None},
                    "context": {"study_system": "deterministic computational scale stress"},
                },
                "expression_matrix": "expression.tsv",
            },
            "extras": {
                name: path.relative_to(dataset).as_posix()
                for name, path in sorted(extra_paths.items())
            },
            "notes": "Computational scale model; not a biological simulator.",
        },
        "Synthetic dataset manifest",
        boundary=package,
    )
    truth_manifest = dataset / "ground-truth-manifest.json"
    atomic_write_json(
        truth_manifest,
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
            "notes": "Known group-varying linear regulatory model.",
        },
        "Synthetic truth manifest",
        boundary=package,
    )
    provenance = dataset / "generator-provenance.json"
    atomic_write_json(
        provenance,
        {
            "schema_version": "1.0",
            "generator_id": GENERATOR_ID,
            "generator_contract": GENERATOR_CONTRACT,
            "dataset_id": dataset_id,
            "seed": seed,
            "component_seeds": component_seeds,
            "parameters": spec,
            "derived": {
                "cells": n_cells,
                "genes": n_genes,
                "transcription_factors": n_tfs,
                "active_transcription_factors": active_tfs,
                "regulated_targets": regulated_targets,
                "low_detection_targets": low_detection_targets,
            },
            "normalization": {
                "method": "library-size-log1p",
                "target_sum": _TARGET_SUM,
                "pre_normalization_library_size": library_summary,
            },
        },
        "Synthetic generator provenance",
        boundary=package,
    )

    benchmark_manifest = package / "benchmark-manifest.json"
    relative = lambda path: path.relative_to(package).as_posix()  # noqa: E731
    atomic_write_json(
        benchmark_manifest,
        {
            "schema_version": "1.0",
            "id": dataset_id,
            "organism": {"taxonomic_group": "synthetic", "ncbi_taxon_id": None},
            "requested_extras": sorted(extra_paths),
            "effective_extras": sorted(extra_paths),
            "base_seed": seed,
            "inputs": {"generator": GENERATOR_ID, "contract": GENERATOR_CONTRACT},
            "runs": [
                {
                    "run_id": dataset_id,
                    "simulator_id": GENERATOR_ID,
                    "replicates": 1,
                    "base_seed": seed,
                    "replicate_seeds": [seed],
                    "resolved_simulator_params": spec,
                    "runtime_resources": {"threads": 1},
                    "ram_gb": max(0.1, n_cells * n_tfs * 8 / 1_000_000_000),
                    "eta_seconds": 0,
                    "eta_source": "not_predicted",
                    "eta_start_seconds": 0,
                    "eta_end_seconds": 0,
                    "eta_provenance": {
                        "kind": "benchmark_owned_deterministic_generator",
                        "prediction_role": "not_applicable",
                    },
                }
            ],
            "tasks": [],
            "execution": {
                "max_parallel_tasks": 1,
                "max_cores": 1,
                "max_ram_gb": max(0.1, n_cells * n_tfs * 8 / 1_000_000_000),
                "eta_total_seconds": 0,
                "waves": [],
                "warnings": [
                    "This panel measures computational scale and is not biological accuracy evidence."
                ],
            },
            "datasets": [
                {
                    "dataset_id": dataset_id,
                    "run_id": dataset_id,
                    "simulator_id": GENERATOR_ID,
                    "seed": seed,
                    "path": relative(dataset),
                    "dataset_manifest": relative(dataset_manifest),
                    "ground_truth_manifest": relative(truth_manifest),
                }
            ],
            "artifacts": [
                {
                    "dataset_id": dataset_id,
                    "run_id": dataset_id,
                    "simulator_id": GENERATOR_ID,
                    "expression_matrix": relative(dataset / "expression.tsv"),
                    "gene_universe": relative(universe),
                    "groups": relative(extra_paths["groups"]),
                    "tf_list": relative(tf_list),
                    "networks": relative(network),
                    "column_phenotypes": relative(extra_paths["column_phenotypes"]),
                    "lineage_tree": relative(extra_paths["lineage_tree"]),
                    "native_outputs": {},
                }
            ],
            "data_axes": {
                "measurement": "rna_expression",
                "resolution": "single_cell",
                "column_kind": "cells",
                "experimental_design": "differentiation",
            },
            "truth_requirements": {"contexts": ["global", "group"]},
            "notes": "Deterministic computational scale dataset.",
        },
        "Synthetic benchmark manifest",
        boundary=package,
    )
    return benchmark_manifest.resolve()


__all__ = ["GENERATOR_CONTRACT", "GENERATOR_ID", "generate_scale_dataset"]
