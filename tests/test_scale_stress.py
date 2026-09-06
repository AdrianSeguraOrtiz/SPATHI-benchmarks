from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

import spathi_bench.scale_stress as scale_stress
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.manifests import datasets_from_benchmark
from spathi_bench.scale_stress import generate_scale_stress_panel, validate_scale_stress_request

REPO_ROOT = Path(__file__).resolve().parents[1]


def _request(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "id": "tiny_scale_stress",
                "description": "Small deterministic contract fixture.",
                "seed_plan": "experiments/v1/seeds/seed-plan.json",
                "scale_grid": {
                    "provenance": "author_defined_public_synthetic_grid",
                    "cells": [10],
                    "genes": [24],
                    "transcription_factors": [5],
                },
                "expression_preparation": {
                    "input_scale": "nonnegative_synthetic_counts_like",
                    "normalization": "library-size-log1p",
                    "target_sum": 10000.0,
                    "library_size_scope": "all_generated_genes_before_any_filtering",
                    "formula": "log1p(target_sum * raw_value / per_cell_library_size)",
                },
                "datasets": [
                    {
                        "id": "tiny_dynamic",
                        "design_role": {
                            "kind": "synthetic_scale_case",
                            "case_id": "tiny_dynamic_fixture",
                            "purpose": "Exercise deterministic generation in a unit test.",
                        },
                        "seed": 271760129,
                        "seed_derivation": {
                            "namespace": "reduced.scalability",
                            "stream": "augmentation",
                            "coordinates": {
                                "phase": "scalability",
                                "dataset_id": "tiny_dynamic",
                                "augmentation_contract": "scale_stress_v1",
                            },
                        },
                        "genes": 24,
                        "tfs": 5,
                        "groups": [
                            {"id": "immature", "cells": 7},
                            {"id": "mature", "cells": 3},
                        ],
                        "network_mode": "dynamic",
                        "active_tf_fraction": 0.8,
                        "regulated_target_fraction": 0.8,
                        "low_detection_target_fraction": 0.25,
                        "regulators_per_target": 2,
                        "rewiring_fraction": 0.5,
                        "base_dropout_probability": 0.1,
                        "low_detection_dropout_probability": 0.8,
                        "library_size_log_sd": 0.2,
                        "latent_factors": 2,
                        "noise_sd": 0.1,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _load_request(name: str) -> dict[str, object]:
    return json.loads(
        (REPO_ROOT / f"experiments/v1/simulations/scale_stress/{name}.json").read_text(
            encoding="utf-8"
        )
    )


def _write_request(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_scale_stress_panels_declare_closed_public_scale_grids() -> None:
    full = validate_scale_stress_request(
        REPO_ROOT,
        REPO_ROOT / "experiments/v1/simulations/scale_stress/scale-full.json",
    )
    reduced = validate_scale_stress_request(
        REPO_ROOT,
        REPO_ROOT / "experiments/v1/simulations/scale_stress/scale-reduced.json",
    )

    assert full["scale_grid"] == {
        "provenance": "author_defined_public_synthetic_grid",
        "cells": [1000, 5000, 10000],
        "genes": [5000, 20000, 25000],
        "transcription_factors": [250, 1250, 1500],
    }
    assert reduced["scale_grid"] == {
        "provenance": "author_defined_public_synthetic_grid",
        "cells": [1000, 3000],
        "genes": [5000, 10000],
        "transcription_factors": [250, 500],
    }
    assert all(
        dataset["design_role"]["kind"] in {"synthetic_scale_case", "synthetic_control"}
        for request in (full, reduced)
        for dataset in request["datasets"]
    )


def test_scale_stress_request_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    request = _request(real_parent / "request.json")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(BenchmarkConfigError, match="contains a symlink"):
        validate_scale_stress_request(REPO_ROOT, linked_parent / request.name)


def test_scale_stress_grid_rejects_a_stale_axis(tmp_path: Path) -> None:
    request = _load_request("scale-reduced")
    request["scale_grid"]["genes"] = [5000]

    with pytest.raises(BenchmarkConfigError, match="must exactly enumerate"):
        validate_scale_stress_request(REPO_ROOT, _write_request(tmp_path / "request.json", request))


def test_scale_stress_binding_rejects_preparation_contract_drift(tmp_path: Path) -> None:
    request = _load_request("scale-reduced")
    request["expression_preparation"]["target_sum"] = 5000.0

    with pytest.raises(BenchmarkConfigError, match="Invalid scale-stress panel request"):
        validate_scale_stress_request(REPO_ROOT, _write_request(tmp_path / "request.json", request))


def test_scale_stress_grid_rejects_dataset_shape_drift(tmp_path: Path) -> None:
    request = _load_request("scale-reduced")
    request["datasets"][0]["genes"] += 1

    with pytest.raises(BenchmarkConfigError, match="must exactly enumerate"):
        validate_scale_stress_request(REPO_ROOT, _write_request(tmp_path / "request.json", request))


def test_scale_stress_rejects_duplicate_design_roles(tmp_path: Path) -> None:
    request = _load_request("scale-full")
    request["datasets"][1]["design_role"] = request["datasets"][0]["design_role"]

    with pytest.raises(BenchmarkConfigError, match="duplicate design-role identities"):
        validate_scale_stress_request(REPO_ROOT, _write_request(tmp_path / "request.json", request))


def test_scale_stress_normalization_matches_library_size_log1p_formula(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.float32"
    raw = np.memmap(raw_path, dtype=np.float32, mode="w+", shape=(2, 2))
    try:
        raw[:] = np.asarray([[1.0, 0.0], [3.0, 2.0]], dtype=np.float32)
        library_sizes, maximum_error = scale_stress._write_normalized_expression(
            tmp_path / "expression.tsv",
            raw_expression=raw,
            gene_names=("gene_1", "gene_2"),
            cells=("cell_1", "cell_2"),
            target_sum=10000.0,
        )
    finally:
        raw.flush()
        raw._mmap.close()  # noqa: SLF001 - this test owns the memmap

    lines = (tmp_path / "expression.tsv").read_text(encoding="utf-8").splitlines()
    observed = np.asarray(
        [[float(value) for value in line.split("\t")[1:]] for line in lines[1:]],
        dtype=np.float64,
    )
    expected = np.log1p(np.asarray([[2500.0, 0.0], [7500.0, 10000.0]]))
    np.testing.assert_array_equal(library_sizes, np.asarray([4.0, 2.0]))
    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)
    assert maximum_error == 0.0


def test_scale_stress_generator_is_deterministic_and_andrea_compatible(tmp_path: Path) -> None:
    request = _request(tmp_path / "request.json")
    first = generate_scale_stress_panel(
        repo_root=REPO_ROOT, request_path=request, output_root=tmp_path / "first"
    )
    second = generate_scale_stress_panel(
        repo_root=REPO_ROOT, request_path=request, output_root=tmp_path / "second"
    )

    first_root = first.parent
    second_root = second.parent
    first_files = sorted(
        path.relative_to(first_root) for path in first_root.rglob("*") if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second_root) for path in second_root.rglob("*") if path.is_file()
    )
    assert first_files == second_files
    for relative in first_files:
        assert (first_root / relative).read_bytes() == (second_root / relative).read_bytes()

    datasets = datasets_from_benchmark(first, "tiny_generation")
    assert [(item["dataset_id"], item["seed"]) for item in datasets] == [
        ("tiny_dynamic", 271760129)
    ]
    dataset_root = first_root / "datasets/tiny_dynamic"
    expression_lines = (dataset_root / "expression.tsv").read_text(encoding="utf-8").splitlines()
    assert len(expression_lines) == 25
    assert len(expression_lines[0].split("\t")) == 11
    expression = np.asarray(
        [[float(value) for value in line.split("\t")[1:]] for line in expression_lines[1:]],
        dtype=np.float64,
    )
    assert np.all(expression >= 0)
    np.testing.assert_allclose(
        np.sum(np.expm1(expression), axis=0),
        np.full(10, 10000.0),
        rtol=1e-12,
        atol=1e-9,
    )
    network = (dataset_root / "truth/networks.csv").read_text(encoding="utf-8")
    assert ",group:immature" in network
    assert ",group:mature" in network
    assert ",global" in network
    provenance = json.loads(
        (dataset_root / "generator-provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["root_seed"] == 271760129
    assert provenance["seed_derivation"]["stream"] == "augmentation"
    assert set(provenance["effective_component_seeds"]) == {
        "groups",
        "regulatory_model",
        "expression",
        "dropout",
    }
    assert all(
        record["seed_derivation"]["stream"] == "augmentation"
        and record["seed_derivation"]["coordinates"]["component"] == component
        for component, record in provenance["effective_component_seeds"].items()
    )
    preparation = provenance["expression_preparation"]
    assert preparation["normalization"] == "library-size-log1p"
    assert preparation["target_sum"] == 10000.0
    assert preparation["library_size_scope"] == "all_generated_genes_before_any_filtering"
    assert preparation["validation"]["positive_finite_library_sizes"] is True
    assert preparation["validation"]["maximum_absolute_linear_scale_sum_error"] < 1e-9
    assert preparation["working_storage"] == {
        "kind": "temporary_disk_backed_float32",
        "retained": False,
    }
    benchmark_manifest = json.loads(first.read_text(encoding="utf-8"))
    assert benchmark_manifest["inputs"]["expression_preparation"] == {
        key: preparation[key]
        for key in ("input_scale", "normalization", "target_sum", "library_size_scope", "formula")
    }
    assert not list(first_root.rglob(".scale-stress-raw-expression-*"))

    andrea_root = REPO_ROOT.parent / "ANDREA/andrea"
    manifests = (
        (
            dataset_root / "dataset-manifest.json",
            andrea_root / "catalog_inference_tools/schemas/dataset-manifest.schema.json",
        ),
        (
            dataset_root / "ground-truth-manifest.json",
            andrea_root / "catalog_simulation_data_tools/schemas/ground-truth-manifest.schema.json",
        ),
        (
            first,
            andrea_root / "catalog_simulation_data_tools/schemas/benchmark-manifest.schema.json",
        ),
    )
    for manifest_path, schema_path in manifests:
        Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )


def test_scale_stress_generator_removes_disk_backed_expression_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path / "request.json")

    def fail_normalization(*_args: object, **_kwargs: object) -> tuple[np.ndarray, float]:
        raise RuntimeError("forced normalization failure")

    monkeypatch.setattr(scale_stress, "_write_normalized_expression", fail_normalization)
    with pytest.raises(RuntimeError, match="forced normalization failure"):
        generate_scale_stress_panel(
            repo_root=REPO_ROOT,
            request_path=request,
            output_root=tmp_path / "failed",
        )

    assert not list(tmp_path.rglob(".scale-stress-raw-expression-*"))
