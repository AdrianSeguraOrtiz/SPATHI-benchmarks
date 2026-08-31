from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def frozen_spathi_params() -> dict[str, Any]:
    return {
        "weight_mode": "cell-distance-group-anchored",
        "distance_space": "pca",
        "n_components": 3,
        "distance_standardization": "none",
        "pca_svd_solver": "auto",
        "distance_metric": "euclidean",
        "kernel": "gaussian",
        "bandwidth": "auto",
        "group_size_correction": "cap-to-target",
        "tree_method": "extra-trees",
        "n_estimators": 25,
        "max_features": 1.0,
        "min_samples_leaf": 1,
        "max_depth": None,
        "bootstrap": None,
        "random_seed": 1729,
    }


@pytest.fixture
def external_tool_inputs(
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
) -> dict[str, Path]:
    expression = tmp_path / "expression.tsv"
    expression.write_text("gene\tcell_1\nTF1\t1.0\n", encoding="utf-8")
    params = tmp_path / "params.json"
    params.write_text(json.dumps(frozen_spathi_params), encoding="utf-8")
    (tmp_path / "execution.json").write_text(
        json.dumps({"mode": "group_native"}),
        encoding="utf-8",
    )
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "tf_list.txt").write_text("TF1\n", encoding="utf-8")
    (extra / "groups.tsv").write_text(
        "column\tcluster\ncell_1\tgroup_a\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    return {
        "expression": expression,
        "params": params,
        "extra": extra,
        "output_dir": output_dir,
    }
