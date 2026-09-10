from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import spathi_bench.convergence_execution as convergence_execution
from spathi_bench.runtime import CommandExecutionError

REPO_ROOT = Path(__file__).resolve().parents[1]
SPATHI_ROOT = REPO_ROOT.parent / "SPATHI"


def _model() -> dict[str, object]:
    return {
        "weight_mode": "cell-distance-group-anchored",
        "distance_space": "pca",
        "n_components": 3,
        "distance_standardization": "none",
        "pca_svd_solver": "auto",
        "distance_metric": "euclidean",
        "kernel": "gaussian",
        "bandwidth": "auto",
        "bandwidth_scale": 1.0,
        "group_size_correction": "cap-to-target",
        "tree_method": "extra-trees",
        "n_estimators": 9,
        "target_eligibility": "all",
        "min_target_detected_cells": 2,
        "min_target_detected_fraction": 0.01,
        "min_target_weighted_detected_fraction": 0.01,
        "min_target_weighted_detected_ess": 1.0,
        "adaptive_trees": False,
        "adaptive_min_estimators": 2,
        "adaptive_tree_step": 1,
        "adaptive_tolerance": 0.01,
        "adaptive_patience": 1,
        "max_features": 1.0,
        "min_samples_leaf": 1,
        "max_depth": None,
        "bootstrap": None,
        "random_seed": 1729,
    }


def test_container_installs_the_adapter_under_its_importable_module_name() -> None:
    dockerfile = (REPO_ROOT / "containers/spathi/Dockerfile").read_text(encoding="utf-8")

    assert "COPY containers/spathi/run_tool.py /usr/local/bin/run_tool.py" in dockerfile
    assert 'ENTRYPOINT ["python", "/usr/local/bin/run_tool.py"]' in dockerfile


def test_physical_command_failure_preserves_its_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_log = tmp_path / "temporary/physical.log"
    durable_log = tmp_path / "logs/physical.log"
    command = ["example", "--flag"]

    def fail(
        observed_command: list[str],
        *,
        cwd: Path,
        log_path: Path,
        environment: dict[str, str] | None = None,
    ) -> None:
        assert observed_command == command
        assert cwd == tmp_path
        assert environment is None
        log_path.parent.mkdir(parents=True)
        log_path.write_text("durable diagnostic\n", encoding="utf-8")
        raise CommandExecutionError(
            command=observed_command,
            return_code=17,
            log_path=log_path,
        )

    monkeypatch.setattr(convergence_execution, "stream_command", fail)

    with pytest.raises(CommandExecutionError) as captured:
        convergence_execution._run_physical_command(
            command,
            cwd=tmp_path,
            temporary_log=temporary_log,
            durable_log=durable_log,
        )

    assert captured.value.return_code == 17
    assert captured.value.log_path == durable_log.resolve()
    assert durable_log.read_text(encoding="utf-8") == "durable diagnostic\n"


def test_runner_fits_one_forest_and_publishes_four_canonical_prefixes(
    tmp_path: Path,
) -> None:
    expression = tmp_path / "expression.tsv"
    expression.write_text(
        "gene\tc1\tc2\tc3\tc4\nTF1\t1\t2\t3\t4\nTF2\t0\t1\t1\t3\nG1\t1\t3\t2\t6\n",
        encoding="utf-8",
    )
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "tf_list.txt").write_text("TF1\nTF2\n", encoding="utf-8")
    (extra / "groups.tsv").write_text(
        "cell\tcluster\nc1\tA\nc2\tA\nc3\tB\nc4\tB\n",
        encoding="utf-8",
    )
    params = tmp_path / "params.json"
    prefixes = [
        {
            "run_id": f"prefix-{count}",
            "configuration_id": f"trees-{count}",
            "configuration_sha256": f"{index + 1}" * 64,
            "n_estimators": count,
        }
        for index, count in enumerate((3, 5, 7, 9))
    ]
    params.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "physical_run_id": "physical-parent-seed",
                "parent_configuration_id": "physical-parent",
                "algorithm_seed": 1729,
                "model": _model(),
                "logical_prefixes": prefixes,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "execution.json").write_text(json.dumps({"mode": "group_native"}), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    subprocess.run(
        [
            str(SPATHI_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "containers/spathi/run_convergence.py"),
            "--input",
            str(expression),
            "--params",
            str(params),
            "--extra",
            str(extra),
            "--output-dir",
            str(output),
            "--threads",
            "2",
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(SPATHI_ROOT / "src")},
    )

    manifest = json.loads((output / "fanout-manifest.json").read_text(encoding="utf-8"))
    assert manifest["physical_fits"] == 1
    assert manifest["logical_results"] == 4
    assert manifest["estimator_counts"] == [3, 5, 7, 9]
    for prefix in prefixes:
        directory = output / "prefixes" / str(prefix["n_estimators"])
        assert {
            "network.csv",
            "cell_weights.tsv.gz",
            "skipped_targets.tsv",
            "model_diagnostics.tsv.gz",
            "weight_diagnostics.tsv",
            "target_eligibility.tsv.gz",
            "run_metadata.json",
        } == {path.name for path in directory.iterdir()}
        assert (
            (directory / "network.csv")
            .read_text(encoding="utf-8")
            .startswith("source,target,score,sign,evidence,context\n")
        )
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        assert metadata["run_id"] == prefix["run_id"]
        assert metadata["n_estimators"] == prefix["n_estimators"]

        # The fan-out is an execution optimization only: every logical network
        # must be byte-identical to an ordinary independent SPATHI run.
        ordinary_root = tmp_path / f"ordinary-{prefix['n_estimators']}"
        request = ordinary_root / "request"
        ordinary_output = ordinary_root / "output"
        request.mkdir(parents=True)
        ordinary_output.mkdir()
        ordinary_model = _model()
        ordinary_model["n_estimators"] = prefix["n_estimators"]
        (request / "params.json").write_text(
            json.dumps(ordinary_model),
            encoding="utf-8",
        )
        (request / "execution.json").write_text(
            json.dumps({"mode": "group_native"}),
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(SPATHI_ROOT / ".venv/bin/python"),
                str(REPO_ROOT / "containers/spathi/run_tool.py"),
                "--input",
                str(expression),
                "--params",
                str(request / "params.json"),
                "--extra",
                str(extra),
                "--output-dir",
                str(ordinary_output),
                "--threads",
                "2",
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": str(SPATHI_ROOT / "src")},
        )
        assert (directory / "network.csv").read_bytes() == (
            ordinary_output / "network.csv"
        ).read_bytes()
