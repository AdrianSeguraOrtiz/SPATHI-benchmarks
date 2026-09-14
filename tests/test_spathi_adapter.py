from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
import types
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_adapter() -> types.ModuleType:
    path = REPO_ROOT / "containers" / "spathi" / "run_tool.py"
    spec = importlib.util.spec_from_file_location("spathi_bench_run_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def adapter() -> types.ModuleType:
    return _load_adapter()


def _namespace(inputs: dict[str, Path], *, threads: int = 2) -> Namespace:
    return Namespace(
        input=inputs["expression"],
        params=inputs["params"],
        extra=inputs["extra"],
        output_dir=inputs["output_dir"],
        threads=threads,
    )


def _write_audit_sidecars(output_dir: Path, *, status: str = "complete") -> None:
    for name, header in (
        ("model_diagnostics.tsv.gz", "target_group\ttarget\tstatus\n"),
    ):
        with gzip.open(output_dir / name, "wt", encoding="utf-8", newline="") as handle:
            handle.write(header)
    (output_dir / "run_metadata.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )


def test_adapter_accepts_the_standardized_cli_contract(
    adapter: types.ModuleType,
    external_tool_inputs: dict[str, Path],
) -> None:
    argv = [
        "--input",
        str(external_tool_inputs["expression"]),
        "--params",
        str(external_tool_inputs["params"]),
        "--extra",
        str(external_tool_inputs["extra"]),
        "--output-dir",
        str(external_tool_inputs["output_dir"]),
        "--threads",
        "3",
    ]

    args = adapter.build_parser().parse_args(argv)

    assert args.input == external_tool_inputs["expression"]
    assert args.params == external_tool_inputs["params"]
    assert args.extra == external_tool_inputs["extra"]
    assert args.output_dir == external_tool_inputs["output_dir"]
    assert args.threads == 3


def test_progress_publication_is_atomic_under_concurrent_updates(
    adapter: types.ModuleType,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"

    def publish(percent: int) -> None:
        adapter._write_progress(
            output_dir,
            percent=percent,
            status="running",
            phase="model_inference",
            message=f"Completed {percent}",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(publish, range(10, 90)))

    payload = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert payload["percent"] in range(10, 90)
    assert payload["message"] == f"Completed {payload['percent']}"
    assert (output_dir / "progress.json").stat().st_mode & 0o777 == 0o644
    assert not list(output_dir.glob(".progress.json.*.tmp"))


@pytest.mark.parametrize("threads", ["0", "-1"])
def test_adapter_cli_rejects_nonpositive_thread_counts(
    adapter: types.ModuleType,
    external_tool_inputs: dict[str, Path],
    threads: str,
) -> None:
    with pytest.raises(SystemExit):
        adapter.build_parser().parse_args(
            [
                "--input",
                str(external_tool_inputs["expression"]),
                "--params",
                str(external_tool_inputs["params"]),
                "--extra",
                str(external_tool_inputs["extra"]),
                "--output-dir",
                str(external_tool_inputs["output_dir"]),
                "--threads",
                threads,
            ]
        )


def test_adapter_requires_all_frozen_params_and_rejects_unknown_ones(
    adapter: types.ModuleType,
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
) -> None:
    params_path = tmp_path / "params.json"
    incomplete = dict(frozen_spathi_params)
    incomplete.pop("kernel")
    params_path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="missing frozen SPATHI parameters: kernel"):
        adapter._validated_params(params_path)

    unknown = dict(frozen_spathi_params, surprise=True)
    params_path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported SPATHI parameters: surprise"):
        adapter._validated_params(params_path)


def test_adapter_accepts_bound_scalability_parameters_without_renaming(
    adapter: types.ModuleType,
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
) -> None:
    params = {
        **frozen_spathi_params,
        "n_estimators": 250,
    }
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    assert adapter._validated_params(params_path) == (params, {"method": "all"})


def test_adapter_resolves_the_frozen_policy_from_observed_group_count(
    adapter: types.ModuleType,
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
) -> None:
    multi = {key: value for key, value in frozen_spathi_params.items() if key != "random_seed"}
    single = {
        **multi,
        "weight_mode": "cell-distance",
        "distance_metric": "euclidean",
        "group_size_correction": "none",
    }
    conditional = {"weight_mode", "distance_metric", "group_size_correction"}
    request = {
        "group_policy": {
            "routing_parameter": "group_count",
            "shared_parameters": sorted(set(multi) - conditional),
            "branches": {
                "single_group": {
                    "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                    "parameters": single,
                },
                "multi_group": {
                    "condition": {"minimum_group_count": 2},
                    "parameters": multi,
                },
            },
        },
        "random_seed": 8675309,
    }
    path = tmp_path / "params.json"
    path.write_text(json.dumps(request), encoding="utf-8")

    resolved_single, single_scope = adapter._validated_params(path, group_count=1)
    resolved_multi, multi_scope = adapter._validated_params(path, group_count=3)

    assert resolved_single == {**single, "random_seed": 8675309}
    assert resolved_multi == {**multi, "random_seed": 8675309}
    assert single_scope == multi_scope == {"method": "all"}


def test_high_tf_scout_hashes_only_gene_ids_and_is_order_independent(
    adapter: types.ModuleType,
    tmp_path: Path,
) -> None:
    genes = [f"G{index:04d}" for index in range(300)]
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    first.write_text(
        "gene\tc1\tc2\n" + "".join(f"{gene}\t1\t2\n" for gene in genes),
        encoding="utf-8",
    )
    second.write_text(
        "gene\tc1\tc2\n" + "".join(f"{gene}\t999\t-3\n" for gene in reversed(genes)),
        encoding="utf-8",
    )
    scope = {
        "method": "sha256_gene_id",
        "size": 256,
        "salt": "spathi-calibration-high-tf-v1",
        "dataset_id": "dataset-a",
    }

    first_targets = adapter._hashed_target_list(
        first, tmp_path / "first-targets.txt", scope, random_seed=1729
    )
    second_targets = adapter._hashed_target_list(
        second, tmp_path / "second-targets.txt", scope, random_seed=1729
    )

    assert first_targets is not None and second_targets is not None
    assert first_targets.read_text(encoding="utf-8") == second_targets.read_text(encoding="utf-8")
    assert len(first_targets.read_text(encoding="utf-8").splitlines()) == 256

    other_scope = {**scope, "dataset_id": "dataset-b"}
    other_targets = adapter._hashed_target_list(
        first, tmp_path / "other-targets.txt", other_scope, random_seed=1729
    )
    other_seed_targets = adapter._hashed_target_list(
        first, tmp_path / "other-seed-targets.txt", scope, random_seed=1730
    )
    assert other_targets is not None and other_seed_targets is not None
    assert first_targets.read_bytes() != other_targets.read_bytes()
    assert first_targets.read_bytes() != other_seed_targets.read_bytes()


def test_adapter_rejects_a_policy_with_divergent_shared_parameters(
    adapter: types.ModuleType,
    tmp_path: Path,
    frozen_spathi_params: dict[str, Any],
) -> None:
    multi = {key: value for key, value in frozen_spathi_params.items() if key != "random_seed"}
    single = {
        **multi,
        "weight_mode": "cell-distance",
        "distance_metric": "euclidean",
        "group_size_correction": "none",
        "kernel": "exponential",
    }
    request = {
        "group_policy": {
            "routing_parameter": "group_count",
            "shared_parameters": sorted(
                set(multi) - {"weight_mode", "distance_metric", "group_size_correction"}
            ),
            "branches": {
                "single_group": {
                    "condition": {"minimum_group_count": 1, "maximum_group_count": 1},
                    "parameters": single,
                },
                "multi_group": {
                    "condition": {"minimum_group_count": 2},
                    "parameters": multi,
                },
            },
        },
        "random_seed": 123,
    }
    path = tmp_path / "params.json"
    path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ValueError, match="shared parameter"):
        adapter._validated_params(path, group_count=1)


def test_adapter_requires_group_native_execution(
    adapter: types.ModuleType,
    tmp_path: Path,
) -> None:
    execution = tmp_path / "execution.json"
    execution.write_text(json.dumps({"mode": "group_emulated"}), encoding="utf-8")

    with pytest.raises(ValueError, match="requires execution.mode=group_native"):
        adapter._validate_execution(execution)


def test_adapter_rejects_missing_required_extra(
    adapter: types.ModuleType,
    external_tool_inputs: dict[str, Path],
) -> None:
    (external_tool_inputs["extra"] / "groups.tsv").unlink()

    with pytest.raises(ValueError, match="required groups input does not exist"):
        adapter._validate_files(_namespace(external_tool_inputs))


def test_network_contract_accepts_header_only_output_and_requires_exact_order(
    adapter: types.ModuleType,
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.csv"
    valid.write_text(
        "source,target,score,sign,evidence,context\n",
        encoding="utf-8",
    )
    adapter._validate_network_header(valid)

    invalid = tmp_path / "invalid.csv"
    invalid.write_text(
        "target,source,score,sign,evidence,context\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Unexpected SPATHI network.csv columns"):
        adapter._validate_network_header(invalid)


def test_run_forwards_contract_and_atomically_publishes_network(
    adapter: types.ModuleType,
    external_tool_inputs: dict[str, Path],
    frozen_spathi_params: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    progress_writes: list[dict[str, Any]] = []

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.output_dir = kwargs["output_dir"]

    def fake_infer(
        config: FakeConfig,
        *,
        progress_callback: Any,
        checkpoint: bool,
    ) -> Any:
        captured["checkpoint"] = checkpoint
        for phase, completed_models, total_models in (
            ("validating_inputs", 0, None),
            ("building_representation", 0, None),
            ("model_inference", 5, 10),
            ("writing_outputs", 10, 10),
            ("complete", 10, 10),
        ):
            progress_callback(
                types.SimpleNamespace(
                    phase=phase,
                    message=f"Progress: {phase}",
                    completed_models=completed_models,
                    total_models=total_models,
                )
            )
        config.output_dir.mkdir()
        network = config.output_dir / "network.csv"
        with network.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(adapter.EXPECTED_NETWORK_COLUMNS)
        _write_audit_sidecars(config.output_dir)
        return types.SimpleNamespace(network_path=network, warnings=("bounded warning",))

    spathi_package = types.ModuleType("spathi")
    spathi_package.__path__ = []
    spathi_package.SpathiConfig = FakeConfig
    spathi_package.infer = fake_infer
    monkeypatch.setitem(sys.modules, "spathi", spathi_package)
    original_write_progress = adapter._write_progress

    def tracked_write_progress(output_dir: Path, **kwargs: Any) -> None:
        progress_writes.append(dict(kwargs))
        original_write_progress(output_dir, **kwargs)

    monkeypatch.setattr(adapter, "_write_progress", tracked_write_progress)

    adapter.run(_namespace(external_tool_inputs, threads=2))

    spathi_output = captured.pop("output_dir")
    assert isinstance(spathi_output, Path)
    assert spathi_output.name == "spathi-run"
    assert spathi_output.parent.name.startswith("spathi-adapter-")
    assert not spathi_output.exists()
    assert captured == {
        "expression": external_tool_inputs["expression"],
        "tf_list": external_tool_inputs["extra"] / "tf_list.txt",
        "groups": external_tool_inputs["extra"] / "groups.tsv",
        "target_list": None,
        "threads": 2,
        "report": False,
        **frozen_spathi_params,
        "checkpoint": False,
    }
    network = external_tool_inputs["output_dir"] / "network.csv"
    assert network.read_text(encoding="utf-8").startswith("source,target,score")
    assert network.stat().st_mode & 0o777 == 0o644
    for name in adapter.AUDIT_SIDECARS:
        assert (external_tool_inputs["output_dir"] / name).is_file()
        assert (external_tool_inputs["output_dir"] / name).stat().st_mode & 0o777 == 0o644
    assert not list(external_tool_inputs["output_dir"].glob(".*.tmp"))
    progress = json.loads(
        (external_tool_inputs["output_dir"] / "progress.json").read_text(encoding="utf-8")
    )
    assert (external_tool_inputs["output_dir"] / "progress.json").stat().st_mode & 0o777 == 0o644
    assert progress == {
        "message": "Published ANDREA-compatible network.csv and SPATHI audit sidecars",
        "percent": 100,
        "phase": "done",
        "status": "completed",
        "warnings": ["bounded warning"],
    }
    assert [write["phase"] for write in progress_writes] == [
        "validating_inputs",
        "validating_inputs",
        "building_representation",
        "model_inference",
        "writing_outputs",
        "complete",
        "done",
    ]
    assert [write["percent"] for write in progress_writes] == [5, 10, 15, 62, 92, 98, 100]


def test_adapter_removes_temporary_network_when_copy_fails(
    adapter: types.ModuleType,
    external_tool_inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.output_dir = kwargs["output_dir"]

    def fake_infer(config: FakeConfig, **_: Any) -> Any:
        config.output_dir.mkdir()
        network = config.output_dir / "network.csv"
        network.write_text(
            "source,target,score,sign,evidence,context\n",
            encoding="utf-8",
        )
        _write_audit_sidecars(config.output_dir)
        return types.SimpleNamespace(network_path=network, warnings=())

    spathi_package = types.ModuleType("spathi")
    spathi_package.__path__ = []
    spathi_package.SpathiConfig = FakeConfig
    spathi_package.infer = fake_infer
    monkeypatch.setitem(sys.modules, "spathi", spathi_package)

    def fail_copy(_source: Path, destination: Path) -> None:
        destination.write_text("partial", encoding="utf-8")
        raise OSError("copy failed")

    monkeypatch.setattr(adapter.shutil, "copyfile", fail_copy)

    with pytest.raises(OSError, match="copy failed"):
        adapter.run(_namespace(external_tool_inputs))

    assert not list(external_tool_inputs["output_dir"].glob(".*.tmp"))
    assert not (external_tool_inputs["output_dir"] / "network.csv").exists()
    assert all(
        not (external_tool_inputs["output_dir"] / name).exists() for name in adapter.AUDIT_SIDECARS
    )


def test_main_records_adapter_failures(
    adapter: types.ModuleType,
    external_tool_inputs: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    (external_tool_inputs["extra"] / "tf_list.txt").unlink()
    argv = [
        "--input",
        str(external_tool_inputs["expression"]),
        "--params",
        str(external_tool_inputs["params"]),
        "--extra",
        str(external_tool_inputs["extra"]),
        "--output-dir",
        str(external_tool_inputs["output_dir"]),
        "--threads",
        "2",
    ]

    assert adapter.main(argv) == 2
    progress = json.loads(
        (external_tool_inputs["output_dir"] / "progress.json").read_text(encoding="utf-8")
    )
    assert (external_tool_inputs["output_dir"] / "progress.json").stat().st_mode & 0o777 == 0o644
    assert progress["status"] == "failed"
    assert progress["percent"] == 100
    assert "required tf_list input does not exist" in progress["message"]
    assert "SPATHI adapter failed" in capsys.readouterr().err
    failure_root = external_tool_inputs["output_dir"] / adapter.FAILURE_DIAGNOSTICS_DIR
    failure = json.loads((failure_root / adapter.FAILURE_MANIFEST).read_text(encoding="utf-8"))
    assert failure == {
        "artifact_type": "spathi_adapter_failure_diagnostics",
        "error": {
            "message": (
                "required tf_list input does not exist: "
                f"{external_tool_inputs['extra'] / 'tf_list.txt'}"
            ),
            "type": "ValueError",
        },
        "files": [],
        "model_diagnostics": None,
        "schema_version": "1.0",
        "status": "failed",
    }
    assert not (external_tool_inputs["output_dir"] / "network.csv").exists()
    assert all(
        not (external_tool_inputs["output_dir"] / name).exists() for name in adapter.AUDIT_SIDECARS
    )


def test_failed_spathi_run_preserves_diagnostics_without_publishing_success(
    adapter: types.ModuleType,
    external_tool_inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Path] = {}

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.output_dir = kwargs["output_dir"]
            captured["scratch_output"] = self.output_dir

    def fake_infer(config: FakeConfig, **_: Any) -> Any:
        config.output_dir.mkdir()
        with gzip.open(
            config.output_dir / "model_diagnostics.tsv.gz",
            "wt",
            encoding="utf-8",
            newline="",
        ) as handle:
            handle.write("target_group\ttarget\tstatus\tmessage\n")
            handle.write("group_a\tG1\tinvalid_feature_importances\tnon-finite values\n")
            handle.write("group_a\tG2\tinvalid_feature_importances\tnon-finite values\n")
            handle.write("group_a\tG3\ttrained\t\n")
        (config.output_dir / "run_metadata.json").write_text(
            json.dumps({"status": "failed", "models": {"failed": 2}}),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"2 models failed; inspect {config.output_dir / 'model_diagnostics.tsv.gz'}"
        )

    spathi_package = types.ModuleType("spathi")
    spathi_package.__path__ = []
    spathi_package.SpathiConfig = FakeConfig
    spathi_package.infer = fake_infer
    monkeypatch.setitem(sys.modules, "spathi", spathi_package)
    args = _namespace(external_tool_inputs)
    argv = [
        "--input",
        str(args.input),
        "--params",
        str(args.params),
        "--extra",
        str(args.extra),
        "--output-dir",
        str(args.output_dir),
        "--threads",
        str(args.threads),
    ]

    assert adapter.main(argv) == 2

    scratch_output = captured["scratch_output"]
    assert not scratch_output.exists()
    failure_root = args.output_dir / adapter.FAILURE_DIAGNOSTICS_DIR
    manifest = json.loads((failure_root / adapter.FAILURE_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"] == {
        "type": "RuntimeError",
        "message": ("2 models failed; inspect failure_diagnostics/model_diagnostics.tsv.gz"),
    }
    assert manifest["model_diagnostics"] == {
        "records": 3,
        "status_counts": {"invalid_feature_importances": 2, "trained": 1},
        "messages": [
            {
                "status": "invalid_feature_importances",
                "message": "non-finite values",
                "count": 2,
            }
        ],
    }
    assert {item["path"] for item in manifest["files"]} == set(adapter.AUDIT_SIDECARS)
    for record in manifest["files"]:
        path = failure_root / record["path"]
        assert path.is_file()
        assert record["size"] == path.stat().st_size
        assert record["sha256"] == adapter._sha256_file(path)
    assert not (args.output_dir / "network.csv").exists()
    assert all(not (args.output_dir / name).exists() for name in adapter.AUDIT_SIDECARS)
    progress = json.loads((args.output_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["phase"] == "failed"
    assert progress["percent"] == 100
    assert progress["message"].endswith("durable diagnostics: failure_diagnostics/failure.json")
    stderr = capsys.readouterr().err
    assert "failure_diagnostics/failure.json" in stderr
    assert "invalid_feature_importances=2, trained=1" in stderr
    assert "spathi-adapter-" not in stderr


def test_run_refuses_to_overwrite_prior_network(
    adapter: types.ModuleType,
    external_tool_inputs: dict[str, Path],
) -> None:
    (external_tool_inputs["output_dir"] / "network.csv").write_text(
        "source,target,score,sign,evidence,context\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        adapter.run(_namespace(external_tool_inputs))


def test_adapter_requires_complete_readable_audit_sidecars(
    adapter: types.ModuleType,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    with pytest.raises(RuntimeError, match="required audit sidecars"):
        adapter._validate_audit_sidecars(output_dir)

    _write_audit_sidecars(output_dir, status="failed")
    with pytest.raises(RuntimeError, match="does not describe a complete run"):
        adapter._validate_audit_sidecars(output_dir)

    (output_dir / "run_metadata.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (output_dir / "model_diagnostics.tsv.gz").write_bytes(b"not gzip")
    with pytest.raises(RuntimeError, match="Cannot read SPATHI audit sidecar"):
        adapter._validate_audit_sidecars(output_dir)


def test_run_refuses_to_overwrite_prior_audit_sidecar(
    adapter: types.ModuleType,
    external_tool_inputs: dict[str, Path],
) -> None:
    (external_tool_inputs["output_dir"] / "run_metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="run_metadata.json"):
        adapter.run(_namespace(external_tool_inputs))
