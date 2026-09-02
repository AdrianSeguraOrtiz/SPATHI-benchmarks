from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from argparse import Namespace
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
        for phase, fraction in (
            ("validating_inputs", None),
            ("building_representation", None),
            ("model_inference", 0.5),
            ("writing_outputs", 1.0),
            ("complete", 1.0),
        ):
            progress_callback(
                types.SimpleNamespace(
                    phase=phase,
                    message=f"Progress: {phase}",
                    model_fraction=fraction,
                )
            )
        config.output_dir.mkdir()
        network = config.output_dir / "network.csv"
        with network.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(adapter.EXPECTED_NETWORK_COLUMNS)
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
        "threads": 2,
        "report": False,
        **frozen_spathi_params,
        "checkpoint": False,
    }
    network = external_tool_inputs["output_dir"] / "network.csv"
    assert network.read_text(encoding="utf-8").startswith("source,target,score")
    assert network.stat().st_mode & 0o777 == 0o644
    assert not list(external_tool_inputs["output_dir"].glob(".network.*.tmp"))
    progress = json.loads(
        (external_tool_inputs["output_dir"] / "progress.json").read_text(encoding="utf-8")
    )
    assert progress == {
        "message": "Published ANDREA-compatible network.csv",
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


def test_adapter_uses_only_the_public_spathi_api() -> None:
    source = (REPO_ROOT / "containers/spathi/run_tool.py").read_text(encoding="utf-8")

    assert "from spathi import SpathiConfig, infer" in source
    assert "spathi.pipeline" not in source


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

    assert not list(external_tool_inputs["output_dir"].glob(".network.*.tmp"))
    assert not (external_tool_inputs["output_dir"] / "network.csv").exists()


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
    assert progress["status"] == "failed"
    assert progress["percent"] == 100
    assert "required tf_list input does not exist" in progress["message"]
    assert "SPATHI adapter failed" in capsys.readouterr().err


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
