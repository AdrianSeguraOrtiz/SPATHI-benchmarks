from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "containers/spathi/run_convergence.py"


def _load_runner(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    adapter = types.ModuleType("run_tool")
    adapter._load_json_object = lambda path, _label: json.loads(path.read_text(encoding="utf-8"))
    adapter._validate_execution = lambda _path: None
    adapter._validate_files = lambda args: (
        args.extra / "tf_list.txt",
        args.extra / "groups.tsv",
        args.params.parent / "execution.json",
    )
    adapter._validated_scalar_params = lambda params: params
    monkeypatch.setitem(sys.modules, "run_tool", adapter)
    spec = importlib.util.spec_from_file_location("test_run_convergence", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fatal_status", [None, "model_fit_failed"])
def test_runner_fits_once_and_fans_out_four_exact_prefixes_or_rejects_fatal_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fatal_status: str | None,
) -> None:
    runner = _load_runner(monkeypatch)
    calls: list[tuple[int, ...]] = []

    class FakePrepared:
        weight_results = ("weight-result",)

        def iter_batches(self, *, target_batch_size: int) -> list[tuple[Any, ...]]:
            assert target_batch_size == 4
            results = []
            for count in (25, 50, 100, 250):
                inference = SimpleNamespace(
                    edges=(),
                    skipped_targets=(),
                    model_stats=(
                        (SimpleNamespace(status=fatal_status),) if fatal_status is not None else ()
                    ),
                    completed_models=3,
                    trained_models=3,
                    duration_seconds=float(count) / 100,
                )
                results.append(SimpleNamespace(n_estimators=count, inference=inference))
            return [tuple(results)]

    class FakeWriter:
        def __init__(self, directory: Path) -> None:
            self.directory = directory

        def __enter__(self) -> FakeWriter:
            (self.directory / "network.csv").write_text(
                "source,target,score,sign,evidence,context\n",
                encoding="utf-8",
            )
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def write_weights(self, _value: object) -> None:
            return None

        def write_weight_diagnostics(self, _value: object) -> None:
            return None

        def write_edges(self, edges: object) -> int:
            assert edges == ()
            return 0

        def write_skipped_targets(self, _value: object) -> None:
            return None

        def write_model_diagnostics(self, _value: object) -> None:
            return None

    spathi = types.ModuleType("spathi")
    spathi.SpathiConfig = lambda **values: SimpleNamespace(**values)

    def prepare(_config: object, *, estimator_counts: tuple[int, ...]) -> FakePrepared:
        calls.append(estimator_counts)
        return FakePrepared()

    spathi.prepare_forest_prefixes = prepare
    diagnostics = types.ModuleType("spathi.diagnostics")
    diagnostics.compute_weight_diagnostics = lambda value, *, emit_warnings: {
        "value": value,
        "emit_warnings": emit_warnings,
    }
    outputs = types.ModuleType("spathi.outputs")
    outputs.IncrementalRunWriter = FakeWriter
    inference = types.ModuleType("spathi.inference")
    inference.FATAL_MODEL_STATUSES = frozenset({"model_fit_failed", "invalid_feature_importances"})
    monkeypatch.setitem(sys.modules, "spathi", spathi)
    monkeypatch.setitem(sys.modules, "spathi.diagnostics", diagnostics)
    monkeypatch.setitem(sys.modules, "spathi.outputs", outputs)
    monkeypatch.setitem(sys.modules, "spathi.inference", inference)

    extra = tmp_path / "extra"
    extra.mkdir()
    params = tmp_path / "params.json"
    params.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "physical_run_id": "forest-candidate-01-seed-7",
                "parent_configuration_id": "candidate-01",
                "algorithm_seed": 7,
                "model": {"random_seed": 7, "n_estimators": 250},
                "logical_prefixes": [
                    {
                        "run_id": f"candidate-01__trees_{count}__seed_7",
                        "configuration_id": "candidate-01",
                        "configuration_sha256": hashlib.sha256(
                            json.dumps(
                                {"random_seed": 7, "n_estimators": count},
                                ensure_ascii=True,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("ascii")
                        ).hexdigest(),
                        "n_estimators": count,
                    }
                    for count in (25, 50, 100, 250)
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()

    arguments = Namespace(
        input=tmp_path / "expression.tsv",
        params=params,
        extra=extra,
        output_dir=output,
        threads=2,
    )
    if fatal_status is not None:
        with pytest.raises(RuntimeError, match="fatal model outcomes"):
            runner.run(arguments)
        assert not any(output.iterdir())
        return

    runner.run(arguments)

    assert calls == [(25, 50, 100, 250)]
    manifest = json.loads((output / "fanout-manifest.json").read_text(encoding="utf-8"))
    assert manifest["physical_fits"] == 1
    assert manifest["logical_results"] == 4
    assert manifest["estimator_counts"] == [25, 50, 100, 250]
    for count in manifest["estimator_counts"]:
        metadata = json.loads(
            (output / "prefixes" / str(count) / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["n_estimators"] == count
        assert metadata["physical_forest_ceiling"] == 250
        assert metadata["logical_runtime_seconds"] is None
        assert metadata["runtime_semantics"] == ("shared_physical_fit_not_attributable_to_prefix")


def test_runner_recognizes_fatal_model_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner(monkeypatch)
    counts = runner._fatal_status_counts(
        (
            SimpleNamespace(status="trained"),
            SimpleNamespace(status="model_fit_failed"),
            SimpleNamespace(status="invalid_feature_importances"),
        ),
        frozenset({"model_fit_failed", "invalid_feature_importances"}),
    )

    assert counts == {
        "model_fit_failed": 1,
        "invalid_feature_importances": 1,
    }
