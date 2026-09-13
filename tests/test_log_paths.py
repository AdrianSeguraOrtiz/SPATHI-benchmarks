from __future__ import annotations

import json
import re
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.pipeline as pipeline
from spathi_bench.artifacts import ArtifactStore, sha256_file
from spathi_bench.runtime import command_log_path

_DATASET_ID = (
    "spathi_scmultisim_dimensional_transfer_tf10_single_group_v1__"
    "scmultisim_transfer_g100_tf10_c500_single_group_reduced__r01"
)
_RUN_ID = f"joint-{_DATASET_ID}-group_only"


def _write_log(path: Path, directory: Path) -> None:
    assert path.parent == directory
    assert path.suffix == ".log"
    assert path.name.isascii()
    assert len(path.name.encode("utf-8")) <= 200
    assert re.fullmatch(r"[A-Za-z0-9._-]+", path.name)
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text("command output\n", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "command output\n"


def test_short_safe_command_log_name_is_readable(tmp_path: Path) -> None:
    path = command_log_path(tmp_path, "50", "dataset", "joint", "run")

    assert path == tmp_path / "50-dataset-joint-run.log"
    _write_log(path, tmp_path)


@pytest.mark.parametrize("stem_length", [195, 196, 197, 500])
def test_log_filename_limit_includes_extension(tmp_path: Path, stem_length: int) -> None:
    component = "a" * stem_length
    path = command_log_path(tmp_path, component)

    _write_log(path, tmp_path)
    if stem_length + len(".log") <= 200:
        assert path.name == f"{component}.log"
    else:
        assert re.search(r"-[0-9a-f]{16}\.log$", path.name)


def test_dimensional_transfer_joint_log_can_be_created(tmp_path: Path) -> None:
    parts = ("50", _DATASET_ID, "joint", _RUN_ID)
    assert len(("-".join(parts) + ".log").encode("utf-8")) > 255

    path = command_log_path(tmp_path / "logs", *parts)

    _write_log(path, tmp_path / "logs")
    assert re.search(r"-[0-9a-f]{16}\.log$", path.name)
    assert path == command_log_path(tmp_path / "logs", *parts)


@pytest.mark.parametrize(
    "component",
    ["../outside", "/absolute/name", "a\\b", "cell group", "TF-β", "细胞" * 200],
)
def test_unsafe_or_unicode_components_stay_inside_log_directory(
    tmp_path: Path,
    component: str,
) -> None:
    path = command_log_path(tmp_path / "logs", "50", component, "run")

    _write_log(path, tmp_path / "logs")
    assert re.search(r"-[0-9a-f]{16}\.log$", path.name)
    assert path == command_log_path(tmp_path / "logs", "50", component, "run")


def test_log_hash_distinguishes_original_components_after_sanitization(
    tmp_path: Path,
) -> None:
    paths = {
        command_log_path(tmp_path, "50", component)
        for component in ("a/b", "a\\b", "a b", "a_b", "a-b")
    }

    assert len(paths) == 5


def test_log_hash_includes_truncated_suffix_and_component_boundaries(tmp_path: Path) -> None:
    prefix = "dataset" * 80
    parts = [
        ("50", prefix, "joint", "run-first"),
        ("50", prefix, "joint", "run-second"),
        ("50", prefix, "joint-run", "first"),
    ]
    paths = [command_log_path(tmp_path, *components) for components in parts]

    assert len(set(paths)) == len(parts)
    assert len({path.stem.rsplit("-", 1)[-1] for path in paths}) == len(parts)
    for path in paths:
        _write_log(path, tmp_path)


def test_evaluation_task_bounds_log_name_without_changing_inference_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    run_root = tmp_path / "run"
    spec = {"dataset_id": _DATASET_ID, "run_id": _RUN_ID}
    task = {
        "dataset": {"dataset_id": _DATASET_ID},
        "inference": {"run_id": _RUN_ID},
        "scope": "joint",
        "run_report": tmp_path / "input" / "run_report.json",
        "ground_truth_manifest": tmp_path / "input" / "ground-truth-manifest.json",
        "key": store.key_for("evaluation", spec),
        "spec": spec,
    }
    original = deepcopy(task)

    def fake_andrea_command(arguments: list[str], **kwargs: Any) -> None:
        _write_log(kwargs["log_path"], run_root / "logs")
        assert arguments[arguments.index("--run-report") + 1] == str(task["run_report"])
        output = Path(arguments[arguments.index("--output-dir") + 1]) / "evaluation"
        output.mkdir(parents=True)
        (output / "evaluation_report.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "andrea_command", fake_andrea_command)
    artifact = pipeline._execute_evaluation_task(
        task,
        store=store,
        repo_root=tmp_path,
        andrea_repo=tmp_path / "andrea",
        run_root=run_root,
    )

    assert task == original
    assert artifact.key == task["key"]
    assert artifact.manifest["spec"] == spec
    assert artifact.file("output/evaluation/evaluation_report.json").is_file()


def test_analysis_export_task_bounds_log_name_without_changing_dataset_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    run_root = tmp_path / "run"
    spec = {"dataset_id": _DATASET_ID, "run_id": _RUN_ID}
    task = {
        "dataset_id": _DATASET_ID,
        "run_id": _RUN_ID,
        "scope": "joint",
        "producer_command": "evaluate-inference",
        "source_root": tmp_path / "source",
        "key": store.key_for("analysis-bundle", spec),
        "spec": spec,
    }
    original = deepcopy(task)

    def fake_contract_command(arguments: list[str], **kwargs: Any) -> None:
        _write_log(kwargs["log_path"], run_root / "logs")
        assert arguments[arguments.index("--dataset-id") + 1] == _DATASET_ID
        archive = Path(arguments[arguments.index("--output") + 1])
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("evaluation_report.json", "{}\n")
        manifest = {
            "schema_version": "1.0",
            "status": "complete",
            "bundle": {"id": "analysis"},
            "archive": {
                "filename": archive.name,
                "sha256": sha256_file(archive),
                "size": archive.stat().st_size,
            },
        }
        archive.with_suffix(".zip.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    monkeypatch.setattr(pipeline, "andrea_contract_command", fake_contract_command)
    artifact = pipeline._execute_analysis_export_task(
        task,
        store=store,
        repo_root=tmp_path,
        andrea_repo=tmp_path / "andrea",
        run_root=run_root,
    )

    assert task == original
    assert artifact.key == task["key"]
    assert artifact.manifest["spec"] == spec
    assert artifact.file("analysis.zip").is_file()
