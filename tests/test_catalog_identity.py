from __future__ import annotations

import json
from pathlib import Path

import pytest

from spathi_bench.catalog_identity import (
    catalog_definition_identity,
    verify_catalog_definitions,
)
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.study_store import StudyStore
from spathi_bench.study_workflow import bind_catalog_definitions


def _definition(root: Path, kind: str, identifier: str, payload: dict[str, object]) -> Path:
    branch, filename = (
        ("catalog_inference_tools/tools", "toolspec.json")
        if kind == "tool"
        else ("catalog_simulation_data_tools/simulators", "simulatorspec.json")
    )
    path = root / "andrea" / branch / identifier / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _task(
    task_id: str, kind: str, spec: dict[str, object], dependencies: list[str] | None = None
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "phase": "comparison",
        "stage": "test",
        "kind": kind,
        "scientific_digest": "0" * 64,
        "executor_contract": "test-v1",
        "spec": spec,
        "dependencies": dependencies or [],
    }


def test_binding_changes_only_tasks_that_consume_changed_catalog_definition(
    tmp_path: Path,
) -> None:
    andrea = tmp_path / "ANDREA"
    _definition(
        andrea,
        "simulator",
        "dyngen",
        {"id": "dyngen", "docker_image": "example/dyngen:1"},
    )
    toolspec = _definition(
        andrea,
        "tool",
        "genie3",
        {"id": "genie3", "docker_image": "example/genie3:1"},
    )
    tasks = [
        _task(
            "generate",
            "generate_dataset",
            {"dataset": {"generator": "dyngen"}},
        ),
        _task(
            "infer",
            "comparison_dataset",
            {"routes": [{"tool_id": "genie3"}, {"tool_id": "spathi"}]},
            ["generate"],
        ),
        _task("select", "comparison_statistics", {}, ["infer"]),
    ]

    before = {task["task_id"]: task for task in bind_catalog_definitions(tasks, andrea)}
    toolspec.write_text(
        json.dumps(
            {
                "id": "genie3",
                "docker_image": "example/genie3:1",
                "description": "changed contract, same image",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    after = {task["task_id"]: task for task in bind_catalog_definitions(tasks, andrea)}

    assert before["generate"]["scientific_digest"] == after["generate"]["scientific_digest"]
    assert before["infer"]["scientific_digest"] != after["infer"]["scientific_digest"]
    assert before["select"]["scientific_digest"] == after["select"]["scientific_digest"]
    assert before["infer"]["task_id"] == after["infer"]["task_id"]
    assert before["infer"]["spec"]["catalog_definitions"].keys() == {"tool:genie3"}

    with StudyStore(tmp_path / "study.sqlite") as store:
        store.initialize(
            {
                "study_id": "catalog-resume",
                "profile": "smoke",
                "design_digest": "design",
                "plan_topology_digest": "topology",
                "scientific_design": {},
            }
        )
        store.record_revision(
            {
                "benchmark_tree": "benchmark",
                "spathi_scientific_tree": "spathi",
                "andrea_scientific_tree": "andrea-before",
            }
        )
        store.add_tasks(before.values())
        for task_id in ("generate", "infer", "select"):
            store.start(task_id)
            store.finish(task_id, {"summary": {}})
        planning = store.add_tasks(after.values())
        assert planning == {"inserted": 0, "reused": 2, "invalidated": 2}
        assert store.task("generate").status == "complete"
        assert store.task("infer").status == "pending"
        assert store.task("select").status == "pending"


def test_execution_rejects_catalog_drift_after_planning(tmp_path: Path) -> None:
    andrea = tmp_path / "ANDREA"
    toolspec = _definition(
        andrea,
        "tool",
        "genie3",
        {"id": "genie3", "docker_image": "example/genie3:1"},
    )
    planned = catalog_definition_identity(andrea, catalog_kind="tool", catalog_id="genie3")
    toolspec.write_text(
        json.dumps({"id": "genie3", "docker_image": "example/genie3:1", "inputs": []}),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkConfigError, match="changed after planning"):
        verify_catalog_definitions(andrea, {"tool:genie3": planned})
