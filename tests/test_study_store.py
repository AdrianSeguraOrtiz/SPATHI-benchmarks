from __future__ import annotations

from pathlib import Path

import pytest

from spathi_bench.config import BenchmarkConfigError
from spathi_bench.study_store import StudyStore


def _manifest() -> dict[str, object]:
    return {
        "study_id": "reduced-1",
        "profile": "reduced",
        "design_digest": "design-a",
        "plan_topology_digest": "topology-a",
        "scientific_design": {"phases": ["calibration", "comparison", "scalability"]},
    }


def _task(task_id: str, *, value: int, dependencies: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "task_id": task_id,
        "phase": "calibration",
        "stage": "screen",
        "kind": "test",
        "executor_contract": "test-v1",
        "spec": {"value": value},
        "dependencies": dependencies,
    }


def _revision(label: str) -> dict[str, str]:
    return {
        "benchmark_commit": label,
        "benchmark_tree": f"benchmark-tree-{label}",
        "benchmark_execution_tree": f"benchmark-execution-tree-{label}",
        "spathi_commit": f"spathi-{label}",
        "spathi_scientific_tree": f"spathi-tree-{label}",
        "andrea_commit": f"andrea-{label}",
        "andrea_scientific_tree": f"andrea-tree-{label}",
    }


def test_resume_preserves_compatible_tasks_across_revisions(tmp_path: Path) -> None:
    with StudyStore(tmp_path / "study.sqlite") as store:
        assert store.initialize(_manifest()) is True
        assert store.initialize(_manifest()) is False
        changed, previous = store.record_revision(
            {
                "benchmark_commit": "a",
                "benchmark_tree": "tree-a",
                "spathi_commit": "s",
                "spathi_scientific_tree": "spathi-tree",
                "andrea_commit": "n",
                "andrea_scientific_tree": "andrea-tree",
            }
        )
        assert changed is False
        assert previous is None
        assert store.add_tasks([_task("a", value=1)]) == {
            "inserted": 1,
            "reused": 0,
            "invalidated": 0,
        }
        store.start("a")
        store.finish("a", {"artifact": "one"})

        changed, _ = store.record_revision(
            {
                "benchmark_commit": "b",
                "benchmark_tree": "tree-b",
                "spathi_commit": "s",
                "spathi_scientific_tree": "spathi-tree",
                "andrea_commit": "n",
                "andrea_scientific_tree": "andrea-tree",
            }
        )
        assert changed is True
        assert store.add_tasks([_task("a", value=1)])["reused"] == 1
        assert store.task("a").status == "complete"


def test_changed_task_invalidates_transitive_descendants(tmp_path: Path) -> None:
    with StudyStore(tmp_path / "study.sqlite") as store:
        store.initialize(_manifest())
        store.record_revision(_revision("one"))
        store.add_tasks(
            [
                _task("a", value=1),
                _task("b", value=2, dependencies=("a",)),
                _task("c", value=3, dependencies=("b",)),
            ]
        )
        for task_id in ("a", "b", "c"):
            store.start(task_id)
            store.finish(task_id, {"ok": True})

        summary = store.add_tasks(
            [
                _task("a", value=9),
                _task("b", value=2, dependencies=("a",)),
                _task("c", value=3, dependencies=("b",)),
            ]
        )
        assert summary == {"inserted": 0, "reused": 2, "invalidated": 3}
        assert [store.task(value).status for value in ("a", "b", "c")] == [
            "pending",
            "pending",
            "pending",
        ]


def test_ready_tasks_require_successful_dependencies(tmp_path: Path) -> None:
    with StudyStore(tmp_path / "study.sqlite") as store:
        store.initialize(_manifest())
        store.record_revision(_revision("one"))
        store.add_tasks([_task("a", value=1), _task("b", value=2, dependencies=("a",))])
        assert [task.task_id for task in store.ready_tasks()] == ["a"]
        store.start("a")
        store.finish("a", {"ok": True})
        assert [task.task_id for task in store.ready_tasks()] == ["b"]


def test_task_graph_rejects_cycles_before_mutating_the_journal(tmp_path: Path) -> None:
    with StudyStore(tmp_path / "study.sqlite") as store:
        store.initialize(_manifest())
        store.record_revision(_revision("one"))
        with pytest.raises(BenchmarkConfigError, match="dependencies contain a cycle"):
            store.add_tasks(
                [
                    _task("a", value=1, dependencies=("b",)),
                    _task("b", value=2, dependencies=("a",)),
                ]
            )
        assert store.tasks() == []


def test_task_graph_rejects_duplicate_dependencies(tmp_path: Path) -> None:
    with StudyStore(tmp_path / "study.sqlite") as store:
        store.initialize(_manifest())
        store.record_revision(_revision("one"))
        store.add_tasks([_task("a", value=1)])
        with pytest.raises(BenchmarkConfigError, match="duplicate dependencies"):
            store.add_tasks([_task("b", value=2, dependencies=("a", "a"))])


def test_read_transaction_keeps_one_consistent_wal_snapshot(tmp_path: Path) -> None:
    database = tmp_path / "study.sqlite"
    with StudyStore(database) as reader:
        reader.initialize(_manifest())
        reader.record_revision(_revision("one"))
        reader.add_tasks([_task("a", value=1)])
        with StudyStore(database) as writer:
            with reader.read_transaction():
                assert reader.task("a").status == "pending"
                writer.start("a")
                writer.finish("a", {"ok": True})
                assert reader.task("a").status == "pending"
            assert reader.task("a").status == "complete"


def test_existing_id_rejects_scientific_design_change(tmp_path: Path) -> None:
    with StudyStore(tmp_path / "study.sqlite") as store:
        store.initialize(_manifest())
        changed = {**_manifest(), "design_digest": "design-b"}
        with pytest.raises(BenchmarkConfigError, match="scientific design changed"):
            store.initialize(changed)


def test_existing_id_rejects_task_topology_change(tmp_path: Path) -> None:
    with StudyStore(tmp_path / "study.sqlite") as store:
        store.initialize(_manifest())
        changed = {**_manifest(), "plan_topology_digest": "topology-b"}
        with pytest.raises(BenchmarkConfigError, match="task topology changed"):
            store.initialize(changed)


def test_interrupted_running_task_becomes_pending(tmp_path: Path) -> None:
    with StudyStore(tmp_path / "study.sqlite") as store:
        store.initialize(_manifest())
        store.record_revision(_revision("one"))
        store.add_tasks([_task("a", value=1)])
        store.start("a")
        assert store.recover_interrupted() == 1
        assert store.task("a").status == "pending"
        [attempt] = store.snapshot()["task_attempts"]
        assert attempt["revision_sequence"] == 1
        assert attempt["status"] == "interrupted"
        assert attempt["error"] == "Interrupted before completion"


def test_attempt_history_binds_each_result_to_its_active_revision(tmp_path: Path) -> None:
    with StudyStore(tmp_path / "study.sqlite") as store:
        store.initialize(_manifest())
        store.record_revision(_revision("one"))
        store.add_tasks([_task("a", value=1)])
        store.start("a")
        store.finish("a", {"artifact": "first"})

        store.record_revision(_revision("two"))
        store.add_tasks([_task("a", value=2)])
        store.start("a")
        store.finish("a", {"artifact": "second"})

        snapshot = store.snapshot()

    assert [revision["sequence"] for revision in snapshot["revisions"]] == [1, 2]
    attempts = snapshot["task_attempts"]
    assert [attempt["revision_sequence"] for attempt in attempts] == [1, 2]
    assert [attempt["status"] for attempt in attempts] == ["complete", "complete"]
    assert [attempt["result"] for attempt in attempts] == [
        {"artifact": "first"},
        {"artifact": "second"},
    ]
    assert attempts[0]["scientific_digest"] != attempts[1]["scientific_digest"]


def test_task_cannot_start_without_an_active_revision(tmp_path: Path) -> None:
    with StudyStore(tmp_path / "study.sqlite") as store:
        store.initialize(_manifest())
        store.add_tasks([_task("a", value=1)])
        with pytest.raises(BenchmarkConfigError, match="active implementation revision"):
            store.start("a")
