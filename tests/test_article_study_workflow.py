from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from spathi_bench.cli import build_parser
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.design import load_design
from spathi_bench.study_store import StudyStore
from spathi_bench.study_workflow import (
    _phase_runtime_provenance,
    _report_snapshot,
    build_task_plan,
    rebuild_study_reports,
    run_study_workflow,
)

REPO = Path(__file__).resolve().parents[1]


def _resource_plan(cpus: int = 8, gib: int = 16) -> dict[str, Any]:
    return {
        "capacity": {
            "cpu_threads": cpus,
            "memory_bytes": gib * 1024**3,
            "memory_gib": float(gib),
        }
    }


def test_plan_contains_only_three_phases_and_equal_calibration_candidates() -> None:
    design = load_design(REPO / "experiments/article-study.json")
    tasks = build_task_plan(design, "smoke", _resource_plan())

    assert {task["phase"] for task in tasks} == {
        "calibration",
        "comparison",
        "scalability",
    }
    screens = [task for task in tasks if task["kind"] == "calibration_screen"]
    transfers = [task for task in tasks if task["kind"] == "calibration_transfer"]
    assert screens
    assert len(transfers) == 1
    for task in screens:
        candidates = task["spec"]["candidates"]
        assert len(candidates) == 27
        assert {candidate["id"] for candidate in candidates} == {
            f"candidate-{index:02d}" for index in range(1, 28)
        }
        assert all("reference" not in candidate["id"] for candidate in candidates)
    validation_ids = {task["task_id"] for task in tasks if task["kind"] == "calibration_validate"}
    assert validation_ids
    assert validation_ids <= set(transfers[0]["dependencies"])


def test_generators_receive_honest_resource_allowances() -> None:
    design = load_design(REPO / "experiments/article-study.json")
    tasks = build_task_plan(design, "full", _resource_plan(cpus=120, gib=1024))
    generation = [task for task in tasks if task["kind"] == "generate_dataset"]
    external = [
        task for task in generation if task["spec"]["dataset"]["generator"] != "scale-stress"
    ]

    assert external
    assert {task["spec"]["resources"]["memory_bytes"] for task in external} == {32 * 1024**3}
    assert {
        task["spec"]["resources"]["cpu_threads"]
        for task in external
        if task["spec"]["dataset"]["generator"] in {"dyngen", "scmultisim"}
    } == {8}
    assert {
        task["spec"]["resources"]["cpu_threads"]
        for task in external
        if task["spec"]["dataset"]["generator"] in {"boolode", "sergio"}
    } == {1}

    large_scale = [
        task for task in generation if task["spec"]["dataset"]["condition_id"] == "p-xlarge"
    ]
    assert large_scale
    assert {tuple(sorted(task["spec"]["resources"].items())) for task in large_scale} == {
        tuple(
            sorted(
                {
                    "cpu_threads": 4,
                    "memory_bytes": 10_000 * 20_000 * 8 * 12 + 1024**3,
                    "exclusive": False,
                }.items()
            )
        )
    }


def test_external_generator_resources_use_the_largest_measured_tier() -> None:
    design = load_design(REPO / "experiments/article-study.json")
    tasks = build_task_plan(design, "smoke", _resource_plan(cpus=6, gib=24))
    external = [
        task
        for task in tasks
        if task["kind"] == "generate_dataset"
        and task["spec"]["dataset"]["generator"] != "scale-stress"
    ]

    assert external
    assert {task["spec"]["resources"]["memory_bytes"] for task in external} == {16 * 1024**3}
    assert {task["spec"]["resources"]["cpu_threads"] for task in external} == {4}


def test_external_generator_requires_a_measured_memory_tier() -> None:
    design = load_design(REPO / "experiments/article-study.json")

    with pytest.raises(
        BenchmarkConfigError, match="External simulator generation requires at least .* RAM"
    ):
        build_task_plan(design, "smoke", _resource_plan(cpus=8, gib=7))


def test_comparison_batches_methods_but_scalability_isolates_them() -> None:
    design = load_design(REPO / "experiments/article-study.json")
    tasks = build_task_plan(design, "reduced", _resource_plan(cpus=32, gib=128))

    comparison = [task for task in tasks if task["kind"] == "comparison_dataset"]
    assert len(comparison) == 11
    assert all(len(task["spec"]["routes"]) > 1 for task in comparison)
    assert all(
        task["executor_contract"].endswith(":machine-relative-memory-slots") for task in comparison
    )
    assert all(
        not task["executor_contract"].endswith(":machine-relative-memory-slots")
        for task in tasks
        if task["kind"] != "comparison_dataset"
    )

    measurements = [task for task in tasks if task["kind"] == "scalability_measurement"]
    assert measurements
    assert all("route" in task["spec"] and "routes" not in task["spec"] for task in measurements)
    assert all(task["spec"]["resources"]["exclusive"] is True for task in measurements)
    assert all(task["spec"]["resources"]["cpu_threads"] == 32 for task in measurements)

    ladder = [task for task in measurements if task["spec"]["measurement"] == "size_ladder"]
    by_route: dict[str, list[dict[str, Any]]] = {}
    for task in ladder:
        key = task["spec"]["route"]["id"]
        by_route.setdefault(key, []).append(task)
    for route_tasks in by_route.values():
        ordered = sorted(
            route_tasks,
            key=lambda item: (
                item["stage"],
                item["spec"]["dataset"]["replicate"],
            ),
        )
        for previous, current in zip(ordered, ordered[1:], strict=False):
            assert previous["task_id"] in current["dependencies"]
            assert current["spec"]["advance_dependency"] == previous["task_id"]

    thread_scaling = [
        task for task in measurements if task["spec"]["measurement"] == "thread_scaling"
    ]
    assert all(
        task["spec"]["effective_threads"] <= task["spec"]["route"]["max_threads"]
        for task in thread_scaling
        if task["spec"]["route"]["max_threads"] is not None
    )
    assert not any(
        task["spec"]["route"]["id"] in {"clr", "ppcor", "lioness"} for task in thread_scaling
    )


def test_phase_barriers_are_explicit_dependencies() -> None:
    design = load_design(REPO / "experiments/article-study.json")
    tasks = build_task_plan(design, "smoke", _resource_plan())
    winner = next(task for task in tasks if task["kind"] == "select_calibration_winner")
    statistics = next(task for task in tasks if task["kind"] == "comparison_statistics")

    comparison_generation = [
        task
        for task in tasks
        if task["kind"] == "generate_dataset" and task["phase"] == "comparison"
    ]
    scalability_generation = [
        task
        for task in tasks
        if task["kind"] == "generate_dataset" and task["phase"] == "scalability"
    ]
    assert all(winner["task_id"] in task["dependencies"] for task in comparison_generation)
    assert all(statistics["task_id"] in task["dependencies"] for task in scalability_generation)


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def execute_task(self, task: Any, dependencies: Any) -> dict[str, Any]:
        self.calls[task.task_id] += 1
        return {
            "summary": {"kind": task.kind},
            "rows": [],
            "decisions": [],
            "failures": [],
            "dependencies_observed": sorted(dependencies),
        }


def _revision(commit: str) -> dict[str, Any]:
    return {
        "benchmark_commit": commit,
        "benchmark_tree": f"benchmark-tree-{commit}",
        "benchmark_execution_tree": "benchmark-execution-tree",
        "spathi_commit": "spathi",
        "spathi_scientific_tree": "spathi-tree",
        "andrea_commit": "andrea",
        "andrea_scientific_tree": "andrea-tree",
    }


def test_resume_across_compatible_commit_does_not_repeat_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import spathi_bench.study_workflow as workflow

    fake = _FakeExecutor()
    revision = [_revision("a")]
    monkeypatch.setattr(workflow, "observe_revisions", lambda *_args: revision[0])
    monkeypatch.setattr(
        workflow,
        "_runtime_plan",
        lambda **_kwargs: _resource_plan(cpus=4, gib=8),
    )
    monkeypatch.setattr(
        workflow,
        "build_study_deliverables",
        lambda _snapshot, root: {"directory": str(root / "resume-test")},
    )

    def factory(**_kwargs: Any) -> _FakeExecutor:
        return fake

    first = run_study_workflow(
        repo_root=REPO,
        andrea_repo=REPO.parent / "ANDREA",
        spathi_repo=REPO.parent / "SPATHI",
        work_root=tmp_path / "work",
        profile="smoke",
        study_id="resume-test",
        executor_factory=factory,
    )
    assert first["status"] == "complete"
    first_calls = fake.calls.copy()
    assert first_calls

    revision[0] = _revision("b")
    second = run_study_workflow(
        repo_root=REPO,
        andrea_repo=REPO.parent / "ANDREA",
        spathi_repo=REPO.parent / "SPATHI",
        work_root=tmp_path / "work",
        profile=None,
        study_id="resume-test",
        executor_factory=factory,
    )
    assert second["status"] == "complete"
    assert fake.calls == first_calls
    with StudyStore(tmp_path / "work/studies/resume-test/study.sqlite") as store:
        snapshot = _report_snapshot(store)
    for phase in snapshot["phases"].values():
        for task in phase["provenance"]["tasks"]:
            assert {attempt["revision_sequence"] for attempt in task["attempts"]} <= {1}
        assert [revision["sequence"] for revision in phase["provenance"]["revisions"]] == [
            1,
            2,
        ]


def test_cli_exposes_only_the_clean_public_commands() -> None:
    parser = build_parser()
    run = parser.parse_args(["run", "--profile", "reduced", "--id", "paper-1"])
    resume = parser.parse_args(["run", "--id", "paper-1"])
    assert run.profile == "reduced"
    assert resume.profile is None
    for command in ("status", "plan", "report", "doctor"):
        arguments = [command]
        if command == "plan":
            arguments += ["--profile", "smoke"]
        if command == "report":
            arguments += ["--id", "paper-1"]
        assert parser.parse_args(arguments).command == command


def test_report_command_supports_an_interrupted_incomplete_study(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    study_root = work_root / "studies/interrupted-1"
    study_root.mkdir(parents=True)
    with StudyStore(study_root / "study.sqlite") as store:
        store.initialize(
            {
                "study_id": "interrupted-1",
                "profile": "smoke",
                "design_digest": "d" * 64,
                "plan_topology_digest": "p" * 64,
                "scientific_design": {},
                "resource_plan": _resource_plan(),
            }
        )
        store.record_revision(_revision("interrupted"))
        store.add_tasks(
            [
                {
                    "task_id": phase,
                    "phase": phase,
                    "stage": "pending-stage",
                    "kind": "pending-task",
                    "executor_contract": "test-v1",
                    "spec": {"phase": phase},
                    "dependencies": [],
                }
                for phase in ("calibration", "comparison", "scalability")
            ]
        )

    summary = rebuild_study_reports(
        repo_root=tmp_path,
        work_root=work_root,
        study_id="interrupted-1",
    )

    assert summary["status"] == "pending"
    output = tmp_path / "deliverables/interrupted-1"
    for filename in (
        "article-report.html",
        "calibration.html",
        "comparison.html",
        "scalability.html",
    ):
        content = (output / filename).read_text(encoding="utf-8")
        assert '"status":"pending"' in content
        assert "<script src=" not in content
        assert "<link " not in content


def test_report_snapshot_exports_runtime_identities_in_phase_provenance(
    tmp_path: Path,
) -> None:
    with StudyStore(tmp_path / "study.sqlite3") as store:
        store.initialize(
            {
                "study_id": "provenance-test",
                "profile": "smoke",
                "design_digest": "d" * 64,
                "plan_topology_digest": "p" * 64,
                "scientific_design": {},
                "resource_plan": _resource_plan(),
            }
        )
        store.record_revision(_revision("provenance"))
        store.add_tasks(
            [
                {
                    "task_id": "generate",
                    "phase": "comparison",
                    "stage": "generation",
                    "kind": "generate_dataset",
                    "scientific_digest": "s" * 64,
                    "executor_contract": "test-v1",
                    "spec": {},
                    "dependencies": [],
                }
            ]
        )
        store.start("generate")
        store.finish(
            "generate",
            {
                "summary": {
                    "generator_identity": {
                        "requested_image": "catalog/dyngen:1",
                        "image_id": "sha256:" + "a" * 64,
                        "repo_digest": "catalog/dyngen@sha256:" + "b" * 64,
                        "origin": "andrea_simulator_catalog",
                        "simulator_id": "dyngen",
                        "definition_sha256": "c" * 64,
                    },
                    "host_generator": {
                        "origin": "benchmark_host_generator",
                        "generator_id": "spathi_scale_stress",
                        "generator_contract": "scale-stress-v1",
                    },
                }
            },
        )

        snapshot = _report_snapshot(store)

    provenance = snapshot["phases"]["comparison"]["provenance"]
    assert provenance["image_identities"] == [
        {
            "requested_image": "catalog/dyngen:1",
            "image_id": "sha256:" + "a" * 64,
            "repo_digest": "catalog/dyngen@sha256:" + "b" * 64,
            "origin": "andrea_simulator_catalog",
            "simulator_id": "dyngen",
            "definition_sha256": "c" * 64,
        }
    ]
    assert provenance["host_generator_identities"] == [
        {
            "origin": "benchmark_host_generator",
            "generator_id": "spathi_scale_stress",
            "generator_contract": "scale-stress-v1",
        }
    ]
    [task] = provenance["tasks"]
    [attempt] = task["attempts"]
    assert attempt["attempt"] == 1
    assert attempt["revision_sequence"] == 1
    assert attempt["status"] == "complete"
    assert attempt["result_sha256"]
    assert attempt["runtime_provenance"]["image_identities"] == provenance["image_identities"]
    assert (
        provenance["all_attempt_runtime_identities"]["image_identities"]
        == provenance["image_identities"]
    )
    assert provenance["revisions"][0]["sequence"] == 1


def test_runtime_provenance_preserves_distinct_definitions_for_one_image() -> None:
    common = {
        "requested_image": "catalog/shared:1",
        "image_id": "sha256:" + "a" * 64,
        "repo_digest": None,
        "origin": "andrea_catalog",
    }
    provenance = _phase_runtime_provenance(
        [
            {"image_identities": {"first": {**common, "definition_sha256": "1" * 64}}},
            {"image_identities": {"second": {**common, "definition_sha256": "2" * 64}}},
        ]
    )

    assert len(provenance["image_identities"]) == 2
    assert {value["definition_sha256"] for value in provenance["image_identities"]} == {
        "1" * 64,
        "2" * 64,
    }
