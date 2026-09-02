"""Interactive report for the bounded SPATHI calibration workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifacts import sha256_file
from ..config import BenchmarkConfigError, load_json_object
from ..state import STATE_SCHEMA_VERSION
from .common import (
    central_report_path,
    path_below,
    repo_file,
    require_array,
    require_completed_workflow,
    require_mapping,
    safe_relative_link,
    verify_sha256,
    write_report_bundle,
)
from .datasets import build_dataset_overview, combine_dataset_overviews

CALIBRATION_STAGE_ORDER = ("convergence", "structure", "estimator", "validation")


def _stage_specifications(search_space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specifications: dict[str, dict[str, Any]] = {}
    for raw_stage in require_array(search_space.get("stages"), "calibration search stages"):
        stage = require_mapping(raw_stage, "calibration stage specification")
        stage_id = stage.get("id")
        if isinstance(stage_id, str) and stage_id:
            specifications[stage_id] = stage
    protocol = search_space.get("selection_protocol")
    if "validation" not in specifications and isinstance(protocol, dict):
        specifications["validation"] = {
            "id": "validation",
            "order": 4,
            "dataset_role": "validation",
            "strategy": "frozen_finalist_selection",
            "purpose": (
                "Choose one global configuration from the predeclared development finalists "
                "without accessing the held-out test panel."
            ),
            "advance_limit": 1,
            "candidates": [],
        }
    return specifications


def _candidate_rationales(stage: dict[str, Any]) -> dict[str, str]:
    rationales = {
        "reference": "Reference configuration from the frozen search space.",
        "inherited": "Configuration inherited unchanged from the preceding shortlist.",
    }
    for raw_candidate in stage.get("candidates", []):
        if not isinstance(raw_candidate, dict):
            continue
        candidate_id = raw_candidate.get("id")
        rationale = raw_candidate.get("rationale")
        if isinstance(candidate_id, str) and isinstance(rationale, str):
            rationales[candidate_id] = rationale
    return rationales


def _parameter_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _stage_payload(
    *,
    stage_id: str,
    ranking: dict[str, Any],
    specification: dict[str, Any],
) -> dict[str, Any]:
    ranked = require_array(ranking.get("ranking"), f"{stage_id} ranking")
    if not ranked:
        raise BenchmarkConfigError(f"Completed calibration stage {stage_id!r} has no candidates")
    selected_candidates = require_array(
        ranking.get("selected_candidates"), f"{stage_id} selected candidates"
    )
    selected_ids = {
        item.get("configuration_id")
        for item in selected_candidates
        if isinstance(item, dict) and isinstance(item.get("configuration_id"), str)
    }
    rationales = _candidate_rationales(specification)
    candidates: list[dict[str, Any]] = []
    parameter_values: dict[str, set[str]] = {}
    for index, raw_record in enumerate(ranked, start=1):
        record = require_mapping(raw_record, f"{stage_id} ranking entry")
        candidate = require_mapping(record.get("candidate"), f"{stage_id} candidate")
        configuration_id = candidate.get("configuration_id")
        variant_id = candidate.get("variant_id")
        parameters = require_mapping(candidate.get("parameters"), f"{stage_id} parameters")
        if not isinstance(configuration_id, str) or not configuration_id:
            raise BenchmarkConfigError(f"{stage_id} ranking candidate has no configuration_id")
        if not isinstance(variant_id, str) or not variant_id:
            raise BenchmarkConfigError(f"{stage_id} ranking candidate has no variant_id")
        for name, value in parameters.items():
            parameter_values.setdefault(name, set()).add(_parameter_value(value))
        candidates.append(
            {
                "rank": index,
                "configuration_id": configuration_id,
                "configuration_sha256": candidate.get("configuration_sha256"),
                "variant_id": variant_id,
                "parent_configuration_id": candidate.get("parent_configuration_id"),
                "parameters": parameters,
                "score": record.get("hierarchical_aupr"),
                "stability": record.get("stability"),
                "median_wall_time_seconds": record.get("median_wall_time_seconds"),
                "median_peak_memory_mb": record.get("median_peak_memory_mb"),
                "simplicity": record.get("simplicity"),
                "dataset_count": record.get("dataset_count"),
                "algorithm_seed_count": record.get("algorithm_seed_count"),
                "selected": configuration_id in selected_ids,
                "rationale": rationales.get(variant_id, "Predeclared calibration candidate."),
            }
        )
    observations = require_array(ranking.get("calibration_rows"), f"{stage_id} calibration rows")
    varying_parameters = sorted(
        name for name, observed_values in parameter_values.items() if len(observed_values) > 1
    )
    memory_available = any(
        candidate["median_peak_memory_mb"] is not None for candidate in candidates
    ) or any(
        isinstance(row, dict) and row.get("peak_memory_mb") is not None for row in observations
    )
    return {
        "id": stage_id,
        "order": specification.get("order"),
        "purpose": specification.get("purpose"),
        "dataset_role": ranking.get("dataset_role"),
        "primary_metric": ranking.get("primary_metric"),
        "strategy": specification.get("strategy"),
        "advance_limit": specification.get("advance_limit"),
        "candidates": candidates,
        "observations": observations,
        "varying_parameters": varying_parameters,
        "compared_parameters": [
            {"name": name, "values": sorted(parameter_values[name])} for name in varying_parameters
        ],
        "selected_ids": sorted(selected_ids),
        "winner": ranking.get("winner"),
        "test_data_accessed": ranking.get("test_data_accessed"),
        "memory_available": memory_available,
    }


def _stage_run_state(
    *,
    repo_root: Path,
    version_root: Path,
    record: dict[str, Any],
    profile_id: Any,
    stage_id: str,
) -> tuple[dict[str, Any], Path]:
    raw_run_root = record.get("run_root")
    if not isinstance(raw_run_root, str) or not raw_run_root:
        raise BenchmarkConfigError(f"{stage_id} workflow stage has no run root")
    candidate = Path(raw_run_root)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    run_root = path_below(candidate, version_root, f"{stage_id} run", file=False)
    state_path = run_root / "state.json"
    state = load_json_object(state_path, f"{stage_id} benchmark state")
    if (
        state.get("schema_version") != STATE_SCHEMA_VERSION
        or state.get("status") != "completed"
        or state.get("profile_id") != profile_id
    ):
        raise BenchmarkConfigError(f"{stage_id} benchmark state is not complete for its profile")
    return state, state_path


def _dataset_identity(state: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    fields = (
        "dataset_id",
        "dataset_role",
        "simulator_id",
        "condition_id",
        "seed",
        "split_id",
        "dataset_manifest",
    )
    return tuple(
        sorted(
            tuple(require_mapping(raw, "benchmark dataset").get(field) for field in fields)
            for raw in require_array(state.get("datasets"), "benchmark datasets")
        )
    )


def _record_group_counts(
    *,
    observations: list[Any],
    dataset_role: Any,
    stage_id: str,
    known: dict[tuple[str, str], int],
) -> set[tuple[str, str]]:
    if not isinstance(dataset_role, str) or not dataset_role:
        raise BenchmarkConfigError(f"{stage_id} ranking has no dataset role")
    observed: set[tuple[str, str]] = set()
    for raw_row in observations:
        row = require_mapping(raw_row, f"{stage_id} calibration row")
        dataset_id = row.get("dataset_id")
        row_role = row.get("dataset_role")
        groups = row.get("n_group_contexts")
        if not isinstance(dataset_id, str) or row_role != dataset_role:
            raise BenchmarkConfigError(
                f"{stage_id} calibration row has inconsistent dataset identity"
            )
        if isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0:
            raise BenchmarkConfigError(f"{stage_id} calibration row has no valid group count")
        key = (dataset_role, dataset_id)
        previous = known.setdefault(key, groups)
        if previous != groups:
            raise BenchmarkConfigError(f"Dataset {dataset_id!r} has inconsistent group counts")
        observed.add(key)
    return observed


def render_calibration_report(
    *, repo_root: Path, version_root: Path, workflow_root: Path
) -> dict[str, Any]:
    """Render a completed calibration without rerunning or reranking any experiment."""

    root = repo_root.resolve()
    version = path_below(
        version_root,
        root / "work",
        "benchmark version root",
        file=False,
    )
    calibration_root = path_below(
        workflow_root,
        version / "workflows/calibration",
        "calibration workflow",
        file=False,
    )
    workflow_path = calibration_root / "workflow.json"
    workflow = require_completed_workflow(workflow_path, "calibration")
    calibration_budget = workflow.get("calibration_budget")
    if calibration_budget not in {"reduced", "full"}:
        raise BenchmarkConfigError("Completed calibration has an invalid evidence budget")
    if calibration_root.parent.name != calibration_budget:
        raise BenchmarkConfigError("Calibration workflow path disagrees with its evidence budget")
    if calibration_root.name != workflow["flow_id"]:
        raise BenchmarkConfigError("Calibration workflow path disagrees with its flow ID")

    search_path = root / "experiments/v1/spathi/search-space.json"
    search_space = load_json_object(search_path, "SPATHI calibration search space")
    specifications = _stage_specifications(search_space)
    stage_records = require_mapping(workflow.get("stages"), "calibration workflow stages")
    if set(stage_records) != set(CALIBRATION_STAGE_ORDER):
        raise BenchmarkConfigError(
            "Completed calibration must contain convergence, structure, estimator and validation"
        )

    report_path = central_report_path(
        version_root=version,
        report_type="calibration",
        scope=calibration_budget,
        flow_id=workflow["flow_id"],
    )
    stages: list[dict[str, Any]] = []
    input_paths = [search_path]
    previous_ranking_path: Path | None = None
    previous_ranking_sha256: str | None = None
    ranking_paths: dict[str, Path] = {}
    dataset_states: dict[str, dict[str, Any]] = {}
    dataset_state_paths: list[Path] = []
    group_counts: dict[tuple[str, str], int] = {}
    test_access_flags: list[bool] = []
    spathi_commits: set[str] = set()
    for stage_id in CALIBRATION_STAGE_ORDER:
        record = require_mapping(stage_records.get(stage_id), f"{stage_id} workflow stage")
        if record.get("status") != "ranked":
            raise BenchmarkConfigError(f"Completed calibration stage {stage_id!r} is not ranked")
        ranking_path = repo_file(
            root,
            record.get("ranking"),
            f"{stage_id} ranking",
            below=calibration_root,
        )
        verify_sha256(ranking_path, record.get("ranking_sha256"), f"{stage_id} ranking")
        ranking = load_json_object(ranking_path, f"{stage_id} ranking")
        if ranking.get("schema_version") != "1.0" or ranking.get("stage_id") != stage_id:
            raise BenchmarkConfigError(f"Invalid {stage_id} calibration ranking contract")
        if ranking.get("calibration_budget") != calibration_budget:
            raise BenchmarkConfigError(f"{stage_id} ranking uses a different calibration budget")
        if ranking.get("profile_id") != record.get("profile_id"):
            raise BenchmarkConfigError(f"{stage_id} ranking profile differs from workflow state")
        if ranking.get("search_space_sha256") != sha256_file(search_path):
            raise BenchmarkConfigError(f"{stage_id} ranking uses a different search space")
        if ranking.get("spathi_commit") != workflow["identity"].get("spathi_commit"):
            raise BenchmarkConfigError(f"{stage_id} ranking uses a different SPATHI commit")
        parent = ranking.get("parent_ranking")
        if previous_ranking_path is None:
            if parent is not None:
                raise BenchmarkConfigError("Convergence ranking unexpectedly declares a parent")
        else:
            parent_record = require_mapping(parent, f"{stage_id} parent ranking")
            parent_path = repo_file(
                root,
                parent_record.get("path"),
                f"{stage_id} parent ranking",
                below=calibration_root,
            )
            if (
                parent_path != previous_ranking_path
                or parent_record.get("sha256") != previous_ranking_sha256
                or parent_record.get("stage_id")
                != CALIBRATION_STAGE_ORDER[CALIBRATION_STAGE_ORDER.index(stage_id) - 1]
            ):
                raise BenchmarkConfigError(f"{stage_id} ranking breaks the calibration chain")
        test_access = ranking.get("test_data_accessed")
        if test_access is not False:
            raise BenchmarkConfigError(
                f"{stage_id} ranking accessed or ambiguously handled test data"
            )
        test_access_flags.append(test_access)
        spathi_commit = ranking.get("spathi_commit")
        if isinstance(spathi_commit, str):
            spathi_commits.add(spathi_commit)
        observations = require_array(
            ranking.get("calibration_rows"), f"{stage_id} calibration rows"
        )
        observed_datasets = _record_group_counts(
            observations=observations,
            dataset_role=ranking.get("dataset_role"),
            stage_id=stage_id,
            known=group_counts,
        )
        stage_state, stage_state_path = _stage_run_state(
            repo_root=root,
            version_root=version,
            record=record,
            profile_id=record.get("profile_id"),
            stage_id=stage_id,
        )
        state_datasets = {
            (str(item.get("dataset_role")), str(item.get("dataset_id")))
            for item in require_array(stage_state.get("datasets"), "benchmark datasets")
            if isinstance(item, dict)
        }
        if state_datasets != observed_datasets:
            raise BenchmarkConfigError(
                f"{stage_id} ranking datasets differ from its completed benchmark state"
            )
        role = str(ranking["dataset_role"])
        previous_state = dataset_states.setdefault(role, stage_state)
        if _dataset_identity(previous_state) != _dataset_identity(stage_state):
            raise BenchmarkConfigError(f"Calibration stages disagree on the {role!r} dataset panel")
        dataset_state_paths.append(stage_state_path)
        stages.append(
            _stage_payload(
                stage_id=stage_id,
                ranking=ranking,
                specification=specifications.get(stage_id, {}),
            )
        )
        ranking_paths[stage_id] = ranking_path
        input_paths.append(ranking_path)
        previous_ranking_path = ranking_path
        previous_ranking_sha256 = sha256_file(ranking_path)

    selected_path = repo_file(
        root,
        workflow.get("selected_config"),
        "frozen SPATHI configuration",
        below=calibration_root,
    )
    verify_sha256(
        selected_path,
        workflow.get("selected_config_sha256"),
        "frozen SPATHI configuration",
    )
    selected = load_json_object(selected_path, "frozen SPATHI configuration")
    if (
        selected.get("schema_version") != "1.0"
        or selected.get("status") != "selected"
        or selected.get("phase") != "frozen"
        or selected.get("calibration_budget") != calibration_budget
        or selected.get("test_accessed_before_freeze") is not False
    ):
        raise BenchmarkConfigError("Frozen SPATHI configuration has an invalid selection contract")
    selected_candidate = require_mapping(
        selected.get("selected_candidate"), "frozen SPATHI candidate"
    )
    validation_winner = require_mapping(stages[-1].get("winner"), "validation winner")
    if (
        selected_candidate.get("candidate_id") != validation_winner.get("candidate_id")
        or selected_candidate.get("configuration_sha256")
        != validation_winner.get("configuration_sha256")
        or selected_candidate.get("spathi_commit") != workflow["identity"].get("spathi_commit")
    ):
        raise BenchmarkConfigError("Frozen selection differs from the validation winner")
    input_paths.append(selected_path)

    dataset_overviews: list[dict[str, Any]] = []
    for role in sorted(dataset_states):
        overview, overview_inputs = build_dataset_overview(
            repo_root=root,
            version_root=version,
            benchmark_state=dataset_states[role],
            group_counts=group_counts,
        )
        dataset_overviews.append(overview)
        input_paths.extend(overview_inputs)
    datasets = combine_dataset_overviews(dataset_overviews)
    input_paths.extend(dataset_state_paths)

    links = [
        {
            "label": "Workflow state",
            "href": safe_relative_link(report_path, workflow_path, calibration_root),
        },
        {
            "label": "Frozen configuration",
            "href": safe_relative_link(report_path, selected_path, calibration_root),
        },
    ]
    links.extend(
        {
            "label": f"{stage_id.title()} ranking",
            "href": safe_relative_link(report_path, ranking_paths[stage_id], calibration_root),
        }
        for stage_id in CALIBRATION_STAGE_ORDER
    )
    payload = {
        "schema_version": "1.0",
        "report_type": "calibration",
        "workflow": {
            "flow_id": workflow["flow_id"],
            "status": workflow["status"],
            "calibration_budget": calibration_budget,
            "identity": workflow["identity"],
        },
        "integrity": {
            "stage_order": list(CALIBRATION_STAGE_ORDER),
            "all_stages_ranked": True,
            "test_data_accessed_before_freeze": False,
            "single_spathi_commit": len(spathi_commits) == 1,
            "selection_frozen": True,
        },
        "objective": search_space.get("objective"),
        "dataset_partition": search_space.get("dataset_partition"),
        "datasets": datasets,
        "selected": selected,
        "stages": stages,
        "links": links,
    }
    return write_report_bundle(
        repo_root=root,
        kind="calibration",
        output_path=report_path,
        payload=payload,
        input_paths=input_paths,
        manifest_name="calibration-report.manifest.json",
    )
