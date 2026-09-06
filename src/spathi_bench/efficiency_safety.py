"""Pre-freeze safety veto for a provisionally selected efficiency policy.

The confirmatory efficiency panel selects at most one provisional candidate.
This module freezes that choice before safety outcomes are inspected and applies
a deliberately descriptive transfer check across six scale/grouping strata. It
cannot promote a policy that failed the confirmatory study and it never searches
for a replacement.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .config import BenchmarkConfigError, load_json_object, validate_json
from .noninferiority import (
    _dataset_observations,
    _validate_plan_binding,
    resolve_efficiency_adoption,
)
from .noninferiority_evidence import verify_noninferiority_evidence
from .safe_io import write_idempotent_json
from .scenario_inputs import scenario_external_input_records
from .seeds import load_seed_plan, plan_seed_record, validate_plan_seed_record

SAFETY_DATASET_ROLE = "efficiency_safety"
SAFETY_SPLIT_ID = "efficiency_safety_v1"
SAFETY_NAMESPACE = "full.efficiency_safety"
SAFETY_CONDITION_SCALES = {
    "g3000_tf100_c300": {
        "genes": 3000,
        "transcription_factors": 100,
        "cells": 300,
        "group_sizes": [100, 100, 100],
        "target_list": None,
    },
    "g3000_tf1500_c300": {
        "genes": 3000,
        "transcription_factors": 1500,
        "cells": 300,
        "group_sizes": [100, 100, 100],
        "target_list": None,
    },
    "g15000_tf100_c300": {
        "genes": 15000,
        "transcription_factors": 100,
        "cells": 300,
        "group_sizes": [100, 100, 100],
        "target_list": None,
    },
    "g100_tf10_c500_single_group": {
        "genes": 100,
        "transcription_factors": 10,
        "cells": 500,
        "group_sizes": [500],
        "target_list": None,
    },
    "g100_tf10_c300_groups_10_40_250": {
        "genes": 100,
        "transcription_factors": 10,
        "cells": 300,
        "group_sizes": [10, 40, 250],
        "target_list": None,
    },
    "g100_tf10_c10000": {
        "genes": 100,
        "transcription_factors": 10,
        "cells": 10000,
        "group_sizes": [3333, 3333, 3334],
        "target_list": None,
    },
}
SAFETY_CONDITIONS = tuple(SAFETY_CONDITION_SCALES)
SAFETY_DATASETS_PER_CONDITION = 3
SAFETY_EXPECTED_DATASETS = len(SAFETY_CONDITIONS) * SAFETY_DATASETS_PER_CONDITION
SAFETY_ANALYSIS_ROLE = "descriptive_safety_veto"
SAFETY_THRESHOLDS = {
    "expected_independent_datasets": SAFETY_EXPECTED_DATASETS,
    "expected_conditions": list(SAFETY_CONDITIONS),
    "expected_datasets_per_condition": SAFETY_DATASETS_PER_CONDITION,
    "minimum_mean_aupr_difference": -0.002,
    "minimum_median_aupr_difference": -0.002,
    "minimum_each_dataset_aupr_difference": -0.01,
    "maximum_median_wall_time_ratio": 1.0,
}


def _inside(repo_root: Path, path: Path, label: str, *, require_file: bool = True) -> str:
    root = repo_root.resolve()
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise BenchmarkConfigError(f"{label} must be inside the benchmark repository") from exc
    if absolute != resolved or path.is_symlink():
        raise BenchmarkConfigError(f"{label} must be a canonical non-symlink path")
    if require_file and not resolved.is_file():
        raise BenchmarkConfigError(f"{label} does not exist: {path}")
    return relative.as_posix()


def _file_reference(repo_root: Path, path: Path, label: str) -> dict[str, str]:
    return {
        "path": _inside(repo_root, path, label),
        "sha256": sha256_file(path),
    }


def _source_results(
    repo_root: Path,
    *,
    efficiency_plan_path: Path,
    efficiency_results_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = load_json_object(efficiency_plan_path, "confirmatory efficiency plan")
    validate_json(
        plan,
        repo_root / "schemas/noninferiority-plan.schema.json",
        "confirmatory efficiency plan",
    )
    _validate_plan_binding(repo_root, plan)
    results = load_json_object(efficiency_results_path, "confirmatory efficiency results")
    validate_json(
        results,
        repo_root / "schemas/noninferiority-results.schema.json",
        "confirmatory efficiency results",
    )
    if (
        plan["budget"]["id"] != "full"
        or plan["budget"]["analysis_role"] != "confirmatory"
        or results["budget"] != "full"
        or results["analysis_role"] != "confirmatory"
        or results["inputs"]["plan_sha256"] != sha256_file(efficiency_plan_path)
    ):
        raise BenchmarkConfigError(
            "Efficiency safety requires the completed full confirmatory efficiency study"
        )
    adoption = resolve_efficiency_adoption(
        plan,
        results["comparisons"],
        results.get("execution_failures", []),
    )
    if adoption != results.get("adoption"):
        raise BenchmarkConfigError(
            "Efficiency adoption does not reproduce from the frozen plan and results"
        )
    return plan, results, adoption


def provisional_candidate(
    repo_root: Path,
    *,
    efficiency_plan_path: Path,
    efficiency_results_path: Path,
) -> dict[str, Any] | None:
    """Return the sole provisional candidate, or ``None`` when exact already won."""

    plan, _results, adoption = _source_results(
        repo_root.resolve(),
        efficiency_plan_path=efficiency_plan_path.resolve(),
        efficiency_results_path=efficiency_results_path.resolve(),
    )
    if adoption["decision"] == "reference_retained":
        return None
    if adoption["decision"] != "candidate_adopted":
        raise BenchmarkConfigError(
            "Only a confirmatory candidate adoption can enter efficiency safety"
        )
    matches = [
        candidate
        for candidate in plan["candidates"]
        if candidate["id"] == adoption["selected_candidate_id"]
    ]
    if len(matches) != 1:
        raise BenchmarkConfigError("Provisional efficiency candidate is not unique in its plan")
    candidate = matches[0]
    if (
        candidate["policy_sha256"] != adoption["selected_policy_sha256"]
        or candidate["group_policy"] != adoption["selected_group_policy"]
    ):
        raise BenchmarkConfigError("Provisional candidate differs from the formal adoption")
    return candidate


def _coverage_thresholds(candidate: Mapping[str, Any]) -> dict[str, Any]:
    gates = candidate.get("coverage_gates")
    if gates is None:
        return {
            "minimum_each_dataset_target_fraction": 1.0,
            "minimum_each_dataset_truth_edge_fraction": 1.0,
            "basis": "full_target_policy_requires_complete_coverage",
        }
    if not isinstance(gates, Mapping):
        raise BenchmarkConfigError("Provisional candidate has invalid coverage gates")
    return {
        "minimum_each_dataset_target_fraction": float(
            gates["minimum_each_dataset_target_fraction"]
        ),
        "minimum_each_dataset_truth_edge_fraction": float(
            gates["minimum_each_dataset_truth_edge_fraction"]
        ),
        "basis": gates["threshold_basis"],
    }


def materialize_safety_plan(
    *,
    repo_root: Path,
    efficiency_plan_path: Path,
    efficiency_results_path: Path,
    panel_path: Path,
    output_path: Path,
) -> Path:
    """Freeze the one provisional candidate before safety outcomes are accessed."""

    root = repo_root.resolve()
    for path, label in (
        (efficiency_plan_path, "confirmatory efficiency plan"),
        (efficiency_results_path, "confirmatory efficiency results"),
        (panel_path, "efficiency safety panel"),
    ):
        _inside(root, path, label)
    _inside(root, output_path, "efficiency safety plan output", require_file=False)
    source_plan, source_results, adoption = _source_results(
        root,
        efficiency_plan_path=efficiency_plan_path,
        efficiency_results_path=efficiency_results_path,
    )
    candidate = provisional_candidate(
        root,
        efficiency_plan_path=efficiency_plan_path,
        efficiency_results_path=efficiency_results_path,
    )
    if candidate is None:
        raise BenchmarkConfigError("The exact policy already won; efficiency safety must not run")
    panel = load_json_object(panel_path, "efficiency safety panel")
    validate_json(
        panel,
        root / "schemas/efficiency-safety-panel.schema.json",
        "efficiency safety panel",
    )
    if (
        panel["selection_validation_manifest_sha256"]
        != source_plan["reference"]["selection_validation_manifest_sha256"]
    ):
        raise BenchmarkConfigError("Safety panel is bound to another calibration selection")
    source_panel_path = root / source_plan["panel"]["path"]
    source_panel = load_json_object(source_panel_path, "efficiency validation panel")
    source_ids = {item["dataset_id"] for item in source_panel["datasets"]}
    safety_ids = {item["dataset_id"] for item in panel["datasets"]}
    if source_ids & safety_ids:
        raise BenchmarkConfigError("Safety panel reuses efficiency-validation dataset IDs")

    seed_plan_path = root / "experiments/v1/seeds/seed-plan.json"
    seed_plan = load_seed_plan(seed_plan_path)
    algorithm_seed = plan_seed_record(
        seed_plan,
        namespace=SAFETY_NAMESPACE,
        stream="inference",
        coordinates={"purpose": "common_algorithm_seed"},
    )
    execution_order_seed = plan_seed_record(
        seed_plan,
        namespace=SAFETY_NAMESPACE,
        stream="execution_order",
        coordinates={"purpose": "counterbalanced_execution_order"},
    )
    payload = {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_safety_plan",
        "study_phase": "efficiency_safety",
        "study": source_plan["study"],
        "source_efficiency": {
            "plan": _file_reference(root, efficiency_plan_path, "confirmatory efficiency plan"),
            "results": _file_reference(
                root, efficiency_results_path, "confirmatory efficiency results"
            ),
            "formal_adoption_sha256": canonical_json_sha256(adoption),
        },
        "panel": _file_reference(root, panel_path, "efficiency safety panel"),
        "budget": {
            "id": "full",
            "analysis_role": SAFETY_ANALYSIS_ROLE,
            "algorithm_seeds": [algorithm_seed["seed"]],
            "execution_order_seed": execution_order_seed["seed"],
            "expected_independent_datasets": SAFETY_EXPECTED_DATASETS,
        },
        "seed_contract": {
            "seed_plan": _file_reference(root, seed_plan_path, "canonical seed plan"),
            "algorithm_seed": algorithm_seed,
            "execution_order_seed": execution_order_seed,
        },
        "reference": source_plan["reference"],
        "analysis": {
            "validation_dataset_role": SAFETY_DATASET_ROLE,
            "split_id": SAFETY_SPLIT_ID,
            "metric": "aupr",
            "level": "directed",
            "primary_target_universe": "full_requested_target_universe",
            "context_aggregation": "equal_weight_mean_within_dataset",
            "independent_unit": "independently_simulated_dataset",
            "statistical_role": "descriptive_veto_without_confidence_intervals_or_p_values",
            "selection_rule": "single_provisional_candidate_no_fallback",
            "condition_scales": panel["condition_scales"],
            "thresholds": {
                **SAFETY_THRESHOLDS,
                **_coverage_thresholds(candidate),
            },
        },
        "candidates": [candidate],
        "interpretation_limit": (
            "This eighteen-dataset, six-condition check is a descriptive pre-freeze safety veto. "
            "It cannot establish equivalence, estimate statistical power, promote a policy "
            "that failed the confirmatory study, or select a fallback candidate."
        ),
    }
    validate_json(
        payload,
        root / "schemas/efficiency-safety-plan.schema.json",
        "efficiency safety plan",
    )
    validate_safety_plan(root, payload)
    return write_idempotent_json(
        output_path,
        payload,
        "efficiency safety plan",
        boundary=root,
    ).resolve()


def _validate_condition_scales(
    repo_root: Path,
    panel: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    scales = panel["condition_scales"]
    if scales != plan["analysis"]["condition_scales"] or set(scales) != set(
        SAFETY_CONDITION_SCALES
    ):
        raise BenchmarkConfigError("Efficiency safety condition scales differ across artifacts")
    topologies: dict[str, tuple[set[str], set[tuple[str, str]]]] = {}
    for condition_id, expected in SAFETY_CONDITION_SCALES.items():
        scale = scales[condition_id]
        if {key: scale[key] for key in expected} != expected:
            raise BenchmarkConfigError(
                f"Efficiency safety condition {condition_id!r} changes its frozen dimensions"
            )
        if sum(scale["group_sizes"]) != scale["cells"]:
            raise BenchmarkConfigError(
                f"Efficiency safety condition {condition_id!r} has inconsistent group sizes"
            )
        scenario_path = repo_root / scale["scenario"]["path"]
        runs_path = repo_root / scale["simulator_runs"]["path"]
        for path, reference, label in (
            (scenario_path, scale["scenario"], f"{condition_id} safety scenario"),
            (runs_path, scale["simulator_runs"], f"{condition_id} safety simulator runs"),
        ):
            _inside(repo_root, path, label)
            if sha256_file(path) != reference["sha256"]:
                raise BenchmarkConfigError(f"{label.title()} differs from the frozen panel")
        runs = load_json_object(runs_path, f"{condition_id} safety simulator runs")["runs"]
        if len(runs) != 1:
            raise BenchmarkConfigError(
                f"Efficiency safety condition {condition_id!r} must declare one run"
            )
        run = runs[0]
        params = run.get("params", {})
        if (
            run.get("simulator_id") != "scmultisim"
            or run.get("replicates") != SAFETY_DATASETS_PER_CONDITION
            or params.get("num_genes") != scale["genes"]
            or params.get("num_cells") != scale["cells"]
            or params.get("population_mode") != "discrete"
            or params.get("discrete_population", {}).get("pop_size") != scale["group_sizes"]
        ):
            raise BenchmarkConfigError(
                f"Efficiency safety run {condition_id!r} changes its frozen scale"
            )
        observed_inputs = scenario_external_input_records(scenario_path, repo_root)
        if scale["scenario_inputs"] != observed_inputs or set(observed_inputs) != {
            "regulatory_network",
            "tree_newick",
        }:
            raise BenchmarkConfigError(
                f"Efficiency safety condition {condition_id!r} changes its external inputs"
            )
        for input_id, record in observed_inputs.items():
            if record["kind"] != "file" or len(record["files"]) != 1:
                raise BenchmarkConfigError(
                    f"Efficiency safety scenario input {condition_id!r}:{input_id!r} "
                    "must bind exactly one file"
                )
        network_relative = observed_inputs["regulatory_network"]["files"][0]["path"]
        network_path = repo_root / network_relative
        try:
            with network_path.open("r", encoding="utf-8", newline="") as handle:
                network_rows = list(csv.DictReader(handle, delimiter="\t"))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise BenchmarkConfigError(
                f"Cannot read {condition_id!r} safety regulatory network: {exc}"
            ) from exc
        edges = {(row.get("target"), row.get("regulator")) for row in network_rows}
        regulators = {regulator for _target, regulator in edges}
        if (
            not network_rows
            or len(edges) != len(network_rows)
            or any(not target or not regulator for target, regulator in edges)
            or None in regulators
            or len(regulators) != scale["transcription_factors"]
        ):
            raise BenchmarkConfigError(
                f"Efficiency safety condition {condition_id!r} has the wrong TF topology"
            )
        if network_relative not in topologies:
            topologies[network_relative] = (set(), set(edges))
        conditions, frozen_edges = topologies[network_relative]
        if frozen_edges != edges:
            raise BenchmarkConfigError("Efficiency safety topology binding is inconsistent")
        conditions.add(condition_id)

    expected_shared_tf10 = {
        "g100_tf10_c500_single_group",
        "g100_tf10_c300_groups_10_40_250",
        "g100_tf10_c10000",
    }
    shared_condition_sets = {frozenset(conditions) for conditions, _edges in topologies.values()}
    if (
        len(topologies) != 4
        or frozenset(expected_shared_tf10) not in shared_condition_sets
        or any(
            len(conditions) != 1 and conditions != expected_shared_tf10
            for conditions, _edges in topologies.values()
        )
    ):
        raise BenchmarkConfigError(
            "Efficiency safety conditions do not bind the four predeclared topologies"
        )
    topology_items = sorted(topologies.items())
    for index, (left_path, (_left_conditions, left_edges)) in enumerate(topology_items):
        for right_path, (_right_conditions, right_edges) in topology_items[index + 1 :]:
            if left_edges & right_edges:
                raise BenchmarkConfigError(
                    "Efficiency safety topologies share directed edges: "
                    f"{left_path} and {right_path}"
                )


def _validate_panel_dataset_identities(panel: Mapping[str, Any]) -> None:
    dataset_ids = [item["dataset_id"] for item in panel["datasets"]]
    generation_seeds = [item["generation_seed"] for item in panel["datasets"]]
    if len(set(dataset_ids)) != SAFETY_EXPECTED_DATASETS:
        raise BenchmarkConfigError("Efficiency safety panel reuses a dataset ID")
    if len(set(generation_seeds)) != SAFETY_EXPECTED_DATASETS:
        raise BenchmarkConfigError("Efficiency safety panel reuses a generation seed")


def validate_safety_plan(repo_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct every source and seed binding in a materialized safety plan."""

    root = repo_root.resolve()
    if not isinstance(plan, dict):
        plan = dict(plan)
    validate_json(
        plan,
        root / "schemas/efficiency-safety-plan.schema.json",
        "efficiency safety plan",
    )
    source_plan_path = root / plan["source_efficiency"]["plan"]["path"]
    source_results_path = root / plan["source_efficiency"]["results"]["path"]
    for path, reference, label in (
        (source_plan_path, plan["source_efficiency"]["plan"], "source efficiency plan"),
        (
            source_results_path,
            plan["source_efficiency"]["results"],
            "source efficiency results",
        ),
    ):
        _inside(root, path, label)
        if sha256_file(path) != reference["sha256"]:
            raise BenchmarkConfigError(f"{label.title()} changed after safety-plan freeze")
    source_plan, _results, adoption = _source_results(
        root,
        efficiency_plan_path=source_plan_path,
        efficiency_results_path=source_results_path,
    )
    candidate = provisional_candidate(
        root,
        efficiency_plan_path=source_plan_path,
        efficiency_results_path=source_results_path,
    )
    if candidate is None:
        raise BenchmarkConfigError("Safety plan exists although exact already won")
    if (
        plan["source_efficiency"]["formal_adoption_sha256"] != canonical_json_sha256(adoption)
        or plan["reference"] != source_plan["reference"]
        or plan["candidates"] != [candidate]
    ):
        raise BenchmarkConfigError("Safety plan changes the frozen provisional decision")

    panel_path = root / plan["panel"]["path"]
    _inside(root, panel_path, "safety panel")
    if sha256_file(panel_path) != plan["panel"]["sha256"]:
        raise BenchmarkConfigError("Safety panel changed after plan materialization")
    panel = load_json_object(panel_path, "efficiency safety panel")
    validate_json(
        panel,
        root / "schemas/efficiency-safety-panel.schema.json",
        "efficiency safety panel",
    )
    if len(panel["datasets"]) != SAFETY_EXPECTED_DATASETS:
        raise BenchmarkConfigError("Efficiency safety panel must contain exactly 18 datasets")
    if (
        panel["selection_validation_manifest_sha256"]
        != source_plan["reference"]["selection_validation_manifest_sha256"]
    ):
        raise BenchmarkConfigError("Safety panel is bound to another calibration selection")
    source_panel = load_json_object(
        root / source_plan["panel"]["path"], "efficiency validation panel"
    )
    source_ids = {item["dataset_id"] for item in source_panel["datasets"]}
    safety_ids = {item["dataset_id"] for item in panel["datasets"]}
    if source_ids & safety_ids:
        raise BenchmarkConfigError("Safety panel reuses efficiency-validation dataset IDs")
    _validate_condition_scales(root, panel, plan)
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for dataset in panel["datasets"]:
        by_condition[dataset["condition_id"]].append(dataset)
    if set(by_condition) != set(SAFETY_CONDITIONS) or any(
        len(members) != SAFETY_DATASETS_PER_CONDITION for members in by_condition.values()
    ):
        raise BenchmarkConfigError(
            "Efficiency safety panel must contain three datasets per frozen condition"
        )
    for condition_id, members in by_condition.items():
        expected_contexts = len(SAFETY_CONDITION_SCALES[condition_id]["group_sizes"])
        if any(len(dataset["contexts"]) != expected_contexts for dataset in members):
            raise BenchmarkConfigError(
                f"Efficiency safety condition {condition_id!r} has the wrong group count"
            )
    _validate_panel_dataset_identities(panel)

    seed_plan_path = root / plan["seed_contract"]["seed_plan"]["path"]
    _inside(root, seed_plan_path, "canonical seed plan")
    if sha256_file(seed_plan_path) != plan["seed_contract"]["seed_plan"]["sha256"]:
        raise BenchmarkConfigError("Canonical seed plan changed after safety-plan freeze")
    seed_plan = load_seed_plan(seed_plan_path)
    algorithm = validate_plan_seed_record(
        seed_plan,
        **plan["seed_contract"]["algorithm_seed"],
        required_namespace=SAFETY_NAMESPACE,
        required_stream="inference",
        required_coordinates={"purpose": "common_algorithm_seed"},
    )
    order = validate_plan_seed_record(
        seed_plan,
        **plan["seed_contract"]["execution_order_seed"],
        required_namespace=SAFETY_NAMESPACE,
        required_stream="execution_order",
        required_coordinates={"purpose": "counterbalanced_execution_order"},
    )
    if (
        plan["budget"]["algorithm_seeds"] != [algorithm["seed"]]
        or plan["budget"]["execution_order_seed"] != order["seed"]
        or plan["analysis"]["thresholds"]
        != {**SAFETY_THRESHOLDS, **_coverage_thresholds(candidate)}
    ):
        raise BenchmarkConfigError("Safety plan seed or threshold contract has drifted")
    return dict(plan)


def load_safety_panel(repo_root: Path, plan: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Load the exact safety panel referenced by an already validated plan."""

    root = repo_root.resolve()
    validate_safety_plan(root, plan)
    reference = plan["panel"]
    panel_path = root / reference["path"]
    _inside(root, panel_path, "efficiency safety panel")
    if sha256_file(panel_path) != reference["sha256"]:
        raise BenchmarkConfigError("Safety panel changed after plan materialization")
    panel = load_json_object(panel_path, "efficiency safety panel")
    validate_json(
        panel,
        root / "schemas/efficiency-safety-panel.schema.json",
        "efficiency safety panel",
    )
    if (
        panel["budget"] != plan["budget"]["id"]
        or panel["dataset_role"] != plan["analysis"]["validation_dataset_role"]
        or panel["split_id"] != plan["analysis"]["split_id"]
        or panel["selection_validation_manifest_sha256"]
        != plan["reference"]["selection_validation_manifest_sha256"]
    ):
        raise BenchmarkConfigError("Efficiency safety panel is not bound to its plan")
    return panel_path, panel


def _analysis_implementation_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/spathi_bench/efficiency_safety.py",
        "src/spathi_bench/noninferiority.py",
        "src/spathi_bench/noninferiority_evidence.py",
        "src/spathi_bench/scenario_inputs.py",
        "schemas/efficiency-safety-plan.schema.json",
        "schemas/efficiency-safety-panel.schema.json",
        "schemas/efficiency-safety-results.schema.json",
        "schemas/noninferiority-evidence.schema.json",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((repo_root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dataset_point_estimates(
    rows: Sequence[Mapping[str, Any]], candidate_id: str
) -> dict[str, tuple[float, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("candidate_id") != candidate_id:
            raise BenchmarkConfigError("Safety evidence contains an unplanned candidate")
        grouped[str(row["dataset_id"])].append(row)
    return {
        dataset_id: (
            fmean(float(row["reference_value"]) for row in members),
            fmean(float(row["candidate_value"]) for row in members),
        )
        for dataset_id, members in grouped.items()
    }


def evaluate_safety_pairs(
    paired: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bool], bool]:
    """Evaluate the frozen descriptive veto without any inferential procedure."""

    expected = int(thresholds["expected_independent_datasets"])
    expected_conditions = tuple(thresholds["expected_conditions"])
    per_condition = int(thresholds["expected_datasets_per_condition"])
    if len(paired) != expected or len({str(row.get("dataset_id")) for row in paired}) != expected:
        raise BenchmarkConfigError(
            f"Safety evidence must contain exactly {expected} distinct dataset pairs"
        )
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired:
        condition_id = row.get("condition_id")
        if condition_id not in expected_conditions:
            raise BenchmarkConfigError("Safety evidence contains an unplanned condition")
        by_condition[str(condition_id)].append(row)
        difference = row.get("aupr_difference")
        ratio = row.get("wall_time_ratio")
        target_fraction = row.get("target_fraction")
        truth_fraction = row.get("truth_edge_fraction")
        numeric = (difference, ratio, target_fraction, truth_fraction)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise BenchmarkConfigError("Safety evidence contains a non-finite numeric value")
        if (
            not -1.0 <= float(difference) <= 1.0
            or float(ratio) <= 0.0
            or not 0.0 <= float(target_fraction) <= 1.0
            or not 0.0 <= float(truth_fraction) <= 1.0
        ):
            raise BenchmarkConfigError("Safety evidence contains a value outside its domain")
        if not isinstance(row.get("reference_complete"), bool) or not isinstance(
            row.get("candidate_complete"), bool
        ):
            raise BenchmarkConfigError("Safety completion indicators must be boolean")
    if set(by_condition) != set(expected_conditions) or any(
        len(rows) != per_condition for rows in by_condition.values()
    ):
        raise BenchmarkConfigError(
            "Safety evidence must contain the exact predeclared condition-by-replicate product"
        )

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        differences = [float(row["aupr_difference"]) for row in rows]
        wall_time_ratios = [float(row["wall_time_ratio"]) for row in rows]
        return {
            "dataset_pairs": len(rows),
            "completed_dataset_pairs": sum(
                bool(row["reference_complete"] and row["candidate_complete"]) for row in rows
            ),
            "mean_aupr_difference": float(fmean(differences)),
            "median_aupr_difference": float(median(differences)),
            "minimum_aupr_difference": min(differences),
            "median_wall_time_ratio": float(median(wall_time_ratios)),
            "geometric_mean_wall_time_ratio": float(
                math.exp(fmean(math.log(value) for value in wall_time_ratios))
            ),
        }

    global_summary = summarize(paired)
    condition_summaries = {
        condition_id: summarize(by_condition[condition_id]) for condition_id in expected_conditions
    }
    summary = {"global": global_summary, "by_condition": condition_summaries}
    gates = {
        "completion_18_of_18": global_summary["completed_dataset_pairs"] == expected,
        "completion_each_condition_3_of_3": all(
            item["completed_dataset_pairs"] == per_condition
            for item in condition_summaries.values()
        ),
        "target_coverage_each_dataset": all(
            float(row["target_fraction"])
            >= float(thresholds["minimum_each_dataset_target_fraction"])
            for row in paired
        ),
        "truth_edge_coverage_each_dataset": all(
            float(row["truth_edge_fraction"])
            >= float(thresholds["minimum_each_dataset_truth_edge_fraction"])
            for row in paired
        ),
        "global_mean_aupr_difference": global_summary["mean_aupr_difference"]
        >= float(thresholds["minimum_mean_aupr_difference"]),
        "global_median_aupr_difference": global_summary["median_aupr_difference"]
        >= float(thresholds["minimum_median_aupr_difference"]),
        "condition_mean_aupr_difference": all(
            item["mean_aupr_difference"] >= float(thresholds["minimum_mean_aupr_difference"])
            for item in condition_summaries.values()
        ),
        "condition_median_aupr_difference": all(
            item["median_aupr_difference"] >= float(thresholds["minimum_median_aupr_difference"])
            for item in condition_summaries.values()
        ),
        "each_dataset_aupr_difference": all(
            float(row["aupr_difference"])
            >= float(thresholds["minimum_each_dataset_aupr_difference"])
            for row in paired
        ),
        "global_median_wall_time_ratio": global_summary["median_wall_time_ratio"]
        < float(thresholds["maximum_median_wall_time_ratio"]),
    }
    return summary, gates, all(gates.values())


def analyze_safety(
    *,
    repo_root: Path,
    plan_path: Path,
    evidence_path: Path,
    output_path: Path,
) -> Path:
    """Apply every descriptive veto and publish the final safety decision."""

    root = repo_root.resolve()
    plan = load_json_object(plan_path, "efficiency safety plan")
    validate_safety_plan(root, plan)
    evidence = verify_noninferiority_evidence(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
    )
    candidate = plan["candidates"][0]
    candidate_id = candidate["id"]
    expected_seeds = set(plan["budget"]["algorithm_seeds"])
    execution_failures = evidence["execution_failures"]
    if any(
        failure["selection_role"] != "candidate" or failure["configuration_id"] != candidate_id
        for failure in execution_failures
    ):
        raise BenchmarkConfigError("Safety evidence reports a failure outside its policy pair")
    coverage_thresholds = plan["analysis"]["thresholds"]
    paired: list[dict[str, Any]] = []
    if execution_failures:
        empty_summary = {
            "dataset_pairs": 0,
            "completed_dataset_pairs": 0,
            "mean_aupr_difference": None,
            "median_aupr_difference": None,
            "minimum_aupr_difference": None,
            "median_wall_time_ratio": None,
            "geometric_mean_wall_time_ratio": None,
        }
        summary = {
            "global": dict(empty_summary),
            "by_condition": {
                condition_id: dict(empty_summary) for condition_id in SAFETY_CONDITIONS
            },
        }
        gates = {
            key: False
            for key in (
                "completion_18_of_18",
                "completion_each_condition_3_of_3",
                "target_coverage_each_dataset",
                "truth_edge_coverage_each_dataset",
                "global_mean_aupr_difference",
                "global_median_aupr_difference",
                "condition_mean_aupr_difference",
                "condition_median_aupr_difference",
                "each_dataset_aupr_difference",
                "global_median_wall_time_ratio",
            )
        }
        passed = False
    else:
        observations = _dataset_observations(
            evidence["rows"],
            expected_seeds=expected_seeds,
            candidate_id=candidate_id,
        )
        expected = plan["analysis"]["thresholds"]["expected_independent_datasets"]
        if len(observations) != expected:
            raise BenchmarkConfigError(
                f"Safety evidence has {len(observations)} datasets, expected {expected}"
            )
        point_estimates = _dataset_point_estimates(evidence["rows"], candidate_id)
        for observation in observations:
            dataset_id = observation["dataset_id"]
            reference_aupr, candidate_aupr = point_estimates[dataset_id]
            target_fraction = (
                1.0 if observation["target_fraction"] is None else observation["target_fraction"]
            )
            truth_fraction = (
                1.0
                if observation["truth_edge_fraction"] is None
                else observation["truth_edge_fraction"]
            )
            reference_complete = (
                observation["reference_run_fraction"] == 1.0
                and observation["reference_model_fraction"] == 1.0
            )
            candidate_complete = (
                observation["candidate_run_fraction"] == 1.0
                and observation["candidate_model_fraction"] == 1.0
            )
            paired.append(
                {
                    "dataset_id": dataset_id,
                    "condition_id": observation["condition_id"],
                    "dataset_manifest_sha256": observation["dataset_manifest_sha256"],
                    "generation_seed": observation["generation_seed"],
                    "algorithm_seed": next(iter(expected_seeds)),
                    "reference_aupr": reference_aupr,
                    "candidate_aupr": candidate_aupr,
                    "aupr_difference": candidate_aupr - reference_aupr,
                    "target_fraction": target_fraction,
                    "truth_edge_fraction": truth_fraction,
                    "reference_complete": reference_complete,
                    "candidate_complete": candidate_complete,
                    "wall_time_ratio": observation["runtime_ratio"],
                }
            )
        paired.sort(key=lambda row: row["dataset_id"])
        summary, gates, passed = evaluate_safety_pairs(paired, coverage_thresholds)
    selected = candidate if passed else plan["reference"]
    payload = {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_safety_results",
        "study_phase": "efficiency_safety",
        "budget": "full",
        "analysis_role": SAFETY_ANALYSIS_ROLE,
        "dataset_role": SAFETY_DATASET_ROLE,
        "inputs": {
            "plan_sha256": sha256_file(plan_path),
            "evidence_sha256": sha256_file(evidence_path),
        },
        "analysis_implementation_sha256": _analysis_implementation_sha256(root),
        "reference_policy_id": plan["reference"]["policy_id"],
        "provisional_candidate_id": candidate_id,
        "provisional_candidate_policy_sha256": candidate["policy_sha256"],
        "condition_scales": plan["analysis"]["condition_scales"],
        "thresholds": coverage_thresholds,
        "execution_failures": execution_failures,
        "paired_datasets": paired,
        "summary": summary,
        "gates": gates,
        "decision": {
            "status": "candidate_authorized" if passed else "reference_retained_safety_veto",
            "selection_rule": "single_provisional_candidate_no_fallback",
            "selected_policy_id": selected["id"] if passed else selected["policy_id"],
            "selected_policy_sha256": selected["policy_sha256"],
            "selected_group_policy": selected["group_policy"],
            "reason": (
                "The sole provisional candidate passed every pre-freeze descriptive "
                "safety and operational veto."
                if passed
                else (
                    "A bound SPATHI execution failed during efficiency safety; exact SPATHI "
                    "is retained and no fallback candidate is tried."
                    if execution_failures
                    else "The sole provisional candidate failed at least one pre-freeze "
                    "descriptive veto; exact SPATHI is retained and no fallback candidate is tried."
                )
            ),
        },
        "interpretation_limit": plan["interpretation_limit"],
    }
    validate_json(
        payload,
        root / "schemas/efficiency-safety-results.schema.json",
        "efficiency safety results",
    )
    return write_idempotent_json(
        output_path,
        payload,
        "efficiency safety results",
        boundary=root,
    ).resolve()


__all__ = [
    "SAFETY_ANALYSIS_ROLE",
    "SAFETY_CONDITIONS",
    "SAFETY_CONDITION_SCALES",
    "SAFETY_DATASET_ROLE",
    "SAFETY_DATASETS_PER_CONDITION",
    "SAFETY_EXPECTED_DATASETS",
    "SAFETY_NAMESPACE",
    "SAFETY_SPLIT_ID",
    "SAFETY_THRESHOLDS",
    "analyze_safety",
    "evaluate_safety_pairs",
    "materialize_safety_plan",
    "load_safety_panel",
    "provisional_candidate",
    "validate_safety_plan",
]
