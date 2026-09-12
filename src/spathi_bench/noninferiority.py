"""Predeclared paired non-inferiority analysis for SPATHI efficiency policies.

The study consumes an explicit paired-evidence artifact. A materialized plan
merges public ``spathi.SpathiConfig`` overrides into one frozen calibration
winner and records complete, content-addressed execution parameters.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .calibration import locked_spathi_identity, resolve_group_policy
from .config import BenchmarkConfigError, load_json_object, validate_json
from .noninferiority_evidence import verify_noninferiority_evidence
from .safe_io import write_idempotent_json
from .seeds import (
    derive_plan_seed,
    load_seed_plan,
    plan_seed_record,
    validate_plan_seed_record,
)

STUDY_RELATIVE_PATH = "experiments/v1/spathi/efficiency-study.json"
RESULT_SCHEMA_VERSION = "1.0"
ANALYSIS_SCHEMA_PATHS = (
    "schemas/efficiency-design-sensitivity.schema.json",
    "schemas/noninferiority-study.schema.json",
    "schemas/noninferiority-plan.schema.json",
    "schemas/noninferiority-panel.schema.json",
    "schemas/noninferiority-artifact-index.schema.json",
    "schemas/noninferiority-evidence.schema.json",
    "schemas/noninferiority-results.schema.json",
)
TARGET_ELIGIBILITY_PARAMETERS = frozenset(
    {
        "target_eligibility",
        "min_target_detected_cells",
        "min_target_detected_fraction",
        "min_target_weighted_detected_fraction",
        "min_target_weighted_detected_ess",
    }
)
ADAPTIVE_TREE_PARAMETERS = frozenset(
    {
        "adaptive_trees",
        "adaptive_min_estimators",
        "adaptive_tree_step",
        "adaptive_tolerance",
        "adaptive_patience",
    }
)


def _relative_to_repository(repo_root: Path, path: Path, label: str) -> str:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise BenchmarkConfigError(
            f"{label} must be stored inside the benchmark repository"
        ) from exc


def _load_validated(
    path: Path,
    *,
    schema_path: Path,
    label: str,
) -> dict[str, Any]:
    payload = load_json_object(path, label)
    validate_json(payload, schema_path, label)
    return payload


def load_noninferiority_study(repo_root: Path, study_path: Path | None = None) -> dict[str, Any]:
    """Load the committed study and enforce its cross-field scientific contract."""

    root = repo_root.resolve()
    path = (study_path or root / STUDY_RELATIVE_PATH).resolve()
    _relative_to_repository(root, path, "Non-inferiority study")
    study = _load_validated(
        path,
        schema_path=root / "schemas/noninferiority-study.schema.json",
        label="SPATHI efficiency non-inferiority study",
    )
    candidates = study["validation_components"]
    identifiers = [candidate["id"] for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise BenchmarkConfigError("Non-inferiority study contains duplicate candidate IDs")
    by_component = {candidate["component"]: candidate for candidate in candidates}
    if set(by_component) != {"target_eligibility", "adaptive_trees", "combined"}:
        raise BenchmarkConfigError(
            "Non-inferiority study must define target_eligibility, adaptive_trees and combined"
        )
    if any(
        candidate["primary_evaluation_policy"] != "full_requested_universe"
        for candidate in candidates
    ):
        raise BenchmarkConfigError(
            "Every efficiency candidate must use the full requested universe for primary accuracy"
        )
    if by_component["adaptive_trees"]["intersection_diagnostic"]:
        raise BenchmarkConfigError("Adaptive trees cannot declare an eligibility intersection")
    for component in ("target_eligibility", "combined"):
        if not by_component[component]["intersection_diagnostic"]:
            raise BenchmarkConfigError(
                f"{component} must retain the eligible-target intersection as a diagnostic"
            )
    expected_parameters = {
        "target_eligibility": TARGET_ELIGIBILITY_PARAMETERS,
        "adaptive_trees": ADAPTIVE_TREE_PARAMETERS,
        "combined": TARGET_ELIGIBILITY_PARAMETERS | ADAPTIVE_TREE_PARAMETERS,
    }
    for component, candidate in by_component.items():
        bound = set(candidate["binding"]["parameters"])
        if bound != expected_parameters[component]:
            raise BenchmarkConfigError(
                f"{component} binding does not exactly match its development-grid parameters"
            )
    eligibility_gates = by_component["target_eligibility"]["coverage_gates"]
    if by_component["combined"]["coverage_gates"] != eligibility_gates:
        raise BenchmarkConfigError(
            "Combined candidate must inherit every target-eligibility coverage guardrail"
        )
    if (
        study["analysis"]["margin_basis"]["value"]
        != study["analysis"]["absolute_noninferiority_margin"]
    ):
        raise BenchmarkConfigError("Efficiency AUPR margin differs from its declared basis")
    if study["budgets"]["reduced"]["analysis_role"] != "descriptive":
        raise BenchmarkConfigError("Reduced non-inferiority budget must remain descriptive")
    if study["budgets"]["full"]["analysis_role"] != "confirmatory":
        raise BenchmarkConfigError("Full non-inferiority budget must remain confirmatory")
    if (
        study["budgets"]["reduced"]["replicates_per_stratum"] != 1
        or study["budgets"]["full"]["replicates_per_stratum"] != 5
        or study["budgets"]["reduced"]["expected_independent_datasets"] != 5
        or study["budgets"]["full"]["expected_independent_datasets"] != 25
    ):
        raise BenchmarkConfigError(
            "Efficiency budgets must retain the frozen one- and five-replicate stratum designs"
        )
    seed_plan = load_seed_plan(root / "experiments/v1/seeds/seed-plan.json")
    for budget_id, budget in study["budgets"].items():
        namespace = f"{budget_id}.efficiency_validation"
        expected_algorithm_seeds = [
            derive_plan_seed(
                seed_plan,
                namespace=namespace,
                stream="inference",
                coordinates={
                    "purpose": "algorithm_replicate",
                    "replicate_index": index,
                },
            )
            for index in range(1, len(budget["algorithm_seeds"]) + 1)
        ]
        expected_execution_order_seed = derive_plan_seed(
            seed_plan,
            namespace=namespace,
            stream="execution_order",
            coordinates={"purpose": "counterbalanced_execution_order"},
        )
        if (
            budget["algorithm_seeds"] != expected_algorithm_seeds
            or budget["execution_order_seed"] != expected_execution_order_seed
        ):
            raise BenchmarkConfigError(
                f"{budget_id.title()} efficiency seeds differ from the canonical seed plan"
            )
        if budget["analysis_role"] == "confirmatory":
            expected_statistical_seed = derive_plan_seed(
                seed_plan,
                namespace=namespace,
                stream="statistical",
                coordinates={"purpose": "noninferiority_analysis"},
            )
            if budget.get("statistical_seed") != expected_statistical_seed:
                raise BenchmarkConfigError(
                    f"{budget_id.title()} efficiency statistical seed differs from the "
                    "canonical seed plan"
                )
        elif "statistical_seed" in budget:
            raise BenchmarkConfigError(
                f"{budget_id.title()} descriptive efficiency budget cannot reserve a "
                "statistical seed"
            )
    phases = study["phases"]
    if (
        phases["policy_predeclaration"]["uses_outcome_data"]
        or phases["efficiency_development"]["dataset_role"] != "efficiency_development"
        or phases["efficiency_validation"]["dataset_role"]
        != study["analysis"]["validation_dataset_role"]
    ):
        raise BenchmarkConfigError(
            "Efficiency policies must be predeclared before outcome data are accessed"
        )
    from .design_sensitivity import load_design_sensitivity_contract
    from .efficiency_development import load_efficiency_development_contract

    development = load_efficiency_development_contract(root)
    if phases["efficiency_development"]["input"] != (
        "experiments/v1/spathi/efficiency-development.json"
    ):
        raise BenchmarkConfigError("Efficiency-development phase points to the wrong contract")
    if set(development["budgets"]) != set(study["budgets"]):
        raise BenchmarkConfigError("Efficiency development and validation budgets differ")
    load_design_sensitivity_contract(root, study=study)
    return study


def _selected_policy(repo_root: Path, selected_config_path: Path, budget: str) -> dict[str, Any]:
    root = repo_root.resolve()
    selected_path = selected_config_path.resolve()
    _relative_to_repository(root, selected_path, "Selected configuration")
    selected = _load_validated(
        selected_path,
        schema_path=root / "schemas/spathi-selected-config.schema.json",
        label="frozen SPATHI selected configuration",
    )
    if selected.get("status") != "selected" or selected.get("phase") != "frozen":
        raise BenchmarkConfigError("Non-inferiority planning requires a frozen SPATHI selection")
    if selected.get("calibration_budget") != budget:
        raise BenchmarkConfigError(
            f"Selected configuration uses {selected.get('calibration_budget')!r}, not {budget!r}"
        )
    policy = selected.get("selected_policy")
    if not isinstance(policy, dict) or not isinstance(policy.get("group_policy"), dict):
        raise BenchmarkConfigError("Frozen SPATHI selection contains no selected policy")
    if policy.get("policy_sha256") != canonical_json_sha256(policy["group_policy"]):
        raise BenchmarkConfigError("Frozen SPATHI selection has a stale policy digest")
    evidence = selected.get("selection_evidence")
    if not isinstance(evidence, dict) or (
        evidence.get("winner_policy_sha256") != policy["policy_sha256"]
        or evidence.get("winner_candidate_id") != policy.get("source_candidate_id")
        or evidence.get("winner_configuration_sha256") != policy.get("source_configuration_sha256")
    ):
        raise BenchmarkConfigError("Frozen SPATHI policy differs from its selection evidence")
    return selected


def _policy_with_overrides(
    group_policy: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply one efficiency policy to every calibrated group-count branch."""

    result = deepcopy(dict(group_policy))
    branches = result.get("branches")
    if not isinstance(branches, dict) or set(branches) != {"single_group", "multi_group"}:
        raise BenchmarkConfigError("Frozen SPATHI group policy has invalid branches")
    for branch_id, branch in branches.items():
        if not isinstance(branch, dict) or not isinstance(branch.get("parameters"), dict):
            raise BenchmarkConfigError(
                f"Frozen SPATHI group policy branch {branch_id!r} has no parameters"
            )
        unknown = sorted(set(overrides) - set(branch["parameters"]))
        if unknown:
            raise BenchmarkConfigError(
                f"Efficiency overrides are not SPATHI parameters in {branch_id!r}: {unknown}"
            )
        branch["parameters"] = {**branch["parameters"], **dict(overrides)}
    return result


def _validate_selection_partition(
    repo_root: Path,
    *,
    panel: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> frozenset[str]:
    """Verify and expose the dataset IDs used to select the frozen reference."""

    reference = panel["selection_validation_ranking"]
    ranking_path = (repo_root / reference["path"]).resolve()
    _relative_to_repository(repo_root, ranking_path, "Selection validation ranking")
    if (
        not ranking_path.is_file()
        or sha256_file(ranking_path) != reference["sha256"]
        or reference["sha256"] != selected["selection_evidence"]["validation_ranking_sha256"]
    ):
        raise BenchmarkConfigError("Panel does not link the frozen selection ranking exactly")
    ranking = load_json_object(ranking_path, "selection validation ranking")
    rows = ranking.get("calibration_rows")
    if (
        ranking.get("stage_id") != "validation"
        or ranking.get("dataset_role") != "validation"
        or not isinstance(rows, list)
        or not rows
    ):
        raise BenchmarkConfigError("Selection ranking has no verifiable validation partition")
    dataset_ids = frozenset(
        str(row.get("dataset_id"))
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("dataset_id"), str)
    )
    if not dataset_ids:
        raise BenchmarkConfigError("Selection ranking contains no validation dataset IDs")
    return dataset_ids


def _validate_panel_replication(
    repo_root: Path,
    *,
    panel: Mapping[str, Any],
    budget_id: str,
    budget: Mapping[str, Any],
) -> None:
    """Require the panel to realize every frozen simulator-condition stratum exactly."""

    from .design_sensitivity import efficiency_profile_strata

    strata = efficiency_profile_strata(repo_root, budget_id)
    expected = Counter(
        {
            (str(stratum["simulator_id"]), str(stratum["condition_id"])): int(stratum["replicates"])
            for stratum in strata
        }
    )
    observed = Counter(
        (str(dataset["simulator_id"]), str(dataset["condition_id"]))
        for dataset in panel["datasets"]
    )
    expected_total = sum(expected.values())
    if expected_total != int(budget["expected_independent_datasets"]):
        raise BenchmarkConfigError("Efficiency profile cardinality differs from the study budget")
    if any(count != int(budget["replicates_per_stratum"]) for count in expected.values()):
        raise BenchmarkConfigError(
            "Efficiency profile does not use the study's frozen per-stratum replication"
        )
    if observed != expected:
        raise BenchmarkConfigError(
            "Predeclared panel does not realize the exact simulator-condition replication "
            f"contract: expected {dict(expected)}, observed {dict(observed)}"
        )


def materialize_noninferiority_plan(
    *,
    repo_root: Path,
    selected_config_path: Path,
    panel_path: Path,
    development_selection_path: Path,
    budget: str,
    output_path: Path,
    study_path: Path | None = None,
) -> Path:
    """Bind frozen efficiency policies to one frozen calibration winner."""

    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError("Non-inferiority budget must be 'reduced' or 'full'")
    root = repo_root.resolve()
    study_file = (study_path or root / STUDY_RELATIVE_PATH).resolve()
    study = load_noninferiority_study(root, study_file)
    selected_path = selected_config_path.resolve()
    selected = _selected_policy(root, selected_path, budget)
    reference = selected["selected_policy"]
    reference_policy = reference["group_policy"]
    reference_digest = reference["policy_sha256"]
    from .efficiency_development import load_efficiency_development_selection

    development_selection_file = development_selection_path.resolve()
    _relative_to_repository(root, development_selection_file, "Efficiency-development selection")
    development_selection = load_efficiency_development_selection(
        root,
        development_selection_file,
        budget=budget,
        selected_config_path=selected_path,
    )
    execution_identity = locked_spathi_identity(root)
    selected_identity = {
        "spathi_commit": reference["spathi_commit"],
        "spathi_build_fingerprint": selected["selection_evidence"]["spathi_build_fingerprint"],
    }
    if selected_identity != {key: execution_identity[key] for key in selected_identity}:
        raise BenchmarkConfigError(
            "The frozen selection predates the current SPATHI build; recalibrate with both "
            "efficiency policies disabled before validating them"
        )
    panel_file = panel_path.resolve()
    _relative_to_repository(root, panel_file, "Predeclared non-inferiority panel")
    panel = _load_validated(
        panel_file,
        schema_path=root / "schemas/noninferiority-panel.schema.json",
        label="predeclared non-inferiority panel",
    )
    if panel["budget"] != budget or panel["dataset_role"] != "efficiency_validation":
        raise BenchmarkConfigError("Predeclared panel does not match the selected budget/role")
    if (
        panel["selection_validation_manifest_sha256"]
        != selected["selection_evidence"]["validation_manifest_sha256"]
    ):
        raise BenchmarkConfigError("Predeclared panel uses a different validation manifest")
    selection_dataset_ids = _validate_selection_partition(root, panel=panel, selected=selected)
    expected_datasets = int(study["budgets"][budget]["expected_independent_datasets"])
    if len(panel["datasets"]) != expected_datasets:
        raise BenchmarkConfigError(
            f"Predeclared panel must contain exactly {expected_datasets} independent datasets"
        )
    _validate_panel_replication(
        root,
        panel=panel,
        budget_id=budget,
        budget=study["budgets"][budget],
    )
    dataset_ids = [item["dataset_id"] for item in panel["datasets"]]
    overlap = sorted(set(dataset_ids) & selection_dataset_ids)
    if overlap:
        raise BenchmarkConfigError(f"Efficiency validation reuses selection datasets: {overlap}")
    independent_units = [
        (item["simulator_id"], item["generation_seed"]) for item in panel["datasets"]
    ]
    if len(dataset_ids) != len(set(dataset_ids)) or len(independent_units) != len(
        set(independent_units)
    ):
        raise BenchmarkConfigError(
            "Predeclared panel contains duplicate datasets/independent units"
        )
    required_reference = study["reference"]["required_execution_parameters"]
    mismatches = {
        f"{branch_id}.{key}": (branch["parameters"].get(key), expected)
        for branch_id, branch in reference_policy["branches"].items()
        for key, expected in required_reference.items()
        if branch["parameters"].get(key) != expected
    }
    if mismatches:
        raise BenchmarkConfigError(
            f"Frozen SPATHI reference enables an efficiency policy: {mismatches}"
        )
    candidates: list[dict[str, Any]] = []
    raw = development_selection["nominee"]
    if raw is not None:
        template = next(
            (
                item
                for item in study["validation_components"]
                if item["component"] == raw["component"]
            ),
            None,
        )
        if template is None:
            raise BenchmarkConfigError(
                f"Efficiency nominee component {raw['component']!r} is not declared for validation"
            )
        overrides = raw["execution_overrides"]
        override_digest = canonical_json_sha256(overrides)
        group_policy = _policy_with_overrides(reference_policy, overrides)
        for branch_id, branch in group_policy["branches"].items():
            parameters = branch["parameters"]
            if parameters["adaptive_trees"] and (
                parameters["adaptive_min_estimators"] >= parameters["n_estimators"]
            ):
                raise BenchmarkConfigError(
                    f"Candidate {raw['id']!r}/{branch_id} requires "
                    "adaptive_min_estimators below the frozen n_estimators maximum"
                )
        candidate = {
            **template,
            "id": raw["id"],
            "validation_component_id": template["id"],
            "execution_overrides": overrides,
            "development_candidate_sha256": raw["candidate_sha256"],
            "development_result": raw["development_result"],
            "execution_overrides_sha256": override_digest,
            "group_policy": group_policy,
            "policy_sha256": canonical_json_sha256(group_policy),
        }
        candidates.append(candidate)
    plan = {
        "schema_version": "1.0",
        "artifact_type": "spathi_efficiency_noninferiority_plan",
        "study_phase": "efficiency_validation",
        "study": {
            "path": _relative_to_repository(root, study_file, "Non-inferiority study"),
            "sha256": sha256_file(study_file),
        },
        "design_sensitivity": {
            "path": study["analysis"]["design_sensitivity_contract"]["path"],
            "sha256": sha256_file(root / study["analysis"]["design_sensitivity_contract"]["path"]),
        },
        "development_selection": {
            "path": _relative_to_repository(
                root,
                development_selection_file,
                "Efficiency-development selection",
            ),
            "sha256": sha256_file(development_selection_file),
        },
        "panel": {
            "path": _relative_to_repository(root, panel_file, "Predeclared panel"),
            "sha256": sha256_file(panel_file),
        },
        "budget": {"id": budget, **study["budgets"][budget]},
        "reference": {
            "selected_config": {
                "path": _relative_to_repository(root, selected_path, "Selected configuration"),
                "sha256": sha256_file(selected_path),
            },
            "policy_id": reference["policy_id"],
            "policy_sha256": reference_digest,
            "group_policy": reference_policy,
            "source_candidate_id": reference["source_candidate_id"],
            "source_configuration_sha256": reference["source_configuration_sha256"],
            "spathi_commit": reference["spathi_commit"],
            "spathi_build_fingerprint": selected["selection_evidence"]["spathi_build_fingerprint"],
            "spathi_image_id": selected["selection_evidence"]["spathi_image_id"],
            "selection_validation_manifest_sha256": selected["selection_evidence"][
                "validation_manifest_sha256"
            ],
        },
        "analysis": study["analysis"],
        "candidates": candidates,
    }
    validate_json(
        plan,
        root / "schemas/noninferiority-plan.schema.json",
        "materialized non-inferiority plan",
    )
    _relative_to_repository(root, output_path, "Non-inferiority plan output")
    return write_idempotent_json(output_path, plan, "non-inferiority plan").resolve()


def _validate_plan_binding(repo_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Reconcile a materialized plan with every source it claims to freeze."""

    root = repo_root.resolve()
    study_reference = plan["study"]
    study_path = root / study_reference["path"]
    study = load_noninferiority_study(root, study_path)
    if sha256_file(study_path) != study_reference["sha256"]:
        raise BenchmarkConfigError("Non-inferiority study changed after plan materialization")
    budget_id = plan["budget"]["id"]
    if plan["budget"] != {"id": budget_id, **study["budgets"][budget_id]}:
        raise BenchmarkConfigError("Non-inferiority plan budget differs from its study")
    if plan["analysis"] != study["analysis"]:
        raise BenchmarkConfigError("Non-inferiority plan analysis differs from its study")
    if plan["study_phase"] != study["analysis"]["study_phase"]:
        raise BenchmarkConfigError("Non-inferiority plan targets the wrong study phase")
    sensitivity_reference = plan["design_sensitivity"]
    sensitivity_path = root / sensitivity_reference["path"]
    if (
        sensitivity_reference["path"] != study["analysis"]["design_sensitivity_contract"]["path"]
        or sha256_file(sensitivity_path) != sensitivity_reference["sha256"]
    ):
        raise BenchmarkConfigError(
            "Prospective efficiency sensitivity contract changed after plan materialization"
        )

    development_reference = plan["development_selection"]
    development_path = root / development_reference["path"]
    from .efficiency_development import load_efficiency_development_selection

    development = load_efficiency_development_selection(
        root,
        development_path,
        budget=budget_id,
        selected_config_path=root / plan["reference"]["selected_config"]["path"],
    )
    if sha256_file(development_path) != development_reference["sha256"]:
        raise BenchmarkConfigError(
            "Efficiency-development selection changed after plan materialization"
        )

    panel_reference = plan["panel"]
    panel_path = root / panel_reference["path"]
    panel = _load_validated(
        panel_path,
        schema_path=root / "schemas/noninferiority-panel.schema.json",
        label="predeclared non-inferiority panel",
    )
    if sha256_file(panel_path) != panel_reference["sha256"]:
        raise BenchmarkConfigError("Predeclared panel changed after plan materialization")
    if (
        panel["budget"] != budget_id
        or panel["dataset_role"] != plan["analysis"]["validation_dataset_role"]
        or panel["selection_validation_manifest_sha256"]
        != plan["reference"]["selection_validation_manifest_sha256"]
    ):
        raise BenchmarkConfigError("Predeclared panel differs from its plan")
    expected_datasets = int(plan["budget"]["expected_independent_datasets"])
    if len(panel["datasets"]) != expected_datasets:
        raise BenchmarkConfigError(
            f"Predeclared panel must contain exactly {expected_datasets} independent datasets"
        )
    _validate_panel_replication(
        root,
        panel=panel,
        budget_id=budget_id,
        budget=plan["budget"],
    )

    selected_reference = plan["reference"]["selected_config"]
    selected_path = root / selected_reference["path"]
    selected = _selected_policy(root, selected_path, budget_id)
    if sha256_file(selected_path) != selected_reference["sha256"]:
        raise BenchmarkConfigError("Frozen SPATHI selection changed after plan materialization")
    selection_dataset_ids = _validate_selection_partition(root, panel=panel, selected=selected)
    overlap = sorted({item["dataset_id"] for item in panel["datasets"]} & selection_dataset_ids)
    if overlap:
        raise BenchmarkConfigError(f"Efficiency validation reuses selection datasets: {overlap}")
    reference = selected["selected_policy"]
    execution_identity = locked_spathi_identity(root)
    if (
        reference["spathi_commit"] != execution_identity["spathi_commit"]
        or selected["selection_evidence"]["spathi_build_fingerprint"]
        != execution_identity["spathi_build_fingerprint"]
    ):
        raise BenchmarkConfigError(
            "The frozen selection and current SPATHI implementation are not identical"
        )
    expected_reference = {
        "policy_id": reference["policy_id"],
        "policy_sha256": reference["policy_sha256"],
        "group_policy": reference["group_policy"],
        "source_candidate_id": reference["source_candidate_id"],
        "source_configuration_sha256": reference["source_configuration_sha256"],
        "spathi_commit": reference["spathi_commit"],
        "spathi_build_fingerprint": selected["selection_evidence"]["spathi_build_fingerprint"],
        "spathi_image_id": selected["selection_evidence"]["spathi_image_id"],
        "selection_validation_manifest_sha256": selected["selection_evidence"][
            "validation_manifest_sha256"
        ],
    }
    if any(plan["reference"].get(key) != value for key, value in expected_reference.items()):
        raise BenchmarkConfigError("Non-inferiority plan references a different SPATHI winner")

    planned = {candidate["component"]: candidate for candidate in plan["candidates"]}
    declared = {candidate["component"]: candidate for candidate in study["validation_components"]}
    nominee = development["nominee"]
    selected = {} if nominee is None else {nominee["component"]: nominee}
    if set(planned) != set(selected) or not set(planned) <= set(declared):
        raise BenchmarkConfigError(
            "Non-inferiority plan components differ from development selection or study"
        )
    for component, candidate in planned.items():
        template = declared[component]
        raw = selected[component]
        expected_template = {
            **template,
            "id": raw["id"],
            "validation_component_id": template["id"],
        }
        if any(candidate.get(key) != value for key, value in expected_template.items()):
            raise BenchmarkConfigError(
                f"Non-inferiority plan component {component!r} differs from its study"
            )
        override_digest = canonical_json_sha256(raw["execution_overrides"])
        group_policy = _policy_with_overrides(reference["group_policy"], raw["execution_overrides"])
        policy_digest = canonical_json_sha256(group_policy)
        if (
            candidate["execution_overrides_sha256"] != override_digest
            or candidate["development_candidate_sha256"] != raw["candidate_sha256"]
            or candidate["development_result"] != raw["development_result"]
            or candidate["group_policy"] != group_policy
            or candidate["policy_sha256"] != policy_digest
        ):
            raise BenchmarkConfigError(
                f"Non-inferiority plan component {component!r} has stale digests"
            )
        for branch_id in ("single_group", "multi_group"):
            candidate_parameters = candidate["group_policy"]["branches"][branch_id]["parameters"]
            reference_parameters = reference["group_policy"]["branches"][branch_id]["parameters"]
            if candidate_parameters["n_estimators"] != reference_parameters["n_estimators"]:
                raise BenchmarkConfigError(
                    f"Non-inferiority candidate {candidate['id']!r}/{branch_id} changes the "
                    "tree-count maximum"
                )
    return study


def _analysis_implementation_sha256(repo_root: Path) -> str:
    """Fingerprint the analyzer and every schema that constrains its inputs/outputs."""

    digest = hashlib.sha256()
    files = (
        ("src/spathi_bench/noninferiority.py", Path(__file__).resolve()),
        (
            "src/spathi_bench/design_sensitivity.py",
            Path(__file__).with_name("design_sensitivity.py").resolve(),
        ),
        (
            "src/spathi_bench/noninferiority_evidence.py",
            Path(__file__).with_name("noninferiority_evidence.py").resolve(),
        ),
    ) + tuple((relative, repo_root / relative) for relative in ANALYSIS_SCHEMA_PATHS)
    for relative, path in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _hierarchical_mean(observations: Sequence[dict[str, Any]], field: str) -> float:
    by_simulator: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        by_simulator[observation["simulator_id"]][observation["condition_id"]].append(
            float(observation[field])
        )
    simulator_means = [
        fmean(fmean(values) for values in conditions.values())
        for conditions in by_simulator.values()
    ]
    return fmean(simulator_means)


def _hierarchical_bootstrap(
    observations: Sequence[dict[str, Any]],
    field: str,
    *,
    resamples: int,
    random_seed: int,
) -> list[float]:
    """Resample independent datasets within fixed simulator/condition strata.

    Simulator and condition strata remain fixed so large panels cannot dominate
    smaller families. Algorithmic seeds have already been averaged within each
    independent simulated dataset and are never treated as biological replicates.
    """

    strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        strata[observation["simulator_id"]][observation["condition_id"]].append(
            float(observation[field])
        )
    rng = random.Random(random_seed)
    estimates: list[float] = []
    for _ in range(resamples):
        simulator_means: list[float] = []
        for conditions in strata.values():
            condition_means: list[float] = []
            for values in conditions.values():
                sampled = [values[rng.randrange(len(values))] for _ in values]
                condition_means.append(fmean(sampled))
            simulator_means.append(fmean(condition_means))
        estimates.append(fmean(simulator_means))
    return estimates


def _same_across_rows(rows: Sequence[dict[str, Any]], field: str, label: str) -> Any:
    values = {json.dumps(row[field], sort_keys=True, separators=(",", ":")) for row in rows}
    if len(values) != 1:
        raise BenchmarkConfigError(f"{label} differs across algorithmic seeds")
    return rows[0][field]


def _validate_candidate_space(
    row: Mapping[str, Any],
    *,
    policy: str,
    intersection_diagnostic: bool,
    candidate_id: str,
) -> None:
    space = row["candidate_space"]
    if space["policy"] != policy:
        raise BenchmarkConfigError(
            f"Evidence for {candidate_id!r} uses candidate-space policy {space['policy']!r}"
        )
    if space["reference_source_universe_sha256"] != space["candidate_source_universe_sha256"]:
        raise BenchmarkConfigError(
            f"Evidence for {candidate_id!r} changes the TF predictor universe"
        )
    if space["n_reference_sources"] != space["n_candidate_sources"]:
        raise BenchmarkConfigError(f"Evidence for {candidate_id!r} changes the TF predictor count")
    if space["omitted_target_prediction_policy"] != "minimum_score_zero":
        raise BenchmarkConfigError(
            f"Evidence for {candidate_id!r} does not score omitted targets at the minimum"
        )
    reference_target = space["reference_target_universe_sha256"]
    candidate_target = space["candidate_target_universe_sha256"]
    reference_evaluation_target = space["reference_evaluation_target_universe_sha256"]
    candidate_evaluation_target = space["candidate_evaluation_target_universe_sha256"]
    reference_count = space["n_reference_targets"]
    candidate_count = space["n_candidate_targets"]
    reference_evaluation_count = space["n_reference_evaluation_targets"]
    candidate_evaluation_count = space["n_candidate_evaluation_targets"]
    coverage = row["target_coverage"]
    if not (
        reference_evaluation_target == candidate_evaluation_target
        and space["reference_evaluation_candidate_space_sha256"]
        == space["candidate_evaluation_candidate_space_sha256"]
        and reference_evaluation_count == candidate_evaluation_count
        and space["n_reference_evaluation_candidates"] == space["n_candidate_evaluation_candidates"]
    ):
        raise BenchmarkConfigError(
            f"Paired evidence for {candidate_id!r} uses different evaluation candidate spaces"
        )
    expected_reference_candidates = (
        space["n_reference_sources"] * reference_evaluation_count
        - space["n_reference_evaluation_source_target_overlaps"]
    )
    expected_candidate_candidates = (
        space["n_candidate_sources"] * candidate_evaluation_count
        - space["n_candidate_evaluation_source_target_overlaps"]
    )
    if space["n_reference_evaluation_source_target_overlaps"] > min(
        space["n_reference_sources"], reference_evaluation_count
    ) or space["n_candidate_evaluation_source_target_overlaps"] > min(
        space["n_candidate_sources"], candidate_evaluation_count
    ):
        raise BenchmarkConfigError(
            f"Evidence for {candidate_id!r} reports impossible source-target overlaps"
        )
    if (
        space["n_reference_evaluation_candidates"] != expected_reference_candidates
        or space["n_candidate_evaluation_candidates"] != expected_candidate_candidates
    ):
        raise BenchmarkConfigError(
            f"Evidence for {candidate_id!r} is not the complete directed TF-by-target "
            "candidate space excluding self-edges"
        )
    outside_truth_count = space["n_truth_edges_outside_candidate_space"]
    outside_truth_examples = space["truth_outside_candidate_space_examples"]
    if len(outside_truth_examples) > outside_truth_count or (
        outside_truth_count == 0 and outside_truth_examples
    ):
        raise BenchmarkConfigError(
            f"Evidence for {candidate_id!r} has inconsistent out-of-space truth diagnostics"
        )
    if policy != "full_requested_universe" or not (
        reference_target == reference_evaluation_target == candidate_evaluation_target
        and reference_count == reference_evaluation_count == candidate_evaluation_count
    ):
        raise BenchmarkConfigError(
            f"Primary accuracy for {candidate_id!r} is not evaluated on the full requested universe"
        )
    if candidate_count > reference_count:
        raise BenchmarkConfigError(
            f"Eligibility evidence for {candidate_id!r} contains more targets than its reference"
        )
    diagnostic = row["intersection_diagnostic"]
    if intersection_diagnostic:
        if coverage is None:
            raise BenchmarkConfigError(f"Eligibility evidence for {candidate_id!r} lacks coverage")
        if coverage["n_reference_truth_edges"] != space["n_truth_edges_in_candidate_space"]:
            raise BenchmarkConfigError(
                f"Eligibility evidence for {candidate_id!r} uses a different evaluable truth set"
            )
        if coverage["n_retained_truth_edges"] > coverage["n_reference_truth_edges"]:
            raise BenchmarkConfigError(
                f"Eligibility evidence for {candidate_id!r} retains impossible truth-edge counts"
            )
        if diagnostic is None:
            if candidate_count and coverage["n_retained_truth_edges"]:
                raise BenchmarkConfigError(
                    f"Eligibility evidence for {candidate_id!r} omits an evaluable diagnostic"
                )
        elif (
            diagnostic["target_universe_sha256"] != candidate_target
            or diagnostic["n_targets"] != candidate_count
            or diagnostic["n_truth_edges"] != coverage["n_retained_truth_edges"]
        ):
            raise BenchmarkConfigError(
                f"Intersection diagnostic for {candidate_id!r} differs from target coverage"
            )
    elif (
        coverage is not None
        or diagnostic is not None
        or not (candidate_target == reference_target and candidate_count == reference_count)
    ):
        raise BenchmarkConfigError(
            f"Non-eligibility candidate {candidate_id!r} changes or diagnoses its target universe"
        )


def _normalize_evidence(
    plan: dict[str, Any],
    evidence: dict[str, Any],
    *,
    failed_candidate_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    candidates = {candidate["id"]: candidate for candidate in plan["candidates"]}
    failed = failed_candidate_ids or set()
    if evidence["budget"] != plan["budget"]["id"]:
        raise BenchmarkConfigError("Evidence and plan use different non-inferiority budgets")
    if (
        evidence["selection_validation_manifest_sha256"]
        != plan["reference"]["selection_validation_manifest_sha256"]
    ):
        raise BenchmarkConfigError(
            "Evidence does not use the validation manifest frozen with the reference"
        )
    if evidence["dataset_role"] != plan["analysis"]["validation_dataset_role"]:
        raise BenchmarkConfigError(
            "Non-inferiority evidence must use the independent efficiency-validation partition"
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str, int]] = set()
    contexts_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in evidence["rows"]:
        contexts_by_dataset[row["dataset_id"]].add(row["context_id"])
    for row in evidence["rows"]:
        candidate_id = row["candidate_id"]
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise BenchmarkConfigError(f"Evidence contains unknown candidate {candidate_id!r}")
        key = (
            candidate_id,
            row["dataset_id"],
            row["context_id"],
            row["algorithm_seed"],
        )
        if key in seen:
            raise BenchmarkConfigError(f"Evidence contains duplicate paired row {key!r}")
        seen.add(key)
        if row["reference_policy_sha256"] != plan["reference"]["policy_sha256"]:
            raise BenchmarkConfigError(f"Evidence for {candidate_id!r} uses the wrong reference")
        if row["candidate_policy_sha256"] != candidate["policy_sha256"]:
            raise BenchmarkConfigError(f"Evidence for {candidate_id!r} uses stale overrides")
        group_count = len(contexts_by_dataset[row["dataset_id"]])
        reference_parameters = resolve_group_policy(plan["reference"]["group_policy"], group_count)
        candidate_parameters = resolve_group_policy(candidate["group_policy"], group_count)
        if row["reference_configuration_sha256"] != canonical_json_sha256(
            reference_parameters
        ) or row["candidate_configuration_sha256"] != canonical_json_sha256(candidate_parameters):
            raise BenchmarkConfigError(
                f"Evidence for {candidate_id!r} uses the wrong group-count policy branch"
            )
        if row["metric"] != plan["analysis"]["metric"] or row["level"] != plan["analysis"]["level"]:
            raise BenchmarkConfigError(f"Evidence for {candidate_id!r} uses the wrong endpoint")
        numeric_values = [float(row["reference_value"]), float(row["candidate_value"])]
        for side in ("reference_measurements", "candidate_measurements"):
            measurement = row[side]
            numeric_values.extend(
                float(measurement[field])
                for field in (
                    "wall_time_seconds",
                    "peak_memory_bytes",
                    "output_bytes",
                    "targets_requested",
                    "targets_processed",
                    "models_requested",
                    "models_completed",
                    "models_trained",
                    "trees_fitted",
                )
            )
            if (
                measurement["models_completed"] > measurement["models_requested"]
                or measurement["models_trained"] > measurement["models_completed"]
                or measurement["targets_processed"] > measurement["targets_requested"]
            ):
                raise BenchmarkConfigError(
                    f"Evidence for {candidate_id!r} contains inconsistent run accounting"
                )
        if any(not math.isfinite(value) for value in numeric_values):
            raise BenchmarkConfigError(f"Evidence for {candidate_id!r} contains non-finite values")
        _validate_candidate_space(
            row,
            policy=candidate["primary_evaluation_policy"],
            intersection_diagnostic=candidate["intersection_diagnostic"],
            candidate_id=candidate_id,
        )
        grouped[candidate_id].append(row)
    # Successful rows belonging to a candidate with even one bound execution
    # failure are still validated above, but are not used for partial inference:
    # the predeclared panel is incomplete and no favourable bound may be made.
    for candidate_id in failed:
        grouped.pop(candidate_id, None)
    missing = sorted(set(candidates) - failed - grouped.keys())
    if missing:
        raise BenchmarkConfigError(f"Evidence omits confirmatory candidates: {missing}")
    return grouped


def _mean_paired_ratio(
    pairs: Mapping[int, Sequence[Mapping[str, Any]]],
    field: str,
    *,
    candidate_id: str,
    dataset_id: str,
) -> float:
    ratios: list[float] = []
    for reference, candidate in pairs.values():
        denominator = float(reference[field])
        if denominator <= 0:
            raise BenchmarkConfigError(
                f"{candidate_id}/{dataset_id} reference {field} is not positive"
            )
        ratios.append(float(candidate[field]) / denominator)
    return fmean(ratios)


def _mean_paired_log_ratio(
    pairs: Mapping[int, Sequence[Mapping[str, Any]]],
    field: str,
    *,
    candidate_id: str,
    dataset_id: str,
) -> float:
    log_ratios: list[float] = []
    for reference, candidate in pairs.values():
        denominator = float(reference[field])
        numerator = float(candidate[field])
        if denominator <= 0 or numerator <= 0:
            raise BenchmarkConfigError(
                f"{candidate_id}/{dataset_id} paired {field} values must be positive"
            )
        log_ratios.append(math.log(numerator / denominator))
    return fmean(log_ratios)


def _mean_completion_fraction(
    pairs: Mapping[int, Sequence[Mapping[str, Any]]],
    side: int,
    completed: str,
    requested: str,
) -> float:
    return fmean(
        float(pair[side][completed]) / float(pair[side][requested]) for pair in pairs.values()
    )


def _dataset_observations(
    rows: Sequence[dict[str, Any]],
    *,
    expected_seeds: set[int],
    candidate_id: str,
) -> list[dict[str, Any]]:
    by_context: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_context[(row["dataset_id"], row["context_id"])].append(row)
    contexts_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (dataset_id, context_id), members in sorted(by_context.items()):
        members.sort(key=lambda row: int(row["algorithm_seed"]))
        seeds = {int(row["algorithm_seed"]) for row in members}
        if seeds != expected_seeds:
            raise BenchmarkConfigError(
                f"{candidate_id}/{dataset_id}/{context_id} algorithm seeds differ from the "
                f"plan: expected={sorted(expected_seeds)}, observed={sorted(seeds)}"
            )
        label = f"{candidate_id}/{dataset_id}/{context_id}"
        space = _same_across_rows(members, "candidate_space", label)
        coverage = _same_across_rows(members, "target_coverage", label)
        raw_intersections = [row["intersection_diagnostic"] for row in members]
        if any(value is None for value in raw_intersections) and not all(
            value is None for value in raw_intersections
        ):
            raise BenchmarkConfigError(f"{label} inconsistently reports intersection diagnostics")
        intersection = raw_intersections[0]
        if intersection is not None:
            contracts = {
                json.dumps(
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"reference_value", "candidate_value"}
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for item in raw_intersections
            }
            if len(contracts) != 1:
                raise BenchmarkConfigError(
                    f"{label} intersection universe differs across algorithmic seeds"
                )
        contexts_by_dataset[dataset_id].append(
            {
                "context_id": context_id,
                "dataset_manifest_sha256": _same_across_rows(
                    members, "dataset_manifest_sha256", label
                ),
                "simulator_id": _same_across_rows(members, "simulator_id", label),
                "condition_id": _same_across_rows(members, "condition_id", label),
                "generation_seed": _same_across_rows(members, "generation_seed", label),
                "difference": fmean(
                    float(row["candidate_value"]) - float(row["reference_value"]) for row in members
                ),
                "intersection_difference": (
                    fmean(
                        float(row["intersection_diagnostic"]["candidate_value"])
                        - float(row["intersection_diagnostic"]["reference_value"])
                        for row in members
                    )
                    if intersection is not None
                    else None
                ),
                "measurements_by_seed": {
                    int(row["algorithm_seed"]): (
                        row["reference_measurements"],
                        row["candidate_measurements"],
                    )
                    for row in members
                },
                "target_fraction": (
                    float(space["n_candidate_targets"]) / float(space["n_reference_targets"])
                    if coverage is not None
                    else None
                ),
                "truth_edge_fraction": (
                    float(coverage["n_retained_truth_edges"])
                    / float(coverage["n_reference_truth_edges"])
                    if coverage is not None
                    else None
                ),
                "candidate_space": space,
                "target_coverage": coverage,
                "intersection_diagnostic": intersection,
            }
        )

    observations: list[dict[str, Any]] = []
    for dataset_id, contexts in sorted(contexts_by_dataset.items()):
        contexts.sort(key=lambda item: item["context_id"])
        metadata = {
            (
                item["dataset_manifest_sha256"],
                item["simulator_id"],
                item["condition_id"],
                item["generation_seed"],
            )
            for item in contexts
        }
        if len(metadata) != 1:
            raise BenchmarkConfigError(
                f"{candidate_id}/{dataset_id} metadata differs across group contexts"
            )
        dataset_manifest, simulator, condition, generation_seed = next(iter(metadata))
        measurements_by_seed: dict[int, set[str]] = defaultdict(set)
        for context in contexts:
            for seed, pair in context["measurements_by_seed"].items():
                measurements_by_seed[seed].add(
                    json.dumps(pair, sort_keys=True, separators=(",", ":"))
                )
        if any(len(measurements) != 1 for measurements in measurements_by_seed.values()):
            raise BenchmarkConfigError(
                f"{candidate_id}/{dataset_id} whole-run measurements differ across contexts"
            )
        pairs = {
            seed: json.loads(next(iter(measurements_by_seed[seed])))
            for seed in sorted(expected_seeds)
        }

        has_coverage = all(context["target_coverage"] is not None for context in contexts)
        has_intersection = all(
            context["intersection_diagnostic"] is not None for context in contexts
        )
        runtime_log_ratio = _mean_paired_log_ratio(
            pairs,
            "wall_time_seconds",
            candidate_id=candidate_id,
            dataset_id=dataset_id,
        )
        observations.append(
            {
                "dataset_id": dataset_id,
                "dataset_manifest_sha256": dataset_manifest,
                "simulator_id": simulator,
                "condition_id": condition,
                "generation_seed": generation_seed,
                "difference": fmean(float(context["difference"]) for context in contexts),
                "intersection_difference": (
                    fmean(float(context["intersection_difference"]) for context in contexts)
                    if has_intersection
                    else None
                ),
                "runtime_log_ratio": runtime_log_ratio,
                "runtime_ratio": math.exp(runtime_log_ratio),
                "peak_memory_ratio": _mean_paired_ratio(
                    pairs,
                    "peak_memory_bytes",
                    candidate_id=candidate_id,
                    dataset_id=dataset_id,
                ),
                "output_bytes_ratio": _mean_paired_ratio(
                    pairs,
                    "output_bytes",
                    candidate_id=candidate_id,
                    dataset_id=dataset_id,
                ),
                "targets_processed_ratio": _mean_paired_ratio(
                    pairs,
                    "targets_processed",
                    candidate_id=candidate_id,
                    dataset_id=dataset_id,
                ),
                "models_requested_ratio": _mean_paired_ratio(
                    pairs,
                    "models_requested",
                    candidate_id=candidate_id,
                    dataset_id=dataset_id,
                ),
                "trees_fitted_ratio": _mean_paired_ratio(
                    pairs,
                    "trees_fitted",
                    candidate_id=candidate_id,
                    dataset_id=dataset_id,
                ),
                "reference_run_fraction": fmean(
                    float(pair[0]["run_completed"]) for pair in pairs.values()
                ),
                "candidate_run_fraction": fmean(
                    float(pair[1]["run_completed"]) for pair in pairs.values()
                ),
                "reference_model_fraction": _mean_completion_fraction(
                    pairs, 0, "models_completed", "models_requested"
                ),
                "candidate_model_fraction": _mean_completion_fraction(
                    pairs, 1, "models_completed", "models_requested"
                ),
                "target_fraction": (
                    fmean(float(context["target_fraction"]) for context in contexts)
                    if has_coverage
                    else None
                ),
                "truth_edge_fraction": (
                    fmean(float(context["truth_edge_fraction"]) for context in contexts)
                    if has_coverage
                    else None
                ),
                "contexts": {
                    context["context_id"]: {
                        "candidate_space": context["candidate_space"],
                        "target_coverage": context["target_coverage"],
                        "intersection_diagnostic": context["intersection_diagnostic"],
                    }
                    for context in contexts
                },
            }
        )
    return observations


def _lower_bound(
    observations: Sequence[dict[str, Any]],
    field: str,
    *,
    alpha: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    point = _hierarchical_mean(observations, field)
    distribution = _hierarchical_bootstrap(
        observations,
        field,
        resamples=resamples,
        random_seed=seed,
    )
    return point, _percentile(distribution, alpha)


def _upper_bound(
    observations: Sequence[dict[str, Any]],
    field: str,
    *,
    alpha: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    point = _hierarchical_mean(observations, field)
    distribution = _hierarchical_bootstrap(
        observations,
        field,
        resamples=resamples,
        random_seed=seed,
    )
    return point, _percentile(distribution, 1.0 - alpha)


def _geometric_ratio_upper_bound(
    observations: Sequence[dict[str, Any]],
    *,
    alpha: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    log_point, log_upper = _upper_bound(
        observations,
        "runtime_log_ratio",
        alpha=alpha,
        resamples=resamples,
        seed=seed,
    )
    return math.exp(log_point), math.exp(log_upper)


def _coverage_endpoint_result(
    observations: Sequence[dict[str, Any]],
    *,
    field: str,
    gates: Mapping[str, Any],
    inferential: bool,
    alpha: float | None,
    resamples: int,
    seed: int | None,
) -> dict[str, Any]:
    estimate = _hierarchical_mean(observations, field)
    lower: float | None = None
    if inferential:
        if alpha is None or seed is None:
            raise BenchmarkConfigError("Confirmatory coverage interval requires alpha and seed")
        estimate, lower = _lower_bound(
            observations,
            field,
            alpha=alpha,
            resamples=resamples,
            seed=seed,
        )
    condition_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for observation in observations:
        condition_values[
            (str(observation["simulator_id"]), str(observation["condition_id"]))
        ].append(float(observation[field]))
    observed_dataset_minimum = min(float(item[field]) for item in observations)
    observed_condition_minimum = min(fmean(values) for values in condition_values.values())
    suffix = "target_fraction" if field == "target_fraction" else "truth_edge_fraction"
    mean_minimum = float(gates[f"minimum_{suffix}"])
    dataset_minimum = float(gates[f"minimum_each_dataset_{suffix}"])
    condition_minimum = float(gates[f"minimum_each_condition_mean_{suffix}"])
    dataset_guardrail_met = observed_dataset_minimum >= dataset_minimum
    condition_guardrail_met = observed_condition_minimum >= condition_minimum
    return {
        "estimate": estimate,
        "one_sided_lower_confidence_bound": lower,
        "minimum": mean_minimum,
        "minimum_each_dataset": dataset_minimum,
        "observed_minimum_dataset": observed_dataset_minimum,
        "observed_dataset_guardrail_met": dataset_guardrail_met,
        "minimum_each_condition_mean": condition_minimum,
        "observed_minimum_condition_mean": observed_condition_minimum,
        "observed_condition_guardrail_met": condition_guardrail_met,
        "inference_status": ("confirmatory_interval" if inferential else "descriptive_only"),
        "passed": (
            bool(
                lower is not None
                and lower >= mean_minimum
                and dataset_guardrail_met
                and condition_guardrail_met
            )
            if inferential
            else None
        ),
    }


def _manifest_matches(
    output_dir: Path,
    plan_digest: str,
    evidence_digest: str,
    implementation_digest: str,
) -> bool:
    canonical_members = ("results.json", "comparisons.csv")
    canonical_disk_files = {*canonical_members, "manifest.json"}
    manifest_path = output_dir / "manifest.json"
    if output_dir.is_symlink() or not output_dir.is_dir():
        return False
    try:
        disk_members = list(output_dir.iterdir())
    except OSError:
        return False
    if (
        len(disk_members) != len(canonical_disk_files)
        or {member.name for member in disk_members} != canonical_disk_files
        or any(member.is_symlink() or not member.is_file() for member in disk_members)
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        return False
    try:
        manifest = load_json_object(manifest_path, "non-inferiority output manifest")
    except BenchmarkConfigError:
        return False
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("artifact_type") != "spathi_efficiency_noninferiority_output_manifest"
    ):
        return False
    if manifest.get("inputs") != {"plan_sha256": plan_digest, "evidence_sha256": evidence_digest}:
        return False
    if manifest.get("analysis_implementation_sha256") != implementation_digest:
        return False
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(canonical_members):
        return False
    observed_paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            return False
        raw_path = item.get("path")
        expected_size = item.get("size")
        if (
            not isinstance(raw_path, str)
            or raw_path not in canonical_members
            or raw_path in observed_paths
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            return False
        observed_paths.add(raw_path)
        member = output_dir / raw_path
        try:
            if (
                member.is_symlink()
                or not member.is_file()
                or member.stat().st_size != expected_size
                or sha256_file(member) != item.get("sha256")
            ):
                return False
        except (OSError, ValueError):
            return False
    return observed_paths == set(canonical_members)


def _paired_ratio_result(
    observations: Sequence[dict[str, Any]],
    field: str,
    *,
    unit: str,
    alpha: float | None,
    resamples: int,
    seed: int | None,
    confidence_level: float | None,
) -> dict[str, Any]:
    estimate = _hierarchical_mean(observations, field)
    upper: float | None = None
    if alpha is not None:
        if seed is None:
            raise BenchmarkConfigError("Confirmatory ratio interval requires a statistical seed")
        estimate, upper = _upper_bound(
            observations,
            field,
            alpha=alpha,
            resamples=resamples,
            seed=seed,
        )
    return {
        "unit": unit,
        "hierarchical_mean_ratio": estimate,
        "median_dataset_ratio": float(median(item[field] for item in observations)),
        "one_sided_upper_confidence_bound": upper,
        "confidence_level": confidence_level,
        "inference_status": ("confirmatory_interval" if alpha is not None else "descriptive_only"),
    }


def decide_efficiency_adoption(
    plan: Mapping[str, Any], comparisons: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Select the exact reference or one approved policy using the frozen rule."""

    reference = plan["reference"]
    selection_rule = plan["analysis"]["adoption_rule"]
    if selection_rule != "lowest_geometric_mean_runtime_ratio_then_candidate_id_among_adoptable":
        raise BenchmarkConfigError(f"Unsupported efficiency adoption rule: {selection_rule!r}")
    if plan["budget"]["analysis_role"] != "confirmatory":
        return {
            "decision": "descriptive_only",
            "selection_rule": selection_rule,
            "selected_policy_id": reference["policy_id"],
            "selected_policy_sha256": reference["policy_sha256"],
            "selected_group_policy": reference["group_policy"],
            "selected_candidate_id": None,
            "reason": (
                "Reduced-budget results are descriptive only: they contain no confidence "
                "intervals, gate decisions or authority to change the production policy."
            ),
        }
    adoptable = [item for item in comparisons if item["decision"] == "adoptable"]
    if not adoptable:
        return {
            "decision": "reference_retained",
            "selection_rule": selection_rule,
            "selected_policy_id": reference["policy_id"],
            "selected_policy_sha256": reference["policy_sha256"],
            "selected_group_policy": reference["group_policy"],
            "selected_candidate_id": None,
            "reason": "No frozen efficiency policy passed every confirmatory adoption gate.",
        }
    winner = min(
        adoptable,
        key=lambda item: (
            float(item["cost"]["runtime"]["hierarchical_mean_ratio"]),
            str(item["candidate_id"]),
        ),
    )
    candidates = {item["id"]: item for item in plan["candidates"]}
    selected = candidates[winner["candidate_id"]]
    return {
        "decision": "candidate_adopted",
        "selection_rule": selection_rule,
        "selected_policy_id": selected["id"],
        "selected_policy_sha256": selected["policy_sha256"],
        "selected_group_policy": selected["group_policy"],
        "selected_candidate_id": selected["id"],
        "reason": "The selected frozen policy passed every gate and had the lowest observed hierarchical geometric-mean runtime ratio among adoptable candidates; this rule does not claim a significant head-to-head speed difference between candidates.",
    }


def resolve_efficiency_adoption(
    plan: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    execution_failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reproduce the frozen decision, including its conservative failure rule."""

    if not execution_failures:
        return decide_efficiency_adoption(plan, comparisons)
    candidate_ids = {candidate["id"] for candidate in plan["candidates"]}
    for failure in execution_failures:
        if (
            failure.get("selection_role") != "candidate"
            or failure.get("configuration_id") not in candidate_ids
        ):
            raise BenchmarkConfigError(
                "Only a bound failure of a predeclared candidate can enter the "
                "conservative efficiency decision"
            )
    reference = plan["reference"]
    inferential = plan["budget"]["analysis_role"] == "confirmatory"
    return {
        "decision": "reference_retained" if inferential else "descriptive_only",
        "selection_rule": plan["analysis"]["adoption_rule"],
        "selected_policy_id": reference["policy_id"],
        "selected_policy_sha256": reference["policy_sha256"],
        "selected_group_policy": reference["group_policy"],
        "selected_candidate_id": None,
        "reason": (
            "At least one predeclared candidate had a bound tool-execution failure. "
            "No result or statistical bound was computed for any incomplete candidate, "
            "and the exact reference policy was retained fail-safe."
            if inferential
            else "At least one predeclared candidate had a bound tool-execution failure; "
            "the reduced study remains descriptive and cannot change the exact policy."
        ),
    }


def analyze_noninferiority(
    *,
    repo_root: Path,
    plan_path: Path,
    evidence_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Validate paired evidence and publish the predeclared one-sided analysis."""

    root = repo_root.resolve()
    _relative_to_repository(root, plan_path, "Non-inferiority plan")
    _relative_to_repository(root, evidence_path, "Non-inferiority evidence")
    _relative_to_repository(root, output_dir, "Non-inferiority output")
    plan = _load_validated(
        plan_path,
        schema_path=root / "schemas/noninferiority-plan.schema.json",
        label="non-inferiority plan",
    )
    evidence = verify_noninferiority_evidence(
        repo_root=root,
        plan_path=plan_path,
        evidence_path=evidence_path,
    )
    plan_digest = sha256_file(plan_path)
    evidence_digest = sha256_file(evidence_path)
    implementation_digest = _analysis_implementation_sha256(root)
    if evidence["plan_sha256"] != plan_digest:
        raise BenchmarkConfigError("Paired evidence targets a different plan digest")
    study = _validate_plan_binding(root, plan)
    if evidence["study_id"] != study["id"]:
        raise BenchmarkConfigError("Paired evidence targets a different study ID")
    if output_dir.exists() or output_dir.is_symlink():
        if _manifest_matches(
            output_dir,
            plan_digest,
            evidence_digest,
            implementation_digest,
        ):
            return tuple(
                (output_dir / name).resolve()
                for name in ("results.json", "comparisons.csv", "manifest.json")
            )
        raise BenchmarkConfigError(
            f"Non-inferiority output already exists with incompatible or damaged evidence: {output_dir}"
        )

    expected_seeds = set(plan["budget"]["algorithm_seeds"])
    expected_datasets = int(plan["budget"]["expected_independent_datasets"])
    candidates = {candidate["id"]: candidate for candidate in plan["candidates"]}
    execution_failures = list(evidence["execution_failures"])
    failure_keys: set[tuple[str, str, int]] = set()
    failed_candidate_ids: set[str] = set()
    for failure in execution_failures:
        candidate_id = failure["configuration_id"]
        if failure["selection_role"] != "candidate":
            raise BenchmarkConfigError(
                "An exact-reference execution failure makes the efficiency study inconclusive"
            )
        if candidate_id not in candidates:
            raise BenchmarkConfigError(
                f"Execution failure names unknown efficiency candidate {candidate_id!r}"
            )
        key = (candidate_id, failure["dataset_id"], failure["algorithm_seed"])
        if key in failure_keys:
            raise BenchmarkConfigError(f"Duplicate bound execution failure: {key!r}")
        failure_keys.add(key)
        failed_candidate_ids.add(candidate_id)
    grouped = _normalize_evidence(
        plan,
        evidence,
        failed_candidate_ids=failed_candidate_ids,
    )
    panels: dict[str, list[dict[str, Any]]] = {}
    for candidate_id, rows in grouped.items():
        observations = _dataset_observations(
            rows,
            expected_seeds=expected_seeds,
            candidate_id=candidate_id,
        )
        if len(observations) != expected_datasets:
            raise BenchmarkConfigError(
                f"{candidate_id!r} has {len(observations)} independent datasets; "
                f"the plan requires exactly {expected_datasets}"
            )
        panels[candidate_id] = observations

    panel_identities = {
        candidate_id: {
            (
                item["dataset_id"],
                item["simulator_id"],
                item["condition_id"],
                item["generation_seed"],
            )
            for item in observations
        }
        for candidate_id, observations in panels.items()
    }
    if panel_identities and len({frozenset(panel) for panel in panel_identities.values()}) != 1:
        raise BenchmarkConfigError("Confirmatory candidates do not use one complete paired panel")
    context_panels = {
        candidate_id: {
            (item["dataset_id"], tuple(sorted(item["contexts"]))) for item in observations
        }
        for candidate_id, observations in panels.items()
    }
    if context_panels and len({frozenset(panel) for panel in context_panels.values()}) != 1:
        raise BenchmarkConfigError(
            "Confirmatory candidates do not use one complete group-context panel"
        )
    identities = next(iter(panel_identities.values()), set())
    independent_units: dict[tuple[str, int], str] = {}
    for dataset_id, simulator_id, _condition_id, generation_seed in identities:
        unit = (simulator_id, generation_seed)
        previous = independent_units.get(unit)
        if previous is not None and previous != dataset_id:
            raise BenchmarkConfigError(
                f"Independent simulation unit {unit!r} is duplicated by {previous!r} and {dataset_id!r}"
            )
        independent_units[unit] = dataset_id

    # Reference inputs must not drift between candidate comparisons. Eligibility
    # and combined candidates must additionally use the exact same selected set.
    by_dataset: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for candidate_id, observations in panels.items():
        for observation in observations:
            by_dataset[observation["dataset_id"]].append((candidate_id, observation))
    for dataset_id, members in by_dataset.items():
        dataset_manifests = {item["dataset_manifest_sha256"] for _candidate_id, item in members}
        if len(dataset_manifests) != 1:
            raise BenchmarkConfigError(
                f"Dataset manifest drifts between candidates for {dataset_id!r}"
            )
        context_ids = sorted(members[0][1]["contexts"])
        for context_id in context_ids:
            reference_spaces = {
                (
                    item["contexts"][context_id]["candidate_space"][
                        "reference_source_universe_sha256"
                    ],
                    item["contexts"][context_id]["candidate_space"][
                        "reference_target_universe_sha256"
                    ],
                    item["contexts"][context_id]["candidate_space"]["n_reference_targets"],
                )
                for _candidate_id, item in members
            }
            if len(reference_spaces) != 1:
                raise BenchmarkConfigError(
                    f"Reference candidate space drifts for {dataset_id!r}/{context_id!r}"
                )
            eligible_spaces = {
                (
                    (
                        item["contexts"][context_id]["intersection_diagnostic"][
                            "candidate_space_sha256"
                        ]
                        if item["contexts"][context_id]["intersection_diagnostic"] is not None
                        else None
                    ),
                    item["contexts"][context_id]["candidate_space"]["n_candidate_targets"],
                    item["contexts"][context_id]["candidate_space"][
                        "candidate_target_universe_sha256"
                    ],
                    item["contexts"][context_id]["target_coverage"]["eligibility_manifest_sha256"],
                    item["contexts"][context_id]["target_coverage"]["n_reference_truth_edges"],
                    item["contexts"][context_id]["target_coverage"]["n_retained_truth_edges"],
                )
                for candidate_id, item in members
                if candidates[candidate_id]["intersection_diagnostic"]
            }
            if len(eligible_spaces) != 1:
                raise BenchmarkConfigError(
                    "Eligibility and combined candidates use different targets for "
                    f"{dataset_id!r}/{context_id!r}"
                )

    analysis = plan["analysis"]
    candidate_count = len(candidates)
    analysis_role = plan["budget"]["analysis_role"]
    inferential = analysis_role == "confirmatory"
    adjusted_alpha = (
        float(analysis["familywise_alpha"]) / candidate_count
        if inferential and candidate_count
        else None
    )
    confidence_level = 1.0 - adjusted_alpha if adjusted_alpha is not None else None
    resamples = int(analysis["bootstrap_resamples"])
    seed_plan = load_seed_plan(root / "experiments/v1/seeds/seed-plan.json")
    seed_namespace = f"{plan['budget']['id']}.efficiency_validation"
    statistical_root = (
        validate_plan_seed_record(
            seed_plan,
            seed=plan["budget"]["statistical_seed"],
            seed_derivation={
                "namespace": seed_namespace,
                "stream": "statistical",
                "coordinates": {"purpose": "noninferiority_analysis"},
            },
            required_namespace=seed_namespace,
            required_stream="statistical",
            required_coordinates={"purpose": "noninferiority_analysis"},
        )
        if inferential
        else None
    )
    effective_statistical_seeds: list[dict[str, Any]] = []

    def endpoint_seed(candidate_id: str, endpoint: str) -> int:
        record = plan_seed_record(
            seed_plan,
            namespace=seed_namespace,
            stream="statistical",
            coordinates={
                "purpose": "hierarchical_bootstrap",
                "candidate_id": candidate_id,
                "endpoint": endpoint,
            },
        )
        effective_statistical_seeds.append(
            {"candidate_id": candidate_id, "endpoint": endpoint, **record}
        )
        return int(record["seed"])

    margin = float(analysis["absolute_noninferiority_margin"])
    results: list[dict[str, Any]] = []
    for candidate_id in sorted(panels):
        candidate = candidates[candidate_id]
        observations = panels[candidate_id]
        estimate = _hierarchical_mean(observations, "difference")
        lower: float | None = None
        if adjusted_alpha is not None:
            estimate, lower = _lower_bound(
                observations,
                "difference",
                alpha=adjusted_alpha,
                resamples=resamples,
                seed=endpoint_seed(candidate_id, "accuracy"),
            )
        accuracy = {
            "metric": analysis["metric"],
            "level": analysis["level"],
            "target_universe": analysis["primary_target_universe"],
            "difference": "candidate_minus_reference",
            "estimate": estimate,
            "absolute_noninferiority_margin": margin,
            "one_sided_lower_confidence_bound": lower,
            "confidence_level": confidence_level,
            "inference_status": ("confirmatory_interval" if inferential else "descriptive_only"),
            "passed": lower >= -margin if lower is not None else None,
        }
        coverage_results: dict[str, Any] | None = None
        coverage_passed: bool | None = True if inferential else None
        gates = candidate.get("coverage_gates")
        if gates is not None:
            target_coverage = _coverage_endpoint_result(
                observations,
                field="target_fraction",
                gates=gates,
                inferential=inferential,
                alpha=adjusted_alpha,
                resamples=resamples,
                seed=(endpoint_seed(candidate_id, "target_coverage") if inferential else None),
            )
            truth_coverage = _coverage_endpoint_result(
                observations,
                field="truth_edge_fraction",
                gates=gates,
                inferential=inferential,
                alpha=adjusted_alpha,
                resamples=resamples,
                seed=(endpoint_seed(candidate_id, "truth_edge_coverage") if inferential else None),
            )
            coverage_passed = (
                bool(target_coverage["passed"] and truth_coverage["passed"])
                if inferential
                else None
            )
            coverage_results = {
                "target_fraction": target_coverage,
                "truth_edge_fraction": truth_coverage,
                "confidence_level": confidence_level,
                "threshold_basis": gates["threshold_basis"],
            }
        runtime_ratio = math.exp(_hierarchical_mean(observations, "runtime_log_ratio"))
        runtime_upper: float | None = None
        if adjusted_alpha is not None:
            runtime_ratio, runtime_upper = _geometric_ratio_upper_bound(
                observations,
                alpha=adjusted_alpha,
                resamples=resamples,
                seed=endpoint_seed(candidate_id, "runtime"),
            )
        runtime_maximum = float(
            analysis["runtime_gate"]["maximum_one_sided_upper_confidence_bound"]
        )
        runtime_passed = runtime_upper < runtime_maximum if runtime_upper is not None else None
        completion_gate = analysis["completion_gate"]
        reference_run_fraction = _hierarchical_mean(observations, "reference_run_fraction")
        candidate_run_fraction = _hierarchical_mean(observations, "candidate_run_fraction")
        reference_model_fraction = _hierarchical_mean(observations, "reference_model_fraction")
        candidate_model_fraction = _hierarchical_mean(observations, "candidate_model_fraction")
        minimum_run_fraction = float(completion_gate["minimum_run_completion_fraction"])
        minimum_model_fraction = float(completion_gate["minimum_model_completion_fraction"])
        completion_passed = (
            reference_run_fraction >= minimum_run_fraction
            and candidate_run_fraction >= minimum_run_fraction
            and reference_model_fraction >= minimum_model_fraction
            and candidate_model_fraction >= minimum_model_fraction
        )
        completion = {
            "reference_run_fraction": reference_run_fraction,
            "candidate_run_fraction": candidate_run_fraction,
            "reference_model_fraction": reference_model_fraction,
            "candidate_model_fraction": candidate_model_fraction,
            "minimum_run_fraction": minimum_run_fraction,
            "minimum_model_fraction": minimum_model_fraction,
            "observed_complete": completion_passed,
        }
        intersection_evaluable = all(
            observation["intersection_difference"] is not None for observation in observations
        )
        intersection_result = (
            {
                "difference": "candidate_minus_reference",
                "estimate": _hierarchical_mean(observations, "intersection_difference"),
                "n_dataset_group_contexts": sum(len(item["contexts"]) for item in observations),
                "interpretation": "diagnostic_only_not_an_adoption_endpoint",
            }
            if candidate["intersection_diagnostic"] and intersection_evaluable
            else None
        )
        passed = bool(
            inferential
            and accuracy["passed"]
            and coverage_passed
            and completion_passed
            and runtime_passed
        )
        result = {
            "candidate_id": candidate_id,
            "component": candidate["component"],
            "policy_sha256": candidate["policy_sha256"],
            "primary_evaluation_policy": candidate["primary_evaluation_policy"],
            "n_independent_datasets": len(observations),
            "n_dataset_group_contexts": sum(len(item["contexts"]) for item in observations),
            "n_algorithm_seeds_per_dataset": len(expected_seeds),
            "accuracy": accuracy,
            "intersection_diagnostic": intersection_result,
            "coverage": coverage_results,
            "completion": completion,
            "cost": {
                "runtime": {
                    "hierarchical_mean_ratio": runtime_ratio,
                    "hierarchical_mean_speedup": 1.0 / runtime_ratio,
                    "median_dataset_ratio": float(
                        median(item["runtime_ratio"] for item in observations)
                    ),
                    "one_sided_upper_confidence_bound": runtime_upper,
                    "maximum": runtime_maximum,
                    "confidence_level": confidence_level,
                    "inference_status": (
                        "confirmatory_interval" if inferential else "descriptive_only"
                    ),
                    "passed": runtime_passed,
                },
                "peak_memory": _paired_ratio_result(
                    observations,
                    "peak_memory_ratio",
                    unit="bytes",
                    alpha=adjusted_alpha,
                    resamples=resamples,
                    seed=(endpoint_seed(candidate_id, "peak_memory") if inferential else None),
                    confidence_level=confidence_level,
                ),
                "output_size": _paired_ratio_result(
                    observations,
                    "output_bytes_ratio",
                    unit="bytes",
                    alpha=adjusted_alpha,
                    resamples=resamples,
                    seed=(endpoint_seed(candidate_id, "output_bytes") if inferential else None),
                    confidence_level=confidence_level,
                ),
                "targets_processed": _paired_ratio_result(
                    observations,
                    "targets_processed_ratio",
                    unit="count",
                    alpha=adjusted_alpha,
                    resamples=resamples,
                    seed=(
                        endpoint_seed(candidate_id, "targets_processed") if inferential else None
                    ),
                    confidence_level=confidence_level,
                ),
                "models_requested": _paired_ratio_result(
                    observations,
                    "models_requested_ratio",
                    unit="count",
                    alpha=adjusted_alpha,
                    resamples=resamples,
                    seed=(endpoint_seed(candidate_id, "models_requested") if inferential else None),
                    confidence_level=confidence_level,
                ),
                "trees_fitted": _paired_ratio_result(
                    observations,
                    "trees_fitted_ratio",
                    unit="count",
                    alpha=adjusted_alpha,
                    resamples=resamples,
                    seed=(endpoint_seed(candidate_id, "trees_fitted") if inferential else None),
                    confidence_level=confidence_level,
                ),
            },
            "decision": (
                "adoptable" if passed else "not_adoptable" if inferential else "descriptive_only"
            ),
        }
        results.append(result)

    adoption = resolve_efficiency_adoption(plan, results, execution_failures)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "artifact_type": "spathi_efficiency_noninferiority_results",
        "study_id": study["id"],
        "budget": plan["budget"]["id"],
        "analysis_role": analysis_role,
        "dataset_role": evidence["dataset_role"],
        "inputs": {"plan_sha256": plan_digest, "evidence_sha256": evidence_digest},
        "analysis_implementation_sha256": implementation_digest,
        "analysis": {
            **analysis,
            "confirmatory_candidate_count": candidate_count,
            "per_candidate_alpha": adjusted_alpha,
            "one_sided_confidence_level": confidence_level,
        },
        "execution_failures": execution_failures,
        "comparisons": results,
        "global_decision": (
            "policy_adopted"
            if adoption["decision"] == "candidate_adopted"
            else "descriptive_only"
            if adoption["decision"] == "descriptive_only"
            else "reference_retained"
        ),
        "adoption": adoption,
        "interpretation_limit": (
            "Primary accuracy uses every requested target and scores targets omitted by an "
            "eligibility policy as empty minimum-score predictions. Retained-target accuracy "
            "is diagnostic only. Full-budget conclusions are conditional on the predeclared "
            "synthetic panel and fixed algorithm-seed ensemble; the AUPR and coverage margins "
            "are author-defined engineering tolerances rather than biologically validated "
            "equivalence thresholds. Reduced-budget results are descriptive only and contain "
            "no confidence intervals or gate decisions. Any bound formal candidate failure "
            "invalidates relative selection globally and retains exact SPATHI; an exact-reference "
            "failure aborts because no paired comparison is interpretable."
        ),
    }
    if statistical_root is not None:
        payload["seed_contract"] = {
            "seed_plan": "experiments/v1/seeds/seed-plan.json",
            "seed_plan_sha256": canonical_json_sha256(seed_plan),
            "root": statistical_root,
            "effective": effective_statistical_seeds,
        }
    validate_json(
        payload,
        root / "schemas/noninferiority-results.schema.json",
        "non-inferiority results",
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=str(output_dir.parent.resolve()))
    )
    try:
        results_path = temporary / "results.json"
        results_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        comparisons_path = temporary / "comparisons.csv"
        fields = (
            "candidate_id",
            "component",
            "primary_evaluation_policy",
            "n_independent_datasets",
            "n_dataset_group_contexts",
            "accuracy_estimate",
            "accuracy_lower_bound",
            "noninferiority_margin",
            "intersection_accuracy_difference_diagnostic",
            "target_coverage_estimate",
            "target_coverage_lower_bound",
            "target_coverage_minimum_dataset",
            "target_coverage_minimum_condition_mean",
            "truth_edge_coverage_estimate",
            "truth_edge_coverage_lower_bound",
            "truth_edge_coverage_minimum_dataset",
            "truth_edge_coverage_minimum_condition_mean",
            "runtime_ratio",
            "runtime_upper_confidence_bound",
            "runtime_gate_maximum",
            "runtime_gate_passed",
            "speedup",
            "peak_memory_ratio",
            "output_bytes_ratio",
            "targets_processed_ratio",
            "models_requested_ratio",
            "trees_fitted_ratio",
            "completion_observed_complete",
            "decision",
        )
        with comparisons_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for result in results:
                coverage = result["coverage"] or {}
                target_coverage = coverage.get("target_fraction", {})
                truth_coverage = coverage.get("truth_edge_fraction", {})
                writer.writerow(
                    {
                        "candidate_id": result["candidate_id"],
                        "component": result["component"],
                        "primary_evaluation_policy": result["primary_evaluation_policy"],
                        "n_independent_datasets": result["n_independent_datasets"],
                        "n_dataset_group_contexts": result["n_dataset_group_contexts"],
                        "accuracy_estimate": result["accuracy"]["estimate"],
                        "accuracy_lower_bound": result["accuracy"][
                            "one_sided_lower_confidence_bound"
                        ],
                        "noninferiority_margin": margin,
                        "intersection_accuracy_difference_diagnostic": (
                            ""
                            if result["intersection_diagnostic"] is None
                            else result["intersection_diagnostic"]["estimate"]
                        ),
                        "target_coverage_estimate": target_coverage.get("estimate", ""),
                        "target_coverage_lower_bound": target_coverage.get(
                            "one_sided_lower_confidence_bound", ""
                        ),
                        "target_coverage_minimum_dataset": target_coverage.get(
                            "observed_minimum_dataset", ""
                        ),
                        "target_coverage_minimum_condition_mean": target_coverage.get(
                            "observed_minimum_condition_mean", ""
                        ),
                        "truth_edge_coverage_estimate": truth_coverage.get("estimate", ""),
                        "truth_edge_coverage_lower_bound": truth_coverage.get(
                            "one_sided_lower_confidence_bound", ""
                        ),
                        "truth_edge_coverage_minimum_dataset": truth_coverage.get(
                            "observed_minimum_dataset", ""
                        ),
                        "truth_edge_coverage_minimum_condition_mean": truth_coverage.get(
                            "observed_minimum_condition_mean", ""
                        ),
                        "runtime_ratio": result["cost"]["runtime"]["hierarchical_mean_ratio"],
                        "runtime_upper_confidence_bound": result["cost"]["runtime"][
                            "one_sided_upper_confidence_bound"
                        ],
                        "runtime_gate_maximum": result["cost"]["runtime"]["maximum"],
                        "runtime_gate_passed": result["cost"]["runtime"]["passed"],
                        "speedup": result["cost"]["runtime"]["hierarchical_mean_speedup"],
                        "peak_memory_ratio": result["cost"]["peak_memory"][
                            "hierarchical_mean_ratio"
                        ],
                        "output_bytes_ratio": result["cost"]["output_size"][
                            "hierarchical_mean_ratio"
                        ],
                        "targets_processed_ratio": result["cost"]["targets_processed"][
                            "hierarchical_mean_ratio"
                        ],
                        "models_requested_ratio": result["cost"]["models_requested"][
                            "hierarchical_mean_ratio"
                        ],
                        "trees_fitted_ratio": result["cost"]["trees_fitted"][
                            "hierarchical_mean_ratio"
                        ],
                        "completion_observed_complete": result["completion"]["observed_complete"],
                        "decision": result["decision"],
                    }
                )
        manifest = {
            "schema_version": "1.0",
            "artifact_type": "spathi_efficiency_noninferiority_output_manifest",
            "inputs": payload["inputs"],
            "analysis_implementation_sha256": implementation_digest,
            "files": [
                {
                    "path": "results.json",
                    "sha256": sha256_file(results_path),
                    "size": results_path.stat().st_size,
                },
                {
                    "path": "comparisons.csv",
                    "sha256": sha256_file(comparisons_path),
                    "size": comparisons_path.stat().st_size,
                },
            ],
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return tuple(
        (output_dir / name).resolve()
        for name in ("results.json", "comparisons.csv", "manifest.json")
    )


__all__ = [
    "STUDY_RELATIVE_PATH",
    "analyze_noninferiority",
    "decide_efficiency_adoption",
    "load_noninferiority_study",
    "materialize_noninferiority_plan",
]
