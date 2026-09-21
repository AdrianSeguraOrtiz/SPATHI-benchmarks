"""Validated scientific design for the three-phase article benchmark.

The design is data, not executable control flow.  This module turns the single
committed JSON document into a small immutable model that planners, executors,
and reports can share.  Random seeds are derived from semantic coordinates, so
neither task order nor the amount of available parallelism can change an
experiment.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_NAMES = ("smoke", "reduced", "full")
PHASE_NAMES = ("calibration", "comparison", "scalability")
THREAD_POINT_NAMES = ("one", "standard", "all")
SEED_DERIVATION = "sha256-semantic-coordinates-v1"
DATASET_CAPABILITIES = frozenset({"group_labels", "tf_list", "ordered_groups", "lineage_tree"})
ROUTE_DELIVERIES = frozenset({"group_native", "group_emulated", "group_aggregated"})
OUTPUT_SEMANTICS = frozenset({"directed_tf_to_target", "directed_all_gene", "undirected_topology"})
ARTICLE_TOOL_IDS = frozenset(
    {
        "aracne3",
        "cespgrn",
        "clr",
        "genie3",
        "grnboost2",
        "infercsn",
        "kscreni",
        "lioness",
        "metasem",
        "pidc",
        "ppcor",
        "scmtni",
        "simic",
        "spathi",
        "tigress",
    }
)


class DesignError(ValueError):
    """Raised when the prospective article design is internally inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DesignError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DesignError(f"{label} must be an array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DesignError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DesignError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignError(f"{label} must be a number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise DesignError(f"{label} must be >= {minimum}")
    return result


def _unique(values: Iterable[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != len(set(result)):
        raise DesignError(f"{label} must not contain duplicate identifiers")
    return result


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise DesignError(f"{label} keys differ (missing={missing}, extra={extra})")


@dataclass(frozen=True, slots=True)
class Dimensions:
    """Synthetic expression dimensions declared before data generation."""

    cells: int
    genes: int
    transcription_factors: int
    groups: int


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One prospective synthetic condition, independent of its replicate seed."""

    id: str
    label: str
    simulator: str
    scenario: str
    dimensions: Dimensions
    capabilities: frozenset[str]
    attributes: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One equally eligible, explicit SPATHI parameter vector."""

    id: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolRoute:
    """One reproducible route through the ANDREA inference contract."""

    id: str
    tool_id: str
    configuration_id: str
    delivery: str
    required_capabilities: frozenset[str]
    inputs: tuple[str, ...]
    output_semantics: str
    parameters: Mapping[str, Any]
    seed_parameter: str | None
    max_threads: int | None

    @property
    def method_id(self) -> str:
        """Stable method identifier used in plans and result tables."""

        return self.id

    @property
    def execution_mode(self) -> str:
        """ANDREA delivery mode for this route."""

        return self.delivery

    @property
    def default_parameter_overrides(self) -> Mapping[str, Any]:
        """Only the prospectively justified deviations from upstream defaults."""

        return self.parameters


@dataclass(frozen=True, slots=True)
class CompetitorFamily:
    """Methods compared under one formally stated information contract."""

    id: str
    label: str
    rationale: str
    statistical_role: str
    route_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalibrationBudget:
    condition_ids: tuple[str, ...]
    development_replicates: int
    validation_replicates: int
    screening_algorithm_seeds: int
    validation_algorithm_seeds: int


@dataclass(frozen=True, slots=True)
class ComparisonBudget:
    replicates: Mapping[str, int]
    task_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class ScalabilityBudget:
    scale_ids: tuple[str, ...]
    ladder_replicates: int
    scaling_scale_id: str
    scaling_replicates: int
    thread_points: tuple[str, ...]
    task_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class Profile:
    """Frozen coverage budget for all three scientific phases."""

    id: str
    calibration: CalibrationBudget
    comparison: ComparisonBudget
    scalability: ScalabilityBudget


@dataclass(frozen=True, slots=True)
class DatasetInstance:
    """A seeded realization of a prospective synthetic condition."""

    dataset_id: str
    split: str
    replicate: int
    seed: int


@dataclass(frozen=True, slots=True)
class PlannedDataset:
    """Execution-ready dataset realization exposed to the study planner."""

    id: str
    phase: str
    role: str
    condition_id: str
    label: str
    generator: str
    scenario: str
    scenario_request: Mapping[str, Any]
    simulator_parameters: Mapping[str, Any]
    generated_inputs: Mapping[str, Mapping[str, Any]]
    dimensions: Dimensions
    capabilities: frozenset[str]
    replicate: int
    seed: int
    order_kind: str


@dataclass(frozen=True, slots=True)
class ProfileView:
    """A profile-expanded view with no runtime- or machine-dependent choices."""

    id: str
    profile: Profile
    candidates: tuple[Candidate, ...]
    datasets: Mapping[str, Mapping[str, tuple[PlannedDataset, ...]]]
    routes: Mapping[str, ToolRoute]
    families: tuple[CompetitorFamily, ...]
    screening_trees: int
    finalists: int
    validation_fit_trees: int
    validation_prefixes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StudyDesign:
    """Complete immutable view of the committed article design."""

    id: str
    master_seed: int
    raw: Mapping[str, Any]
    profiles: Mapping[str, Profile]
    candidates: tuple[Candidate, ...]
    calibration_conditions: Mapping[str, DatasetSpec]
    comparison_datasets: Mapping[str, DatasetSpec]
    tool_routes: Mapping[str, ToolRoute]
    competitor_families: tuple[CompetitorFamily, ...]
    scalability_scales: Mapping[str, DatasetSpec]
    screening_trees: int
    finalists: int
    validation_fit_trees: int
    validation_prefixes: tuple[int, ...]
    fingerprint: str

    def profile(self, profile_id: str) -> Profile:
        try:
            return self.profiles[profile_id]
        except KeyError as exc:
            raise DesignError(f"Unknown study profile {profile_id!r}") from exc

    def calibration_fit_count(self, profile_id: str) -> int:
        profile = self.profile(profile_id)
        budget = profile.calibration
        screening = (
            len(self.candidates)
            * len(budget.condition_ids)
            * budget.development_replicates
            * budget.screening_algorithm_seeds
        )
        validation = (
            self.finalists
            * len(budget.condition_ids)
            * budget.validation_replicates
            * budget.validation_algorithm_seeds
        )
        return screening + validation

    def comparison_dataset_count(self, profile_id: str) -> int:
        return sum(self.profile(profile_id).comparison.replicates.values())


def derive_seed(
    design: StudyDesign,
    *,
    phase: str,
    stream: str,
    coordinates: Mapping[str, str | int | bool],
) -> int:
    """Derive a positive 31-bit seed from stable scientific coordinates."""

    if phase not in PHASE_NAMES:
        raise DesignError(f"Unknown study phase {phase!r}")
    stream = _text(stream, "seed stream")
    normalized: dict[str, str | int | bool] = {}
    for key, value in coordinates.items():
        if not isinstance(key, str) or not key:
            raise DesignError("Seed coordinate names must be non-empty strings")
        if isinstance(value, (str, bool)) or (type(value) is int):
            normalized[key] = value
        else:
            raise DesignError(f"Seed coordinate {key!r} has an unsupported value")
    payload = {
        "coordinates": normalized,
        "derivation": SEED_DERIVATION,
        "master_seed": design.master_seed,
        "phase": phase,
        "stream": stream,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).digest()
    return (int.from_bytes(digest[:8], "big") % 2_147_483_646) + 1


def calibration_instances(
    design: StudyDesign, profile_id: str, *, split: str
) -> tuple[DatasetInstance, ...]:
    """Expand the nested calibration dataset budget without execution-order seeds."""

    if split not in {"development", "validation"}:
        raise DesignError("Calibration split must be 'development' or 'validation'")
    budget = design.profile(profile_id).calibration
    replicate_count = (
        budget.development_replicates if split == "development" else budget.validation_replicates
    )
    return tuple(
        DatasetInstance(
            dataset_id=dataset_id,
            split=split,
            replicate=replicate,
            seed=derive_seed(
                design,
                phase="calibration",
                stream="generation",
                coordinates={
                    "dataset": dataset_id,
                    "replicate": replicate,
                    "split": split,
                },
            ),
        )
        for dataset_id in budget.condition_ids
        for replicate in range(1, replicate_count + 1)
    )


def comparison_instances(design: StudyDesign, profile_id: str) -> tuple[DatasetInstance, ...]:
    """Expand held-out comparison datasets in declared panel order."""

    replicates = design.profile(profile_id).comparison.replicates
    return tuple(
        DatasetInstance(
            dataset_id=dataset_id,
            split="test",
            replicate=replicate,
            seed=derive_seed(
                design,
                phase="comparison",
                stream="generation",
                coordinates={"dataset": dataset_id, "replicate": replicate, "split": "test"},
            ),
        )
        for dataset_id in design.comparison_datasets
        if dataset_id in replicates
        for replicate in range(1, replicates[dataset_id] + 1)
    )


def eligible_routes(design: StudyDesign, dataset_id: str) -> tuple[ToolRoute, ...]:
    """Return routes admitted solely by the dataset/tool information contract."""

    try:
        dataset = design.comparison_datasets[dataset_id]
    except KeyError:
        try:
            dataset = design.scalability_scales[dataset_id]
        except KeyError as exc:
            raise DesignError(f"Unknown comparison or scalability dataset {dataset_id!r}") from exc
    return tuple(
        route
        for route in design.tool_routes.values()
        if route.required_capabilities <= dataset.capabilities
    )


def _order_kind(dataset: DatasetSpec) -> str:
    ordered = "ordered_groups" in dataset.capabilities
    lineage = "lineage_tree" in dataset.capabilities
    if ordered and lineage:
        return "ordered_and_lineage"
    if lineage:
        return "lineage"
    if ordered:
        return "ordered"
    return "unordered"


def _planned_dataset(
    design: StudyDesign,
    dataset: DatasetSpec,
    *,
    phase: str,
    role: str,
    replicate: int,
) -> PlannedDataset:
    split = "test" if phase == "comparison" else ("scale" if phase == "scalability" else role)
    scenario_request, generated_inputs = _scenario_request(
        design,
        dataset,
        phase=phase,
        role=split,
        replicate=replicate,
    )
    return PlannedDataset(
        id=f"{dataset.id}--{split}--r{replicate:02d}",
        phase=phase,
        role=role,
        condition_id=dataset.id,
        label=dataset.label,
        generator=dataset.simulator,
        scenario=dataset.scenario,
        scenario_request=scenario_request,
        simulator_parameters=_simulator_parameters(dataset),
        generated_inputs=generated_inputs,
        dimensions=dataset.dimensions,
        capabilities=dataset.capabilities,
        replicate=replicate,
        seed=derive_seed(
            design,
            phase=phase,
            stream="generation",
            coordinates={"dataset": dataset.id, "replicate": replicate, "split": split},
        ),
        order_kind=_order_kind(dataset),
    )


def _group_sizes(cells: int, groups: int, *, imbalanced: bool) -> list[int]:
    if groups == 1:
        return [cells]
    if imbalanced and groups == 3:
        first = max(1, cells // 10)
        second = max(1, (cells * 3) // 10)
        return [first, second, cells - first - second]
    base, remainder = divmod(cells, groups)
    return [base + int(index < remainder) for index in range(groups)]


def _simulator_parameters(dataset: DatasetSpec) -> dict[str, Any]:
    """Resolve complete ANDREA simulator parameters from one frozen condition."""

    dimensions = dataset.dimensions
    attributes = dataset.attributes
    if dataset.simulator == "dyngen":
        housekeeping = max(
            0, min(dimensions.genes // 10, dimensions.genes - dimensions.transcription_factors - 1)
        )
        targets = dimensions.genes - dimensions.transcription_factors - housekeeping
        return {
            "backbone_template": (
                "bifurcating" if attributes.get("topology") == "branching" else "linear"
            ),
            "num_cells": dimensions.cells,
            "num_tfs": dimensions.transcription_factors,
            "num_targets": targets,
            "num_hks": housekeeping,
            "distance_metric": "pearson",
            "simulation_params": {
                "num_simulations": 32,
                "census_interval": 4.0,
                "num_knockdown_simulations": 0,
                "compute_dimred": False,
                "compute_rna_velocity": False,
                "kinetics_noise_kind": "simple",
                "kinetics_noise_mean": 1.0,
                "kinetics_noise_sd": 0.02 if attributes.get("noise") == "high" else 0.005,
            },
            "experiment_params": {
                "kind": "snapshot",
                "map_reference_cpm": False,
                "map_reference_ls": False,
            },
        }
    if dataset.simulator == "scmultisim":
        imbalanced = attributes.get("balance") == "imbalanced"
        noisy = attributes.get("noise") not in {None, "low"}
        return {
            "grn_source": "input_tsv",
            "tree_preset": "input_newick",
            "population_mode": "discrete",
            "discrete_population": {
                "pop_size": _group_sizes(dimensions.cells, dimensions.groups, imbalanced=imbalanced)
            },
            "num_cells": dimensions.cells,
            "num_genes": dimensions.genes,
            "num_cifs": min(50, max(10, dimensions.genes // 20)),
            "speed_up": True,
            "velocity": {"enabled": False},
            "dynamic_grn": {
                "enabled": True,
                "num_steps": 5 if dimensions.genes <= 100 else 10,
                "cell_per_step": 1,
                "num_changing_edges": min(50, max(6, dimensions.genes // 20)),
                "create_tf_edges": False,
                "weight_mean": 0,
                "weight_sd": 0.8,
            },
            "technical_noise": (
                {
                    "enabled": True,
                    "protocol": "UMI",
                    "alpha_mean": 0.1,
                    "alpha_sd": 0.02,
                    "depth_mean": 100000,
                    "depth_sd": 3000,
                    "n_pcr1": 16,
                    "n_pcr2": 10,
                    "atac_obs_prob": 0.3,
                    "atac_sd_frac": 0.5,
                }
                if noisy
                else {"enabled": False}
            ),
            "batch_effect": {"enabled": False},
        }
    if dataset.simulator == "sergio":
        return {
            "input_preset": "demo_steady_state",
            "simulation_mode": "steady_state",
            "number_genes": 100,
            "number_bins": 9,
            "number_sc": 40,
            "noise_params": 0.3,
            "noise_type": "dpd",
            "decays": 0.8,
            "sampling_state": 1,
            "window_length": 100,
            "dt": 0.01,
            "shared_coop_state": 2,
            "technical_noise": {
                "outlier_enabled": False,
                "library_size_enabled": False,
                "dropout_enabled": False,
                "convert_to_umi": False,
            },
        }
    if dataset.simulator == "boolode":
        return {
            "model_preset": "GSD",
            "model_type": "hill",
            "simulation_time": 8,
            "num_cells": dimensions.cells,
            "n_clusters": dimensions.groups,
            "sample_cells": False,
            "sample_parameters": True,
            "sample_std": 0.1,
            "identical_parameters": False,
            "integration_step_size": 0.01,
            "dropout": {"enabled": False},
        }
    if dataset.simulator == "scale-stress":
        return {
            "groups": _group_sizes(
                dimensions.cells,
                dimensions.groups,
                imbalanced=False,
            ),
            "genes": dimensions.genes,
            "transcription_factors": dimensions.transcription_factors,
            "network_mode": "dynamic",
            "active_tf_fraction": 0.8,
            "regulated_target_fraction": 0.75,
            "low_detection_target_fraction": 0.1,
            "regulators_per_target": min(3, dimensions.transcription_factors),
            "rewiring_fraction": 0.34,
            "latent_factors": min(8, max(2, dimensions.transcription_factors // 10)),
            "library_size_log_sd": 0.25,
            "noise_sd": 0.25,
            "base_dropout_probability": 0.08,
            "low_detection_dropout_probability": 0.65,
        }
    raise DesignError(f"No materialization contract exists for simulator {dataset.simulator!r}")


def _scenario_request(
    design: StudyDesign,
    dataset: DatasetSpec,
    *,
    phase: str,
    role: str,
    replicate: int,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Build the exact scenario payload and deterministic synthetic input recipes."""

    extras = ["groups", "tf_list"]
    if "ordered_groups" in dataset.capabilities:
        extras.append("column_phenotypes")
    if "lineage_tree" in dataset.capabilities:
        extras.append("lineage_tree")
    scenario: dict[str, Any] = {
        "schema_version": "1.0",
        "id": f"article_{phase}_{dataset.id.replace('-', '_')}_{role}_r{replicate:02d}",
        "data_axes": {
            "measurement": "rna_expression",
            "resolution": "single_cell",
            "column_kind": "cells",
            "experimental_design": (
                "steady_state"
                if dataset.simulator == "sergio"
                else ("differentiation" if dataset.simulator == "scmultisim" else "trajectory")
            ),
        },
        "truth_requirements": {"contexts": ["global", "group"]},
        "organism": {"taxonomic_group": "synthetic", "ncbi_taxon_id": None},
        "requested_extras": extras,
    }
    generated_inputs: dict[str, Mapping[str, Any]] = {}
    if dataset.simulator == "scmultisim":
        network_seed = derive_seed(
            design,
            phase=phase,
            stream="network",
            coordinates={"dataset": dataset.id, "replicate": replicate, "role": role},
        )
        scenario["inputs"] = {
            "regulatory_network": {
                "path": "inputs/regulatory-network.tsv",
                "format": "tsv",
                "organism": {"taxonomic_group": "synthetic", "ncbi_taxon_id": None},
                "description": "Role-exclusive deterministic synthetic regulatory topology.",
            },
            "tree_newick": {
                "path": "inputs/tree.newick",
                "format": "newick",
                "description": "Deterministic population tree for this synthetic condition.",
            },
        }
        generated_inputs = {
            "regulatory_network": {
                "kind": "directed_scale_free_v1",
                "path": "inputs/regulatory-network.tsv",
                "genes": dataset.dimensions.genes,
                "transcription_factors": dataset.dimensions.transcription_factors,
                "edges_per_target": 2,
                "seed": network_seed,
            },
            "tree_newick": {
                "kind": "balanced_population_tree_v1",
                "path": "inputs/tree.newick",
                "groups": dataset.dimensions.groups,
            },
        }
    return scenario, generated_inputs


def profile_view(design: StudyDesign, profile_id: str) -> ProfileView:
    """Expand one coverage profile into the central planner-facing shape.

    The returned dataset mapping is keyed first by phase, then by scientific
    role.  It contains only prospective, deterministic inputs; resource counts
    and task order are resolved later by the execution engine.
    """

    profile = design.profile(profile_id)
    calibration: dict[str, tuple[PlannedDataset, ...]] = {}
    for role, count in (
        ("development", profile.calibration.development_replicates),
        ("validation", profile.calibration.validation_replicates),
    ):
        calibration[role] = tuple(
            _planned_dataset(
                design,
                design.calibration_conditions[condition_id],
                phase="calibration",
                role=role,
                replicate=replicate,
            )
            for condition_id in profile.calibration.condition_ids
            for replicate in range(1, count + 1)
        )

    comparison = tuple(
        _planned_dataset(
            design,
            design.comparison_datasets[dataset_id],
            phase="comparison",
            role="test",
            replicate=replicate,
        )
        for dataset_id in design.comparison_datasets
        if dataset_id in profile.comparison.replicates
        for replicate in range(1, profile.comparison.replicates[dataset_id] + 1)
    )

    scalability_ladder = tuple(
        _planned_dataset(
            design,
            design.scalability_scales[scale_id],
            phase="scalability",
            role="ladder",
            replicate=replicate,
        )
        for scale_id in profile.scalability.scale_ids
        for replicate in range(1, profile.scalability.ladder_replicates + 1)
    )
    scaling_dataset = design.scalability_scales[profile.scalability.scaling_scale_id]
    thread_scaling = tuple(
        _planned_dataset(
            design,
            scaling_dataset,
            phase="scalability",
            role="thread_scaling",
            replicate=replicate,
        )
        for replicate in range(1, profile.scalability.scaling_replicates + 1)
    )
    return ProfileView(
        id=profile_id,
        profile=profile,
        candidates=design.candidates,
        datasets={
            "calibration": calibration,
            "comparison": {"test": comparison},
            "scalability": {"ladder": scalability_ladder, "thread_scaling": thread_scaling},
        },
        routes=design.tool_routes,
        families=design.competitor_families,
        screening_trees=design.screening_trees,
        finalists=design.finalists,
        validation_fit_trees=design.validation_fit_trees,
        validation_prefixes=design.validation_prefixes,
    )


def resolve_thread_points(points: Sequence[str], available_cores: int) -> tuple[int, ...]:
    """Resolve portable thread tokens against the current cgroup/affinity budget."""

    available_cores = _integer(available_cores, "available cores", minimum=1)
    resolved: list[int] = []
    for point in points:
        if point not in THREAD_POINT_NAMES:
            raise DesignError(f"Unknown thread point {point!r}")
        value = {"one": 1, "standard": min(8, available_cores), "all": available_cores}[point]
        if value not in resolved:
            resolved.append(value)
    return tuple(resolved)


def _dimensions(value: object, label: str) -> Dimensions:
    item = _object(value, label)
    _require_exact_keys(item, {"cells", "genes", "transcription_factors", "groups"}, label)
    result = Dimensions(
        cells=_integer(item["cells"], f"{label}.cells", minimum=1),
        genes=_integer(item["genes"], f"{label}.genes", minimum=2),
        transcription_factors=_integer(
            item["transcription_factors"], f"{label}.transcription_factors", minimum=1
        ),
        groups=_integer(item["groups"], f"{label}.groups", minimum=1),
    )
    if result.transcription_factors > result.genes:
        raise DesignError(f"{label} declares more transcription factors than genes")
    return result


def _datasets(value: object, label: str) -> dict[str, DatasetSpec]:
    items = _array(value, label)
    ids = _unique((_text(_object(item, label).get("id"), f"{label} id") for item in items), label)
    result: dict[str, DatasetSpec] = {}
    for dataset_id, raw_item in zip(ids, items, strict=True):
        item = _object(raw_item, f"{label}.{dataset_id}")
        _require_exact_keys(
            item,
            {"id", "label", "simulator", "scenario", "dimensions", "capabilities", "attributes"},
            f"{label}.{dataset_id}",
        )
        capabilities = frozenset(
            _text(member, f"{label}.{dataset_id}.capability")
            for member in _array(item["capabilities"], f"{label}.{dataset_id}.capabilities")
        )
        if len(capabilities) != len(item["capabilities"]):
            raise DesignError(f"{label}.{dataset_id}.capabilities contains duplicates")
        unknown_capabilities = sorted(capabilities - DATASET_CAPABILITIES)
        if unknown_capabilities:
            raise DesignError(
                f"{label}.{dataset_id}.capabilities contains unsupported values: "
                f"{unknown_capabilities}"
            )
        if not {"group_labels", "tf_list"} <= capabilities:
            raise DesignError(f"{label}.{dataset_id} must provide group_labels and tf_list")
        result[dataset_id] = DatasetSpec(
            id=dataset_id,
            label=_text(item["label"], f"{label}.{dataset_id}.label"),
            simulator=_text(item["simulator"], f"{label}.{dataset_id}.simulator"),
            scenario=_text(item["scenario"], f"{label}.{dataset_id}.scenario"),
            dimensions=_dimensions(item["dimensions"], f"{label}.{dataset_id}.dimensions"),
            capabilities=capabilities,
            attributes=dict(_object(item["attributes"], f"{label}.{dataset_id}.attributes")),
        )
    return result


def _candidates(calibration: Mapping[str, Any]) -> tuple[Candidate, ...]:
    domains = _object(calibration.get("parameter_domains"), "calibration.parameter_domains")
    expected_parameters = set(domains)
    if not expected_parameters:
        raise DesignError("calibration.parameter_domains must not be empty")
    candidates_raw = _array(calibration.get("candidates"), "calibration.candidates")
    if len(candidates_raw) != 27:
        raise DesignError("Calibration must declare exactly 27 equally eligible candidates")
    identifiers = _unique(
        (
            _text(_object(item, "calibration candidate").get("id"), "candidate id")
            for item in candidates_raw
        ),
        "calibration candidate ids",
    )
    candidates: list[Candidate] = []
    observed: set[bytes] = set()
    covered: dict[str, set[bytes]] = {parameter: set() for parameter in domains}
    for candidate_id, raw_candidate in zip(identifiers, candidates_raw, strict=True):
        candidate = _object(raw_candidate, f"calibration candidate {candidate_id}")
        _require_exact_keys(candidate, {"id", "parameters"}, f"candidate {candidate_id}")
        parameters = dict(_object(candidate["parameters"], f"candidate {candidate_id}.parameters"))
        _require_exact_keys(parameters, expected_parameters, f"candidate {candidate_id}.parameters")
        encoded = _canonical_bytes(parameters)
        if encoded in observed:
            raise DesignError(f"Candidate {candidate_id!r} duplicates another parameter vector")
        observed.add(encoded)
        for parameter, allowed_raw in domains.items():
            allowed = _array(allowed_raw, f"parameter domain {parameter}")
            value_bytes = _canonical_bytes(parameters[parameter])
            allowed_bytes = {_canonical_bytes(value) for value in allowed}
            if value_bytes not in allowed_bytes:
                raise DesignError(
                    f"Candidate {candidate_id!r} uses an undeclared value for {parameter!r}"
                )
            covered[parameter].add(value_bytes)
        candidates.append(Candidate(id=candidate_id, parameters=parameters))
    for parameter, allowed_raw in domains.items():
        allowed = {_canonical_bytes(value) for value in _array(allowed_raw, parameter)}
        if covered[parameter] != allowed:
            raise DesignError(f"Candidates do not cover every declared value of {parameter!r}")
        counts = [
            sum(
                _canonical_bytes(candidate.parameters[parameter]) == value
                for candidate in candidates
            )
            for value in allowed
        ]
        if max(counts) - min(counts) > 1:
            raise DesignError(f"Candidates do not balance the declared values of {parameter!r}")
    for left, right in itertools.combinations(domains, 2):
        observed_pairs = {
            (
                _canonical_bytes(candidate.parameters[left]),
                _canonical_bytes(candidate.parameters[right]),
            )
            for candidate in candidates
        }
        expected_pairs = {
            (_canonical_bytes(left_value), _canonical_bytes(right_value))
            for left_value in _array(domains[left], left)
            for right_value in _array(domains[right], right)
        }
        if observed_pairs != expected_pairs:
            raise DesignError(
                f"Candidates do not cover every value pair for {left!r} and {right!r}"
            )
    return tuple(candidates)


def _routes(comparison: Mapping[str, Any]) -> dict[str, ToolRoute]:
    routes_raw = _array(comparison.get("routes"), "comparison.routes")
    ids = _unique(
        (_text(_object(item, "comparison route").get("id"), "route id") for item in routes_raw),
        "comparison route ids",
    )
    routes: dict[str, ToolRoute] = {}
    for route_id, raw_route in zip(ids, routes_raw, strict=True):
        route = _object(raw_route, f"comparison.routes.{route_id}")
        _require_exact_keys(
            route,
            {
                "id",
                "tool_id",
                "configuration_id",
                "delivery",
                "required_capabilities",
                "inputs",
                "output_semantics",
                "parameters",
                "seed_parameter",
                "max_threads",
            },
            f"comparison.routes.{route_id}",
        )
        tool_id = _text(route["tool_id"], f"{route_id}.tool_id")
        if tool_id not in ARTICLE_TOOL_IDS:
            raise DesignError(f"Route {route_id!r} names unsupported tool id {tool_id!r}")
        delivery = _text(route["delivery"], f"{route_id}.delivery")
        if delivery not in ROUTE_DELIVERIES:
            raise DesignError(f"Route {route_id!r} has unsupported delivery {delivery!r}")
        required_capabilities = frozenset(
            _text(value, f"{route_id}.required_capabilities")
            for value in _array(route["required_capabilities"], f"{route_id}.required_capabilities")
        )
        if len(required_capabilities) != len(route["required_capabilities"]):
            raise DesignError(f"Route {route_id!r} repeats a required capability")
        unknown_capabilities = sorted(required_capabilities - DATASET_CAPABILITIES)
        if unknown_capabilities:
            raise DesignError(
                f"Route {route_id!r} requires unsupported capabilities: {unknown_capabilities}"
            )
        inputs = tuple(
            _text(value, f"{route_id}.inputs")
            for value in _array(route["inputs"], f"{route_id}.inputs")
        )
        if len(inputs) != len(set(inputs)) or not inputs or inputs[0] != "expression":
            raise DesignError(f"Route {route_id!r} inputs must be unique and start with expression")
        output_semantics = _text(route["output_semantics"], f"{route_id}.output_semantics")
        if output_semantics not in OUTPUT_SEMANTICS:
            raise DesignError(
                f"Route {route_id!r} has unsupported output semantics {output_semantics!r}"
            )
        routes[route_id] = ToolRoute(
            id=route_id,
            tool_id=tool_id,
            configuration_id=_text(route["configuration_id"], f"{route_id}.configuration_id"),
            delivery=delivery,
            required_capabilities=required_capabilities,
            inputs=inputs,
            output_semantics=output_semantics,
            parameters=dict(_object(route["parameters"], f"{route_id}.parameters")),
            seed_parameter=(
                _text(route["seed_parameter"], f"{route_id}.seed_parameter")
                if route["seed_parameter"] is not None
                else None
            ),
            max_threads=(
                _integer(route["max_threads"], f"{route_id}.max_threads", minimum=1)
                if route["max_threads"] is not None
                else None
            ),
        )
    return routes


def _families(
    comparison: Mapping[str, Any], routes: Mapping[str, ToolRoute]
) -> tuple[CompetitorFamily, ...]:
    families_raw = _array(comparison.get("families"), "comparison.families")
    ids = _unique(
        (_text(_object(item, "competitor family").get("id"), "family id") for item in families_raw),
        "competitor family ids",
    )
    families: list[CompetitorFamily] = []
    primary_count = 0
    for family_id, raw_family in zip(ids, families_raw, strict=True):
        family = _object(raw_family, f"comparison.families.{family_id}")
        _require_exact_keys(
            family,
            {"id", "label", "rationale", "statistical_role", "route_ids"},
            f"comparison.families.{family_id}",
        )
        route_ids = _unique(
            (
                _text(value, f"{family_id}.route_ids")
                for value in _array(family["route_ids"], f"{family_id}.route_ids")
            ),
            f"{family_id}.route_ids",
        )
        unknown = sorted(set(route_ids) - set(routes))
        if unknown:
            raise DesignError(f"Family {family_id!r} names unknown routes: {unknown}")
        role = _text(family["statistical_role"], f"{family_id}.statistical_role")
        if role == "primary_friedman_holm":
            primary_count += 1
        elif role != "separate_family_analysis":
            raise DesignError(f"Unknown statistical role {role!r}")
        families.append(
            CompetitorFamily(
                id=family_id,
                label=_text(family["label"], f"{family_id}.label"),
                rationale=_text(family["rationale"], f"{family_id}.rationale"),
                statistical_role=role,
                route_ids=route_ids,
            )
        )
    if primary_count != 1:
        raise DesignError("Exactly one information-matched family must define the primary ranking")
    for family in families:
        if "spathi" not in family.route_ids:
            raise DesignError(f"Competitor family {family.id!r} must include SPATHI")
    return tuple(families)


def _profile(
    profile_id: str,
    raw: object,
    *,
    condition_ids: set[str],
    comparison_ids: set[str],
    scale_ids: set[str],
) -> Profile:
    profile = _object(raw, f"profiles.{profile_id}")
    _require_exact_keys(profile, set(PHASE_NAMES), f"profiles.{profile_id}")
    calibration = _object(profile["calibration"], f"profiles.{profile_id}.calibration")
    _require_exact_keys(
        calibration,
        {
            "condition_ids",
            "development_replicates",
            "validation_replicates",
            "screening_algorithm_seeds",
            "validation_algorithm_seeds",
        },
        f"profiles.{profile_id}.calibration",
    )
    selected_conditions = _unique(
        (
            _text(value, "calibration condition id")
            for value in _array(calibration["condition_ids"], "calibration.condition_ids")
        ),
        f"{profile_id} calibration conditions",
    )
    if not selected_conditions or not set(selected_conditions) <= condition_ids:
        raise DesignError(f"Profile {profile_id!r} selects unknown calibration conditions")

    comparison = _object(profile["comparison"], f"profiles.{profile_id}.comparison")
    _require_exact_keys(
        comparison,
        {"replicates", "task_timeout_seconds"},
        f"profiles.{profile_id}.comparison",
    )
    replicate_map = _object(comparison["replicates"], f"{profile_id}.comparison.replicates")
    if not replicate_map or not set(replicate_map) <= comparison_ids:
        raise DesignError(f"Profile {profile_id!r} selects unknown comparison datasets")
    normalized_replicates = {
        str(dataset_id): _integer(count, f"{profile_id}.{dataset_id} replicates", minimum=1)
        for dataset_id, count in replicate_map.items()
    }

    scalability = _object(profile["scalability"], f"profiles.{profile_id}.scalability")
    _require_exact_keys(
        scalability,
        {
            "scale_ids",
            "ladder_replicates",
            "scaling_scale_id",
            "scaling_replicates",
            "thread_points",
            "task_timeout_seconds",
        },
        f"profiles.{profile_id}.scalability",
    )
    selected_scales = _unique(
        (
            _text(value, "scale id")
            for value in _array(scalability["scale_ids"], "scalability.scale_ids")
        ),
        f"{profile_id} scalability scales",
    )
    if not selected_scales or not set(selected_scales) <= scale_ids:
        raise DesignError(f"Profile {profile_id!r} selects unknown scalability scales")
    scaling_scale_id = _text(scalability["scaling_scale_id"], "scaling scale id")
    if scaling_scale_id not in selected_scales:
        raise DesignError(f"Profile {profile_id!r} scales threads on an absent dataset")
    thread_points = tuple(
        _text(value, "thread point")
        for value in _array(scalability["thread_points"], "scalability.thread_points")
    )
    if not thread_points or any(value not in THREAD_POINT_NAMES for value in thread_points):
        raise DesignError(f"Profile {profile_id!r} contains an unknown thread point")
    return Profile(
        id=profile_id,
        calibration=CalibrationBudget(
            condition_ids=selected_conditions,
            development_replicates=_integer(
                calibration["development_replicates"], "development replicates", minimum=1
            ),
            validation_replicates=_integer(
                calibration["validation_replicates"], "validation replicates", minimum=1
            ),
            screening_algorithm_seeds=_integer(
                calibration["screening_algorithm_seeds"],
                "screening algorithm seeds",
                minimum=1,
            ),
            validation_algorithm_seeds=_integer(
                calibration["validation_algorithm_seeds"],
                "validation algorithm seeds",
                minimum=1,
            ),
        ),
        comparison=ComparisonBudget(
            replicates=normalized_replicates,
            task_timeout_seconds=_integer(
                comparison["task_timeout_seconds"],
                "comparison task timeout",
                minimum=1,
            ),
        ),
        scalability=ScalabilityBudget(
            scale_ids=selected_scales,
            ladder_replicates=_integer(
                scalability["ladder_replicates"], "ladder replicates", minimum=1
            ),
            scaling_scale_id=scaling_scale_id,
            scaling_replicates=_integer(
                scalability["scaling_replicates"], "scaling replicates", minimum=1
            ),
            thread_points=thread_points,
            task_timeout_seconds=_integer(
                scalability["task_timeout_seconds"], "task timeout", minimum=1
            ),
        ),
    )


def parse_design(raw: Mapping[str, Any]) -> StudyDesign:
    """Validate and parse a design document without consulting runtime state."""

    raw = _object(raw, "article design")
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "study_id",
            "seed",
            "profiles",
            "calibration",
            "comparison",
            "scalability",
        },
        "article design",
    )
    if raw["schema_version"] != 1:
        raise DesignError("Only article design schema_version 1 is supported")
    study_id = _text(raw["study_id"], "study_id")
    seed = _object(raw["seed"], "seed")
    _require_exact_keys(seed, {"master", "derivation"}, "seed")
    if seed["derivation"] != SEED_DERIVATION:
        raise DesignError(f"Unsupported seed derivation {seed['derivation']!r}")
    master_seed = _integer(seed["master"], "master seed", minimum=1)

    calibration = _object(raw["calibration"], "calibration")
    _require_exact_keys(
        calibration,
        {"conditions", "parameter_domains", "parameter_policy", "candidates", "protocol"},
        "calibration",
    )
    conditions = _datasets(calibration["conditions"], "calibration.conditions")
    single_group_conditions = sorted(
        condition.id for condition in conditions.values() if condition.dimensions.groups == 1
    )
    if single_group_conditions:
        raise DesignError(
            "Calibration may contain only multi-group conditions; single-group data require "
            f"a different SPATHI weighting contract: {single_group_conditions}"
        )
    candidates = _candidates(calibration)
    parameter_policy = _object(calibration["parameter_policy"], "calibration.parameter_policy")
    _require_exact_keys(
        parameter_policy,
        {
            "calibrated",
            "calibrated_by_tree_prefix",
            "search_design",
            "fixed_scientific",
            "operational_not_tuned",
            "unavailable_in_synthetic_study",
        },
        "calibration.parameter_policy",
    )
    calibrated = _unique(
        (
            _text(value, "calibrated parameter")
            for value in _array(parameter_policy["calibrated"], "calibrated parameters")
        ),
        "calibrated parameters",
    )
    domains = _object(calibration["parameter_domains"], "calibration.parameter_domains")
    if set(calibrated) != set(domains):
        raise DesignError("Calibration policy must name every and only declared parameter")
    if parameter_policy["calibrated_by_tree_prefix"] != "n_estimators":
        raise DesignError("n_estimators must be calibrated through exact forest prefixes")
    search_design = _object(parameter_policy["search_design"], "calibration search design")
    _require_exact_keys(
        search_design,
        {"method", "seed", "runs", "guarantees", "scope"},
        "calibration search design",
    )
    if search_design["method"] != "deterministic-balanced-pairwise-covering-array":
        raise DesignError("Calibration must use the declared balanced pairwise covering array")
    if _integer(search_design["seed"], "search design seed", minimum=1) != master_seed:
        raise DesignError("Calibration covering-array seed must equal the study master seed")
    if _integer(search_design["runs"], "search design runs", minimum=1) != len(candidates):
        raise DesignError("Calibration covering-array run count differs from candidates")
    required_guarantees = {
        "every_declared_value_present",
        "marginal_level_counts_differ_by_at_most_one",
        "every_pair_of_declared_values_present",
    }
    if set(_array(search_design["guarantees"], "search design guarantees")) != required_guarantees:
        raise DesignError("Calibration covering-array guarantees are incomplete")
    if search_design["scope"] != "main_effects_and_two_factor_screening_not_full_factorial":
        raise DesignError("Calibration covering-array scope is unsupported")
    protocol = _object(calibration["protocol"], "calibration.protocol")
    _require_exact_keys(
        protocol, {"screening", "advancement", "validation", "selection"}, "calibration.protocol"
    )
    screening = _object(protocol["screening"], "calibration.protocol.screening")
    _require_exact_keys(screening, {"fit_trees", "dataset_split"}, "calibration.protocol.screening")
    if screening["dataset_split"] != "development":
        raise DesignError("Calibration screening must use the development split")
    screening_trees = _integer(screening["fit_trees"], "screening trees", minimum=1)
    advancement = _object(protocol["advancement"], "calibration.protocol.advancement")
    _require_exact_keys(
        advancement,
        {"finalists", "primary_metric", "tie_breakers", "privileged_candidates"},
        "calibration.protocol.advancement",
    )
    finalists = _integer(advancement["finalists"], "calibration finalists", minimum=1)
    if advancement["privileged_candidates"] != []:
        raise DesignError("Calibration candidates must compete without privileged advancement")
    if finalists != 4:
        raise DesignError("Calibration must advance exactly four candidates")
    validation = _object(protocol["validation"], "calibration.protocol.validation")
    _require_exact_keys(
        validation,
        {"fit_trees", "score_exact_prefixes", "dataset_split"},
        "calibration.protocol.validation",
    )
    if validation["dataset_split"] != "validation":
        raise DesignError("Finalists must be assessed on the held-out validation split")
    validation_fit_trees = _integer(validation["fit_trees"], "validation trees", minimum=1)
    validation_prefixes = tuple(
        _integer(value, "validation tree prefix", minimum=1)
        for value in _array(validation["score_exact_prefixes"], "validation prefixes")
    )
    if tuple(sorted(set(validation_prefixes))) != validation_prefixes:
        raise DesignError("Validation tree prefixes must be unique and increasing")
    if not validation_prefixes or validation_prefixes[-1] != validation_fit_trees:
        raise DesignError("The final exact tree prefix must equal the fitted tree count")
    selection = _object(protocol["selection"], "calibration.protocol.selection")
    _require_exact_keys(
        selection,
        {"primary_metric", "practical_equivalence_fraction", "tie_breakers"},
        "calibration.protocol.selection",
    )
    if selection["primary_metric"] != "aupr":
        raise DesignError("Calibration selection must use directed AUPR")
    equivalence = _number(
        selection["practical_equivalence_fraction"],
        "calibration practical-equivalence fraction",
        minimum=0.0,
    )
    if not 0.0 < equivalence < 1.0:
        raise DesignError("Calibration practical-equivalence fraction must be in (0, 1)")
    expected_tie_breakers = (
        "between_dataset_stability",
        "declared_tree_count",
    )
    if (
        tuple(_array(selection["tie_breakers"], "calibration selection tie-breakers"))
        != expected_tie_breakers
    ):
        raise DesignError("Calibration selection tie-breakers differ from the declared protocol")

    comparison = _object(raw["comparison"], "comparison")
    _require_exact_keys(
        comparison,
        {"datasets", "routes", "families", "execution", "statistics"},
        "comparison",
    )
    comparison_datasets = _datasets(comparison["datasets"], "comparison.datasets")
    single_group_comparisons = sorted(
        dataset.id for dataset in comparison_datasets.values() if dataset.dimensions.groups == 1
    )
    if single_group_comparisons:
        raise DesignError(
            "The multi-group calibration winner cannot be applied to single-group comparison "
            f"datasets: {single_group_comparisons}"
        )
    routes = _routes(comparison)
    families = _families(comparison, routes)
    execution = _object(comparison["execution"], "comparison.execution")
    _require_exact_keys(
        execution,
        {"parameter_rule", "dataset_rule", "failure_rule"},
        "comparison.execution",
    )
    if execution["dataset_rule"] != "all_contract_compatible":
        raise DesignError("Comparison datasets must be assigned only by contract compatibility")
    statistics = _object(comparison["statistics"], "comparison.statistics")
    _require_exact_keys(
        statistics,
        {
            "primary_family",
            "metrics",
            "omnibus",
            "post_hoc",
            "blocking_unit",
            "estimand",
            "alpha",
        },
        "comparison.statistics",
    )
    estimand = _object(statistics["estimand"], "comparison.statistics.estimand")
    _require_exact_keys(
        estimand,
        {"scenario_weighting", "sampling_unit", "missing_data_rule"},
        "comparison.statistics.estimand",
    )
    expected_estimand = {
        "scenario_weighting": "equal_replicates_per_scenario",
        "sampling_unit": "independent_seeded_dataset",
        "missing_data_rule": "complete_blocks_within_family_and_metric",
    }
    if dict(estimand) != expected_estimand:
        raise DesignError("Comparison estimand differs from the declared balanced design")

    scalability = _object(raw["scalability"], "scalability")
    _require_exact_keys(
        scalability,
        {"scales", "route_rule", "resource_resolution", "execution", "desist_reasons"},
        "scalability",
    )
    scales = _datasets(scalability["scales"], "scalability.scales")
    if scalability["route_rule"] != "all_contract_eligible_comparison_routes":
        raise DesignError("Scalability must not use a manually curated tool subset")
    scale_execution = _object(scalability["execution"], "scalability.execution")
    _require_exact_keys(
        scale_execution,
        {
            "ordering",
            "advance_rule",
            "timeout_rule",
        },
        "scalability.execution",
    )
    if scale_execution["ordering"] != "breadth_first_by_scale":
        raise DesignError("Scalability must schedule breadth-first by scale")
    if scale_execution["advance_rule"] != "previous_scale_completed":
        raise DesignError("A method may advance only after completing its previous scale")
    if scale_execution["timeout_rule"] != "andrea_physical_timeout":
        raise DesignError("Scalability must use ANDREA's physical-run timeout contract")
    declared_reasons = set(
        _text(value, "desist reason")
        for value in _array(scalability["desist_reasons"], "scalability.desist_reasons")
    )
    required_reasons = {
        "eligible",
        "estimated_infeasible_time",
        "completed_with_warnings",
        "timeout",
        "memory_limit",
        "runtime_failure",
        "not_attempted",
        "completed",
    }
    if declared_reasons != required_reasons:
        raise DesignError("Scalability must declare the complete canonical desist vocabulary")

    profiles_raw = _object(raw["profiles"], "profiles")
    if set(profiles_raw) != set(PROFILE_NAMES):
        raise DesignError("Profiles must be exactly smoke, reduced, and full")
    profiles = {
        profile_id: _profile(
            profile_id,
            profiles_raw[profile_id],
            condition_ids=set(conditions),
            comparison_ids=set(comparison_datasets),
            scale_ids=set(scales),
        )
        for profile_id in PROFILE_NAMES
    }
    design = StudyDesign(
        id=study_id,
        master_seed=master_seed,
        raw=dict(raw),
        profiles=profiles,
        candidates=candidates,
        calibration_conditions=conditions,
        comparison_datasets=comparison_datasets,
        tool_routes=routes,
        competitor_families=families,
        scalability_scales=scales,
        screening_trees=screening_trees,
        finalists=finalists,
        validation_fit_trees=validation_fit_trees,
        validation_prefixes=validation_prefixes,
        fingerprint=hashlib.sha256(_canonical_bytes(raw)).hexdigest(),
    )
    if design.calibration_fit_count("reduced") != 155:
        raise DesignError("The reduced calibration budget must contain exactly 155 physical fits")
    if design.calibration_fit_count("full") != 390:
        raise DesignError("The full calibration budget must contain exactly 390 physical fits")
    if design.comparison_dataset_count("reduced") != 8:
        raise DesignError("The reduced held-out comparison panel must contain 8 datasets")
    full_replication = set(design.profile("full").comparison.replicates.values())
    if full_replication != {3} or design.comparison_dataset_count("full") != 24:
        raise DesignError(
            "The full held-out comparison must contain three replicates of every scenario"
        )
    return design


def load_design(repo_root_or_path: Path | str) -> StudyDesign:
    """Load the article design from a repository root or its exact JSON path."""

    source = Path(repo_root_or_path)
    if source.is_dir():
        source = source / "experiments" / "article-study.json"
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DesignError(f"Cannot load article design {source}: {exc}") from exc
    return parse_design(_object(raw, "article design"))
