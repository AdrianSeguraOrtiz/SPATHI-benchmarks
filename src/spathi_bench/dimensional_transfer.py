"""Contracts and analysis for post-freeze dimensional accuracy transfer.

The transfer panel is intentionally distinct from the conjunctive scalability
generator. scMultiSim supplies dynamic group-specific truth here, so the panel
supports synthetic accuracy measurements at prospectively declared public
engineering scales without claiming biological realism.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .artifacts import (
    canonical_json_sha256,
    sha256_file,
)
from .config import (
    BenchmarkConfigError,
    load_json_object,
    resolve_repo_path,
    validate_json,
    validate_profile_references,
)
from .pipeline import validate_inference_attribution_contract
from .provenance import (
    frozen_path,
    frozen_profile,
    frozen_repository_root,
    validate_run_inputs,
)
from .safe_io import write_idempotent_json
from .seeds import derive_plan_seed, load_seed_plan
from .state import load_state
from .summarize import validate_summary_reports

PANEL_PATH = "experiments/v1/simulations/scmultisim/dimensional_transfer/panel.json"
TOPOLOGY_GENERATOR = "sha256-counter-balanced-regulators-v1"
EDGES_PER_TARGET = 2
GENE_LEVELS = (100, 3_000, 15_000)
TF_LEVELS = (10, 100, 1_500)
CELL_LEVELS = (300, 500, 10_000)
CONDITION_IDS = (
    "g3000_tf100_c300",
    "g15000_tf100_c300",
    "g3000_tf1500_c300",
    "g100_tf10_c300",
    "g100_tf10_c10000",
    "g100_tf10_c500_single_group",
    "g100_tf10_c300_groups_10_40_250",
)
REDUCED_CONDITION_IDS = (
    "g3000_tf100_c300",
    "g100_tf10_c300",
    "g100_tf10_c500_single_group",
    "g100_tf10_c300_groups_10_40_250",
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _counter_bytes(domain: str, target_index: int, edge_index: int, counter: int) -> bytes:
    payload = {
        "counter": counter,
        "domain": domain,
        "edge_index": edge_index,
        "generator": TOPOLOGY_GENERATOR,
        "target_index": target_index,
    }
    return hashlib.sha256(_canonical_json(payload)).digest()


def topology_rows(
    *, transcription_factors: int, targets: int | None = None
) -> list[tuple[str, str, str]]:
    """Return a deterministic sparse regulatory core with balanced TF coverage."""

    expected_targets = {10: 50, 100: 1_000, 1_500: 1_000}
    if transcription_factors not in expected_targets:
        raise BenchmarkConfigError(
            f"Transfer topology requires one of {TF_LEVELS} TFs, got {transcription_factors}"
        )
    target_count = expected_targets[transcription_factors] if targets is None else targets
    if target_count != expected_targets[transcription_factors]:
        raise BenchmarkConfigError("Transfer topology has a non-canonical target count")
    domain = f"spathi-dimensional-transfer-tf-{transcription_factors}-targets-{target_count}"
    rows: list[tuple[str, str, str]] = []
    used: dict[str, int] = defaultdict(int)
    for target_index in range(1, target_count + 1):
        selected: set[int] = set()
        for edge_index in range(EDGES_PER_TARGET):
            # The first TF-count edges guarantee that every declared regulator
            # occurs in the topology. Remaining edges use counter-mode SHA-256.
            ordinal = (target_index - 1) * EDGES_PER_TARGET + edge_index
            if ordinal < transcription_factors:
                regulator_index = ordinal + 1
                digest = _counter_bytes(domain, target_index, edge_index, 0)
            else:
                counter = 0
                while True:
                    digest = _counter_bytes(domain, target_index, edge_index, counter)
                    regulator_index = int.from_bytes(digest[:8], "big") % transcription_factors + 1
                    if regulator_index not in selected:
                        break
                    counter += 1
            selected.add(regulator_index)
            regulator = f"TF{regulator_index:04d}"
            target = f"G{target_index:05d}"
            magnitude = 0.6 + 0.1 * (digest[8] % 7)
            effect = magnitude if digest[9] % 2 == 0 else -magnitude
            rows.append((target, regulator, f"{effect:.1f}"))
            used[regulator] += 1
    expected_regulators = {f"TF{index:04d}" for index in range(1, transcription_factors + 1)}
    if set(used) != expected_regulators:
        raise AssertionError("Deterministic transfer topology does not cover every TF")
    if len(rows) != target_count * EDGES_PER_TARGET:
        raise AssertionError("Deterministic transfer topology has the wrong edge count")
    return rows


def topology_text(*, transcription_factors: int, targets: int | None = None) -> str:
    rows = topology_rows(transcription_factors=transcription_factors, targets=targets)
    return "target\tregulator\teffect\n" + "".join("\t".join(row) + "\n" for row in rows)


def _expected_condition_map() -> dict[str, tuple[int, int, int, int]]:
    return {
        "g3000_tf100_c300": (3_000, 100, 300, 1_000),
        "g15000_tf100_c300": (15_000, 100, 300, 1_000),
        "g3000_tf1500_c300": (3_000, 1_500, 300, 1_000),
        "g100_tf10_c300": (100, 10, 300, 50),
        "g100_tf10_c10000": (100, 10, 10_000, 50),
        "g100_tf10_c500_single_group": (100, 10, 500, 50),
        "g100_tf10_c300_groups_10_40_250": (100, 10, 300, 50),
    }


def condition_ids_for_budget(budget: str) -> tuple[str, ...]:
    """Return the exact ordered condition panel admitted by one budget."""

    if budget == "reduced":
        return REDUCED_CONDITION_IDS
    if budget == "full":
        return CONDITION_IDS
    raise BenchmarkConfigError(f"Unknown dimensional-transfer budget: {budget!r}")


def _validate_runs(
    repo_root: Path,
    *,
    path: str,
    condition_id: str,
    genes: int,
    tfs: int,
    cells: int,
    group_sizes: Sequence[int],
    budget: str,
) -> dict[str, Any]:
    runs_path = resolve_repo_path(repo_root, path, f"{condition_id} {budget} simulator runs")
    payload = load_json_object(runs_path, f"{condition_id} {budget} simulator runs")
    runs = payload.get("runs")
    expected_replicates = 3 if budget == "full" and condition_id == "g3000_tf100_c300" else 1
    if payload.get("schema_version") != "1.0" or not isinstance(runs, list) or len(runs) != 1:
        raise BenchmarkConfigError(f"{condition_id} {budget} must contain one simulator run")
    run = runs[0]
    params = run.get("params") if isinstance(run, Mapping) else None
    expected_params = {
        "grn_source": "input_tsv",
        "tree_preset": "input_newick",
        "population_mode": "discrete",
        "discrete_population": {"pop_size": list(group_sizes)},
        "num_cells": cells,
        "num_genes": genes,
        "num_cifs": 10 if tfs == 10 else 50,
        "speed_up": True,
        "velocity": {"enabled": False},
        "dynamic_grn": {
            "enabled": True,
            "num_steps": 5,
            "cell_per_step": 1,
            "num_changing_edges": 10 if tfs == 10 else 50,
            "create_tf_edges": False,
            "weight_mean": 0,
            "weight_sd": 0.8,
        },
        "technical_noise": {
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
        },
        "batch_effect": {"enabled": False},
    }
    if (
        not isinstance(params, Mapping)
        or run.get("simulator_id") != "scmultisim"
        or run.get("replicates") != expected_replicates
        or dict(params) != expected_params
    ):
        raise BenchmarkConfigError(f"{condition_id} {budget} differs from the transfer design")
    return {
        "path": runs_path,
        "run": dict(run),
    }


def validate_transfer_panel(repo_root: Path, panel_path: Path | None = None) -> dict[str, Any]:
    """Validate the full scale-contrast design and its bounded reduced subset."""

    root = repo_root.resolve()
    requested_path = panel_path or root / PANEL_PATH
    path = requested_path.resolve()
    if (
        Path(os.path.abspath(requested_path)) != path
        or not path.is_relative_to(root)
        or requested_path.is_symlink()
    ):
        raise BenchmarkConfigError("Dimensional-transfer panel must be a regular repository file")
    panel = load_json_object(path, "dimensional-transfer panel")
    validate_json(
        panel,
        root / "schemas/dimensional-transfer-panel.schema.json",
        "dimensional-transfer panel",
    )
    if panel.get("seed_plan") != "experiments/v1/seeds/seed-plan.json":
        raise BenchmarkConfigError("Dimensional-transfer panel must use the canonical seed plan")
    design = panel["design"]
    if (
        design["cell_levels"] != list(CELL_LEVELS)
        or design["gene_levels"] != list(GENE_LEVELS)
        or design["tf_levels"] != list(TF_LEVELS)
        or design["edges_per_target"] != EDGES_PER_TARGET
    ):
        raise BenchmarkConfigError("Dimensional-transfer panel differs from its v1 design")

    expected = _expected_condition_map()
    conditions = panel["conditions"]
    if [item["id"] for item in conditions] != list(CONDITION_IDS):
        raise BenchmarkConfigError("Dimensional-transfer condition order is not canonical")
    observed = {
        item["id"]: (
            item["genes"],
            item["transcription_factors"],
            item["cells"],
            item["targets_in_regulatory_core"],
        )
        for item in conditions
    }
    if observed != expected:
        raise BenchmarkConfigError(
            "Dimensional-transfer panel is not the two anchors plus three one-axis scale endpoints"
        )
    expected_roles = {
        "g3000_tf100_c300": ("scale_anchor", 3),
        "g15000_tf100_c300": ("scale_endpoint", 3),
        "g3000_tf1500_c300": ("scale_endpoint", 3),
        "g100_tf10_c300": ("scale_anchor", 3),
        "g100_tf10_c10000": ("scale_endpoint", 3),
        "g100_tf10_c500_single_group": ("single_group_routing_control", 1),
        "g100_tf10_c300_groups_10_40_250": ("extreme_group_imbalance_control", 3),
    }
    if {
        str(item["id"]): (item.get("analysis_role"), item.get("expected_group_contexts"))
        for item in conditions
    } != expected_roles:
        raise BenchmarkConfigError("Dimensional-transfer condition roles are not canonical")
    expected_replicates_by_condition = {
        "g3000_tf100_c300": {"reduced": 1, "full": 3},
        "g15000_tf100_c300": {"full": 1},
        "g3000_tf1500_c300": {"full": 1},
        "g100_tf10_c300": {"reduced": 1, "full": 1},
        "g100_tf10_c10000": {"full": 1},
        "g100_tf10_c500_single_group": {"reduced": 1, "full": 1},
        "g100_tf10_c300_groups_10_40_250": {"reduced": 1, "full": 1},
    }
    if {
        str(item["id"]): item.get("replicates") for item in conditions
    } != expected_replicates_by_condition:
        raise BenchmarkConfigError("Dimensional-transfer replicate budgets are not canonical")
    plan = load_seed_plan(root / panel["seed_plan"])
    generation_seeds: set[int] = set()
    for condition in conditions:
        condition_id = condition["id"]
        expected_budgets = set(expected_replicates_by_condition[condition_id])
        if (
            set(condition["runs"]) != expected_budgets
            or set(condition["replicates"]) != expected_budgets
        ):
            raise BenchmarkConfigError(
                f"{condition_id} is assigned to a non-canonical transfer budget"
            )
        if sum(int(value) for value in condition["group_sizes"]) != int(condition["cells"]):
            raise BenchmarkConfigError(f"{condition_id} group sizes do not sum to cell count")
        scenario_path = resolve_repo_path(
            root, condition["scenario"], f"{condition_id} scMultiSim scenario"
        )
        scenario = load_json_object(scenario_path, f"{condition_id} scMultiSim scenario")
        raw_inputs = scenario.get("inputs")
        if (
            scenario.get("schema_version") != "1.0"
            or scenario.get("truth_requirements") != {"contexts": ["global", "group"]}
            or not isinstance(raw_inputs, Mapping)
            or set(raw_inputs) != {"regulatory_network", "tree_newick"}
        ):
            raise BenchmarkConfigError(f"{condition_id} has an invalid scMultiSim scenario")
        topology_candidate = scenario_path.parent / raw_inputs["regulatory_network"]["path"]
        topology_path = topology_candidate.resolve()
        expected_topology = topology_text(
            transcription_factors=int(condition["transcription_factors"]),
            targets=int(condition["targets_in_regulatory_core"]),
        )
        if (
            Path(os.path.abspath(topology_candidate)) != topology_path
            or not topology_path.is_relative_to(root)
            or topology_candidate.is_symlink()
            or topology_path.read_text(encoding="utf-8") != expected_topology
        ):
            raise BenchmarkConfigError(f"{condition_id} regulatory topology is not canonical")
        tree_candidate = scenario_path.parent / raw_inputs["tree_newick"]["path"]
        tree_path = tree_candidate.resolve()
        expected_tree = (
            "(A:1.5);\n"
            if condition_id == "g100_tf10_c500_single_group"
            else "((A:1,B:1):1,C:2);\n"
        )
        if (
            Path(os.path.abspath(tree_candidate)) != tree_path
            or not tree_path.is_relative_to(root)
            or tree_candidate.is_symlink()
            or tree_path.read_text(encoding="utf-8") != expected_tree
        ):
            raise BenchmarkConfigError(
                f"{condition_id} does not use its canonical synthetic lineage tree"
            )
        for budget in condition["runs"]:
            validated = _validate_runs(
                root,
                path=condition["runs"][budget],
                condition_id=condition_id,
                genes=int(condition["genes"]),
                tfs=int(condition["transcription_factors"]),
                cells=int(condition["cells"]),
                group_sizes=[int(value) for value in condition["group_sizes"]],
                budget=budget,
            )
            run = validated["run"]
            run_path = validated["path"].relative_to(root).as_posix()
            expected_span = int(condition["replicates"][budget])
            for replicate_index in range(1, expected_span + 1):
                expected_seed = _registered_generation_seed(
                    plan,
                    path=run_path,
                    run_id=str(run["run_id"]),
                    replicate_index=replicate_index,
                )
                observed_seed = int(run["base_seed"]) + replicate_index - 1
                if observed_seed != expected_seed or observed_seed in generation_seeds:
                    raise BenchmarkConfigError(
                        f"{condition_id} {budget} has stale or reused generation seeds"
                    )
                generation_seeds.add(observed_seed)
    return panel


def _registered_generation_seed(
    plan: Mapping[str, Any], *, path: str, run_id: str, replicate_index: int
) -> int:
    from .seeds import generation_seed_for_consumer

    return generation_seed_for_consumer(
        plan,
        path=path,
        run_id=run_id,
        replicate_index=replicate_index,
    )


def transfer_algorithm_seed(repo_root: Path, *, budget: str) -> tuple[int, dict[str, Any]]:
    if budget not in {"reduced", "full"}:
        raise BenchmarkConfigError(f"Unknown dimensional-transfer budget: {budget!r}")
    derivation = {
        "namespace": f"{budget}.dimensional_transfer",
        "stream": "inference",
        "coordinates": {"purpose": "algorithm_replicate", "replicate_index": 1},
    }
    plan = load_seed_plan(repo_root / "experiments/v1/seeds/seed-plan.json")
    return (
        derive_plan_seed(
            plan,
            namespace=derivation["namespace"],
            stream=derivation["stream"],
            coordinates=derivation["coordinates"],
        ),
        derivation,
    )


def _reference(repo_root: Path, path: Path) -> dict[str, str]:
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve()
    if (
        path.is_symlink()
        or absolute != resolved
        or not resolved.is_file()
        or not resolved.is_relative_to(repo_root.resolve())
    ):
        raise BenchmarkConfigError(f"Dimensional-transfer evidence input is missing: {path}")
    return {
        "path": resolved.relative_to(repo_root.resolve()).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _manifest_file(manifest: Mapping[str, Any], *, relative: str, label: str) -> Mapping[str, Any]:
    raw_files = manifest.get("files")
    matches = (
        [
            entry
            for entry in raw_files
            if isinstance(entry, Mapping) and entry.get("path") == relative
        ]
        if isinstance(raw_files, list)
        else []
    )
    if len(matches) != 1:
        raise BenchmarkConfigError(f"Frozen input snapshot does not bind exactly one {label}")
    return matches[0]


def _canonical_artifact_store(repo_root: Path, run_root: Path, state: Mapping[str, Any]) -> Path:
    resolved_run = run_root.resolve()
    absolute_run = Path(os.path.abspath(run_root))
    work_root = (repo_root.resolve() / "work").resolve()
    try:
        version_root = resolved_run.parents[1]
    except IndexError as exc:  # pragma: no cover - guarded by normal workflow layout
        raise BenchmarkConfigError("Dimensional-transfer run root is too shallow") from exc
    if (
        absolute_run != resolved_run
        or not resolved_run.is_relative_to(work_root)
        or not version_root.is_relative_to(work_root)
        or version_root.name != "v1"
    ):
        raise BenchmarkConfigError(
            "Dimensional-transfer run root is outside the canonical work tree layout"
        )
    expected = version_root / "artifacts"
    raw_store = state.get("artifact_store")
    if not isinstance(raw_store, str) or not raw_store:
        raise BenchmarkConfigError("Dimensional-transfer state has no artifact store")
    declared = Path(os.path.abspath(raw_store))
    if declared != expected or declared.resolve() != expected or declared.is_symlink():
        raise BenchmarkConfigError(
            "Dimensional-transfer state does not bind the canonical shared artifact store"
        )
    return expected


def _bound_analysis_inputs(
    *,
    repo_root: Path,
    profile_path: Path,
    selected_config_path: Path,
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    """Load analysis inputs only after proving they are the pipeline's frozen bytes."""

    root = repo_root.resolve()
    live_profile_reference = _reference(root, profile_path)
    live_profile = load_json_object(profile_path, "dimensional-transfer runtime profile")
    validate_json(
        live_profile,
        root / "schemas/profile.schema.json",
        "dimensional-transfer runtime profile",
    )
    profile_id = live_profile.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        raise BenchmarkConfigError("Dimensional-transfer runtime profile has no identity")

    state = load_state(run_root, profile_id)
    snapshot_path = (run_root / "input_snapshot/manifest.json").resolve()
    raw_snapshot = state.get("input_snapshot")
    if (
        not isinstance(raw_snapshot, str)
        or Path(os.path.abspath(raw_snapshot)) != snapshot_path
        or snapshot_path.is_symlink()
    ):
        raise BenchmarkConfigError(
            "Dimensional-transfer state does not bind its canonical frozen input snapshot"
        )
    manifest = validate_run_inputs(root, profile_id, run_root)
    profile_source = manifest.get("profile_source")
    if not isinstance(profile_source, Mapping):
        raise BenchmarkConfigError("Frozen dimensional-transfer profile source is missing")
    profile_entry = _manifest_file(
        manifest,
        relative=live_profile_reference["path"],
        label="runtime profile",
    )
    if (
        profile_source.get("path") != live_profile_reference["path"]
        or profile_source.get("sha256") != live_profile_reference["sha256"]
        or profile_entry.get("sha256") != live_profile_reference["sha256"]
    ):
        raise BenchmarkConfigError(
            "Dimensional-transfer runtime profile differs from the frozen pipeline input"
        )

    frozen_root = frozen_repository_root(run_root)
    profile = frozen_profile(run_root, profile_id)
    validate_profile_references(frozen_root, profile)
    if profile != live_profile:
        raise BenchmarkConfigError(
            "Dimensional-transfer runtime profile differs from its frozen copy"
        )

    raw_production = profile.get("selected_config")
    if not isinstance(raw_production, str) or not raw_production:
        raise BenchmarkConfigError(
            "Dimensional-transfer runtime profile has no frozen production policy"
        )
    live_production_reference = _reference(root, selected_config_path)
    if live_production_reference["path"] != raw_production:
        raise BenchmarkConfigError(
            "Dimensional-transfer analyzer was given a different production policy"
        )
    production_entry = _manifest_file(
        manifest,
        relative=raw_production,
        label="production policy",
    )
    frozen_production_path = frozen_path(
        run_root, raw_production, "frozen dimensional-transfer production policy"
    )
    if (
        production_entry.get("sha256") != live_production_reference["sha256"]
        or sha256_file(frozen_production_path) != live_production_reference["sha256"]
    ):
        raise BenchmarkConfigError(
            "Dimensional-transfer production policy differs from the frozen pipeline input"
        )
    production = load_json_object(
        frozen_production_path, "frozen dimensional-transfer production policy"
    )
    validate_json(
        production,
        frozen_root / "schemas/spathi-selected-config.schema.json",
        "frozen SPATHI production config",
    )
    policy = production.get("selected_policy")
    group_policy = policy.get("group_policy") if isinstance(policy, Mapping) else None
    if not isinstance(group_policy, Mapping) or policy.get(
        "policy_sha256"
    ) != canonical_json_sha256(dict(group_policy)):
        raise BenchmarkConfigError("Frozen SPATHI production policy has a stale digest")

    canonical_store = _canonical_artifact_store(root, run_root, state)
    validate_inference_attribution_contract(
        repo_root=root,
        profile_id=profile_id,
        run_root=run_root,
    )
    return profile, production, state, frozen_root, canonical_store


def _finite(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError(f"Dimensional-transfer metric {label} is not numeric") from exc
    if not math.isfinite(number):
        raise BenchmarkConfigError(f"Dimensional-transfer metric {label} is not finite")
    return number


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return fmean(observed) if observed else None


def analyze_dimensional_transfer(
    *,
    repo_root: Path,
    budget: str,
    panel_path: Path,
    profile_path: Path,
    selected_config_path: Path,
    run_root: Path,
    output_path: Path,
) -> Path:
    """Build fail-closed, descriptive transfer evidence from ANDREA evaluations."""

    root = repo_root.resolve()
    panel = validate_transfer_panel(root, panel_path)
    profile, production, state, frozen_root, canonical_store = _bound_analysis_inputs(
        repo_root=root,
        profile_path=profile_path,
        selected_config_path=selected_config_path,
        run_root=run_root,
    )
    if (
        budget not in {"reduced", "full"}
        or profile.get("id") != f"dimensional-transfer-{budget}-runtime"
        or profile.get("purpose") != "dimensional_transfer"
        or production.get("calibration_budget") != budget
    ):
        raise BenchmarkConfigError(
            "Dimensional-transfer profile, production policy, and requested budget differ"
        )
    production_policy = production["selected_policy"]
    production_id = str(production_policy["policy_id"])
    production_configuration_sha256 = canonical_json_sha256({"group_policy": production_policy["group_policy"]})
    metrics_path, summary_path = validate_summary_reports(state, run_root)
    if state.get("purpose") != "dimensional_transfer" or state.get("status") != "completed":
        raise BenchmarkConfigError("Dimensional-transfer pipeline is not completely summarized")
    datasets = state.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise BenchmarkConfigError("Dimensional-transfer state contains no datasets")
    active_condition_ids = condition_ids_for_budget(budget)
    active_conditions = [
        item for item in panel["conditions"] if str(item["id"]) in active_condition_ids
    ]
    expected_counts = {
        str(item["id"]): int(item["replicates"][budget]) for item in active_conditions
    }
    expected_datasets = sum(expected_counts.values())
    if len(datasets) != expected_datasets:
        raise BenchmarkConfigError(
            f"Dimensional-transfer requires {expected_datasets} datasets, found {len(datasets)}"
        )
    expected_jobs = {
        str(condition["id"]): str(condition["runs"][budget]) for condition in active_conditions
    }
    profile_jobs = profile.get("generation_jobs")
    if not isinstance(profile_jobs, list):
        raise BenchmarkConfigError("Dimensional-transfer profile has no generation jobs")
    condition_by_job = {str(job["id"]): str(job["condition_id"]) for job in profile_jobs}
    if (
        len(condition_by_job) != len(expected_jobs)
        or {str(job["condition_id"]): str(job["simulator_runs"]) for job in profile_jobs}
        != expected_jobs
    ):
        raise BenchmarkConfigError("Dimensional-transfer runtime profile differs from its panel")
    expected_seeds_by_job: dict[str, set[int]] = {}
    for job in profile_jobs:
        runs = load_json_object(
            frozen_root / str(job["simulator_runs"]),
            f"{job['id']} dimensional simulator runs",
        )["runs"]
        expected_seeds_by_job[str(job["id"])] = {
            int(run["base_seed"]) + offset
            for run in runs
            for offset in range(int(run["replicates"]))
        }
    dimension_by_condition = {
        str(item["id"]): (
            int(item["genes"]),
            int(item["transcription_factors"]),
            int(item["cells"]),
        )
        for item in active_conditions
    }
    expected_contexts = {
        str(item["id"]): int(item["expected_group_contexts"]) for item in active_conditions
    }
    analysis_role_by_condition = {
        str(item["id"]): str(item["analysis_role"]) for item in active_conditions
    }
    dataset_condition: dict[str, str] = {}
    generation_seeds: dict[str, int] = {}
    dataset_state_by_id: dict[str, Mapping[str, Any]] = {}
    observed_seeds_by_job: dict[str, set[int]] = defaultdict(set)
    for raw in datasets:
        if not isinstance(raw, Mapping):
            raise BenchmarkConfigError("Dimensional-transfer state contains an invalid dataset")
        generation_job = str(raw.get("generation_job"))
        condition = condition_by_job.get(generation_job)
        dataset_id = raw.get("dataset_id")
        if (
            not isinstance(dataset_id, str)
            or not condition
            or raw.get("dataset_role") != "dimensional_transfer"
            or raw.get("simulator_id") != "scmultisim"
        ):
            raise BenchmarkConfigError("Dimensional-transfer dataset provenance is incomplete")
        if dataset_id in dataset_condition:
            raise BenchmarkConfigError("Dimensional-transfer state repeats a dataset")
        generation_seed = raw.get("seed")
        if not isinstance(generation_seed, int) or isinstance(generation_seed, bool):
            raise BenchmarkConfigError("Dimensional-transfer generation seed is missing")
        dataset_condition[dataset_id] = condition
        generation_seeds[dataset_id] = generation_seed
        dataset_state_by_id[dataset_id] = raw
        observed_seeds_by_job[generation_job].add(generation_seed)
    if dict(observed_seeds_by_job) != expected_seeds_by_job:
        raise BenchmarkConfigError("Dimensional-transfer state uses undeclared generation seeds")
    counts = defaultdict(int)
    for condition in dataset_condition.values():
        counts[condition] += 1
    if dict(counts) != expected_counts:
        raise BenchmarkConfigError("Dimensional-transfer dataset panel is incomplete")

    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    directed_all = [
        row
        for row in metric_rows
        if row.get("level") == "directed" and row.get("dataset_id") in dataset_condition
    ]
    if not directed_all or any(
        row.get("status") != "ok"
        or row.get("dataset_role") != "dimensional_transfer"
        or row.get("regime") != "group_only"
        or row.get("family") != "spathi"
        or row.get("configuration_id") != production_id
        or row.get("configuration_sha256")
        != production_configuration_sha256
        for row in directed_all
    ):
        raise BenchmarkConfigError("Dimensional-transfer contains missing or invalid metrics")
    directed = [row for row in directed_all if row["configuration_id"] == production_id]
    algorithm_seed, seed_derivation = transfer_algorithm_seed(frozen_root, budget=budget)
    if {int(row["algorithm_seed"]) for row in directed_all} != {algorithm_seed}:
        raise BenchmarkConfigError("Dimensional-transfer metrics use an undeclared algorithm seed")

    per_dataset_contexts: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    observed_contexts: dict[str, set[str]] = defaultdict(set)
    for row in directed:
        dataset_id = str(row["dataset_id"])
        context = str(row["context"])
        if not context.startswith("group:") or context in observed_contexts[dataset_id]:
            raise BenchmarkConfigError("Dimensional-transfer contexts are missing or duplicated")
        observed_contexts[dataset_id].add(context)
        n_candidates = int(row["n_candidates"])
        n_truth_edges = int(row["n_truth_edges"])
        if n_candidates < 1 or not 0 < n_truth_edges < n_candidates:
            raise BenchmarkConfigError("Dimensional-transfer candidate universe is invalid")
        prevalence = n_truth_edges / n_candidates
        aupr = _finite(row["aupr"], f"{dataset_id}/{context}/aupr")
        adjusted = (aupr - prevalence) / (1.0 - prevalence)
        auroc = None if not row.get("auroc") else _finite(row["auroc"], "auroc")
        per_dataset_contexts[dataset_id].append(
            {
                "aupr": aupr,
                "adjusted_aupr": adjusted,
                "auroc": auroc,
                "f1": _finite(row["f1_at_truth_count"], "f1_at_truth_count"),
                "epr": _finite(row["epr_at_truth_count"], "epr_at_truth_count"),
                "prevalence": prevalence,
                "wall_time": _finite(row["wall_time_seconds"], "wall_time_seconds"),
            }
        )
    if set(per_dataset_contexts) != set(dataset_condition) or any(
        len(rows) != expected_contexts[dataset_condition[dataset_id]]
        for dataset_id, rows in per_dataset_contexts.items()
    ):
        raise BenchmarkConfigError(
            "Every dimensional-transfer dataset must expose its declared group contexts"
        )

    per_condition: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    dataset_results = []
    for dataset_id, rows in per_dataset_contexts.items():
        condition_id = dataset_condition[dataset_id]
        reduced = {
            "aupr": fmean(float(row["aupr"]) for row in rows),
            "adjusted_aupr": fmean(float(row["adjusted_aupr"]) for row in rows),
            "auroc": _mean_or_none([row["auroc"] for row in rows]),
            "f1": fmean(float(row["f1"]) for row in rows),
            "epr": fmean(float(row["epr"]) for row in rows),
            "prevalence": fmean(float(row["prevalence"]) for row in rows),
            "wall_time": float(rows[0]["wall_time"]),
        }
        per_condition[condition_id].append(reduced)
        genes, tfs, cells = dimension_by_condition[condition_id]
        dataset_results.append(
            {
                "dataset_id": dataset_id,
                "condition_id": condition_id,
                "analysis_role": analysis_role_by_condition[condition_id],
                "genes": genes,
                "transcription_factors": tfs,
                "cells": cells,
                "generation_seed": generation_seeds[dataset_id],
                "contexts": len(rows),
                "directed_aupr": reduced["aupr"],
                "adjusted_aupr": reduced["adjusted_aupr"],
                "directed_auroc": reduced["auroc"],
                "f1_at_truth_count": reduced["f1"],
                "epr_at_truth_count": reduced["epr"],
                "candidate_prevalence": reduced["prevalence"],
                "wall_time_seconds": reduced["wall_time"],
            }
        )

    conditions = []
    for condition_id in active_condition_ids:
        rows = per_condition[condition_id]
        genes, tfs, cells = dimension_by_condition[condition_id]
        conditions.append(
            {
                "id": condition_id,
                "analysis_role": analysis_role_by_condition[condition_id],
                "genes": genes,
                "transcription_factors": tfs,
                "cells": cells,
                "replicates": len(rows),
                "contexts": len(rows) * expected_contexts[condition_id],
                "directed_aupr": fmean(float(row["aupr"]) for row in rows),
                "adjusted_aupr": fmean(float(row["adjusted_aupr"]) for row in rows),
                "directed_auroc": _mean_or_none([row["auroc"] for row in rows]),
                "f1_at_truth_count": fmean(float(row["f1"]) for row in rows),
                "epr_at_truth_count": fmean(float(row["epr"]) for row in rows),
                "candidate_prevalence": fmean(float(row["prevalence"]) for row in rows),
                "median_wall_time_seconds": median(float(row["wall_time"]) for row in rows),
            }
        )

    contrast_specs = (
        (
            ("gene_scale_contrast", "g3000_tf100_c300", "g15000_tf100_c300"),
            ("tf_scale_contrast", "g3000_tf100_c300", "g3000_tf1500_c300"),
            ("cell_scale_contrast", "g100_tf10_c300", "g100_tf10_c10000"),
        )
        if budget == "full"
        else ()
    )
    contrasts = []
    for contrast_id, left_id, right_id in contrast_specs:
        left = [float(row["adjusted_aupr"]) for row in per_condition[left_id]]
        right = [float(row["adjusted_aupr"]) for row in per_condition[right_id]]
        estimate = fmean(right) - fmean(left)
        contrasts.append(
            {
                "id": contrast_id,
                "estimand": "difference_in_mean_adjusted_aupr",
                "estimate": estimate,
            }
        )
    payload = {
        "schema_version": "1.0",
        "artifact_type": "spathi_dimensional_transfer_results",
        "budget": budget,
        "evidence_role": "publication_descriptive" if budget == "full" else "operational",
        "claim_scope": "truth_bearing_synthetic_dimensional_transfer",
        "status": "complete",
        "panel": _reference(root, panel_path),
        "profile": _reference(root, profile_path),
        "selected_config": {
            **_reference(root, selected_config_path),
            "policy_id": production["selected_policy"]["policy_id"],
            "policy_sha256": production["selected_policy"]["policy_sha256"],
        },
        "algorithm_seed": {"value": algorithm_seed, "derivation": seed_derivation},
        "summary_inputs": {
            "metrics_csv": _reference(root, metrics_path),
            "summary_json": _reference(root, summary_path),
            "pipeline_state": _reference(root, run_root / "state.json"),
        },
        "summary": {
            "datasets_requested": expected_datasets,
            "datasets_completed": len(per_dataset_contexts),
            "contexts_evaluated": sum(len(rows) for rows in per_dataset_contexts.values()),
            "all_completed": True,
        },
        "datasets": sorted(dataset_results, key=lambda item: str(item["dataset_id"])),
        "conditions": conditions,
        "contrasts": contrasts,
        "limitations": [
            "The panel is generated by scMultiSim from a fully public synthetic design and makes no biological-realism claim.",
            "The full budget expands genes, candidate TFs, and cells only in separate one-axis pairs; their joint interaction is not estimated. The reduced budget contains no scale endpoint and reports no scale contrast.",
            "The full-budget gene and TF contrasts share one anchor, while the cell contrast uses a smaller regulatory core; every result is interpreted only within its declared pair.",
            "The TF scale contrast also changes the regulatory topology and which TFs regulate targets, so it does not causally identify a pure effect of candidate-TF count.",
            "One frozen SPATHI policy is measured, so the phase neither ranks tools nor selects parameters.",
            "The fixed inference seed makes dataset generation, not algorithmic repetition, the independent unit.",
            "The full budget repeats the bounded anchor three times but schedules one prospectively admitted dataset at each expensive scale endpoint; the reduced budget omits all three endpoints.",
            "The single-group and declared 10/40/250 group-size conditions are routing controls; they are reported separately and do not define dimensional contrasts.",
            "No confidence interval is reported for a contrast containing a single endpoint dataset; no non-inferiority margin or winner rule is declared.",
        ],
    }
    validate_json(
        payload,
        root / "schemas/dimensional-transfer-results.schema.json",
        "dimensional-transfer results",
    )
    return write_idempotent_json(
        output_path,
        payload,
        "dimensional-transfer results",
        boundary=root,
    )


__all__ = [
    "analyze_dimensional_transfer",
    "condition_ids_for_budget",
    "topology_rows",
    "topology_text",
    "transfer_algorithm_seed",
    "validate_transfer_panel",
]
