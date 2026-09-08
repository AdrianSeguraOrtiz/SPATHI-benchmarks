from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from spathi_bench.artifacts import canonical_json_sha256
from spathi_bench.reproducibility import (
    ReproducibilityAuditError,
    audit_reproducibility,
    write_reproducibility_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _copy_design(destination: Path) -> Path:
    root = destination / "benchmark"
    shutil.copytree(REPO_ROOT / "experiments", root / "experiments")
    shutil.copytree(REPO_ROOT / "schemas", root / "schemas")
    return root


def test_committed_design_passes_a_stable_schema_validated_audit() -> None:
    result = audit_reproducibility(REPO_ROOT)
    schema = _read(REPO_ROOT / "schemas/reproducibility-audit.schema.json")
    Draft202012Validator(schema).validate(result)

    unsigned = deepcopy(result)
    digest = unsigned.pop("audit_sha256")
    assert digest == canonical_json_sha256(unsigned)
    assert result["status"] == "passed"
    assert result["coverage"]["simulator_replicates"] > 0
    assert result["coverage"]["stochastic_inference_runs"] > 0
    assert result["coverage"]["scale_stress_generator_datasets"] > 0
    assert result["coverage"] == {
        "simulator_manifests": 97,
        "simulator_runs": 97,
        "simulator_replicates": 223,
        "inference_metadata_documents": 7,
        "stochastic_inference_runs": 44,
        "deterministic_inference_runs": 6,
        "scale_stress_generator_requests": 2,
        "scale_stress_generator_datasets": 9,
        "scale_stress_component_seeds": 36,
        "calibration_algorithm_seeds": 34,
        "statistical_contract_documents": 1,
        "statistical_root_seeds": 1,
        "comparison_algorithm_seeds": 7,
        "efficiency_algorithm_seeds": 7,
        "efficiency_statistical_root_seeds": 1,
        "efficiency_execution_order_seeds": 2,
        "efficiency_design_sensitivity_contracts": 1,
        "efficiency_safety_generation_seeds": 18,
        "efficiency_safety_algorithm_seeds": 1,
        "efficiency_safety_execution_order_seeds": 1,
        "scalability_algorithm_seeds": 9,
        "dimensional_transfer_algorithm_seeds": 2,
    }

    serialized = json.dumps(result, sort_keys=True)
    assert str(REPO_ROOT) not in serialized
    assert "timestamp" not in serialized.lower()


def test_audit_and_persisted_bytes_are_checkout_location_independent(tmp_path: Path) -> None:
    first_root = _copy_design(tmp_path / "first")
    second_root = _copy_design(tmp_path / "second")
    first_output = tmp_path / "first-audit.json"
    second_output = tmp_path / "second-audit.json"

    first = write_reproducibility_audit(first_root, first_output)
    second = write_reproducibility_audit(second_root, second_output)

    assert first == second
    assert first_output.read_bytes() == second_output.read_bytes()


def test_audit_rejects_a_simulator_seed_outside_its_registered_block(tmp_path: Path) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/simulations/dyngen/smoke-runs.json"
    payload = _read(path)
    payload["runs"][0]["base_seed"] += 1
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="seed/span"):
        audit_reproducibility(root)


def test_audit_rejects_undeclared_or_mismatched_inference_derivations(
    tmp_path: Path,
) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/inference/group_only/smoke-baselines-metadata.json"
    payload = _read(path)
    payload["runs"][0].pop("seed_derivation")
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="must contain exactly"):
        audit_reproducibility(root)

    payload = _read(REPO_ROOT / "experiments/v1/inference/group_only/smoke-baselines-metadata.json")
    payload["runs"][0]["algorithm_seed"] += 1
    _write(path, payload)
    request_path = root / "experiments/v1/inference/group_only/smoke-baselines.json"
    request = _read(request_path)
    request["runs"][0]["params"]["seed"] = payload["runs"][0]["algorithm_seed"]
    _write(request_path, request)

    with pytest.raises(ReproducibilityAuditError, match="resolves to"):
        audit_reproducibility(root)


def test_audit_rejects_metadata_and_request_seed_disagreement(tmp_path: Path) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/inference/group_only/smoke-baselines.json"
    payload = _read(path)
    payload["runs"][0]["params"]["seed"] += 1
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="does not expose metadata seed"):
        audit_reproducibility(root)


def test_audit_rejects_a_mismatched_scale_stress_generator_seed(tmp_path: Path) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/simulations/scale_stress/scale-reduced.json"
    payload = _read(path)
    payload["datasets"][0]["seed"] += 1
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="resolves to"):
        audit_reproducibility(root)


def test_audit_rejects_a_noncanonical_seed_plan(tmp_path: Path) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/seeds/seed-plan.json"
    payload = _read(path)
    payload["allocations"][0]["seed"] += 1
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="Invalid canonical seed plan"):
        audit_reproducibility(root)


def test_audit_rejects_a_mismatched_efficiency_seed(tmp_path: Path) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/spathi/efficiency-study.json"
    payload = _read(path)
    payload["budgets"]["reduced"]["algorithm_seeds"][0] += 1
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="algorithm seed 1 derives"):
        audit_reproducibility(root)


def test_audit_rejects_a_mismatched_calibration_seed(tmp_path: Path) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/spathi/search-space.json"
    payload = _read(path)
    payload["calibration_budgets"]["full"]["algorithmic_seeds"]["geometry"][0] += 1
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="Calibration full.geometry seed 1"):
        audit_reproducibility(root)


def test_audit_rejects_a_mismatched_statistical_root(tmp_path: Path) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/inference/plans/full-v1.json"
    payload = _read(path)
    payload["statistics"]["random_seed"] += 1
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="resolves to"):
        audit_reproducibility(root)


def test_audit_rejects_a_mismatched_efficiency_execution_order_seed(
    tmp_path: Path,
) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/spathi/efficiency-study.json"
    payload = _read(path)
    payload["budgets"]["full"]["execution_order_seed"] += 1
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="execution-order seed derives"):
        audit_reproducibility(root)


def test_audit_rejects_scale_stress_generation_stream_masquerading_as_augmentation(
    tmp_path: Path,
) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/simulations/scale_stress/scale-reduced.json"
    payload = _read(path)
    payload["datasets"][0]["seed_derivation"]["stream"] = "generation"
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="must be 'augmentation'"):
        audit_reproducibility(root)


def test_audit_rejects_an_inference_request_without_seed_metadata(tmp_path: Path) -> None:
    root = _copy_design(tmp_path)
    path = root / "experiments/v1/inference/group_only/smoke-baselines.json"
    payload = _read(path)
    payload["runs"].append(
        {
            "run_id": "orphan_stochastic_run",
            "tool_id": "genie3",
            "params": {"seed": 1},
            "execution": {"mode": "group_emulated"},
        }
    )
    _write(path, payload)

    with pytest.raises(ReproducibilityAuditError, match="lack seed metadata"):
        audit_reproducibility(root)
