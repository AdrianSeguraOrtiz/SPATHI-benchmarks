"""Content-based exclusivity contract for benchmark dataset partitions.

Development, validation, and test data must be independent. Paths and human
labels are provenance only: exclusivity is computed for each effective
deterministic simulation task after combining scenario content, simulator
parameters, and the seed ANDREA will actually use. Request fragments are never
exclusive in isolation, so a stratified design may reuse simulator parameters
with disjoint scenario seeds. External inputs are additionally identified by
content so datasets derived from the same topology remain in one partition.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .config import BenchmarkConfigError, load_json_object, resolve_repo_path

IDENTITY_CONTRACT_VERSION = "1.0"
PROTECTED_DATASET_ROLES = frozenset(
    {
        "development",
        "validation",
        "efficiency_validation",
        "efficiency_safety",
        "test",
        "dimensional_transfer",
    }
)
_MAX_SEED_32BIT = 2_147_483_646
_TRUTH_CONTEXT_ORDER = ("global", "group", "column")


def _require_seed(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise BenchmarkConfigError(f"{label} must be an integer >= 1")
    return value


def _replicate_count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise BenchmarkConfigError(f"{label} must be an integer >= 1")
    return value


def _stable_seed_base(*, request_id: str, semantic_key: str, simulator_id: str) -> int:
    digest = hashlib.sha256(f"{request_id}|{semantic_key}|{simulator_id}".encode()).hexdigest()
    return (int(digest[:8], 16) % _MAX_SEED_32BIT) + 1


def _semantic_key(scenario: dict[str, Any], label: str) -> str:
    axes = scenario.get("data_axes")
    truth = scenario.get("truth_requirements")
    if not isinstance(axes, dict) or not isinstance(truth, dict):
        raise BenchmarkConfigError(f"{label} must define data_axes and truth_requirements")
    axis_names = ("measurement", "resolution", "column_kind", "experimental_design")
    axis_values = [axes.get(name) for name in axis_names]
    if not all(isinstance(value, str) and value for value in axis_values):
        raise BenchmarkConfigError(f"{label}.data_axes is incomplete")
    raw_contexts = truth.get("contexts")
    if not isinstance(raw_contexts, list) or not all(
        isinstance(context, str) and context for context in raw_contexts
    ):
        raise BenchmarkConfigError(f"{label}.truth_requirements.contexts is invalid")
    unknown = sorted(set(raw_contexts) - set(_TRUTH_CONTEXT_ORDER))
    if unknown:
        raise BenchmarkConfigError(f"{label} contains unknown truth contexts: {unknown}")
    contexts = [context for context in _TRUTH_CONTEXT_ORDER if context in raw_contexts]
    return "|".join([*axis_values, ",".join(contexts)])


def _tree_content_identity(path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        return {"kind": "file", "sha256": sha256_file(resolved)}
    if resolved.is_dir():
        members: dict[str, str] = {}
        for member in sorted(resolved.rglob("*")):
            if member.is_symlink():
                raise BenchmarkConfigError(f"{label} contains a symlink: {member}")
            if member.is_file():
                members[member.relative_to(resolved).as_posix()] = sha256_file(member)
        return {"kind": "directory", "members": members}
    raise BenchmarkConfigError(f"{label} does not exist: {resolved}")


def _normalized_inputs(
    scenario_path: Path,
    scenario: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    raw_inputs = scenario.get("inputs", {})
    if not isinstance(raw_inputs, dict):
        raise BenchmarkConfigError(f"{label}.inputs must be an object")
    normalized: dict[str, Any] = {}
    for input_id, raw_input in sorted(raw_inputs.items()):
        if not isinstance(input_id, str) or not input_id or not isinstance(raw_input, dict):
            raise BenchmarkConfigError(f"{label}.inputs contains an invalid entry")
        raw_path = raw_input.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise BenchmarkConfigError(f"{label}.inputs.{input_id}.path is invalid")
        input_path = Path(raw_path).expanduser()
        if not input_path.is_absolute():
            input_path = scenario_path.parent / input_path
        metadata = {
            key: value for key, value in raw_input.items() if key not in {"path", "description"}
        }
        normalized[input_id] = {
            **metadata,
            "content": _tree_content_identity(
                input_path,
                f"{label}.inputs.{input_id}.path",
            ),
        }
    return normalized


def _scenario_contract(
    scenario_path: Path,
    scenario: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], int | None, str, str]:
    request_id = scenario.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise BenchmarkConfigError(f"{label}.id must be a non-empty string")
    semantic_key = _semantic_key(scenario, label)
    raw_seed = scenario.get("base_seed")
    base_seed = None if raw_seed is None else _require_seed(raw_seed, f"{label}.base_seed")
    requested_extras = scenario.get("requested_extras", [])
    if not isinstance(requested_extras, list) or not all(
        isinstance(item, str) and item for item in requested_extras
    ):
        raise BenchmarkConfigError(f"{label}.requested_extras must be an array of strings")
    truth = scenario["truth_requirements"]
    normalized_contexts = [
        context for context in _TRUTH_CONTEXT_ORDER if context in truth["contexts"]
    ]
    contract = {
        "data_axes": scenario["data_axes"],
        "truth_requirements": {"contexts": normalized_contexts},
        "organism": scenario.get("organism"),
        "requested_extras": sorted(requested_extras),
        "inputs": _normalized_inputs(scenario_path, scenario, label),
        "effective_base_seed": base_seed,
    }
    return contract, base_seed, request_id, semantic_key


def _normalized_run(raw_run: dict[str, Any], label: str) -> dict[str, Any]:
    simulator_id = raw_run.get("simulator_id")
    if not isinstance(simulator_id, str) or not simulator_id:
        raise BenchmarkConfigError(f"{label}.simulator_id must be a non-empty string")
    params = raw_run.get("params")
    if not isinstance(params, dict):
        raise BenchmarkConfigError(f"{label}.params must be an object")
    native_outputs = raw_run.get("native_outputs", [])
    if not isinstance(native_outputs, list) or not all(
        isinstance(item, str) and item for item in native_outputs
    ):
        raise BenchmarkConfigError(f"{label}.native_outputs must be an array of strings")
    base_seed = raw_run.get("base_seed")
    if base_seed is not None:
        base_seed = _require_seed(base_seed, f"{label}.base_seed")
    return {
        "simulator_id": simulator_id,
        "params": params,
        "replicates": _replicate_count(raw_run.get("replicates"), f"{label}.replicates"),
        "native_outputs": sorted(native_outputs),
        "base_seed": base_seed,
    }


def _runs_contract(
    runs_payload: dict[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    raw_runs = runs_payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise BenchmarkConfigError(f"{label}.runs must be a non-empty array")
    ordered: list[dict[str, Any]] = []
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise BenchmarkConfigError(f"{label}.runs[{index}] must be an object")
        run_id = raw_run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise BenchmarkConfigError(f"{label}.runs[{index}].run_id must be non-empty")
        run = _normalized_run(raw_run, f"{label}.runs[{index}]")
        ordered.append({"run_id": run_id, **run})
    return ordered


def _record(
    *,
    identity_type: str,
    identity_payload: dict[str, Any],
    profile_id: str,
    job: dict[str, Any],
    source: dict[str, str],
    input_id: str | None = None,
    run_id: str | None = None,
    replicate_index: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "identity_type": identity_type,
        "identity_sha256": canonical_json_sha256(
            {
                "contract_version": IDENTITY_CONTRACT_VERSION,
                "identity_type": identity_type,
                "payload": identity_payload,
            }
        ),
        "dataset_role": job["dataset_role"],
        "profile_id": profile_id,
        "generation_job_id": job["id"],
        "source": source,
    }
    if run_id is not None:
        record["run_id"] = run_id
    if input_id is not None:
        record["input_id"] = input_id
    if replicate_index is not None:
        record["replicate_index"] = replicate_index
    if seed is not None:
        record["seed"] = seed
    return record


def build_dataset_partition_contract(
    repo_root: Path,
    profiles: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build the auditable content identities used to detect partition leakage."""

    records: list[dict[str, Any]] = []
    for profile in sorted(profiles, key=lambda item: str(item.get("id", ""))):
        profile_id = str(profile.get("id", ""))
        for job in profile.get("generation_jobs", []):
            role = str(job.get("dataset_role", ""))
            if role not in PROTECTED_DATASET_ROLES:
                continue
            job_id = str(job.get("id", ""))
            scenario_path = resolve_repo_path(
                repo_root,
                str(job.get("scenario", "")),
                f"{profile_id}:{job_id} partition scenario",
            )
            runs_path = resolve_repo_path(
                repo_root,
                str(job.get("simulator_runs", "")),
                f"{profile_id}:{job_id} partition simulator runs",
            )
            scenario = load_json_object(scenario_path, f"{profile_id}:{job_id} scenario")
            runs_payload = load_json_object(runs_path, f"{profile_id}:{job_id} simulator runs")
            (
                scenario_payload,
                scenario_base_seed,
                scenario_request_id,
                scenario_semantic_key,
            ) = _scenario_contract(
                scenario_path,
                scenario,
                f"{profile_id}:{job_id} scenario",
            )
            ordered_runs = _runs_contract(
                runs_payload,
                f"{profile_id}:{job_id} simulator runs",
            )
            source = {
                "scenario": scenario_path.relative_to(repo_root.resolve()).as_posix(),
                "simulator_runs": runs_path.relative_to(repo_root.resolve()).as_posix(),
            }
            for input_id, input_contract in scenario_payload["inputs"].items():
                records.append(
                    _record(
                        identity_type="related_input",
                        identity_payload={"content": input_contract["content"]},
                        profile_id=profile_id,
                        job=job,
                        source=source,
                        input_id=input_id,
                    )
                )
            seed_offset = 0
            scenario_for_task = {
                key: value
                for key, value in scenario_payload.items()
                if key != "effective_base_seed"
            }
            for run in ordered_runs:
                replicates = int(run["replicates"])
                if run["base_seed"] is None:
                    if scenario_base_seed is None:
                        base_seed = _stable_seed_base(
                            request_id=scenario_request_id,
                            semantic_key=scenario_semantic_key,
                            simulator_id=f"{run['run_id']}|{run['simulator_id']}",
                        )
                    else:
                        base_seed = ((scenario_base_seed - 1 + seed_offset) % _MAX_SEED_32BIT) + 1
                else:
                    base_seed = int(run["base_seed"])
                run_for_task = {
                    key: value
                    for key, value in run.items()
                    if key not in {"run_id", "replicates", "base_seed"}
                }
                for replicate_index in range(1, replicates + 1):
                    seed = ((base_seed - 1 + replicate_index - 1) % _MAX_SEED_32BIT) + 1
                    records.append(
                        _record(
                            identity_type="generation_task",
                            identity_payload={
                                "scenario": scenario_for_task,
                                "simulator_run": run_for_task,
                                "seed": seed,
                            },
                            profile_id=profile_id,
                            job=job,
                            source=source,
                            run_id=str(run["run_id"]),
                            replicate_index=replicate_index,
                            seed=seed,
                        )
                    )
                seed_offset += replicates
    records.sort(
        key=lambda item: (
            item["identity_type"],
            item["identity_sha256"],
            item["dataset_role"],
            item["profile_id"],
            item["generation_job_id"],
            item.get("input_id", ""),
            item.get("run_id", ""),
            item.get("replicate_index", 0),
        )
    )
    return {
        "schema_version": "1.0",
        "identity_contract_version": IDENTITY_CONTRACT_VERSION,
        "protected_roles": sorted(PROTECTED_DATASET_ROLES),
        "identity_types": ["generation_task", "related_input"],
        "records": records,
    }


def validate_dataset_partition_exclusivity(
    repo_root: Path,
    profiles: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Reject shared effective tasks or related external inputs across protected roles."""

    contract = build_dataset_partition_contract(repo_root, profiles)
    by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in contract["records"]:
        by_identity[(record["identity_type"], record["identity_sha256"])].append(record)
    conflicts = []
    for (identity_type, digest), records in sorted(by_identity.items()):
        roles = sorted({record["dataset_role"] for record in records})
        if len(roles) < 2:
            continue
        locations = sorted(
            {
                f"{record['profile_id']}:{record['generation_job_id']}"
                + (f":input-{record['input_id']}" if "input_id" in record else "")
                + (f":{record['run_id']}" if "run_id" in record else "")
                + (f":replicate-{record['replicate_index']}" if "replicate_index" in record else "")
                for record in records
            }
        )
        conflicts.append(
            f"{identity_type} sha256:{digest} is shared by roles {roles} at {locations}"
        )
    if conflicts:
        raise BenchmarkConfigError(
            "Dataset partition leakage across development/validation/test: " + "; ".join(conflicts)
        )
    return contract
