"""Small physical qualification of SPATHI sidecars and ANDREA evaluation.

This is a scientific contract test using real local executables. It deliberately
does not claim to qualify Docker simulators, competitors or all study phases.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .config import BenchmarkConfigError, load_json_object
from .safe_io import atomic_write_text, path_below_no_symlinks
from .spathi_diagnostics import read_model_accounting

_GENES = ("TF1", "TF2", "G1", "G2", "G3", "G4", "Rare", "Constant")
_SOURCES = ("TF1", "TF2")
_CONTEXTS = ("group:A", "group:B")
_SEED = 738291
_COVERED = (
    "Real SPATHI CLI: fixed trees and explicit target lists",
    "Complete model sidecars: task seeds, numerical skips, tree counts and metadata counters",
    "Deterministic networks with identical seeds and different thread budgets",
    "Real ANDREA evaluation of external grouped networks and completed empty networks",
    "AUPR and AUROC compared with an independent dense oracle, including score ties",
    "Truth edges outside the TF candidate space are excluded and counted",
    "Real SPATHI and ANDREA HTML report generation",
)
_NOT_COVERED = (
    "Docker image execution and simulator-specific contracts",
    "The five complete study workflows or interruption of their expensive tasks",
    "Competitor wrappers, scalability, performance measurements or runtime forecasts",
    "Scientific evidence for a configuration choice or published tool ranking",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        "qualification JSON",
    )


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _checkout_identity(path: Path) -> dict[str, str]:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise BenchmarkConfigError(f"Qualification requires a source checkout: {path}")
    diff = subprocess.run(
        ["git", "-C", str(path), "diff", "HEAD", "--binary"], capture_output=True, check=True
    )
    import hashlib

    return {
        "commit": result.stdout.strip(),
        "worktree_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
    }


def _interpreter(repo: Path) -> Path:
    candidate = repo / ".venv/bin/python"
    return candidate if candidate.is_file() else Path(sys.executable)


def _software_identity(repo_root: Path, andrea_repo: Path, spathi_repo: Path) -> dict[str, Any]:
    def runtime(repo: Path) -> dict[str, Any]:
        code = (
            "import importlib.metadata as m,json,sys;"
            "print(json.dumps({'python':sys.version,'packages':"
            "{d.metadata['Name']:d.version for d in m.distributions()}}))"
        )
        observed = subprocess.run(
            [str(_interpreter(repo)), "-c", code], capture_output=True, text=True, check=False
        )
        if observed.returncode:
            raise BenchmarkConfigError(f"Cannot inspect qualification Python environment: {repo}")
        return json.loads(observed.stdout)

    return {
        "benchmark": _checkout_identity(repo_root),
        "andrea": _checkout_identity(andrea_repo),
        "spathi": _checkout_identity(spathi_repo),
        "environments": {
            "benchmark": runtime(repo_root),
            "andrea": runtime(andrea_repo),
            "spathi": runtime(spathi_repo),
        },
        "qualification_sha256": sha256_file(Path(__file__)),
        "diagnostics_sha256": sha256_file(Path(__file__).with_name("spathi_diagnostics.py")),
    }


def _completed_receipt(
    root: Path, *, software: Mapping[str, Any], resources: Mapping[str, Any]
) -> dict[str, Any] | None:
    receipt_path = root / "receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = load_json_object(receipt_path, "qualification receipt")
    if receipt.get("status") != "complete":
        return None
    if receipt.get("run_id") != root.name or receipt.get("receipt_path") != str(receipt_path):
        raise BenchmarkConfigError("Completed qualification has an inconsistent identity")
    attempt_value = receipt.get("attempt")
    files = receipt.get("files")
    if not isinstance(attempt_value, str) or not isinstance(files, dict) or not files:
        raise BenchmarkConfigError("Completed qualification has no output fingerprints")
    attempt = path_below_no_symlinks(
        Path(attempt_value), root, "qualification attempt", require_directory=True
    )
    for relative, digest in files.items():
        path = path_below_no_symlinks(
            attempt / relative, attempt, "qualification output", require_file=True
        )
        if sha256_file(path) != digest:
            raise BenchmarkConfigError(f"Completed qualification output changed: {path}")
    if receipt.get("software") == software and receipt.get("requested_resources") == resources:
        return receipt
    return None


def _run(command: list[str], *, repo: Path, log_path: Path, timeout: int = 180) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(repo / "src"), str(repo)))
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NO_COLOR": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "cwd": str(repo)}) + "\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BenchmarkConfigError(
                f"Qualification command failed; see {log_path}: {exc}"
            ) from exc
    if result.returncode:
        raise BenchmarkConfigError(
            f"Qualification command exited with {result.returncode}; see {log_path}"
        )


def dense_metrics(
    scores: Mapping[tuple[str, str], float],
    *,
    sources: Sequence[str],
    targets: Sequence[str],
    truth: set[tuple[str, str]],
) -> dict[str, float]:
    """Dense independent AP/AUC oracle for tiny qualification networks.

    AP groups tied scores at the same threshold. AUC uses every positive/negative
    pair, awarding half a win to ties. Missing predictions have score zero.
    """
    candidates = [(source, target) for source in sources for target in targets if source != target]
    observations = [(edge in truth, float(scores.get(edge, 0.0))) for edge in candidates]
    positives = [score for positive, score in observations if positive]
    negatives = [score for positive, score in observations if not positive]
    if not positives or not negatives or any(not math.isfinite(score) for _, score in observations):
        raise BenchmarkConfigError("Qualification oracle needs finite scores and both classes")
    thresholds = sorted({score for _, score in observations}, reverse=True)
    ap = 0.0
    for threshold in thresholds:
        new_positives = sum(positive and score == threshold for positive, score in observations)
        retained = [(positive, score) for positive, score in observations if score >= threshold]
        ap += (
            new_positives
            / len(positives)
            * sum(positive for positive, _ in retained)
            / len(retained)
        )
    auc = sum(
        (positive > negative) + 0.5 * (positive == negative)
        for positive in positives
        for negative in negatives
    ) / (len(positives) * len(negatives))
    return {"aupr": ap, "auroc": auc}


def _dataset(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(_SEED)
    tf1 = [rng.randint(1, 20) for _ in range(24)]
    tf2 = [rng.randint(1, 20) for _ in range(24)]
    values = (
        tf1,
        tf2,
        [2 * x + i % 3 for i, x in enumerate(tf1)],
        [3 * x + i % 2 for i, x in enumerate(tf2)],
        [x + y for x, y in zip(tf1, tf2, strict=True)],
        [rng.randint(1, 25) for _ in range(24)],
        [1] + [0] * 23,
        [5] * 24,
    )
    cells = [f"c{i:02d}" for i in range(24)]
    with (root / "expression.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["gene", *cells])
        writer.writerows([gene, *row] for gene, row in zip(_GENES, values, strict=True))
    atomic_write_text(
        root / "groups.tsv",
        "column\tcluster\n"
        + "".join(f"{cell}\t{'A' if i < 12 else 'B'}\n" for i, cell in enumerate(cells)),
        "qualification groups",
    )
    atomic_write_text(root / "tf_list.txt", "\n".join(_SOURCES) + "\n", "qualification TFs")
    atomic_write_text(root / "gene_universe.txt", "\n".join(_GENES) + "\n", "qualification genes")
    truth = {("TF1", "G1"), ("TF2", "G2"), ("TF1", "G3")}
    truth_rows = [
        {
            "source": source,
            "target": target,
            "score": 1,
            "sign": "+",
            "evidence": "simulated_truth",
            "context": context,
        }
        for context in ("global", *_CONTEXTS)
        for source, target in sorted(truth | {("G4", "G1")})
    ]
    _write_csv(
        root / "networks.csv",
        ("source", "target", "score", "sign", "evidence", "context"),
        truth_rows,
    )
    fingerprint = {
        "algorithm": "sha256",
        "value": canonical_json_sha256(
            {
                "expression": sha256_file(root / "expression.tsv"),
                "groups": sha256_file(root / "groups.tsv"),
                "tf_list": sha256_file(root / "tf_list.txt"),
            }
        ),
    }
    manifest = {
        "schema_version": "1.0",
        "dataset_id": "qualification_synthetic_v1",
        "dataset_fingerprint": fingerprint,
        "simulator_id": "qualification_synthetic",
        "data_axes": {
            "measurement": "rna_expression",
            "resolution": "single_cell",
            "column_kind": "cells",
            "experimental_design": "steady_state",
        },
        "truth_requirements": {"contexts": ["global", "group"]},
        "outputs": {"gene_universe": "gene_universe.txt", "networks": "networks.csv"},
        "candidate_space": {
            "sources": "tf_list.txt",
            "targets": "gene_universe.txt",
            "allow_self_edges": False,
        },
    }
    _write_json(root / "ground-truth-manifest.json", manifest)
    return {"root": root, "fingerprint": fingerprint, "truth": truth, "manifest": manifest}


def _infer(
    *,
    spathi_repo: Path,
    dataset: Mapping[str, Any],
    output: Path,
    threads: int,
    targets: Sequence[str] = _GENES,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    inputs = dataset["root"]
    command = [
        str(_interpreter(spathi_repo)),
        "-m",
        "spathi",
        "infer",
        "--expression",
        str(inputs / "expression.tsv"),
        "--groups",
        str(inputs / "groups.tsv"),
        "--tf-list",
        str(inputs / "tf_list.txt"),
        "--output-dir",
        str(output),
        "--random-seed",
        str(_SEED),
        "--threads",
        str(threads),
        "--n-components",
        "3",
        "--distance-metric",
        "euclidean",
        "--group-size-correction",
        "cap-to-target",
        "--n-estimators",
        "30",
        "--no-progress",
        "--report",
    ]
    if tuple(targets) != _GENES:
        target_list = output.parent / f"{output.name}-targets.txt"
        atomic_write_text(target_list, "".join(f"{target}\n" for target in targets), "target list")
        command.extend(("--target-list", str(target_list)))
    _run(command, repo=spathi_repo, log_path=output.parent / f"{output.name}.log")
    metadata = load_json_object(output / "run_metadata.json")
    accounting = read_model_accounting(
        output / "model_diagnostics.tsv.gz",
        contexts=_CONTEXTS,
        targets=targets,
        parameters=metadata["requested_parameters"],
        algorithm_seed=_SEED,
    )
    accounting.validate_metadata(metadata["models"])
    if not (output / "report.html").is_file():
        raise BenchmarkConfigError("Real SPATHI HTML report was not generated")
    with (output / "network.csv").open(encoding="utf-8", newline="") as handle:
        network = list(csv.DictReader(handle))
    if any(row["target"] not in targets for row in network):
        raise BenchmarkConfigError("SPATHI network exceeds the requested target list")
    return network, {
        "models": accounting.models_requested,
        "trees": accounting.trees_fitted,
        "status_counts": accounting.status_counts,
        "threads": threads,
        "dependency_versions": metadata["dependency_versions"],
        "warnings": metadata.get("warnings", []),
    }


def _evaluate(
    *,
    andrea_repo: Path,
    root: Path,
    dataset: Mapping[str, Any],
    networks: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, Any]:
    merged = [{**row, "tool_id": tool} for tool, rows in networks.items() for row in rows]
    _write_csv(
        root / "merged_network_raw.csv",
        ("source", "target", "score", "sign", "evidence", "context", "tool_id"),
        merged,
    )
    tools = list(networks)
    run_report = {
        "run_id": "qualification_external_inference",
        "status": "executed",
        "output_profile": "full",
        "dataset": {
            "id": dataset["manifest"]["dataset_id"],
            "fingerprint": dataset["fingerprint"],
            "column_kind": "cells",
            "expression_profile": "scrna",
            "genes": len(_GENES),
            "columns": 24,
        },
        "inputs": {},
        "outputs": {
            "merged_network_raw": "merged_network_raw.csv",
            "rows_per_tool": {tool: len(rows) for tool, rows in networks.items()},
        },
        "tools": {
            "selected": tools,
            "completed": tools,
            "failed": {},
            "skipped": {},
            "status_by_tool": dict.fromkeys(tools, "completed"),
            "completed_contexts": {tool: list(_CONTEXTS) for tool in tools},
            "catalog_tool_ids": {tool: "custom_" + tool for tool in tools},
            "tool_origins": dict.fromkeys(tools, "custom"),
            "results": {tool: {"execution": {"mode": "group_native"}} for tool in tools},
            "output_capabilities": {
                tool: {
                    "tool_origin": "custom",
                    "catalog_tool_id": "custom_" + tool,
                    "directed": True,
                    "sign": "none",
                }
                for tool in tools
            },
        },
        "execution": {},
        "issues": [],
        "notes": ["External local qualification handoff; no runtime claim."],
    }
    _write_json(root / "run_report.json", run_report)
    code = (
        "import json,sys;from pathlib import Path;"
        "from andrea.core.commands.evaluate_inference import evaluate_inference;"
        "r=evaluate_inference(run_report_path=Path(sys.argv[1]),ground_truth_manifest_path=Path(sys.argv[2]),"
        "output_dir=Path(sys.argv[3]),generate_view=True);"
        "Path(sys.argv[4]).write_text(json.dumps(r,indent=2)+'\\n',encoding='utf-8')"
    )
    _run(
        [
            str(_interpreter(andrea_repo)),
            "-c",
            code,
            str(root / "run_report.json"),
            str(dataset["root"] / "ground-truth-manifest.json"),
            str(root / "evaluation"),
            str(root / "evaluation.json"),
        ],
        repo=andrea_repo,
        log_path=root / "evaluation.log",
    )
    report = load_json_object(root / "evaluation.json")
    directed = [row for row in report["metrics"] if row["level"] == "directed"]
    if len(directed) != len(networks) * len(_CONTEXTS):
        raise BenchmarkConfigError("Real ANDREA evaluation omitted completed network contexts")
    for row in directed:
        scores = {
            (edge["source"], edge["target"]): float(edge["score"])
            for edge in networks[row["tool_id"]]
            if edge["context"] == row["context"]
        }
        expected = dense_metrics(scores, sources=_SOURCES, targets=_GENES, truth=dataset["truth"])
        if (
            row["status"] != "ok"
            or row["n_truth_rows_outside_candidate_space"] != 1
            or any(
                not math.isclose(row[metric], value, abs_tol=1e-12, rel_tol=1e-12)
                for metric, value in expected.items()
            )
        ):
            raise BenchmarkConfigError(f"Real ANDREA metrics differ from dense oracle: {row}")
    view = root / "evaluation" / report["outputs"]["evaluation_view"]
    if not view.is_file():
        raise BenchmarkConfigError("Real ANDREA evaluation HTML report was not generated")
    return {"directed_metrics": directed, "report": str(view)}


def run_qualification(
    *,
    repo_root: Path,
    andrea_repo: Path,
    spathi_repo: Path,
    work_root: Path,
    run_id: str | None = None,
    **resources: Any,
) -> dict[str, Any]:
    """Run the inexpensive real qualification and persist a complete/failed receipt."""
    identity = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identity):
        raise BenchmarkConfigError("Invalid qualification run ID")
    qualification_root = path_below_no_symlinks(
        work_root / "qualification" / identity, work_root, "qualification root"
    )
    software = _software_identity(repo_root, andrea_repo, spathi_repo)
    completed = _completed_receipt(qualification_root, software=software, resources=resources)
    if completed is not None:
        return completed
    attempt = qualification_root / ("attempt-" + uuid.uuid4().hex[:12])
    attempt.mkdir(parents=True)
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": identity,
        "status": "running",
        "scope": "local_physical_scientific_contracts",
        "covers": list(_COVERED),
        "does_not_cover": list(_NOT_COVERED),
        "seed": _SEED,
        "requested_resources": resources,
        "attempt": str(attempt),
        "receipt_path": str(qualification_root / "receipt.json"),
        "software": software,
    }
    _write_json(qualification_root / "receipt.json", receipt)
    try:
        dataset = _dataset(attempt / "dataset")
        receipt["dataset"] = {
            "generator": "qualification_synthetic_v1",
            "genes": len(_GENES),
            "cells": 24,
            "groups": 2,
            "tfs": len(_SOURCES),
            "fingerprint": dataset["fingerprint"],
        }
        threads = min(2, max(1, int(resources.get("threads") or resources.get("cpu_threads") or 2)))
        fixed, fixed_counts = _infer(
            spathi_repo=spathi_repo,
            dataset=dataset,
            output=attempt / "fixed",
            threads=1,
        )
        repeated, repeated_counts = _infer(
            spathi_repo=spathi_repo,
            dataset=dataset,
            output=attempt / "fixed-repeat",
            threads=threads,
        )
        if fixed != repeated:
            raise BenchmarkConfigError(
                "SPATHI networks differ between identical-seed qualification runs"
            )
        subset, subset_counts = _infer(
            spathi_repo=spathi_repo,
            dataset=dataset,
            output=attempt / "explicit-targets",
            threads=threads,
            targets=("TF1", "G1", "G2"),
        )
        golden = [
            {
                "source": source,
                "target": target,
                "score": str(score),
                "sign": "?",
                "evidence": "association",
                "context": context,
            }
            for context in _CONTEXTS
            for source, target, score in (
                ("TF1", "G1", 0.9),
                ("TF1", "G2", 0.5),
                ("TF2", "G2", 0.5),
            )
        ]
        evaluation = _evaluate(
            andrea_repo=andrea_repo,
            root=attempt / "handoff",
            dataset=dataset,
            networks={
                "fixed": fixed,
                "explicit_targets": subset,
                "golden_ties": golden,
                "empty": [],
            },
        )
        receipt.update(
            {
                "status": "complete",
                "deterministic_network_sha256": canonical_json_sha256(fixed),
                "models": {
                    "fixed": fixed_counts,
                    "repeat": repeated_counts,
                    "explicit_targets": subset_counts,
                },
                "metrics": evaluation["directed_metrics"],
                "andrea_report": evaluation["report"],
                "threads_tested": sorted({1, threads}),
            }
        )
        if threads == 1:
            receipt["covers"][2] = "Repeated same-seed SPATHI runs produce identical networks"
            receipt["does_not_cover"].append(
                "Cross-thread reproducibility (only one thread was requested)"
            )
        metrics_html = "".join(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row[key]))}</td>"
                for key in ("tool_id", "context", "aupr", "auroc")
            )
            + "</tr>"
            for row in receipt["metrics"]
        )
        report = "<!doctype html><html lang='en'><meta charset='utf-8'><title>Scientific contract qualification</title>"
        report += "<style>body{font:16px system-ui;max-width:1000px;margin:3em auto}td,th{padding:.5em;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>"
        report += "<h1>Scientific contract qualification: passed</h1><p>Synthetic fixture: 8 genes, 24 cells, 2 groups, 2 TFs. These are software checks, not a scientific ranking.</p>"
        report += (
            "<h2>Covered</h2><ul>"
            + "".join(f"<li>{html.escape(value)}</li>" for value in receipt["covers"])
            + "</ul>"
        )
        report += (
            "<h2>Not covered</h2><ul>"
            + "".join(f"<li>{html.escape(value)}</li>" for value in receipt["does_not_cover"])
            + "</ul>"
        )
        report += (
            "<h2>Metrics checked against the independent oracle</h2><table><tr><th>Run</th><th>Context</th><th>AUPR</th><th>AUROC</th></tr>"
            + metrics_html
            + "</table></html>"
        )
        atomic_write_text(attempt / "qualification.html", report, "qualification report")
        receipt["report"] = str(attempt / "qualification.html")
        receipt["files"] = {
            path.relative_to(attempt).as_posix(): sha256_file(path)
            for path in sorted(attempt.rglob("*"))
            if path.is_file()
        }
    except Exception as exc:
        receipt.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        receipt["wall_time_seconds"] = time.monotonic() - started
        _write_json(attempt / "receipt.json", receipt)
        _write_json(qualification_root / "receipt.json", receipt)
    return receipt
