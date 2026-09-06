from __future__ import annotations

import json
from pathlib import Path

import pytest

from spathi_bench.artifacts import ArtifactStore, canonical_json_sha256
from spathi_bench.config import BenchmarkConfigError
from spathi_bench.evaluation_identity import load_evaluation_space_identity


def _truth_artifact(
    store: ArtifactStore,
    *,
    variant: str,
    genes: list[str],
    source: str,
    truth_target: str,
):
    dataset_id = "same-cardinalities"
    fingerprint = {
        "algorithm": "sha256",
        "value": canonical_json_sha256({"dataset": dataset_id, "variant": variant}),
    }
    spec = {"fixture": variant}
    key = store.key_for("dataset", spec)
    with store.staging_directory(key) as staging:
        root = staging / "output" / "datasets" / dataset_id
        (root / "truth").mkdir(parents=True)
        (root / "extras").mkdir()
        (root / "truth" / "gene_universe.txt").write_text(
            "".join(f"{gene}\n" for gene in genes), encoding="utf-8"
        )
        (root / "extras" / "tf_list.txt").write_text(f"{source}\n", encoding="utf-8")
        (root / "truth" / "networks.csv").write_text(
            "source,target,score,sign,evidence,context\n"
            f"{source},{truth_target},1,+,simulation,group:a\n",
            encoding="utf-8",
        )
        manifest = root / "ground-truth-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "dataset_fingerprint": fingerprint,
                    "outputs": {
                        "gene_universe": "truth/gene_universe.txt",
                        "networks": "truth/networks.csv",
                    },
                    "candidate_space": {
                        "sources": "extras/tf_list.txt",
                        "targets": "truth/gene_universe.txt",
                        "allow_self_edges": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        artifact = store.publish_from_temp(staging, kind="dataset", spec=spec)
    relative = Path("output") / "datasets" / dataset_id / "ground-truth-manifest.json"
    return artifact, artifact.file(relative.as_posix()), fingerprint


def _report(fingerprint: dict[str, str]) -> dict[str, object]:
    return {
        "inputs": {
            "inference_dataset_id": "same-cardinalities",
            "ground_truth_dataset_id": "same-cardinalities",
            "inference_dataset_fingerprint": fingerprint,
            "ground_truth_dataset_fingerprint": fingerprint,
        },
        "ground_truth": {
            "dataset_id": "same-cardinalities",
            "dataset_fingerprint": fingerprint,
            "gene_universe_size": 3,
            "candidate_space": {
                "allow_self_edges": False,
                "n_sources": 1,
                "n_targets": 3,
                "n_source_target_overlap": 1,
                "n_candidates_by_level": {
                    "topology": 2,
                    "directed": 2,
                    "signed": 4,
                },
            },
        },
    }


def test_semantic_hashes_distinguish_equal_size_gene_and_edge_universes(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    first_artifact, first_manifest, _ = _truth_artifact(
        store,
        variant="first",
        genes=["TF1", "G1", "G2"],
        source="TF1",
        truth_target="G1",
    )
    second_artifact, second_manifest, _ = _truth_artifact(
        store,
        variant="second",
        genes=["TF2", "G1", "G2"],
        source="TF2",
        truth_target="G2",
    )
    third_artifact, third_manifest, _ = _truth_artifact(
        store,
        variant="third",
        genes=["TF1", "G1", "G2"],
        source="TF1",
        truth_target="G2",
    )
    first = load_evaluation_space_identity(
        dataset_artifact=first_artifact,
        ground_truth_manifest_path=first_manifest,
        expected_dataset_id="same-cardinalities",
    )
    second = load_evaluation_space_identity(
        dataset_artifact=second_artifact,
        ground_truth_manifest_path=second_manifest,
        expected_dataset_id="same-cardinalities",
    )
    third = load_evaluation_space_identity(
        dataset_artifact=third_artifact,
        ground_truth_manifest_path=third_manifest,
        expected_dataset_id="same-cardinalities",
    )

    assert first.candidate_count_by_level == second.candidate_count_by_level
    assert first.source_universe_sha256 != second.source_universe_sha256
    assert (
        first.candidate_space_sha256_by_level["directed"]
        != second.candidate_space_sha256_by_level["directed"]
    )
    assert (
        first.truth_edges_sha256_by_context_level[("group:a", "directed")]
        != second.truth_edges_sha256_by_context_level[("group:a", "directed")]
    )
    assert (
        first.candidate_space_sha256_by_level["directed"]
        == third.candidate_space_sha256_by_level["directed"]
    )
    assert (
        first.truth_edges_sha256_by_context_level[("group:a", "directed")]
        != third.truth_edges_sha256_by_context_level[("group:a", "directed")]
    )


def test_identity_validates_report_and_metric_against_immutable_truth(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact, manifest, fingerprint = _truth_artifact(
        store,
        variant="bound",
        genes=["TF1", "G1", "G2"],
        source="TF1",
        truth_target="G1",
    )
    identity = load_evaluation_space_identity(
        dataset_artifact=artifact,
        ground_truth_manifest_path=manifest,
        expected_dataset_id="same-cardinalities",
    )
    report = _report(fingerprint)
    metric = {
        "context": "group:a",
        "level": "directed",
        "n_candidates": 2,
        "n_candidate_genes": 3,
        "n_candidate_sources": 1,
        "n_candidate_targets": 3,
        "n_truth_edges": 1,
    }

    identity.validate_evaluation_report(report, dataset_id="same-cardinalities")
    identity.validate_metric(metric)
    assert set(identity.metric_provenance(context="group:a", level="directed")) == {
        "dataset_fingerprint_sha256",
        "candidate_source_universe_sha256",
        "candidate_target_universe_sha256",
        "candidate_edge_universe_sha256",
        "truth_edge_universe_sha256",
    }

    report["ground_truth"]["candidate_space"]["n_sources"] = 2  # type: ignore[index]
    with pytest.raises(BenchmarkConfigError, match="n_sources"):
        identity.validate_evaluation_report(report, dataset_id="same-cardinalities")

    metric["n_truth_edges"] = 2
    with pytest.raises(BenchmarkConfigError, match="truth-edge count"):
        identity.validate_metric(metric)
