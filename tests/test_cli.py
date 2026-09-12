from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.cli as cli
from spathi_bench.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_path_conversion_preserves_symlink_evidence(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)

    converted = cli._path(str(linked))

    assert converted == linked
    assert converted.is_symlink()


def test_cli_help_describes_bounded_calibration_and_conditional_inference() -> None:
    help_text = " ".join(cli.build_parser().format_help().split())

    assert "bounded six-stage SPATHI calibration" in help_text
    assert "inferential tests only when authorized" in help_text


def test_run_all_help_describes_the_low_level_stage_contract() -> None:
    completed = subprocess.run(
        [str(REPO_ROOT / "scripts/run_all.sh"), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Runs a runnable low-level profile" in completed.stdout
    assert "preflight, image, generate" in completed.stdout


def test_study_script_exposes_the_identity_handoff_contract() -> None:
    completed = subprocess.run(
        [str(REPO_ROOT / "scripts/study.sh"), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "complete content-bound SPATHI benchmark study" in completed.stdout
    assert "workflow IDs are carried automatically" in completed.stdout
    assert "--calibration-run-id ID" in completed.stdout


def test_study_cli_forwards_budget_and_resource_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "run_study_workflow",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )
    repo = tmp_path / "repo"

    assert (
        main(
            [
                "study",
                "--repo-root",
                str(repo),
                "--andrea-repo",
                str(tmp_path / "ANDREA"),
                "--spathi-repo",
                str(tmp_path / "SPATHI"),
                "--profile",
                "reduced",
                "--calibration-run-id",
                "calibration-current",
                "--max-cores",
                "12",
                "--reserve-ram-gb",
                "4",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "repo_root": repo.resolve(),
            "andrea_repo": (tmp_path / "ANDREA").resolve(),
            "spathi_repo": (tmp_path / "SPATHI").resolve(),
            "work_root": repo.resolve() / "work",
            "profile": "reduced",
            "calibration_run_id": "calibration-current",
            "max_cores": 12,
            "max_ram_gb": None,
            "reserve_cores": 0,
            "reserve_ram_gb": 4.0,
        }
    ]


def test_run_all_rejects_unknown_flags_with_usage() -> None:
    completed = subprocess.run(
        [
            str(REPO_ROOT / "scripts/run_all.sh"),
            "smoke",
            "argument-contract-test",
            "--not-a-real-option",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Unknown argument: --not-a-real-option" in completed.stderr
    assert "Usage: run_all.sh [PROFILE] [RUN_ID] [--from STAGE] [--until STAGE]" in completed.stderr


def test_high_level_calibrate_cli_uses_default_work_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "run_calibration_workflow",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )
    repo = tmp_path / "repo"

    assert (
        main(
            [
                "calibrate",
                "--repo-root",
                str(repo),
                "--andrea-repo",
                str(tmp_path / "ANDREA"),
                "--spathi-repo",
                str(tmp_path / "SPATHI"),
            ]
        )
        == 0
    )
    assert calls[0]["work_root"] == repo.resolve() / "work"
    assert calls[0]["calibration_budget"] == "full"
    assert calls[0]["run_id"] is None


def test_high_level_calibrate_cli_forwards_reduced_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "run_calibration_workflow",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )

    assert (
        main(
            [
                "calibrate",
                "--repo-root",
                str(tmp_path / "repo"),
                "--andrea-repo",
                str(tmp_path / "ANDREA"),
                "--spathi-repo",
                str(tmp_path / "SPATHI"),
                "--profile",
                "reduced",
            ]
        )
        == 0
    )
    assert calls[0]["calibration_budget"] == "reduced"


def test_high_level_compare_cli_defaults_to_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "run_comparison_workflow",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )

    assert (
        main(
            [
                "compare",
                "--repo-root",
                str(tmp_path / "repo"),
                "--andrea-repo",
                str(tmp_path / "ANDREA"),
                "--spathi-repo",
                str(tmp_path / "SPATHI"),
            ]
        )
        == 0
    )
    assert calls[0]["profile"] == "full"
    assert calls[0]["efficiency_run_id"] is None


def test_high_level_scalability_cli_forwards_frozen_policy_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "run_scalability_workflow",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )
    repo = tmp_path / "repo"

    assert (
        main(
            [
                "scalability",
                "--repo-root",
                str(repo),
                "--andrea-repo",
                str(tmp_path / "ANDREA"),
                "--spathi-repo",
                str(tmp_path / "SPATHI"),
                "--profile",
                "reduced",
                "--efficiency-run-id",
                "efficiency-fixture",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "repo_root": repo.resolve(),
            "andrea_repo": (tmp_path / "ANDREA").resolve(),
            "spathi_repo": (tmp_path / "SPATHI").resolve(),
            "work_root": repo.resolve() / "work",
            "profile": "reduced",
            "efficiency_run_id": "efficiency-fixture",
            "max_cores": None,
            "max_ram_gb": None,
            "reserve_cores": 0,
            "reserve_ram_gb": 0.0,
        }
    ]


def test_high_level_dimensional_transfer_cli_forwards_frozen_policy_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "run_dimensional_transfer_workflow",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )
    repo = tmp_path / "repo"

    assert (
        main(
            [
                "dimensional-transfer",
                "--repo-root",
                str(repo),
                "--andrea-repo",
                str(tmp_path / "ANDREA"),
                "--spathi-repo",
                str(tmp_path / "SPATHI"),
                "--profile",
                "reduced",
                "--efficiency-run-id",
                "efficiency-fixture",
            ]
        )
        == 0
    )
    assert calls == [
        {
            "repo_root": repo.resolve(),
            "andrea_repo": (tmp_path / "ANDREA").resolve(),
            "spathi_repo": (tmp_path / "SPATHI").resolve(),
            "work_root": repo.resolve() / "work",
            "profile": "reduced",
            "efficiency_run_id": "efficiency-fixture",
            "max_cores": None,
            "max_ram_gb": None,
            "reserve_cores": 0,
            "reserve_ram_gb": 0.0,
        }
    ]


@pytest.mark.parametrize("command", ["calibrate", "compare"])
def test_high_level_report_only_uses_published_outputs(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_calls: list[dict[str, Any]] = []
    comparison_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        cli,
        "rebuild_calibration_report",
        lambda **kwargs: calibration_calls.append(kwargs) or {"status": "rendered"},
    )
    monkeypatch.setattr(
        cli,
        "rebuild_comparison_report",
        lambda **kwargs: comparison_calls.append(kwargs) or {"status": "rendered"},
    )
    repo = tmp_path / "repo"
    arguments = [
        command,
        "--repo-root",
        str(repo),
        "--andrea-repo",
        str(tmp_path / "ANDREA"),
        "--spathi-repo",
        str(tmp_path / "SPATHI"),
        "--profile",
        "reduced",
        "--report-only",
    ]
    if command == "calibrate":
        arguments.extend(["--run-id", "chosen"])
    else:
        arguments.extend(["--efficiency-run-id", "chosen"])

    assert main(arguments) == 0

    if command == "calibrate":
        assert comparison_calls == []
        assert calibration_calls == [
            {
                "repo_root": repo.resolve(),
                "work_root": repo.resolve() / "work",
                "calibration_budget": "reduced",
                "run_id": "chosen",
            }
        ]
    else:
        assert calibration_calls == []
        assert comparison_calls == [
            {
                "repo_root": repo.resolve(),
                "work_root": repo.resolve() / "work",
                "profile": "reduced",
                "efficiency_run_id": "chosen",
            }
        ]
