from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

import spathi_bench.runtime as runtime


def test_stream_command_preserves_all_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    warning = "warning that must remain visible"
    log = tmp_path / "logs" / "command.log"

    runtime.stream_command(
        [sys.executable, "-c", f"print({warning!r})"],
        cwd=tmp_path,
        log_path=log,
    )

    assert warning in capsys.readouterr().out
    assert warning in log.read_text(encoding="utf-8")


def test_stream_command_exposes_failure_evidence(tmp_path: Path) -> None:
    log = tmp_path / "failed.log"

    with pytest.raises(runtime.CommandExecutionError) as error:
        runtime.stream_command(
            [sys.executable, "-c", "import sys; print('evidence'); sys.exit(7)"],
            cwd=tmp_path,
            log_path=log,
        )

    assert error.value.return_code == 7
    assert error.value.command[0] == sys.executable
    assert error.value.log_path == log.resolve()
    assert "evidence" in log.read_text(encoding="utf-8")


def test_andrea_command_uses_current_interpreter_and_requested_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_stream(command: list[str], **kwargs: Any) -> None:
        observed["command"] = command
        observed.update(kwargs)

    monkeypatch.setattr(runtime.sys, "executable", "/active/venv/bin/python")
    monkeypatch.setattr(runtime, "stream_command", fake_stream)
    andrea = tmp_path / "ANDREA"

    runtime.andrea_command(
        ["infer-network", "--help"],
        repo_root=tmp_path,
        andrea_repo=andrea,
        log_path=tmp_path / "andrea.log",
    )

    assert observed["command"][:3] == [
        "/active/venv/bin/python",
        "-m",
        "andrea",
    ]
    assert observed["command"][3:] == ["infer-network", "--help"]
    assert observed["environment"]["PYTHONPATH"].split(os.pathsep)[0] == str(andrea.resolve())
