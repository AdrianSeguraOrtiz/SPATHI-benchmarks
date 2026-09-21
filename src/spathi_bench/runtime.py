"""Transparent subprocess and environment helpers.

Commands are streamed unchanged to the console and a durable log. There are
no warning filters, image locks, or hidden interpreter overrides: the process
running ``spathi-bench`` is the process that runs ANDREA.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from .config import BenchmarkConfigError


class CommandExecutionError(RuntimeError):
    """A streamed external command exited unsuccessfully."""

    def __init__(self, *, command: Sequence[str], return_code: int, log_path: Path) -> None:
        self.command = tuple(str(item) for item in command)
        self.return_code = int(return_code)
        self.log_path = Path(log_path).resolve()
        super().__init__(f"Command failed with exit code {self.return_code}; see {self.log_path}")


def stream_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Run a command while copying merged output verbatim to console and log."""

    argv = [str(item) for item in command]
    if not argv:
        raise ValueError("command must not be empty")
    working_directory = Path(cwd).resolve()
    if not working_directory.is_dir():
        raise BenchmarkConfigError(f"Command working directory does not exist: {working_directory}")
    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(argv)
    print(f"+ {rendered}", flush=True)
    try:
        with destination.open("w", encoding="utf-8") as log:
            log.write(f"$ {rendered}\n")
            process = subprocess.Popen(
                argv,
                cwd=working_directory,
                env=dict(environment) if environment is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            _copy_stream(process.stdout, log)
            return_code = process.wait()
    except OSError as exc:
        raise BenchmarkConfigError(f"Cannot execute {argv[0]!r}: {exc}") from exc
    if return_code != 0:
        raise CommandExecutionError(
            command=argv,
            return_code=return_code,
            log_path=destination,
        )


def _copy_stream(source: TextIO, log: TextIO) -> None:
    for line in source:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)


def _andrea_environment(andrea_repo: Path) -> dict[str, str]:
    """Prefer the requested ANDREA checkout over installed copies."""

    environment = os.environ.copy()
    source = Path(andrea_repo).resolve()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source) + (os.pathsep + existing if existing else "")
    return environment


def andrea_command(
    arguments: Sequence[str],
    *,
    repo_root: Path,
    andrea_repo: Path,
    log_path: Path,
) -> None:
    """Execute ANDREA from ``andrea_repo`` with the current interpreter."""

    stream_command(
        [sys.executable, "-m", "andrea", *(str(item) for item in arguments)],
        cwd=repo_root,
        log_path=log_path,
        environment=_andrea_environment(andrea_repo),
    )
