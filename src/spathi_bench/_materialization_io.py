"""Atomic publication primitives shared by workflow materializers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def publish_new_json_directory(
    output_dir: Path,
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    """Atomically publish a new directory of deterministic JSON documents."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for filename, payload in payloads.items():
            (temporary / filename).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def publish_new_json_file(output_path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a new deterministic JSON file without overwriting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}-", dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
