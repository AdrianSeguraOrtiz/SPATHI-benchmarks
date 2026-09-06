from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

import spathi_bench.runtime as runtime


def _details(image_id: str, repo_digest: str) -> dict[str, Any]:
    return {
        "reference": "catalog/tool:1.0",
        "id": image_id,
        "repo_digests": [repo_digest],
        "created": "2026-08-28T00:00:00Z",
        "labels": {},
    }


def test_matching_catalog_image_leases_run_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_digest = "catalog/tool@sha256:" + "a" * 64
    image_id = "sha256:" + "1" * 64
    lock_entry = {"tag": "catalog/tool:1.0", "repo_digest": repo_digest}
    current = _details(image_id, repo_digest)
    monkeypatch.setattr(runtime, "docker_image_details", lambda _tag: deepcopy(current))
    monkeypatch.setattr(
        runtime,
        "pin_catalog_image",
        lambda _entry: pytest.fail("a matching image must not be repinned"),
    )
    entered = threading.Event()
    release = threading.Event()

    def contender() -> None:
        with runtime.catalog_image_lease(
            lock_entry,
            expected_id=image_id,
            lock_root=tmp_path / "locks",
        ):
            entered.set()
            release.wait(timeout=2)

    with runtime.catalog_image_lease(
        lock_entry,
        expected_id=image_id,
        lock_root=tmp_path / "locks",
    ) as leased:
        assert leased["locked_repo_digest"] == repo_digest
        thread = threading.Thread(target=contender)
        thread.start()
        assert entered.wait(timeout=1), "matching shared lease was needlessly serialized"
        release.set()
        thread.join(timeout=2)

    assert not thread.is_alive()


def test_conflicting_catalog_image_waits_until_active_lease_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest_a = "catalog/tool@sha256:" + "a" * 64
    digest_b = "catalog/tool@sha256:" + "b" * 64
    image_a = "sha256:" + "1" * 64
    image_b = "sha256:" + "2" * 64
    entry_a = {"tag": "catalog/tool:1.0", "repo_digest": digest_a}
    entry_b = {"tag": "catalog/tool:1.0", "repo_digest": digest_b}
    current = _details(image_a, digest_a)
    pin_calls: list[str] = []

    def inspect(_tag: str) -> dict[str, Any]:
        return deepcopy(current)

    def pin(entry: dict[str, Any]) -> dict[str, Any]:
        pin_calls.append(entry["repo_digest"])
        current.clear()
        current.update(_details(image_b, digest_b))
        return deepcopy(current)

    monkeypatch.setattr(runtime, "docker_image_details", inspect)
    monkeypatch.setattr(runtime, "pin_catalog_image", pin)
    attempting = threading.Event()
    entered = threading.Event()

    def contender() -> None:
        attempting.set()
        with runtime.catalog_image_lease(
            entry_b,
            expected_id=image_b,
            lock_root=tmp_path / "locks",
        ):
            entered.set()

    with runtime.catalog_image_lease(
        entry_a,
        expected_id=image_a,
        lock_root=tmp_path / "locks",
    ):
        thread = threading.Thread(target=contender)
        thread.start()
        assert attempting.wait(timeout=1)
        assert not entered.wait(timeout=0.1), "conflicting tag was changed during an active lease"
        assert pin_calls == []

    thread.join(timeout=2)
    assert not thread.is_alive()
    assert entered.is_set()
    assert pin_calls == [digest_b]


def test_catalog_image_leases_reject_conflicting_duplicate_requirements(tmp_path: Path) -> None:
    tag = "catalog/tool:1.0"
    first = {"tag": tag, "repo_digest": "catalog/tool@sha256:" + "a" * 64}
    second = {"tag": tag, "repo_digest": "catalog/tool@sha256:" + "b" * 64}

    with pytest.raises(runtime.BenchmarkConfigError, match="Conflicting immutable requirements"):
        with runtime.catalog_image_leases(
            [(first, "sha256:" + "1" * 64), (second, "sha256:" + "2" * 64)],
            lock_root=tmp_path / "locks",
        ):
            pass


def test_docker_runtime_fingerprint_uses_build_contract_not_context_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "docker"
    executable.write_bytes(b"docker-cli-fixture")
    payload = {
        "Client": {
            "Version": "25.0.1",
            "ApiVersion": "1.44",
            "GitCommit": "client-commit",
            "Context": "mutable-alias",
        },
        "Server": {
            "Version": "25.0.1",
            "ApiVersion": "1.44",
            "GitCommit": "server-commit",
            "Components": [{"Name": "runc", "Version": "1.1.12"}],
        },
    }
    monkeypatch.setattr(runtime.shutil, "which", lambda _value: str(executable))
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **_kwargs: CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    observed = runtime.docker_runtime_environment()

    assert "Context" not in observed["client"]
    assert observed["server"]["Components"][0]["Version"] == "1.1.12"
    assert observed["fingerprint"] == runtime.canonical_json_sha256(
        {key: value for key, value in observed.items() if key != "fingerprint"}
    )


def test_validate_andrea_import_uses_configured_andrea_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    andrea_repo = tmp_path / "ANDREA"
    module = andrea_repo / "andrea" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    configured = tmp_path / "andrea-venv" / "bin" / "python"
    configured.parent.mkdir(parents=True)
    configured.write_text("", encoding="utf-8")
    seen: list[list[str]] = []

    monkeypatch.setenv("ANDREA_PYTHON", str(configured))
    monkeypatch.setattr(runtime.shutil, "which", lambda value: value)

    def run(command: list[str], **_kwargs: Any) -> CompletedProcess[str]:
        seen.append(command)
        return CompletedProcess(command, 0, stdout=str(module) + "\n", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", run)

    assert runtime.validate_andrea_import(andrea_repo) == str(module.resolve())
    assert seen[0][0] == str(configured.absolute())
