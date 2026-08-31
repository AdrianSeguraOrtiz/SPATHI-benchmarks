"""External-process helpers shared by incremental benchmark stages."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256, sha256_file
from .config import BenchmarkConfigError


def _docker_tag_lock_root() -> Path:
    """Return a host-wide lock directory shared by benchmark run IDs.

    Docker tags live outside any one run or work tree.  Keeping their advisory
    locks below the per-user temporary directory therefore coordinates separate
    benchmark checkouts that talk to the same local Docker daemon.
    """

    uid = os.getuid() if hasattr(os, "getuid") else 0
    root = Path(tempfile.gettempdir()) / f"spathi-benchmarks-docker-tags-{uid}"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise BenchmarkConfigError(f"Docker tag lock root is unsafe: {root}")
    if hasattr(os, "getuid") and root.stat().st_uid != uid:
        raise BenchmarkConfigError(f"Docker tag lock root has the wrong owner: {root}")
    return root


def _catalog_image_matches(
    details: Mapping[str, Any] | None,
    *,
    repo_digest: str,
    expected_id: str | None,
) -> bool:
    if details is None or repo_digest not in details.get("repo_digests", []):
        return False
    return expected_id is None or details.get("id") == expected_id


@contextmanager
def catalog_image_lease(
    lock_entry: Mapping[str, Any],
    *,
    expected_id: str | None = None,
    lock_root: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Keep one mutable catalog tag pinned while a caller uses it.

    A matching tag takes a shared lock, so jobs using the same immutable image
    still run concurrently.  A mismatching tag drops the shared lock, takes the
    exclusive lock, revalidates, and only then retags.  The exclusive lock is
    retained for that first execution; later matching callers use shared locks.
    """

    try:
        tag = str(lock_entry["tag"])
        repo_digest = str(lock_entry["repo_digest"])
    except KeyError as exc:
        raise BenchmarkConfigError("Catalog image lock is missing tag or repo_digest") from exc
    if not tag or not repo_digest:
        raise BenchmarkConfigError("Catalog image tag and repo_digest must be non-empty")

    directory = (lock_root or _docker_tag_lock_root()).resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = directory / f"{canonical_json_sha256(tag)}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            details = docker_image_details(tag)
            if not _catalog_image_matches(
                details,
                repo_digest=repo_digest,
                expected_id=expected_id,
            ):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                details = docker_image_details(tag)
                if not _catalog_image_matches(
                    details,
                    repo_digest=repo_digest,
                    expected_id=expected_id,
                ):
                    details = pin_catalog_image(dict(lock_entry))
                if not _catalog_image_matches(
                    details,
                    repo_digest=repo_digest,
                    expected_id=expected_id,
                ):
                    raise BenchmarkConfigError(
                        f"Catalog tag {tag} did not resolve to its locked image"
                    )
            leased_details = dict(details)
            leased_details["locked_repo_digest"] = repo_digest
            yield leased_details
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def catalog_image_leases(
    requirements: Sequence[tuple[Mapping[str, Any], str]],
    *,
    lock_root: Path | None = None,
) -> Iterator[dict[str, dict[str, Any]]]:
    """Acquire several tag leases in stable order without deadlocking."""

    by_tag: dict[str, tuple[Mapping[str, Any], str]] = {}
    for lock_entry, expected_id in requirements:
        tag = str(lock_entry.get("tag", ""))
        previous = by_tag.get(tag)
        if previous is not None and (
            str(previous[0].get("repo_digest")) != str(lock_entry.get("repo_digest"))
            or previous[1] != expected_id
        ):
            raise BenchmarkConfigError(f"Conflicting immutable requirements for Docker tag {tag}")
        by_tag[tag] = (lock_entry, expected_id)

    acquired: dict[str, dict[str, Any]] = {}
    with ExitStack() as stack:
        for tag, (lock_entry, expected_id) in sorted(by_tag.items()):
            acquired[tag] = stack.enter_context(
                catalog_image_lease(
                    lock_entry,
                    expected_id=expected_id,
                    lock_root=lock_root,
                )
            )
        yield acquired


def andrea_environment(andrea_repo: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(andrea_repo.resolve()) + (
        os.pathsep + existing if existing else ""
    )
    return environment


def _andrea_python_executable() -> Path:
    configured = os.environ.get("ANDREA_PYTHON", sys.executable)
    resolved = shutil.which(configured)
    if resolved is None:
        raise BenchmarkConfigError(f"ANDREA Python {configured!r} is unavailable")
    # Do not Path.resolve() here: dereferencing a virtualenv's Python symlink
    # can bypass its pyvenv.cfg and execute in the base environment instead.
    return Path(os.path.abspath(resolved))


def stream_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(command)
    print(f"+ {rendered}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}; see {log_path}")


def andrea_command(
    arguments: list[str],
    *,
    repo_root: Path,
    andrea_repo: Path,
    log_path: Path,
) -> None:
    executable = _andrea_python_executable()
    stream_command(
        [str(executable), "-m", "andrea", *arguments],
        cwd=repo_root,
        log_path=log_path,
        environment=andrea_environment(andrea_repo),
    )


def andrea_contract_command(
    arguments: list[str],
    *,
    repo_root: Path,
    andrea_repo: Path,
    log_path: Path,
) -> None:
    """Run a SPATHI-benchmarks contract helper in ANDREA's exact environment."""

    executable = _andrea_python_executable()
    environment = andrea_environment(andrea_repo)
    benchmark_source = str((repo_root / "src").resolve())
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = benchmark_source + (
        os.pathsep + existing if existing else ""
    )
    stream_command(
        [str(executable), "-m", "spathi_bench.andrea_bridge", *arguments],
        cwd=repo_root,
        log_path=log_path,
        environment=environment,
    )


def andrea_host_environment() -> dict[str, Any]:
    """Fingerprint the exact host Python environment used to orchestrate ANDREA."""

    launcher = _andrea_python_executable()
    probe = """
import importlib.metadata
import json
import platform
import sys

packages = sorted({
    str(distribution.metadata.get("Name") or "").lower(): distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}.items())
print(json.dumps({
    "python_executable": sys.executable,
    "python_prefix": sys.prefix,
    "python_base_prefix": sys.base_prefix,
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "system": platform.system(),
    "machine": platform.machine(),
    "packages": [{"name": name, "version": version} for name, version in packages],
}, sort_keys=True))
""".strip()
    completed = subprocess.run(
        [str(launcher), "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BenchmarkConfigError(
            "Cannot inspect the ANDREA Python environment: "
            + (completed.stderr or completed.stdout).strip()
        )
    try:
        details = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise BenchmarkConfigError("ANDREA Python environment probe returned invalid JSON") from exc
    if not isinstance(details, dict) or not isinstance(details.get("python_executable"), str):
        raise BenchmarkConfigError("ANDREA Python environment probe returned an invalid payload")
    interpreter = Path(os.path.abspath(details["python_executable"]))
    details.update(
        {
            "launcher": str(launcher),
            "launcher_sha256": sha256_file(launcher),
            "python_executable": str(interpreter),
            "python_executable_sha256": sha256_file(interpreter),
        }
    )
    details["fingerprint"] = canonical_json_sha256(details)
    return details


def benchmark_host_environment() -> dict[str, Any]:
    """Fingerprint the Python interpreter executing SPATHI-benchmarks itself."""

    import platform

    executable = Path(os.path.abspath(sys.executable))
    details: dict[str, Any] = {
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    details["fingerprint"] = canonical_json_sha256(details)
    return details


def check_docker() -> None:
    if shutil.which("docker") is None:
        raise BenchmarkConfigError("Docker CLI is unavailable")
    completed = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BenchmarkConfigError(
            "Docker daemon is unavailable: " + (completed.stderr or completed.stdout).strip()
        )


def docker_image_details(reference: str) -> dict[str, Any] | None:
    completed = subprocess.run(
        ["docker", "image", "inspect", reference],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise BenchmarkConfigError(f"Unexpected Docker inspect response for {reference}")
    image = payload[0]
    return {
        "reference": reference,
        "id": image.get("Id"),
        "repo_digests": sorted(image.get("RepoDigests") or []),
        "created": image.get("Created"),
        "labels": (image.get("Config") or {}).get("Labels") or {},
    }


def pin_catalog_image(lock_entry: dict[str, Any]) -> dict[str, Any]:
    """Materialize a locked repository digest under the catalog tag."""

    check_docker()
    tag = str(lock_entry["tag"])
    repo_digest = str(lock_entry["repo_digest"])
    completed = subprocess.run(
        ["docker", "pull", repo_digest],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise BenchmarkConfigError(
            f"Cannot pull locked image {repo_digest}: "
            + (completed.stderr or completed.stdout).strip()
        )
    tagged = subprocess.run(
        ["docker", "tag", repo_digest, tag],
        check=False,
        capture_output=True,
        text=True,
    )
    if tagged.returncode != 0:
        raise BenchmarkConfigError(
            f"Cannot tag locked image {repo_digest} as {tag}: "
            + (tagged.stderr or tagged.stdout).strip()
        )
    details = docker_image_details(tag)
    if details is None or repo_digest not in details["repo_digests"]:
        raise BenchmarkConfigError(
            f"Catalog tag {tag} does not resolve to locked digest {repo_digest}"
        )
    details["locked_repo_digest"] = repo_digest
    return details


def validate_andrea_import(andrea_repo: Path) -> str:
    executable = _andrea_python_executable()
    completed = subprocess.run(
        [
            str(executable),
            "-c",
            "import pathlib,andrea; print(pathlib.Path(andrea.__file__).resolve())",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=andrea_environment(andrea_repo),
    )
    if completed.returncode != 0:
        raise BenchmarkConfigError(
            "Cannot import locked ANDREA source: " + (completed.stderr or completed.stdout).strip()
        )
    imported = Path(completed.stdout.strip()).resolve()
    if not imported.is_relative_to(andrea_repo.resolve()):
        raise BenchmarkConfigError(f"ANDREA import resolved outside locked repository: {imported}")
    return str(imported)
