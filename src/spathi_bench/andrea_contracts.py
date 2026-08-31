"""Load output contracts from the exact ANDREA checkout locked by a run."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import BenchmarkConfigError


def _module_belongs_to(module: Any, repository: Path) -> bool:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        return False
    try:
        return Path(raw_path).resolve().is_relative_to(repository.resolve())
    except OSError:
        return False


@contextmanager
def _exact_andrea_import(andrea_repo: Path) -> Iterator[None]:
    """Temporarily import ``andrea`` from one explicit source checkout.

    SPATHI-benchmarks can coexist with another editable ANDREA installation.
    Silently using that installation would make bundle layouts depend on the
    caller's shell, so an already imported checkout must match exactly.
    """

    repository = andrea_repo.resolve()
    package_init = repository / "andrea" / "__init__.py"
    if not package_init.is_file():
        raise BenchmarkConfigError(f"ANDREA package is missing from {repository}")

    existing_names = {
        name for name in sys.modules if name == "andrea" or name.startswith("andrea.")
    }
    existing_root = sys.modules.get("andrea")
    if existing_root is not None and not _module_belongs_to(existing_root, repository):
        raise BenchmarkConfigError(
            "A different ANDREA checkout is already imported in the benchmark process"
        )

    sys.path.insert(0, str(repository))
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("andrea")
        if not _module_belongs_to(module, repository):
            raise BenchmarkConfigError(
                f"ANDREA imported from an unexpected location: {getattr(module, '__file__', None)}"
            )
        yield
    finally:
        try:
            sys.path.remove(str(repository))
        except ValueError:
            pass
        if not existing_names:
            for name in list(sys.modules):
                if name == "andrea" or name.startswith("andrea."):
                    del sys.modules[name]
        importlib.invalidate_caches()


def validate_inference_analysis_contract(andrea_repo: Path, run_report: Path) -> None:
    """Validate a joint inference view with ANDREA's own analysis contract."""

    with _exact_andrea_import(andrea_repo):
        from andrea.core.commands.evaluate_inference import (  # type: ignore[import-not-found]
            validate_inference_analysis_inputs,
        )
        from andrea.core.commands.infer_network import (  # type: ignore[import-not-found]
            bundles,
        )

        validate_inference_analysis_inputs(run_report_path=run_report)
        resolution = bundles.resolve_bundle(bundle_id="analysis", run_dir=run_report.parent)
        if not resolution.available:
            raise BenchmarkConfigError(
                "Integrated inference does not satisfy ANDREA's analysis bundle: "
                + ", ".join(resolution.missing_required)
            )


def resolve_analysis_bundle(
    andrea_repo: Path,
    *,
    command: str,
    root: Path,
    dataset_id: str | None = None,
) -> Any:
    """Resolve an official ``analysis`` bundle from the locked ANDREA checkout."""

    with _exact_andrea_import(andrea_repo):
        if command == "generate-data":
            from andrea.core.commands.generate_data import (  # type: ignore[import-not-found]
                bundles,
            )

            return bundles.resolve_bundle(
                bundle_id="analysis",
                benchmark_root=root,
                dataset_id=dataset_id,
            )
        if command == "infer-network":
            from andrea.core.commands.infer_network import (  # type: ignore[import-not-found]
                bundles,
            )

            return bundles.resolve_bundle(bundle_id="analysis", run_dir=root)
        if command == "evaluate-inference":
            from andrea.core.commands.evaluate_inference import (  # type: ignore[import-not-found]
                bundles,
            )

            return bundles.resolve_bundle(bundle_id="analysis", evaluation_dir=root)
    raise ValueError(f"Unsupported ANDREA analysis bundle command: {command}")
