from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_exact_andrea_import_is_atomic_across_worker_threads(tmp_path: Path) -> None:
    repository = tmp_path / "andrea-repository"
    package = repository / "andrea"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("MARKER = 'locked checkout'\n", encoding="utf-8")

    program = textwrap.dedent(
        """
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        from threading import Event

        from spathi_bench.andrea_contracts import _exact_andrea_import

        repository = Path(sys.argv[1])
        first_entered = Event()
        release_first = Event()
        second_entered = Event()

        def first_import():
            with _exact_andrea_import(repository):
                first_entered.set()
                if not release_first.wait(5):
                    raise RuntimeError("test did not release first import")

        def second_import():
            if not first_entered.wait(5):
                raise RuntimeError("first import did not start")
            with _exact_andrea_import(repository):
                second_entered.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(first_import)
            if not first_entered.wait(5):
                raise RuntimeError("first import did not enter its context")
            second = executor.submit(second_import)
            if second_entered.wait(0.2):
                raise RuntimeError("ANDREA import contexts overlapped")
            release_first.set()
            first.result(timeout=5)
            second.result(timeout=5)
            if not second_entered.is_set():
                raise RuntimeError("second import did not run after serialization")
        """
    )

    subprocess.run(
        [sys.executable, "-c", program, str(repository)],
        check=True,
        text=True,
        capture_output=True,
    )
