"""Transactional study journal for crash-safe, cross-revision resumption.

The journal stores orchestration state only.  Scientific outputs remain immutable
files below ``artifacts/`` and are addressed by the SHA-256 identity of the task
that produced them.  A single orchestrator writes SQLite; worker processes only
return results to it.  The transaction boundary prevents partially written state
from being mistaken for completed work after an interruption.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import canonical_json_sha256
from .config import BenchmarkConfigError

SCHEMA_VERSION = 1
TASK_STATES = frozenset({"pending", "running", "complete"})


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _decode(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """One immutable scientific request plus its mutable execution state."""

    task_id: str
    phase: str
    stage: str
    kind: str
    scientific_digest: str
    executor_contract: str
    spec: dict[str, Any]
    dependencies: tuple[str, ...]
    status: str
    attempt: int
    result: dict[str, Any] | None
    error: str | None


class StudyStore:
    """Small SQLite repository with explicit transitions and durable events."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=60.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._create_schema()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> StudyStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield self._connection
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    @contextmanager
    def read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Expose one consistent WAL snapshot without blocking concurrent writers."""

        try:
            self._connection.execute("BEGIN")
            yield self._connection
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _create_schema(self) -> None:
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    observed_at TEXT NOT NULL,
                    benchmark_commit TEXT,
                    benchmark_tree TEXT NOT NULL,
                    spathi_commit TEXT,
                    spathi_scientific_tree TEXT NOT NULL,
                    andrea_commit TEXT,
                    andrea_scientific_tree TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    phase TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    scientific_digest TEXT NOT NULL,
                    executor_contract TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    dependencies_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    CHECK(status IN ('pending','running','complete'))
                );
                CREATE INDEX IF NOT EXISTS tasks_phase_stage
                    ON tasks(phase, stage, status);
                CREATE TABLE IF NOT EXISTS task_attempts (
                    task_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    revision_sequence INTEGER NOT NULL,
                    scientific_digest TEXT NOT NULL,
                    executor_contract TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    result_json TEXT,
                    error TEXT,
                    PRIMARY KEY(task_id, attempt),
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id),
                    FOREIGN KEY(revision_sequence) REFERENCES revisions(sequence),
                    CHECK(status IN ('running','complete','interrupted'))
                );
                CREATE INDEX IF NOT EXISTS task_attempts_revision
                    ON task_attempts(revision_sequence, task_id, attempt);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    level TEXT NOT NULL,
                    event TEXT NOT NULL,
                    task_id TEXT,
                    details_json TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                """
            )
            observed = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if observed is None:
                connection.execute(
                    "INSERT INTO metadata(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(observed["value"]) != SCHEMA_VERSION:
                raise BenchmarkConfigError(
                    f"Unsupported study database schema {observed['value']!r}"
                )

    def initialize(self, manifest: Mapping[str, Any]) -> bool:
        """Create study metadata once; return ``True`` for a new journal."""

        required = {
            "study_id",
            "profile",
            "design_digest",
            "plan_topology_digest",
            "scientific_design",
        }
        missing = sorted(required.difference(manifest))
        if missing:
            raise BenchmarkConfigError(f"Study manifest is missing: {', '.join(missing)}")
        encoded = {key: _json(value) for key, value in manifest.items()}
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key='study_id'"
            ).fetchone()
            if existing is None:
                connection.executemany(
                    "INSERT INTO metadata(key,value) VALUES(?,?)",
                    sorted(encoded.items()),
                )
                connection.execute(
                    "INSERT INTO metadata(key,value) VALUES('created_at',?)",
                    (_json(utc_now()),),
                )
                return True
            study_id = _decode(existing["value"], None)
            if study_id != manifest["study_id"]:
                raise BenchmarkConfigError(
                    f"Study database belongs to {study_id!r}, not {manifest['study_id']!r}"
                )
            stored_profile = self._metadata_value(connection, "profile")
            if stored_profile != manifest["profile"]:
                raise BenchmarkConfigError("An existing study ID cannot change profile")
            stored_design = self._metadata_value(connection, "design_digest")
            if stored_design != manifest["design_digest"]:
                raise BenchmarkConfigError(
                    "The scientific design changed; use a new study ID. "
                    "Implementation-only revisions may resume the existing ID."
                )
            stored_topology = self._metadata_value(connection, "plan_topology_digest")
            if stored_topology != manifest["plan_topology_digest"]:
                raise BenchmarkConfigError("The study task topology changed; use a new study ID")
            return False

    @staticmethod
    def _metadata_value(connection: sqlite3.Connection, key: str) -> Any:
        row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        if row is None:
            raise BenchmarkConfigError(f"Study metadata is missing {key!r}")
        return _decode(row["value"], None)

    def metadata(self) -> dict[str, Any]:
        rows = self._connection.execute("SELECT key,value FROM metadata ORDER BY key").fetchall()
        return {str(row["key"]): _decode(row["value"], None) for row in rows}

    def set_metadata(self, key: str, value: Any) -> None:
        if key in {
            "study_id",
            "profile",
            "design_digest",
            "plan_topology_digest",
            "scientific_design",
        }:
            raise BenchmarkConfigError(f"Immutable study metadata cannot change: {key}")
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, _json(value)),
            )

    def record_revision(self, revision: Mapping[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        """Record implementation provenance and report whether it changed.

        Revisions are evidence, never cache keys by themselves.  Individual task
        digests decide reuse from the exact scientific inputs and executor
        contract, so documentation/reporting commits do not invalidate inference.
        """

        fields = (
            "benchmark_tree",
            "spathi_scientific_tree",
            "andrea_scientific_tree",
        )
        for field in fields:
            if not isinstance(revision.get(field), str) or not revision[field]:
                raise BenchmarkConfigError(f"Revision is missing {field}")
        previous_row = self._connection.execute(
            "SELECT details_json FROM revisions ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous = _decode(previous_row["details_json"], None) if previous_row else None
        current = dict(revision)
        changed = previous is not None and any(
            previous.get(key) != current.get(key) for key in fields
        )
        if previous == current:
            return False, previous
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO revisions(
                    observed_at, benchmark_commit, benchmark_tree,
                    spathi_commit, spathi_scientific_tree,
                    andrea_commit, andrea_scientific_tree, details_json
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    utc_now(),
                    current.get("benchmark_commit"),
                    current["benchmark_tree"],
                    current.get("spathi_commit"),
                    current["spathi_scientific_tree"],
                    current.get("andrea_commit"),
                    current["andrea_scientific_tree"],
                    _json(current),
                ),
            )
        return changed, previous

    def add_tasks(self, tasks: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        """Insert tasks and invalidate only changed tasks plus their descendants."""

        prepared = []
        for raw in tasks:
            spec = dict(raw.get("spec", {}))
            digest = str(raw.get("scientific_digest") or canonical_json_sha256(spec))
            dependencies = tuple(str(value) for value in raw.get("dependencies", ()))
            prepared.append(
                {
                    "task_id": str(raw["task_id"]),
                    "phase": str(raw["phase"]),
                    "stage": str(raw["stage"]),
                    "kind": str(raw["kind"]),
                    "scientific_digest": digest,
                    "executor_contract": str(raw["executor_contract"]),
                    "spec": spec,
                    "dependencies": dependencies,
                }
            )
        identifiers = [task["task_id"] for task in prepared]
        if len(identifiers) != len(set(identifiers)):
            raise BenchmarkConfigError("Planned task IDs must be unique")
        existing_rows = self._connection.execute(
            "SELECT task_id,dependencies_json FROM tasks"
        ).fetchall()
        dependency_graph = {
            str(row["task_id"]): tuple(_decode(row["dependencies_json"], []))
            for row in existing_rows
        }
        dependency_graph.update({str(task["task_id"]): task["dependencies"] for task in prepared})
        known = set(dependency_graph)
        for task in prepared:
            if len(task["dependencies"]) != len(set(task["dependencies"])):
                raise BenchmarkConfigError(
                    f"Task {task['task_id']!r} contains duplicate dependencies"
                )
            missing = sorted(set(task["dependencies"]).difference(known))
            if missing:
                raise BenchmarkConfigError(
                    f"Task {task['task_id']!r} has unknown dependencies: {missing}"
                )

        # Reject a malformed planner result before inserting even one row.  A
        # cycle would otherwise surface only after every reachable expensive
        # task had run and the scheduler finally reported a blocked study.
        remaining = {
            task_id: set(dependencies) for task_id, dependencies in dependency_graph.items()
        }
        ready = sorted(task_id for task_id, dependencies in remaining.items() if not dependencies)
        visited = 0
        while ready:
            completed = ready.pop()
            visited += 1
            for task_id, dependencies in remaining.items():
                if completed not in dependencies:
                    continue
                dependencies.remove(completed)
                if not dependencies:
                    ready.append(task_id)
        if visited != len(remaining):
            cyclic = sorted(task_id for task_id, dependencies in remaining.items() if dependencies)
            raise BenchmarkConfigError(
                "Planned task dependencies contain a cycle: " + ", ".join(cyclic)
            )

        inserted = reused = invalidated = 0
        changed: set[str] = set()
        with self.transaction() as connection:
            for task in prepared:
                row = connection.execute(
                    "SELECT scientific_digest,executor_contract,status,attempt "
                    "FROM tasks WHERE task_id=?",
                    (task["task_id"],),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO tasks(
                            task_id,phase,stage,kind,scientific_digest,executor_contract,
                            spec_json,dependencies_json,status,created_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            task["task_id"],
                            task["phase"],
                            task["stage"],
                            task["kind"],
                            task["scientific_digest"],
                            task["executor_contract"],
                            _json(task["spec"]),
                            _json(list(task["dependencies"])),
                            "pending",
                            utc_now(),
                        ),
                    )
                    inserted += 1
                    continue
                if (
                    row["scientific_digest"] == task["scientific_digest"]
                    and row["executor_contract"] == task["executor_contract"]
                ):
                    reused += 1
                    continue
                if row["status"] == "running":
                    self._interrupt_attempt(
                        connection,
                        task_id=task["task_id"],
                        attempt=int(row["attempt"]),
                        reason="Scientific request changed before completion",
                    )
                connection.execute(
                    """
                    UPDATE tasks SET phase=?,stage=?,kind=?,scientific_digest=?,
                        executor_contract=?,spec_json=?,dependencies_json=?,status='pending',
                        result_json=NULL,error=NULL,started_at=NULL,finished_at=NULL
                    WHERE task_id=?
                    """,
                    (
                        task["phase"],
                        task["stage"],
                        task["kind"],
                        task["scientific_digest"],
                        task["executor_contract"],
                        _json(task["spec"]),
                        _json(list(task["dependencies"])),
                        task["task_id"],
                    ),
                )
                changed.add(task["task_id"])
                invalidated += 1

            # A changed input invalidates every transitive descendant, even when
            # its own literal request did not change.
            frontier = set(changed)
            while frontier:
                next_frontier: set[str] = set()
                rows = connection.execute(
                    "SELECT task_id,dependencies_json,status,attempt FROM tasks"
                ).fetchall()
                for row in rows:
                    task_id = str(row["task_id"])
                    if task_id in changed:
                        continue
                    dependencies = set(_decode(row["dependencies_json"], []))
                    if not dependencies.intersection(frontier):
                        continue
                    if row["status"] == "running":
                        self._interrupt_attempt(
                            connection,
                            task_id=task_id,
                            attempt=int(row["attempt"]),
                            reason="Dependency changed before completion",
                        )
                    connection.execute(
                        """
                        UPDATE tasks SET status='pending',result_json=NULL,error=NULL,
                            started_at=NULL,finished_at=NULL WHERE task_id=?
                        """,
                        (task_id,),
                    )
                    changed.add(task_id)
                    next_frontier.add(task_id)
                    invalidated += 1
                frontier = next_frontier
        return {"inserted": inserted, "reused": reused, "invalidated": invalidated}

    def recover_interrupted(self) -> int:
        """Return orphaned running tasks to pending after a process interruption."""

        with self.transaction() as connection:
            running = connection.execute(
                "SELECT task_id,attempt FROM tasks WHERE status='running' ORDER BY task_id"
            ).fetchall()
            for row in running:
                self._interrupt_attempt(
                    connection,
                    task_id=str(row["task_id"]),
                    attempt=int(row["attempt"]),
                    reason="Interrupted before completion",
                )
            connection.execute(
                """
                UPDATE tasks SET status='pending',error='Interrupted before completion',
                    started_at=NULL,finished_at=NULL WHERE status='running'
                """
            )
            return len(running)

    def ready_tasks(
        self, *, phase: str | None = None, stage: str | None = None
    ) -> list[TaskRecord]:
        """Return pending tasks whose dependencies completed successfully."""

        clauses = ["status='pending'"]
        params: list[str] = []
        if phase is not None:
            clauses.append("phase=?")
            params.append(phase)
        if stage is not None:
            clauses.append("stage=?")
            params.append(stage)
        rows = self._connection.execute(
            f"SELECT * FROM tasks WHERE {' AND '.join(clauses)} ORDER BY task_id",  # noqa: S608
            params,
        ).fetchall()
        statuses = {
            str(row["task_id"]): str(row["status"])
            for row in self._connection.execute("SELECT task_id,status FROM tasks").fetchall()
        }
        return [
            self._record(row)
            for row in rows
            if all(
                statuses.get(dependency) == "complete"
                for dependency in _decode(row["dependencies_json"], [])
            )
        ]

    def start(self, task_id: str) -> TaskRecord:
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise BenchmarkConfigError(f"Unknown task: {task_id}")
            if row["status"] != "pending":
                raise BenchmarkConfigError(f"Task {task_id!r} cannot start from {row['status']!r}")
            revision = connection.execute(
                "SELECT sequence FROM revisions ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            if revision is None:
                raise BenchmarkConfigError(
                    "Cannot start a task before recording the active implementation revision"
                )
            started_at = utc_now()
            attempt = int(row["attempt"]) + 1
            connection.execute(
                """
                UPDATE tasks SET status='running',attempt=attempt+1,started_at=?,
                    finished_at=NULL,error=NULL WHERE task_id=?
                """,
                (started_at, task_id),
            )
            connection.execute(
                """
                INSERT INTO task_attempts(
                    task_id,attempt,revision_sequence,scientific_digest,
                    executor_contract,status,started_at
                ) VALUES(?,?,?,?,?,'running',?)
                """,
                (
                    task_id,
                    attempt,
                    int(revision["sequence"]),
                    str(row["scientific_digest"]),
                    str(row["executor_contract"]),
                    started_at,
                ),
            )
        return self.task(task_id)

    def finish(self, task_id: str, result: Mapping[str, Any]) -> None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT status,attempt FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise BenchmarkConfigError(f"Unknown task: {task_id}")
            if row["status"] != "running":
                raise BenchmarkConfigError(f"Task {task_id!r} cannot finish from {row['status']!r}")
            finished_at = utc_now()
            encoded_result = _json(dict(result))
            connection.execute(
                """
                UPDATE tasks SET status=?,result_json=?,error=?,finished_at=? WHERE task_id=?
                """,
                ("complete", encoded_result, None, finished_at, task_id),
            )
            cursor = connection.execute(
                """
                UPDATE task_attempts SET status='complete',result_json=?,error=NULL,
                    finished_at=? WHERE task_id=? AND attempt=? AND status='running'
                """,
                (encoded_result, finished_at, task_id, int(row["attempt"])),
            )
            if cursor.rowcount != 1:
                raise BenchmarkConfigError(
                    f"Task {task_id!r} has no matching running attempt provenance"
                )

    @staticmethod
    def _interrupt_attempt(
        connection: sqlite3.Connection,
        *,
        task_id: str,
        attempt: int,
        reason: str,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE task_attempts SET status='interrupted',finished_at=?,error=?
            WHERE task_id=? AND attempt=? AND status='running'
            """,
            (utc_now(), reason, task_id, attempt),
        )
        if cursor.rowcount != 1:
            raise BenchmarkConfigError(
                f"Task {task_id!r} has no matching running attempt provenance"
            )

    def task(self, task_id: str) -> TaskRecord:
        row = self._connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise BenchmarkConfigError(f"Unknown task: {task_id}")
        return self._record(row)

    def tasks(self, *, phase: str | None = None) -> list[TaskRecord]:
        if phase is None:
            rows = self._connection.execute("SELECT * FROM tasks ORDER BY task_id").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM tasks WHERE phase=? ORDER BY task_id", (phase,)
            ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _record(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=str(row["task_id"]),
            phase=str(row["phase"]),
            stage=str(row["stage"]),
            kind=str(row["kind"]),
            scientific_digest=str(row["scientific_digest"]),
            executor_contract=str(row["executor_contract"]),
            spec=dict(_decode(row["spec_json"], {})),
            dependencies=tuple(_decode(row["dependencies_json"], [])),
            status=str(row["status"]),
            attempt=int(row["attempt"]),
            result=None if row["result_json"] is None else dict(_decode(row["result_json"], {})),
            error=None if row["error"] is None else str(row["error"]),
        )

    def event(
        self,
        event: str,
        *,
        level: str = "info",
        task_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO events(occurred_at,level,event,task_id,details_json) VALUES(?,?,?,?,?)",
                (utc_now(), level, event, task_id, _json(dict(details or {}))),
            )

    def snapshot(self) -> dict[str, Any]:
        tasks = self.tasks()
        revisions = [
            {
                **dict(_decode(row["details_json"], {})),
                "sequence": int(row["sequence"]),
                "observed_at": str(row["observed_at"]),
            }
            for row in self._connection.execute(
                "SELECT sequence,observed_at,details_json FROM revisions ORDER BY sequence"
            ).fetchall()
        ]
        attempts = [
            {
                "task_id": str(row["task_id"]),
                "attempt": int(row["attempt"]),
                "revision_sequence": int(row["revision_sequence"]),
                "scientific_digest": str(row["scientific_digest"]),
                "executor_contract": str(row["executor_contract"]),
                "status": str(row["status"]),
                "started_at": str(row["started_at"]),
                "finished_at": (None if row["finished_at"] is None else str(row["finished_at"])),
                "result": (
                    None if row["result_json"] is None else dict(_decode(row["result_json"], {}))
                ),
                "error": None if row["error"] is None else str(row["error"]),
            }
            for row in self._connection.execute(
                "SELECT * FROM task_attempts ORDER BY task_id,attempt"
            ).fetchall()
        ]
        events = [
            {
                "sequence": int(row["sequence"]),
                "occurred_at": str(row["occurred_at"]),
                "level": str(row["level"]),
                "event": str(row["event"]),
                "task_id": row["task_id"],
                "details": dict(_decode(row["details_json"], {})),
            }
            for row in self._connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        ]
        counts = {
            state: sum(task.status == state for task in tasks) for state in sorted(TASK_STATES)
        }
        return {
            "metadata": self.metadata(),
            "revisions": revisions,
            "task_attempts": attempts,
            "task_counts": counts,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "phase": task.phase,
                    "stage": task.stage,
                    "kind": task.kind,
                    "scientific_digest": task.scientific_digest,
                    "executor_contract": task.executor_contract,
                    "dependencies": list(task.dependencies),
                    "status": task.status,
                    "attempt": task.attempt,
                    "result": task.result,
                    "error": task.error,
                }
                for task in tasks
            ],
            "events": events,
        }


__all__ = [
    "SCHEMA_VERSION",
    "TASK_STATES",
    "StudyStore",
    "TaskRecord",
    "utc_now",
]
