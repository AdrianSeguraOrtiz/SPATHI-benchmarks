#!/usr/bin/env bash

set -euo pipefail

BENCH_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bench_init() {
  BENCH_PROFILE="${1:-smoke}"
  BENCH_RUN_ID="${2:-manual}"

  if [[ ! "$BENCH_PROFILE" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    echo "Invalid profile identifier: $BENCH_PROFILE" >&2
    return 2
  fi
  if [[ ! "$BENCH_RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
    echo "Invalid run identifier: $BENCH_RUN_ID" >&2
    return 2
  fi

  BENCH_WORK_ROOT="${WORK_ROOT:-$BENCH_REPO_ROOT/work}"
  BENCH_RUN_ROOT="$BENCH_WORK_ROOT/v1/$BENCH_PROFILE/$BENCH_RUN_ID"
  BENCH_ANDREA_REPO="${ANDREA_REPO:-$BENCH_REPO_ROOT/../ANDREA}"
  BENCH_SPATHI_REPO="${SPATHI_REPO:-$BENCH_REPO_ROOT/../SPATHI}"
  if [[ -z "${BENCH_PYTHON:-}" ]]; then
    if [[ -x "$BENCH_REPO_ROOT/.venv/bin/python" ]]; then
      BENCH_PYTHON="$BENCH_REPO_ROOT/.venv/bin/python"
    else
      BENCH_PYTHON="python3"
    fi
  fi

}

bench_command() {
  PYTHONPATH="$BENCH_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$BENCH_PYTHON" -m spathi_bench.cli "$@"
}

bench_cli() {
  bench_command "$@" \
    --repo-root "$BENCH_REPO_ROOT" \
    --profile "$BENCH_PROFILE" \
    --run-root "$BENCH_RUN_ROOT"
}
