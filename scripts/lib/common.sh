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

  export BENCH_PROFILE BENCH_RUN_ID BENCH_WORK_ROOT BENCH_RUN_ROOT
  export BENCH_ANDREA_REPO BENCH_SPATHI_REPO BENCH_REPO_ROOT
  export PYTHONPATH="$BENCH_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
}

bench_cli() {
  python3 -m spathi_bench.cli "$@" \
    --repo-root "$BENCH_REPO_ROOT" \
    --profile "$BENCH_PROFILE" \
    --run-root "$BENCH_RUN_ROOT"
}
