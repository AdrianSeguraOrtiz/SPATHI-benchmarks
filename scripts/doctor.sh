#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_ROOT/lib/common.sh"
bench_init smoke doctor

"${BENCH_PYTHON:-python3}" -m spathi_bench.cli doctor \
  --repo-root "$BENCH_REPO_ROOT" \
  --andrea-repo "$BENCH_ANDREA_REPO" \
  --spathi-repo "$BENCH_SPATHI_REPO" \
  --python "${ANDREA_PYTHON:-${BENCH_PYTHON:-python3}}" \
  "$@"
