#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/calibrate.sh [--run-id ID]"
  echo "Runs or resumes the tree-count, structure, estimator and validation stages."
  echo
  echo "Options:"
  echo "  --run-id ID  Name this calibration; the default is derived from locked inputs."
  echo "  -h, --help   Show this help."
  echo
  echo "Repeating the same command resumes and verifies completed work."
  exit 0
fi
source "$SCRIPT_ROOT/lib/common.sh"
bench_init smoke calibrate

bench_command calibrate \
  --repo-root "$BENCH_REPO_ROOT" \
  --andrea-repo "$BENCH_ANDREA_REPO" \
  --spathi-repo "$BENCH_SPATHI_REPO" \
  --work-root "$BENCH_WORK_ROOT" \
  "$@"
