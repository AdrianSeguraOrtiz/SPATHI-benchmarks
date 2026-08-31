#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/compare.sh [--profile full|reduced] [--calibration-run-id ID]"
  echo "Runs or resumes the selected SPATHI comparison."
  echo
  echo "Options:"
  echo "  --profile full|reduced    Use held-out full (default) or development reduced."
  echo "  --calibration-run-id ID   Use a calibration created with an explicit --run-id."
  echo "  -h, --help                Show this help."
  echo
  echo "Repeating the same command resumes and verifies completed work."
  exit 0
fi
source "$SCRIPT_ROOT/lib/common.sh"
bench_init smoke compare

bench_command compare \
  --repo-root "$BENCH_REPO_ROOT" \
  --andrea-repo "$BENCH_ANDREA_REPO" \
  --spathi-repo "$BENCH_SPATHI_REPO" \
  --work-root "$BENCH_WORK_ROOT" \
  "$@"
