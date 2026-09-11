#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_ROOT/lib/common.sh"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/study.sh [--profile full|reduced]"
  echo "Runs or resumes the complete content-bound SPATHI benchmark study."
  echo
  echo "Options:"
  echo "  --profile full|reduced    Use publication full (default) or operational reduced."
  bench_resource_help
  echo "  -h, --help                Show this help."
  echo
  echo "Calibration and efficiency workflow IDs are carried automatically between phases."
  echo "Historical workflows in work/ never need to be deleted."
  exit 0
fi

bench_environment

bench_command study \
  --repo-root "$BENCH_REPO_ROOT" \
  --andrea-repo "$BENCH_ANDREA_REPO" \
  --spathi-repo "$BENCH_SPATHI_REPO" \
  --work-root "$BENCH_WORK_ROOT" \
  "$@"
