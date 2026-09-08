#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_ROOT/lib/common.sh"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/justify-efficiency.sh [--profile full|reduced] [--calibration-run-id ID]"
  echo "Runs or resumes the independent SPATHI efficiency-justification study."
  echo
  echo "Options:"
  echo "  --profile full|reduced    Use the full (default) or reduced evidence budget."
  echo "                            Calibration must use the same profile and current image."
  echo "  --calibration-run-id ID   Use a same-profile calibration with an explicit ID."
  bench_resource_help
  echo "  -h, --help                Show this help."
  echo
  echo "Development first screens the six declared component policies using only timing"
  echo "and memory, then nominates exactly one policy without seeing validation outcomes."
  echo "An independent, counterbalanced accuracy panel compares that nominee with exact"
  echo "SPATHI. Full applies non-inferiority, cost, and conditional safety gates; reduced"
  echo "is descriptive and retains exact. Repeating the command resumes verified work."
  exit 0
fi

bench_environment

bench_command justify-efficiency \
  --repo-root "$BENCH_REPO_ROOT" \
  --andrea-repo "$BENCH_ANDREA_REPO" \
  --spathi-repo "$BENCH_SPATHI_REPO" \
  --work-root "$BENCH_WORK_ROOT" \
  "$@"
