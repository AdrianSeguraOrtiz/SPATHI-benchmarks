#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/justify-efficiency.sh [--profile full|reduced] [--calibration-run-id ID]"
  echo "Runs or resumes the independent SPATHI efficiency-justification study."
  echo
  echo "Options:"
  echo "  --profile full|reduced    Use the full (default) or reduced evidence budget."
  echo "                            Calibration must use the same profile and current image."
  echo "  --calibration-run-id ID   Use a same-profile calibration with an explicit ID."
  echo "  -h, --help                Show this help."
  echo
  echo "The command creates an efficiency_validation panel and runs the exact, target-"
  echo "eligibility, adaptive-tree and combined policies in counterbalanced order. Full"
  echo "uses non-inferiority and cost gates; reduced is descriptive and retains exact."
  echo "A full provisional winner must also pass the conditional safety panel before the"
  echo "production policy is frozen. Repeating the command resumes verified work."
  exit 0
fi

source "$SCRIPT_ROOT/lib/common.sh"
bench_environment

bench_command justify-efficiency \
  --repo-root "$BENCH_REPO_ROOT" \
  --andrea-repo "$BENCH_ANDREA_REPO" \
  --spathi-repo "$BENCH_SPATHI_REPO" \
  --work-root "$BENCH_WORK_ROOT" \
  "$@"
