#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_ROOT/lib/common.sh"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/dimensional-transfer.sh [--profile full|reduced] [--efficiency-run-id ID]"
  echo "Runs or resumes the post-freeze SPATHI dimensional accuracy-transfer study."
  echo
  echo "Options:"
  echo "  --profile full|reduced    Full descriptive evidence (default) or reduced operational check."
  echo "  --efficiency-run-id ID    Use an explicit completed, same-profile efficiency workflow."
  bench_resource_help
  echo "  -h, --help                Show this help."
  echo
  echo "The command executes only the production policy frozen by justify-efficiency."
  echo "Full estimates three descriptive one-axis scale contrasts; reduced runs only four"
  echo "moderate/routing conditions. Neither can alter selection or enter the tool ranking."
  exit 0
fi

bench_environment

bench_command dimensional-transfer \
  --repo-root "$BENCH_REPO_ROOT" \
  --andrea-repo "$BENCH_ANDREA_REPO" \
  --spathi-repo "$BENCH_SPATHI_REPO" \
  --work-root "$BENCH_WORK_ROOT" \
  "$@"
