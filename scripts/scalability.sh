#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_ROOT/lib/common.sh"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/scalability.sh [--profile full|reduced] [--efficiency-run-id ID]"
  echo "Runs or resumes the post-efficiency SPATHI computational scale study."
  echo
  echo "Options:"
  echo "  --profile full|reduced    Full publication evidence (default) or reduced operational check."
  echo "  --efficiency-run-id ID    Use an explicit completed, same-profile efficiency workflow."
  bench_resource_help
  echo "  -h, --help                Show this help."
  echo
  echo "The study uses only the frozen production policy emitted by justify-efficiency."
  echo "Its synthetic scale-stress panels cover the intended application-scale envelope; they do not claim"
  echo "biological realism and are not used as a confirmatory accuracy benchmark."
  exit 0
fi

bench_environment

bench_command scalability \
  --repo-root "$BENCH_REPO_ROOT" \
  --andrea-repo "$BENCH_ANDREA_REPO" \
  --spathi-repo "$BENCH_SPATHI_REPO" \
  --work-root "$BENCH_WORK_ROOT" \
  "$@"
