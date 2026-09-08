#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_ROOT/lib/common.sh"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/performance.sh [--profile full|reduced] [--efficiency-run-id ID]"
  echo "Runs or resumes the formal isolated computational-performance study."
  echo
  echo "Options:"
  echo "  --profile full|reduced    Publication protocol (default) or short descriptive check."
  echo "  --efficiency-run-id ID    Use an explicit completed, same-profile efficiency workflow."
  bench_resource_help
  echo "  -h, --help                Show this help."
  echo
  echo "Reduced measures four methods on one dataset at 1/8 threads once, plus warm-ups."
  echo "Full measures six methods on small/medium datasets at 1/2/4/8 threads with"
  echo "three system repetitions (144 measured runs), plus excluded warm-ups."
  echo "Every run is fresh, sequential, CPU-pinned, and retains full ANDREA outputs."
  exit 0
fi

bench_environment

bench_command performance \
  --repo-root "$BENCH_REPO_ROOT" \
  --andrea-repo "$BENCH_ANDREA_REPO" \
  --spathi-repo "$BENCH_SPATHI_REPO" \
  --work-root "$BENCH_WORK_ROOT" \
  "$@"
