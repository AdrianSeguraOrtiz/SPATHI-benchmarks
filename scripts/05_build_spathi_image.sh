#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
bench_init "$@"

bench_cli build-image --spathi-repo "$BENCH_SPATHI_REPO"
