#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-smoke}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"

"$SCRIPT_ROOT/00_preflight.sh" "$PROFILE" "$RUN_ID"
"$SCRIPT_ROOT/05_build_spathi_image.sh" "$PROFILE" "$RUN_ID"
"$SCRIPT_ROOT/10_generate_data.sh" "$PROFILE" "$RUN_ID"
"$SCRIPT_ROOT/20_run_inference.sh" "$PROFILE" "$RUN_ID"
"$SCRIPT_ROOT/30_validate_handoff.sh" "$PROFILE" "$RUN_ID"
"$SCRIPT_ROOT/40_evaluate.sh" "$PROFILE" "$RUN_ID"
"$SCRIPT_ROOT/50_summarize.sh" "$PROFILE" "$RUN_ID"

echo "Completed SPATHI benchmark: $PROFILE / $RUN_ID"
