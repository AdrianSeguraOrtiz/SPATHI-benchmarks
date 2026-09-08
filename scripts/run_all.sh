#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  printf 'Usage: %s [PROFILE] [RUN_ID] [--from STAGE] [--until STAGE]\n' "${0##*/}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  printf '%s\n' \
    'Runs a runnable low-level profile through the selected contiguous stage interval.' \
    'Default profile: smoke. Stages: preflight, image, generate, infer, validate,' \
    'integrate, evaluate, export, summarize, statistics.'
  exit 0
fi

PROFILE="${1:-smoke}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
shift "$(( $# >= 2 ? 2 : $# ))"

FROM_STAGE="preflight"
UNTIL_STAGE="statistics"
while (( $# > 0 )); do
  case "$1" in
    --from)
      FROM_STAGE="${2:?--from requires a stage}"
      shift 2
      ;;
    --until)
      UNTIL_STAGE="${2:?--until requires a stage}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

STAGES=(preflight image generate infer validate integrate evaluate export summarize statistics)
SCRIPTS=(
  00_preflight.sh
  05_build_spathi_image.sh
  10_generate_data.sh
  20_run_inference.sh
  30_validate_handoff.sh
  35_integrate_inference.sh
  40_evaluate.sh
  45_export_analysis.sh
  50_summarize.sh
  60_statistics.sh
)

FROM_INDEX=-1
UNTIL_INDEX=-1
for index in "${!STAGES[@]}"; do
  [[ "${STAGES[$index]}" == "$FROM_STAGE" ]] && FROM_INDEX="$index"
  [[ "${STAGES[$index]}" == "$UNTIL_STAGE" ]] && UNTIL_INDEX="$index"
done
if (( FROM_INDEX < 0 || UNTIL_INDEX < 0 || FROM_INDEX > UNTIL_INDEX )); then
  echo "Invalid stage interval: $FROM_STAGE -> $UNTIL_STAGE" >&2
  exit 2
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "run_all.sh requires flock to protect benchmark resource measurements" >&2
  exit 2
fi
BENCH_LEASE_ROOT="/tmp/spathi-benchmarks-${UID:-0}"
mkdir -p -m 700 "$BENCH_LEASE_ROOT"
BENCH_LEASE_PATH="$BENCH_LEASE_ROOT/spathi-benchmarks-execution-${UID:-0}.lock"
exec 9>"$BENCH_LEASE_PATH"
if ! flock -n 9; then
  echo "Another SPATHI benchmark workflow owns the host execution lease" >&2
  exit 2
fi
printf '{"pid":%d,"workload":"run_all:%s:%s"}\n' \
  "$$" "$PROFILE" "$RUN_ID" >&9

for index in "${!STAGES[@]}"; do
  if (( index >= FROM_INDEX && index <= UNTIL_INDEX )); then
    "$SCRIPT_ROOT/${SCRIPTS[$index]}" "$PROFILE" "$RUN_ID"
  fi
done

echo "Completed SPATHI benchmark stages $FROM_STAGE -> $UNTIL_STAGE: $PROFILE / $RUN_ID"
