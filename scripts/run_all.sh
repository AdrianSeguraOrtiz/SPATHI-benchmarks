#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${1:-smoke}"
RUN_ID="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
shift "$(( $# >= 2 ? 2 : $# ))"

FROM_STAGE="preflight"
UNTIL_STAGE="summarize"
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
    --resume)
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

STAGES=(preflight image generate infer validate evaluate summarize)
SCRIPTS=(
  00_preflight.sh
  05_build_spathi_image.sh
  10_generate_data.sh
  20_run_inference.sh
  30_validate_handoff.sh
  40_evaluate.sh
  50_summarize.sh
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

for index in "${!STAGES[@]}"; do
  if (( index >= FROM_INDEX && index <= UNTIL_INDEX )); then
    "$SCRIPT_ROOT/${SCRIPTS[$index]}" "$PROFILE" "$RUN_ID"
  fi
done

echo "Completed SPATHI benchmark stages $FROM_STAGE -> $UNTIL_STAGE: $PROFILE / $RUN_ID"
