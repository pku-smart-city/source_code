#!/usr/bin/env bash
set -euo pipefail

# Batch runner for reproducible multi-seed evaluations.
# Default: 5 seeds, 100 routes each.

MODE="${MODE:-wm}"  # wm | ablate
SEEDS="${SEEDS:-41 42 43 44 45}"
NUM_ROUTES="${NUM_ROUTES:-100}"
MAX_ROUTE_TICKS="${MAX_ROUTE_TICKS:-4000}"
CARLA_TIMEOUT="${CARLA_TIMEOUT:-300}"
TM_PORT="${TM_PORT:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_WM="${ROOT_DIR}/scripts/run_llm_wm_tm_hybrid.sh"
RUN_ABLATE="${ROOT_DIR}/scripts/run_llm_wm_tm_hybrid_ablate_wm.sh"

if [[ "${MODE}" == "wm" ]]; then
  RUNNER="${RUN_WM}"
  LOG_ROOT_DEFAULT="src/experiments/wm_hybrid_100routes"
elif [[ "${MODE}" == "ablate" ]]; then
  RUNNER="${RUN_ABLATE}"
  LOG_ROOT_DEFAULT="src/experiments/wm_hybrid_ablate_100routes"
else
  echo "Unsupported MODE=${MODE}, expected wm|ablate" >&2
  exit 2
fi

LOG_ROOT="${LOG_ROOT:-${LOG_ROOT_DEFAULT}}"

cd "${ROOT_DIR}"
echo "Batch config:"
echo "  MODE=${MODE}"
echo "  SEEDS=${SEEDS}"
echo "  NUM_ROUTES=${NUM_ROUTES}"
echo "  MAX_ROUTE_TICKS=${MAX_ROUTE_TICKS}"
echo "  CARLA_TIMEOUT=${CARLA_TIMEOUT}"
echo "  TM_PORT=${TM_PORT}"
echo "  LOG_ROOT=${LOG_ROOT}"
echo

for s in ${SEEDS}; do
  echo "===== Seed ${s} ====="
  NUM_ROUTES="${NUM_ROUTES}" \
  MAX_ROUTE_TICKS="${MAX_ROUTE_TICKS}" \
  CARLA_TIMEOUT="${CARLA_TIMEOUT}" \
  TM_PORT="${TM_PORT}" \
  LOG_ROOT="${LOG_ROOT}" \
  SEED="${s}" \
  bash "${RUNNER}"
  echo
done

echo "Done."
