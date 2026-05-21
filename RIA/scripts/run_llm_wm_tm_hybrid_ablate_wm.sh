#!/usr/bin/env bash
set -euo pipefail

# Hybrid WM-ablation runner:
# - Keep LLM decision loop active in TM mode
# - Disable WM contribution to subaction and TM hint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="/root/anaconda3/etc/profile.d/conda.sh"

NUM_ROUTES="${NUM_ROUTES:-10}"
MAX_ROUTE_TICKS="${MAX_ROUTE_TICKS:-4000}"
CARLA_TIMEOUT="${CARLA_TIMEOUT:-120}"
LOG_ROOT="${LOG_ROOT:-src/experiments/wm_hybrid_ablate}"
SEED="${SEED:-42}"
TM_PORT="${TM_PORT:-8000}"

export MADA_DISABLE_VIDEO_RECORDING="${MADA_DISABLE_VIDEO_RECORDING:-1}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.deepseek.com/v1}"
export MADA_LLM_MODEL="${MADA_LLM_MODEL:-deepseek-reasoner}"
export MADA_ENABLE_THINKING="${MADA_ENABLE_THINKING:-true}"

TM_AGENT_LOOP_INTERVAL="${TM_AGENT_LOOP_INTERVAL:-5}"
TM_ACTION_HINT_SPEED_UP_DIFF="${TM_ACTION_HINT_SPEED_UP_DIFF:--12.0}"
TM_ACTION_HINT_SLOW_DOWN_DIFF="${TM_ACTION_HINT_SLOW_DOWN_DIFF:-20.0}"
TM_ACTION_HINT_LANE_CHANGE_PCT="${TM_ACTION_HINT_LANE_CHANGE_PCT:-35.0}"

source "${CONDA_SH}"
conda activate carla_env
cd "${ROOT_DIR}"

CMD=(
  python src/automatic_control_wm_hybrid.py
  --host 127.0.0.1 --port 2000
  --sync --seed "${SEED}"
  --carla-timeout "${CARLA_TIMEOUT}"
  --tm-port "${TM_PORT}"
  --ablate-wm
  --num-routes "${NUM_ROUTES}"
  --max-route-ticks "${MAX_ROUTE_TICKS}"
  --collision-grace-ticks 20
  --max-red-light-hold-ticks 220
  --tm-near-goal-distance 30
  --tm-max-replans-per-route 2
  --tm-replan-stagnation-ticks 450
  --tm-progress-success-threshold 0.70
  --tm-progress-success-min-waypoints 0
  --tm-progress-stable-ticks 400
  --tm-unblock-trigger-ticks 350
  --tm-unblock-duration-ticks 260
  --tm-unblock-max-per-route 3
  --tm-unblock-reserve-near-goal 1
  --tm-unblock-speed-kmh 4.0
  --tm-force-lane-change-from-unblock 2
  --tm-force-lane-change-min-no-progress-ticks 500
  --tm-force-lane-change-cooldown-ticks 500
  --tm-force-lane-change-from-loop-escape 2
  --tm-loop-escape-distance-m 45.0
  --tm-loop-escape-min-no-progress-ticks 300
  --tm-loop-escape-lock-ticks 220
  --tm-loop-escape-cooldown-ticks 700
  --tm-max-soft-resets-per-route 1
  --tm-soft-reset-trigger-ticks 1200
  --tm-soft-reset-max-progress 0.45
  --tm-red-hold-soft-reset-ticks 0
  --tm-agent-loop-interval-ticks "${TM_AGENT_LOOP_INTERVAL}"
  --tm-action-hint-speed-up-diff "${TM_ACTION_HINT_SPEED_UP_DIFF}"
  --tm-action-hint-slow-down-diff "${TM_ACTION_HINT_SLOW_DOWN_DIFF}"
  --tm-action-hint-lane-change-pct "${TM_ACTION_HINT_LANE_CHANGE_PCT}"
  --log-root "${LOG_ROOT}"
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Preset: hybrid TM + LLM active + WM ablation\n'
printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
