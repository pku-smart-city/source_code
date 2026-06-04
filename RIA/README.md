# Reason–Imagine–Act: Closed-Loop LLM Decision Making with World Models for Autonomous Driving

Short name: `RIA`

This repository implements a closed-loop LLM + world-model driving framework:
- `Reason`: an LLM proposes high-level action templates and candidate sub-actions.
- `Imagine`: a world model (WM) performs short-horizon rollout verification.
- `Act`: the safest candidate is executed via CARLA Traffic Manager (TM), and physical feedback is written back to the next LLM step.

## Upstream Baseline Reference

- MADA (Multi-alignment-Driving-Agent) official repository: https://github.com/AIR-DISCOVER/Multi-alignment-Driving-Agent

## Project Scope

Main target task:
- CARLA point-goal closed-loop navigation.
- Inference-time deployment only (no policy/value fine-tuning during evaluation).
- Unified interface for ablation and baseline comparison.

## Method-to-Code Mapping

- `Reason` (LLM reasoning and structured prompt):
  - `src/agents/navigation/behavior_agent.py`
- `Imagine` (WM feature extraction, rollout, safety scoring):
  - `src/agents/navigation/behavior_agent_wm.py`
  - `src/agents/navigation/wm_core/features.py`
  - `src/agents/navigation/wm_core/wm_adapter.py`
- `Act` (TM execution, recovery logic, route-level evaluation loop):
  - `src/automatic_control_wm_hybrid.py`

Main runnable entry scripts:
- `scripts/run_llm_wm_tm_hybrid.sh` (RIA full pipeline)
- `scripts/run_llm_wm_tm_hybrid_ablate_wm.sh` (LLM + TM with WM ablated)
- `scripts/run_wm_hybrid_multi_seed.sh` (batch multi-seed runner)

## Reported Results

Closed-loop point-goal navigation over 1000 episodes:

| Method | RC (%) | AR (%) | ColR (%) |
|---|---:|---:|---:|
| LLM w/o WM verification | 61.21 | 30.20 | 0.40 |
| CARLA TM | 46.47 | 13.40 | 16.00 |
| [MADA baseline policy](https://github.com/AIR-DISCOVER/Multi-alignment-Driving-Agent) | 21.88 | 22.40 | 53.00 |
| LLM+WM (Ours) | **80.05** | **51.10** | **0.20** |

Metrics:
- `RC`: average route completion.
- `AR`: arrival rate.
- `ColR`: collision rate.

## Environment

- CARLA: `0.9.15`
- Python: `3.7` / `3.8` recommended
- Core packages used by code: `carla`, `pygame`, `opencv-python`, `numpy`, `torch`, `flask`, `openai`

If you are using the provided environment:

```bash
conda activate carla_env
```

## Quick Start

1) Start CARLA server (example):

```bash
cd /path/to/carla_sim
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=2000
```

2) Configure runtime env vars (example):

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_API_BASE="https://api.deepseek.com/v1"
export MADA_LLM_MODEL="deepseek-reasoner"
export MADA_ENABLE_THINKING="true"
export MADA_DISABLE_VIDEO_RECORDING="1"
# optional
# export MADA_DRIVER_PROMPT="src/prompt/demonstration_cautious.txt"
```

3) Run RIA full pipeline:

```bash
bash scripts/run_llm_wm_tm_hybrid.sh
```

4) Run WM ablation:

```bash
bash scripts/run_llm_wm_tm_hybrid_ablate_wm.sh
```

5) Multi-seed batch:

```bash
MODE=wm SEEDS="41 42 43 44 45" NUM_ROUTES=1000 \
bash scripts/run_wm_hybrid_multi_seed.sh
```

## Logs and Outputs

Typical output roots:
- `src/experiments/wm_hybrid/`
- `src/experiments/wm_hybrid_ablate/`
- `src/experiments/wm/`

Typical files per run:
- `run_args.txt`
- `status.txt`
- `times.txt`
- `route_eval_live.json`
- `route_eval_summary.json`
- `tm_event_log.jsonl` (hybrid runner)
- `gpt_log*.csv`
- `canbus_log*.csv`
- `driving.mp4` (if recording is enabled)

## Analysis Utility

Use route-level aggregation and significance testing:

```bash
python scripts/analyze_route_stats.py \
  --spec wm='src/experiments/wm_hybrid_100routes/automatic_control_*' \
  --spec ablate='src/experiments/wm_hybrid_ablate_100routes/automatic_control_*' \
  --require-complete
```

## Important Notes

- This README follows the current repository implementation.
- Some historical docs/scripts in `docs/`, `backups/`, and `src/experiments/` are archival and may reflect earlier setups.
- `src/agents/navigation/behavior_agent.py` currently contains a hardcoded API-key style placeholder; replace it with your secure key management before real deployment.
- A Flask thread is started by `behavior_agent.py` (default port `5000`); check port conflicts if startup fails.
