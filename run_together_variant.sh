#!/usr/bin/env bash
# Together AI PAIR Table 2-style launcher.
#
# Edit the "Experiment Settings" block below, then run:
#   bash run_together_variant.sh

set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

###############################################################################
# Experiment Settings
###############################################################################

# Dataset size controls.
# - RUN_MODE="full" runs all 100 JailbreakBench harmful behaviors.
# - RUN_MODE="limit" runs NUM_BEHAVIORS rows starting at START_INDEX.
# - To run the first XX rows, set RUN_MODE="limit", START_INDEX=0,
#   NUM_BEHAVIORS=XX.
RUN_MODE="limit"                # dry-run | limit | full
NUM_BEHAVIORS=50                # used only when RUN_MODE="limit" or "dry-run"
START_INDEX=0                   # zero-based dataset index; 0 means start from first row
PHASE="dev"                     # dev | test | eval; full mode always uses test
RESUME=1                        # 1 skips completed rows in status.jsonl
CONTINUE_ON_ERROR=1             # 1 keeps going if one behavior fails

# Table 2-style roles. The script runs TARGET_MODELS sequentially.
ATTACK_MODEL="qwen-2.5-7b-instruct-turbo"
JUDGE_MODEL="llama-guard-4-12b"
TARGET_MODELS=(
  "llama-3-8b-instruct-lite"
  "gemma-3n-e4b-it"
)

# Use 0 for new Together targets such as Gemma. Use 1 for paper/JBB targets
# such as llama-2-7b-chat-hf if you want the JailbreakBench target wrapper.
USE_JAILBREAKBENCH_TARGET=0

# PAIR budget. Paper-style budget is 30 streams x 3 iterations.
N_STREAMS=30
N_ITERATIONS=3
ATTACK_MAX_N_TOKENS=1024
TARGET_MAX_N_TOKENS=150
JUDGE_MAX_N_TOKENS=64

# Together API key. together-api.key is ignored by git via *.key.
# If the file contains multiple keys, API_KEY_INDEX is 1-based among non-empty,
# non-comment lines. Leave empty to use the last key.
TOGETHER_API_KEY_FILE="together-api.key"
API_KEY_INDEX=""
WANDB_MODE_VALUE="offline"      # offline | online | disabled

# Conda env and outputs. This reuses the same env as the UCSD variant.
CONDA_ENV_PATH="envs/pair"
LOG_ROOT="logs/pair_together_variant"
RESULTS_DIR="results/pair_together_variant"
RUN_POSTPROCESS=1               # 1 creates tables/figures after all targets finish
RUN_WITH_NOHUP=1                 # 1 submits this script to background via nohup
NOHUP_LOG_DIR="_launcher"        # relative to LOG_ROOT; stores launcher stdout/stderr

###############################################################################
# Available Together Model Names Registered In config.py
###############################################################################
#
# Together-hosted models currently registered in this repo:
#   qwen-2.5-7b-instruct-turbo  # serverless attacker candidate
#   mixtral                     # currently not available as serverless on this account
#   gemma-3n-e4b-it
#   gemma-4-31b-it              # API returns empty content in smoke tests
#   llama-3-8b-instruct-lite
#   llama-2-7b-chat-hf
#   llama-3.3-70b-instruct-turbo
#   llama-guard-4-12b
#
# For the current experiment setup, keep:
#   ATTACK_MODEL="qwen-2.5-7b-instruct-turbo"
#   JUDGE_MODEL="llama-guard-4-12b"
# and change only TARGET_MODELS.

###############################################################################
# Launcher Implementation
###############################################################################

if [[ "$CONDA_ENV_PATH" = /* ]]; then
  ENV_DIR="$CONDA_ENV_PATH"
else
  ENV_DIR="$PROJ/$CONDA_ENV_PATH"
fi
PYTHON="$ENV_DIR/bin/python"

if [[ "$TOGETHER_API_KEY_FILE" = /* ]]; then
  TOGETHER_API_KEY_PATH="$TOGETHER_API_KEY_FILE"
else
  TOGETHER_API_KEY_PATH="$PROJ/$TOGETHER_API_KEY_FILE"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python env: $PYTHON" >&2
  echo "Create it with:" >&2
  echo "  conda create -y -p \"$ENV_DIR\" python=3.11" >&2
  echo "  \"$PYTHON\" -m pip install \"litellm==1.52.0\" \"fschat>=0.2.36\" jailbreakbench wandb pandas psutil accelerate" >&2
  exit 1
fi

if [[ ! -s "$TOGETHER_API_KEY_PATH" ]]; then
  echo "Missing Together API key file: $TOGETHER_API_KEY_PATH" >&2
  echo "Create it with: echo \"YOUR_TOGETHER_KEY\" > \"$TOGETHER_API_KEY_PATH\"" >&2
  exit 1
fi

if [[ "$RUN_WITH_NOHUP" == "1" && "${PAIR_NOHUP_CHILD:-0}" != "1" ]]; then
  launcher_dir="$PROJ/$LOG_ROOT/$NOHUP_LOG_DIR"
  mkdir -p "$launcher_dir"
  run_stamp="$(date +"%Y%m%d_%H%M%S")"
  launcher_log="$launcher_dir/run_${run_stamp}.log"
  launcher_pid="$launcher_dir/run_${run_stamp}.pid"

  echo "Submitting Together PAIR run with nohup..."
  echo "Launcher log: $launcher_log"
  echo "PID file: $launcher_pid"

  PAIR_NOHUP_CHILD=1 nohup bash "$0" "$@" > "$launcher_log" 2>&1 < /dev/null &
  child_pid="$!"
  echo "$child_pid" > "$launcher_pid"
  echo "Background PID: $child_pid"
  exit 0
fi

KEY_SELECTION="$("$PYTHON" - "$TOGETHER_API_KEY_PATH" "$API_KEY_INDEX" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
index_arg = sys.argv[2].strip()
keys = [
    (position, line.strip())
    for position, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
    if line.strip() and not line.lstrip().startswith("#")
]
if not keys:
    raise SystemExit(f"No non-empty API keys found in {path}")

if index_arg:
    try:
        selected_position = int(index_arg)
    except ValueError:
        raise SystemExit("API_KEY_INDEX must be a 1-based integer")
    if selected_position < 1 or selected_position > len(keys):
        raise SystemExit(f"API_KEY_INDEX must be between 1 and {len(keys)}")
    position, key = keys[selected_position - 1]
else:
    position, key = keys[-1]

print(position)
print(key)
PY
)"

export TOGETHER_API_KEY_INDEX="${KEY_SELECTION%%$'\n'*}"
export TOGETHER_API_KEY="${KEY_SELECTION#*$'\n'}"
export WANDB_MODE="$WANDB_MODE_VALUE"
export PIP_CACHE_DIR="$PROJ/.pipcache"
export TMPDIR="$PROJ/.tmp"
export HF_HOME="$PROJ/.hf"
export HF_DATASETS_CACHE="$PROJ/.hf/datasets"
export TRANSFORMERS_CACHE="$PROJ/.hf/transformers"
export WANDB_DIR="$PROJ/.wandb"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$WANDB_DIR"

cd "$PROJ"

run_target() {
  local target_model="$1"
  shift
  local experiment_name="${ATTACK_MODEL}__${target_model}__${JUDGE_MODEL}"
  experiment_name="${experiment_name//\//_}"
  experiment_name="${experiment_name//:/_}"
  local log_dir="$PROJ/$LOG_ROOT/$experiment_name"

  local args=(
    --log-dir "$log_dir"
    --wandb-mode "$WANDB_MODE_VALUE"
    --attack-model "$ATTACK_MODEL"
    --target-model "$target_model"
    --judge-model "$JUDGE_MODEL"
    --n-streams "$N_STREAMS"
    --n-iterations "$N_ITERATIONS"
    --attack-max-n-tokens "$ATTACK_MAX_N_TOKENS"
    --target-max-n-tokens "$TARGET_MAX_N_TOKENS"
    --judge-max-n-tokens "$JUDGE_MAX_N_TOKENS"
    --phase "$PHASE"
  )
  if [[ "$USE_JAILBREAKBENCH_TARGET" == "0" ]]; then
    args+=(--not-jailbreakbench-target)
  fi

  case "$RUN_MODE" in
    dry-run)
      args+=(--dry-run --limit "$NUM_BEHAVIORS")
      ;;
    limit)
      args+=(--limit "$NUM_BEHAVIORS")
      ;;
    full)
      args+=(--full)
      ;;
    *)
      echo "RUN_MODE must be one of: dry-run, limit, full" >&2
      exit 1
      ;;
  esac

  if [[ "$START_INDEX" != "0" ]]; then
    args+=(--start-index "$START_INDEX")
  fi
  if [[ "$RESUME" == "1" ]]; then
    args+=(--resume)
  fi
  if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
    args+=(--continue-on-error)
  fi

  echo
  echo "============================================================"
  echo "Running target: $target_model"
  echo "Using conda env: $ENV_DIR"
  echo "Results dir: $log_dir"
  echo "Attacker: $ATTACK_MODEL"
  echo "Judge: $JUDGE_MODEL"
  echo "Budget: $N_STREAMS streams x $N_ITERATIONS iterations"
  if [[ "$RUN_MODE" == "full" ]]; then
    echo "Dataset: all 100 JailbreakBench harmful behaviors"
  else
    echo "Dataset: $NUM_BEHAVIORS behavior(s), starting at index $START_INDEX"
  fi
  echo "============================================================"

  "$PYTHON" run_table2_llama.py "${args[@]}" "$@"
}

for target_model in "${TARGET_MODELS[@]}"; do
  run_target "$target_model" "$@"
done

if [[ "$RUN_POSTPROCESS" == "1" && "$RUN_MODE" != "dry-run" ]]; then
  "$PYTHON" summarize_pair_results.py \
    --input-root "$PROJ/$LOG_ROOT" \
    --output-dir "$PROJ/$RESULTS_DIR"
fi
