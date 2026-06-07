#!/usr/bin/env bash
# UCSD TritonAI PAIR variant launcher.
#
# Edit the "Experiment Settings" block below, then run:
#   bash run_variant.sh

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
RUN_MODE="full"                 # dry-run | limit | full
NUM_BEHAVIORS=1                 # used only when RUN_MODE="limit" or "dry-run"
START_INDEX=0                   # zero-based dataset index; 0 means start from first row
RESUME=1                        # 1 skips completed rows in status.jsonl
CONTINUE_ON_ERROR=1             # 1 keeps going if one behavior fails

# Model roles. Keep attacker/judge fixed if you only want target-model variants.
ATTACK_MODEL="api-deepseek-v4-flash"
TARGET_MODEL="api-llama-4-scout"
JUDGE_MODEL="claude-sonnet-4-6"

# PAIR budget. Paper-style budget is 30 streams x 3 iterations.
N_STREAMS=20
N_ITERATIONS=3
ATTACK_MAX_N_TOKENS=1024
TARGET_MAX_N_TOKENS=150

# API and runtime. API_KEY_INDEX is 1-based among non-empty, non-comment lines in
# api-key.txt. Leave empty to auto-use the last key that can list gateway models.
API_KEY_FILE="api-key.txt"
API_KEY_INDEX="2"                # example: "2"
BASE_URL="https://tritonai-api.ucsd.edu/v1"
WANDB_MODE_VALUE="offline"      # offline | online | disabled

# Conda env and outputs. The existing env is ./envs/pair.
CONDA_ENV_PATH="envs/pair"
LOG_ROOT="logs/pair_ucsd_variant"

###############################################################################
# Available UCSD Model Names Registered In config.py
###############################################################################
#
# Good targets to try:
#   api-llama-4-scout
#   api-mistral-small-3.2-2506
#   mistral.mistral-large-3-675b-instruct
#
# Also registered, but better used carefully:
#   api-deepseek-v4-flash          # current attacker
#   claude-sonnet-4-6              # current judge
#
# The gateway catalog also has extra models, but they must be added to config.py
# before this runner can use them.

###############################################################################
# Launcher Implementation
###############################################################################

if [[ "$CONDA_ENV_PATH" = /* ]]; then
  ENV_DIR="$CONDA_ENV_PATH"
else
  ENV_DIR="$PROJ/$CONDA_ENV_PATH"
fi
PYTHON="$ENV_DIR/bin/python"

if [[ "$API_KEY_FILE" = /* ]]; then
  API_KEY_PATH="$API_KEY_FILE"
else
  API_KEY_PATH="$PROJ/$API_KEY_FILE"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python env: $PYTHON" >&2
  echo "Create it with:" >&2
  echo "  conda create -y -p \"$ENV_DIR\" python=3.11" >&2
  echo "  \"$PYTHON\" -m pip install \"litellm==1.52.0\" \"fschat>=0.2.36\" jailbreakbench wandb pandas psutil accelerate" >&2
  exit 1
fi

if [[ ! -s "$API_KEY_PATH" ]]; then
  echo "Missing UCSD API key file: $API_KEY_PATH" >&2
  echo "Create it with: echo \"YOUR_TRITON_KEY\" > \"$API_KEY_PATH\"" >&2
  exit 1
fi

KEY_SELECTION="$("$PYTHON" - "$API_KEY_PATH" "$API_KEY_INDEX" "$BASE_URL" <<'PY'
from pathlib import Path
import json
import sys
import urllib.request

path = Path(sys.argv[1])
index_arg = sys.argv[2].strip()
base_url = sys.argv[3].rstrip("/")
keys = [
    (position, line_no, line.strip())
    for position, (line_no, line) in enumerate(
        enumerate(path.read_text(encoding="utf-8").splitlines(), start=1),
        start=1,
    )
    if line.strip() and not line.lstrip().startswith("#")
]
if not keys:
    raise SystemExit(f"No non-empty API keys found in {path}")

def can_list_models(key: str) -> bool:
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            json.loads(response.read().decode("utf-8"))
        return True
    except Exception:
        return False

if index_arg:
    try:
        selected_position = int(index_arg)
    except ValueError:
        raise SystemExit("API_KEY_INDEX must be a 1-based integer")
    if selected_position < 1 or selected_position > len(keys):
        raise SystemExit(f"API_KEY_INDEX must be between 1 and {len(keys)}")
    position, _, key = keys[selected_position - 1]
else:
    position = key = None
    for candidate_position, _, candidate_key in reversed(keys):
        if can_list_models(candidate_key):
            position, key = candidate_position, candidate_key
            break
    if key is None:
        raise SystemExit("No API key in the key file could access the gateway model catalog.")

print(position)
print(key)
PY
)"

export PAIR_API_KEY_INDEX="${KEY_SELECTION%%$'\n'*}"
export OPENAI_API_KEY="${KEY_SELECTION#*$'\n'}"
export OPENAI_BASE_URL="$BASE_URL"
export WANDB_MODE="$WANDB_MODE_VALUE"
export PIP_CACHE_DIR="$PROJ/.pipcache"
export TMPDIR="$PROJ/.tmp"
export HF_HOME="$PROJ/.hf"
export WANDB_DIR="$PROJ/.wandb"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$WANDB_DIR"

EXPERIMENT_NAME="${ATTACK_MODEL}__${TARGET_MODEL}__${JUDGE_MODEL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME//\//_}"
EXPERIMENT_NAME="${EXPERIMENT_NAME//:/_}"
LOG_DIR="$PROJ/$LOG_ROOT/$EXPERIMENT_NAME"

ARGS=(
  --api-key-file "$API_KEY_PATH"
  --base-url "$BASE_URL"
  --log-dir "$LOG_DIR"
  --wandb-mode "$WANDB_MODE_VALUE"
  --attack-model "$ATTACK_MODEL"
  --target-model "$TARGET_MODEL"
  --judge-model "$JUDGE_MODEL"
  --n-streams "$N_STREAMS"
  --n-iterations "$N_ITERATIONS"
  --attack-max-n-tokens "$ATTACK_MAX_N_TOKENS"
  --target-max-n-tokens "$TARGET_MAX_N_TOKENS"
)

case "$RUN_MODE" in
  dry-run)
    ARGS+=(--dry-run --limit "$NUM_BEHAVIORS")
    ;;
  limit)
    ARGS+=(--limit "$NUM_BEHAVIORS")
    ;;
  full)
    ARGS+=(--full)
    ;;
  *)
    echo "RUN_MODE must be one of: dry-run, limit, full" >&2
    exit 1
    ;;
esac

if [[ "$START_INDEX" != "0" ]]; then
  ARGS+=(--start-index "$START_INDEX")
fi
if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  ARGS+=(--continue-on-error)
fi

echo "Using conda env: $ENV_DIR"
echo "Results dir: $LOG_DIR"
echo "Attacker: $ATTACK_MODEL"
echo "Target: $TARGET_MODEL"
echo "Judge: $JUDGE_MODEL"
echo "Budget: $N_STREAMS streams x $N_ITERATIONS iterations"
if [[ "$RUN_MODE" == "full" ]]; then
  echo "Dataset: all 100 JailbreakBench harmful behaviors"
else
  echo "Dataset: $NUM_BEHAVIORS behavior(s), starting at index $START_INDEX"
fi

cd "$PROJ"
exec "$PYTHON" run_ucsd_pair_variant.py "${ARGS[@]}" "$@"
