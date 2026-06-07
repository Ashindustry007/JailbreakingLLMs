#!/usr/bin/env bash
# Together AI JBC Table 2-style baseline launcher.
#
# Edit the "Experiment Settings" block below, then run:
#   bash run_jbc_together_variant.sh

set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

###############################################################################
# Experiment Settings
###############################################################################

# Dataset size controls.
# - RUN_MODE="full" runs all 100 JailbreakBench harmful behaviors.
# - RUN_MODE="limit" runs NUM_BEHAVIORS rows starting at START_INDEX.
# - RUN_MODE="dry-run" validates config without making API calls.
RUN_MODE="limit"                # dry-run | limit | full
NUM_BEHAVIORS=50                # used only when RUN_MODE="limit" or "dry-run"
START_INDEX=0                   # zero-based dataset index; 0 means start from first row
RESUME=1                        # 1 skips completed rows in status.jsonl
CONTINUE_ON_ERROR=1             # 1 keeps going if one target fails

# JBC baseline roles. This evaluates static jailbreak templates, no attacker LLM.
JUDGE_MODEL="llama-guard-4-12b"
TARGET_MODELS=(
  "llama-3-8b-instruct-lite"
  "gemma-3n-e4b-it"
)

# Generation budget. JBC sends each filled template once to the target.
TARGET_MAX_N_TOKENS=150
JUDGE_MAX_N_TOKENS=20
TEMPLATES_FILE="jbc_templates.json"

# Together API key. together-api.key is ignored by git via *.key.
# If the file contains multiple keys, API_KEY_INDEX is 1-based among non-empty,
# non-comment lines. Leave empty to use the last key.
TOGETHER_API_KEY_FILE="together-api.key"
API_KEY_INDEX=""

# Conda env and outputs. This reuses the same env as the PAIR variant.
CONDA_ENV_PATH="envs/pair"
LOG_ROOT="logs/jbc_together_variant"
RESULTS_DIR="results/jbc_together_variant"
RUN_POSTPROCESS=1               # 1 creates aggregate CSV/Markdown after all targets finish
RUN_WITH_NOHUP=1                 # 1 submits this script to background via nohup
PREVENT_SLEEP=1                  # 1 uses caffeinate on macOS if available
NOHUP_LOG_DIR="_launcher"        # relative to LOG_ROOT; stores launcher stdout/stderr

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

if [[ ! -s "$PROJ/$TEMPLATES_FILE" ]]; then
  echo "Missing JBC templates file: $PROJ/$TEMPLATES_FILE" >&2
  exit 1
fi

if [[ "$RUN_WITH_NOHUP" == "1" && "${JBC_NOHUP_CHILD:-0}" != "1" ]]; then
  launcher_dir="$PROJ/$LOG_ROOT/$NOHUP_LOG_DIR"
  mkdir -p "$launcher_dir"
  run_stamp="$(date +"%Y%m%d_%H%M%S")"
  launcher_log="$launcher_dir/run_${run_stamp}.log"
  launcher_pid="$launcher_dir/run_${run_stamp}.pid"

  echo "Submitting Together JBC baseline with nohup..."
  echo "Launcher log: $launcher_log"
  echo "PID file: $launcher_pid"

  if [[ "$PREVENT_SLEEP" == "1" ]] && command -v caffeinate >/dev/null 2>&1; then
    JBC_NOHUP_CHILD=1 nohup caffeinate -dimsu bash "$0" "$@" > "$launcher_log" 2>&1 < /dev/null &
  else
    JBC_NOHUP_CHILD=1 nohup bash "$0" "$@" > "$launcher_log" 2>&1 < /dev/null &
  fi
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
export PIP_CACHE_DIR="$PROJ/.pipcache"
export TMPDIR="$PROJ/.tmp"
export HF_HOME="$PROJ/.hf"
export HF_DATASETS_CACHE="$PROJ/.hf/datasets"
export TRANSFORMERS_CACHE="$PROJ/.hf/transformers"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

cd "$PROJ"

run_target() {
  local target_model="$1"
  shift
  local experiment_name="jbc__${target_model}__${JUDGE_MODEL}"
  experiment_name="${experiment_name//\//_}"
  experiment_name="${experiment_name//:/_}"
  local log_dir="$PROJ/$LOG_ROOT/$experiment_name"

  local args=(
    --log-dir "$log_dir"
    --target-model "$target_model"
    --judge-model "$JUDGE_MODEL"
    --target-max-n-tokens "$TARGET_MAX_N_TOKENS"
    --judge-max-n-tokens "$JUDGE_MAX_N_TOKENS"
    --templates-file "$TEMPLATES_FILE"
  )

  case "$RUN_MODE" in
    dry-run)
      args+=(--dry-run --limit "$NUM_BEHAVIORS")
      ;;
    limit)
      args+=(--limit "$NUM_BEHAVIORS")
      ;;
    full)
      args+=(--limit 100)
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

  echo
  echo "============================================================"
  echo "Running JBC target: $target_model"
  echo "Using conda env: $ENV_DIR"
  echo "Results dir: $log_dir"
  echo "Judge: $JUDGE_MODEL"
  echo "Templates: $TEMPLATES_FILE"
  if [[ "$RUN_MODE" == "full" ]]; then
    echo "Dataset: all 100 JailbreakBench harmful behaviors"
  else
    echo "Dataset: $NUM_BEHAVIORS behavior(s), starting at index $START_INDEX"
  fi
  echo "============================================================"

  if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
    "$PYTHON" jbc_eval.py "${args[@]}" "$@" || echo "Target failed: $target_model" >&2
  else
    "$PYTHON" jbc_eval.py "${args[@]}" "$@"
  fi
}

for target_model in "${TARGET_MODELS[@]}"; do
  run_target "$target_model" "$@"
done

if [[ "$RUN_POSTPROCESS" == "1" && "$RUN_MODE" != "dry-run" ]]; then
  "$PYTHON" - "$PROJ/$LOG_ROOT" "$PROJ/$RESULTS_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
results_dir = Path(sys.argv[2])
results_dir.mkdir(parents=True, exist_ok=True)
rows = []
for summary_path in sorted(log_root.glob("jbc__*/summary.json")):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows.append({
        "target_model": data["target_model"],
        "completed_behaviors": data["behaviors"],
        "templates": data["templates"],
        "avg_template_jailbreak_percent": round(data["avg_template_jailbreak_percent"], 2),
        "any_template_jailbreak_percent": round(data["any_template_jailbreak_percent"], 2),
    })

csv_path = results_dir / "table2_jbc_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "target_model",
        "completed_behaviors",
        "templates",
        "avg_template_jailbreak_percent",
        "any_template_jailbreak_percent",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

md_lines = ["# JBC Together Variant Summary", ""]
for row in rows:
    md_lines.append(
        f"- `{row['target_model']}`: Avg. Jailbreak % "
        f"{row['avg_template_jailbreak_percent']} across {row['templates']} templates "
        f"({row['completed_behaviors']} behaviors); any-template JB% "
        f"{row['any_template_jailbreak_percent']}"
    )
(results_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"Wrote JBC aggregate results to {results_dir}")
PY
fi
