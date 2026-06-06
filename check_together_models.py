"""Probe which Together.ai model endpoints are currently callable with your key.

Run AFTER `set -a && source .env && set +a` (so TOGETHER_API_KEY is set):

    python check_together_models.py
"""

import os
import litellm

KEY = os.environ.get("TOGETHER_API_KEY")
if not KEY:
    raise SystemExit("TOGETHER_API_KEY not set. Run: set -a && source .env && set +a")

CANDIDATES = {
    "--- Llama Guard 4 (target judge) ---": [],
    "Llama-Guard-4-12B":                     "together_ai/meta-llama/Llama-Guard-4-12B",
    "Llama-Guard-4-12B (alt)":               "together_ai/meta-llama/Meta-Llama-Guard-4-12B",
    "Llama-Guard-3-8B (compare)":            "together_ai/meta-llama/Llama-Guard-3-8B",
    "--- Confirmed working ---": [],
    "Llama-3.3-70B-Instruct-Turbo":          "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen2.5-7B-Instruct-Turbo":             "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
}

for label, model in CANDIDATES.items():
    if not model:
        print(f"\n{label}")
        continue
    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            api_key=KEY,
            max_tokens=5,
        )
        print(f"  OK    {label:45} {model}")
    except Exception as e:
        msg = str(e)
        status = "no serverless" if "non-serverless" in msg else "not found" if "404" in msg else "error"
        print(f"  FAIL({status:12}) {label:45} {model}")
