"""Extract source jailbreak prompts from PAIR run logs.

Shared by the transferability and defense experiments: both replay the prompts
that jailbroke an undefended target. Reads the per-behavior PAIR logs written by
the main runner.
"""
from __future__ import annotations
import os, re, json

# PAIR run dirs that may contain source jailbreak prompts.
SOURCE_DIRS = ["logs/paper_qwen_target", "logs/paper_gemini"]
PROMPT_RE = re.compile(r"Example Jailbreak PROMPT:\s*(.*?)\nExample Jailbreak RESPONSE:", re.S)


def collect_sources(source_dirs=None):
    """Return {source_model: [(index, jailbreak_prompt), ...]} from PAIR runs."""
    out = {}
    for d in (source_dirs or SOURCE_DIRS):
        sp = os.path.join(d, "status.jsonl")
        if not os.path.isfile(sp):
            continue
        rows = [json.loads(l) for l in open(sp)]
        if not rows:
            continue
        src = rows[-1]["target_model"]
        jbs = []
        for r in rows:
            if r.get("jailbroken") is not True:
                continue
            lf = r.get("log_file")
            if not lf or not os.path.isfile(lf):
                continue
            m = PROMPT_RE.search(open(lf, errors="replace").read())
            if m:
                jbs.append((int(r["index"]), m.group(1).strip()))
        if jbs:
            out[src] = jbs
    return out
