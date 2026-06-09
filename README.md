# PAIR Reproduction — DSC 291 Trustworthy ML

A reproduction of **PAIR** (Prompt Automatic Iterative Refinement; Chao et al.,
2023) on contemporary open- and closed-source models, plus four additional
experiments. PAIR is a black-box jailbreak: an *attacker* LLM iteratively refines
an adversarial prompt against a *target* LLM, scored by a *judge*, usually
succeeding in under twenty queries.

## Setup

All models are reached through their native providers via `litellm`.

| Role | Default |
|------|---------|
| Attacker | Qwen2.5-7B-Instruct-Turbo (Together) |
| Judge | Llama-Guard-4-12B (Together) |
| Dataset | JailbreakBench `harmful`, first 50 behaviors |
| Budget | N = 30 streams, K = 3 iterations |

Targets reproduced: Qwen3-235B, Gemini-2.5-Flash, Llama-3-8B-Instruct-Lite,
Gemma-3n-E4B-it, GPT-3.5-Turbo, GPT-4o.

## Layout

```
config.py, judges.py, sources.py   shared modules (model routing, judge, prompt extraction)
run_ucsd_pair_variant.py           main PAIR runner (attacker / target / judge)
jbc_eval.py                        JailbreakChat static-template baseline (Table 2 "JBC")
experiments/
  stress_test/        PAIR methodology against 2026 SOTA models (gateway study)
  judge_prompt/       judge-prompt sensitivity: original vs refusal-aware Llama-Guard
  defense/            SmoothLLM and perplexity-filter defenses
  transferability/    replay jailbreak prompts across targets (Table 3)
figures/              plotting code for the report figures
results/              per-target result summaries (including collaborators' runs)
```
The upstream PAIR algorithm (`conversers.py`, `system_prompts.py`, `common.py`,
`loggers.py`, `main.py`, `language_models.py`) is unchanged; see
`README_UPSTREAM_PAIR.md`.

## Running

Export the keys your targets need, then run from the repository root:

```bash
export TOGETHER_API_KEY=...      # attacker, judge, Together targets
export GEMINI_API_KEY=...        # Gemini target
export OPENAI_API_KEY=...        # GPT targets

python run_ucsd_pair_variant.py --target-model qwen3-235b-instruct --limit 50
python jbc_eval.py               --target-model qwen3-235b-instruct --limit 50
```

Each experiment folder has its own README and result files.

## Note on the judge

The reported ASR depends on the Llama-Guard prompt. The default JailbreakBench
prompt over-counts: it flags refusals and benign on-topic answers as jailbreaks,
and the inflation grows with how much a model refuses (e.g. Llama-3-8B 90% vs 30%
under a refusal-aware prompt). See `experiments/judge_prompt/`.
